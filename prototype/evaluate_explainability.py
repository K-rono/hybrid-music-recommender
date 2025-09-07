#!/usr/bin/env python3
"""
Batch evaluation script for explainability metrics.
Evaluates Stability, Coverage, and Efficiency for 5000 users.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import time

# Add the prototype directory to the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from app_core.data_loader import load_music_and_behavior
from app_core.features import load_artifacts
from app_core.recommender import (
    HybridConfig, build_tfidf_interactions, train_svd_cf,
    build_content_matrix, make_item_hybrid, build_ann, recommend_hybrid
)
from app_core.explainer import RecommendationExplainer
from app_core.explainability_evaluator import ExplainabilityEvaluator


def get_sample_users(behavior_df: pd.DataFrame, n_users: int = 100, seed: int = 42) -> pd.DataFrame:
    """Get a sample of users for evaluation"""
    # Get users with at least 10 interactions for reliable evaluation
    user_counts = behavior_df['user_id'].value_counts()
    active_users = user_counts[user_counts >= 10].index
    
    # Sample users
    np.random.seed(seed)
    if len(active_users) >= n_users:
        sampled_users = np.random.choice(active_users, size=n_users, replace=False)
    else:
        sampled_users = active_users
    
    # Filter behavior data to sampled users
    sampled_behavior = behavior_df[behavior_df['user_id'].isin(sampled_users)].copy()
    
    print(f"Sampled {len(sampled_users)} users with {len(sampled_behavior)} interactions")
    return sampled_behavior


def build_hybrid_components(behavior_df: pd.DataFrame, item_features_df: pd.DataFrame, config: HybridConfig):
    """Build hybrid recommendation components"""
    print("Building hybrid components...")
    
    # Data splitting
    behavior_df = shuffle(behavior_df, random_state=42)
    train_df = behavior_df.groupby('user_id', group_keys=False).apply(
        lambda x: x.sample(frac=0.8, random_state=42)
    )
    test_df = behavior_df.drop(train_df.index)
    
    # Get catalog items
    catalog_items = item_features_df["track_id"].astype(str).tolist()
    
    # Build TF-IDF interactions
    print("Building TF-IDF interactions...")
    X_train, user_to_idx, item_to_idx, all_users = build_tfidf_interactions(
        train_df, test_df, catalog_items
    )
    
    # Train SVD
    print("Training SVD...")
    cf_pack = train_svd_cf(X_train, n_components=config.n_components)
    
    # Build content matrix
    print("Building content matrix...")
    item_content = build_content_matrix(catalog_items, item_features_df)
    
    # Create hybrid item vectors
    print("Creating hybrid item vectors...")
    item_hybrid = make_item_hybrid(
        cf_pack["V"], item_content, config.lambda_cf, config.lambda_cb
    )
    
    # Build ANN index
    print("Building ANN index...")
    ann = build_ann(item_hybrid, n_neighbors=config.topk_ann)
    
    return {
        "cf_pack": cf_pack,
        "ann": ann,
        "all_users": all_users,
        "catalog_items": catalog_items,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "item_content": item_content,
        "item_hybrid": item_hybrid,
        "train_df": train_df,
    }


def generate_recommendations_for_users(
    hybrid_components: dict, 
    config: HybridConfig, 
    user_ids: list, 
    k: int = 10
) -> dict:
    """Generate recommendations for all users"""
    print(f"Generating recommendations for {len(user_ids)} users...")
    
    user_recommendations = {}
    
    for i, user_id in enumerate(user_ids):
        if i % 100 == 0:
            print(f"Processed {i}/{len(user_ids)} users...")
        
        try:
            recommendations = recommend_hybrid(
                user_id=user_id,
                cf_pack=hybrid_components["cf_pack"],
                ann=hybrid_components["ann"],
                train_df=hybrid_components["train_df"],
                all_users=hybrid_components["all_users"],
                catalog_items=hybrid_components["catalog_items"],
                user_to_idx=hybrid_components["user_to_idx"],
                item_to_idx=hybrid_components["item_to_idx"],
                item_content=hybrid_components["item_content"],
                item_hybrid=hybrid_components["item_hybrid"],
                config=config,
                k=k
            )
            
            if recommendations:
                user_recommendations[user_id] = recommendations
                
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {e}")
            continue
    
    print(f"Generated recommendations for {len(user_recommendations)} users")
    return user_recommendations


def main():
    """Main evaluation function"""
    print("🎯 Starting Explainability Evaluation (100 users)")
    print("=" * 50)
    
    # Configuration
    config = HybridConfig(
        n_components=64,
        lambda_cf=0.7,
        lambda_cb=0.3,
        topk_ann=200,
        reco_k=10
    )
    
    # Load data
    print("Loading data...")
    music_df, behavior_df = load_music_and_behavior()
    artifacts = load_artifacts()
    
    # Sample users
    print("Sampling users...")
    sampled_behavior = get_sample_users(behavior_df, n_users=100, seed=42)
    sampled_users = sampled_behavior['user_id'].unique().tolist()
    
    # Build hybrid components
    hybrid_components = build_hybrid_components(
        sampled_behavior, artifacts.item_features_df, config
    )
    
    # Generate recommendations
    user_recommendations = generate_recommendations_for_users(
        hybrid_components, config, sampled_users, k=10
    )
    
    # Create explainer
    print("Creating explainer...")
    explainer = ExplainabilityEvaluator(
        RecommendationExplainer(
            cf_pack=hybrid_components["cf_pack"],
            item_content=hybrid_components["item_content"],
            item_hybrid=hybrid_components["item_hybrid"],
            feature_columns=artifacts.feature_columns,
            train_df=hybrid_components["train_df"],
            user_to_idx=hybrid_components["user_to_idx"],
            item_to_idx=hybrid_components["item_to_idx"],
            all_users=hybrid_components["all_users"],
            catalog_items=hybrid_components["catalog_items"],
            music_df=music_df,
            config=config
        )
    )
    
    # Run evaluation
    print("Running explainability evaluation...")
    start_time = time.time()
    
    results = explainer.evaluate_batch(
        user_recommendations,
        stability_runs=3,
        efficiency_runs=5
    )
    
    evaluation_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"Total Users Evaluated: {results.total_users}")
    print(f"Total Recommendations: {results.total_recommendations}")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    
    print("\n🎯 STABILITY METRICS")
    print(f"Feature Consistency: {results.avg_stability.feature_consistency:.3f}")
    print(f"User Consistency: {results.avg_stability.user_consistency:.3f}")
    print(f"Score Consistency: {results.avg_stability.score_consistency:.3f}")
    
    print("\n📈 COVERAGE METRICS")
    print(f"Coverage Percentage: {results.avg_coverage.coverage_percentage:.2f}%")
    print(f"Successful Explanations: {results.avg_coverage.successful_explanations}")
    print(f"Failed Explanations: {results.avg_coverage.failed_explanations}")
    
    print("\n⚡ EFFICIENCY METRICS")
    print(f"Avg Generation Time: {results.avg_efficiency.avg_generation_time:.4f} seconds")
    print(f"Min Generation Time: {results.avg_efficiency.min_generation_time:.4f} seconds")
    print(f"Max Generation Time: {results.avg_efficiency.max_generation_time:.4f} seconds")
    
    # Save results
    output_file = PROJECT_ROOT / "explainability_evaluation_results.json"
    explainer.save_results(results, str(output_file))
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
