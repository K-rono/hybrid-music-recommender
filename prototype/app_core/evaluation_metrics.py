"""
Optimized evaluation metrics for music recommendation systems.

This module provides efficient implementations of common recommendation evaluation metrics
including NDCG@k, Novelty@k, Diversity@k, and Serendipity@k.

All functions are optimized for large datasets with vectorized operations and efficient
data structures to minimize memory usage and computation time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Union
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def _user_item_sets(
    df: pd.DataFrame, 
    user_to_idx: Dict[str, int], 
    item_to_idx: Dict[str, int]
) -> Dict[int, Set[int]]:
    """
    Efficiently create user-item sets from interaction dataframe.
    
    Args:
        df: DataFrame with 'user_id' and 'track_id' columns
        user_to_idx: Mapping from user_id to user index
        item_to_idx: Mapping from track_id to item index
        
    Returns:
        Dictionary mapping user indices to sets of item indices
    """
    # Convert to string and filter valid mappings
    df = df.copy()
    df['user_id'] = df['user_id'].astype(str)
    df['track_id'] = df['track_id'].astype(str)
    
    # Filter to only valid users and items
    valid_users = set(user_to_idx.keys())
    valid_items = set(item_to_idx.keys())
    
    df = df[
        (df['user_id'].isin(valid_users)) & 
        (df['track_id'].isin(valid_items))
    ].copy()
    
    if df.empty:
        return {}
    
    # Vectorized mapping to indices
    df['user_idx'] = df['user_id'].map(user_to_idx)
    df['item_idx'] = df['track_id'].map(item_to_idx)
    
    # Group by user and create sets
    user_item_sets = {}
    for user_idx, group in df.groupby('user_idx'):
        user_item_sets[user_idx] = set(group['item_idx'].values)
    
    return user_item_sets


def _dcg(recommendations: np.ndarray, relevant_items: Set[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain for a single user.
    
    Args:
        recommendations: Array of recommended item indices
        relevant_items: Set of relevant item indices
        k: Number of top recommendations to consider
        
    Returns:
        DCG score
    """
    dcg = 0.0
    for rank in range(min(k, len(recommendations))):
        item_idx = recommendations[rank]
        if item_idx in relevant_items:
            # Binary relevance (1 if relevant, 0 otherwise)
            gain = 1.0
            # Discount factor: 1 / log2(rank + 2) since rank is 0-based
            discount = 1.0 / np.log2(rank + 2.0)
            dcg += gain * discount
    return dcg


def ndcg_at_k(
    recommendations: Dict[int, np.ndarray],
    test_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
    k: int = 10
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@k.
    
    Args:
        recommendations: Dict mapping user indices to recommendation arrays
        test_df: Test interactions DataFrame
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping
        k: Number of top recommendations to evaluate
        
    Returns:
        Average NDCG@k across all users
    """
    # Get relevant items for each user from test set
    test_sets = _user_item_sets(test_df, user_to_idx, item_to_idx)
    
    if not test_sets:
        return 0.0
    
    ndcg_scores = []
    
    for user_idx, rec in recommendations.items():
        relevant_items = test_sets.get(user_idx, set())
        
        if not relevant_items:
            continue
            
        # Calculate DCG
        dcg = _dcg(rec, relevant_items, k)
        
        # Calculate IDCG (ideal DCG)
        # IDCG is the DCG if all relevant items were ranked at the top
        num_relevant = min(k, len(relevant_items))
        idcg = sum(1.0 / np.log2(rank + 2.0) for rank in range(num_relevant))
        
        # Calculate NDCG
        if idcg > 0:
            ndcg = dcg / idcg
            ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def novelty_at_k(
    recommendations: Dict[int, np.ndarray],
    train_df: pd.DataFrame,
    item_to_idx: Dict[str, int],
    k: int = 10
) -> float:
    """
    Calculate Novelty@k - measures how "unpopular" recommended items are.
    
    Args:
        recommendations: Dict mapping user indices to recommendation arrays
        train_df: Training interactions DataFrame
        item_to_idx: Item ID to index mapping
        k: Number of top recommendations to evaluate
        
    Returns:
        Average Novelty@k across all users
    """
    # Calculate item popularity in training data
    train_df = train_df.copy()
    train_df['track_id'] = train_df['track_id'].astype(str)
    
    # Filter to catalog items only
    valid_items = set(item_to_idx.keys())
    train_filtered = train_df[train_df['track_id'].isin(valid_items)].copy()
    
    if train_filtered.empty:
        return 0.0
    
    # Count unique users per item (popularity)
    item_popularity = train_filtered.groupby('track_id')['user_id'].nunique()
    n_users = train_filtered['user_id'].nunique()
    
    # Create popularity mapping
    pop_counts = {}
    for track_id, count in item_popularity.items():
        if track_id in item_to_idx:
            pop_counts[item_to_idx[track_id]] = count
    
    def _novelty_item(item_idx: int) -> float:
        """Calculate novelty for a single item."""
        count = pop_counts.get(item_idx, 0)
        # Laplace smoothing to avoid log(0)
        p = (count + 1.0) / (n_users + 1.0)
        return -np.log2(p)
    
    # Calculate average novelty for each user
    novelty_scores = []
    
    for user_idx, rec in recommendations.items():
        user_novelties = []
        for rank in range(min(k, len(rec))):
            item_idx = rec[rank]
            novelty = _novelty_item(item_idx)
            user_novelties.append(novelty)
        
        if user_novelties:
            avg_novelty = np.mean(user_novelties)
            novelty_scores.append(avg_novelty)
    
    return np.mean(novelty_scores) if novelty_scores else 0.0


def diversity_ild_at_k(
    recommendations: Dict[int, np.ndarray],
    item_content: np.ndarray,
    k: int = 10
) -> float:
    """
    Calculate Inter-List Diversity@k - measures diversity within recommendation lists.
    
    Args:
        recommendations: Dict mapping user indices to recommendation arrays
        item_content: Content feature matrix (n_items, n_features)
        k: Number of top recommendations to evaluate
        
    Returns:
        Average Diversity(ILD)@k across all users
    """
    diversity_scores = []
    
    for user_idx, rec in recommendations.items():
        # Get content vectors for recommended items
        rec_items = rec[:k]
        
        if len(rec_items) < 2:
            continue  # Need at least 2 items for diversity calculation
        
        # Get content vectors
        content_vectors = item_content[rec_items]
        
        # Calculate pairwise cosine similarities
        # Since vectors are L2-normalized, dot product gives cosine similarity
        similarity_matrix = content_vectors @ content_vectors.T
        
        # Calculate pairwise dissimilarities (1 - cosine_similarity)
        dissimilarity_matrix = 1.0 - similarity_matrix
        
        # Get upper triangle (excluding diagonal) for unique pairs
        upper_tri_indices = np.triu_indices(len(rec_items), k=1)
        pairwise_dissimilarities = dissimilarity_matrix[upper_tri_indices]
        
        # Average pairwise dissimilarity
        avg_dissimilarity = np.mean(pairwise_dissimilarities)
        diversity_scores.append(avg_dissimilarity)
    
    return np.mean(diversity_scores) if diversity_scores else 0.0


def serendipity_at_k(
    recommendations: Dict[int, np.ndarray],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
    item_content: np.ndarray,
    k: int = 10
) -> float:
    """
    Calculate Serendipity@k - measures surprising but relevant recommendations.
    
    Args:
        recommendations: Dict mapping user indices to recommendation arrays
        train_df: Training interactions DataFrame
        test_df: Test interactions DataFrame
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping
        item_content: Content feature matrix
        k: Number of top recommendations to evaluate
        
    Returns:
        Average Serendipity@k across all users
    """
    # Get user historical items from training data
    train_sets = _user_item_sets(train_df, user_to_idx, item_to_idx)
    test_sets = _user_item_sets(test_df, user_to_idx, item_to_idx)
    
    if not train_sets or not test_sets:
        return 0.0
    
    serendipity_scores = []
    
    for user_idx, rec in recommendations.items():
        historical_items = train_sets.get(user_idx, set())
        relevant_items = test_sets.get(user_idx, set())
        
        if not historical_items or not relevant_items:
            continue
        
        # Get content vectors for historical items
        historical_vectors = item_content[list(historical_items)]
        historical_centroid = np.mean(historical_vectors, axis=0)
        
        # Calculate serendipity for relevant recommendations
        relevant_recs = []
        for rank in range(min(k, len(rec))):
            item_idx = rec[rank]
            if item_idx in relevant_items:
                relevant_recs.append(item_idx)
        
        if not relevant_recs:
            continue
        
        # Calculate dissimilarity between relevant recommendations and user history
        rec_vectors = item_content[relevant_recs]
        
        # Calculate cosine dissimilarity with historical centroid
        similarities = cosine_similarity(rec_vectors, historical_centroid.reshape(1, -1)).flatten()
        dissimilarities = 1.0 - similarities
        
        # Average dissimilarity for relevant recommendations
        avg_dissimilarity = np.mean(dissimilarities)
        serendipity_scores.append(avg_dissimilarity)
    
    return np.mean(serendipity_scores) if serendipity_scores else 0.0


# Coverage metrics removed per project decision


def evaluate_recommendations(
    recommendations: Dict[int, np.ndarray],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
    item_content: np.ndarray,
    k_values: List[int] = [5, 10, 20],
    metrics: List[str] = None
) -> Dict[str, Dict[int, float]]:
    """
    Comprehensive evaluation of recommendations across multiple metrics and k values.
    
    Args:
        recommendations: Dict mapping user indices to recommendation arrays
        train_df: Training interactions DataFrame
        test_df: Test interactions DataFrame
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping
        item_content: Content feature matrix
        k_values: List of k values to evaluate
        metrics: List of metrics to calculate (None for all)
        
    Returns:
        Nested dictionary with results: {metric: {k: score}}
    """
    if metrics is None:
        metrics = ['ndcg', 'novelty', 'diversity', 'serendipity']
    
    results = {}
    total_items = len(item_to_idx)
    total_users = len(user_to_idx)
    
    print(f"Evaluating {len(recommendations)} users across {len(k_values)} k values...")
    
    for metric in metrics:
        results[metric] = {}
        print(f"  Calculating {metric}...")
        
        for k in k_values:
            if metric == 'ndcg':
                score = ndcg_at_k(recommendations, test_df, user_to_idx, item_to_idx, k)
            elif metric == 'novelty':
                score = novelty_at_k(recommendations, train_df, item_to_idx, k)
            elif metric == 'diversity':
                score = diversity_ild_at_k(recommendations, item_content, k)
            elif metric == 'serendipity':
                score = serendipity_at_k(recommendations, train_df, test_df, user_to_idx, item_to_idx, item_content, k)
            elif metric == 'catalog_coverage':
                score = catalog_coverage_at_k(recommendations, total_items, k)
            elif metric == 'user_coverage':
                score = user_coverage_at_k(recommendations, total_users)
            else:
                print(f"    Warning: Unknown metric '{metric}', skipping...")
                continue
            
            results[metric][k] = score
            print(f"    {metric}@{k}: {score:.4f}")
    
    return results


def print_evaluation_summary(results: Dict[str, Dict[int, float]]) -> None:
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Results dictionary from evaluate_recommendations
    """
    print("\n" + "="*60)
    print("RECOMMENDATION EVALUATION SUMMARY")
    print("="*60)
    
    for metric, k_scores in results.items():
        print(f"\n{metric.upper()}:")
        print("-" * 20)
        for k, score in k_scores.items():
            print(f"  @{k:2d}: {score:6.4f}")
    
    print("\n" + "="*60)


# Convenience function for quick evaluation
def quick_evaluate(
    recommendations: Dict[int, np.ndarray],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
    item_content: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Quick evaluation with a single k value for all metrics.
    
    Args:
        recommendations: Dict mapping user indices to recommendation arrays
        train_df: Training interactions DataFrame
        test_df: Test interactions DataFrame
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping
        item_content: Content feature matrix
        k: Number of top recommendations to evaluate
        
    Returns:
        Dictionary with metric scores
    """
    results = evaluate_recommendations(
        recommendations, train_df, test_df, user_to_idx, item_to_idx, item_content, [k]
    )
    
    # Flatten results
    flat_results = {}
    for metric, k_scores in results.items():
        flat_results[metric] = k_scores[k]
    
    return flat_results
