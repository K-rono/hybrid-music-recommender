from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from app_core.data_loader import load_music_and_behavior, get_all_user_ids, validate_user_id
from app_core.features import artifacts_exist, build_and_save_artifacts, load_artifacts
from app_core.recommender import (
    HybridConfig,
    build_tfidf_interactions,
    train_svd_cf,
    build_content_matrix,
    make_item_hybrid,
    build_ann,
    recommend_hybrid,
)
from app_core.spotify_client import SpotifyClient, SpotifyAuthError
from app_core.ui_helpers import render_song_card, render_explanation_card, render_user_evaluation_metrics, render_individual_evaluation_metrics
from app_core.explainer import RecommendationExplainer, get_cached_explanation, cache_explanation
from app_core.explainability_evaluator import ExplainabilityEvaluator

from dotenv import load_dotenv
load_dotenv()  

st.set_page_config(page_title="Music Recommender", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    return load_music_and_behavior()


@st.cache_resource(show_spinner=True)
def get_artifacts(music_df: pd.DataFrame):
    if artifacts_exist():
        return load_artifacts()
    return build_and_save_artifacts(music_df)


@st.cache_resource(show_spinner=False)
def get_data_split(behavior_df: pd.DataFrame, sample_users: bool = True):
    """Cache the data split separately from model components."""
    from sklearn.utils import shuffle
    
    if sample_users:
        # Get users with at least 10 interactions for faster processing
        user_counts = behavior_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= 10].index
        behavior_df = behavior_df[behavior_df['user_id'].isin(active_users)]
    
    behavior_df = shuffle(behavior_df, random_state=42)
    train_df = behavior_df.groupby('user_id', group_keys=False).apply(
        lambda x: x.sample(frac=0.8, random_state=42)
    )
    test_df = behavior_df.drop(train_df.index)
    
    return train_df, test_df


@st.cache_resource(show_spinner=False)
def get_hybrid_components(train_df: pd.DataFrame, test_df: pd.DataFrame, item_features_df: pd.DataFrame, config: HybridConfig):
    """Build all hybrid recommender components with caching."""
    import time
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    progress_bar.progress(10)
    
    # Get catalog items from features
    status_text.text("📋 Preparing catalog items...")
    catalog_items = item_features_df["track_id"].astype(str).tolist()
    progress_bar.progress(15)
    
    # Build TF-IDF interactions
    status_text.text("🔢 Building TF-IDF interactions matrix...")
    start_time = time.time()
    X_train, user_to_idx, item_to_idx, all_users = build_tfidf_interactions(
        train_df, test_df, catalog_items
    )
    tfidf_time = time.time() - start_time
    st.info(f"✅ TF-IDF matrix built in {tfidf_time:.2f}s - Shape: {X_train.shape}")
    progress_bar.progress(35)
    
    # Train SVD for collaborative filtering
    status_text.text("🧮 Training SVD for collaborative filtering...")
    start_time = time.time()
    cf_pack = train_svd_cf(X_train, n_components=config.n_components)
    svd_time = time.time() - start_time
    st.info(f"✅ SVD trained in {svd_time:.2f}s - Components: {config.n_components}")
    progress_bar.progress(55)
    
    # Build content matrix
    status_text.text("🎵 Building content feature matrix...")
    start_time = time.time()
    item_content = build_content_matrix(catalog_items, item_features_df)
    content_time = time.time() - start_time
    st.info(f"✅ Content matrix built in {content_time:.2f}s - Shape: {item_content.shape}")
    progress_bar.progress(75)
    
    # Create hybrid item vectors
    status_text.text("🔗 Creating hybrid item vectors...")
    start_time = time.time()
    item_hybrid = make_item_hybrid(
        cf_pack["V"], item_content, config.lambda_cf, config.lambda_cb
    )
    hybrid_time = time.time() - start_time
    st.info(f"✅ Hybrid vectors created in {hybrid_time:.2f}s - Shape: {item_hybrid.shape}")
    progress_bar.progress(85)
    
    # Build ANN index
    status_text.text("🔍 Building ANN index for fast retrieval...")
    start_time = time.time()
    ann = build_ann(item_hybrid, n_neighbors=config.topk_ann)
    ann_time = time.time() - start_time
    st.info(f"✅ ANN index built in {ann_time:.2f}s - Neighbors: {config.topk_ann}")
    progress_bar.progress(100)
    
    total_time = tfidf_time + svd_time + content_time + hybrid_time + ann_time
    status_text.text(f"🎉 All components ready! Total time: {total_time:.2f}s")
    
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


@st.cache_resource(show_spinner=False)
def get_spotify_client():
    try:
        client = SpotifyClient()
        # Trigger token fetch to validate env early
        client._ensure_token()
        return client
    except SpotifyAuthError:
        return None


def get_explainer(hybrid_components: dict, artifacts: dict, config: HybridConfig, music_df: pd.DataFrame):
    """Create the recommendation explainer (not cached to avoid Streamlit state issues)"""
    return RecommendationExplainer(
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


def main():
    st.title("Hybrid Music Recommender")
    music_df, behavior_df = load_data()

    # Load artifacts
    artifacts = get_artifacts(music_df)

    # Configuration sidebar
    st.sidebar.header("Configuration")
    lambda_cf = st.sidebar.slider("Collaborative Weight", 0.0, 1.0, 0.7, 0.1)
    lambda_cb = st.sidebar.slider("Content Weight", 0.0, 1.0, 0.3, 0.1)
    n_components = st.sidebar.slider("SVD Components", 16, 128, 64, 16)
    
    # Normalize weights
    total_weight = lambda_cf + lambda_cb
    lambda_cf = lambda_cf / total_weight
    lambda_cb = lambda_cb / total_weight
    
    config = HybridConfig(
        n_components=n_components,
        lambda_cf=lambda_cf,
        lambda_cb=lambda_cb,
        topk_ann=200,
        reco_k=20
    )

    # Sample users for faster processing (optional optimization)
    sample_users = st.sidebar.checkbox("Sample users for faster processing", value=True)
    
    # Get data split (cached separately)
    st.info("📊 Getting data split...(Might take a couple minutes)")
    train_df, test_df = get_data_split(behavior_df, sample_users)
    
    with st.expander("🔍 Performance Debug Info", expanded=False):
        st.write("**Dataset Statistics:**")
        st.write(f"- Total users: {behavior_df['user_id'].nunique():,}")
        st.write(f"- Total tracks: {behavior_df['track_id'].nunique():,}")
        st.write(f"- Total interactions: {len(behavior_df):,}")
        st.write(f"- Train interactions: {len(train_df):,}")
        st.write(f"- Test interactions: {len(test_df):,}")
        st.write(f"- Feature dimensions: {len(artifacts.feature_columns):,}")
        st.write(f"- SVD components: {config.n_components}")
        st.write(f"- CF weight: {config.lambda_cf:.2f}, Content weight: {config.lambda_cb:.2f}")
    
    # Build hybrid components (only rebuilds when config changes)
    hybrid_components = get_hybrid_components(train_df, test_df, artifacts.item_features_df, config)
    
    # Initialize explainer
    explainer = get_explainer(hybrid_components, artifacts, config, music_df)
    
    # Initialize evaluator
    evaluator = ExplainabilityEvaluator(explainer)

    st.sidebar.header("User Selection")
    user_input = st.sidebar.text_input("User ID", value="")
    top_n = st.sidebar.number_input("Top N", min_value=1, max_value=50, value=10)
    submitted = st.sidebar.button("Recommend")

    available_users = get_all_user_ids(behavior_df)
    st.sidebar.caption(f"Available users: {len(available_users):,}")

    if submitted:
        if not validate_user_id(user_input, behavior_df):
            st.error("Please enter a valid user_id.")
            return

        with st.spinner("Generating hybrid recommendations..."):
            import time
            
            rec_progress = st.progress(0)
            rec_status = st.empty()
            
            rec_status.text("👤 Building hybrid user profile...")
            start_time = time.time()
            
            recommendations = recommend_hybrid(
                user_id=user_input,
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
                k=int(top_n)
            )
            
            rec_time = time.time() - start_time
            rec_progress.progress(100)
            rec_status.text(f"✅ Recommendations generated in {rec_time:.2f}s - Found {len(recommendations)} items")

        if not recommendations:
            st.warning("No recommendations available for this user.")
            return

        # Extract track IDs and scores
        rec_track_ids = [rec[0] for rec in recommendations]
        rec_scores = [rec[1] for rec in recommendations]

        # Prepare display df
        rec_df = music_df[music_df["track_id"].astype(str).isin(rec_track_ids)].copy()
        # Preserve order according to recommendations
        rec_df["_order"] = rec_df["track_id"].astype(str).apply(
            lambda x: rec_track_ids.index(str(x)) if str(x) in rec_track_ids else 999999
        )
        rec_df = rec_df.sort_values("_order").drop(columns=["_order"])
        
        # Add scores
        rec_df["hybrid_score"] = rec_scores

        spotify_ids = rec_df["spotify_id"].astype(str).tolist()
        spotify_client = get_spotify_client()
        track_meta_map: Dict[str, dict] = {}
        if spotify_client is not None:
            try:
                track_meta_map = spotify_client.fetch_tracks(spotify_ids)
            except Exception:
                track_meta_map = {}
        else:
            st.info("Spotify credentials not configured. Skipping cover art.")

        # Store recommendations in session state to prevent regeneration
        st.session_state['recommendations'] = {
            'rec_df': rec_df,
            'rec_track_ids': rec_track_ids,
            'rec_scores': rec_scores,
            'track_meta_map': track_meta_map,
            'user_input': user_input,
            'lambda_cf': lambda_cf,
            'lambda_cb': lambda_cb
        }

    # Display recommendations if they exist in session state
    if 'recommendations' in st.session_state:
        rec_data = st.session_state['recommendations']
        rec_df = rec_data['rec_df']
        rec_track_ids = rec_data['rec_track_ids']
        rec_scores = rec_data['rec_scores']
        track_meta_map = rec_data['track_meta_map']
        user_input = rec_data['user_input']
        lambda_cf = rec_data['lambda_cf']
        lambda_cb = rec_data['lambda_cb']

        st.subheader("Hybrid Recommendations with Explanations")
        st.caption(f"CF Weight: {lambda_cf:.2f}, Content Weight: {lambda_cb:.2f}")
        
        # Add explanation toggle
        show_explanations = st.checkbox("Show detailed explanations", value=True)
        
        # Add evaluation metrics toggle
        show_evaluation = st.checkbox("Show explainability evaluation metrics", value=False)
        
        # Evaluation metrics for the current user
        if show_evaluation:
            with st.expander("📊 Explainability Evaluation Metrics", expanded=False):
                render_user_evaluation_metrics(evaluator, user_input, rec_track_ids, rec_scores)
        
        for _, row in rec_df.iterrows():
            track_id = str(row["track_id"])
            score = row['hybrid_score']
            
            if show_explanations:
                # Generate explanation
                with st.spinner(f"Generating explanation for {row['name']}..."):
                    # Check cache first
                    cached_explanation = get_cached_explanation(user_input, track_id, score)
                    
                    if cached_explanation:
                        explanation = cached_explanation
                    else:
                        # Generate new explanation
                        explanation = explainer.explain_recommendation(user_input, track_id, score)
                        # Cache it
                        cache_explanation(user_input, track_id, score, explanation)
                
                # Render with explanation
                meta = track_meta_map.get(str(row["spotify_id"])) if track_meta_map else None
                preview_url = row.get("spotify_preview_url")
                render_explanation_card(explanation, meta, preview_url)
                
                # Show individual evaluation metrics if enabled
                if show_evaluation:
                    render_individual_evaluation_metrics(evaluator, user_input, track_id, score)
            else:
                # Render simple card without explanation
                meta = track_meta_map.get(str(row["spotify_id"])) if track_meta_map else None
                render_song_card(row.to_dict(), meta)
                st.caption(f"Hybrid Score: {score:.3f}")


if __name__ == "__main__":
    main()


