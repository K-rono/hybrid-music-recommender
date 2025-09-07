from __future__ import annotations

from typing import Dict, Optional, List
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from .explainer import RecommendationExplanation, FeatureContribution, CollaborativeInsight, SimilarHistoricalItem
from .explainability_evaluator import ExplainabilityEvaluator, StabilityMetrics, CoverageMetrics, EfficiencyMetrics


def render_song_card(row: dict, track_meta: Optional[dict]) -> None:
    cols = st.columns([1, 3])
    with cols[0]:
        image_url = None
        if track_meta and track_meta.get("album", {}).get("images"):
            image_url = track_meta["album"]["images"][1]["url"] if len(track_meta["album"]["images"]) > 1 else track_meta["album"]["images"][0]["url"]
        if image_url:
            st.image(image_url)
    with cols[1]:
        title = row.get("name") or (track_meta.get("name") if track_meta else None) or "Unknown Title"
        artist = row.get("artist") or (
            ", ".join(a.get("name") for a in track_meta.get("artists", [])) if track_meta else None
        ) or "Unknown Artist"
        st.markdown(f"**{title}** — {artist}")
        preview_url = row.get("spotify_preview_url")
        if preview_url:
            st.audio(preview_url, format="audio/mp3")
        else:
            st.caption("Preview unavailable")
        if track_meta and track_meta.get("external_urls", {}).get("spotify"):
            st.link_button("Open in Spotify", track_meta["external_urls"]["spotify"])


def render_explanation_card(explanation: RecommendationExplanation, track_meta: Optional[dict] = None, preview_url: Optional[str] = None) -> None:
    """Render a song card with explanation details"""
    
    # Main song card
    with st.container():
        st.markdown("---")
        cols = st.columns([1, 3, 1])
        
        with cols[0]:
            # Album art
            image_url = None
            if track_meta and track_meta.get("album", {}).get("images"):
                image_url = track_meta["album"]["images"][1]["url"] if len(track_meta["album"]["images"]) > 1 else track_meta["album"]["images"][0]["url"]
            if image_url:
                st.image(image_url)
        
        with cols[1]:
            # Song info
            st.markdown(f"**{explanation.track_name}** — {explanation.track_artist}")
            st.caption(f"Recommendation Score: {explanation.total_score:.3f}")
            
            # Preview audio - try both sources
            audio_url = None
            if track_meta and track_meta.get("preview_url"):
                audio_url = track_meta["preview_url"]
            elif preview_url:
                audio_url = preview_url
            
            if audio_url:
                st.audio(audio_url, format="audio/mp3")
            else:
                st.caption("Preview unavailable")
        
        with cols[2]:
            # Spotify link
            if track_meta and track_meta.get("external_urls", {}).get("spotify"):
                st.link_button("Open in Spotify", track_meta["external_urls"]["spotify"])
    
    # Explanation details in expandable section
    with st.expander("🔍 Why was this recommended?", expanded=False):
        render_explanation_details(explanation)


def render_explanation_details(explanation: RecommendationExplanation) -> None:
    """Render detailed explanation components"""
    
    # Score breakdown
    st.subheader("📊 Score Breakdown")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Collaborative Score", 
            f"{explanation.score_breakdown['collaborative_score']:.3f}",
            help="Based on users with similar taste"
        )
    
    with col2:
        st.metric(
            "Content Score", 
            f"{explanation.score_breakdown['content_score']:.3f}",
            help="Based on audio features and content similarity"
        )
    
    with col3:
        st.metric(
            "Total Score", 
            f"{explanation.score_breakdown['total_score']:.3f}",
            help="Combined recommendation score"
        )
    
    # Content feature analysis
    st.subheader("🎵 Audio Feature Analysis")
    render_content_analysis(explanation.content_contributions)
    
    # Collaborative insights
    st.subheader("👥 Collaborative Insights")
    render_collaborative_insights(explanation.collaborative_insight)
    
    # Similar historical items
    if explanation.similar_historical_items:
        st.subheader("🎧 Similar Songs You've Liked")
        render_similar_historical_items(explanation.similar_historical_items)


def render_content_analysis(contributions: List[FeatureContribution]) -> None:
    """Render content feature analysis with radar chart and feature list"""
    
    # Create two columns: radar chart and feature list
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create radar chart
        radar_fig = create_audio_radar_chart(contributions)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with col2:
        # Feature contribution list
        st.markdown("**Top Contributing Features:**")
        for i, contrib in enumerate(contributions, 1):
            st.markdown(f"{i}. **{contrib.feature_name.replace('_', ' ').title()}**")
            st.caption(f"   Value: {contrib.qualitative_description}")
            st.caption(f"   Contribution: {contrib.contribution_score:.3f}")


def create_audio_radar_chart(contributions: List[FeatureContribution]) -> go.Figure:
    """Create radar chart for audio features"""
    
    # Audio features in order for radar chart
    audio_features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                     'liveness', 'valence', 'speechiness']
    
    # Create mapping from contributions
    contrib_map = {c.feature_name: c.item_value for c in contributions}
    
    # Get values for radar chart
    values = []
    labels = []
    
    for feature in audio_features:
        if feature in contrib_map:
            values.append(contrib_map[feature])
            labels.append(feature.title())
        else:
            values.append(0)
            labels.append(feature.title())
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Audio Profile',
        line_color='rgb(32, 201, 151)',
        fillcolor='rgba(32, 201, 151, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            )),
        showlegend=False,
        title="Audio Feature Profile",
        font=dict(size=12),
        height=400
    )
    
    return fig


def render_collaborative_insights(insight: CollaborativeInsight) -> None:
    """Render collaborative filtering insights"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Collaborative Similarity", 
            f"{insight.cf_similarity_score:.3f}",
            help="How similar this recommendation is to collaborative patterns"
        )
    
    with col2:
        st.metric(
            "CF Contribution", 
            f"{insight.cf_contribution:.3f}",
            help="How much collaborative filtering contributed to the score"
        )
    
    if insight.similar_users:
        st.markdown("**Users with similar taste:**")
        for user_id, similarity in insight.similar_users:
            st.caption(f"• User {user_id} (similarity: {similarity:.3f})")


def render_similar_historical_items(items: List[SimilarHistoricalItem]) -> None:
    """Render similar historical items"""
    
    for item in items:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{item.name}** — {item.artist}")
            
            with col2:
                st.caption(f"Similarity: {item.similarity_score:.3f}")


def render_explanation_summary(explanation: RecommendationExplanation) -> None:
    """Render a compact explanation summary"""
    
    with st.container():
        # Top contributing features
        top_features = explanation.content_contributions[:3]
        feature_text = ", ".join([f"{f.feature_name.replace('_', ' ').title()} ({f.qualitative_description})" 
                                 for f in top_features])
        
        # Similar users
        similar_users_text = ", ".join([f"User {uid}" for uid, _ in explanation.collaborative_insight.similar_users[:2]])
        
        # Similar historical items
        similar_songs_text = ", ".join([f"{item.name}" for item in explanation.similar_historical_items[:2]])
        
        # Create explanation text
        explanation_parts = []
        
        if feature_text:
            explanation_parts.append(f"**Audio features:** {feature_text}")
        
        if similar_users_text:
            explanation_parts.append(f"**Similar users:** {similar_users_text}")
        
        if similar_songs_text:
            explanation_parts.append(f"**Similar to:** {similar_songs_text}")
        
        if explanation_parts:
            st.markdown("**Why recommended:** " + " | ".join(explanation_parts))


def render_user_evaluation_metrics(evaluator: ExplainabilityEvaluator, user_id: str, track_ids: List[str], scores: List[float]) -> None:
    """Render evaluation metrics for a user's recommendations"""
    
    # Evaluate coverage
    recommendations = list(zip(track_ids, scores))
    coverage = evaluator.evaluate_coverage(recommendations, user_id)
    
    # Evaluate stability and efficiency for first few recommendations
    stability_results = []
    efficiency_results = []
    
    for i, (track_id, score) in enumerate(recommendations[:3]):  # Test first 3 recommendations
        # Stability evaluation
        stability = evaluator.evaluate_stability(user_id, track_id, score, n_runs=3)
        stability_results.append(stability)
        
        # Efficiency evaluation
        efficiency = evaluator.evaluate_efficiency(user_id, track_id, score, n_runs=3)
        efficiency_results.append(efficiency)
    
    # Display metrics
    st.subheader("📊 User-Level Evaluation Metrics")
    
    # Coverage metrics
    st.markdown("**Coverage Metrics:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Coverage %", f"{coverage.coverage_percentage:.1f}%")
    with col2:
        st.metric("Successful", coverage.successful_explanations)
    with col3:
        st.metric("Failed", coverage.failed_explanations)
    
    if stability_results:
        # Stability metrics
        st.markdown("**Stability Metrics:**")
        avg_stability = StabilityMetrics(
            feature_consistency=np.mean([s.feature_consistency for s in stability_results]),
            user_consistency=np.mean([s.user_consistency for s in stability_results]),
            score_consistency=np.mean([s.score_consistency for s in stability_results])
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Feature Consistency", f"{avg_stability.feature_consistency:.3f}")
        with col2:
            st.metric("User Consistency", f"{avg_stability.user_consistency:.3f}")
        with col3:
            st.metric("Score Consistency", f"{avg_stability.score_consistency:.3f}")
    
    if efficiency_results:
        # Efficiency metrics
        st.markdown("**Efficiency Metrics:**")
        avg_efficiency = EfficiencyMetrics(
            avg_generation_time=np.mean([e.avg_generation_time for e in efficiency_results]),
            min_generation_time=np.min([e.min_generation_time for e in efficiency_results]),
            max_generation_time=np.max([e.max_generation_time for e in efficiency_results])
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Time (s)", f"{avg_efficiency.avg_generation_time:.4f}")
        with col2:
            st.metric("Min Time (s)", f"{avg_efficiency.min_generation_time:.4f}")
        with col3:
            st.metric("Max Time (s)", f"{avg_efficiency.max_generation_time:.4f}")


def render_individual_evaluation_metrics(evaluator: ExplainabilityEvaluator, user_id: str, track_id: str, score: float) -> None:
    """Render evaluation metrics for an individual recommendation"""
    
    with st.expander("🔍 Individual Evaluation Metrics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Quick stability test
            st.markdown("**Stability Test (3 runs):**")
            try:
                stability = evaluator.evaluate_stability(user_id, track_id, score, n_runs=3)
                st.metric("Feature Consistency", f"{stability.feature_consistency:.3f}")
                st.metric("User Consistency", f"{stability.user_consistency:.3f}")
            except Exception as e:
                st.error(f"Stability test failed: {e}")
        
        with col2:
            # Quick efficiency test
            st.markdown("**Efficiency Test (3 runs):**")
            try:
                efficiency = evaluator.evaluate_efficiency(user_id, track_id, score, n_runs=3)
                st.metric("Avg Time (s)", f"{efficiency.avg_generation_time:.4f}")
                st.metric("Min Time (s)", f"{efficiency.min_generation_time:.4f}")
            except Exception as e:
                st.error(f"Efficiency test failed: {e}")


