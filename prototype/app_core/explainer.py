from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px

from .recommender import HybridConfig


@dataclass
class FeatureContribution:
    """Container for feature contribution analysis"""
    feature_name: str
    user_value: float
    item_value: float
    contribution_score: float
    qualitative_description: str


@dataclass
class CollaborativeInsight:
    """Container for collaborative filtering insights"""
    cf_similarity_score: float
    similar_users: List[Tuple[str, float]]  # (user_id, similarity)
    cf_contribution: float


@dataclass
class SimilarHistoricalItem:
    """Container for similar historical items"""
    track_id: str
    similarity_score: float
    name: str
    artist: str


@dataclass
class RecommendationExplanation:
    """Complete explanation for a recommendation"""
    track_id: str
    track_name: str
    track_artist: str
    total_score: float
    content_contributions: List[FeatureContribution]
    collaborative_insight: CollaborativeInsight
    similar_historical_items: List[SimilarHistoricalItem]
    score_breakdown: Dict[str, float]


class RecommendationExplainer:
    """Main class for generating recommendation explanations"""
    
    def __init__(
        self,
        cf_pack: Dict,
        item_content: np.ndarray,
        item_hybrid: np.ndarray,
        feature_columns: List[str],
        train_df: pd.DataFrame,
        user_to_idx: Dict[str, int],
        item_to_idx: Dict[str, int],
        all_users: List[str],
        catalog_items: List[str],
        music_df: pd.DataFrame,
        config: HybridConfig
    ):
        self.cf_pack = cf_pack
        self.item_content = item_content
        self.item_hybrid = item_hybrid
        self.feature_columns = feature_columns
        self.train_df = train_df
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.all_users = all_users
        self.catalog_items = catalog_items
        self.music_df = music_df
        self.config = config
        
        # Create mappings for quick lookups
        self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        
        # Feature name mappings for better display
        self.feature_display_names = self._create_feature_display_names()
        
        # Build similarity index for historical items
        self._build_similarity_index()
    
    def _create_feature_display_names(self) -> Dict[str, str]:
        """Create human-readable feature names"""
        display_names = {}
        
        # Audio features
        audio_features = {
            'danceability': 'Danceability',
            'energy': 'Energy', 
            'loudness': 'Loudness',
            'speechiness': 'Speechiness',
            'acousticness': 'Acousticness',
            'instrumentalness': 'Instrumentalness',
            'liveness': 'Liveness',
            'valence': 'Valence',
            'tempo': 'Tempo',
            'year': 'Year'
        }
        
        # Genre features (one-hot encoded)
        genre_features = {}
        for col in self.feature_columns:
            if col.startswith('genre_'):
                genre_name = col.replace('genre_', '').replace('_', ' ').title()
                genre_features[col] = f"Genre: {genre_name}"
        
        # Hashed artist features
        hashed_artist_features = {}
        for col in self.feature_columns:
            if col.startswith('hashed_artist_'):
                hashed_artist_features[col] = "Artist Similarity"
        
        # Tag embedding features
        tag_features = {}
        for col in self.feature_columns:
            if col.startswith('tag_embedding_'):
                tag_features[col] = "Tag Similarity"
        
        display_names.update(audio_features)
        display_names.update(genre_features)
        display_names.update(hashed_artist_features)
        display_names.update(tag_features)
        
        return display_names
    
    def _build_similarity_index(self):
        """Build similarity index for finding similar historical items"""
        self.similarity_nn = NearestNeighbors(
            metric="cosine", 
            n_neighbors=50, 
            n_jobs=-1
        )
        self.similarity_nn.fit(self.item_hybrid)
    
    def _get_qualitative_description(self, feature_name: str, value: float) -> str:
        """Convert numeric feature value to qualitative description"""
        if feature_name in ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                           'liveness', 'valence', 'speechiness']:
            # 0-1 scale features
            if value >= 0.8:
                return "Very High"
            elif value >= 0.6:
                return "High"
            elif value >= 0.4:
                return "Medium"
            elif value >= 0.2:
                return "Low"
            else:
                return "Very Low"
        
        elif feature_name == 'loudness':
            # Loudness is typically -60 to 0 dB
            if value >= -5:
                return "Very Loud"
            elif value >= -15:
                return "Loud"
            elif value >= -30:
                return "Medium"
            elif value >= -45:
                return "Quiet"
            else:
                return "Very Quiet"
        
        elif feature_name == 'tempo':
            # Tempo in BPM
            if value >= 160:
                return "Very Fast"
            elif value >= 120:
                return "Fast"
            elif value >= 80:
                return "Medium"
            elif value >= 60:
                return "Slow"
            else:
                return "Very Slow"
        
        elif feature_name == 'year':
            # Year
            current_year = 2024
            age = current_year - value
            if age <= 2:
                return "Very Recent"
            elif age <= 5:
                return "Recent"
            elif age <= 10:
                return "Moderate Age"
            elif age <= 20:
                return "Older"
            else:
                return "Classic"
        
        else:
            # Default for other features
            if value >= 0.8:
                return "Very High"
            elif value >= 0.6:
                return "High"
            elif value >= 0.4:
                return "Medium"
            elif value >= 0.2:
                return "Low"
            else:
                return "Very Low"
    
    def _analyze_content_features(
        self, 
        user_hybrid: np.ndarray, 
        item_hybrid: np.ndarray, 
        item_idx: int
    ) -> List[FeatureContribution]:
        """Analyze content feature contributions"""
        # Extract content portions of hybrid vectors
        cf_dim = self.cf_pack["U"].shape[1]
        user_content = user_hybrid[cf_dim:]
        item_content = item_hybrid[cf_dim:]
        
        # Calculate feature-level contributions
        contributions = []
        
        for i, feature_name in enumerate(self.feature_columns):
            if i < len(user_content) and i < len(item_content):
                user_val = user_content[i]
                item_val = item_content[i]
                
                # Contribution is the product of user preference and item feature
                contribution = user_val * item_val
                
                # Get qualitative description
                qual_desc = self._get_qualitative_description(feature_name, item_val)
                
                contributions.append(FeatureContribution(
                    feature_name=feature_name,
                    user_value=float(user_val),
                    item_value=float(item_val),
                    contribution_score=float(contribution),
                    qualitative_description=qual_desc
                ))
        
        # Sort by contribution score and return top 5
        contributions.sort(key=lambda x: abs(x.contribution_score), reverse=True)
        return contributions[:5]
    
    def _analyze_collaborative_signal(
        self, 
        user_id: str, 
        user_hybrid: np.ndarray, 
        item_hybrid: np.ndarray
    ) -> CollaborativeInsight:
        """Analyze collaborative filtering contribution"""
        # Extract CF portions
        cf_dim = self.cf_pack["U"].shape[1]
        user_cf = user_hybrid[:cf_dim]
        item_cf = item_hybrid[:cf_dim]
        
        # Calculate CF similarity
        cf_similarity = np.dot(user_cf, item_cf)
        
        # Find similar users based on CF embeddings
        user_idx = self.user_to_idx[user_id]
        user_cf_embedding = self.cf_pack["U"][user_idx]
        
        # Calculate similarities with all other users
        similarities = []
        for other_user_idx, other_user_id in self.idx_to_user.items():
            if other_user_idx != user_idx:
                other_cf = self.cf_pack["U"][other_user_idx]
                sim = np.dot(user_cf_embedding, other_cf)
                similarities.append((other_user_id, float(sim)))
        
        # Get top 5 similar users
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = similarities[:5]
        
        # Calculate CF contribution to total score
        cf_contribution = cf_similarity * self.config.lambda_cf
        
        return CollaborativeInsight(
            cf_similarity_score=float(cf_similarity),
            similar_users=similar_users,
            cf_contribution=float(cf_contribution)
        )
    
    def _find_similar_historical_items(
        self, 
        user_id: str, 
        recommended_track_id: str, 
        item_idx: int
    ) -> List[SimilarHistoricalItem]:
        """Find similar items from user's history"""
        # Get user's historical items
        user_history = self.train_df[self.train_df["user_id"] == user_id]
        if user_history.empty:
            return []
        
        # Get recommended item's hybrid vector
        recommended_vector = self.item_hybrid[item_idx].reshape(1, -1)
        
        # Find similar items using the similarity index
        distances, indices = self.similarity_nn.kneighbors(recommended_vector, n_neighbors=20)
        distances, indices = distances[0], indices[0]
        
        # Filter to only user's historical items
        similar_items = []
        user_track_ids = set(user_history["track_id"].astype(str))
        
        for idx, distance in zip(indices, distances):
            track_id = self.catalog_items[idx]
            if track_id in user_track_ids and track_id != recommended_track_id:
                # Get track info
                track_info = self.music_df[self.music_df["track_id"].astype(str) == track_id]
                if not track_info.empty:
                    similar_items.append(SimilarHistoricalItem(
                        track_id=track_id,
                        similarity_score=float(1 - distance),  # Convert distance to similarity
                        name=track_info.iloc[0].get("name", "Unknown"),
                        artist=track_info.iloc[0].get("artist", "Unknown")
                    ))
        
        # Return top 3 similar historical items
        return similar_items[:3]
    
    def explain_recommendation(
        self, 
        user_id: str, 
        track_id: str, 
        score: float
    ) -> RecommendationExplanation:
        """Generate complete explanation for a recommendation"""
        # Get track info
        track_info = self.music_df[self.music_df["track_id"].astype(str) == track_id]
        track_name = track_info.iloc[0].get("name", "Unknown") if not track_info.empty else "Unknown"
        track_artist = track_info.iloc[0].get("artist", "Unknown") if not track_info.empty else "Unknown"
        
        # Get indices
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[track_id]
        
        # Build user hybrid vector
        from .recommender import make_user_hybrid
        user_hybrid = make_user_hybrid(
            user_idx, self.cf_pack, self.train_df, self.item_content,
            self.item_to_idx, self.all_users, self.config.lambda_cf, self.config.lambda_cb
        )
        
        # Get item hybrid vector
        item_hybrid = self.item_hybrid[item_idx]
        
        # Analyze content features
        content_contributions = self._analyze_content_features(
            user_hybrid, item_hybrid, item_idx
        )
        
        # Analyze collaborative signal
        collaborative_insight = self._analyze_collaborative_signal(
            user_id, user_hybrid, item_hybrid
        )
        
        # Find similar historical items
        similar_historical = self._find_similar_historical_items(
            user_id, track_id, item_idx
        )
        
        # Calculate score breakdown
        cf_dim = self.cf_pack["U"].shape[1]
        user_cf = user_hybrid[:cf_dim]
        item_cf = item_hybrid[:cf_dim]
        user_content = user_hybrid[cf_dim:]
        item_content = item_hybrid[cf_dim:]
        
        cf_score = np.dot(user_cf, item_cf) * self.config.lambda_cf
        content_score = np.dot(user_content, item_content) * self.config.lambda_cb
        
        score_breakdown = {
            "collaborative_score": float(cf_score),
            "content_score": float(content_score),
            "total_score": float(score)
        }
        
        return RecommendationExplanation(
            track_id=track_id,
            track_name=track_name,
            track_artist=track_artist,
            total_score=float(score),
            content_contributions=content_contributions,
            collaborative_insight=collaborative_insight,
            similar_historical_items=similar_historical,
            score_breakdown=score_breakdown
        )
    
    def create_radar_chart(self, explanation: RecommendationExplanation) -> go.Figure:
        """Create radar chart for audio features"""
        # Get audio features only
        audio_features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'speechiness']
        
        # Find corresponding contributions
        feature_values = []
        feature_names = []
        
        for contrib in explanation.content_contributions:
            if contrib.feature_name in audio_features:
                feature_values.append(contrib.item_value)
                feature_names.append(self.feature_display_names.get(
                    contrib.feature_name, contrib.feature_name
                ))
        
        if not feature_values:
            # Create empty radar chart
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[0, 0, 0, 0, 0, 0, 0],
                theta=['Danceability', 'Energy', 'Acousticness', 'Instrumentalness', 
                      'Liveness', 'Valence', 'Speechiness'],
                fill='toself',
                name='Audio Profile'
            ))
        else:
            # Create radar chart
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=feature_values,
                theta=feature_names,
                fill='toself',
                name='Audio Profile',
                line_color='rgb(32, 201, 151)'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Audio Feature Profile",
            font=dict(size=12)
        )
        
        return fig


# Caching system for explanations
class ExplanationCache:
    """Simple in-memory cache for explanations"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key: str) -> Optional[RecommendationExplanation]:
        """Get explanation from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, explanation: RecommendationExplanation) -> None:
        """Store explanation in cache"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = explanation
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()


# Global cache instance
_explanation_cache = ExplanationCache()


def get_cached_explanation(user_id: str, track_id: str, score: float) -> Optional[RecommendationExplanation]:
    """Get cached explanation"""
    cache_key = f"{user_id}_{track_id}_{score:.3f}"
    return _explanation_cache.get(cache_key)


def cache_explanation(user_id: str, track_id: str, score: float, explanation: RecommendationExplanation) -> None:
    """Cache explanation"""
    cache_key = f"{user_id}_{track_id}_{score:.3f}"
    _explanation_cache.put(cache_key, explanation)


def clear_explanation_cache() -> None:
    """Clear explanation cache"""
    _explanation_cache.clear()
