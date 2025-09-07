from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


# Hybrid recommender configuration
@dataclass
class HybridConfig:
    n_components: int = 64  # SVD components
    lambda_cf: float = 0.7  # Collaborative filtering weight
    lambda_cb: float = 0.3  # Content-based weight
    topk_ann: int = 200     # ANN retrieval size
    reco_k: int = 20        # Final recommendations
    n_trees: int = 50       # Annoy trees


def build_tfidf_interactions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_items: List[str],
) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int], List[str]]:
    """
    Build TF-IDF weighted user-item interaction matrix for hybrid approach.
    
    Args:
        train_df: Training interactions
        test_df: Test interactions (for user space)
        catalog_items: List of track IDs in catalog
        
    Returns:
        Sparse matrix, user mappings, item mappings, user list
    """
    import time
    
    # Get all users from both train and test
    all_users = pd.Index(pd.concat([train_df["user_id"], test_df["user_id"]]).unique())
    catalog_items = pd.Index(catalog_items)
    
    # Create mappings
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    item_to_idx = {it: j for j, it in enumerate(catalog_items)}
    
    # Filter to catalog items only
    train_filtered = train_df[train_df["track_id"].isin(catalog_items)].copy()
    
    if train_filtered.empty:
        # Return empty matrix if no valid interactions
        return csr_matrix((len(all_users), len(catalog_items))), user_to_idx, item_to_idx, all_users.tolist()
    
    # Build interaction weights (log1p(playcount) with user normalization)
    train_filtered["log_play"] = np.log1p(train_filtered["playcount"].astype(float))
    
    # User-relative normalization
    user_sums = train_filtered.groupby("user_id")["log_play"].sum()
    train_filtered["normalized_play"] = train_filtered["log_play"] / train_filtered["user_id"].map(user_sums)
    
    # Filter out zero values
    train_filtered = train_filtered[train_filtered["normalized_play"] > 0].copy()
    
    if train_filtered.empty:
        return csr_matrix((len(all_users), len(catalog_items))), user_to_idx, item_to_idx, all_users.tolist()
    
    # Create sparse matrix
    rows = train_filtered["user_id"].map(user_to_idx).values
    cols = train_filtered["track_id"].map(item_to_idx).values
    data = train_filtered["normalized_play"].values.astype(np.float32)
    
    X = coo_matrix((data, (rows, cols)), shape=(len(all_users), len(catalog_items))).tocsr()
    
    # Apply TF-IDF weighting to reduce popularity bias
    n_users = X.shape[0]
    df_i = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log((n_users + 1) / (df_i + 1)) + 1.0
    X = X @ diags(idf.astype(np.float32), format="csr")
    
    return X, user_to_idx, item_to_idx, all_users.tolist()


def train_svd_cf(X_train: csr_matrix, n_components: int = 64, random_state: int = 42) -> Dict:
    """
    Train TruncatedSVD on user-item matrix to get collaborative filtering embeddings.
    
    Args:
        X_train: Sparse user-item matrix
        n_components: Number of SVD components
        random_state: Random seed
        
    Returns:
        Dictionary with user embeddings, item embeddings, and fitted models
    """
    import time
    import psutil
    
    # User embeddings
    start_time = time.time()
    svd_users = TruncatedSVD(
        n_components=n_components, 
        random_state=random_state,
        n_iter=3  # Reduce iterations for speed
    )
    U = svd_users.fit_transform(X_train)  # (n_users, k)
    user_time = time.time() - start_time
    
    # Item embeddings
    start_time = time.time()
    svd_items = TruncatedSVD(
        n_components=n_components, 
        random_state=random_state,
        n_iter=3  # Reduce iterations for speed
    )
    V = svd_items.fit_transform(X_train.T)  # (n_items, k)
    item_time = time.time() - start_time
    
    # L2 normalize for cosine similarity
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    
    return {
        "U": U.astype(np.float32),
        "V": V.astype(np.float32), 
        "svd_users": svd_users,
        "svd_items": svd_items
    }


def build_content_matrix(catalog_items: List[str], item_features_df: pd.DataFrame) -> np.ndarray:
    """
    Build content feature matrix aligned to catalog items.
    
    Args:
        catalog_items: List of track IDs in catalog
        item_features_df: DataFrame with item features
        
    Returns:
        Content feature matrix (n_items, n_features)
    """
    # Align features to catalog
    catalog_as_str = pd.Index([str(t) for t in catalog_items])
    
    # Get features for catalog items
    base = item_features_df.copy()
    base["track_id"] = base["track_id"].astype(str)
    base = base[base["track_id"].isin(catalog_as_str)]
    base = base.set_index("track_id").reindex(catalog_as_str)
    
    # Select numeric features only
    feature_columns = [col for col in base.columns if col != "track_id" and base[col].dtype in [np.number]]
    Xc = base[feature_columns].fillna(0.0).values.astype(np.float32)
    
    # L2 normalize rows for cosine similarity
    norms = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    Xc = Xc / norms
    
    return Xc


def make_item_hybrid(
    item_cf: np.ndarray, 
    item_content: np.ndarray, 
    lambda_cf: float = 0.7, 
    lambda_cb: float = 0.3
) -> np.ndarray:
    """
    Create hybrid item vectors by fusing CF and content features.
    
    Args:
        item_cf: Collaborative filtering item embeddings
        item_content: Content feature matrix
        lambda_cf: CF weight
        lambda_cb: Content weight
        
    Returns:
        Hybrid item vectors
    """
    # Normalize both parts
    V_cf = item_cf / (np.linalg.norm(item_cf, axis=1, keepdims=True) + 1e-12)
    V_cb = item_content / (np.linalg.norm(item_content, axis=1, keepdims=True) + 1e-12)
    
    # Concatenate with weights
    V = np.hstack([lambda_cf * V_cf, lambda_cb * V_cb]).astype(np.float32)
    
    # Final L2 normalization
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    
    return V


def make_user_hybrid(
    user_idx: int,
    cf_pack: Dict,
    train_df: pd.DataFrame,
    item_content: np.ndarray,
    item_to_idx: Dict[str, int],
    all_users: List[str],
    lambda_cf: float = 0.7,
    lambda_cb: float = 0.3
) -> np.ndarray:
    """
    Create hybrid user vector by fusing CF embedding with content profile.
    
    Args:
        user_idx: User index in the user space
        cf_pack: Collaborative filtering results
        train_df: Training interactions
        item_content: Content feature matrix
        item_to_idx: Item to index mapping
        all_users: List of all users
        lambda_cf: CF weight
        lambda_cb: Content weight
        
    Returns:
        Hybrid user vector
    """
    # Get CF embedding
    u_cf = cf_pack["U"][user_idx]
    
    # Build content profile from training history
    user_id = all_users[user_idx]
    user_interactions = train_df[train_df["user_id"] == user_id]
    
    if user_interactions.empty:
        u_cb = np.zeros(item_content.shape[1], dtype=np.float32)
    else:
        # Get content features for user's tracks
        user_track_ids = user_interactions["track_id"].astype(str).tolist()
        user_playcounts = user_interactions["playcount"].astype(float).values
        
        # Weight by playcount
        content_vectors = []
        weights = []
        
        for track_id, playcount in zip(user_track_ids, user_playcounts):
            if track_id in item_to_idx:
                item_idx = item_to_idx[track_id]
                content_vectors.append(item_content[item_idx])
                weights.append(playcount)
        
        if content_vectors:
            content_vectors = np.array(content_vectors)
            weights = np.array(weights)
            
            # Weighted average
            u_cb = np.average(content_vectors, axis=0, weights=weights)
            u_cb = u_cb / (np.linalg.norm(u_cb) + 1e-12)
        else:
            u_cb = np.zeros(item_content.shape[1], dtype=np.float32)
    
    # Concatenate with weights
    u = np.concatenate([lambda_cf * u_cf, lambda_cb * u_cb]).astype(np.float32)
    u = u / (np.linalg.norm(u) + 1e-12)
    
    return u


def build_ann(item_hybrid: np.ndarray, n_neighbors: int = 200) -> NearestNeighbors:
    """
    Build Approximate Nearest Neighbors index for hybrid item vectors.
    
    Args:
        item_hybrid: Hybrid item vectors
        n_neighbors: Number of neighbors to retrieve
        
    Returns:
        Fitted NearestNeighbors model
    """
    nn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(item_hybrid)
    return nn


def recommend_hybrid(
    user_id: str,
    cf_pack: Dict,
    ann: NearestNeighbors,
    train_df: pd.DataFrame,
    all_users: List[str],
    catalog_items: List[str],
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
    item_content: np.ndarray,
    item_hybrid: np.ndarray,
    config: HybridConfig,
    k: int = 20
) -> List[Tuple[str, float]]:
    """
    Generate hybrid recommendations for a user.
    
    Args:
        user_id: User ID to recommend for
        cf_pack: Collaborative filtering results
        ann: Fitted ANN model
        train_df: Training interactions
        all_users: List of all users
        catalog_items: List of catalog items
        user_to_idx: User to index mapping
        item_to_idx: Item to index mapping
        item_content: Content feature matrix
        item_hybrid: Hybrid item vectors
        config: Hybrid configuration
        k: Number of recommendations
        
    Returns:
        List of (track_id, score) tuples
    """
    if user_id not in user_to_idx:
        return []
    
    user_idx = user_to_idx[user_id]
    
    # Build hybrid user vector
    u = make_user_hybrid(
        user_idx, cf_pack, train_df, item_content, item_to_idx, all_users,
        config.lambda_cf, config.lambda_cb
    )
    
    # Query ANN
    distances, indices = ann.kneighbors(u.reshape(1, -1), n_neighbors=min(config.topk_ann, len(catalog_items)))
    distances, indices = distances[0], indices[0]
    scores = 1.0 - distances  # Convert distance to similarity
    
    # Filter out training items
    seen = set(train_df[train_df["user_id"] == user_id]["track_id"].astype(str))
    
    recommendations = []
    for idx, score in zip(indices, scores):
        track_id = catalog_items[idx]
        if track_id not in seen:
            recommendations.append((track_id, float(score)))
            if len(recommendations) >= k:
                break
    
    return recommendations


def build_sparse_user_item_matrix(
    train_df: pd.DataFrame,
    allowed_track_ids: List[str],
) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
    # Convert to string once and filter early
    train_df = train_df.copy()
    train_df["user_id"] = train_df["user_id"].astype(str)
    train_df["track_id"] = train_df["track_id"].astype(str)
    
    # Filter to only allowed tracks early to reduce data size
    allowed_track_ids = [str(t) for t in allowed_track_ids]
    allowed_track_set = set(allowed_track_ids)
    train_df = train_df[train_df["track_id"].isin(allowed_track_set)].copy()
    
    if train_df.empty:
        # Return empty matrix if no valid tracks
        user_ids = []
        user_id_to_row = {}
        track_id_to_col = {t: i for i, t in enumerate(allowed_track_ids)}
        return csr_matrix((0, len(allowed_track_ids))), user_id_to_row, track_id_to_col
    
    # Map users and tracks to indices
    user_ids = train_df["user_id"].unique().tolist()
    user_id_to_row = {u: i for i, u in enumerate(user_ids)}
    track_id_to_col = {t: i for i, t in enumerate(allowed_track_ids)}

    # Vectorized log1p calculation
    train_df["log_play"] = np.log1p(train_df["playcount"].astype(float))
    
    # Vectorized normalization using groupby
    denom_per_user = train_df.groupby("user_id")["log_play"].sum()
    train_df["normalized_play"] = train_df["log_play"] / train_df["user_id"].map(denom_per_user)
    
    # Filter out zero values
    train_df = train_df[train_df["normalized_play"] > 0].copy()
    
    if train_df.empty:
        # Return empty matrix if no valid interactions
        return csr_matrix((len(user_ids), len(allowed_track_ids))), user_id_to_row, track_id_to_col
    
    # Vectorized mapping to indices
    rows = train_df["user_id"].map(user_id_to_row).values
    cols = train_df["track_id"].map(track_id_to_col).values
    data = train_df["normalized_play"].values.astype(float)
    
    # Create sparse matrix
    uim = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(allowed_track_ids)))
    return uim, user_id_to_row, track_id_to_col


def build_annoy_index(item_features_df: pd.DataFrame, feature_columns: List[str]) -> Tuple[AnnoyIndex, Dict[int, str], Dict[str, int]]:
    print(f"Building Annoy index for {len(item_features_df)} items with {len(feature_columns)} features...")
    
    feature_length = len(feature_columns)
    annoy_index = AnnoyIndex(feature_length, "angular")

    # Reset index to ensure contiguous 0-based indexing for AnnoyIndex
    df_reset = item_features_df.reset_index(drop=True)
    
    # Pre-extract feature matrix and track_ids for vectorized operations
    print("Preparing feature matrix...")
    feature_matrix = df_reset[feature_columns].values.astype("float32")
    track_ids = df_reset["track_id"].astype(str).values
    
    # Create mappings efficiently using list comprehensions
    print("Creating index mappings...")
    idx_to_track_id = {i: str(track_id) for i, track_id in enumerate(track_ids)}
    track_id_to_idx = {str(track_id): i for i, track_id in enumerate(track_ids)}

    # Add items to AnnoyIndex using vectorized operations
    print("Adding items to Annoy index...")
    for i in range(len(feature_matrix)):
        annoy_index.add_item(i, feature_matrix[i])
        if i % 10000 == 0 and i > 0:  # Progress indicator for large datasets
            print(f"  Added {i:,} items...")

    # Build with adaptive number of trees for better accuracy
    n_trees = min(100, max(10, len(feature_matrix) // 1000))
    print(f"Building Annoy index with {n_trees} trees...")
    annoy_index.build(n_trees)
    
    print("Annoy index built successfully!")
    return annoy_index, idx_to_track_id, track_id_to_idx


def build_user_profile_sparse(
    user_id: str,
    uim: csr_matrix,
    user_id_to_row: Dict[str, int],
    track_id_to_col: Dict[str, int],
    item_features_df: pd.DataFrame,
    feature_columns: List[str],
) -> np.ndarray | None:
    if user_id not in user_id_to_row:
        return None
    row_idx = user_id_to_row[user_id]
    user_row = uim.getrow(row_idx)
    if user_row.nnz == 0:
        return None

    # Precompute reverse mapping from column index to track_id
    reverse_cols = [None] * uim.shape[1]
    for tid, c in track_id_to_col.items():
        reverse_cols[c] = tid

    # Precompute track_id to feature row mapping
    track_id_to_df_index = {str(t): i for i, t in enumerate(item_features_df["track_id"].astype(str).values)}
    
    # Pre-extract feature matrix for efficiency
    feature_matrix = item_features_df[feature_columns].values.astype(np.float64)
    
    profile = np.zeros(len(feature_columns), dtype=np.float64)
    
    for c, w in zip(user_row.indices, user_row.data):
        tid = reverse_cols[c]
        df_i = track_id_to_df_index.get(tid)
        if df_i is None:
            continue
        profile += w * feature_matrix[df_i]

    norm = np.linalg.norm(profile)
    if norm == 0:
        return None
    return profile / norm


def recommend_for_user_sparse(
    user_id: str,
    uim: csr_matrix,
    user_id_to_row: Dict[str, int],
    track_id_to_col: Dict[str, int],
    item_features_df: pd.DataFrame,
    feature_columns: List[str],
    annoy_index: AnnoyIndex,
    idx_to_track_id: Dict[int, str],
    top_n: int = 10,
    exclude_interacted_track_ids: List[str] | None = None,
) -> List[str]:
    user_profile = build_user_profile_sparse(
        user_id,
        uim,
        user_id_to_row,
        track_id_to_col,
        item_features_df,
        feature_columns,
    )
    if user_profile is None:
        return []

    nearest_idx = annoy_index.get_nns_by_vector(
        user_profile.astype("float32"),
        top_n * 3,
        include_distances=False,
    )
    rec_track_ids = [idx_to_track_id[idx] for idx in nearest_idx]

    if exclude_interacted_track_ids:
        exclude = set(str(t) for t in exclude_interacted_track_ids)
        rec_track_ids = [t for t in rec_track_ids if str(t) not in exclude]

    return rec_track_ids[:top_n]


class RecommendationScore(NamedTuple):
    """Container for recommendation with score"""
    track_id: str
    score: float


def get_content_based_scores(
    user_id: str,
    uim: csr_matrix,
    user_id_to_row: Dict[str, int],
    track_id_to_col: Dict[str, int],
    item_features_df: pd.DataFrame,
    feature_columns: List[str],
    annoy_index: AnnoyIndex,
    idx_to_track_id: Dict[int, str],
    candidate_track_ids: List[str] | None = None,
    exclude_interacted_track_ids: List[str] | None = None,
) -> List[RecommendationScore]:
    """
    Get content-based similarity scores for hybrid recommendation.
    
    Args:
        user_id: User ID to get recommendations for
        uim: User-item interaction matrix
        user_id_to_row: Mapping from user_id to matrix row
        track_id_to_col: Mapping from track_id to matrix column
        item_features_df: DataFrame with item features
        feature_columns: List of feature column names
        annoy_index: Annoy index for similarity search
        idx_to_track_id: Mapping from index to track_id
        candidate_track_ids: Optional list of candidate tracks to score
        exclude_interacted_track_ids: Optional list of tracks to exclude
        
    Returns:
        List of RecommendationScore tuples (track_id, score) sorted by score descending
    """
    user_profile = build_user_profile_sparse(
        user_id,
        uim,
        user_id_to_row,
        track_id_to_col,
        item_features_df,
        feature_columns,
    )
    if user_profile is None:
        return []

    # Get candidate tracks to score
    if candidate_track_ids is None:
        # Get all tracks from the index
        candidate_track_ids = list(idx_to_track_id.values())
    
    # Filter out excluded tracks
    if exclude_interacted_track_ids:
        exclude_set = set(str(t) for t in exclude_interacted_track_ids)
        candidate_track_ids = [t for t in candidate_track_ids if str(t) not in exclude_set]
    
    # Get feature matrix for candidate tracks
    candidate_features = []
    valid_candidate_ids = []
    
    for track_id in candidate_track_ids:
        if track_id in idx_to_track_id:
            # Find the index in the Annoy index
            track_idx = None
            for idx, tid in idx_to_track_id.items():
                if tid == track_id:
                    track_idx = idx
                    break
            
            if track_idx is not None:
                # Get features from item_features_df
                track_features = item_features_df[item_features_df["track_id"] == track_id]
                if not track_features.empty:
                    feature_vector = track_features[feature_columns].values[0].astype(np.float64)
                    candidate_features.append(feature_vector)
                    valid_candidate_ids.append(track_id)
    
    if not candidate_features:
        return []
    
    # Calculate similarity scores
    candidate_features = np.array(candidate_features)
    scores = []
    
    for i, track_id in enumerate(valid_candidate_ids):
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(user_profile, candidate_features[i])
        scores.append(RecommendationScore(track_id=track_id, score=similarity))
    
    # Sort by score descending
    scores.sort(key=lambda x: x.score, reverse=True)
    
    return scores


def get_content_based_scores_fast(
    user_id: str,
    uim: csr_matrix,
    user_id_to_row: Dict[str, int],
    track_id_to_col: Dict[str, int],
    item_features_df: pd.DataFrame,
    feature_columns: List[str],
    annoy_index: AnnoyIndex,
    idx_to_track_id: Dict[int, str],
    candidate_track_ids: List[str] | None = None,
    exclude_interacted_track_ids: List[str] | None = None,
    top_n: int = 1000,
) -> List[RecommendationScore]:
    """
    Fast version using Annoy index for initial filtering, then exact similarity calculation.
    More efficient for large candidate sets.
    
    Args:
        user_id: User ID to get recommendations for
        uim: User-item interaction matrix
        user_id_to_row: Mapping from user_id to matrix row
        track_id_to_col: Mapping from track_id to matrix column
        item_features_df: DataFrame with item features
        feature_columns: List of feature column names
        annoy_index: Annoy index for similarity search
        idx_to_track_id: Mapping from index to track_id
        candidate_track_ids: Optional list of candidate tracks to score
        exclude_interacted_track_ids: Optional list of tracks to exclude
        top_n: Number of top candidates to get from Annoy index
        
    Returns:
        List of RecommendationScore tuples (track_id, score) sorted by score descending
    """
    user_profile = build_user_profile_sparse(
        user_id,
        uim,
        user_id_to_row,
        track_id_to_col,
        item_features_df,
        feature_columns,
    )
    if user_profile is None:
        return []

    # Get initial candidates from Annoy index
    if candidate_track_ids is None:
        # Get top candidates from Annoy index
        nearest_idx = annoy_index.get_nns_by_vector(
            user_profile.astype("float32"),
            top_n,
            include_distances=True,
        )
        candidate_track_ids = [idx_to_track_id[idx] for idx in nearest_idx[0]]
    else:
        # Filter candidate tracks
        if exclude_interacted_track_ids:
            exclude_set = set(str(t) for t in exclude_interacted_track_ids)
            candidate_track_ids = [t for t in candidate_track_ids if str(t) not in exclude_set]
    
    # Calculate exact similarity scores for candidates
    scores = []
    
    for track_id in candidate_track_ids:
        # Get features for this track
        track_features = item_features_df[item_features_df["track_id"] == track_id]
        if not track_features.empty:
            feature_vector = track_features[feature_columns].values[0].astype(np.float64)
            # Calculate cosine similarity
            similarity = 1 - cosine(user_profile, feature_vector)
            scores.append(RecommendationScore(track_id=track_id, score=similarity))
    
    # Sort by score descending
    scores.sort(key=lambda x: x.score, reverse=True)
    
    return scores


def normalize_scores(scores: List[RecommendationScore], method: str = "minmax") -> List[RecommendationScore]:
    """
    Normalize scores to a 0-1 range for hybrid recommendation.
    
    Args:
        scores: List of RecommendationScore tuples
        method: Normalization method ("minmax", "zscore", "rank")
        
    Returns:
        List of normalized RecommendationScore tuples
    """
    if not scores:
        return scores
    
    score_values = [s.score for s in scores]
    
    if method == "minmax":
        min_score = min(score_values)
        max_score = max(score_values)
        if max_score == min_score:
            # All scores are the same
            normalized_scores = [0.5] * len(scores)
        else:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in score_values]
    
    elif method == "zscore":
        mean_score = np.mean(score_values)
        std_score = np.std(score_values)
        if std_score == 0:
            normalized_scores = [0.5] * len(scores)
        else:
            normalized_scores = [(s - mean_score) / std_score for s in score_values]
            # Convert to 0-1 range
            min_norm = min(normalized_scores)
            max_norm = max(normalized_scores)
            if max_norm != min_norm:
                normalized_scores = [(s - min_norm) / (max_norm - min_norm) for s in normalized_scores]
    
    elif method == "rank":
        # Rank-based normalization
        sorted_indices = sorted(range(len(score_values)), key=lambda i: score_values[i], reverse=True)
        normalized_scores = [0.0] * len(scores)
        for rank, idx in enumerate(sorted_indices):
            normalized_scores[idx] = 1.0 - (rank / (len(scores) - 1)) if len(scores) > 1 else 1.0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return [RecommendationScore(track_id=s.track_id, score=norm_score) 
            for s, norm_score in zip(scores, normalized_scores)]


