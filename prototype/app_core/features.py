from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "prototype" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "year",
]


@dataclass
class FeatureArtifacts:
    scaler: MinMaxScaler
    encoder: OneHotEncoder
    hasher_artist: FeatureHasher
    feature_columns: List[str]
    item_features_df: pd.DataFrame  # includes track_id as a column


def _fit_transform_features(music_df: pd.DataFrame) -> FeatureArtifacts:
    # Preserve track ids
    original_track_ids = music_df["track_id"].values

    # Numeric scaling
    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(music_df[NUMERIC_FEATURES])
    scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=NUMERIC_FEATURES)

    # One-hot encode genre
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_genre = encoder.fit_transform(music_df[["genre"]])
    ohe_feature_names = encoder.get_feature_names_out(["genre"]).tolist()
    encoded_genre_df = pd.DataFrame(encoded_genre, columns=ohe_feature_names)

    # Feature hashing for artist and sentence-embeddings for tags
    artist_input = [[str(x)] if pd.notna(x) else [] for x in music_df["artist"]]
    tags_text = music_df["tags"].fillna("").astype(str).tolist()

    hasher_artist = FeatureHasher(n_features=500, input_type="string")
    hashed_artist = hasher_artist.fit_transform(artist_input).toarray()
    hashed_artist_df = pd.DataFrame(
        hashed_artist, columns=[f"hashed_artist_{i}" for i in range(500)]
    )

    # Sentence embedding for tags
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    tag_embeddings = sentence_model.encode(tags_text, convert_to_numpy=True)
    tag_embeddings_df = pd.DataFrame(
        tag_embeddings, columns=[f"tag_embedding_{i}" for i in range(tag_embeddings.shape[1])]
    )

    item_features_df = pd.concat(
        [scaled_numeric_df, hashed_artist_df, tag_embeddings_df], axis=1
    )
    item_features_df["track_id"] = original_track_ids

    feature_columns = item_features_df.columns.drop("track_id").tolist()

    return FeatureArtifacts(
        scaler=scaler,
        encoder=encoder,
        hasher_artist=hasher_artist,
        feature_columns=feature_columns,
        item_features_df=item_features_df,
    )


def build_and_save_artifacts(music_df: pd.DataFrame) -> FeatureArtifacts:
    artifacts = _fit_transform_features(music_df)
    # Save artifacts
    joblib.dump(artifacts.scaler, ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(artifacts.encoder, ARTIFACTS_DIR / "encoder.joblib")
    joblib.dump(artifacts.hasher_artist, ARTIFACTS_DIR / "hasher_artist.joblib")
    artifacts.item_features_df.to_parquet(ARTIFACTS_DIR / "item_features.parquet")
    Path(ARTIFACTS_DIR / "feature_columns.json").write_text(
        json.dumps(artifacts.feature_columns)
    )
    return artifacts


def load_artifacts() -> FeatureArtifacts:
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    encoder = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
    hasher_artist = joblib.load(ARTIFACTS_DIR / "hasher_artist.joblib")
    item_features_df = pd.read_parquet(ARTIFACTS_DIR / "item_features.parquet")
    feature_columns = json.loads(Path(ARTIFACTS_DIR / "feature_columns.json").read_text())
    return FeatureArtifacts(
        scaler=scaler,
        encoder=encoder,
        hasher_artist=hasher_artist,
        feature_columns=feature_columns,
        item_features_df=item_features_df,
    )


def artifacts_exist() -> bool:
    expected = [
        ARTIFACTS_DIR / "scaler.joblib",
        ARTIFACTS_DIR / "encoder.joblib",
        ARTIFACTS_DIR / "hasher_artist.joblib",
        ARTIFACTS_DIR / "item_features.parquet",
        ARTIFACTS_DIR / "feature_columns.json",
    ]
    return all(p.exists() for p in expected)


