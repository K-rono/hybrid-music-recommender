from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "training"


def get_dataset_paths() -> Tuple[Path, Path]:
    music_csv = TRAINING_DIR / "music_list.csv"
    behavior_csv = TRAINING_DIR / "user_behavior_list.csv"
    return music_csv, behavior_csv


def load_music_and_behavior() -> Tuple[pd.DataFrame, pd.DataFrame]:
    music_csv, behavior_csv = get_dataset_paths()
    music_df = pd.read_csv(music_csv)
    behavior_df = pd.read_csv(behavior_csv)
    return music_df, behavior_df


def get_all_user_ids(behavior_df: pd.DataFrame) -> List[str]:
    user_ids = behavior_df["user_id"].dropna().astype(str).unique().tolist()
    return user_ids


def validate_user_id(user_id: str, behavior_df: pd.DataFrame) -> bool:
    if user_id is None or user_id == "":
        return False
    return user_id in set(behavior_df["user_id"].astype(str))


