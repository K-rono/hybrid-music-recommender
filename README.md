# Music Recommender (Hybrid + Explainability)

A hybrid music recommender that combines collaborative filtering (TruncatedSVD on TF–IDF user–item interactions) with content features (audio features, artist hashing, tag embeddings). It provides a Streamlit UI for interactive recommendations and explanations, plus a script to evaluate explainability metrics.

## Quickstart (Conda)

1) Create and activate environment
```bash
conda create -n musicrec python=3.11 -y
conda activate musicrec
```

2) Install dependencies
```bash
pip install -r prototype/requirements.txt
```

3) (Optional) Configure Spotify credentials
Some UI features (cover art/links/previews) use Spotify’s API.
```bash
# Windows PowerShell
env:SPOTIFY_CLIENT_ID = "<your_client_id>"
env:SPOTIFY_CLIENT_SECRET = "<your_client_secret>"
```

4) Run the Streamlit app
```bash
streamlit run prototype/streamlit_app.py
```
This will:
- Load datasets from `training/music_list.csv` and `training/user_behavior_list.csv`
- Build/load content artifacts under `prototype/artifacts/`
- Train a small SVD CF model, fuse with content, and serve recommendations with explanations

5) Generating recommendations
```
17a35e6658c1d128f24ff6b79a8d04f0125cd82f
09bea918cfe86e003a0b7bf860746e63995f15b0
ff6b6619a334a4489beb608266e4b9ff5ff86d23
```
To test, you can either choose a random ID from user_behavior_list.csv or choose a random one provided below

## Project Structure
- `prototype/`
  - `streamlit_app.py`: UI entry point
  - `app_core/`: core modules
    - `data_loader.py`: reads CSVs from `training/`
    - `features.py`: builds/saves/loads content features and artifacts
    - `recommender.py`: TF–IDF interactions, SVD CF, hybrid fusion, retrieval
    - `explainer.py`: per‑recommendation explanations
    - `evaluation_metrics.py`, `explainability_evaluator.py`: metrics and explainability evaluation
  - `artifacts/`: saved scaler/encoder/hasher and item feature parquet
  - `evaluate_explainability.py`: batch explainability evaluation (saves JSON)
  - `explainability_evaluation_results.json`: sample results
- `training/`
  - `music_list.csv`, `user_behavior_list.csv`: datasets required by the app
  - `standalone_evaluation_metrics.py`: portable metrics module
  - Notebooks: baselines/experiments (retained)

## Common Tasks

- Run explainability evaluation (small sampled run):
```bash
python prototype/evaluate_explainability.py
```
Outputs to `prototype/explainability_evaluation_results.json`.

- Rebuild content artifacts (first app run does this automatically):
```bash
python - << 'PY'
from prototype.app_core.features import build_and_save_artifacts
from prototype.app_core.data_loader import load_music_and_behavior
music_df, _ = load_music_and_behavior()
build_and_save_artifacts(music_df)
PY
```

## Notes
- Python 3.11 recommended.
- First run may download the `sentence-transformers` model; it will be cached subsequently.

## Contributors and File Ownership (FOR ACADEMIC EVALUATION PURPOSES)

This project was collaboratively developed. Primary areas of contribution are attributed as follows:

- JUNYI: UI/UX implementation and Spotify integration
- QIHONG: Explainability methods, metrics, and evaluation
- JAMAN: Hybrid filtering (collaborative + content) and retrieval pipeline

### File-level ownership (with overlaps noted)

- UI and Spotify (JUNYI)
  - `prototype/streamlit_app.py` — primary
  - `prototype/app_core/ui_helpers.py` — primary
  - `prototype/app_core/spotify_client.py` — primary

- Explainability (QIHONG)
  - `prototype/app_core/explainer.py` — primary
  - `prototype/evaluate_explainability.py` — primary
  - `prototype/app_core/evaluation_metrics.py` — primary (shared usage by recommender and evaluation)
  - `training/standalone_evaluation_metrics.py` — primary (portable metrics for experiments)

- Hybrid filtering and retrieval (JAMAN)
  - `prototype/app_core/recommender.py` — primary (SVD CF, hybrid fusion, ranking)
  - `prototype/hybrid_example.py` — primary (illustrative usage)
  - Notebooks in `training/` (e.g., `hybrid.py`, `Hybrid.ipynb`) — primary for experimentation

- Shared / supporting modules (overlap of authorship)
  - `prototype/app_core/data_loader.py` — shared (used by UI, hybrid, and explainability)
  - `prototype/app_core/features.py` — shared (content features/artifacts consumed by hybrid and explanations)
  - `prototype/artifacts/` — generated outputs (shared provenance)

Notes on overlaps:
- `evaluation_metrics.py` is authored by QIHONG but is consumed by both the hybrid and explainability evaluation flows.
- `features.py` underpins both the hybrid recommender (JAMAN) and explanation generation (QIHONG); UI (JUNYI) loads these artifacts.
- Data loading and schemas (`data_loader.py`) are shared across UI, hybrid, and explainability components.