# Intro Cloud Computing Final Project - TMDB Analysis

## Abstract

Perform movie analysis on the TMDB dataset and present the analytics and machine learning prediction through Streamlit.

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` and fill in your TMDB credentials. Prefer the v4 bearer token; the v3 API key is used as a fallback.

   ```bash
   cp .env.example .env
   ```

3. Python version is pinned in [`.python-version`](.python-version) (3.13 locally; the project targets Python 3.12+ per [AGENTS.md](AGENTS.md)). All commands run via `uv run`.

## Pipeline commands

The ingestion pipeline has three stages, all idempotent:

| Stage  | Output                                                       | Command                              |
| ------ | ------------------------------------------------------------ | ------------------------------------ |
| Bronze | raw gzipped TMDB JSON (`discover` pages + per-movie details) | `uv run python -m src.ingest bronze` |
| Silver | typed Parquet: `movies`, `cast`, `crew`                      | `uv run python -m src.ingest silver` |
| Gold   | modeling-ready `gold_movies.parquet`                         | `uv run python -m src.ingest gold`   |
| All    | Bronze then Silver then Gold                                 | `uv run python -m src.ingest all`    |

The ML module has its own entry point (runs after Gold is built):

| Step   | Output                                                                 | Command                             |
| ------ | ---------------------------------------------------------------------- | ----------------------------------- |
| Train  | `data/ml/model_{revenue,rating}.joblib`, `metrics.json`, `model_card.md` | `uv run python -m src.ml train`     |
| Probe  | stdout revenue + rating prediction                                     | `uv run python -m src.ml predict …` |

Tunables are centralized in [src/config.py](src/config.py) and overridable via `.env` or process env. Common knobs:

- `SAMPLE_COUNTS` (default `1000`) — hard cap on number of movies to pull.
- `MIN_VOTE_COUNT` (default `10`) — TMDB `vote_count.gte` filter on discover.
- `START_YEAR` (default `1980`).
- `MIN_BUDGET_USD` (default `100000`) — Gold-only filter.
- `REQUESTS_PER_SECOND` (default `40`), `CONCURRENCY` (default `20`).

**ML / embedding overrides** (optional):

- `TMDB_ML_DEVICE` — `CPU` or `GPU` for CatBoost and the sentence-encoder device probe; unset auto-picks GPU when CUDA is available.
- `TMDB_ML_EMBEDDINGS` — set to `0` to skip MiniLM during Gold build (deterministic tiny-hash fallback vectors; useful offline / CI).
- `TMDB_ML_TABPFN` — set to `0` to disable the rating TabPFN blend (CatBoost-only; faster CI).
- `TMDB_ML_TABPFN_DEVICE` — `cpu`, `cuda`, or `mps` for TabPFN only. Default auto skips Apple MPS (TabPFN often OOMs on unified memory); use `mps` only if you have headroom.
- `TMDB_ML_TABPFN_BATCH` — TabPFN predict chunk size (default `256`).
- `TABPFN_TOKEN` — API key from [Prior Labs](https://ux.priorlabs.ai) to download TabPFN weights in non-interactive environments.

Quick end-to-end smoke run:

```bash
SAMPLE_COUNTS=50 uv run python -m src.ingest all
```

## Data layout

```
data/
  bronze/
    discover/<run-date>/page_NNNN.json.gz   # raw discover pages
    movies/<movie_id>.json.gz               # per-movie details + credits
  silver/
    movies.parquet                          # one row per movie (en, non-adult, 1980+)
                                            # + collection, tagline, overview
    cast.parquet                            # long, top-5 cast by billing order
    crew.parquet                            # long: Director + Producer
  gold/
    gold_movies.parquet                     # wide, budget>=100k and revenue>0, month grain
                                            # + cast_2..5, producer counts, collection,
                                            #   has_tagline, counts, overview_embedding (16 floats)
    overview_pca.joblib                     # PCA basis for overview text (Streamlit + Gold)
  ml/
    model_revenue.joblib                    # CatBoost pipeline + feature spec
    model_rating.joblib                     # CatBoost (+ optional TabPFN blend) + feature spec
    metrics.json                            # holdout, 5-fold CV, Ridge baseline, importances
    model_card.md                           # human-readable summary after each train
```

The Gold table is the modeling-ready dataset for the Streamlit analytics and ML pages. It joins director, top-5 cast, and producer aggregates onto Silver movies, adds `budget_musd`, `revenue_musd`, `roi`, `lead_production_company`, franchise/collection fields, tagline flag, list counts, and a 16-dimensional PCA-compressed MiniLM embedding of `overview` (see [src/ingest/embeddings.py](src/ingest/embeddings.py)).

## Streamlit app

Launch the app (pages defined under `src/app/`):

```bash
uv run streamlit run src/app/Home.py
```

### Pages

1. [x] **Overview** (`src/app/Home.py`) — abstract, pipeline diagram, scope constraints, and per-layer data metadata (file count, size, row count, last updated) plus headline Gold stats.
2. [x] **Sample data pull / inspect** (`src/app/pages/1_Sample_Data.py`) — filter the Gold table by year, genres, director, and budget; preview rows with selectable columns, summary statistics, and CSV download.
3. [x] **Data analytics** (`src/app/pages/2_Analytics.py`) — genre ROI, director leaderboard, hit/flop scatter, plus a release-month × genre seasonality heatmap and a top-production-companies leaderboard. All charts share sidebar filters (year range, minimum `vote_count`, genres).
4. [x] **ML prediction** (`src/app/pages/3_ML_Prediction.py`) — **revenue**: `CatBoostRegressor`. **rating**: CatBoost blended with `TabPFNRegressor` when weights are available and TabPFN fits successfully; otherwise CatBoost-only. Findings tab shows hold-out / 5-fold CV metrics vs a Ridge + `TargetEncoder` baseline, permutation importances, predicted-vs-actual scatter, and blend component metrics when applicable. Predict tab runs single-row inference (budget, runtime, year/month, genres, director, studio, top-5 cast, producer, collection, tagline flag, optional overview text embedded with the same MiniLM + PCA as Gold).

### Analytics page

- Average ROI for each genre (bar chart)
- Top director by avg vote average and avg movie budget (line on bar chart)
- ROI scatter plot with 3 regions: hit, average, flop
- Release-month × genre seasonality heatmap — median ROI / median revenue / movie-count toggle; cells with `n < 3` are blanked to avoid over-reading thin samples. Ties directly to the `release_month` ML feature.
- Top production companies leaderboard — avg revenue bars coloured by median ROI; studio-level complement to the director chart and ML studio multi-hot features.

### ML prediction page

**Models.** Revenue uses a **`CatBoostRegressor`** pipeline (native categorical columns, GPU when available, CPU fallback). Rating uses the same CatBoost model, optionally combined with **`TabPFNRegressor`** (tabular transformer): out-of-fold predictions from both models are stacked and a small **`Ridge(positive=True)`** meta-learner learns convex blend weights. If TabPFN is unavailable (import failure, missing `TABPFN_TOKEN`, or fit/predict error), training falls back to CatBoost-only. A **Ridge + `TargetEncoder`** baseline still target-encodes only the three original high-cardinality columns (`director_name`, `lead_production_company`, `lead_cast_name`) for a fair linear comparator.

**Features (v2).**

- Numeric: `budget_musd`, `runtime`, `release_year`, `n_production_companies`, `n_genres`, `n_cast`, `n_producers`, `has_tagline`, `has_collection`.
- Cyclical: `release_month` → `month_sin`, `month_cos`.
- Multi-hot: top-25 genres and top-30 production companies at fit time.
- Categorical (strings, CatBoost `cat_features`): director, lead studio, lead cast, `cast_2_name` … `cast_5_name`, `lead_producer_name`, `collection_name` (`Standalone` when not in a franchise).
- Text: 16 columns `overview_emb_0` … `overview_emb_15` from Gold’s cached embedding (Streamlit embeds user overview text with the persisted PCA in `data/gold/overview_pca.joblib`).

**Targets (2 models).**

- **Revenue:** trained on `log1p(revenue_musd)`; predictions are back-transformed with `expm1` and reported in million USD.
- **Rating:** trained on `vote_average` (0–10) with row weights `log1p(vote_count)` so highly-rated obscure titles do not dominate. Predictions are clipped to `[0, 10]`.

**Evaluation.** 80/20 hold-out split plus 5-fold CV for CatBoost (R², MAE, RMSE). Ridge + `TargetEncoder` hold-out metrics are reported as a sanity baseline. Permutation importances (scoring = R²) are computed on the hold-out split from the CatBoost pipeline.

**CLI predict** (optional extra flags for the v2 feature frame):

```bash
uv run python -m src.ml predict \
  --budget 80 --runtime 120 --year 2024 --month 6 \
  --genres Action,Adventure \
  --director "Christopher Nolan" \
  --studio "Warner Bros. Pictures" \
  --cast "Leonardo DiCaprio" \
  --producer "Emma Thomas" \
  --collection "Standalone"
```

**Artifacts.** Each bundle (`data/ml/model_*.joblib`) carries the fitted predictor, the feature spec (top genres and top studios), reported metrics, permutation importances, and hold-out predictions for the Streamlit scatter plot. `model_card.md` is regenerated on every successful `train` run.

## Tests & checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

See [AGENTS.md](AGENTS.md) for the authoritative project rules (package manager, scope, data contracts).
