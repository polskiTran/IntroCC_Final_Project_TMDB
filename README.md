# Intro Cloud Computing Final Project - TMDB Analysis

## Abstract
Perform movie analysis on TMDB dataset and present the analytics and machine learning prediction through Streamlit.

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Copy `.env.example` to `.env` and fill in your TMDB credentials. Prefer the v4 bearer token; the v3 API key is used as a fallback.
   ```bash
   cp .env.example .env
   ```
3. Python 3.13+ is required (see `.python-version`). All commands run via `uv run`.

## Pipeline commands

The ingestion pipeline has three stages, all idempotent:

| Stage  | Output                                                      | Command                                   |
| ------ | ----------------------------------------------------------- | ----------------------------------------- |
| Bronze | raw gzipped TMDB JSON (`discover` pages + per-movie details) | `uv run python -m src.ingest bronze`      |
| Silver | typed Parquet: `movies`, `cast`, `crew`                      | `uv run python -m src.ingest silver`      |
| Gold   | modeling-ready `gold_movies.parquet`                         | `uv run python -m src.ingest gold`        |
| All    | Bronze then Silver then Gold                                 | `uv run python -m src.ingest all`         |

The ML module has its own entry point (runs after Gold is built):

| Step  | Output                                  | Command                             |
| ----- | --------------------------------------- | ----------------------------------- |
| Train | `models/revenue_model/model_revenue.joblib`, `models/rating_model/model_rating.joblib`, `models/metrics.json`, `models/model_card.md` | `uv run python -m src.ml train`     |
| Probe | stdout revenue + rating prediction      | `uv run python -m src.ml predict …` |

Tunables are centralized in [src/config.py](src/config.py) and overridable via `.env` or process env. Common knobs:

- `DATA_BACKEND` (`local` or `s3`, default `local`) — store Bronze/Silver/Gold under `data/` or in an S3 bucket. When `s3`, set `S3_BUCKET` (required), optional `S3_PREFIX`, and `AWS_REGION`; use standard AWS credentials (env vars, profile, or IAM role). The project depends on `awscrt` so boto3 can use AWS IAM Identity Center / SSO-style login providers when needed.
- `SAMPLE_COUNTS` (default `1000`) — hard cap on number of movies to pull.
- `MIN_VOTE_COUNT` (default `10`) — TMDB `vote_count.gte` filter on discover.
- `START_YEAR` (default `1980`).
- `MIN_BUDGET_USD` (default `100000`) — Gold-only filter.
- `REQUESTS_PER_SECOND` (default `40`), `CONCURRENCY` (default `20`).
- `DISCOVER_PAGE_CONCURRENCY` (default `15`) — concurrent `/discover/movie` pages (bounded).
- `SAVE_DISCOVER_PAGES` (default `false`) — persist raw discover JSON pages under `bronze/discover/` (off by default).

Quick end-to-end smoke run:

```bash
SAMPLE_COUNTS=50 uv run python -m src.ingest all
```

## Data layout

### Local (`DATA_BACKEND=local`, default)

```
data/
  bronze/
    manifests/run_date=YYYY-MM-DD/run_manifest.json   # sampled ids + fetch stats
    discover/<run-date>/page_NNNN.json.gz   # optional raw discover pages (SAVE_DISCOVER_PAGES=true)
    movies/id_prefix=NNN/<movie_id>.json.gz           # per-movie details + credits (hive-style prefixes)
  silver/
    movies.parquet                          # one row per movie (en, non-adult, 1980+)
    cast.parquet                            # long, top-N lead cast
    crew.parquet                            # long, filtered to Director and Producer
  gold/
    gold_movies.parquet                     # wide, budget>=100k and revenue>0, month grain
models/                                     # ML artifacts (not under data/)
  revenue_model/
    model_revenue.joblib                    # HGB pipeline + fit-time feature spec
  rating_model/
    model_rating.joblib                     # HGB pipeline + fit-time feature spec
  metrics.json                              # holdout, 5-fold CV, Ridge baseline, importances (both targets)
  model_card.md                             # human-readable card (overwritten each train run)
```

### S3 (`DATA_BACKEND=s3`)

The same relative paths under `s3://<S3_BUCKET>/<S3_PREFIX>/` (omit prefix segments when `S3_PREFIX` is empty):

- `bronze/manifests/...`, `bronze/discover/...`, `bronze/movies/id_prefix=NNN/...`
- `silver/*.parquet`
- `gold/gold_movies.parquet`

Trained models and metrics remain on disk under `models/` (not uploaded to S3).

The Gold table is the modeling-ready dataset for the Streamlit analytics and ML pages. It joins lead director and lead cast onto the Silver movies and adds `budget_musd`, `revenue_musd`, `roi`, and `lead_production_company`.

## Streamlit app

Launch the app (pages defined under `src/app/`):

```bash
uv run streamlit run src/app/Home.py
```

### Pages
1. [x] **Overview** (`src/app/Home.py`) — abstract, pipeline diagram, scope constraints, and per-layer data metadata (file count, size, row count, last updated) plus headline Gold stats.
2. [x] **Sample data pull / inspect** (`src/app/pages/1_Sample_Data.py`) — filter the Gold table by year, genres, director, and budget; preview rows with selectable columns, summary statistics, and CSV download.
3. [x] **Data analytics** (`src/app/pages/2_Analytics.py`) — genre ROI, director leaderboard, hit/flop scatter, plus a release-month × genre seasonality heatmap and a top-production-companies leaderboard. All charts share sidebar filters (year range, minimum `vote_count`, genres).
4. [x] **ML prediction** (`src/app/pages/3_ML_Prediction.py`) — two `HistGradientBoostingRegressor` models (revenue & rating) with leakage-safe target encoding. Findings tab shows hold-out / 5-fold CV metrics vs a Ridge baseline, permutation importances, and predicted-vs-actual scatter. Predict tab runs single-row inference from user inputs.

### Analytics page
- Average ROI for each genre (bar chart)
- Top director by avg vote average and avg movie budget (line on bar chart)
- ROI scatter plot with 3 regions: hit, average, flop
- Release-month × genre seasonality heatmap — median ROI / median revenue / movie-count toggle; cells with `n < 3` are blanked to avoid over-reading thin samples. Ties directly to the `release_month` ML feature.
- Top production companies leaderboard — avg revenue bars coloured by median ROI; studio-level complement to the director chart and the second target-encoded feature in the ML page.

### ML prediction page

**Model.** `sklearn.ensemble.HistGradientBoostingRegressor` wrapped in a `Pipeline`
with a `ColumnTransformer`. Gradient-boosted trees are a strong default for small
tabular data with a mix of numeric, cyclical, multi-hot, and high-cardinality
categorical features. The same architecture is used for both targets, so the
only difference between the two models is the target transform and sample
weighting.

**Features.**
- `budget_musd`, `runtime` — numeric passthrough.
- `release_month` — cyclical: `sin(2πm/12)`, `cos(2πm/12)`.
- `genres` — multi-hot over the top-15 most frequent genres seen at fit time.
- `director_name`, `lead_production_company`, `lead_cast_name` — leakage-safe
  `sklearn.preprocessing.TargetEncoder` (5-fold out-of-fold encoding, auto
  smoothing). Unseen categories at inference time fall back to the learned
  global prior.

**Targets (2 models).**
- Model A — revenue: trained on `log1p(revenue_musd)`; predictions are
  back-transformed with `expm1` and reported in million USD.
- Model B — rating: trained on `vote_average` (0–10) with row weights
  `log1p(vote_count)` so highly-rated obscure titles don't dominate. Predictions
  are clipped to `[0, 10]`.

**Evaluation.** 80/20 hold-out split plus 5-fold CV (R², MAE, RMSE). A Ridge
regression with the same feature pipeline is reported as a sanity baseline.
Permutation importances (scoring = R²) are computed on the hold-out split.

**Artifacts.** Each bundle under `models/revenue_model/` and `models/rating_model/`
carries the fitted pipeline, the feature spec (top genres), all reported metrics,
permutation importances, and the hold-out predictions used by the Streamlit scatter
plot so the app never has to retrain. A combined `models/metrics.json` is written
at the models root for easy inspection.

## Tests & checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

See [AGENTS.md](AGENTS.md) for the authoritative project rules (package manager, scope, data contracts).
