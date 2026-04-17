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

Tunables are centralized in [src/config.py](src/config.py) and overridable via `.env` or process env. Common knobs:

- `SAMPLE_COUNTS` (default `1000`) — hard cap on number of movies to pull.
- `MIN_VOTE_COUNT` (default `10`) — TMDB `vote_count.gte` filter on discover.
- `START_YEAR` (default `1980`).
- `MIN_BUDGET_USD` (default `100000`) — Gold-only filter.
- `REQUESTS_PER_SECOND` (default `40`), `CONCURRENCY` (default `20`).

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
    cast.parquet                            # long, top-N lead cast
    crew.parquet                            # long, filtered to Director and Producer
  gold/
    gold_movies.parquet                     # wide, budget>=100k and revenue>0, month grain
```

The Gold table is the modeling-ready dataset for the Streamlit analytics and ML pages. It joins lead director and lead cast onto the Silver movies and adds `budget_musd`, `revenue_musd`, `roi`, and `lead_production_company`.

## Streamlit app

Launch the app (pages defined under `src/app/`):

```bash
uv run streamlit run src/app/Home.py
```

Note: the Streamlit pages are not yet implemented; this is the intended entry point once pages are added.

### Planned pages
1. Overview — project abstract + data metadata
2. Sample data pull / inspect
3. Data analytics
4. ML prediction

### Analytics page
- Average ROI for each genre (bar chart)
- Top director by avg vote average and avg movie budget (line on bar chart)
- ROI scatter plot with 3 regions: hit, average, flop

### ML prediction page

Input features:
- genres (one-hot encoding)
- budget (million dollars)
- release month (1–12)
- director (target encoding)
- production (target encoding)
- 1 lead cast (target encoding)

Targets (2 models):
- Model A: Revenue (million dollars)
- Model B: User rating

## Tests & checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

See [AGENTS.md](AGENTS.md) for the authoritative project rules (package manager, scope, data contracts).
