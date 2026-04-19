# Model card — TMDB revenue & rating regressors

This file is **overwritten** on every `uv run python -m src.ml train`.

## Overview

Two **HistGradientBoostingRegressor** pipelines (revenue in M USD, user rating 0–10) with the same tabular features as described in the project README. See `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/models/metrics.json` for machine-readable metrics.

## Run

- **UTC time**: `2026-04-19T19:41:40+00:00`
- **Gold data source**: `s3://tmdb-datalake-424865813912-us-east-2-an/gold/gold_movies.parquet`

## Artifacts

- **Models root**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/models`
- **Metrics JSON**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/models/metrics.json`
- **Revenue bundle**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/models/revenue_model/model_revenue.joblib`
- **Rating bundle**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/models/rating_model/model_rating.joblib`

## Reproducibility

| Parameter | Value |
| --- | --- |
| cv_splits | `5` |
| random_state | `42` |
| test_size | `0.2` |

## Training data (per target)

### revenue

- **Rows (after filters)**: 5535 (train 4428, holdout 1107)
- **Target**: `revenue_musd` — Revenue (million USD)
- **Target transform**: `log1p`
- **Top genres (multi-hot)**: Drama, Comedy, Thriller, Action, Adventure, Romance, Crime, Horror, Science Fiction, Family, Fantasy, Mystery, Animation, History, Music
- **Feature columns (22)**: `budget_musd, runtime, month_sin, month_cos, genre_Drama, genre_Comedy, genre_Thriller, genre_Action, genre_Adventure, genre_Romance, genre_Crime, genre_Horror, genre_Science Fiction, genre_Family, genre_Fantasy, genre_Mystery, genre_Animation, genre_History, genre_Music, director_name, lead_production_company, lead_cast_name`

### rating

- **Rows (after filters)**: 5535 (train 4428, holdout 1107)
- **Target**: `vote_average` — User rating (0-10)
- **Target transform**: `identity`
- **Top genres (multi-hot)**: Drama, Comedy, Thriller, Action, Adventure, Romance, Crime, Horror, Science Fiction, Family, Fantasy, Mystery, Animation, History, Music
- **Feature columns (22)**: `budget_musd, runtime, month_sin, month_cos, genre_Drama, genre_Comedy, genre_Thriller, genre_Action, genre_Adventure, genre_Romance, genre_Crime, genre_Horror, genre_Science Fiction, genre_Family, genre_Fantasy, genre_Mystery, genre_Animation, genre_History, genre_Music, director_name, lead_production_company, lead_cast_name`

## Evaluation

### revenue

| Split | R² | MAE | RMSE |
| --- | --- | --- | --- |
| Holdout (HGB) | 0.5875 | 0.8565 | 1.0801 |
| 5-fold CV (HGB) | 0.5595 | 0.8476 | 1.0917 |
| Holdout Ridge baseline | 0.4884 | 0.9678 | 1.2029 |

**Revenue (original units, M USD)**

| Metric | Value |
| --- | --- |
| mae_musd | 69.9128 |
| mape_over_1M | 1.2285 |
| rmse_musd | 141.6586 |

**Permutation importance (top 12)**

| Feature | Importance (mean) | Std |
| --- | --- | --- |
| budget_musd | 0.543661 | 0.037807 |
| lead_production_company | 0.054081 | 0.011239 |
| director_name | 0.052715 | 0.008273 |
| lead_cast_name | 0.019763 | 0.005655 |
| runtime | 0.013001 | 0.004111 |
| genre_Drama | 0.009159 | 0.005289 |
| month_sin | 0.005299 | 0.001033 |
| genre_Adventure | 0.004897 | 0.000380 |
| genre_Romance | 0.004676 | 0.000775 |
| genre_Horror | 0.003978 | 0.001213 |
| genre_Comedy | 0.003328 | 0.001723 |
| genre_Thriller | 0.003036 | 0.001614 |

### rating

| Split | R² | MAE | RMSE |
| --- | --- | --- | --- |
| Holdout (HGB) | 0.3403 | 0.4526 | 0.5765 |
| 5-fold CV (HGB) | 0.3121 | 0.4473 | 0.5691 |
| Holdout Ridge baseline | 0.3381 | 0.4502 | 0.5775 |

**Permutation importance (top 12)**

| Feature | Importance (mean) | Std |
| --- | --- | --- |
| runtime | 0.265062 | 0.012190 |
| director_name | 0.095610 | 0.007955 |
| budget_musd | 0.058061 | 0.005511 |
| genre_Animation | 0.043307 | 0.006121 |
| genre_Drama | 0.028596 | 0.006627 |
| lead_cast_name | 0.026540 | 0.006530 |
| genre_Horror | 0.012358 | 0.004083 |
| lead_production_company | 0.010573 | 0.003242 |
| genre_Thriller | 0.008359 | 0.002254 |
| genre_Action | 0.008304 | 0.002223 |
| genre_Comedy | 0.006820 | 0.001288 |
| genre_Adventure | 0.004139 | 0.002185 |

## System / device

- **Platform**: `macOS-26.4.1-arm64-arm-64bit-Mach-O`
- **Machine**: `arm64`
- **Processor**: `arm`
- **Python**: `3.13.11`
- **CPU cores (logical)**: 12
- **Approx. RAM**: 32.0 GiB
- **Libraries**: `joblib=1.5.3`, `numpy=2.4.4`, `polars=1.39.3`, `scikit-learn=1.8.0`

**GPU / accelerator**

```text
No NVIDIA GPU detected (nvidia-smi not available or no devices).
```
