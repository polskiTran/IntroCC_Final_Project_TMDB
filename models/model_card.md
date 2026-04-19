# Model card — TMDB revenue & rating regressors

This file is **overwritten** on every `uv run python -m src.ml train`.

## Overview

Two **HistGradientBoostingRegressor** pipelines (revenue in M USD, user rating 0–10) with the same tabular features as described in the project README. See `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/models/metrics.json` for machine-readable metrics.

## Run

- **UTC time**: `2026-04-19T18:06:47+00:00`
- **Gold data source**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/data/gold/gold_movies.parquet`

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
| Holdout (HGB) | 0.5876 | 0.8557 | 1.0801 |
| 5-fold CV (HGB) | 0.5578 | 0.8502 | 1.0936 |
| Holdout Ridge baseline | 0.4884 | 0.9679 | 1.2030 |

**Revenue (original units, M USD)**

| Metric | Value |
| --- | --- |
| mae_musd | 69.7418 |
| mape_over_1M | 1.2303 |
| rmse_musd | 140.8505 |

**Permutation importance (top 12)**

| Feature | Importance (mean) | Std |
| --- | --- | --- |
| budget_musd | 0.543334 | 0.036146 |
| director_name | 0.051926 | 0.008397 |
| lead_production_company | 0.051826 | 0.010463 |
| lead_cast_name | 0.017151 | 0.005572 |
| runtime | 0.015374 | 0.003002 |
| genre_Drama | 0.011068 | 0.006231 |
| month_sin | 0.005434 | 0.001868 |
| genre_Romance | 0.005162 | 0.001235 |
| genre_Adventure | 0.004189 | 0.000268 |
| genre_Horror | 0.003422 | 0.001136 |
| genre_Comedy | 0.003376 | 0.001600 |
| genre_Thriller | 0.002927 | 0.001608 |

### rating

| Split | R² | MAE | RMSE |
| --- | --- | --- | --- |
| Holdout (HGB) | 0.3409 | 0.4530 | 0.5763 |
| 5-fold CV (HGB) | 0.3113 | 0.4469 | 0.5694 |
| Holdout Ridge baseline | 0.3380 | 0.4502 | 0.5775 |

**Permutation importance (top 12)**

| Feature | Importance (mean) | Std |
| --- | --- | --- |
| runtime | 0.262676 | 0.012824 |
| director_name | 0.097951 | 0.008459 |
| budget_musd | 0.055507 | 0.007163 |
| genre_Animation | 0.043820 | 0.004837 |
| lead_cast_name | 0.030094 | 0.005550 |
| genre_Drama | 0.026774 | 0.006307 |
| genre_Horror | 0.010495 | 0.003694 |
| genre_Comedy | 0.009232 | 0.001291 |
| genre_Action | 0.008250 | 0.002825 |
| genre_Thriller | 0.007108 | 0.002080 |
| lead_production_company | 0.006443 | 0.003493 |
| genre_Adventure | 0.004381 | 0.002091 |

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
