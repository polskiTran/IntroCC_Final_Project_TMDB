# Model card — TMDB revenue & rating regressors

This file is **overwritten** on every `uv run python -m src.ml train`.

## Overview

Two **HistGradientBoostingRegressor** pipelines (revenue in M USD, user rating 0–10) with the same tabular features as described in the project README. See `data/ml/metrics.json` for machine-readable metrics.

## Run

- **UTC time**: `2026-04-19T16:54:16+00:00`
- **Gold data source**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/data/gold/gold_movies.parquet`

## Artifacts

- **ML directory**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/data/ml`
- **Metrics JSON**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/data/ml/metrics.json`
- **Revenue bundle**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/data/ml/model_revenue.joblib`
- **Rating bundle**: `/Users/polski/Documents/GitHub/IntroCC_Final_Project_TMDB/data/ml/model_rating.joblib`

## Reproducibility

| Parameter | Value |
| --- | --- |
| cv_splits | `5` |
| random_state | `42` |
| test_size | `0.2` |

## Training data (per target)

### revenue

- **Rows (after filters)**: 5947 (train 4757, holdout 1190)
- **Target**: `revenue_musd` — Revenue (million USD)
- **Target transform**: `log1p`
- **Top genres (multi-hot)**: Drama, Comedy, Thriller, Action, Adventure, Crime, Romance, Horror, Science Fiction, Family, Fantasy, Mystery, Animation, History, Music
- **Feature columns (22)**: `budget_musd, runtime, month_sin, month_cos, genre_Drama, genre_Comedy, genre_Thriller, genre_Action, genre_Adventure, genre_Crime, genre_Romance, genre_Horror, genre_Science Fiction, genre_Family, genre_Fantasy, genre_Mystery, genre_Animation, genre_History, genre_Music, director_name, lead_production_company, lead_cast_name`

### rating

- **Rows (after filters)**: 5947 (train 4757, holdout 1190)
- **Target**: `vote_average` — User rating (0-10)
- **Target transform**: `identity`
- **Top genres (multi-hot)**: Drama, Comedy, Thriller, Action, Adventure, Crime, Romance, Horror, Science Fiction, Family, Fantasy, Mystery, Animation, History, Music
- **Feature columns (22)**: `budget_musd, runtime, month_sin, month_cos, genre_Drama, genre_Comedy, genre_Thriller, genre_Action, genre_Adventure, genre_Crime, genre_Romance, genre_Horror, genre_Science Fiction, genre_Family, genre_Fantasy, genre_Mystery, genre_Animation, genre_History, genre_Music, director_name, lead_production_company, lead_cast_name`

## Evaluation

### revenue

| Split | R² | MAE | RMSE |
| --- | --- | --- | --- |
| Holdout (HGB) | 0.5412 | 0.8453 | 1.0802 |
| 5-fold CV (HGB) | 0.5394 | 0.8549 | 1.1027 |
| Holdout Ridge baseline | 0.4883 | 0.9034 | 1.1407 |

**Revenue (original units, M USD)**

| Metric | Value |
| --- | --- |
| mae_musd | 61.4060 |
| mape_over_1M | 1.2467 |
| rmse_musd | 139.8778 |

**Permutation importance (top 12)**

| Feature | Importance (mean) | Std |
| --- | --- | --- |
| budget_musd | 0.427313 | 0.029438 |
| lead_production_company | 0.055397 | 0.005246 |
| director_name | 0.054574 | 0.002887 |
| lead_cast_name | 0.024707 | 0.003111 |
| runtime | 0.017762 | 0.003268 |
| genre_Drama | 0.013855 | 0.001891 |
| genre_Comedy | 0.009555 | 0.002004 |
| month_sin | 0.005338 | 0.001795 |
| genre_Family | 0.004375 | 0.000908 |
| genre_Science Fiction | 0.003828 | 0.001225 |
| genre_Horror | 0.003234 | 0.002689 |
| genre_Adventure | 0.002792 | 0.000801 |

### rating

| Split | R² | MAE | RMSE |
| --- | --- | --- | --- |
| Holdout (HGB) | 0.3041 | 0.4946 | 0.6398 |
| 5-fold CV (HGB) | 0.3191 | 0.4810 | 0.6198 |
| Holdout Ridge baseline | 0.3073 | 0.4943 | 0.6383 |

**Permutation importance (top 12)**

| Feature | Importance (mean) | Std |
| --- | --- | --- |
| runtime | 0.211571 | 0.017498 |
| genre_Animation | 0.083924 | 0.013884 |
| director_name | 0.051880 | 0.008903 |
| lead_cast_name | 0.025141 | 0.004970 |
| budget_musd | 0.023896 | 0.008248 |
| genre_Drama | 0.017273 | 0.007611 |
| genre_Action | 0.014439 | 0.002439 |
| genre_Thriller | 0.009950 | 0.003910 |
| lead_production_company | 0.007064 | 0.002118 |
| genre_Horror | 0.006660 | 0.003124 |
| genre_Comedy | 0.004412 | 0.001474 |
| genre_Adventure | 0.002700 | 0.002284 |

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
