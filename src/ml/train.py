"""Train the TMDB revenue and rating models from the Gold parquet.

Produces two self-contained bundles under ``settings.ml_dir``:

- ``revenue_model/model_revenue.joblib``: predicts ``revenue_musd`` (trained on ``log1p``).
- ``rating_model/model_rating.joblib``: predicts ``vote_average`` in [0, 10].

Each bundle carries the fitted ``sklearn`` Pipeline, the ``FeatureSpec`` (top
genres), the target transform metadata, hold-out and 5-fold CV metrics, a Ridge
baseline, permutation importances, and a small held-out frame used to render
the predicted-vs-actual scatter in the Streamlit page without retraining.

The training run is idempotent: it fully rebuilds both bundles,
``metrics.json`` (under ``settings.ml_dir``), and the model card markdown at
``settings.model_card_path`` on every invocation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import Settings, get_settings
from src.ml.features import (
    NUMERIC_COLS,
    CYCLICAL_COLS,
    TARGET_ENCODED_COLS,
    FeatureSpec,
    build_feature_frame,
    fit_feature_spec,
)
from src.ml.model_card import (
    collect_system_info,
    format_system_info_for_log,
    render_model_card_markdown,
    utc_now_iso,
    write_model_card,
)
from src.ml.paths import (
    metrics_json_path,
    rating_bundle_path,
    revenue_bundle_path,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
CV_SPLITS = 5
TEST_SIZE = 0.2


@dataclass
class Metrics:
    r2: float
    mae: float
    rmse: float

    def as_dict(self) -> dict[str, float]:
        return {"r2": self.r2, "mae": self.mae, "rmse": self.rmse}


@dataclass
class ModelBundle:
    target: str
    target_label: str
    target_transform: str
    pipeline: Pipeline
    feature_spec: FeatureSpec
    feature_columns: list[str]
    holdout_metrics: Metrics
    cv_metrics: Metrics
    baseline_holdout_metrics: Metrics
    revenue_space_metrics: dict[str, float] | None
    permutation_importance: list[dict[str, str | float]]
    holdout_predictions: pd.DataFrame
    n_train: int
    n_test: int
    n_rows_total: int
    extra: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "target": self.target,
            "target_label": self.target_label,
            "target_transform": self.target_transform,
            "feature_columns": self.feature_columns,
            "holdout_metrics": self.holdout_metrics.as_dict(),
            "cv_metrics": self.cv_metrics.as_dict(),
            "baseline_ridge_holdout_metrics": self.baseline_holdout_metrics.as_dict(),
            "permutation_importance": self.permutation_importance,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_rows_total": self.n_rows_total,
            "top_genres": self.feature_spec.top_genres,
        }
        if self.revenue_space_metrics is not None:
            meta["revenue_space_metrics"] = self.revenue_space_metrics
        meta.update(self.extra)
        return meta


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        r2=float(r2_score(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def _build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    multihot = spec.multihot_cols
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(NUMERIC_COLS)),
            ("cyclical", "passthrough", list(CYCLICAL_COLS)),
            ("multihot", "passthrough", multihot),
            (
                "target_enc",
                TargetEncoder(
                    smooth="auto",
                    cv=CV_SPLITS,
                    random_state=RANDOM_STATE,
                ),
                list(TARGET_ENCODED_COLS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _build_hgb_pipeline(spec: FeatureSpec) -> Pipeline:
    preproc = _build_preprocessor(spec)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=500,
        max_depth=None,
        min_samples_leaf=10,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preproc", preproc), ("model", model)])


def _build_ridge_pipeline(spec: FeatureSpec) -> Pipeline:
    preproc = _build_preprocessor(spec)
    return Pipeline(
        [
            ("preproc", preproc),
            ("scale", StandardScaler(with_mean=True)),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )


def _cv_metrics(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Metrics:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def _score(scorer: str) -> np.ndarray:
        return cross_val_score(pipeline, X, y, scoring=scorer, cv=cv, n_jobs=1)

    r2 = float(_score("r2").mean())
    mae = float(-_score("neg_mean_absolute_error").mean())
    rmse = float(-_score("neg_root_mean_squared_error").mean())
    return Metrics(r2=r2, mae=mae, rmse=rmse)


def _permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: list[str],
) -> list[dict[str, str | float]]:
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=1,
    )
    rows = [
        {
            "feature": name,
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i, name in enumerate(feature_columns)
    ]
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    return rows


def _prepare_frame(df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    keep = df.drop_nulls(["budget_musd", "release_month", "genres", target_col])
    if target_col == "revenue_musd":
        keep = keep.filter(pl.col("revenue_musd") > 0)
    return keep


def _train_single(
    df: pl.DataFrame,
    *,
    target_col: str,
    target_label: str,
    target_transform: str,
    sample_weight_col: str | None = None,
) -> ModelBundle:
    prepared = _prepare_frame(df, target_col)
    if prepared.height < 20:
        raise ValueError(
            f"Not enough rows to train {target_col}: got {prepared.height}, need >= 20."
        )

    spec = fit_feature_spec(prepared)
    X_all = build_feature_frame(prepared, spec)
    y_raw = prepared[target_col].to_numpy().astype(np.float64)
    y_trans = np.log1p(y_raw) if target_transform == "log1p" else y_raw
    weights = (
        prepared[sample_weight_col].to_numpy().astype(np.float64)
        if sample_weight_col is not None
        else None
    )
    if weights is not None:
        weights = np.log1p(np.clip(weights, 0.0, None))

    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X_all,
        y_trans,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    if weights is not None:
        w_train, _ = train_test_split(
            weights, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    else:
        w_train = None

    pipeline = _build_hgb_pipeline(spec)
    fit_kwargs: dict[str, Any] = {}
    if w_train is not None:
        fit_kwargs["model__sample_weight"] = w_train
    pipeline.fit(X_train, y_train, **fit_kwargs)

    y_pred_test = pipeline.predict(X_test)
    holdout = _metrics(y_test, y_pred_test)
    cv = _cv_metrics(_build_hgb_pipeline(spec), X_all, y_trans)

    baseline_pipeline = _build_ridge_pipeline(spec)
    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_metrics = _metrics(y_test, baseline_pred)

    perm = _permutation_importance(pipeline, X_test, y_test, spec.feature_columns)

    rev_metrics: dict[str, float] | None = None
    pred_original = y_pred_test
    if target_transform == "log1p":
        pred_original = np.expm1(y_pred_test)
        mae_musd = float(mean_absolute_error(y_raw_test, pred_original))
        rmse_musd = float(np.sqrt(mean_squared_error(y_raw_test, pred_original)))
        mape_mask = y_raw_test >= 1.0
        if mape_mask.any():
            mape = float(
                np.mean(
                    np.abs(
                        (y_raw_test[mape_mask] - pred_original[mape_mask])
                        / y_raw_test[mape_mask]
                    )
                )
            )
        else:
            mape = float("nan")
        rev_metrics = {
            "mae_musd": mae_musd,
            "rmse_musd": rmse_musd,
            "mape_over_1M": mape,
        }

    holdout_predictions = pd.DataFrame(
        {
            "y_true": y_raw_test,
            "y_pred": pred_original,
        }
    )

    return ModelBundle(
        target=target_col,
        target_label=target_label,
        target_transform=target_transform,
        pipeline=pipeline,
        feature_spec=spec,
        feature_columns=list(spec.feature_columns),
        holdout_metrics=holdout,
        cv_metrics=cv,
        baseline_holdout_metrics=baseline_metrics,
        revenue_space_metrics=rev_metrics,
        permutation_importance=perm,
        holdout_predictions=holdout_predictions,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        n_rows_total=int(prepared.height),
    )


def train(
    df: pl.DataFrame | None = None,
    settings: Settings | None = None,
) -> dict[str, ModelBundle]:
    """Train both models, save bundles, metrics.json, and the model card markdown."""
    settings = settings or get_settings()
    system_info = collect_system_info()
    logger.info(
        "Training environment (CPU/GPU/device):\n%s",
        format_system_info_for_log(system_info),
    )
    run_started_at = utc_now_iso()

    reproducibility = {
        "random_state": RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "test_size": TEST_SIZE,
    }

    stage_labels = (
        "Load Gold",
        "Train revenue (HGB)",
        "Train rating (HGB)",
        "Save joblib bundles",
        "Write metrics.json",
        "Write model_card.md",
    )
    revenue: ModelBundle | None = None
    rating: ModelBundle | None = None
    metrics_summary: dict[str, Any] = {}
    gold_source = ""

    with logging_redirect_tqdm():
        with tqdm(
            total=len(stage_labels),
            desc="Training",
            unit="stage",
        ) as pbar:
            # --- stage 1: Gold ---
            if df is None:
                gold_file = settings.gold_dir / "gold_movies.parquet"
                if not gold_file.is_file():
                    raise FileNotFoundError(
                        f"Gold parquet not found at {gold_file}. "
                        "Run `uv run python -m src.ingest all` first."
                    )
                df_in = pl.read_parquet(gold_file)
                gold_source = str(gold_file.resolve())
            else:
                df_in = df
                gold_source = "in-memory DataFrame"

            if df_in.height == 0:
                raise RuntimeError(
                    "Gold table is empty. Run `uv run python -m src.ingest all` first."
                )
            pbar.set_postfix_str("gold ready")
            pbar.update(1)

            # --- stage 2–3: fit models ---
            logger.info("training revenue model on %d rows", df_in.height)
            revenue = _train_single(
                df_in,
                target_col="revenue_musd",
                target_label="Revenue (million USD)",
                target_transform="log1p",
            )
            pbar.set_postfix_str("revenue done")
            pbar.update(1)

            logger.info("training rating model on %d rows", df_in.height)
            rating = _train_single(
                df_in,
                target_col="vote_average",
                target_label="User rating (0-10)",
                target_transform="identity",
                sample_weight_col="vote_count",
            )
            pbar.set_postfix_str("rating done")
            pbar.update(1)

            assert revenue is not None and rating is not None

            models_root = settings.ml_dir
            models_root.mkdir(parents=True, exist_ok=True)
            revenue_bundle_path(settings).parent.mkdir(parents=True, exist_ok=True)
            rating_bundle_path(settings).parent.mkdir(parents=True, exist_ok=True)
            _dump_bundle(revenue, revenue_bundle_path(settings))
            _dump_bundle(rating, rating_bundle_path(settings))
            pbar.set_postfix_str("bundles saved")
            pbar.update(1)

            metrics_summary = {
                "revenue": revenue.metadata(),
                "rating": rating.metadata(),
            }
            metrics_json_path(settings).write_text(
                json.dumps(metrics_summary, indent=2, default=float)
            )
            pbar.set_postfix_str("metrics.json")
            pbar.update(1)

            markdown = render_model_card_markdown(
                metrics_summary=metrics_summary,
                system_info=system_info,
                settings=settings,
                gold_source=gold_source,
                run_started_at_utc=run_started_at,
                reproducibility=reproducibility,
            )
            write_model_card(settings.model_card_path, markdown)
            pbar.set_postfix_str("model_card.md")
            pbar.update(1)

    logger.info(
        "saved models under %s (revenue holdout R2=%.3f, rating holdout R2=%.3f); "
        "model card -> %s",
        settings.ml_dir,
        revenue.holdout_metrics.r2,
        rating.holdout_metrics.r2,
        settings.model_card_path,
    )
    return {"revenue": revenue, "rating": rating}


def _dump_bundle(bundle: ModelBundle, path: Path) -> None:
    payload = {
        "pipeline": bundle.pipeline,
        "feature_spec": asdict(bundle.feature_spec),
        "feature_columns": bundle.feature_columns,
        "target": bundle.target,
        "target_label": bundle.target_label,
        "target_transform": bundle.target_transform,
        "holdout_metrics": bundle.holdout_metrics.as_dict(),
        "cv_metrics": bundle.cv_metrics.as_dict(),
        "baseline_holdout_metrics": bundle.baseline_holdout_metrics.as_dict(),
        "revenue_space_metrics": bundle.revenue_space_metrics,
        "permutation_importance": bundle.permutation_importance,
        "holdout_predictions": bundle.holdout_predictions,
        "n_train": bundle.n_train,
        "n_test": bundle.n_test,
        "n_rows_total": bundle.n_rows_total,
    }
    joblib.dump(payload, path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train()


if __name__ == "__main__":
    main()
