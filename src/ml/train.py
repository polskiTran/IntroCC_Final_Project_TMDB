"""Train the TMDB revenue and rating models from the Gold parquet.

Produces two self-contained bundles under ``settings.ml_dir``:

- ``model_revenue.joblib``: predicts ``revenue_musd`` (trained on ``log1p``).
- ``model_rating.joblib``: predicts ``vote_average`` in [0, 10].

Each bundle carries the fitted ``CatBoostRegressor`` (wrapped in a sklearn
``Pipeline`` for API parity), the ``FeatureSpec`` (top genres), the target
transform metadata, hold-out and 5-fold CV metrics, a Ridge + ``TargetEncoder``
baseline, permutation importances, and a small held-out frame used to render
the predicted-vs-actual scatter in the Streamlit page without retraining.

Director / lead production company / lead cast are fed to CatBoost as native
``cat_features`` (no target encoding) so rare categories keep their identity
instead of being smoothed toward the global prior.

The training run is idempotent: it fully rebuilds both bundles and
``metrics.json`` on every invocation. A GPU is used automatically when
available (probed once per run), otherwise it falls back to CPU. Override with
``TMDB_ML_DEVICE=CPU`` or ``TMDB_ML_DEVICE=GPU``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import catboost
import joblib
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from tqdm.auto import tqdm

from src.config import Settings, get_settings
from src.ml.features import (
    CATEGORICAL_COLS,
    CYCLICAL_COLS,
    NUMERIC_COLS,
    TARGET_ENCODED_COLS,
    FeatureSpec,
    build_feature_frame,
    categorical_feature_indices,
    fit_feature_spec,
)
from src.ml.model_card import write_model_card
from src.ml import tabpfn_model

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
CV_SPLITS = 5
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.15
CATBOOST_ITERATIONS = 2000
CATBOOST_CV_ITERATIONS = 600
CATBOOST_EARLY_STOPPING = 50

_TASK_TYPE_CACHE: str | None = None


def _tqdm_disable(show_progress: bool | None) -> bool | None:
    """Map ``show_progress`` to tqdm's ``disable`` (``None`` = auto / non-TTY)."""
    if show_progress is True:
        return False
    if show_progress is False:
        return True
    return None


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
    pipeline: Any
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


def _resolve_task_type() -> str:
    """Pick ``"GPU"`` when CatBoost can train on the host, else ``"CPU"``.

    Honors the ``TMDB_ML_DEVICE`` env var as a hard override. Probe result is
    cached in a module global so we only pay the ~1-2s detection cost once per
    process.
    """
    global _TASK_TYPE_CACHE
    if _TASK_TYPE_CACHE is not None:
        return _TASK_TYPE_CACHE

    override = os.environ.get("TMDB_ML_DEVICE", "").strip().upper()
    if override in {"CPU", "GPU"}:
        logger.info("CatBoost task_type forced to %s via TMDB_ML_DEVICE", override)
        _TASK_TYPE_CACHE = override
        return override

    try:
        probe = CatBoostRegressor(
            iterations=2,
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
        )
        probe.fit(
            np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0.0, 1.0, 2.0, 3.0])
        )
    except Exception as exc:  # noqa: BLE001 - CatBoost raises many types on missing CUDA
        logger.info("CatBoost GPU unavailable (%s); falling back to CPU", exc)
        _TASK_TYPE_CACHE = "CPU"
    else:
        logger.info("CatBoost GPU detected; using task_type=GPU")
        _TASK_TYPE_CACHE = "GPU"
    return _TASK_TYPE_CACHE


def _catboost_kwargs(task_type: str, *, iterations: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "iterations": iterations,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": task_type,
    }
    if task_type == "GPU":
        kwargs["devices"] = "0"
    else:
        kwargs["thread_count"] = -1
    return kwargs


def _build_catboost_model(
    spec: FeatureSpec, *, task_type: str, iterations: int = CATBOOST_ITERATIONS
) -> Pipeline:
    cat_idx = categorical_feature_indices(spec)
    model = CatBoostRegressor(
        cat_features=cat_idx, **_catboost_kwargs(task_type, iterations=iterations)
    )
    return Pipeline([("model", model)])


def _build_ridge_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
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


def _build_ridge_pipeline(spec: FeatureSpec) -> Pipeline:
    return Pipeline(
        [
            ("preproc", _build_ridge_preprocessor(spec)),
            ("scale", StandardScaler(with_mean=True)),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )


def _cv_metrics(
    spec: FeatureSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: str,
    *,
    tqdm_disable: bool | None = None,
    cv_desc: str = "CV",
) -> Metrics:
    """5-fold CV with CatBoost. Fixed iterations (no early stopping per fold)
    to keep the per-fold budget identical and comparable.

    Uses a manual KFold loop instead of ``cross_val_score`` because sklearn's
    ``clone`` can't round-trip ``CatBoostRegressor``'s ``cat_features``
    parameter (CatBoost mutates it at construction).
    """
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    r2_scores: list[float] = []
    mae_scores: list[float] = []
    rmse_scores: list[float] = []
    for tr_idx, va_idx in tqdm(
        cv.split(X),
        total=CV_SPLITS,
        desc=cv_desc,
        leave=False,
        disable=tqdm_disable,
        unit="fold",
    ):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        fold_pipeline = _build_catboost_model(
            spec, task_type=task_type, iterations=CATBOOST_CV_ITERATIONS
        )
        fold_pipeline.fit(X_tr, y_tr, model__verbose=False)
        pred = fold_pipeline.predict(X_va)
        r2_scores.append(float(r2_score(y_va, pred)))
        mae_scores.append(float(mean_absolute_error(y_va, pred)))
        rmse_scores.append(float(np.sqrt(mean_squared_error(y_va, pred))))
    return Metrics(
        r2=float(np.mean(r2_scores)),
        mae=float(np.mean(mae_scores)),
        rmse=float(np.mean(rmse_scores)),
    )


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


class BlendedPredictor:
    """Picklable wrapper that averages CatBoost and TabPFN predictions.

    Stores the fitted CatBoost pipeline, the fitted ``TabPFNComponent`` (may
    be ``None`` if TabPFN was unavailable at training time), and the blend
    weights fit on out-of-fold predictions. Exposes ``.predict(X)`` so the
    downstream loader (``src.ml.predict``) needs no changes.
    """

    def __init__(
        self,
        cb_pipeline: Pipeline,
        tabpfn: Any | None,
        weights: dict[str, float],
    ) -> None:
        self.cb_pipeline = cb_pipeline
        self.tabpfn = tabpfn
        self.weights = weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cb = np.asarray(self.cb_pipeline.predict(X), dtype=np.float64)
        if self.tabpfn is None:
            return cb
        tp = np.asarray(self.tabpfn.predict(X), dtype=np.float64)
        return self.weights["catboost"] * cb + self.weights["tabpfn"] * tp


def _oof_catboost(
    spec: FeatureSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: str,
    weights: np.ndarray | None = None,
    *,
    tqdm_disable: bool | None = None,
    oof_desc: str = "OOF CatBoost",
) -> np.ndarray:
    """5-fold out-of-fold predictions for CatBoost."""
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr_idx, va_idx in tqdm(
        cv.split(X),
        total=CV_SPLITS,
        desc=oof_desc,
        leave=False,
        disable=tqdm_disable,
        unit="fold",
    ):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]
        w_tr = weights[tr_idx] if weights is not None else None
        fold_pipeline = _build_catboost_model(
            spec, task_type=task_type, iterations=CATBOOST_CV_ITERATIONS
        )
        fit_kwargs: dict[str, Any] = {"model__verbose": False}
        if w_tr is not None:
            fit_kwargs["model__sample_weight"] = w_tr
        fold_pipeline.fit(X_tr, y_tr, **fit_kwargs)
        oof[va_idx] = fold_pipeline.predict(X_va)
    return oof


def _oof_tabpfn(
    spec: FeatureSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    tqdm_disable: bool | None = None,
    oof_desc: str = "OOF TabPFN",
) -> np.ndarray:
    """5-fold out-of-fold predictions for TabPFN."""
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr_idx, va_idx in tqdm(
        cv.split(X),
        total=CV_SPLITS,
        desc=oof_desc,
        leave=False,
        disable=tqdm_disable,
        unit="fold",
    ):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        component = tabpfn_model.TabPFNComponent(spec, random_state=RANDOM_STATE)
        component.fit(X_tr, y[tr_idx])
        oof[va_idx] = component.predict(X_va)
        del component
        tabpfn_model._empty_cache()
    return oof


def _fit_blend_weights(
    oof_cb: np.ndarray, oof_tp: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Fit non-negative blend weights via Ridge; normalise so they sum to 1."""
    from sklearn.linear_model import Ridge as _Ridge  # noqa: PLC0415

    stack = np.column_stack([oof_cb, oof_tp])
    ridge = _Ridge(alpha=1.0, positive=True, fit_intercept=False)
    ridge.fit(stack, y)
    coefs = np.clip(ridge.coef_, 0.0, None)
    total = float(coefs.sum())
    if total <= 0:
        return {"catboost": 1.0, "tabpfn": 0.0}
    w_cb, w_tp = coefs / total
    return {"catboost": float(w_cb), "tabpfn": float(w_tp)}


def _train_single(
    df: pl.DataFrame,
    *,
    target_col: str,
    target_label: str,
    target_transform: str,
    task_type: str,
    sample_weight_col: str | None = None,
    enable_blend: bool = False,
    tqdm_disable: bool | None = None,
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

    # Carve a validation slice off training for CatBoost early stopping.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
    )
    if w_train is not None:
        w_tr, _w_val = train_test_split(
            w_train, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE
        )
    else:
        w_tr = None

    pipeline = _build_catboost_model(spec, task_type=task_type)
    fit_kwargs: dict[str, Any] = {
        "model__eval_set": (X_val, y_val),
        "model__early_stopping_rounds": CATBOOST_EARLY_STOPPING,
        "model__use_best_model": True,
        "model__verbose": False,
    }
    if w_tr is not None:
        fit_kwargs["model__sample_weight"] = w_tr
    pipeline.fit(X_tr, y_tr, **fit_kwargs)

    y_pred_cb = pipeline.predict(X_test)
    cb_holdout = _metrics(y_test, y_pred_cb)
    cv = _cv_metrics(
        spec,
        X_all,
        y_trans,
        task_type=task_type,
        tqdm_disable=tqdm_disable,
        cv_desc=f"CV {target_col}",
    )

    baseline_pipeline = _build_ridge_pipeline(spec)
    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_metrics = _metrics(y_test, baseline_pred)

    blend_weights: dict[str, float] | None = None
    tp_holdout: Metrics | None = None
    final_predictor: Any = pipeline
    y_pred_test = y_pred_cb
    if enable_blend and tabpfn_model.available():
        logger.info(
            "TabPFN available (%s); fitting blend component for %s",
            tabpfn_model.version(),
            target_col,
        )
        try:
            tp_component = tabpfn_model.TabPFNComponent(spec, random_state=RANDOM_STATE)
            tp_component.fit(X_train, y_train)
            y_pred_tp = tp_component.predict(X_test)
            tp_holdout = _metrics(y_test, y_pred_tp)

            oof_cb = _oof_catboost(
                spec,
                X_train,
                y_train,
                task_type=task_type,
                weights=w_train,
                tqdm_disable=tqdm_disable,
                oof_desc=f"OOF CatBoost {target_col}",
            )
            oof_tp = _oof_tabpfn(
                spec,
                X_train,
                y_train,
                tqdm_disable=tqdm_disable,
                oof_desc=f"OOF TabPFN {target_col}",
            )
            blend_weights = _fit_blend_weights(oof_cb, oof_tp, y_train)
            logger.info(
                "blend weights for %s: catboost=%.3f tabpfn=%.3f",
                target_col,
                blend_weights["catboost"],
                blend_weights["tabpfn"],
            )
            y_pred_test = (
                blend_weights["catboost"] * y_pred_cb
                + blend_weights["tabpfn"] * y_pred_tp
            )
            final_predictor = BlendedPredictor(
                cb_pipeline=pipeline,
                tabpfn=tp_component,
                weights=blend_weights,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TabPFN blend failed for %s (%s); falling back to CatBoost-only",
                target_col,
                exc,
            )
            blend_weights = None
            tp_holdout = None
            final_predictor = pipeline
            y_pred_test = y_pred_cb

    holdout = _metrics(y_test, y_pred_test)
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

    best_iteration = int(pipeline.named_steps["model"].get_best_iteration() or 0)
    model_name = "catboost+tabpfn" if blend_weights is not None else "catboost"
    extra: dict[str, Any] = {
        "model": model_name,
        "task_type": task_type,
        "catboost_version": catboost.__version__,
        "best_iteration": best_iteration,
        "cat_features": list(CATEGORICAL_COLS),
        "catboost_holdout_metrics": cb_holdout.as_dict(),
    }
    if blend_weights is not None:
        extra["blend_weights"] = blend_weights
        extra["tabpfn_version"] = tabpfn_model.version()
        if tp_holdout is not None:
            extra["tabpfn_holdout_metrics"] = tp_holdout.as_dict()

    return ModelBundle(
        target=target_col,
        target_label=target_label,
        target_transform=target_transform,
        pipeline=final_predictor,
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
        extra=extra,
    )


def train(
    df: pl.DataFrame | None = None,
    settings: Settings | None = None,
    *,
    show_progress: bool | None = None,
) -> dict[str, ModelBundle]:
    """Train both models, save bundles and metrics.json. Returns the bundles.

    ``show_progress``: when ``True``, always show tqdm bars; when ``False``,
    never show; when ``None`` (default), tqdm hides bars on non-TTY stderr.
    """
    settings = settings or get_settings()
    if df is None:
        gold_file = settings.gold_dir / "gold_movies.parquet"
        if not gold_file.is_file():
            raise FileNotFoundError(
                f"Gold parquet not found at {gold_file}. "
                "Run `uv run python -m src.ingest all` first."
            )
        df = pl.read_parquet(gold_file)
    if df.height == 0:
        raise RuntimeError(
            "Gold table is empty. Run `uv run python -m src.ingest all` first."
        )

    task_type = _resolve_task_type()
    tqdm_disable = _tqdm_disable(show_progress)
    t_train_start = time.perf_counter()

    logger.info(
        "training revenue model on %d rows (task_type=%s)", df.height, task_type
    )
    with tqdm(
        total=2,
        desc="ML train",
        unit="model",
        disable=tqdm_disable,
    ) as bar:
        bar.set_postfix_str("revenue_musd")
        revenue = _train_single(
            df,
            target_col="revenue_musd",
            target_label="Revenue (million USD)",
            target_transform="log1p",
            task_type=task_type,
            tqdm_disable=tqdm_disable,
        )
        bar.update(1)
        logger.info(
            "training rating model on %d rows (task_type=%s)", df.height, task_type
        )
        bar.set_postfix_str("vote_average")
        rating = _train_single(
            df,
            target_col="vote_average",
            target_label="User rating (0-10)",
            target_transform="identity",
            task_type=task_type,
            sample_weight_col="vote_count",
            enable_blend=True,
            tqdm_disable=tqdm_disable,
        )
        bar.update(1)

    train_elapsed_s = time.perf_counter() - t_train_start
    logger.info("ML training wall time: %.1fs", train_elapsed_s)

    ml_dir = settings.ml_dir
    ml_dir.mkdir(parents=True, exist_ok=True)
    _dump_bundle(revenue, ml_dir / "model_revenue.joblib")
    _dump_bundle(rating, ml_dir / "model_rating.joblib")

    metrics_summary = {
        "revenue": revenue.metadata(),
        "rating": rating.metadata(),
    }
    (ml_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2, default=float)
    )

    logger.info(
        "saved models to %s (revenue holdout R2=%.3f, rating holdout R2=%.3f)",
        ml_dir,
        revenue.holdout_metrics.r2,
        rating.holdout_metrics.r2,
    )

    try:
        write_model_card({"revenue": revenue, "rating": rating}, settings=settings)
    except Exception as exc:  # noqa: BLE001 - model card is a side-deliverable
        logger.warning("failed to write model card: %s", exc)

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
        "extra": bundle.extra,
    }
    joblib.dump(payload, path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train()


if __name__ == "__main__":
    main()
