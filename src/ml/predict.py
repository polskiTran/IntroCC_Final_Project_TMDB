"""Inference helpers for the TMDB models.

Loads the joblib bundles produced by :mod:`src.ml.train` and exposes a small
functional API used by both the CLI and the Streamlit page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.ml.features import FeatureSpec, build_single_row


_TARGET_FILES = {
    "revenue": "model_revenue.joblib",
    "rating": "model_rating.joblib",
}


@dataclass
class LoadedBundle:
    target: str
    target_label: str
    target_transform: str
    pipeline: Any
    feature_spec: FeatureSpec
    feature_columns: list[str]
    metadata: dict[str, Any]
    holdout_predictions: pd.DataFrame


def bundle_path(target: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    if target not in _TARGET_FILES:
        raise ValueError(
            f"Unknown target {target!r}; expected one of {list(_TARGET_FILES)}"
        )
    return settings.ml_dir / _TARGET_FILES[target]


def bundle_exists(target: str, settings: Settings | None = None) -> bool:
    return bundle_path(target, settings).is_file()


def load_bundle(target: str, settings: Settings | None = None) -> LoadedBundle:
    path = bundle_path(target, settings)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run `uv run python -m src.ml train`."
        )
    payload = joblib.load(path)
    spec = FeatureSpec(**payload["feature_spec"])
    metadata = {
        k: payload[k]
        for k in (
            "holdout_metrics",
            "cv_metrics",
            "baseline_holdout_metrics",
            "revenue_space_metrics",
            "permutation_importance",
            "n_train",
            "n_test",
            "n_rows_total",
        )
    }
    metadata["top_genres"] = spec.top_genres
    extra = payload.get("extra", {}) or {}
    metadata["extra"] = extra
    for key in (
        "model",
        "task_type",
        "catboost_version",
        "best_iteration",
        "blend_weights",
        "tabpfn_version",
        "catboost_holdout_metrics",
        "tabpfn_holdout_metrics",
    ):
        if key in extra:
            metadata[key] = extra[key]
    return LoadedBundle(
        target=payload["target"],
        target_label=payload["target_label"],
        target_transform=payload["target_transform"],
        pipeline=payload["pipeline"],
        feature_spec=spec,
        feature_columns=payload["feature_columns"],
        metadata=metadata,
        holdout_predictions=payload["holdout_predictions"],
    )


def _post_process(target: str, transform: str, raw_pred: float) -> float:
    if transform == "log1p":
        value = float(np.expm1(raw_pred))
        return max(value, 0.0)
    if target == "vote_average":
        return float(np.clip(raw_pred, 0.0, 10.0))
    return float(raw_pred)


def predict_one(
    bundle: LoadedBundle,
    *,
    budget_musd: float,
    runtime: float,
    release_year: int,
    release_month: int,
    genres: list[str],
    director_name: str,
    lead_production_company: str,
    lead_cast_name: str,
    cast_2_name: str = "Unknown",
    cast_3_name: str = "Unknown",
    cast_4_name: str = "Unknown",
    cast_5_name: str = "Unknown",
    lead_producer_name: str = "Unknown",
    collection_name: str = "Standalone",
    has_tagline: bool = False,
    overview_embedding: np.ndarray | list[float] | None = None,
    production_companies: list[str] | None = None,
) -> float:
    X = build_single_row(
        spec=bundle.feature_spec,
        budget_musd=budget_musd,
        runtime=runtime,
        release_year=release_year,
        release_month=release_month,
        genres=genres,
        production_companies=production_companies,
        director_name=director_name,
        lead_production_company=lead_production_company,
        lead_cast_name=lead_cast_name,
        cast_2_name=cast_2_name,
        cast_3_name=cast_3_name,
        cast_4_name=cast_4_name,
        cast_5_name=cast_5_name,
        lead_producer_name=lead_producer_name,
        collection_name=collection_name,
        has_tagline=has_tagline,
        overview_embedding=overview_embedding,
    )
    raw = float(bundle.pipeline.predict(X)[0])
    return _post_process(bundle.target, bundle.target_transform, raw)
