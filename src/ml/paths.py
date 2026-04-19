"""Canonical filesystem paths for trained ML artifacts under ``settings.ml_dir``."""

from __future__ import annotations

from pathlib import Path

from src.config import Settings

REVENUE_MODEL_SUBDIR = "revenue_model"
RATING_MODEL_SUBDIR = "rating_model"

REVENUE_BUNDLE_FILENAME = "model_revenue.joblib"
RATING_BUNDLE_FILENAME = "model_rating.joblib"

METRICS_FILENAME = "metrics.json"


def revenue_bundle_path(settings: Settings) -> Path:
    return settings.ml_dir / REVENUE_MODEL_SUBDIR / REVENUE_BUNDLE_FILENAME


def rating_bundle_path(settings: Settings) -> Path:
    return settings.ml_dir / RATING_MODEL_SUBDIR / RATING_BUNDLE_FILENAME


def bundle_path(target: str, settings: Settings) -> Path:
    """Return the joblib path for ``target`` (``revenue`` or ``rating``)."""
    if target == "revenue":
        return revenue_bundle_path(settings)
    if target == "rating":
        return rating_bundle_path(settings)
    raise ValueError(f"Unknown target {target!r}; expected 'revenue' or 'rating'")


def metrics_json_path(settings: Settings) -> Path:
    """Combined metrics for both models (same schema as before the layout move)."""
    return settings.ml_dir / METRICS_FILENAME
