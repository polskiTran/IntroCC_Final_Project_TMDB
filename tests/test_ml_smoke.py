"""Smoke tests for the ML module.

Synthesize a tiny Gold-like frame, run the training entry point, and verify
the resulting bundles load and produce sensible predictions. No network, no
real TMDB data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config import Settings
from src.ml import predict as predict_mod
from src.ml import train as train_mod
from src.ml.paths import metrics_json_path, rating_bundle_path, revenue_bundle_path
from src.ml.features import (
    TARGET_ENCODED_COLS,
    build_feature_frame,
    build_single_row,
    fit_feature_spec,
)


def _synth_gold(n: int = 60, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    genres_pool = [
        ["Action"],
        ["Drama"],
        ["Action", "Adventure"],
        ["Comedy"],
        ["Drama", "Romance"],
        ["Action", "Thriller"],
        ["Science Fiction", "Adventure"],
        ["Horror"],
    ]
    directors = [f"Director {i}" for i in range(8)]
    studios = [f"Studio {i}" for i in range(6)]
    leads = [f"Lead {i}" for i in range(12)]

    rows: dict[str, list] = {
        "movie_id": [],
        "title": [],
        "original_title": [],
        "release_year": [],
        "release_month": [],
        "budget": [],
        "budget_musd": [],
        "revenue": [],
        "revenue_musd": [],
        "roi": [],
        "runtime": [],
        "vote_average": [],
        "vote_count": [],
        "popularity": [],
        "genres": [],
        "production_companies": [],
        "lead_production_company": [],
        "director_id": [],
        "director_name": [],
        "lead_cast_id": [],
        "lead_cast_name": [],
    }
    for i in range(n):
        budget_musd = float(rng.uniform(1.0, 200.0))
        month = int(rng.integers(1, 13))
        director = directors[int(rng.integers(0, len(directors)))]
        studio = studios[int(rng.integers(0, len(studios)))]
        lead = leads[int(rng.integers(0, len(leads)))]
        genres = genres_pool[int(rng.integers(0, len(genres_pool)))]
        revenue_musd = max(
            0.5,
            budget_musd * (0.8 + 0.6 * ("Action" in genres)) + rng.normal(0.0, 15.0),
        )
        rating = float(np.clip(6.0 + 0.02 * budget_musd + rng.normal(0, 0.5), 1.0, 9.5))
        rows["movie_id"].append(1000 + i)
        rows["title"].append(f"Movie {i}")
        rows["original_title"].append(f"Movie {i}")
        rows["release_year"].append(2000 + (i % 20))
        rows["release_month"].append(month)
        rows["budget"].append(int(budget_musd * 1_000_000))
        rows["budget_musd"].append(budget_musd)
        rows["revenue"].append(int(revenue_musd * 1_000_000))
        rows["revenue_musd"].append(revenue_musd)
        rows["roi"].append((revenue_musd - budget_musd) / budget_musd)
        rows["runtime"].append(int(rng.integers(80, 160)))
        rows["vote_average"].append(rating)
        rows["vote_count"].append(int(rng.integers(50, 5000)))
        rows["popularity"].append(float(rng.uniform(1.0, 100.0)))
        rows["genres"].append(genres)
        rows["production_companies"].append([studio])
        rows["lead_production_company"].append(studio)
        rows["director_id"].append(i)
        rows["director_name"].append(director)
        rows["lead_cast_id"].append(i + 100)
        rows["lead_cast_name"].append(lead)

    return pl.DataFrame(rows)


def test_feature_spec_and_frame_shape() -> None:
    df = _synth_gold(30)
    spec = fit_feature_spec(df, top_k=5)
    assert 0 < len(spec.top_genres) <= 5
    feats = build_feature_frame(df, spec)
    assert feats.shape[0] == df.height
    for col in ("budget_musd", "runtime", "month_sin", "month_cos"):
        assert col in feats.columns
    for col in TARGET_ENCODED_COLS:
        assert col in feats.columns
    for col in spec.multihot_cols:
        assert col in feats.columns
        assert set(feats[col].unique()).issubset({0, 1})


def test_build_single_row_round_trip() -> None:
    df = _synth_gold(20)
    spec = fit_feature_spec(df, top_k=4)
    row = build_single_row(
        spec=spec,
        budget_musd=50.0,
        runtime=120.0,
        release_month=7,
        genres=["Action", "Made-up"],
        director_name="Someone",
        lead_production_company="Studio X",
        lead_cast_name="Actor Y",
    )
    assert row.shape == (1, len(spec.feature_columns))
    assert row["budget_musd"].iloc[0] == 50.0
    assert math.isclose(row["month_sin"].iloc[0], math.sin(2 * math.pi * 6 / 12))


def test_train_and_predict_end_to_end(tmp_path: Path) -> None:
    settings = Settings(
        ml_dir=tmp_path / "models",
        gold_dir=tmp_path / "gold",
        silver_dir=tmp_path / "silver",
        bronze_dir=tmp_path / "bronze",
        model_card_path=tmp_path / "models" / "model_card.md",
    )
    df = _synth_gold(80)
    bundles = train_mod.train(df=df, settings=settings)
    assert set(bundles) == {"revenue", "rating"}

    assert revenue_bundle_path(settings).is_file(), "missing revenue bundle"
    assert rating_bundle_path(settings).is_file(), "missing rating bundle"

    metrics_file = metrics_json_path(settings)
    assert metrics_file.is_file()

    card = settings.model_card_path.read_text(encoding="utf-8")
    assert settings.model_card_path.is_file()
    assert "# Model card" in card
    assert "## Evaluation" in card
    assert "revenue" in card and "rating" in card

    loaded_rev = predict_mod.load_bundle("revenue", settings=settings)
    loaded_rate = predict_mod.load_bundle("rating", settings=settings)

    rev = predict_mod.predict_one(
        loaded_rev,
        budget_musd=80.0,
        runtime=115.0,
        release_month=6,
        genres=["Action", "Adventure"],
        director_name="Director 1",
        lead_production_company="Studio 2",
        lead_cast_name="Lead 3",
    )
    rate = predict_mod.predict_one(
        loaded_rate,
        budget_musd=80.0,
        runtime=115.0,
        release_month=6,
        genres=["Action", "Adventure"],
        director_name="Director 1",
        lead_production_company="Studio 2",
        lead_cast_name="Lead 3",
    )
    assert math.isfinite(rev) and rev >= 0.0
    assert math.isfinite(rate) and 0.0 <= rate <= 10.0


def test_train_refuses_empty() -> None:
    empty = pl.DataFrame(schema={"revenue_musd": pl.Float64})
    with pytest.raises(RuntimeError):
        train_mod.train(df=empty)
