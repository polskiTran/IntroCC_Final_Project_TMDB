"""Smoke tests for the ML module.

Synthesize a tiny Gold-like frame, run the training entry point, and verify
the resulting bundles load and produce sensible predictions. No network, no
real TMDB data.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

os.environ.setdefault("TMDB_ML_DEVICE", "CPU")
os.environ.setdefault("TMDB_ML_TABPFN", "0")
os.environ.setdefault("TMDB_ML_EMBEDDINGS", "0")

from src.config import Settings  # noqa: E402
from src.ml import predict as predict_mod  # noqa: E402
from src.ml import tabpfn_model  # noqa: E402
from src.ml import train as train_mod  # noqa: E402
from src.ml.features import (  # noqa: E402
    OVERVIEW_EMBED_DIM,
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
    producers = [f"Producer {i}" for i in range(5)]
    collections = [None, "Franchise A", "Franchise B", "Franchise C"]

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
        "n_production_companies": [],
        "n_genres": [],
        "director_id": [],
        "director_name": [],
        "lead_cast_id": [],
        "lead_cast_name": [],
        "cast_2_name": [],
        "cast_3_name": [],
        "cast_4_name": [],
        "cast_5_name": [],
        "n_cast": [],
        "lead_producer_name": [],
        "n_producers": [],
        "collection_name": [],
        "has_tagline": [],
        "overview": [],
        "overview_embedding": [],
    }
    for i in range(n):
        budget_musd = float(rng.uniform(1.0, 200.0))
        month = int(rng.integers(1, 13))
        director = directors[int(rng.integers(0, len(directors)))]
        studio = studios[int(rng.integers(0, len(studios)))]
        lead = leads[int(rng.integers(0, len(leads)))]
        producer = producers[int(rng.integers(0, len(producers)))]
        collection = collections[int(rng.integers(0, len(collections)))]
        genres = genres_pool[int(rng.integers(0, len(genres_pool)))]
        revenue_musd = max(
            0.5,
            budget_musd * (0.8 + 0.6 * ("Action" in genres)) + rng.normal(0.0, 15.0),
        )
        rating = float(np.clip(6.0 + 0.02 * budget_musd + rng.normal(0, 0.5), 1.0, 9.5))
        emb = rng.normal(0.0, 1.0, OVERVIEW_EMBED_DIM).astype(np.float32).tolist()
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
        rows["n_production_companies"].append(1)
        rows["n_genres"].append(len(genres))
        rows["director_id"].append(i)
        rows["director_name"].append(director)
        rows["lead_cast_id"].append(i + 100)
        rows["lead_cast_name"].append(lead)
        rows["cast_2_name"].append(leads[(i + 1) % len(leads)])
        rows["cast_3_name"].append(leads[(i + 2) % len(leads)])
        rows["cast_4_name"].append(leads[(i + 3) % len(leads)])
        rows["cast_5_name"].append(leads[(i + 4) % len(leads)])
        rows["n_cast"].append(5)
        rows["lead_producer_name"].append(producer)
        rows["n_producers"].append(int(rng.integers(1, 4)))
        rows["collection_name"].append(collection or "Standalone")
        rows["has_tagline"].append(int(rng.integers(0, 2)))
        rows["overview"].append(f"Synthetic overview {i}")
        rows["overview_embedding"].append(emb)

    return pl.DataFrame(
        rows,
        schema_overrides={
            "overview_embedding": pl.List(pl.Float32),
            "has_tagline": pl.Int8,
        },
    )


def test_feature_spec_and_frame_shape() -> None:
    df = _synth_gold(30)
    spec = fit_feature_spec(df, top_k_genres=5, top_k_studios=4)
    assert 0 < len(spec.top_genres) <= 5
    assert 0 < len(spec.top_studios) <= 4
    feats = build_feature_frame(df, spec)
    assert feats.shape[0] == df.height
    for col in ("budget_musd", "runtime", "release_year", "month_sin", "month_cos"):
        assert col in feats.columns
    for col in TARGET_ENCODED_COLS:
        assert col in feats.columns
    for col in spec.multihot_cols:
        assert col in feats.columns
        assert set(feats[col].unique()).issubset({0, 1})
    for i in range(OVERVIEW_EMBED_DIM):
        assert f"overview_emb_{i}" in feats.columns


def test_build_single_row_round_trip() -> None:
    df = _synth_gold(20)
    spec = fit_feature_spec(df, top_k_genres=4, top_k_studios=4)
    row = build_single_row(
        spec=spec,
        budget_musd=50.0,
        runtime=120.0,
        release_year=2024,
        release_month=7,
        genres=["Action", "Made-up"],
        director_name="Someone",
        lead_production_company="Studio X",
        lead_cast_name="Actor Y",
    )
    assert row.shape == (1, len(spec.feature_columns))
    assert row["budget_musd"].iloc[0] == 50.0
    assert math.isclose(row["month_sin"].iloc[0], math.sin(2 * math.pi * 6 / 12))
    assert row["collection_name"].iloc[0] == "Standalone"
    assert row["overview_emb_0"].iloc[0] == 0.0


def test_train_and_predict_end_to_end(tmp_path: Path) -> None:
    settings = Settings(
        ml_dir=tmp_path / "ml",
        gold_dir=tmp_path / "gold",
        silver_dir=tmp_path / "silver",
        bronze_dir=tmp_path / "bronze",
    )
    df = _synth_gold(80)
    bundles = train_mod.train(df=df, settings=settings)
    assert set(bundles) == {"revenue", "rating"}

    for target in ("revenue", "rating"):
        path = settings.ml_dir / f"model_{target}.joblib"
        assert path.is_file(), f"missing bundle for {target}"

    metrics_file = settings.ml_dir / "metrics.json"
    assert metrics_file.is_file()

    loaded_rev = predict_mod.load_bundle("revenue", settings=settings)
    loaded_rate = predict_mod.load_bundle("rating", settings=settings)

    assert loaded_rev.metadata.get("model") == "catboost"
    assert loaded_rate.metadata.get("model") in {"catboost", "catboost+tabpfn"}
    assert loaded_rev.metadata.get("task_type") in {"CPU", "GPU"}

    rev = predict_mod.predict_one(
        loaded_rev,
        budget_musd=80.0,
        runtime=115.0,
        release_year=2024,
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
        release_year=2024,
        release_month=6,
        genres=["Action", "Adventure"],
        director_name="Director 1",
        lead_production_company="Studio 2",
        lead_cast_name="Lead 3",
    )
    assert math.isfinite(rev) and rev >= 0.0
    assert math.isfinite(rate) and 0.0 <= rate <= 10.0


def test_rating_bundle_accepts_tabpfn_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When TabPFN is disabled, the rating training must still succeed."""
    monkeypatch.setattr(tabpfn_model, "available", lambda: False)
    settings = Settings(
        ml_dir=tmp_path / "ml",
        gold_dir=tmp_path / "gold",
        silver_dir=tmp_path / "silver",
        bronze_dir=tmp_path / "bronze",
    )
    df = _synth_gold(80)
    bundles = train_mod.train(df=df, settings=settings)
    rating_meta = predict_mod.load_bundle("rating", settings=settings).metadata
    assert rating_meta.get("model") == "catboost"
    assert "blend_weights" not in rating_meta
    assert bundles["rating"].pipeline is not None


def test_train_refuses_empty() -> None:
    empty = pl.DataFrame(schema={"revenue_musd": pl.Float64})
    with pytest.raises(RuntimeError):
        train_mod.train(df=empty)
