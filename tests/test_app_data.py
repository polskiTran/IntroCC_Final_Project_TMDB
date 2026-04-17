"""Smoke tests for src/app/_data.py helpers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.app._data import (
    classify_roi,
    gold_path,
    gold_path_exists,
    layer_metadata,
    load_gold,
    month_genre_matrix,
    roi_by_genre,
    scope_constraints,
    top_directors,
    top_production_companies,
)
from src.config import Settings
from src.ingest import gold, silver


def _write_bronze_movie(dir_path: Path, doc: dict[str, Any]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{doc['id']}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(doc, f)


def _movie_fixture(movie_id: int) -> dict[str, Any]:
    return {
        "id": movie_id,
        "title": f"Movie {movie_id}",
        "original_title": f"Movie {movie_id}",
        "release_date": "2005-06-15",
        "budget": 10_000_000,
        "revenue": 50_000_000,
        "runtime": 120,
        "vote_average": 7.5,
        "vote_count": 2000,
        "popularity": 42.0,
        "original_language": "en",
        "adult": False,
        "genres": [{"id": 28, "name": "Action"}],
        "production_companies": [{"id": 1, "name": "Acme Studios"}],
        "credits": {
            "cast": [{"order": 0, "id": 100, "name": "Lead Star", "character": "Hero"}],
            "crew": [{"job": "Director", "id": 200, "name": "The Director"}],
        },
    }


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return Settings(
        tmdb_api_key="test",
        tmdb_bearer_token=None,
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        sample_counts=10,
        min_vote_count=10,
        start_year=1980,
        min_budget_usd=100_000,
    )


@pytest.fixture(autouse=True)
def _clear_gold_cache() -> None:
    load_gold.clear()


def test_gold_path_exists_flips_after_build(tmp_settings: Settings) -> None:
    assert gold_path_exists(tmp_settings) is False

    _write_bronze_movie(tmp_settings.movies_bronze_dir, _movie_fixture(1))
    silver.build(tmp_settings)
    gold.build(tmp_settings)

    assert gold_path_exists(tmp_settings) is True
    assert gold_path(tmp_settings).is_file()


def test_load_gold_returns_rows_after_build(tmp_settings: Settings) -> None:
    _write_bronze_movie(tmp_settings.movies_bronze_dir, _movie_fixture(1))
    _write_bronze_movie(tmp_settings.movies_bronze_dir, _movie_fixture(2))
    silver.build(tmp_settings)
    gold.build(tmp_settings)

    df = load_gold(str(gold_path(tmp_settings)))
    assert df.height == 2
    assert {"title", "budget_musd", "revenue_musd", "roi"}.issubset(df.columns)


def test_load_gold_empty_when_missing(tmp_settings: Settings) -> None:
    df = load_gold(str(gold_path(tmp_settings)))
    assert df.height == 0


def test_layer_metadata_lists_all_expected_layers(tmp_settings: Settings) -> None:
    _write_bronze_movie(tmp_settings.movies_bronze_dir, _movie_fixture(1))
    _write_bronze_movie(tmp_settings.movies_bronze_dir, _movie_fixture(2))
    silver.build(tmp_settings)
    gold.build(tmp_settings)

    layers = layer_metadata(tmp_settings)
    labels = [layer.layer for layer in layers]
    assert labels == [
        "Bronze / discover",
        "Bronze / movies",
        "Silver / movies",
        "Silver / cast",
        "Silver / crew",
        "Gold / gold_movies",
    ]

    by_label = {layer.layer: layer for layer in layers}
    assert by_label["Bronze / movies"].files == 2
    assert by_label["Bronze / movies"].rows == 2
    assert by_label["Silver / movies"].rows == 2
    assert by_label["Silver / cast"].rows == 2
    assert by_label["Silver / crew"].rows == 2
    assert by_label["Gold / gold_movies"].rows == 2
    assert by_label["Gold / gold_movies"].files == 1
    assert by_label["Gold / gold_movies"].size_mb > 0


def test_scope_constraints_reflects_settings(tmp_settings: Settings) -> None:
    scope = scope_constraints(tmp_settings)
    assert "Movies only" in scope["Media type"]
    assert str(tmp_settings.start_year) in scope["Release window"]
    assert f"{tmp_settings.min_budget_usd:,}" in scope["Gold budget floor"]
    assert scope["Release granularity"] == "Month"


def _analytics_fixture() -> pl.DataFrame:
    """Tiny synthetic frame mirroring the Gold schema used by the helpers."""

    return pl.DataFrame(
        {
            "title": [f"M{i}" for i in range(6)],
            "release_year": [2020] * 6,
            "release_month": [1, 1, 1, 6, 6, 12],
            "budget_musd": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "revenue_musd": [30.0, 60.0, 60.0, 200.0, 250.0, 20.0],
            "roi": [2.0, 2.0, 1.0, 4.0, 4.0, -0.7],
            "vote_average": [6.0, 7.0, 8.0, 7.5, 8.5, 5.0],
            "director_name": [
                "Alice",
                "Alice",
                "Alice",
                "Bob",
                "Bob",
                "Solo",
            ],
            "lead_production_company": [
                "Acme",
                "Acme",
                "Acme",
                "Globex",
                "Globex",
                "Solo Co",
            ],
            "genres": [
                ["Action"],
                ["Action", "Drama"],
                ["Drama"],
                ["Action"],
                ["Action"],
                ["Horror"],
            ],
        },
        schema_overrides={
            "release_month": pl.Int32,
            "release_year": pl.Int32,
            "genres": pl.List(pl.String),
        },
    )


def test_roi_by_genre_explodes_and_aggregates() -> None:
    out = roi_by_genre(_analytics_fixture())
    by_genre = {row["genre"]: row for row in out.to_dicts()}

    assert set(by_genre) == {"Action", "Drama", "Horror"}
    assert by_genre["Action"]["n"] == 4
    assert by_genre["Drama"]["n"] == 2
    assert by_genre["Horror"]["n"] == 1
    assert by_genre["Action"]["avg_roi"] == pytest.approx((2.0 + 2.0 + 4.0 + 4.0) / 4)
    assert by_genre["Drama"]["avg_roi"] == pytest.approx((2.0 + 1.0) / 2)
    avg_rois = out["avg_roi"].to_list()
    assert avg_rois == sorted(avg_rois, reverse=True)


def test_top_directors_respects_min_movies_and_sort() -> None:
    out = top_directors(_analytics_fixture(), n=10, min_movies=2)
    names = out["director_name"].to_list()

    assert "Solo" not in names
    assert names == ["Bob", "Alice"]
    bob = out.filter(pl.col("director_name") == "Bob").row(0, named=True)
    assert bob["n_movies"] == 2
    assert bob["avg_budget_musd"] == pytest.approx(45.0)
    assert bob["avg_vote_average"] == pytest.approx(8.0)


def test_classify_roi_boundaries() -> None:
    df = pl.DataFrame({"roi": [-0.5, 0.0, 1.5, 3.0, 5.0]})
    out = classify_roi(df, hit=3.0, flop=0.0)
    assert out["roi_bucket"].to_list() == [
        "Flop",
        "Flop",
        "Average",
        "Hit",
        "Hit",
    ]


def test_month_genre_matrix_null_masks_sparse_cells() -> None:
    out = month_genre_matrix(_analytics_fixture(), metric="median_roi", min_n=3)
    january_action = out.filter(
        (pl.col("release_month") == 1) & (pl.col("genre") == "Action")
    ).row(0, named=True)
    january_drama = out.filter(
        (pl.col("release_month") == 1) & (pl.col("genre") == "Drama")
    ).row(0, named=True)

    assert january_action["n"] == 2
    assert january_action["value"] is None
    assert january_drama["n"] == 2
    assert january_drama["value"] is None

    count_matrix = month_genre_matrix(_analytics_fixture(), metric="count")
    values = {
        (row["release_month"], row["genre"]): row["value"]
        for row in count_matrix.to_dicts()
    }
    assert values[(1, "Action")] == 2.0
    assert values[(6, "Action")] == 2.0


def test_top_production_companies_respects_min_movies() -> None:
    out = top_production_companies(_analytics_fixture(), n=10, min_movies=2)
    names = out["lead_production_company"].to_list()

    assert "Solo Co" not in names
    assert names == ["Globex", "Acme"]
    globex = out.filter(pl.col("lead_production_company") == "Globex").row(
        0, named=True
    )
    assert globex["n_movies"] == 2
    assert globex["avg_revenue_musd"] == pytest.approx(225.0)
