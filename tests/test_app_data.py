"""Smoke tests for src/app/_data.py helpers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from src.app._data import (
    gold_path,
    gold_path_exists,
    layer_metadata,
    load_gold,
    scope_constraints,
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
