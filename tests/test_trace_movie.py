"""Tests for ``src.tools.trace_movie`` pipeline tracer."""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Any

import pytest

from src.config import Settings
from src.ingest import gold, silver
from src.ingest.storage import bronze_movie_path
from src.tools.trace_movie import trace_movie


def _write_bronze_movie(dir_path: Path, doc: dict[str, Any]) -> None:
    path = bronze_movie_path(dir_path, doc["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(doc, f)


def _movie_doc(
    movie_id: int,
    *,
    language: str = "en",
    adult: bool = False,
    release_date: str = "2005-06-15",
    budget: int = 10_000_000,
    revenue: int = 50_000_000,
) -> dict[str, Any]:
    return {
        "id": movie_id,
        "title": f"Movie {movie_id}",
        "original_title": f"Movie {movie_id}",
        "release_date": release_date,
        "budget": budget,
        "revenue": revenue,
        "runtime": 120,
        "vote_average": 7.5,
        "vote_count": 2000,
        "popularity": 42.0,
        "original_language": language,
        "adult": adult,
        "genres": [{"id": 28, "name": "Action"}],
        "production_companies": [{"id": 1, "name": "Acme Studios"}],
        "credits": {
            "cast": [
                {"order": 0, "id": 100, "name": "Lead Star", "character": "Hero"},
            ],
            "crew": [
                {"job": "Director", "id": 200, "name": "The Director"},
            ],
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


def test_trace_shows_silver_drop_for_non_english(tmp_settings: Settings) -> None:
    mid = 301
    _write_bronze_movie(tmp_settings.movies_bronze_dir, _movie_doc(mid, language="fr"))
    silver.build(tmp_settings)

    buf = io.StringIO()
    trace_movie(tmp_settings, mid, sink=buf)
    text = buf.getvalue()
    assert "SILVER (predict from bronze): would DROP" in text
    assert "original_language" in text
    assert "SILVER (on disk): ABSENT" in text


def test_trace_shows_gold_drop_when_revenue_zero(tmp_settings: Settings) -> None:
    mid = 302
    doc = _movie_doc(mid, revenue=0)
    _write_bronze_movie(tmp_settings.movies_bronze_dir, doc)
    silver.build(tmp_settings)
    gold.build(tmp_settings)

    buf = io.StringIO()
    trace_movie(tmp_settings, mid, sink=buf)
    text = buf.getvalue()
    assert "SILVER (on disk): PRESENT" in text
    assert "GOLD (on disk): ABSENT" in text
    assert "GOLD (predict from financials): would DROP" in text
    assert "revenue > 0" in text


def test_trace_end_to_end_kept(tmp_settings: Settings) -> None:
    mid = 303
    _write_bronze_movie(tmp_settings.movies_bronze_dir, _movie_doc(mid))
    silver.build(tmp_settings)
    gold.build(tmp_settings)

    buf = io.StringIO()
    trace_movie(tmp_settings, mid, sink=buf)
    text = buf.getvalue()
    assert "SILVER (predict from bronze): would KEEP" in text
    assert "SILVER (on disk): PRESENT" in text
    assert "GOLD (on disk): PRESENT" in text
