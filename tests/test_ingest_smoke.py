"""Smoke tests for the ingestion pipeline.

These tests synthesize Bronze JSON on disk and drive the Silver builder,
plus a tiny unit test for TMDB client credential resolution. No network.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TMDB_ML_EMBEDDINGS", "0")

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.config import Settings  # noqa: E402
from src.ingest import gold, silver  # noqa: E402
from src.ingest.tmdb_client import TMDBClient, TMDBError  # noqa: E402


def _write_bronze_movie(dir_path: Path, doc: dict[str, Any]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{doc['id']}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(doc, f)


def _movie_fixture(
    movie_id: int,
    *,
    language: str = "en",
    adult: bool = False,
    release_date: str = "2005-06-15",
) -> dict[str, Any]:
    return {
        "id": movie_id,
        "title": f"Movie {movie_id}",
        "original_title": f"Movie {movie_id}",
        "release_date": release_date,
        "budget": 10_000_000,
        "revenue": 50_000_000,
        "runtime": 120,
        "vote_average": 7.5,
        "vote_count": 2000,
        "popularity": 42.0,
        "original_language": language,
        "adult": adult,
        "genres": [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}],
        "production_companies": [{"id": 1, "name": "Acme Studios"}],
        "belongs_to_collection": {"id": 9000, "name": "Demo Franchise"},
        "tagline": "One line that sells the movie",
        "overview": "A gripping ensemble chase across continents.",
        "credits": {
            "cast": [
                {"order": 0, "id": 100, "name": "Lead Star", "character": "Hero"},
                {"order": 1, "id": 101, "name": "Second", "character": "Sidekick"},
                {"order": 2, "id": 102, "name": "Third", "character": "Villain"},
                {"order": 3, "id": 103, "name": "Fourth", "character": "Mentor"},
                {"order": 4, "id": 104, "name": "Fifth", "character": "Comic"},
                {"order": 5, "id": 105, "name": "Sixth", "character": "Cameo"},
            ],
            "crew": [
                {"job": "Director", "id": 200, "name": "The Director"},
                {"job": "Producer", "id": 201, "name": "The Producer"},
                {"job": "Producer", "id": 202, "name": "Another Producer"},
                {"job": "Editor", "id": 203, "name": "Not Kept"},
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


def test_silver_build_produces_parquet(tmp_settings: Settings) -> None:
    movies_bronze = tmp_settings.movies_bronze_dir
    _write_bronze_movie(movies_bronze, _movie_fixture(1))
    _write_bronze_movie(movies_bronze, _movie_fixture(2, release_date="1999-12-31"))
    _write_bronze_movie(movies_bronze, _movie_fixture(3, language="fr"))
    _write_bronze_movie(movies_bronze, _movie_fixture(4, adult=True))
    _write_bronze_movie(movies_bronze, _movie_fixture(5, release_date="1970-01-01"))

    counts = silver.build(tmp_settings)

    assert counts["movies"] == 2
    assert counts["cast"] == 2 * silver.CAST_TOP_N
    assert silver.CAST_TOP_N == 5
    assert counts["crew"] == 2 * 3  # 1 director + 2 producers per kept movie

    movies_df = pl.read_parquet(tmp_settings.silver_dir / "movies.parquet")
    assert set(movies_df["id"].to_list()) == {1, 2}
    assert movies_df.schema["release_date"] == pl.Date
    assert movies_df.schema["genres"] == pl.List(pl.String)
    assert movies_df.schema["budget"] == pl.Int64
    for col in ("collection_id", "collection_name", "tagline", "overview"):
        assert col in movies_df.columns

    cast_df = pl.read_parquet(tmp_settings.silver_dir / "cast.parquet")
    assert set(cast_df["movie_id"].to_list()) == {1, 2}

    crew_df = pl.read_parquet(tmp_settings.silver_dir / "crew.parquet")
    assert set(crew_df["job"].unique().to_list()) <= {"Director", "Producer"}


def test_silver_build_is_idempotent(tmp_settings: Settings) -> None:
    _write_bronze_movie(tmp_settings.movies_bronze_dir, _movie_fixture(1))
    first = silver.build(tmp_settings)
    second = silver.build(tmp_settings)
    assert first == second


def test_tmdb_client_prefers_bearer_token() -> None:
    settings = Settings(tmdb_api_key="apikey123", tmdb_bearer_token="bearer456")
    client = TMDBClient(settings)
    try:
        auth = client._client.headers.get("Authorization")
        assert auth == "Bearer bearer456"
        assert "api_key" not in dict(client._client.params)
    finally:
        asyncio.run(client._client.aclose())


def test_tmdb_client_falls_back_to_api_key() -> None:
    settings = Settings(tmdb_api_key="apikey123", tmdb_bearer_token=None)
    client = TMDBClient(settings)
    try:
        assert "Authorization" not in client._client.headers
        assert dict(client._client.params).get("api_key") == "apikey123"
    finally:
        asyncio.run(client._client.aclose())


def test_tmdb_client_requires_credentials() -> None:
    settings = Settings(tmdb_api_key="", tmdb_bearer_token=None)
    with pytest.raises(TMDBError):
        TMDBClient(settings)


def test_gold_applies_scope_filters(tmp_settings: Settings) -> None:
    movies_bronze = tmp_settings.movies_bronze_dir
    low_budget = _movie_fixture(10)
    low_budget["budget"] = 50_000
    no_revenue = _movie_fixture(11)
    no_revenue["revenue"] = 0
    keeper = _movie_fixture(12)
    keeper["budget"] = 1_000_000
    keeper["revenue"] = 5_000_000
    _write_bronze_movie(movies_bronze, low_budget)
    _write_bronze_movie(movies_bronze, no_revenue)
    _write_bronze_movie(movies_bronze, keeper)
    silver.build(tmp_settings)

    counts = gold.build(tmp_settings)
    assert counts["gold_movies"] == 1
    assert counts["dropped"] == 2

    df = pl.read_parquet(tmp_settings.gold_dir / "gold_movies.parquet")
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["movie_id"] == 12
    assert row["budget_musd"] == pytest.approx(1.0)
    assert row["revenue_musd"] == pytest.approx(5.0)
    assert row["roi"] == pytest.approx(4.0)
    assert "release_date" not in df.columns
    for col in (
        "cast_2_name",
        "cast_3_name",
        "cast_4_name",
        "cast_5_name",
        "n_cast",
        "lead_producer_name",
        "n_producers",
        "collection_name",
        "has_tagline",
        "n_production_companies",
        "n_genres",
        "overview_embedding",
    ):
        assert col in df.columns
    assert 0 < row["n_cast"] <= 5
    assert row["collection_name"] == "Demo Franchise"
    assert row["has_tagline"] == 1
    assert row["n_producers"] == 2
    assert row["lead_producer_name"] in {"The Producer", "Another Producer"}


def test_gold_joins_director_and_lead_cast(tmp_settings: Settings) -> None:
    doc = _movie_fixture(20)
    doc["credits"] = {
        "cast": [
            {"order": 1, "id": 901, "name": "Second Billed", "character": "Side"},
            {"order": 0, "id": 900, "name": "Top Billed", "character": "Lead"},
        ],
        "crew": [
            {"job": "Producer", "id": 801, "name": "A Producer"},
            {"job": "Director", "id": 800, "name": "Jane Director"},
        ],
    }
    _write_bronze_movie(tmp_settings.movies_bronze_dir, doc)
    silver.build(tmp_settings)

    gold.build(tmp_settings)

    df = pl.read_parquet(tmp_settings.gold_dir / "gold_movies.parquet")
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["director_name"] == "Jane Director"
    assert row["director_id"] == 800
    assert row["lead_cast_name"] == "Top Billed"
    assert row["lead_cast_id"] == 900
    assert row["lead_production_company"] == "Acme Studios"
    assert row["n_cast"] == 2
    assert row["lead_producer_name"] == "A Producer"


def test_gold_defaults_collection_and_tagline(tmp_settings: Settings) -> None:
    doc = _movie_fixture(30)
    doc.pop("belongs_to_collection")
    doc["tagline"] = ""
    doc["overview"] = None
    _write_bronze_movie(tmp_settings.movies_bronze_dir, doc)
    silver.build(tmp_settings)
    gold.build(tmp_settings)

    df = pl.read_parquet(tmp_settings.gold_dir / "gold_movies.parquet")
    row = df.row(0, named=True)
    assert row["collection_name"] == "Standalone"
    assert row["has_tagline"] == 0
    assert row["overview"] is None
    embedding = row["overview_embedding"]
    assert embedding is not None and len(embedding) == 16
