"""Smoke tests for the ingestion pipeline.

These tests synthesize Bronze JSON on disk and drive the Silver builder,
plus a tiny unit test for TMDB client credential resolution. No network.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.config import Settings
from src.ingest import silver
from src.ingest.tmdb_client import TMDBClient, TMDBError


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
        "credits": {
            "cast": [
                {"order": 0, "id": 100, "name": "Lead Star", "character": "Hero"},
                {"order": 1, "id": 101, "name": "Second", "character": "Sidekick"},
            ],
            "crew": [
                {"job": "Director", "id": 200, "name": "The Director"},
                {"job": "Producer", "id": 201, "name": "The Producer"},
                {"job": "Editor", "id": 202, "name": "Not Kept"},
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
        sample_counts=10,
        min_vote_count=10,
        start_year=1980,
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
    assert counts["cast"] == 4
    assert counts["crew"] == 4

    movies_df = pl.read_parquet(tmp_settings.silver_dir / "movies.parquet")
    assert set(movies_df["id"].to_list()) == {1, 2}
    assert movies_df.schema["release_date"] == pl.Date
    assert movies_df.schema["genres"] == pl.List(pl.String)
    assert movies_df.schema["budget"] == pl.Int64

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
