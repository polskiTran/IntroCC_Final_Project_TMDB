"""Tests for Bronze streaming ingestion (mock TMDB client, no network)."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.config import Settings
from src.ingest import bronze
from src.ingest.storage import bronze_movie_path


def _minimal_movie(movie_id: int) -> dict[str, Any]:
    return {
        "id": movie_id,
        "title": f"T{movie_id}",
        "original_title": f"T{movie_id}",
        "release_date": "2005-06-15",
        "budget": 1_000_000,
        "revenue": 5_000_000,
        "runtime": 100,
        "vote_average": 7.0,
        "vote_count": 100,
        "popularity": 10.0,
        "original_language": "en",
        "adult": False,
        "genres": [{"id": 1, "name": "Action"}],
        "production_companies": [{"id": 1, "name": "Acme"}],
        "credits": {"cast": [], "crew": []},
    }


class FakeTMDBClient:
    """Records call order; slows discover pages >= 2 so detail calls can race ahead."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def discover_movies(
        self, page: int, filters: dict[str, Any]
    ) -> dict[str, Any]:
        if page >= 2:
            await asyncio.sleep(0.5)
        self.calls.append(("discover", page))
        if page == 1:
            return {
                "total_pages": 5,
                "total_results": 500,
                "results": [{"id": 101}, {"id": 102}],
            }
        if page == 2:
            return {"results": [{"id": 103}]}
        return {"results": []}

    async def movie_details(
        self, movie_id: int, *, append: str = "credits"
    ) -> dict[str, Any]:
        self.calls.append(("details", movie_id))
        return _minimal_movie(movie_id)


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return Settings(
        tmdb_api_key="test",
        tmdb_bearer_token=None,
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        sample_counts=3,
        min_vote_count=10,
        start_year=1980,
        concurrency=4,
        discover_page_concurrency=5,
        save_discover_pages=False,
    )


def test_details_start_before_later_discover_pages_return(
    tmp_settings: Settings,
) -> None:
    fake = FakeTMDBClient()
    asyncio.run(bronze.run_bronze(tmp_settings, client=fake))

    first_detail_idx = next(
        i for i, (kind, _) in enumerate(fake.calls) if kind == "details"
    )
    idx_discover_2 = next(
        i for i, (kind, v) in enumerate(fake.calls) if kind == "discover" and v == 2
    )
    assert first_detail_idx < idx_discover_2


def test_run_bronze_writes_manifest_and_prefixed_files(tmp_settings: Settings) -> None:
    fake = FakeTMDBClient()
    manifest = asyncio.run(bronze.run_bronze(tmp_settings, client=fake))

    assert manifest["discovered_movie_ids"] == [101, 102, 103]
    assert manifest["details"]["fetched"] == 3

    manifest_path = (
        tmp_settings.bronze_dir
        / "manifests"
        / f"run_date={date.today().isoformat()}"
        / "run_manifest.json"
    )
    assert manifest_path.is_file()
    disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert disk["discovered_movie_ids"] == [101, 102, 103]

    for mid in (101, 102, 103):
        assert bronze_movie_path(tmp_settings.movies_bronze_dir, mid).is_file()


def test_run_bronze_skips_existing_movie_files(tmp_settings: Settings) -> None:
    fake1 = FakeTMDBClient()
    asyncio.run(bronze.run_bronze(tmp_settings, client=fake1))

    fake2 = FakeTMDBClient()
    manifest2 = asyncio.run(bronze.run_bronze(tmp_settings, client=fake2))

    assert manifest2["details"]["skipped"] == 3
    assert manifest2["details"]["fetched"] == 0
