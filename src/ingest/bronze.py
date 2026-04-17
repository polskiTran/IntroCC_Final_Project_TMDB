"""Bronze layer: fetch raw TMDB JSON and persist it gzipped.

Two phases:
  1) Discover movie ids matching the scope filter, saving each page.
  2) Fetch per-movie details+credits, saving one gzipped JSON per movie.

Idempotent: existing per-movie files are skipped on re-run.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm

from src.config import Settings, get_settings
from src.ingest.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)

_DISCOVER_MAX_PAGE = 500


def _discover_filters(settings: Settings) -> dict[str, Any]:
    today = date.today().isoformat()
    return {
        "with_original_language": "en",
        "include_adult": "false",
        "primary_release_date.gte": f"{settings.start_year}-01-01",
        "primary_release_date.lte": today,
        "vote_count.gte": settings.min_vote_count,
        "sort_by": "popularity.desc",
    }


def _write_json_gz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)


async def _discover_phase(client: TMDBClient, settings: Settings) -> list[int]:
    run_dir = settings.discover_dir / date.today().isoformat()
    filters = _discover_filters(settings)
    movie_ids: list[int] = []
    seen: set[int] = set()

    first = await client.discover_movies(page=1, filters=filters)
    _write_json_gz(run_dir / "page_0001.json.gz", first)
    for r in first.get("results", []):
        mid = r.get("id")
        if isinstance(mid, int) and mid not in seen:
            seen.add(mid)
            movie_ids.append(mid)

    total_pages = min(int(first.get("total_pages", 1)), _DISCOVER_MAX_PAGE)
    logger.info(
        "discover: total_pages=%d total_results=%s target_sample=%d",
        total_pages,
        first.get("total_results"),
        settings.sample_counts,
    )

    page = 2
    while page <= total_pages and len(movie_ids) < settings.sample_counts:
        payload = await client.discover_movies(page=page, filters=filters)
        _write_json_gz(run_dir / f"page_{page:04d}.json.gz", payload)
        for r in payload.get("results", []):
            mid = r.get("id")
            if isinstance(mid, int) and mid not in seen:
                seen.add(mid)
                movie_ids.append(mid)
                if len(movie_ids) >= settings.sample_counts:
                    break
        page += 1

    return movie_ids[: settings.sample_counts]


async def _fetch_one(
    client: TMDBClient, movie_id: int, out_dir: Path
) -> tuple[int, bool]:
    out_path = out_dir / f"{movie_id}.json.gz"
    if out_path.exists():
        return movie_id, False
    payload = await client.movie_details(movie_id, append="credits")
    _write_json_gz(out_path, payload)
    return movie_id, True


async def _details_phase(
    client: TMDBClient, settings: Settings, movie_ids: list[int]
) -> None:
    out_dir = settings.movies_bronze_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [_fetch_one(client, mid, out_dir) for mid in movie_ids]
    fetched = 0
    skipped = 0
    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="movies"):
        _, did_fetch = await coro
        if did_fetch:
            fetched += 1
        else:
            skipped += 1
    logger.info("details: fetched=%d skipped=%d", fetched, skipped)


async def run_bronze(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    async with TMDBClient(settings) as client:
        movie_ids = await _discover_phase(client, settings)
        logger.info("collected %d movie ids", len(movie_ids))
        await _details_phase(client, settings, movie_ids)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(run_bronze())


if __name__ == "__main__":
    main()
