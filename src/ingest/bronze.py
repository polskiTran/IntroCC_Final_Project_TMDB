"""Bronze layer: fetch raw TMDB JSON and persist it gzipped.

Streaming pipeline:
  - Discover pages (page 1 sequential, further pages in bounded concurrent batches).
  - Each discovered movie id is queued immediately so detail fetches can start
    while discover continues.
  - Per-movie files live under id_prefix=NNN/ for S3-friendly prefixes.

Idempotent: existing per-movie files are skipped on re-run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from tqdm.contrib.logging import tqdm_logging_redirect
from tqdm.std import tqdm as std_tqdm

from src.config import Settings, get_settings
from src.ingest.storage import (
    bronze_movie_exists,
    write_bronze_movie_json,
    write_json_gz,
)
from src.ingest.tmdb_client import TMDBClient
from src.io.store import write_gzipped_json_s3, write_text_s3

logger = logging.getLogger(__name__)

_DISCOVER_MAX_PAGE = 500


class _StopSentinel:
    """Queue marker so workers know discover is finished."""


STOP = _StopSentinel()


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


def _manifest_path(settings: Settings) -> Path:
    return (
        settings.bronze_dir
        / "manifests"
        / f"run_date={date.today().isoformat()}"
        / "run_manifest.json"
    )


def _write_manifest(settings: Settings, payload: dict[str, Any]) -> None:
    if settings.data_backend != "s3":
        path = _manifest_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
        return
    write_text_s3(
        settings,
        "bronze",
        "manifests",
        f"run_date={date.today().isoformat()}",
        "run_manifest.json",
        text=json.dumps(payload, indent=2),
    )


@dataclass
class _DetailStats:
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


async def _fetch_one(
    client: TMDBClient,
    movie_id: int,
    settings: Settings,
    stats: _DetailStats,
) -> None:
    if bronze_movie_exists(settings, movie_id):
        stats.skipped += 1
        return
    try:
        payload = await client.movie_details(movie_id, append="credits")
        await asyncio.to_thread(write_bronze_movie_json, settings, movie_id, payload)
        stats.fetched += 1
    except Exception as exc:  # noqa: BLE001 — record and continue
        stats.failed += 1
        msg = f"movie_id={movie_id}: {exc}"
        stats.errors.append(msg)
        logger.warning("bronze detail fetch failed %s", msg)


async def _detail_worker(
    worker_id: int,
    client: TMDBClient,
    queue: asyncio.Queue[int | _StopSentinel],
    settings: Settings,
    stats: _DetailStats,
    pbar: std_tqdm,
) -> None:
    del worker_id  # reserved for debugging
    while True:
        item = await queue.get()
        try:
            if isinstance(item, _StopSentinel):
                return
            assert isinstance(item, int)
            await _fetch_one(client, item, settings, stats)
            pbar.update(1)
        finally:
            queue.task_done()


async def _discover_producer(
    client: TMDBClient,
    settings: Settings,
    queue: asyncio.Queue[int | _StopSentinel],
    discovered_ids: list[int],
    num_workers: int,
) -> None:
    filters = _discover_filters(settings)
    seen: set[int] = set()
    run_dir = settings.discover_dir / date.today().isoformat()

    async def fetch_page(page: int) -> dict[str, Any]:
        return await client.discover_movies(page=page, filters=filters)

    async def ingest_results(payload: dict[str, Any]) -> None:
        for r in payload.get("results", []):
            mid = r.get("id")
            if not isinstance(mid, int) or mid in seen:
                continue
            if len(seen) >= settings.sample_counts:
                break
            seen.add(mid)
            discovered_ids.append(mid)
            await queue.put(mid)

    first = await fetch_page(1)
    if settings.save_discover_pages:
        if settings.data_backend != "s3":
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json_gz(run_dir / "page_0001.json.gz", first)
        else:
            await asyncio.to_thread(
                write_gzipped_json_s3,
                settings,
                "bronze",
                "discover",
                date.today().isoformat(),
                "page_0001.json.gz",
                payload=first,
            )

    await ingest_results(first)

    total_pages = min(int(first.get("total_pages", 1)), _DISCOVER_MAX_PAGE)
    logger.info(
        "discover: total_pages=%d total_results=%s target_sample=%d",
        total_pages,
        first.get("total_results"),
        settings.sample_counts,
    )

    page_batch = settings.discover_page_concurrency
    page = 2
    while page <= total_pages and len(seen) < settings.sample_counts:
        batch_end = min(page + page_batch - 1, total_pages)
        pages = list(range(page, batch_end + 1))
        results = await asyncio.gather(*[fetch_page(p) for p in pages])

        for p, payload in zip(pages, results, strict=True):
            if len(seen) >= settings.sample_counts:
                break
            if settings.save_discover_pages:
                if settings.data_backend != "s3":
                    run_dir.mkdir(parents=True, exist_ok=True)
                    write_json_gz(run_dir / f"page_{p:04d}.json.gz", payload)
                else:
                    await asyncio.to_thread(
                        write_gzipped_json_s3,
                        settings,
                        "bronze",
                        "discover",
                        date.today().isoformat(),
                        f"page_{p:04d}.json.gz",
                        payload=payload,
                    )
            await ingest_results(payload)

        page = batch_end + 1

    for _ in range(num_workers):
        await queue.put(STOP)


@contextmanager
def _suppress_http_library_info_logging() -> Iterator[None]:
    """Avoid httpx/httpcore per-request INFO lines (e.g. ``... 200 OK``) during tqdm."""
    names = (
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
    )
    saved: list[tuple[logging.Logger, int]] = []
    for name in names:
        log = logging.getLogger(name)
        saved.append((log, log.level))
        if log.level <= logging.INFO:
            log.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for log, prev in saved:
            log.setLevel(prev)


async def run_bronze(
    settings: Settings | None = None,
    *,
    client: Any = None,
) -> dict[str, Any]:
    """Run Bronze ingestion. Optional injected ``client`` is for tests."""
    settings = settings or get_settings()
    if client is not None:
        return await _run_bronze_with_client(settings, client)

    async with TMDBClient(settings) as tmcd:
        return await _run_bronze_with_client(settings, tmcd)


async def _run_bronze_with_client(
    settings: Settings,
    client: Any,
) -> dict[str, Any]:
    if settings.data_backend != "s3":
        settings.movies_bronze_dir.mkdir(parents=True, exist_ok=True)

    discovered_ids: list[int] = []
    stats = _DetailStats()

    queue_max = max(200, settings.sample_counts + settings.concurrency * 4)
    queue: asyncio.Queue[int | _StopSentinel] = asyncio.Queue(maxsize=queue_max)

    num_workers = max(1, settings.concurrency)

    # tqdm_logging_redirect: route console logging through tqdm.write so lines
    # don't clobber the bar; httpx INFO request logs are muted for the same reason.
    with _suppress_http_library_info_logging():
        with tqdm_logging_redirect(
            total=settings.sample_counts,
            desc="movie details",
            unit="movie",
            dynamic_ncols=True,
            mininterval=0.25,
        ) as pbar:
            producer_task = asyncio.create_task(
                _discover_producer(client, settings, queue, discovered_ids, num_workers)
            )

            workers = [
                asyncio.create_task(
                    _detail_worker(
                        i,
                        client,
                        queue,
                        settings,
                        stats,
                        pbar,
                    )
                )
                for i in range(num_workers)
            ]

            await asyncio.gather(producer_task, *workers)

    manifest: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "filters": _discover_filters(settings),
        "sample_counts_target": settings.sample_counts,
        "discovered_movie_ids": discovered_ids,
        "details": {
            "fetched": stats.fetched,
            "skipped": stats.skipped,
            "failed": stats.failed,
            "errors": stats.errors[:50],
        },
        "save_discover_pages": settings.save_discover_pages,
    }
    _write_manifest(settings, manifest)
    logger.info(
        "bronze complete: discovered=%d fetched=%d skipped=%d failed=%d",
        len(discovered_ids),
        stats.fetched,
        stats.skipped,
        stats.failed,
    )
    return manifest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(run_bronze())


if __name__ == "__main__":
    main()
