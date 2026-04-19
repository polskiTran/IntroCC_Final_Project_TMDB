"""Upload local Bronze/Silver/Gold trees to S3 using the same key layout as readers."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.config import Settings
from src.io.store import get_s3_client, s3_object_key

logger = logging.getLogger(__name__)


def _iter_local_files_and_keys(settings: Settings) -> list[tuple[Path, str]]:
    jobs: list[tuple[Path, str]] = []
    for base, layer in (
        (settings.bronze_dir, "bronze"),
        (settings.silver_dir, "silver"),
        (settings.gold_dir, "gold"),
    ):
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                rel = path.relative_to(base)
                key = s3_object_key(settings, layer, *rel.parts)
                jobs.append((path, key))
    return jobs


def upload_medallion_to_s3(settings: Settings) -> dict[str, Any]:
    """Upload everything under ``bronze_dir``, ``silver_dir``, and ``gold_dir`` to the bucket."""
    bucket = str(settings.s3_bucket).strip()
    if not bucket:
        msg = "s3_bucket is required for upload-s3"
        raise ValueError(msg)

    jobs = _iter_local_files_and_keys(settings)
    if not jobs:
        logger.warning("upload-s3: no files under bronze/silver/gold directories")
        return {"files": 0, "bytes": 0}

    client = get_s3_client(settings)
    total_bytes = sum(p.stat().st_size for p, _ in jobs)
    n = len(jobs)
    logger.info(
        "upload-s3: uploading %d files (~%.1f MB) to s3://%s/",
        n,
        total_bytes / 1_000_000,
        bucket,
    )

    done = 0
    uploaded_bytes = 0
    lock = threading.Lock()
    workers = max(1, settings.s3_upload_concurrency)

    def _upload(path: Path, key: str) -> int:
        client.upload_file(str(path), bucket, key)
        return path.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_upload, p, k): (p, k) for p, k in jobs}
        for fut in as_completed(futs):
            sz = fut.result()
            with lock:
                done += 1
                uploaded_bytes += sz
                step = max(1, n // 10)
                if done == 1 or done == n or done % step == 0:
                    logger.info(
                        "upload-s3: progress %d/%d files (~%.1f / ~%.1f MB)",
                        done,
                        n,
                        uploaded_bytes / 1_000_000,
                        total_bytes / 1_000_000,
                    )

    logger.info(
        "upload-s3: finished %d files (~%.1f MB)",
        n,
        uploaded_bytes / 1_000_000,
    )
    return {"files": n, "bytes": uploaded_bytes}
