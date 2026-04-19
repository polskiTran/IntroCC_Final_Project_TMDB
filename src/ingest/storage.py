"""Filesystem helpers for Bronze artifacts (local paths and S3 keys).

Movie files are stored under hashed prefix folders to avoid huge flat directories:

    <movies_root>/id_prefix=012/<movie_id>.json.gz
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from src.config import Settings
from src.io.store import s3_object_exists, write_gzipped_json_s3


def id_prefix_dir_name(movie_id: int) -> str:
    """Hive-style folder name: id_prefix is movie_id // 1000 (000–999)."""
    return f"id_prefix={movie_id // 1000:03d}"


def bronze_movie_path(movies_bronze_dir: Path, movie_id: int) -> Path:
    """Path to gzipped raw movie JSON under the Bronze movies root."""
    return movies_bronze_dir / id_prefix_dir_name(movie_id) / f"{movie_id}.json.gz"


def write_json_gz(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write of one gzipped JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)


def exists(path: Path) -> bool:
    return path.is_file()


def write_bronze_movie_json(
    settings: Settings, movie_id: int, payload: dict[str, Any]
) -> None:
    if settings.data_backend != "s3":
        out = bronze_movie_path(settings.movies_bronze_dir, movie_id)
        write_json_gz(out, payload)
        return
    write_gzipped_json_s3(
        settings,
        "bronze",
        "movies",
        id_prefix_dir_name(movie_id),
        f"{movie_id}.json.gz",
        payload=payload,
    )


def bronze_movie_exists(settings: Settings, movie_id: int) -> bool:
    if settings.data_backend != "s3":
        return bronze_movie_path(settings.movies_bronze_dir, movie_id).is_file()
    return s3_object_exists(
        settings,
        "bronze",
        "movies",
        id_prefix_dir_name(movie_id),
        f"{movie_id}.json.gz",
    )
