"""Filesystem helpers for Bronze artifacts (local today; S3-compatible paths).

Movie files are stored under hashed prefix folders to avoid huge flat directories:

    <movies_root>/id_prefix=012/<movie_id>.json.gz
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


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
