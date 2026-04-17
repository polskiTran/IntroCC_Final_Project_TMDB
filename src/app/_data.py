"""Shared data helpers for the Streamlit pages.

Kept free of Streamlit UI code so the helpers are easy to unit test. The
only Streamlit dependency is `st.cache_data` on `load_gold`, which is safe
to import without a running Streamlit server.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
import streamlit as st

from src.config import Settings, get_settings


GOLD_FILENAME = "gold_movies.parquet"
SILVER_FILES = ("movies.parquet", "cast.parquet", "crew.parquet")


@dataclass(frozen=True)
class LayerInfo:
    layer: str
    path: str
    files: int
    size_mb: float
    rows: int | None
    last_updated: datetime | None


def gold_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.gold_dir / GOLD_FILENAME


def gold_path_exists(settings: Settings | None = None) -> bool:
    return gold_path(settings).is_file()


@st.cache_data(show_spinner=False)
def load_gold(path_str: str | None = None) -> pl.DataFrame:
    """Load the Gold parquet. Empty frame if missing.

    `path_str` is the cache key; pass `str(gold_path())` from the caller so
    the cache invalidates when the file is replaced.
    """
    path = Path(path_str) if path_str else gold_path()
    if not path.is_file():
        return pl.DataFrame()
    return pl.read_parquet(path)


def _dir_stats(paths: list[Path]) -> tuple[int, int, datetime | None]:
    files = 0
    total_bytes = 0
    latest_mtime: float | None = None
    for p in paths:
        if not p.is_file():
            continue
        stat = p.stat()
        files += 1
        total_bytes += stat.st_size
        if latest_mtime is None or stat.st_mtime > latest_mtime:
            latest_mtime = stat.st_mtime
    last = datetime.fromtimestamp(latest_mtime) if latest_mtime is not None else None
    return files, total_bytes, last


def _parquet_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        df = pl.scan_parquet(path).select(pl.len()).collect()
        if isinstance(df, pl.DataFrame):
            return int(df.item())
        return None
    except Exception:
        return None


def _bronze_json_count(path: Path) -> int | None:
    """Return count of *.json.gz files under a directory, or None if missing."""
    if not path.is_dir():
        return None
    return sum(1 for _ in path.rglob("*.json.gz"))


def layer_metadata(settings: Settings | None = None) -> list[LayerInfo]:
    settings = settings or get_settings()

    out: list[LayerInfo] = []

    discover_dir = settings.discover_dir
    discover_files = (
        list(discover_dir.rglob("*.json.gz")) if discover_dir.is_dir() else []
    )
    files, size, last = _dir_stats(discover_files)
    out.append(
        LayerInfo(
            layer="Bronze / discover",
            path=str(discover_dir),
            files=files,
            size_mb=round(size / 1_000_000, 3),
            rows=None,
            last_updated=last,
        )
    )

    movies_bronze = settings.movies_bronze_dir
    movie_files = (
        list(movies_bronze.glob("*.json.gz")) if movies_bronze.is_dir() else []
    )
    files, size, last = _dir_stats(movie_files)
    out.append(
        LayerInfo(
            layer="Bronze / movies",
            path=str(movies_bronze),
            files=files,
            size_mb=round(size / 1_000_000, 3),
            rows=_bronze_json_count(movies_bronze),
            last_updated=last,
        )
    )

    for name in SILVER_FILES:
        p = settings.silver_dir / name
        files, size, last = _dir_stats([p])
        out.append(
            LayerInfo(
                layer=f"Silver / {name.removesuffix('.parquet')}",
                path=str(p),
                files=files,
                size_mb=round(size / 1_000_000, 3),
                rows=_parquet_rows(p),
                last_updated=last,
            )
        )

    gp = gold_path(settings)
    files, size, last = _dir_stats([gp])
    out.append(
        LayerInfo(
            layer="Gold / gold_movies",
            path=str(gp),
            files=files,
            size_mb=round(size / 1_000_000, 3),
            rows=_parquet_rows(gp),
            last_updated=last,
        )
    )

    return out


def scope_constraints(settings: Settings | None = None) -> dict[str, str]:
    settings = settings or get_settings()
    return {
        "Media type": "Movies only (no TV)",
        "Language": "English only (original_language == 'en')",
        "Adult content": "Excluded (adult == False)",
        "Release window": f"{settings.start_year} to today",
        "Release granularity": "Month",
        "Gold budget floor": f"budget >= ${settings.min_budget_usd:,}",
        "Gold revenue floor": "revenue > 0",
        "Discover vote_count.gte": str(settings.min_vote_count),
        "Sample cap": f"{settings.sample_counts:,} movies",
    }
