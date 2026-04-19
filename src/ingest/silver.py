"""Silver layer: parse Bronze movie JSON into typed Parquet tables.

Reads all `*.json.gz` documents under ``settings.movies_bronze_dir``
(recursively), including hive-style ``id_prefix=NNN/<id>.json.gz`` paths.

Produces three files under `settings.silver_dir`:
  - movies.parquet  (one row per movie)
  - cast.parquet    (long, top-N cast per movie)
  - crew.parquet    (long, filtered to Director and Producer)

The builder is idempotent: it rebuilds from Bronze every run, overwriting
the output files in one pass.

Writes are always to local disk; use ``python -m src.ingest upload-s3`` to copy to S3.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

CAST_TOP_N = 1
CREW_JOBS = frozenset({"Director", "Producer"})


def _iter_bronze_movies(bronze_dir: Path) -> Iterator[dict[str, Any]]:
    if not bronze_dir.exists():
        return
    for p in sorted(bronze_dir.rglob("*.json.gz")):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                yield json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable bronze file %s: %s", p, exc)


def _parse_release_date(raw: Any) -> date | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _movie_row(doc: dict[str, Any]) -> dict[str, Any] | None:
    mid = doc.get("id")
    if not isinstance(mid, int):
        return None
    release_date = _parse_release_date(doc.get("release_date"))
    genres = [
        g["name"]
        for g in doc.get("genres") or []
        if isinstance(g, dict) and g.get("name")
    ]
    companies = [
        c["name"]
        for c in doc.get("production_companies") or []
        if isinstance(c, dict) and c.get("name")
    ]
    return {
        "id": mid,
        "title": doc.get("title"),
        "original_title": doc.get("original_title"),
        "release_date": release_date,
        "release_year": release_date.year if release_date else None,
        "release_month": release_date.month if release_date else None,
        "budget": int(doc.get("budget") or 0),
        "revenue": int(doc.get("revenue") or 0),
        "runtime": doc.get("runtime"),
        "vote_average": float(doc.get("vote_average") or 0.0),
        "vote_count": int(doc.get("vote_count") or 0),
        "popularity": float(doc.get("popularity") or 0.0),
        "original_language": doc.get("original_language"),
        "adult": bool(doc.get("adult", False)),
        "genres": genres,
        "production_companies": companies,
    }


def _cast_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    mid = doc.get("id")
    credits = doc.get("credits") or {}
    cast = credits.get("cast") or []
    ordered = sorted(
        (c for c in cast if isinstance(c, dict)),
        key=lambda c: c.get("order", 10_000),
    )[:CAST_TOP_N]
    return [
        {
            "movie_id": mid,
            "order": c.get("order"),
            "person_id": c.get("id"),
            "name": c.get("name"),
            "character": c.get("character"),
        }
        for c in ordered
    ]


def _crew_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    mid = doc.get("id")
    credits = doc.get("credits") or {}
    crew = credits.get("crew") or []
    return [
        {
            "movie_id": mid,
            "job": c.get("job"),
            "person_id": c.get("id"),
            "name": c.get("name"),
        }
        for c in crew
        if isinstance(c, dict) and c.get("job") in CREW_JOBS
    ]


def _movies_schema() -> dict[str, pl.DataType]:
    return {
        "id": pl.Int64(),
        "title": pl.String(),
        "original_title": pl.String(),
        "release_date": pl.Date(),
        "release_year": pl.Int32(),
        "release_month": pl.Int32(),
        "budget": pl.Int64(),
        "revenue": pl.Int64(),
        "runtime": pl.Int32(),
        "vote_average": pl.Float64(),
        "vote_count": pl.Int64(),
        "popularity": pl.Float64(),
        "original_language": pl.String(),
        "adult": pl.Boolean(),
        "genres": pl.List(pl.String()),
        "production_companies": pl.List(pl.String()),
    }


def _cast_schema() -> dict[str, pl.DataType]:
    return {
        "movie_id": pl.Int64(),
        "order": pl.Int32(),
        "person_id": pl.Int64(),
        "name": pl.String(),
        "character": pl.String(),
    }


def _crew_schema() -> dict[str, pl.DataType]:
    return {
        "movie_id": pl.Int64(),
        "job": pl.String(),
        "person_id": pl.Int64(),
        "name": pl.String(),
    }


def _write_silver_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def build(settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    settings.silver_dir.mkdir(parents=True, exist_ok=True)

    movies: list[dict[str, Any]] = []
    cast: list[dict[str, Any]] = []
    crew: list[dict[str, Any]] = []

    start_date = date(settings.start_year, 1, 1)
    today = date.today()

    for doc in _iter_bronze_movies(settings.movies_bronze_dir):
        row = _movie_row(doc)
        if row is None:
            continue
        if row["original_language"] != "en":
            continue
        if row["adult"]:
            continue
        rd = row["release_date"]
        if rd is None or rd < start_date or rd > today:
            continue
        movies.append(row)
        cast.extend(_cast_rows(doc))
        crew.extend(_crew_rows(doc))

    movies_df = pl.DataFrame(movies, schema=_movies_schema())
    cast_df = pl.DataFrame(cast, schema=_cast_schema())
    crew_df = pl.DataFrame(crew, schema=_crew_schema())

    _write_silver_parquet(movies_df, settings.silver_dir / "movies.parquet")
    _write_silver_parquet(cast_df, settings.silver_dir / "cast.parquet")
    _write_silver_parquet(crew_df, settings.silver_dir / "crew.parquet")

    counts = {
        "movies": movies_df.height,
        "cast": cast_df.height,
        "crew": crew_df.height,
    }
    logger.info("silver build complete: %s", counts)
    return counts


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    build()


if __name__ == "__main__":
    main()
