"""Trace one movie across Bronze → Silver → Gold and print where it stops.

Usage:
  uv run python -m src.tools.trace_movie --movie-id 419430
  uv run python -m src.tools.trace_movie --title "Get Out"

If ``--title`` matches more than one bronze file, the command exits with an
error listing IDs (use ``--movie-id``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TextIO

import polars as pl

from src.config import Settings, get_settings
from src.ingest.silver import _movie_row


def _load_bronze_doc(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _find_bronze_by_title(bronze_dir: Path, needle: str) -> list[tuple[int, Path]]:
    needle_cf = needle.casefold()
    matches: list[tuple[int, Path]] = []
    if not bronze_dir.is_dir():
        return matches
    for path in sorted(bronze_dir.glob("*.json.gz")):
        try:
            doc = _load_bronze_doc(path)
        except (OSError, json.JSONDecodeError):
            continue
        mid = doc.get("id")
        if not isinstance(mid, int):
            continue
        title = str(doc.get("title") or "")
        otitle = str(doc.get("original_title") or "")
        if needle_cf in title.casefold() or needle_cf in otitle.casefold():
            matches.append((mid, path))
    return matches


def _silver_violations(row: dict[str, Any], settings: Settings) -> list[str]:
    """Reason strings if silver.build would skip this row; empty if it keeps it."""
    out: list[str] = []
    start = date(settings.start_year, 1, 1)
    today = date.today()
    if row["original_language"] != "en":
        out.append(
            f"original_language={row['original_language']!r} (silver requires 'en')"
        )
    if row["adult"]:
        out.append("adult=True (silver requires non-adult)")
    rd = row["release_date"]
    if rd is None:
        out.append("release_date missing or not parseable as ISO date")
    else:
        if rd < start:
            out.append(f"release_date {rd} before silver window start {start}")
        if rd > today:
            out.append(f"release_date {rd} after today {today}")
    return out


def _gold_violations(budget: int, revenue: int, settings: Settings) -> list[str]:
    out: list[str] = []
    if budget < settings.min_budget_usd:
        out.append(
            f"budget={budget} < min_budget_usd={settings.min_budget_usd} "
            "(gold requires budget above threshold)"
        )
    if revenue <= 0:
        out.append(f"revenue={revenue} (gold requires revenue > 0)")
    return out


@dataclass(frozen=True)
class TracePaths:
    bronze_movie: Path
    silver_movies: Path
    gold_movies: Path


def _paths(settings: Settings) -> TracePaths:
    return TracePaths(
        bronze_movie=settings.movies_bronze_dir,
        silver_movies=settings.silver_dir / "movies.parquet",
        gold_movies=settings.gold_dir / "gold_movies.parquet",
    )


def trace_movie(
    settings: Settings,
    movie_id: int,
    *,
    bronze_doc: dict[str, Any] | None = None,
    bronze_path: Path | None = None,
    sink: TextIO | None = None,
) -> int:
    """Print trace to ``sink`` (stdout by default). Returns process exit code."""
    paths = _paths(settings)
    lines: list[str] = []

    def out(msg: str) -> None:
        lines.append(msg)

    out(f"=== Trace movie_id={movie_id} ===")
    out(
        f"settings: start_year={settings.start_year} min_budget_usd={settings.min_budget_usd}"
    )

    # --- Bronze ---
    expected = paths.bronze_movie / f"{movie_id}.json.gz"
    doc = bronze_doc
    path_used = bronze_path or expected
    bronze_read_error: str | None = None
    if doc is None and expected.is_file():
        try:
            doc = _load_bronze_doc(expected)
            path_used = expected
        except (OSError, json.JSONDecodeError) as exc:
            bronze_read_error = str(exc)

    if bronze_read_error is not None:
        out(f"BRONZE: file exists but failed to read {expected}: {bronze_read_error}")
    elif doc is None:
        out(f"BRONZE: no file at {expected}")
    else:
        out(f"BRONZE: ok — {path_used}")
        out(
            f"  title={doc.get('title')!r} original_title={doc.get('original_title')!r} "
            f"budget={doc.get('budget')} revenue={doc.get('revenue')}"
        )

    # --- Silver (from bronze doc rules) ---
    row = _movie_row(doc) if doc is not None else None
    if row is None and doc is not None:
        out("SILVER (predict from bronze): DROP — could not build row (id must be int)")
    elif row is None:
        out("SILVER (predict from bronze): (no bronze doc — skip prediction)")
    else:
        bad = _silver_violations(row, settings)
        if bad:
            out("SILVER (predict from bronze): would DROP")
            for b in bad:
                out(f"  - {b}")
        else:
            out("SILVER (predict from bronze): would KEEP")

    in_silver = False
    silver_budget: int | None = None
    silver_revenue: int | None = None
    if paths.silver_movies.is_file():
        sm = pl.read_parquet(paths.silver_movies).filter(pl.col("id") == movie_id)
        if sm.height == 1:
            in_silver = True
            r = sm.row(0, named=True)
            silver_budget = int(r["budget"])
            silver_revenue = int(r["revenue"])
            out(
                "SILVER (on disk): PRESENT in movies.parquet — "
                f"budget={silver_budget} revenue={silver_revenue} "
                f"release_date={r.get('release_date')}"
            )
        elif sm.height == 0:
            out("SILVER (on disk): ABSENT from movies.parquet")
        else:
            out(f"SILVER (on disk): ERROR — {sm.height} rows for this id (unexpected)")
    else:
        out(f"SILVER (on disk): no file at {paths.silver_movies}")

    # --- Gold ---
    budget_for_gold = silver_budget
    revenue_for_gold = silver_revenue
    if budget_for_gold is None and row is not None:
        budget_for_gold = int(row["budget"])
        revenue_for_gold = int(row["revenue"])

    if budget_for_gold is not None and revenue_for_gold is not None:
        gbad = _gold_violations(budget_for_gold, revenue_for_gold, settings)
        if gbad:
            out("GOLD (predict from financials): would DROP")
            for b in gbad:
                out(f"  - {b}")
        else:
            out("GOLD (predict from financials): would KEEP")

    in_gold = False
    if paths.gold_movies.is_file():
        gm = pl.read_parquet(paths.gold_movies).filter(pl.col("movie_id") == movie_id)
        if gm.height == 1:
            in_gold = True
            out("GOLD (on disk): PRESENT in gold_movies.parquet")
        elif gm.height == 0:
            out("GOLD (on disk): ABSENT from gold_movies.parquet")
        else:
            out(f"GOLD (on disk): ERROR — {gm.height} rows for this id (unexpected)")
    else:
        out(f"GOLD (on disk): no file at {paths.gold_movies}")

    # --- Consistency hints ---
    if in_silver and not in_gold and silver_budget is not None:
        gbad = _gold_violations(silver_budget, silver_revenue or 0, settings)
        if gbad:
            out(
                "NOTE: Silver present but Gold absent is explained by gold financial filter:"
            )
            for b in gbad:
                out(f"  - {b}")
    if not in_silver and in_gold:
        out(
            "NOTE: Gold row exists without Silver row — rebuild gold or check paths/stale data."
        )

    text = "\n".join(lines) + "\n"
    (sink or sys.stdout).write(text)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--movie-id", type=int, help="TMDB movie id (bronze file name)")
    g.add_argument(
        "--title", type=str, help="Case-insensitive substring on title fields"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    if args.movie_id is not None:
        sys.exit(trace_movie(settings, args.movie_id))

    matches = _find_bronze_by_title(settings.movies_bronze_dir, args.title)
    if not matches:
        sys.stderr.write(
            f"No bronze movie files matched title substring {args.title!r} under "
            f"{settings.movies_bronze_dir}\n"
        )
        sys.exit(1)
    if len(matches) > 1:
        sys.stderr.write(
            f"Multiple bronze matches ({len(matches)}) for title {args.title!r}:\n"
        )
        for mid, path in matches[:50]:
            sys.stderr.write(f"  id={mid}  {path.name}\n")
        if len(matches) > 50:
            sys.stderr.write(f"  ... and {len(matches) - 50} more\n")
        sys.stderr.write("Pass --movie-id to trace a single title.\n")
        sys.exit(2)

    movie_id, path = matches[0]
    doc = _load_bronze_doc(path)
    sys.exit(trace_movie(settings, movie_id, bronze_doc=doc, bronze_path=path))


if __name__ == "__main__":
    main()
