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
        list(movies_bronze.rglob("*.json.gz")) if movies_bronze.is_dir() else []
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


def roi_by_genre(df: pl.DataFrame) -> pl.DataFrame:
    """Average ROI per genre after exploding the list column.

    Returns columns `genre, avg_roi, n` sorted by `avg_roi` desc. Empty input
    yields an empty frame with the same schema.
    """
    if df.height == 0 or "genres" not in df.columns:
        return pl.DataFrame(
            {"genre": [], "avg_roi": [], "n": []},
            schema={"genre": pl.String, "avg_roi": pl.Float64, "n": pl.UInt32},
        )
    return (
        df.select("genres", "roi")
        .explode("genres")
        .drop_nulls("genres")
        .group_by("genres")
        .agg(
            pl.col("roi").mean().alias("avg_roi"),
            pl.len().alias("n"),
        )
        .rename({"genres": "genre"})
        .sort("avg_roi", descending=True)
    )


def top_directors(df: pl.DataFrame, n: int = 15, min_movies: int = 3) -> pl.DataFrame:
    """Top directors by average budget with a companion avg-vote-average column.

    Filters to directors with at least `min_movies` entries, then keeps the
    top `n` by `avg_budget_musd`.
    """
    if df.height == 0 or "director_name" not in df.columns:
        return pl.DataFrame(
            schema={
                "director_name": pl.String,
                "n_movies": pl.UInt32,
                "avg_budget_musd": pl.Float64,
                "avg_vote_average": pl.Float64,
            }
        )
    return (
        df.drop_nulls("director_name")
        .group_by("director_name")
        .agg(
            pl.len().alias("n_movies"),
            pl.col("budget_musd").mean().alias("avg_budget_musd"),
            pl.col("vote_average").mean().alias("avg_vote_average"),
        )
        .filter(pl.col("n_movies") >= min_movies)
        .sort("avg_budget_musd", descending=True)
        .head(n)
    )


def classify_roi(df: pl.DataFrame, hit: float = 3.0, flop: float = 0.0) -> pl.DataFrame:
    """Attach a `roi_bucket` column with values in {Flop, Average, Hit}.

    Flop when `roi <= flop`, Hit when `roi >= hit`, else Average.
    """
    if df.height == 0 or "roi" not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("roi_bucket"))
    return df.with_columns(
        pl.when(pl.col("roi") <= flop)
        .then(pl.lit("Flop"))
        .when(pl.col("roi") >= hit)
        .then(pl.lit("Hit"))
        .otherwise(pl.lit("Average"))
        .alias("roi_bucket")
    )


_MONTH_GENRE_METRICS = {
    "median_roi": ("roi", "median"),
    "median_revenue_musd": ("revenue_musd", "median"),
    "count": (None, "count"),
}


def month_genre_matrix(
    df: pl.DataFrame, metric: str = "median_roi", min_n: int = 3
) -> pl.DataFrame:
    """Aggregate by `release_month` x exploded `genres`.

    Returns columns `release_month, genre, value, n`. For non-`count` metrics,
    `value` is null when the cell has fewer than `min_n` observations so the
    caller can grey-out sparse cells.
    """
    if metric not in _MONTH_GENRE_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {list(_MONTH_GENRE_METRICS)}"
        )
    empty_schema = {
        "release_month": pl.Int32,
        "genre": pl.String,
        "value": pl.Float64,
        "n": pl.UInt32,
    }
    if df.height == 0 or "genres" not in df.columns:
        return pl.DataFrame(schema=empty_schema)

    exploded = (
        df.select("release_month", "genres", "roi", "revenue_musd")
        .explode("genres")
        .drop_nulls(["release_month", "genres"])
        .rename({"genres": "genre"})
    )
    source_col, op = _MONTH_GENRE_METRICS[metric]
    if op == "count":
        agg_expr = pl.len().cast(pl.Float64).alias("value")
    else:
        assert source_col is not None
        agg_expr = pl.col(source_col).median().alias("value")

    grouped = (
        exploded.group_by(["release_month", "genre"])
        .agg(agg_expr, pl.len().alias("n"))
        .sort(["release_month", "genre"])
    )
    if op != "count" and min_n > 0:
        grouped = grouped.with_columns(
            pl.when(pl.col("n") < min_n)
            .then(None)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    return grouped


def top_production_companies(
    df: pl.DataFrame, n: int = 15, min_movies: int = 3
) -> pl.DataFrame:
    """Top lead production companies by average revenue (million USD)."""
    if df.height == 0 or "lead_production_company" not in df.columns:
        return pl.DataFrame(
            schema={
                "lead_production_company": pl.String,
                "n_movies": pl.UInt32,
                "avg_revenue_musd": pl.Float64,
                "median_roi": pl.Float64,
            }
        )
    return (
        df.drop_nulls("lead_production_company")
        .group_by("lead_production_company")
        .agg(
            pl.len().alias("n_movies"),
            pl.col("revenue_musd").mean().alias("avg_revenue_musd"),
            pl.col("roi").median().alias("median_roi"),
        )
        .filter(pl.col("n_movies") >= min_movies)
        .sort("avg_revenue_musd", descending=True)
        .head(n)
    )


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
