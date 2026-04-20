"""Shared data helpers for the Streamlit pages.

Kept free of Streamlit UI code so the helpers are easy to unit test. The
only Streamlit dependency is ``st.cache_data`` on ``load_gold``, which is safe
to import without a running Streamlit server.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
import streamlit as st

from src.config import Settings, get_settings
from src.io.store import (
    GOLD_FILENAME,
    SILVER_PARQUET_NAMES,
    get_s3_client,
    gold_parquet_ref,
    polars_storage_options,
    s3_object_exists,
    s3_object_key,
    s3_prefix_metrics,
    s3_uri,
)

SILVER_FILES = SILVER_PARQUET_NAMES

# Subset of Gold columns used by the Streamlit Analytics page (smaller IO / memory).
GOLD_ANALYTICS_COLUMNS: tuple[str, ...] = (
    "release_year",
    "vote_count",
    "genres",
    "roi",
    "director_name",
    "budget_musd",
    "revenue_musd",
    "vote_average",
    "title",
    "release_month",
    "lead_production_company",
)

# TMDB movie genre `name` strings (same as Silver `genres` from the API). Used for
# Analytics sidebar options so we never scan the full Gold table to build the list.
TMDB_CHART_GENRE_OPTIONS: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
)

# Default genre filter on the Analytics page (smaller working set on first load).
DEFAULT_ANALYTICS_GENRES: tuple[str, ...] = ("Drama",)


@dataclass(frozen=True)
class LayerInfo:
    layer: str
    path: str
    files: int
    size_mb: float
    rows: int | None
    last_updated: datetime | None


def gold_path(settings: Settings | None = None) -> Path | str:
    settings = settings or get_settings()
    if settings.data_backend != "s3":
        return settings.gold_dir / GOLD_FILENAME
    return s3_uri(settings, "gold", GOLD_FILENAME)


def gold_path_exists(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    ref, opts = gold_parquet_ref(settings)
    if opts is None:
        return Path(ref).is_file()
    return s3_object_exists(settings, "gold", GOLD_FILENAME)


def gold_parquet_stamp(settings: Settings | None = None) -> str:
    """Cheap fingerprint of the Gold parquet on disk or in S3 (mtime+size or ETag+size)."""
    settings = settings or get_settings()
    if settings.data_backend != "s3":
        p = settings.gold_dir / GOLD_FILENAME
        if not p.is_file():
            return "missing"
        st_ = p.stat()
        return f"{st_.st_mtime_ns}:{st_.st_size}"
    if not s3_object_exists(settings, "gold", GOLD_FILENAME):
        return "missing"
    client = get_s3_client(settings)
    key = s3_object_key(settings, "gold", GOLD_FILENAME)
    head = client.head_object(Bucket=settings.s3_bucket, Key=key)
    etag = str(head.get("ETag") or "").strip('"')
    cl = int(head.get("ContentLength") or 0)
    return f"{etag}:{cl}"


@st.cache_data(show_spinner=False)
def load_gold(
    path_str: str,
    parquet_stamp: str,
    columns: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Load the Gold parquet. Empty frame if missing.

    Pass ``parquet_stamp=gold_parquet_stamp(settings)`` so the cache invalidates
    when the file is replaced in place (same path, new bytes).

    Pass ``columns=GOLD_ANALYTICS_COLUMNS`` (or another tuple of names) to read
    only those columns from Parquet.
    """
    _ = parquet_stamp
    cols: list[str] | None = list(columns) if columns is not None else None
    if not path_str.startswith("s3://"):
        p = Path(path_str)
        if not p.is_file():
            return pl.DataFrame()
        return pl.read_parquet(p, columns=cols)

    settings = get_settings()
    if not s3_object_exists(settings, "gold", GOLD_FILENAME):
        return pl.DataFrame()
    so = polars_storage_options(settings) or {}
    try:
        return pl.read_parquet(path_str, storage_options=so, columns=cols)
    except Exception:
        return pl.DataFrame()


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


def _parquet_rows_s3(settings: Settings, *uri_parts: str) -> int | None:
    if not s3_object_exists(settings, *uri_parts):
        return None
    uri = s3_uri(settings, *uri_parts)
    so = polars_storage_options(settings) or {}
    try:
        df = pl.scan_parquet(uri, storage_options=so).select(pl.len()).collect()
        if isinstance(df, pl.DataFrame):
            return int(df.item())
        return None
    except Exception:
        return None


def gold_parquet_row_count(settings: Settings | None = None) -> int | None:
    """Row count for Gold parquet via scan (no full table load)."""
    settings = settings or get_settings()
    if settings.data_backend != "s3":
        return _parquet_rows(settings.gold_dir / GOLD_FILENAME)
    return _parquet_rows_s3(settings, "gold", GOLD_FILENAME)


def _layer_metadata_s3(settings: Settings) -> list[LayerInfo]:
    out: list[LayerInfo] = []

    d_files, d_size, d_last = s3_prefix_metrics(settings, "bronze", "discover")
    out.append(
        LayerInfo(
            layer="Bronze / discover",
            path=s3_uri(settings, "bronze", "discover"),
            files=d_files,
            size_mb=round(d_size, 3),
            rows=None,
            last_updated=d_last,
        )
    )

    m_files, m_size, m_last = s3_prefix_metrics(settings, "bronze", "movies")
    out.append(
        LayerInfo(
            layer="Bronze / movies",
            path=s3_uri(settings, "bronze", "movies"),
            files=m_files,
            size_mb=round(m_size, 3),
            rows=m_files,
            last_updated=m_last,
        )
    )

    for name in SILVER_FILES:
        stem = name.removesuffix(".parquet")
        s_files, s_size, s_last = s3_prefix_metrics(settings, "silver", name)
        fcount = 1 if s_files > 0 else 0
        out.append(
            LayerInfo(
                layer=f"Silver / {stem}",
                path=s3_uri(settings, "silver", name),
                files=fcount,
                size_mb=round(s_size, 3),
                rows=_parquet_rows_s3(settings, "silver", name),
                last_updated=s_last,
            )
        )

    g_files, g_size, g_last = s3_prefix_metrics(settings, "gold", GOLD_FILENAME)
    gfcount = 1 if g_files > 0 else 0
    out.append(
        LayerInfo(
            layer="Gold / gold_movies",
            path=s3_uri(settings, "gold", GOLD_FILENAME),
            files=gfcount,
            size_mb=round(g_size, 3),
            rows=_parquet_rows_s3(settings, "gold", GOLD_FILENAME),
            last_updated=g_last,
        )
    )

    return out


def layer_metadata(settings: Settings | None = None) -> list[LayerInfo]:
    settings = settings or get_settings()
    if settings.data_backend == "s3":
        return _layer_metadata_s3(settings)

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

    gp = settings.gold_dir / GOLD_FILENAME
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
