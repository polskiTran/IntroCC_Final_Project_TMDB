"""Gold layer: modeling-ready table joining Silver movies with director and lead cast.

Produces `gold_movies.parquet` under `settings.gold_dir` with the scope
constraints from AGENTS.md enforced: `budget >= min_budget_usd` and `revenue > 0`.
Release granularity is monthly (release_date column is dropped).

Gold is a full rebuild from Silver on every run, so the step is idempotent.

Writes are always to local disk; use ``python -m src.ingest upload-s3`` to copy to S3.
"""

from __future__ import annotations

import logging

import polars as pl

from src.config import Settings, get_settings
from src.io.store import GOLD_FILENAME

logger = logging.getLogger(__name__)


def _read_silver_parquet(settings: Settings, name: str) -> pl.DataFrame:
    path = settings.silver_dir / name
    return pl.read_parquet(path)


_GOLD_COLUMNS = [
    "movie_id",
    "title",
    "original_title",
    "release_year",
    "release_month",
    "budget",
    "budget_musd",
    "revenue",
    "revenue_musd",
    "roi",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "genres",
    "production_companies",
    "lead_production_company",
    "director_id",
    "director_name",
    "lead_cast_id",
    "lead_cast_name",
]


def _load_director(settings: Settings) -> pl.DataFrame:
    crew = _read_silver_parquet(settings, "crew.parquet")
    return (
        crew.filter(pl.col("job") == "Director")
        .group_by("movie_id")
        .agg(
            pl.col("person_id").first().alias("director_id"),
            pl.col("name").first().alias("director_name"),
        )
    )


def _load_lead_cast(settings: Settings) -> pl.DataFrame:
    cast = _read_silver_parquet(settings, "cast.parquet")
    return (
        cast.sort("order")
        .group_by("movie_id")
        .agg(
            pl.col("person_id").first().alias("lead_cast_id"),
            pl.col("name").first().alias("lead_cast_name"),
        )
    )


def build(settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    logger.info("gold: build starting from local silver at %s", settings.silver_dir)
    settings.gold_dir.mkdir(parents=True, exist_ok=True)

    movies = _read_silver_parquet(settings, "movies.parquet")
    director = _load_director(settings)
    lead_cast = _load_lead_cast(settings)

    joined = (
        movies.rename({"id": "movie_id"})
        .join(director, on="movie_id", how="left")
        .join(lead_cast, on="movie_id", how="left")
    )

    filtered = joined.filter(
        (pl.col("budget") >= settings.min_budget_usd) & (pl.col("revenue") > 0)
    )

    enriched = filtered.with_columns(
        (pl.col("budget") / 1_000_000).alias("budget_musd"),
        (pl.col("revenue") / 1_000_000).alias("revenue_musd"),
        ((pl.col("revenue") - pl.col("budget")) / pl.col("budget")).alias("roi"),
        pl.col("production_companies").list.first().alias("lead_production_company"),
    )

    gold = enriched.select(_GOLD_COLUMNS)
    out = settings.gold_dir / GOLD_FILENAME
    tmp = out.with_suffix(out.suffix + ".tmp")
    gold.write_parquet(tmp)
    tmp.replace(out)

    counts = {
        "silver_movies": movies.height,
        "gold_movies": gold.height,
        "dropped": movies.height - gold.height,
    }
    logger.info("gold build complete: %s", counts)
    return counts


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    build()


if __name__ == "__main__":
    main()
