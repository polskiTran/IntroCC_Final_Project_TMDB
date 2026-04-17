"""Gold layer: modeling-ready table joining Silver movies with director and lead cast.

Produces `gold_movies.parquet` under `settings.gold_dir` with the scope
constraints from AGENTS.md enforced: `budget >= min_budget_usd` and `revenue > 0`.
Release granularity is monthly (release_date column is dropped).

Gold is a full rebuild from Silver on every run, so the step is idempotent.
"""

from __future__ import annotations

import logging

import polars as pl

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


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


def _load_director(silver_dir) -> pl.DataFrame:  # type: ignore[no-untyped-def]
    crew = pl.read_parquet(silver_dir / "crew.parquet")
    return (
        crew.filter(pl.col("job") == "Director")
        .group_by("movie_id")
        .agg(
            pl.col("person_id").first().alias("director_id"),
            pl.col("name").first().alias("director_name"),
        )
    )


def _load_lead_cast(silver_dir) -> pl.DataFrame:  # type: ignore[no-untyped-def]
    cast = pl.read_parquet(silver_dir / "cast.parquet")
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
    silver_dir = settings.silver_dir
    gold_dir = settings.gold_dir
    gold_dir.mkdir(parents=True, exist_ok=True)

    movies = pl.read_parquet(silver_dir / "movies.parquet")
    director = _load_director(silver_dir)
    lead_cast = _load_lead_cast(silver_dir)

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
    gold.write_parquet(gold_dir / "gold_movies.parquet")

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
