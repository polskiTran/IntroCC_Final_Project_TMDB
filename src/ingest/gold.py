"""Gold layer: modeling-ready table joining Silver movies with director, cast, producer.

Produces `gold_movies.parquet` under `settings.gold_dir` with the scope
constraints from AGENTS.md enforced: `budget >= min_budget_usd` and `revenue > 0`.
Release granularity is monthly (release_date column is dropped).

Gold is a full rebuild from Silver on every run, so the step is idempotent.
Adds the MiniLM overview embedding as a 16-dim PCA-compressed List[Float32]
column so downstream ML can join it without re-running the sentence encoder.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from src.config import Settings, get_settings
from src.ingest.embeddings import OVERVIEW_EMBED_DIM, embed_overviews

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
    "n_production_companies",
    "n_genres",
    "director_id",
    "director_name",
    "lead_cast_id",
    "lead_cast_name",
    "cast_2_name",
    "cast_3_name",
    "cast_4_name",
    "cast_5_name",
    "n_cast",
    "lead_producer_name",
    "n_producers",
    "collection_name",
    "has_tagline",
    "overview",
    "overview_embedding",
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


def _load_producers(silver_dir) -> pl.DataFrame:  # type: ignore[no-untyped-def]
    crew = pl.read_parquet(silver_dir / "crew.parquet")
    return (
        crew.filter(pl.col("job") == "Producer")
        .group_by("movie_id")
        .agg(
            pl.col("name").first().alias("lead_producer_name"),
            pl.len().cast(pl.Int32).alias("n_producers"),
        )
    )


def _load_cast(silver_dir) -> pl.DataFrame:  # type: ignore[no-untyped-def]
    """Pivot top-5 cast into wide columns + lead_cast_id/name + n_cast."""
    cast = pl.read_parquet(silver_dir / "cast.parquet")
    if cast.height == 0:
        return pl.DataFrame(
            schema={
                "movie_id": pl.Int64(),
                "lead_cast_id": pl.Int64(),
                "lead_cast_name": pl.String(),
                "cast_2_name": pl.String(),
                "cast_3_name": pl.String(),
                "cast_4_name": pl.String(),
                "cast_5_name": pl.String(),
                "n_cast": pl.Int32(),
            }
        )
    ranked = cast.sort(["movie_id", "order"]).with_columns(
        pl.col("order").cum_count().over("movie_id").alias("rank")
    )
    lead = (
        ranked.filter(pl.col("rank") == 1)
        .select(
            pl.col("movie_id"),
            pl.col("person_id").alias("lead_cast_id"),
            pl.col("name").alias("lead_cast_name"),
        )
    )
    counts = (
        cast.group_by("movie_id").agg(pl.len().cast(pl.Int32).alias("n_cast"))
    )
    wide = lead.join(counts, on="movie_id", how="left")
    for rank in (2, 3, 4, 5):
        col = f"cast_{rank}_name"
        part = (
            ranked.filter(pl.col("rank") == rank)
            .select(pl.col("movie_id"), pl.col("name").alias(col))
        )
        wide = wide.join(part, on="movie_id", how="left")
    return wide


def _overview_embedding_column(
    overviews: list[str | None], settings: Settings
) -> list[list[float]]:
    arr = embed_overviews(overviews, settings=settings)
    if arr.size == 0:
        return []
    if arr.shape[1] != OVERVIEW_EMBED_DIM:
        raise ValueError(
            f"expected {OVERVIEW_EMBED_DIM}-dim overview embedding, got {arr.shape[1]}"
        )
    return arr.astype(np.float32).tolist()


def build(settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    silver_dir = settings.silver_dir
    gold_dir = settings.gold_dir
    gold_dir.mkdir(parents=True, exist_ok=True)

    movies = pl.read_parquet(silver_dir / "movies.parquet")
    director = _load_director(silver_dir)
    producers = _load_producers(silver_dir)
    cast_wide = _load_cast(silver_dir)

    joined = (
        movies.rename({"id": "movie_id"})
        .join(director, on="movie_id", how="left")
        .join(cast_wide, on="movie_id", how="left")
        .join(producers, on="movie_id", how="left")
    )

    filtered = joined.filter(
        (pl.col("budget") >= settings.min_budget_usd) & (pl.col("revenue") > 0)
    )

    enriched = filtered.with_columns(
        (pl.col("budget") / 1_000_000).alias("budget_musd"),
        (pl.col("revenue") / 1_000_000).alias("revenue_musd"),
        ((pl.col("revenue") - pl.col("budget")) / pl.col("budget")).alias("roi"),
        pl.col("production_companies").list.first().alias("lead_production_company"),
        pl.col("production_companies")
        .list.len()
        .cast(pl.Int32)
        .alias("n_production_companies"),
        pl.col("genres").list.len().cast(pl.Int32).alias("n_genres"),
        pl.col("n_cast").fill_null(0),
        pl.col("n_producers").fill_null(0),
        pl.col("collection_name").fill_null("Standalone"),
        (pl.col("tagline").is_not_null() & (pl.col("tagline").str.len_chars() > 0))
        .cast(pl.Int8)
        .alias("has_tagline"),
    )

    overviews = enriched["overview"].to_list()
    embeddings = _overview_embedding_column(overviews, settings=settings)
    if not embeddings:
        embeddings = [[0.0] * OVERVIEW_EMBED_DIM for _ in range(enriched.height)]
    enriched = enriched.with_columns(
        pl.Series(
            "overview_embedding",
            embeddings,
            dtype=pl.List(pl.Float32),
        )
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
