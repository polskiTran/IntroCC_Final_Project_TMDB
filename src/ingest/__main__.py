"""CLI entry for the ingestion pipeline.

Usage:
    uv run python -m src.ingest bronze
    uv run python -m src.ingest silver
    uv run python -m src.ingest all
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from src.config import get_settings
from src.ingest import bronze, silver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.ingest")
    parser.add_argument(
        "stage",
        choices=("bronze", "silver", "all"),
        help="Which stage of the pipeline to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    settings = get_settings()

    match args.stage:
        case "bronze":
            asyncio.run(bronze.run_bronze(settings))
        case "silver":
            silver.build(settings)
        case "all":
            asyncio.run(bronze.run_bronze(settings))
            silver.build(settings)


if __name__ == "__main__":
    main()
