"""CLI entry for the ingestion pipeline.

Usage:
    uv run python -m src.ingest bronze
    uv run python -m src.ingest silver
    uv run python -m src.ingest gold
    uv run python -m src.ingest all
    uv run python -m src.ingest upload-s3
    uv run python -m src.ingest all --upload-s3
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from src.config import get_settings
from src.ingest import bronze, gold, silver
from src.ingest.upload_s3 import upload_medallion_to_s3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.ingest")
    parser.add_argument(
        "stage",
        choices=("bronze", "silver", "gold", "all", "upload-s3"),
        help="Which stage of the pipeline to run.",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="After `all`, run upload-s3 to sync local medallion dirs to S3.",
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
        case "gold":
            gold.build(settings)
        case "all":
            asyncio.run(bronze.run_bronze(settings))
            silver.build(settings)
            gold.build(settings)
            if args.upload_s3:
                upload_medallion_to_s3(settings)
        case "upload-s3":
            upload_medallion_to_s3(settings)


if __name__ == "__main__":
    main()
