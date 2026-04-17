"""CLI entry for the ML module.

Usage:
    uv run python -m src.ml train
    uv run python -m src.ml predict \\
        --budget 80 --runtime 120 --month 6 \\
        --genres Action,Adventure \\
        --director "Christopher Nolan" \\
        --studio "Warner Bros. Pictures" \\
        --cast "Leonardo DiCaprio"
"""

from __future__ import annotations

import argparse
import logging

from src.ml import predict, train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.ml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train both models from the Gold parquet.")

    pred = sub.add_parser(
        "predict",
        help="Load the saved bundles and predict revenue and rating for one movie.",
    )
    pred.add_argument("--budget", type=float, required=True, help="Budget in $M.")
    pred.add_argument("--runtime", type=float, default=110.0, help="Runtime in min.")
    pred.add_argument("--year", type=int, default=2024, help="Release year.")
    pred.add_argument("--month", type=int, default=6, help="Release month (1-12).")
    pred.add_argument(
        "--genres",
        type=str,
        default="",
        help="Comma-separated genre names (e.g. 'Action,Adventure').",
    )
    pred.add_argument("--director", type=str, default="Unknown")
    pred.add_argument("--studio", type=str, default="Unknown")
    pred.add_argument("--cast", type=str, default="Unknown")
    pred.add_argument("--cast2", type=str, default="Unknown")
    pred.add_argument("--cast3", type=str, default="Unknown")
    pred.add_argument("--cast4", type=str, default="Unknown")
    pred.add_argument("--cast5", type=str, default="Unknown")
    pred.add_argument("--producer", type=str, default="Unknown")
    pred.add_argument("--collection", type=str, default="Standalone")
    pred.add_argument("--has-tagline", action="store_true")

    return parser


def _cmd_predict(args: argparse.Namespace) -> None:
    genres = [g.strip() for g in args.genres.split(",") if g.strip()]

    def _run(bundle: predict.LoadedBundle) -> float:
        return predict.predict_one(
            bundle,
            budget_musd=float(args.budget),
            runtime=float(args.runtime),
            release_year=int(args.year),
            release_month=int(args.month),
            genres=genres,
            director_name=str(args.director),
            lead_production_company=str(args.studio),
            lead_cast_name=str(args.cast),
            cast_2_name=str(args.cast2),
            cast_3_name=str(args.cast3),
            cast_4_name=str(args.cast4),
            cast_5_name=str(args.cast5),
            lead_producer_name=str(args.producer),
            collection_name=str(args.collection),
            has_tagline=bool(args.has_tagline),
        )

    revenue = _run(predict.load_bundle("revenue"))
    rating = _run(predict.load_bundle("rating"))
    print(f"Predicted revenue: ${revenue:,.1f}M")
    print(f"Predicted rating:  {rating:.2f} / 10")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    args = _build_parser().parse_args()
    match args.command:
        case "train":
            train.train()
        case "predict":
            _cmd_predict(args)


if __name__ == "__main__":
    main()
