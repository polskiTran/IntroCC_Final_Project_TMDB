"""Feature engineering for the TMDB ML models.

Takes the Gold `polars.DataFrame` and produces a `pandas.DataFrame` plus the
column groups that `train.py`'s `ColumnTransformer` needs:

- numeric passthrough: ``budget_musd``, ``runtime``
- cyclical: ``release_month`` -> ``month_sin``, ``month_cos``
- multi-hot: one ``genre_<name>`` column per top-K genre seen at fit time
- target-encoded: ``director_name``, ``lead_production_company``,
  ``lead_cast_name`` (handed to sklearn ``TargetEncoder`` as strings)

The fit-time state (top genres list) is serialized alongside the trained model
so the Streamlit page can produce identical features for a single-row inference.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl


TOP_GENRE_K = 15
UNKNOWN = "Unknown"

NUMERIC_COLS: tuple[str, ...] = ("budget_musd", "runtime")
CYCLICAL_COLS: tuple[str, ...] = ("month_sin", "month_cos")
TARGET_ENCODED_COLS: tuple[str, ...] = (
    "director_name",
    "lead_production_company",
    "lead_cast_name",
)


@dataclass
class FeatureSpec:
    """Fit-time state required to transform new rows the same way."""

    top_genres: list[str] = field(default_factory=list)

    @property
    def multihot_cols(self) -> list[str]:
        return [f"genre_{g}" for g in self.top_genres]

    @property
    def feature_columns(self) -> list[str]:
        return (
            list(NUMERIC_COLS)
            + list(CYCLICAL_COLS)
            + self.multihot_cols
            + list(TARGET_ENCODED_COLS)
        )


def fit_feature_spec(df: pl.DataFrame, top_k: int = TOP_GENRE_K) -> FeatureSpec:
    """Determine the top-K genres across the training frame."""
    if df.height == 0 or "genres" not in df.columns:
        return FeatureSpec(top_genres=[])
    counter: Counter[str] = Counter()
    for row in df["genres"].to_list():
        if not row:
            continue
        for g in row:
            if g:
                counter[g] += 1
    top = [g for g, _ in counter.most_common(top_k)]
    return FeatureSpec(top_genres=top)


def _cyclical_month(months: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = months.fill_null(1).cast(pl.Float64).to_numpy()
    radians = 2 * np.pi * (arr - 1) / 12.0
    return np.sin(radians), np.cos(radians)


def _multihot_genres(genres: pl.Series, top_genres: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {
        f"genre_{g}": np.zeros(len(genres), dtype=np.int8) for g in top_genres
    }
    for i, row in enumerate(genres.to_list()):
        if not row:
            continue
        for g in row:
            col = f"genre_{g}"
            if col in out:
                out[col][i] = 1
    return out


def build_feature_frame(df: pl.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Transform a Gold polars frame into a sklearn-ready pandas frame.

    Missing categorical values are replaced with the sentinel ``"Unknown"`` so
    ``TargetEncoder`` treats them as a coherent unseen-category bucket.
    """
    n = df.height
    if n == 0:
        return pd.DataFrame(columns=spec.feature_columns)

    budget = df["budget_musd"].fill_null(0.0).cast(pl.Float64).to_numpy()
    runtime = df["runtime"].fill_null(0).cast(pl.Float64).to_numpy()
    sin_m, cos_m = _cyclical_month(df["release_month"])
    multihot = _multihot_genres(df["genres"], spec.top_genres)

    out = pd.DataFrame(
        {
            "budget_musd": budget,
            "runtime": runtime,
            "month_sin": sin_m,
            "month_cos": cos_m,
            **multihot,
        }
    )
    for col in TARGET_ENCODED_COLS:
        values = df[col].fill_null(UNKNOWN).cast(pl.String).to_list()
        out[col] = [v if v else UNKNOWN for v in values]

    return out[spec.feature_columns]


def build_single_row(
    *,
    spec: FeatureSpec,
    budget_musd: float,
    runtime: float,
    release_month: int,
    genres: list[str],
    director_name: str,
    lead_production_company: str,
    lead_cast_name: str,
) -> pd.DataFrame:
    """Build a 1-row feature frame for inference using the fitted spec."""
    row_df = pl.DataFrame(
        {
            "budget_musd": [float(budget_musd)],
            "runtime": [float(runtime)],
            "release_month": [int(release_month)],
            "genres": [list(genres) if genres else []],
            "director_name": [director_name or UNKNOWN],
            "lead_production_company": [lead_production_company or UNKNOWN],
            "lead_cast_name": [lead_cast_name or UNKNOWN],
        }
    )
    return build_feature_frame(row_df, spec)
