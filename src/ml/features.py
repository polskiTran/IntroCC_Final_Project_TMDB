"""Feature engineering for the TMDB ML models (v2 — richer feature set).

Takes the Gold `polars.DataFrame` and produces a `pandas.DataFrame` with:

- numeric passthrough: budget, runtime, release year, counts, tagline / collection flags
- cyclical: ``release_month`` -> ``month_sin``, ``month_cos``
- multi-hot: top-K genres and top-K production companies seen at fit time
- categorical (string, consumed as native CatBoost cat_features):
  ``director_name``, ``lead_production_company``, ``lead_cast_name``,
  ``cast_2_name`` .. ``cast_5_name``, ``lead_producer_name``,
  ``collection_name``
- overview text: 16 pre-computed PCA-reduced MiniLM embedding components

The Ridge + TargetEncoder baseline keeps target-encoding *only* the original
three high-cardinality cols (director, studio, lead cast) via
``TARGET_ENCODED_COLS``. CatBoost consumes the full ``CATEGORICAL_COLS`` set.

The fit-time state (top genres, top studios) is serialized alongside the
trained model so the Streamlit page can produce identical features for a
single-row inference.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl


TOP_GENRE_K = 25
TOP_STUDIO_K = 30
OVERVIEW_EMBED_DIM = 16
UNKNOWN = "Unknown"
STANDALONE = "Standalone"

NUMERIC_COLS: tuple[str, ...] = (
    "budget_musd",
    "runtime",
    "release_year",
    "n_production_companies",
    "n_genres",
    "n_cast",
    "n_producers",
    "has_tagline",
    "has_collection",
)
CYCLICAL_COLS: tuple[str, ...] = ("month_sin", "month_cos")
TARGET_ENCODED_COLS: tuple[str, ...] = (
    "director_name",
    "lead_production_company",
    "lead_cast_name",
)
EXTRA_CATEGORICAL_COLS: tuple[str, ...] = (
    "cast_2_name",
    "cast_3_name",
    "cast_4_name",
    "cast_5_name",
    "lead_producer_name",
    "collection_name",
)
CATEGORICAL_COLS: tuple[str, ...] = TARGET_ENCODED_COLS + EXTRA_CATEGORICAL_COLS
OVERVIEW_EMBED_COLS: tuple[str, ...] = tuple(
    f"overview_emb_{i}" for i in range(OVERVIEW_EMBED_DIM)
)


@dataclass
class FeatureSpec:
    """Fit-time state required to transform new rows the same way."""

    top_genres: list[str] = field(default_factory=list)
    top_studios: list[str] = field(default_factory=list)

    @property
    def genre_multihot_cols(self) -> list[str]:
        return [f"genre_{g}" for g in self.top_genres]

    @property
    def studio_multihot_cols(self) -> list[str]:
        return [f"studio_{s}" for s in self.top_studios]

    @property
    def multihot_cols(self) -> list[str]:
        return self.genre_multihot_cols + self.studio_multihot_cols

    @property
    def feature_columns(self) -> list[str]:
        return (
            list(NUMERIC_COLS)
            + list(CYCLICAL_COLS)
            + self.multihot_cols
            + list(CATEGORICAL_COLS)
            + list(OVERVIEW_EMBED_COLS)
        )


def categorical_feature_indices(spec: FeatureSpec) -> list[int]:
    """Positions of categorical columns inside ``spec.feature_columns``.

    Used to build the ``cat_features`` argument for ``CatBoostRegressor``.
    """
    cols = spec.feature_columns
    return [cols.index(c) for c in CATEGORICAL_COLS]


def fit_feature_spec(
    df: pl.DataFrame,
    top_k_genres: int = TOP_GENRE_K,
    top_k_studios: int = TOP_STUDIO_K,
) -> FeatureSpec:
    """Determine the top-K genres and production companies in the training frame."""
    if df.height == 0:
        return FeatureSpec()
    genre_counter: Counter[str] = Counter()
    if "genres" in df.columns:
        for row in df["genres"].to_list():
            if not row:
                continue
            for g in row:
                if g:
                    genre_counter[g] += 1
    top_genres = [g for g, _ in genre_counter.most_common(top_k_genres)]

    studio_counter: Counter[str] = Counter()
    if "production_companies" in df.columns:
        for row in df["production_companies"].to_list():
            if not row:
                continue
            for s in row:
                if s:
                    studio_counter[s] += 1
    top_studios = [s for s, _ in studio_counter.most_common(top_k_studios)]

    return FeatureSpec(top_genres=top_genres, top_studios=top_studios)


def _cyclical_month(months: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = months.fill_null(1).cast(pl.Float64).to_numpy()
    radians = 2 * np.pi * (arr - 1) / 12.0
    return np.sin(radians), np.cos(radians)


def _multihot(values: pl.Series, top: list[str], prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {
        f"{prefix}_{item}": np.zeros(len(values), dtype=np.int8) for item in top
    }
    for i, row in enumerate(values.to_list()):
        if not row:
            continue
        for item in row:
            col = f"{prefix}_{item}"
            if col in out:
                out[col][i] = 1
    return out


def _expand_embedding(
    series: pl.Series | None, n_rows: int
) -> dict[str, np.ndarray]:
    cols: dict[str, np.ndarray] = {
        c: np.zeros(n_rows, dtype=np.float32) for c in OVERVIEW_EMBED_COLS
    }
    if series is None:
        return cols
    for i, row in enumerate(series.to_list()):
        if not row:
            continue
        for j, v in enumerate(row):
            if j >= OVERVIEW_EMBED_DIM:
                break
            cols[OVERVIEW_EMBED_COLS[j]][i] = float(v)
    return cols


def _col_or_null(df: pl.DataFrame, name: str, dtype: pl.DataType) -> pl.Series:
    if name in df.columns:
        return df[name]
    return pl.Series(name, [None] * df.height, dtype=dtype)


def build_feature_frame(df: pl.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Transform a Gold polars frame into a model-ready pandas frame.

    Missing categorical values are replaced with the sentinel ``"Unknown"``
    (or ``"Standalone"`` for ``collection_name``). Numeric nulls fall back to
    0. Overview embedding defaults to zeros when the column is absent.
    """
    n = df.height
    if n == 0:
        return pd.DataFrame(columns=spec.feature_columns)

    budget = df["budget_musd"].fill_null(0.0).cast(pl.Float64).to_numpy()
    runtime = df["runtime"].fill_null(0).cast(pl.Float64).to_numpy()
    release_year = (
        _col_or_null(df, "release_year", pl.Int64())
        .fill_null(0)
        .cast(pl.Float64)
        .to_numpy()
    )
    n_prod = (
        _col_or_null(df, "n_production_companies", pl.Int32())
        .fill_null(0)
        .cast(pl.Float64)
        .to_numpy()
    )
    n_gen = (
        _col_or_null(df, "n_genres", pl.Int32())
        .fill_null(0)
        .cast(pl.Float64)
        .to_numpy()
    )
    n_cast = (
        _col_or_null(df, "n_cast", pl.Int32())
        .fill_null(0)
        .cast(pl.Float64)
        .to_numpy()
    )
    n_producers = (
        _col_or_null(df, "n_producers", pl.Int32())
        .fill_null(0)
        .cast(pl.Float64)
        .to_numpy()
    )
    has_tagline = (
        _col_or_null(df, "has_tagline", pl.Int8())
        .fill_null(0)
        .cast(pl.Float64)
        .to_numpy()
    )
    collection_names = (
        _col_or_null(df, "collection_name", pl.String())
        .fill_null(STANDALONE)
        .to_list()
    )
    has_collection = np.array(
        [0.0 if v in (None, "", STANDALONE) else 1.0 for v in collection_names],
        dtype=np.float64,
    )

    sin_m, cos_m = _cyclical_month(df["release_month"])

    genre_multihot = _multihot(df["genres"], spec.top_genres, "genre")
    if "production_companies" in df.columns:
        studio_multihot = _multihot(
            df["production_companies"], spec.top_studios, "studio"
        )
    else:
        studio_multihot = {
            f"studio_{s}": np.zeros(n, dtype=np.int8) for s in spec.top_studios
        }

    out = pd.DataFrame(
        {
            "budget_musd": budget,
            "runtime": runtime,
            "release_year": release_year,
            "n_production_companies": n_prod,
            "n_genres": n_gen,
            "n_cast": n_cast,
            "n_producers": n_producers,
            "has_tagline": has_tagline,
            "has_collection": has_collection,
            "month_sin": sin_m,
            "month_cos": cos_m,
            **genre_multihot,
            **studio_multihot,
        }
    )

    for col in CATEGORICAL_COLS:
        default = STANDALONE if col == "collection_name" else UNKNOWN
        values = (
            _col_or_null(df, col, pl.String()).fill_null(default).cast(pl.String).to_list()
        )
        out[col] = [v if v else default for v in values]

    emb_series = (
        df["overview_embedding"] if "overview_embedding" in df.columns else None
    )
    emb_cols = _expand_embedding(emb_series, n)
    for col, arr in emb_cols.items():
        out[col] = arr

    return out[spec.feature_columns]


def build_single_row(
    *,
    spec: FeatureSpec,
    budget_musd: float,
    runtime: float,
    release_year: int,
    release_month: int,
    genres: list[str],
    production_companies: list[str] | None = None,
    director_name: str,
    lead_production_company: str,
    lead_cast_name: str,
    cast_2_name: str = UNKNOWN,
    cast_3_name: str = UNKNOWN,
    cast_4_name: str = UNKNOWN,
    cast_5_name: str = UNKNOWN,
    lead_producer_name: str = UNKNOWN,
    collection_name: str = STANDALONE,
    has_tagline: bool = False,
    overview_embedding: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Build a 1-row feature frame for inference using the fitted spec."""
    companies = (
        list(production_companies)
        if production_companies
        else ([lead_production_company] if lead_production_company else [])
    )
    n_cast = 1 + sum(
        1
        for c in (cast_2_name, cast_3_name, cast_4_name, cast_5_name)
        if c and c != UNKNOWN
    )
    n_producers = 1 if lead_producer_name and lead_producer_name != UNKNOWN else 0

    if overview_embedding is None:
        emb_list: list[float] = [0.0] * OVERVIEW_EMBED_DIM
    else:
        emb = np.asarray(overview_embedding, dtype=np.float32).ravel().tolist()
        emb_list = (emb + [0.0] * OVERVIEW_EMBED_DIM)[:OVERVIEW_EMBED_DIM]

    row_df = pl.DataFrame(
        {
            "budget_musd": [float(budget_musd)],
            "runtime": [float(runtime)],
            "release_year": [int(release_year)],
            "release_month": [int(release_month)],
            "genres": [list(genres) if genres else []],
            "production_companies": [companies],
            "n_production_companies": [len(companies)],
            "n_genres": [len(genres) if genres else 0],
            "n_cast": [n_cast],
            "n_producers": [n_producers],
            "has_tagline": [1 if has_tagline else 0],
            "director_name": [director_name or UNKNOWN],
            "lead_production_company": [lead_production_company or UNKNOWN],
            "lead_cast_name": [lead_cast_name or UNKNOWN],
            "cast_2_name": [cast_2_name or UNKNOWN],
            "cast_3_name": [cast_3_name or UNKNOWN],
            "cast_4_name": [cast_4_name or UNKNOWN],
            "cast_5_name": [cast_5_name or UNKNOWN],
            "lead_producer_name": [lead_producer_name or UNKNOWN],
            "collection_name": [collection_name or STANDALONE],
            "overview_embedding": [emb_list],
        },
        schema_overrides={
            "overview_embedding": pl.List(pl.Float32),
        },
    )
    return build_feature_frame(row_df, spec)
