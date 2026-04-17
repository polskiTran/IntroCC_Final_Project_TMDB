"""Sample data inspector over the Gold modeling table."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import polars as pl  # noqa: E402
import streamlit as st  # noqa: E402

from src.app._data import (  # noqa: E402
    GOLD_FILENAME,
    gold_path,
    gold_path_exists,
    load_gold,
)
from src.config import get_settings  # noqa: E402

DEFAULT_COLUMNS = [
    "title",
    "release_year",
    "release_month",
    "genres",
    "budget_musd",
    "revenue_musd",
    "roi",
    "vote_average",
    "director_name",
    "lead_cast_name",
]


st.set_page_config(page_title="Sample Data", layout="wide")
st.title("Sample data — Gold")
st.caption(
    f"Inspect the modeling-ready table (`{GOLD_FILENAME}`) built by the ingestion "
    "pipeline. Filters below operate on the already-ingested local parquet; "
    "re-run the CLI pipeline to refresh."
)

settings = get_settings()

if not gold_path_exists(settings):
    st.error(
        "Gold table not found. Run `uv run python -m src.ingest all` to build "
        "the pipeline before using this page."
    )
    st.stop()

df = load_gold(str(gold_path(settings)))
if df.height == 0:
    st.warning("Gold parquet is empty — the pipeline produced no rows.")
    st.stop()

st.markdown(f"**Total rows in Gold:** {df.height:,}")

with st.expander("Schema", expanded=False):
    schema_df = pl.DataFrame(
        {
            "column": list(df.schema.keys()),
            "dtype": [str(dt) for dt in df.schema.values()],
        }
    )
    st.dataframe(schema_df, width="stretch", hide_index=True)

st.sidebar.header("Filters")

year_min = int(cast(int, df["release_year"].min() or 1980))
year_max = int(cast(int, df["release_year"].max() or 2025))
year_range = st.sidebar.slider(
    "Release year",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1,
)

all_genres = sorted({g for gs in df["genres"].to_list() if gs for g in gs})
selected_genres = st.sidebar.multiselect("Genres (any match)", options=all_genres)

director_query = st.sidebar.text_input("Director name contains").strip()

budget_min = float(cast(float, df["budget_musd"].min() or 0.0))
budget_max = float(cast(float, df["budget_musd"].max() or 0.0))
budget_range = st.sidebar.slider(
    "Budget (million USD)",
    min_value=float(round(budget_min, 2)),
    max_value=float(round(budget_max, 2)),
    value=(float(round(budget_min, 2)), float(round(budget_max, 2))),
)

filtered = df.filter(
    (pl.col("release_year") >= year_range[0])
    & (pl.col("release_year") <= year_range[1])
    & (pl.col("budget_musd") >= budget_range[0])
    & (pl.col("budget_musd") <= budget_range[1])
)

if selected_genres:
    filtered = filtered.filter(
        pl.col("genres").list.eval(pl.element().is_in(selected_genres)).list.any()
    )

if director_query:
    filtered = filtered.filter(
        pl.col("director_name")
        .fill_null("")
        .str.to_lowercase()
        .str.contains(director_query.lower(), literal=True)
    )

st.markdown(f"**Filtered rows:** {filtered.height:,}")

available_columns = list(df.schema.keys())
default_cols = [c for c in DEFAULT_COLUMNS if c in available_columns]
selected_cols = st.multiselect(
    "Columns to show",
    options=available_columns,
    default=default_cols,
)

preview_n = st.slider(
    "Preview row count", min_value=10, max_value=1000, value=100, step=10
)

if filtered.height == 0:
    st.info("No rows match the current filters.")
else:
    display_cols = selected_cols or default_cols
    st.dataframe(
        filtered.select(display_cols).head(preview_n),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Summary statistics (filtered subset)")
    numeric_cols = [
        c
        for c in ["budget_musd", "revenue_musd", "roi", "vote_average", "vote_count"]
        if c in filtered.columns
    ]
    if numeric_cols:
        st.dataframe(
            filtered.select(numeric_cols).describe(),
            width="stretch",
            hide_index=True,
        )

    csv_frame = filtered.with_columns(
        [
            pl.col(name).list.join(", ").alias(name)
            for name, dtype in filtered.schema.items()
            if isinstance(dtype, pl.List)
        ]
    )
    st.download_button(
        "Download filtered CSV",
        data=csv_frame.write_csv().encode("utf-8"),
        file_name="tmdb_gold_sample.csv",
        mime="text/csv",
    )
