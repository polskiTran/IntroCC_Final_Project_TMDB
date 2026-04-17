"""Overview landing page for the TMDB Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import polars as pl  # noqa: E402
import streamlit as st  # noqa: E402

from src.app._data import (  # noqa: E402
    gold_path,
    gold_path_exists,
    layer_metadata,
    load_gold,
    scope_constraints,
)
from src.config import get_settings  # noqa: E402


st.set_page_config(page_title="TMDB Analysis", layout="wide")

settings = get_settings()

st.title("TMDB Analysis")
st.caption("Intro Cloud Computing Final Project")

st.header("Abstract")
st.markdown(
    "Perform movie analysis on the TMDB dataset and present the analytics "
    "and machine-learning predictions through Streamlit."
)

st.subheader("Planned pages")
st.markdown(
    """
    1. **Overview** — project abstract + data metadata (this page)
    2. **Sample data pull / inspect** — filter and preview the Gold table
    3. **Data analytics** — ROI by genre, top directors, hit/flop scatter
    4. **ML prediction** — revenue and rating models
    """
)

st.header("Ingestion pipeline")
st.graphviz_chart(
    """
    digraph pipeline {
        rankdir=LR;
        node [shape=box, style=rounded];
        TMDB [label="TMDB API"];
        Bronze [label="Bronze\\n(raw JSON.gz)\\ndiscover/ + movies/"];
        Silver [label="Silver\\n(typed Parquet)\\nmovies, cast, crew"];
        Gold [label="Gold\\n(modeling-ready)\\ngold_movies.parquet"];
        TMDB -> Bronze -> Silver -> Gold;
    }
    """
)
st.caption(
    "Run `uv run python -m src.ingest all` to rebuild every layer end-to-end. "
    "Each stage is idempotent."
)

st.header("Scope constraints")
scope = scope_constraints(settings)
scope_df = pl.DataFrame(
    {"Constraint": list(scope.keys()), "Value": list(scope.values())}
)
st.dataframe(scope_df, width="stretch", hide_index=True)

st.header("Data metadata")
layers = layer_metadata(settings)
meta_df = pl.DataFrame(
    {
        "Layer": [layer.layer for layer in layers],
        "Path": [layer.path for layer in layers],
        "Files": [layer.files for layer in layers],
        "Size (MB)": [layer.size_mb for layer in layers],
        "Rows": [layer.rows for layer in layers],
        "Last updated": [
            layer.last_updated.strftime("%Y-%m-%d %H:%M:%S")
            if layer.last_updated
            else None
            for layer in layers
        ],
    }
)
st.dataframe(meta_df, width="stretch", hide_index=True)

if not gold_path_exists(settings):
    st.warning(
        "Gold table not found. Run `uv run python -m src.ingest all` to build "
        "the Bronze, Silver, and Gold layers before exploring the app."
    )
else:
    st.header("Gold at a glance")
    gold = load_gold(str(gold_path(settings)))
    if gold.height == 0:
        st.info("Gold parquet exists but is empty.")
    else:
        year_col = gold["release_year"].drop_nulls()
        total_rows = gold.height
        distinct_directors = gold["director_name"].drop_nulls().n_unique()
        year_min = cast(int, year_col.min()) if year_col.len() else None
        year_max = cast(int, year_col.max()) if year_col.len() else None
        median_budget = float(cast(float, gold["budget_musd"].median() or 0.0))
        median_roi = float(cast(float, gold["roi"].median() or 0.0))

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Movies", f"{total_rows:,}")
        col2.metric("Directors", f"{distinct_directors:,}")
        col3.metric(
            "Year range",
            f"{year_min}–{year_max}" if year_min and year_max else "—",
        )
        col4.metric("Median budget", f"${median_budget:,.1f} M")
        col5.metric("Median ROI", f"{median_roi:.2f}x")
