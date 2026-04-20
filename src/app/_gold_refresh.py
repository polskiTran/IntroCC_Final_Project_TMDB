"""Sidebar control to reload Gold from disk or S3 and report whether it changed."""

from __future__ import annotations

import streamlit as st

from src.app._data import (
    gold_parquet_row_count,
    gold_parquet_stamp,
    gold_path_exists,
    load_gold,
)
from src.config import Settings

_SNAP_KEY = "_gold_refresh_snap"


def render_gold_refresh_sidebar(settings: Settings) -> None:
    if not gold_path_exists(settings):
        return

    snap = st.session_state.pop(_SNAP_KEY, None)
    if snap is not None:
        after_stamp = gold_parquet_stamp(settings)
        after_rows = gold_parquet_row_count(settings)
        before_stamp = snap["before_stamp"]
        before_rows = snap["before_rows"]
        updated = before_stamp != after_stamp or before_rows != after_rows
        if updated:
            br = "?" if before_rows is None else f"{before_rows:,}"
            ar = "?" if after_rows is None else f"{after_rows:,}"
            st.sidebar.success(
                f"Gold data updated — rows **{br}** → **{ar}** (file version changed)."
            )
        else:
            st.sidebar.info("Already current — Gold file unchanged.")

    st.sidebar.subheader("Gold data")
    if st.sidebar.button(
        "Refresh from Gold",
        help="Reload from local data/gold or S3 per DATA_BACKEND.",
    ):
        st.session_state[_SNAP_KEY] = {
            "before_stamp": gold_parquet_stamp(settings),
            "before_rows": gold_parquet_row_count(settings),
        }
        load_gold.clear()
        st.rerun()
