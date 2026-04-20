"""Data analytics page: genre ROI, director leaderboard, hit/flop scatter, plus
release seasonality heatmap and top production companies."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import altair as alt  # noqa: E402
import polars as pl  # noqa: E402
import streamlit as st  # noqa: E402

from src.app._data import (  # noqa: E402
    DEFAULT_ANALYTICS_GENRES,
    GOLD_ANALYTICS_COLUMNS,
    GOLD_FILENAME,
    TMDB_CHART_GENRE_OPTIONS,
    classify_roi,
    gold_parquet_stamp,
    gold_path,
    gold_path_exists,
    load_gold,
    month_genre_matrix,
    roi_by_genre,
    top_directors,
    top_production_companies,
)
from src.app._gold_refresh import render_gold_refresh_sidebar  # noqa: E402
from src.config import get_settings  # noqa: E402

_SCATTER_CHART_MAX_ROWS = 6_000

_MONTH_LABEL_ORDER = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
_RELEASE_MONTH_LOOKUP = pl.DataFrame(
    {
        "release_month": list(range(1, 13)),
        "month_label": _MONTH_LABEL_ORDER,
    },
    schema={"release_month": pl.Int32, "month_label": pl.String},
)

# One chart per full rerun keeps the page responsive; sidebar changes still rerun
# the script, but Polars + Altair only run for the visible chart unless "All" is on.
_VIEW_GENRE = "1 · ROI by genre"
_VIEW_DIRECTORS = "2 · Top directors"
_VIEW_HIT_FLOP = "3 · Hit / flop"
_VIEW_SEASON = "4 · Seasonality"
_VIEW_STUDIOS = "5 · Top studios"
_VIEW_ALL = "All charts (slower)"
_CHART_VIEWS: tuple[str, ...] = (
    _VIEW_GENRE,
    _VIEW_DIRECTORS,
    _VIEW_HIT_FLOP,
    _VIEW_SEASON,
    _VIEW_STUDIOS,
    _VIEW_ALL,
)

st.set_page_config(page_title="Analytics", layout="wide")
st.title("Data analytics")
st.caption(
    f"Charts over the Gold modeling table (`{GOLD_FILENAME}`). Sidebar filters "
    "apply to the chart you select below. Pick one chart at a time for best "
    "responsiveness, or open “All charts” to render everything at once."
)

settings = get_settings()

if not gold_path_exists(settings):
    st.error(
        "Gold table not found. Run `uv run python -m src.ingest all` to build "
        "the pipeline before using this page."
    )
    st.stop()

render_gold_refresh_sidebar(settings)
_gold_stamp = gold_parquet_stamp(settings)
df = load_gold(str(gold_path(settings)), _gold_stamp, columns=GOLD_ANALYTICS_COLUMNS)
if df.height == 0:
    st.warning("Gold parquet is empty — the pipeline produced no rows.")
    st.stop()

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

vote_count_max = int(cast(int, df["vote_count"].max() or 0))
min_votes = st.sidebar.slider(
    "Minimum vote_count",
    min_value=0,
    max_value=min(vote_count_max, 5_000),
    value=min(50, vote_count_max),
    step=10,
    help="Suppress noisy ROI outliers from obscure titles.",
)

selected_genres = st.sidebar.multiselect(
    "Genres (any match; clear = all)",
    options=list(TMDB_CHART_GENRE_OPTIONS),
    default=list(DEFAULT_ANALYTICS_GENRES),
    help=(
        "Official TMDB genre names. Defaults to Drama for a smaller first load; "
        "clear the selection to include every genre."
    ),
)

filtered = df.filter(
    (pl.col("release_year") >= year_range[0])
    & (pl.col("release_year") <= year_range[1])
    & (pl.col("vote_count") >= min_votes)
)
if selected_genres:
    filtered = filtered.filter(
        pl.col("genres").list.eval(pl.element().is_in(selected_genres)).list.any()
    )

st.markdown(f"**Movies in view:** {filtered.height:,} of {df.height:,}")

if filtered.height == 0:
    st.info("No rows match the current filters.")
    st.stop()

chart_focus = st.segmented_control(
    "Show chart",
    options=list(_CHART_VIEWS),
    default=_VIEW_GENRE,
    key="analytics_chart_focus",
    help=(
        "Streamlit reruns the whole script when sidebar filters change. Showing "
        "one chart avoids recomputing every figure on each change."
    ),
    label_visibility="visible",
)


def _show_chart(which: str) -> bool:
    return chart_focus == _VIEW_ALL or chart_focus == which


def _section_genre_roi(data: pl.DataFrame) -> None:
    st.header("1. Average ROI by genre")
    genre_roi = roi_by_genre(data)
    if genre_roi.height == 0:
        st.info("No genre data available.")
        return
    chart = (
        alt.Chart(genre_roi.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("avg_roi:Q", title="Average ROI (x)"),
            y=alt.Y("genre:N", sort="-x", title=None),
            color=alt.Color(
                "avg_roi:Q", scale=alt.Scale(scheme="viridis"), legend=None
            ),
            tooltip=[
                alt.Tooltip("genre:N", title="Genre"),
                alt.Tooltip("avg_roi:Q", title="Avg ROI", format=".2f"),
                alt.Tooltip("n:Q", title="Movies"),
            ],
        )
        .properties(height=28 * max(genre_roi.height, 1) + 40)
    )
    st.altair_chart(chart, width="stretch")


if _show_chart(_VIEW_GENRE):
    _section_genre_roi(filtered)


@st.fragment
def _section_top_directors(data: pl.DataFrame) -> None:
    st.header("2. Top directors — avg budget vs avg rating")
    col_a, col_b = st.columns(2)
    with col_a:
        director_n = st.slider("Top N directors", 5, 30, 15, step=1)
    with col_b:
        director_min = st.slider("Minimum movies per director", 1, 10, 3, step=1)

    directors = top_directors(data, n=director_n, min_movies=director_min)
    if directors.height == 0:
        st.info(
            "No directors meet the minimum-movies threshold for the current filter."
        )
        return
    directors_pd = directors.to_pandas()
    order = directors_pd["director_name"].tolist()
    base = alt.Chart(directors_pd).encode(
        x=alt.X(
            "director_name:N", sort=order, title=None, axis=alt.Axis(labelAngle=-35)
        )
    )
    bars = base.mark_bar(color="#4C78A8").encode(
        y=alt.Y("avg_budget_musd:Q", title="Avg budget (million USD)"),
        tooltip=[
            alt.Tooltip("director_name:N", title="Director"),
            alt.Tooltip("n_movies:Q", title="Movies"),
            alt.Tooltip("avg_budget_musd:Q", title="Avg budget ($M)", format=".1f"),
            alt.Tooltip("avg_vote_average:Q", title="Avg vote", format=".2f"),
        ],
    )
    line = base.mark_line(color="#F58518", point=True).encode(
        y=alt.Y(
            "avg_vote_average:Q",
            title="Avg vote_average",
            scale=alt.Scale(domain=[0, 10]),
        ),
    )
    chart = alt.layer(bars, line).resolve_scale(y="independent").properties(height=420)
    st.altair_chart(chart, width="stretch")


if _show_chart(_VIEW_DIRECTORS):
    _section_top_directors(filtered)


@st.fragment
def _section_hit_flop_scatter(data: pl.DataFrame) -> None:
    st.header("3. Hit / Average / Flop — budget vs revenue")
    col_c, col_d = st.columns(2)
    with col_c:
        hit_threshold = st.slider(
            "Hit ROI threshold", min_value=1.0, max_value=10.0, value=3.0, step=0.5
        )
    with col_d:
        flop_threshold = st.slider(
            "Flop ROI threshold", min_value=-1.0, max_value=1.0, value=0.0, step=0.1
        )

    scatter_full = (
        classify_roi(data, hit=hit_threshold, flop=flop_threshold)
        .filter((pl.col("budget_musd") > 0) & (pl.col("revenue_musd") > 0))
        .select(
            "title",
            "director_name",
            "release_year",
            "budget_musd",
            "revenue_musd",
            "roi",
            "roi_bucket",
        )
    )
    if scatter_full.height == 0:
        st.info("No positive-budget/positive-revenue rows in the current filter.")
        return

    if scatter_full.height > _SCATTER_CHART_MAX_ROWS:
        scatter_chart_df = scatter_full.sample(
            n=_SCATTER_CHART_MAX_ROWS, seed=0, shuffle=True
        )
        st.caption(
            f"Scatter shows {_SCATTER_CHART_MAX_ROWS:,} randomly sampled movies "
            f"of {scatter_full.height:,} (full dataset used for bucket counts below)."
        )
    else:
        scatter_chart_df = scatter_full

    scatter_pd = scatter_chart_df.to_pandas()
    color_scale = alt.Scale(
        domain=["Flop", "Average", "Hit"],
        range=["#E45756", "#BAB0AC", "#54A24B"],
    )
    points = (
        alt.Chart(scatter_pd)
        .mark_circle(opacity=0.6)
        .encode(
            x=alt.X(
                "budget_musd:Q",
                scale=alt.Scale(type="log"),
                title="Budget (million USD, log)",
            ),
            y=alt.Y(
                "revenue_musd:Q",
                scale=alt.Scale(type="log"),
                title="Revenue (million USD, log)",
            ),
            color=alt.Color("roi_bucket:N", scale=color_scale, title="Bucket"),
            size=alt.Size("roi:Q", scale=alt.Scale(range=[30, 300]), legend=None),
            tooltip=[
                alt.Tooltip("title:N", title="Title"),
                alt.Tooltip("director_name:N", title="Director"),
                alt.Tooltip("release_year:Q", title="Year"),
                alt.Tooltip("budget_musd:Q", title="Budget ($M)", format=".1f"),
                alt.Tooltip("revenue_musd:Q", title="Revenue ($M)", format=".1f"),
                alt.Tooltip("roi:Q", title="ROI", format=".2f"),
                alt.Tooltip("roi_bucket:N", title="Bucket"),
            ],
        )
    )
    budget_min = float(cast(float, scatter_full["budget_musd"].min() or 0.1))
    budget_max = float(cast(float, scatter_full["budget_musd"].max() or 1.0))
    ref_df = pl.DataFrame(
        {
            "budget_musd": [budget_min, budget_max, budget_min, budget_max],
            "revenue_musd": [
                budget_min,
                budget_max,
                budget_min * (1 + hit_threshold),
                budget_max * (1 + hit_threshold),
            ],
            "label": [
                "Break-even",
                "Break-even",
                f"Hit line (roi={hit_threshold:g})",
                f"Hit line (roi={hit_threshold:g})",
            ],
        }
    ).to_pandas()
    rules = (
        alt.Chart(ref_df)
        .mark_line(strokeDash=[4, 4])
        .encode(
            x="budget_musd:Q",
            y="revenue_musd:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=["Break-even", f"Hit line (roi={hit_threshold:g})"],
                    range=["#888888", "#54A24B"],
                ),
                title=None,
            ),
        )
    )
    st.altair_chart((points + rules).properties(height=500), width="stretch")

    counts = (
        scatter_full.group_by("roi_bucket")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    st.dataframe(counts, width="stretch", hide_index=True)


if _show_chart(_VIEW_HIT_FLOP):
    _section_hit_flop_scatter(filtered)


@st.fragment
def _section_month_genre_heatmap(data: pl.DataFrame) -> None:
    st.header("4. Release-month seasonality by genre")
    st.caption(
        "Darker cells mean higher metric value. Cells with fewer than 3 movies are "
        "left blank to avoid over-reading thin samples. Directly informs the "
        "`release_month` feature in the ML page."
    )
    metric_label = st.radio(
        "Metric",
        options=["Median ROI", "Median revenue ($M)", "Movie count"],
        horizontal=True,
    )
    metric_key = {
        "Median ROI": "median_roi",
        "Median revenue ($M)": "median_revenue_musd",
        "Movie count": "count",
    }[metric_label]
    matrix = month_genre_matrix(data, metric=metric_key)
    if matrix.height == 0:
        st.info("No release-month data for the current filter.")
        return

    matrix_pd = (
        matrix.with_columns(pl.col("release_month").cast(pl.Int32))
        .join(_RELEASE_MONTH_LOOKUP, on="release_month", how="left")
        .to_pandas()
    )
    heatmap = (
        alt.Chart(matrix_pd)
        .mark_rect()
        .encode(
            x=alt.X(
                "month_label:N",
                sort=_MONTH_LABEL_ORDER,
                title="Release month",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("genre:N", sort="-x", title=None),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="magma"),
                title=metric_label,
            ),
            tooltip=[
                alt.Tooltip("month_label:N", title="Month"),
                alt.Tooltip("genre:N", title="Genre"),
                alt.Tooltip("value:Q", title=metric_label, format=".2f"),
                alt.Tooltip("n:Q", title="Movies"),
            ],
        )
        .properties(height=22 * matrix_pd["genre"].nunique() + 60)
    )
    st.altair_chart(heatmap, width="stretch")


if _show_chart(_VIEW_SEASON):
    _section_month_genre_heatmap(filtered)


@st.fragment
def _section_top_studios(data: pl.DataFrame) -> None:
    st.header("5. Top production companies")
    st.caption(
        "Studio-level complement to the director leaderboard. Bar length shows "
        "average revenue; color shows median ROI."
    )
    col_e, col_f = st.columns(2)
    with col_e:
        studio_n = st.slider("Top N studios", 5, 30, 15, step=1, key="studio_n")
    with col_f:
        studio_min = st.slider(
            "Minimum movies per studio", 1, 10, 3, step=1, key="studio_min"
        )

    studios = top_production_companies(data, n=studio_n, min_movies=studio_min)
    if studios.height == 0:
        st.info(
            "No production companies meet the minimum-movies threshold for the "
            "current filter."
        )
        return
    studios_pd = studios.to_pandas()
    chart = (
        alt.Chart(studios_pd)
        .mark_bar()
        .encode(
            x=alt.X("avg_revenue_musd:Q", title="Avg revenue (million USD)"),
            y=alt.Y("lead_production_company:N", sort="-x", title=None),
            color=alt.Color(
                "median_roi:Q",
                scale=alt.Scale(scheme="viridis"),
                title="Median ROI",
            ),
            tooltip=[
                alt.Tooltip("lead_production_company:N", title="Studio"),
                alt.Tooltip("n_movies:Q", title="Movies"),
                alt.Tooltip(
                    "avg_revenue_musd:Q", title="Avg revenue ($M)", format=".1f"
                ),
                alt.Tooltip("median_roi:Q", title="Median ROI", format=".2f"),
            ],
        )
        .properties(height=28 * studios.height + 40)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(studios, width="stretch", hide_index=True)


if _show_chart(_VIEW_STUDIOS):
    _section_top_studios(filtered)
