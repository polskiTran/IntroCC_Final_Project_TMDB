"""ML prediction page: findings from the trained models and interactive inference.

The page never retrains. Everything it shows (metrics, permutation importance,
the predicted-vs-actual scatter) comes from the joblib bundles produced by
``uv run python -m src.ml train``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import streamlit as st  # noqa: E402

from src.app._data import gold_path, gold_path_exists, load_gold  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.ml.predict import LoadedBundle, bundle_exists, load_bundle, predict_one  # noqa: E402


st.set_page_config(page_title="ML prediction", layout="wide")
st.title("ML prediction")
st.caption(
    "Two `HistGradientBoostingRegressor` models (revenue & rating) trained on "
    "the Gold table with leakage-safe target encoding for director / studio / "
    "lead cast. Run `uv run python -m src.ml train` to refresh the artefacts."
)

settings = get_settings()

if not gold_path_exists(settings):
    st.error(
        "Gold table not found. Run `uv run python -m src.ingest all` to build "
        "the pipeline before using this page."
    )
    st.stop()

if not (bundle_exists("revenue", settings) and bundle_exists("rating", settings)):
    st.error(
        "Model bundles not found under `data/ml/`. Train them with "
        "`uv run python -m src.ml train` and reload this page."
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def _load(target: str, _cache_key: str) -> LoadedBundle:
    return load_bundle(target)


def _cache_key(target: str) -> str:
    path = settings.ml_dir / f"model_{target}.joblib"
    return f"{path}:{path.stat().st_mtime_ns if path.is_file() else 0}"


revenue = _load("revenue", _cache_key("revenue"))
rating = _load("rating", _cache_key("rating"))

gold = load_gold(str(gold_path(settings)))


MONTH_LABELS = [
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


def _metrics_frame(bundle: LoadedBundle) -> pd.DataFrame:
    hold = bundle.metadata["holdout_metrics"]
    cv = bundle.metadata["cv_metrics"]
    ridge = bundle.metadata["baseline_holdout_metrics"]
    rows = [
        {
            "Split": "Holdout (HGB)",
            "R²": hold["r2"],
            "MAE": hold["mae"],
            "RMSE": hold["rmse"],
        },
        {
            "Split": "5-fold CV (HGB)",
            "R²": cv["r2"],
            "MAE": cv["mae"],
            "RMSE": cv["rmse"],
        },
        {
            "Split": "Holdout (Ridge baseline)",
            "R²": ridge["r2"],
            "MAE": ridge["mae"],
            "RMSE": ridge["rmse"],
        },
    ]
    return pd.DataFrame(rows)


def _render_findings(
    bundle: LoadedBundle,
    *,
    y_label: str,
    y_format: str,
    extra_note: str | None = None,
) -> None:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Train rows", f"{bundle.metadata['n_train']:,}")
    col_b.metric("Test rows", f"{bundle.metadata['n_test']:,}")
    col_c.metric("Holdout R²", f"{bundle.metadata['holdout_metrics']['r2']:.3f}")

    st.subheader("Metrics")
    st.caption(
        "Metrics are in the **training space** (log1p for revenue, native for "
        "rating). Revenue has an additional $M panel below."
    )
    st.dataframe(
        _metrics_frame(bundle),
        width="stretch",
        hide_index=True,
        column_config={
            "R²": st.column_config.NumberColumn(format="%.3f"),
            "MAE": st.column_config.NumberColumn(format="%.3f"),
            "RMSE": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    rev_space = bundle.metadata.get("revenue_space_metrics")
    if rev_space:
        st.caption("Revenue back-transformed to $M:")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "MAE ($M)": rev_space["mae_musd"],
                        "RMSE ($M)": rev_space["rmse_musd"],
                        "MAPE (revenue ≥ $1M)": rev_space["mape_over_1M"],
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "MAE ($M)": st.column_config.NumberColumn(format="%.1f"),
                "RMSE ($M)": st.column_config.NumberColumn(format="%.1f"),
                "MAPE (revenue ≥ $1M)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    st.subheader("Permutation importance (holdout, scoring=R²)")
    imp = pd.DataFrame(bundle.metadata["permutation_importance"])
    if imp.empty:
        st.info("No importance data available.")
    else:
        top = imp.head(15)
        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("importance_mean:Q", title="Drop in R² when shuffled"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color(
                    "importance_mean:Q",
                    scale=alt.Scale(scheme="viridis"),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("feature:N"),
                    alt.Tooltip("importance_mean:Q", format=".4f"),
                    alt.Tooltip("importance_std:Q", format=".4f"),
                ],
            )
            .properties(height=28 * len(top) + 40)
        )
        st.altair_chart(chart, width="stretch")

    st.subheader("Predicted vs actual (holdout)")
    scatter_df = bundle.holdout_predictions.copy()
    scatter_df.columns = ["Actual", "Predicted"]
    lo = float(min(scatter_df["Actual"].min(), scatter_df["Predicted"].min()))
    hi = float(max(scatter_df["Actual"].max(), scatter_df["Predicted"].max()))
    line_df = pd.DataFrame({"Actual": [lo, hi], "Predicted": [lo, hi]})
    points = (
        alt.Chart(scatter_df)
        .mark_circle(opacity=0.55)
        .encode(
            x=alt.X("Actual:Q", title=f"Actual {y_label}"),
            y=alt.Y("Predicted:Q", title=f"Predicted {y_label}"),
            tooltip=[
                alt.Tooltip("Actual:Q", format=y_format),
                alt.Tooltip("Predicted:Q", format=y_format),
            ],
        )
    )
    ref = (
        alt.Chart(line_df)
        .mark_line(strokeDash=[4, 4], color="#888888")
        .encode(x="Actual:Q", y="Predicted:Q")
    )
    st.altair_chart((points + ref).properties(height=380), width="stretch")

    if extra_note:
        st.caption(extra_note)


@st.cache_data(show_spinner=False)
def _top_values(col: str, _cache_key: str, limit: int = 400) -> list[str]:
    if col not in gold.columns:
        return []
    values = (
        gold.drop_nulls(col)
        .group_by(col)
        .agg(pl.len().alias("n"))
        .sort(["n", col], descending=[True, False])
        .head(limit)[col]
        .to_list()
    )
    return [str(v) for v in values if v]


_GOLD_CACHE_KEY = str(gold_path(settings))


tab_findings, tab_predict = st.tabs(["Findings", "Predict"])

with tab_findings:
    st.markdown(
        "### Revenue model\n"
        "Target `revenue_musd` (million USD) trained on `log1p(revenue)` so the "
        "heavy tail doesn't dominate the loss. Predictions are back-transformed "
        "to $M before reporting."
    )
    _render_findings(
        revenue,
        y_label="revenue ($M)",
        y_format=".1f",
        extra_note=(
            "HGB uses target-encoded director / studio / lead cast, multi-hot "
            "genres, cyclical release month, budget, and runtime. Budget and "
            "the target-encoded studio are consistently the strongest signals."
        ),
    )

    st.markdown("---")

    st.markdown(
        "### Rating model\n"
        "Target `vote_average` (0–10) with row weights `log1p(vote_count)` to "
        "down-weight noisy low-vote titles. Predictions are clipped to [0, 10]."
    )
    _render_findings(
        rating,
        y_label="rating",
        y_format=".2f",
        extra_note=(
            "Rating is noticeably harder than revenue: explanatory power is "
            "modest and a linear Ridge baseline can hold its own. Runtime, "
            "director, and lead cast dominate the remaining signal."
        ),
    )

    st.markdown("---")
    st.markdown(
        "### Takeaways\n"
        "- **Revenue** is largely a *budget + studio/cast prestige* story; cyclical "
        "release month adds a small seasonality bump that matches the heatmap in "
        "the analytics page.\n"
        "- **Rating** is dominated by crew/cast identity with a runtime nudge. "
        "Lists of target-encoded categoricals help, but headroom is limited by "
        "genuine noise in user ratings.\n"
        "- Target encoding + HGB comfortably beats a Ridge baseline on revenue; "
        "on rating the two are close, which is itself a useful finding about "
        "the signal-to-noise ratio."
    )


with tab_predict:
    st.markdown(
        "Fill in the hypothetical movie's attributes. Unseen director / studio / "
        "cast names fall back to the global prior learned by the target encoder."
    )

    genres_options = revenue.feature_spec.top_genres or sorted(
        {g for gs in gold["genres"].to_list() if gs for g in gs}
    )
    director_options = ["Unknown", *_top_values("director_name", _GOLD_CACHE_KEY)]
    studio_options = [
        "Unknown",
        *_top_values("lead_production_company", _GOLD_CACHE_KEY),
    ]
    cast_options = ["Unknown", *_top_values("lead_cast_name", _GOLD_CACHE_KEY)]

    col_l, col_r = st.columns(2)
    with col_l:
        budget = st.number_input(
            "Budget (million USD)",
            min_value=0.1,
            max_value=1000.0,
            value=60.0,
            step=5.0,
        )
        runtime = st.number_input(
            "Runtime (minutes)",
            min_value=40,
            max_value=300,
            value=110,
            step=5,
        )
        month = st.selectbox(
            "Release month",
            options=list(range(1, 13)),
            index=5,
            format_func=lambda m: MONTH_LABELS[m - 1],
        )
        genres_selected = st.multiselect(
            "Genres",
            options=genres_options,
            default=["Action", "Adventure"]
            if "Action" in genres_options and "Adventure" in genres_options
            else [],
            key="predict_genres",
        )
    with col_r:
        director = st.selectbox(
            "Director", options=director_options, index=0, key="predict_director"
        )
        studio = st.selectbox(
            "Lead production company",
            options=studio_options,
            index=0,
            key="predict_studio",
        )
        cast = st.selectbox(
            "Lead cast", options=cast_options, index=0, key="predict_cast"
        )

    predict_clicked = st.button("Predict", type="primary")
    if predict_clicked:

        def _predict(bundle: LoadedBundle) -> float:
            return predict_one(
                bundle,
                budget_musd=float(budget),
                runtime=float(runtime),
                release_month=int(month),
                genres=list(genres_selected),
                director_name=str(director),
                lead_production_company=str(studio),
                lead_cast_name=str(cast),
            )

        predicted_revenue = _predict(revenue)
        predicted_rating = _predict(rating)
        roi = (predicted_revenue - float(budget)) / float(budget) if budget > 0 else 0.0

        m_a, m_b, m_c = st.columns(3)
        m_a.metric("Predicted revenue", f"${predicted_revenue:,.1f}M")
        m_b.metric("Predicted rating", f"{predicted_rating:.2f} / 10")
        m_c.metric("Implied ROI", f"{roi:.2f}x")

        st.caption(
            "Revenue uses the HGB model trained on `log1p(revenue)` and is "
            "back-transformed with `expm1`. Rating is clipped to [0, 10]. ROI "
            "is computed from the predicted revenue and the budget you entered."
        )
