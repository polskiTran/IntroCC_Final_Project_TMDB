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
from src.ingest.embeddings import embed_single_overview  # noqa: E402
from src.ml.model_card import read_model_card  # noqa: E402
from src.ml.predict import LoadedBundle, bundle_exists, load_bundle, predict_one  # noqa: E402


st.set_page_config(page_title="ML prediction", layout="wide")
st.title("ML prediction")
st.caption(
    "Revenue uses a `CatBoostRegressor`; rating is an ensemble of CatBoost + "
    "`TabPFNRegressor` blended on out-of-fold predictions (falls back to "
    "CatBoost-only when TabPFN isn't available). Director / studio / top-5 "
    "cast / producer / collection are consumed as **native categorical "
    "features**. GPU-accelerated when available, CPU otherwise. Run "
    "`uv run python -m src.ml train` to refresh the artefacts."
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
    blend_weights = bundle.metadata.get("blend_weights")
    cb_hold = bundle.metadata.get("catboost_holdout_metrics")
    tp_hold = bundle.metadata.get("tabpfn_holdout_metrics")

    rows: list[dict[str, float | str]] = []
    if blend_weights:
        rows.append(
            {
                "Split": "Holdout (blend)",
                "R²": hold["r2"],
                "MAE": hold["mae"],
                "RMSE": hold["rmse"],
            }
        )
        if cb_hold:
            rows.append(
                {
                    "Split": "Holdout (CatBoost)",
                    "R²": cb_hold["r2"],
                    "MAE": cb_hold["mae"],
                    "RMSE": cb_hold["rmse"],
                }
            )
        if tp_hold:
            rows.append(
                {
                    "Split": "Holdout (TabPFN)",
                    "R²": tp_hold["r2"],
                    "MAE": tp_hold["mae"],
                    "RMSE": tp_hold["rmse"],
                }
            )
    else:
        rows.append(
            {
                "Split": "Holdout (CatBoost)",
                "R²": hold["r2"],
                "MAE": hold["mae"],
                "RMSE": hold["rmse"],
            }
        )
    rows.append(
        {
            "Split": "5-fold CV (CatBoost)",
            "R²": cv["r2"],
            "MAE": cv["mae"],
            "RMSE": cv["rmse"],
        }
    )
    rows.append(
        {
            "Split": "Holdout (Ridge baseline)",
            "R²": ridge["r2"],
            "MAE": ridge["mae"],
            "RMSE": ridge["rmse"],
        }
    )
    return pd.DataFrame(rows)


def _render_findings(
    bundle: LoadedBundle,
    *,
    y_label: str,
    y_format: str,
    extra_note: str | None = None,
) -> None:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Train rows", f"{bundle.metadata['n_train']:,}")
    col_b.metric("Test rows", f"{bundle.metadata['n_test']:,}")
    col_c.metric("Holdout R²", f"{bundle.metadata['holdout_metrics']['r2']:.3f}")
    task_type = bundle.metadata.get("task_type", "CPU")
    best_iter = bundle.metadata.get("best_iteration")
    blend_weights = bundle.metadata.get("blend_weights")
    if blend_weights:
        device_label = (
            f"CatBoost+TabPFN ({task_type}) · "
            f"{blend_weights['catboost']:.2f}/{blend_weights['tabpfn']:.2f}"
        )
    else:
        device_label = f"CatBoost ({task_type})"
        if best_iter:
            device_label += f" · {best_iter} trees"
    col_d.metric("Model", device_label)

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


tab_findings, tab_predict, tab_card = st.tabs(["Findings", "Predict", "Model card"])

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
            "CatBoost consumes director / studio / lead cast as **native "
            "categoricals** (ordered boosting handles leakage), alongside "
            "multi-hot genres, cyclical release month, budget, and runtime. "
            "Budget and studio identity are consistently the strongest signals."
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
            "Rating is noticeably harder than revenue. The v2 pipeline blends "
            "CatBoost (tree-based, strong on categoricals) with TabPFN "
            "(transformer pretrained for small tabular regression) using "
            "Ridge-fitted non-negative weights. When the blend is active, "
            "component R² rows above break out each model's standalone "
            "performance on the same holdout."
        ),
    )

    st.markdown("---")
    st.markdown(
        "### Takeaways\n"
        "- **Revenue** is largely a *budget + studio/cast prestige* story; cyclical "
        "release month adds a small seasonality bump that matches the heatmap in "
        "the analytics page.\n"
        "- **Rating** is dominated by crew/cast identity with a runtime nudge. "
        "Native CatBoost categoricals preserve rare directors / studios / cast "
        "instead of smoothing them toward the global prior, but headroom is "
        "limited by genuine noise in user ratings.\n"
        "- CatBoost with native categorical features comfortably beats the "
        "Ridge + target-encoding baseline on revenue; on rating the two are "
        "closer, which is itself a useful finding about the signal-to-noise "
        "ratio."
    )


with tab_predict:
    st.markdown(
        "Fill in the hypothetical movie's attributes. Unseen director / studio / "
        "cast / producer / collection names are treated as a fresh category by "
        "CatBoost and fall back to the model's prior for unknowns (same as the "
        "training-time ``\"Unknown\"`` / ``\"Standalone\"`` sentinels). The "
        "overview text is embedded with the same MiniLM + PCA pipeline used in "
        "the Gold build."
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
    cast2_options = ["Unknown", *_top_values("cast_2_name", _GOLD_CACHE_KEY)]
    cast3_options = ["Unknown", *_top_values("cast_3_name", _GOLD_CACHE_KEY)]
    cast4_options = ["Unknown", *_top_values("cast_4_name", _GOLD_CACHE_KEY)]
    cast5_options = ["Unknown", *_top_values("cast_5_name", _GOLD_CACHE_KEY)]
    producer_options = [
        "Unknown",
        *_top_values("lead_producer_name", _GOLD_CACHE_KEY),
    ]
    collection_options = [
        "Standalone",
        *_top_values("collection_name", _GOLD_CACHE_KEY),
    ]

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
        release_year = st.number_input(
            "Release year",
            min_value=settings.start_year,
            max_value=2035,
            value=2024,
            step=1,
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
        collection = st.selectbox(
            "Franchise / collection",
            options=collection_options,
            index=0,
            key="predict_collection",
        )
        has_tagline = st.checkbox("Has a marketing tagline", value=True)
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
        producer = st.selectbox(
            "Lead producer",
            options=producer_options,
            index=0,
            key="predict_producer",
        )
        cast = st.selectbox(
            "Lead cast", options=cast_options, index=0, key="predict_cast"
        )
        cast_2 = st.selectbox(
            "Cast #2", options=cast2_options, index=0, key="predict_cast2"
        )
        cast_3 = st.selectbox(
            "Cast #3", options=cast3_options, index=0, key="predict_cast3"
        )
        cast_4 = st.selectbox(
            "Cast #4", options=cast4_options, index=0, key="predict_cast4"
        )
        cast_5 = st.selectbox(
            "Cast #5", options=cast5_options, index=0, key="predict_cast5"
        )

    overview_text = st.text_area(
        "Plot overview (optional)",
        value="",
        placeholder=(
            "A one-paragraph synopsis. Embedded on the fly with MiniLM + the "
            "persisted PCA basis."
        ),
        help=(
            "Leave blank to use a zero-vector. Requires `sentence-transformers` "
            "to be importable and the PCA file at `data/gold/overview_pca.joblib`."
        ),
    )

    predict_clicked = st.button("Predict", type="primary")
    if predict_clicked:
        overview_vec = None
        if overview_text.strip():
            overview_vec = embed_single_overview(overview_text, settings=settings)

        def _predict(bundle: LoadedBundle) -> float:
            return predict_one(
                bundle,
                budget_musd=float(budget),
                runtime=float(runtime),
                release_year=int(release_year),
                release_month=int(month),
                genres=list(genres_selected),
                director_name=str(director),
                lead_production_company=str(studio),
                lead_cast_name=str(cast),
                cast_2_name=str(cast_2),
                cast_3_name=str(cast_3),
                cast_4_name=str(cast_4),
                cast_5_name=str(cast_5),
                lead_producer_name=str(producer),
                collection_name=str(collection),
                has_tagline=bool(has_tagline),
                overview_embedding=overview_vec,
            )

        predicted_revenue = _predict(revenue)
        predicted_rating = _predict(rating)
        roi = (predicted_revenue - float(budget)) / float(budget) if budget > 0 else 0.0

        m_a, m_b, m_c = st.columns(3)
        m_a.metric("Predicted revenue", f"${predicted_revenue:,.1f}M")
        m_b.metric("Predicted rating", f"{predicted_rating:.2f} / 10")
        m_c.metric("Implied ROI", f"{roi:.2f}x")

        st.caption(
            "Revenue uses the CatBoost model trained on `log1p(revenue)` and "
            "is back-transformed with `expm1`. Rating uses the CatBoost + "
            "TabPFN blend when available (falls back to CatBoost-only) and is "
            "clipped to [0, 10]. ROI is computed from the predicted revenue "
            "and the budget you entered."
        )


with tab_card:
    card = read_model_card(settings)
    if card is None:
        st.info(
            "No model card yet. Run `uv run python -m src.ml train` to "
            "generate `data/ml/model_card.md`."
        )
    else:
        st.markdown(card)
        st.caption(
            "Auto-generated from the trained bundles on every training run. "
            "Stored at `data/ml/model_card.md`."
        )
