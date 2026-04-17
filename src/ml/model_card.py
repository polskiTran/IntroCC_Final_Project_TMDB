"""Generate a human-readable model card after each training run.

Called from the end of :func:`src.ml.train.train`. Produces a single
``model_card.md`` under ``settings.ml_dir`` that the Streamlit ML page renders
verbatim. The card is fully rebuilt on every training run (idempotent).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import Settings, get_settings

if TYPE_CHECKING:
    from src.ml.train import ModelBundle


MODEL_CARD_FILENAME = "model_card.md"


def model_card_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.ml_dir / MODEL_CARD_FILENAME


def _fmt(x: float | int | str | None, spec: str = ".3f") -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    try:
        if x != x:  # NaN
            return "n/a"
    except TypeError:
        return str(x)
    return format(x, spec)


def _metrics_table(bundle: "ModelBundle") -> str:
    hold = bundle.holdout_metrics.as_dict()
    cv = bundle.cv_metrics.as_dict()
    ridge = bundle.baseline_holdout_metrics.as_dict()
    extra = bundle.extra or {}
    rows: list[tuple[str, dict[str, float]]] = []
    cb_hold = extra.get("catboost_holdout_metrics")
    tp_hold = extra.get("tabpfn_holdout_metrics")
    is_blend = bool(extra.get("blend_weights"))
    if is_blend:
        rows.append(("Holdout (blend)", hold))
        if cb_hold:
            rows.append(("Holdout (CatBoost)", cb_hold))
        if tp_hold:
            rows.append(("Holdout (TabPFN)", tp_hold))
    else:
        rows.append(("Holdout (CatBoost)", hold))
    rows.append(("5-fold CV (CatBoost)", cv))
    rows.append(("Holdout (Ridge + TargetEncoder baseline)", ridge))
    lines = ["| Split | R² | MAE | RMSE |", "| --- | ---: | ---: | ---: |"]
    for label, m in rows:
        lines.append(
            f"| {label} | {_fmt(m['r2'])} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} |"
        )
    return "\n".join(lines)


def _revenue_space_block(bundle: "ModelBundle") -> str:
    rev = bundle.revenue_space_metrics
    if not rev:
        return ""
    return (
        "\n**Revenue ($M, back-transformed):** "
        f"MAE = {_fmt(rev.get('mae_musd'), '.2f')} · "
        f"RMSE = {_fmt(rev.get('rmse_musd'), '.2f')} · "
        f"MAPE (revenue ≥ $1M) = {_fmt(rev.get('mape_over_1M'), '.3f')}\n"
    )


def _importance_block(bundle: "ModelBundle", top_k: int = 10) -> str:
    rows = bundle.permutation_importance[:top_k]
    if not rows:
        return "_No permutation importance data._"
    lines = [
        "| Feature | Δ R² (mean) | Δ R² (std) |",
        "| --- | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['feature']}` | {_fmt(r['importance_mean'], '.4f')} | "
            f"{_fmt(r['importance_std'], '.4f')} |"
        )
    return "\n".join(lines)


def _bundle_section(bundle: "ModelBundle") -> str:
    extra = bundle.extra or {}
    task_type = extra.get("task_type", "CPU")
    best_iter = extra.get("best_iteration", "?")
    cb_ver = extra.get("catboost_version", "?")
    blend_weights = extra.get("blend_weights")
    tp_ver = extra.get("tabpfn_version")
    if blend_weights:
        engine_line = (
            f"- **Engine:** CatBoost {cb_ver} + TabPFN {tp_ver or '?'} · "
            f"`task_type={task_type}` · best iteration = {best_iter}\n"
            f"- **Blend weights:** catboost = {blend_weights['catboost']:.3f}, "
            f"tabpfn = {blend_weights['tabpfn']:.3f}\n"
        )
    else:
        engine_line = (
            f"- **Engine:** CatBoost {cb_ver} · `task_type={task_type}` · "
            f"best iteration = {best_iter}\n"
        )
    feature_list = (
        "- **Feature set:** numeric (budget, runtime, release_year, counts, "
        "has_tagline, has_collection) + cyclical month + multi-hot genres "
        "(top 25) + multi-hot studios (top 30) + 16-dim MiniLM overview "
        "embedding + native categoricals (director, studio, top-5 cast, "
        "producer, collection).\n"
    )
    return (
        f"## {bundle.target_label}\n\n"
        f"- **Target:** `{bundle.target}`\n"
        f"- **Training transform:** `{bundle.target_transform}`\n"
        f"- **Rows (total / train / test):** "
        f"{bundle.n_rows_total:,} / {bundle.n_train:,} / {bundle.n_test:,}\n"
        + engine_line
        + "- **Categorical features (native):** "
        + ", ".join(f"`{c}`" for c in extra.get("cat_features", []))
        + "\n"
        + feature_list
        + "\n"
        + "### Metrics\n\n"
        + f"{_metrics_table(bundle)}\n"
        + f"{_revenue_space_block(bundle)}\n"
        + "### Top permutation importances (holdout, scoring = R²)\n\n"
        + f"{_importance_block(bundle)}\n"
    )


def render_model_card(bundles: dict[str, "ModelBundle"]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    order = ("revenue", "rating")
    ordered = [bundles[k] for k in order if k in bundles]
    ordered += [b for k, b in bundles.items() if k not in order]

    header = (
        "# TMDB model card\n\n"
        f"_Generated {generated_at}._ "
        "Rebuilt on every `uv run python -m src.ml train` invocation.\n\n"
        "Two models trained on the Gold table: **revenue** uses "
        "`CatBoostRegressor` only; **rating** is an ensemble of "
        "`CatBoostRegressor` and `TabPFNRegressor` (a consumer-GPU transformer "
        "pretrained for small tabular regression), blended with a non-negative "
        "Ridge meta-learner fit on out-of-fold predictions. High-cardinality "
        "categoricals (director, studio, top-5 cast, lead producer, collection) "
        "are consumed as **native categorical features** — no target encoding, "
        "no smoothing. A Ridge + `TargetEncoder` baseline is reported for each "
        "model as a linear sanity-check comparator.\n\n"
    )
    sections = "\n---\n\n".join(_bundle_section(b) for b in ordered)
    return header + sections + "\n"


def write_model_card(
    bundles: dict[str, "ModelBundle"],
    settings: Settings | None = None,
) -> Path:
    settings = settings or get_settings()
    settings.ml_dir.mkdir(parents=True, exist_ok=True)
    path = model_card_path(settings)
    path.write_text(render_model_card(bundles))
    return path


def read_model_card(settings: Settings | None = None) -> str | None:
    path = model_card_path(settings)
    if not path.is_file():
        return None
    return path.read_text()
