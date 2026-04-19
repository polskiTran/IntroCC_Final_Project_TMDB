"""Generate ``model_card.md`` (see ``settings.model_card_path``) after ML training runs.

Collects best-effort system/device info (CPU, RAM, optional NVIDIA GPU via
``nvidia-smi``) and renders evaluation metadata from ``metrics_summary``.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import Settings
from src.ml.paths import (
    metrics_json_path,
    rating_bundle_path,
    revenue_bundle_path,
)


def collect_system_info() -> dict[str, Any]:
    """Runtime environment facts for reproducibility and debugging."""
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "",
        "cpu_count_logical": os.cpu_count(),
    }
    mem_gib = _total_memory_gib()
    if mem_gib is not None:
        info["memory_total_gib"] = round(mem_gib, 2)
    info["packages"] = _package_versions()
    info["gpu"] = _nvidia_gpu_summary()
    return info


def _total_memory_gib() -> float | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=3
            )
            return int(out.strip()) / (1024**3)
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass
    return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for import_name, label in (
        ("numpy", "numpy"),
        ("polars", "polars"),
        ("sklearn", "scikit-learn"),
        ("joblib", "joblib"),
    ):
        try:
            mod = __import__(import_name)
            versions[label] = getattr(mod, "__version__", "?")
        except ImportError:
            versions[label] = "n/a"
    return versions


def _nvidia_gpu_summary() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "No NVIDIA GPU detected (nvidia-smi not available or no devices)."


def format_system_info_for_log(info: dict[str, Any]) -> str:
    """Multi-line string for logging (no tqdm interference)."""
    lines = [
        f"  Python: {info.get('python_version', '?')} ({info.get('platform', '')})",
        f"  CPU (logical): {info.get('cpu_count_logical', '?')}",
    ]
    if "memory_total_gib" in info:
        lines.append(f"  RAM (approx): {info['memory_total_gib']} GiB")
    pkgs = info.get("packages", {})
    if pkgs:
        lines.append(
            "  Packages: " + ", ".join(f"{k}={v}" for k, v in sorted(pkgs.items()))
        )
    gpu = info.get("gpu", "")
    lines.append("  GPU:")
    for gline in str(gpu).splitlines() or ["(no detail)"]:
        lines.append(f"    {gline}")
    return "\n".join(lines)


def _md_escape_cell(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _fmt_metric(x: Any, decimals: int = 4) -> str:
    if x is None:
        return "nan"
    try:
        return f"{float(x):.{decimals}f}"
    except (TypeError, ValueError):
        return str(x)


def render_model_card_markdown(
    *,
    metrics_summary: dict[str, Any],
    system_info: dict[str, Any],
    settings: Settings,
    gold_source: str,
    run_started_at_utc: str,
    reproducibility: dict[str, Any],
    top_importance_rows: int = 12,
) -> str:
    """Build deterministic markdown for ``model_card.md``."""
    lines: list[str] = [
        "# Model card — TMDB revenue & rating regressors",
        "",
        "This file is **overwritten** on every `uv run python -m src.ml train`.",
        "",
        "## Overview",
        "",
        "Two **HistGradientBoostingRegressor** pipelines (revenue in M USD, user rating 0–10) "
        "with the same tabular features as described in the project README. "
        f"See `{metrics_json_path(settings).as_posix()}` for machine-readable metrics.",
        "",
        "## Run",
        "",
        f"- **UTC time**: `{run_started_at_utc}`",
        f"- **Gold data source**: `{gold_source}`",
        "",
        "## Artifacts",
        "",
        f"- **Models root**: `{settings.ml_dir.resolve()}`",
        f"- **Metrics JSON**: `{metrics_json_path(settings).resolve()}`",
        f"- **Revenue bundle**: `{revenue_bundle_path(settings).resolve()}`",
        f"- **Rating bundle**: `{rating_bundle_path(settings).resolve()}`",
        "",
        "## Reproducibility",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
    ]
    for key in sorted(reproducibility.keys()):
        lines.append(
            f"| {_md_escape_cell(key)} | `{_md_escape_cell(str(reproducibility[key]))}` |"
        )
    lines.extend(["", "## Training data (per target)", ""])

    for target_key in ("revenue", "rating"):
        block = metrics_summary.get(target_key, {})
        if not block:
            continue
        lines.append(f"### {target_key}")
        lines.append("")
        lines.append(
            f"- **Rows (after filters)**: {block.get('n_rows_total', '?')} "
            f"(train {block.get('n_train', '?')}, holdout {block.get('n_test', '?')})"
        )
        lines.append(
            f"- **Target**: `{block.get('target', '')}` — {block.get('target_label', '')}"
        )
        lines.append(f"- **Target transform**: `{block.get('target_transform', '')}`")
        tg = block.get("top_genres")
        if isinstance(tg, list):
            lines.append(f"- **Top genres (multi-hot)**: {', '.join(tg)}")
        feats = block.get("feature_columns")
        if isinstance(feats, list):
            lines.append(f"- **Feature columns ({len(feats)})**: `{', '.join(feats)}`")
        lines.append("")

    lines.extend(["## Evaluation", ""])

    for target_key in ("revenue", "rating"):
        block = metrics_summary.get(target_key, {})
        if not block:
            continue
        lines.append(f"### {target_key}")
        lines.append("")
        ho = block.get("holdout_metrics") or {}
        cv = block.get("cv_metrics") or {}
        base = block.get("baseline_ridge_holdout_metrics") or {}
        lines.append("| Split | R² | MAE | RMSE |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(
            "| Holdout (HGB) | "
            f"{_fmt_metric(ho.get('r2'))} | {_fmt_metric(ho.get('mae'))} | "
            f"{_fmt_metric(ho.get('rmse'))} |"
        )
        lines.append(
            "| 5-fold CV (HGB) | "
            f"{_fmt_metric(cv.get('r2'))} | {_fmt_metric(cv.get('mae'))} | "
            f"{_fmt_metric(cv.get('rmse'))} |"
        )
        lines.append(
            "| Holdout Ridge baseline | "
            f"{_fmt_metric(base.get('r2'))} | {_fmt_metric(base.get('mae'))} | "
            f"{_fmt_metric(base.get('rmse'))} |"
        )
        rev_sp = block.get("revenue_space_metrics")
        if isinstance(rev_sp, dict):
            lines.append("")
            lines.append("**Revenue (original units, M USD)**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            for rk, rv in sorted(rev_sp.items()):
                if isinstance(rv, float) and rv != rv:  # NaN
                    disp = "nan"
                else:
                    disp = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
                lines.append(f"| {_md_escape_cell(rk)} | {disp} |")
        perm = block.get("permutation_importance")
        if isinstance(perm, list) and perm:
            lines.append("")
            lines.append(
                f"**Permutation importance (top {min(top_importance_rows, len(perm))})**"
            )
            lines.append("")
            lines.append("| Feature | Importance (mean) | Std |")
            lines.append("| --- | --- | --- |")
            for row in perm[:top_importance_rows]:
                lines.append(
                    "| "
                    f"{_md_escape_cell(str(row.get('feature', '')))} | "
                    f"{float(row.get('importance_mean', 0.0)):.6f} | "
                    f"{float(row.get('importance_std', 0.0)):.6f} |"
                )
        lines.append("")

    lines.extend(["## System / device", ""])
    lines.append(
        f"- **Platform**: `{_md_escape_cell(str(system_info.get('platform', '')))}`"
    )
    lines.append(
        f"- **Machine**: `{_md_escape_cell(str(system_info.get('machine', '')))}`"
    )
    if system_info.get("processor"):
        lines.append(
            f"- **Processor**: `{_md_escape_cell(str(system_info.get('processor')))}`"
        )
    lines.append(
        f"- **Python**: `{_md_escape_cell(str(system_info.get('python_version', '')))}`"
    )
    cc = system_info.get("cpu_count_logical")
    if cc is not None:
        lines.append(f"- **CPU cores (logical)**: {cc}")
    if "memory_total_gib" in system_info:
        lines.append(f"- **Approx. RAM**: {system_info['memory_total_gib']} GiB")

    pkgs = system_info.get("packages", {})
    if pkgs:
        lines.append(
            "- **Libraries**: "
            + ", ".join(f"`{k}={v}`" for k, v in sorted(pkgs.items()))
        )

    lines.extend(["", "**GPU / accelerator**", ""])
    gpu_raw = str(system_info.get("gpu", ""))
    lines.append("```text")
    lines.extend(gpu_raw.splitlines() or ["(none)"])
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def write_model_card(path: Path, content: str) -> None:
    """Atomically replace ``path`` with ``content``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".model_card_", suffix=".md", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        Path(tmp_name).replace(path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
