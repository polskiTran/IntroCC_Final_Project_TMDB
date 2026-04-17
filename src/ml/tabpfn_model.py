"""TabPFN wrapper used by the rating blend.

TabPFN v2 is a transformer pre-trained for small tabular regression (<=10k
rows). It's an ideal ensemble partner for CatBoost on the ~6k-row Gold
movie frame and runs fast on a consumer GPU.

This module isolates the TabPFN dependency so the main training pipeline
keeps working if ``tabpfn`` / ``torch`` can't be imported or the pretrained
weights can't be downloaded. Set ``TMDB_ML_TABPFN=0`` to force the blend off
(used by CI and offline environments).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from src.ml.features import FeatureSpec, categorical_feature_indices

logger = logging.getLogger(__name__)

ENV_TOGGLE = "TMDB_ML_TABPFN"
ENV_BATCH = "TMDB_ML_TABPFN_BATCH"
ENV_DEVICE = "TMDB_ML_TABPFN_DEVICE"
DEFAULT_BATCH_SIZE = 256


def _batch_size() -> int:
    raw = os.environ.get(ENV_BATCH, "").strip()
    if not raw:
        return DEFAULT_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BATCH_SIZE
    return max(1, value)


def _env_enabled() -> bool:
    raw = os.environ.get(ENV_TOGGLE, "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def available() -> bool:
    """Return True if TabPFN can be used in this process."""
    if not _env_enabled():
        return False
    try:
        import tabpfn  # noqa: PLC0415, F401
    except Exception as exc:  # noqa: BLE001
        logger.info("TabPFN unavailable (%s); blend disabled", exc)
        return False
    return True


def version() -> str | None:
    try:
        import tabpfn  # noqa: PLC0415

        return getattr(tabpfn, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def _device_for_tabpfn() -> str:
    """Pick a device string for TabPFN.

    - ``TMDB_ML_TABPFN_DEVICE`` is a hard override (``cpu`` / ``cuda`` / ``mps``).
    - ``TMDB_ML_DEVICE=CPU`` forces CPU.
    - ``TMDB_ML_DEVICE=GPU`` picks CUDA when available, else CPU.
    - Auto mode prefers CUDA, then CPU. **MPS is skipped by default** because
      Apple's unified memory is shared with the host and TabPFN's full-train
      context regularly OOMs on consumer Macs (problem #2 in TabPFN's own
      error message — batching test samples doesn't help). Set
      ``TMDB_ML_TABPFN_DEVICE=mps`` to force it.
    """
    explicit = os.environ.get(ENV_DEVICE, "").strip().lower()
    if explicit in {"cpu", "cuda", "mps"}:
        return explicit

    override = os.environ.get("TMDB_ML_DEVICE", "").strip().upper()
    if override == "CPU":
        return "cpu"

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass

    if override == "GPU":
        logger.info("TMDB_ML_DEVICE=GPU but no CUDA; using CPU for TabPFN")
    return "cpu"


def _empty_cache() -> None:
    """Best-effort free of torch caches between folds."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and getattr(mps, "is_available", lambda: False)():
            empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
            if empty is not None:
                empty()
    except Exception:  # noqa: BLE001
        pass


def _to_numeric_frame(X: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """TabPFN wants numeric columns; map categoricals to integer codes.

    CatBoost handles strings natively, but TabPFN's sklearn API expects
    numeric features with ``categorical_features_indices`` listing the
    integer-coded columns. We use pandas ``Categorical`` codes fit on the
    column itself so unseen categories during inference get ``-1``.
    """
    X_num = X.copy()
    from src.ml.features import CATEGORICAL_COLS  # noqa: PLC0415

    for col in CATEGORICAL_COLS:
        if col in X_num.columns:
            cats = pd.Categorical(X_num[col].astype("string").fillna("Unknown"))
            X_num[col] = cats.codes.astype(np.int64)
    # Order must match spec.feature_columns so categorical indices line up.
    return X_num[spec.feature_columns]


class TabPFNComponent:
    """Fit/predict wrapper. Stores the categorical level maps so inference
    uses the same integer codes as training."""

    def __init__(self, spec: FeatureSpec, *, random_state: int = 42) -> None:
        self.spec = spec
        self.random_state = random_state
        self._model: Any | None = None
        self._cat_levels: dict[str, dict[str, int]] = {}

    def _encode(self, X: pd.DataFrame, *, train: bool) -> pd.DataFrame:
        from src.ml.features import CATEGORICAL_COLS  # noqa: PLC0415

        X_out = X.copy()
        for col in CATEGORICAL_COLS:
            if col not in X_out.columns:
                continue
            values = X_out[col].astype("string").fillna("Unknown").tolist()
            if train:
                levels: dict[str, int] = {}
                codes = []
                for v in values:
                    if v not in levels:
                        levels[v] = len(levels)
                    codes.append(levels[v])
                self._cat_levels[col] = levels
                X_out[col] = np.asarray(codes, dtype=np.int64)
            else:
                levels = self._cat_levels.get(col, {})
                X_out[col] = np.asarray(
                    [levels.get(v, len(levels)) for v in values], dtype=np.int64
                )
        return X_out[self.spec.feature_columns]

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,  # noqa: ARG002 - TabPFN ignores
    ) -> "TabPFNComponent":
        from tabpfn import TabPFNRegressor  # noqa: PLC0415

        _empty_cache()
        X_enc = self._encode(X, train=True)
        cat_idx = categorical_feature_indices(self.spec)
        device = _device_for_tabpfn()
        logger.info(
            "fitting TabPFN on %d rows × %d features (device=%s)",
            X_enc.shape[0],
            X_enc.shape[1],
            device,
        )
        self._model = TabPFNRegressor(
            categorical_features_indices=cat_idx,
            device=device,
            random_state=self.random_state,
            ignore_pretraining_limits=True,
            n_estimators=4,
        )
        self._model.fit(X_enc.to_numpy(), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Batched predict to keep consumer-GPU memory bounded.

        TabPFN holds the full training set in device memory, then reruns the
        transformer over each test row; predicting the whole holdout at once
        can blow the MPS / CUDA budget (observed ~1k rows -> 17 GiB). Chunking
        the test set keeps the peak bounded and is numerically identical.
        Batch size tunable via ``TMDB_ML_TABPFN_BATCH``.
        """
        if self._model is None:
            raise RuntimeError("TabPFNComponent.predict called before fit")
        X_enc = self._encode(X, train=False).to_numpy()
        batch = _batch_size()
        if len(X_enc) <= batch:
            return np.asarray(self._model.predict(X_enc), dtype=np.float64)
        parts: list[np.ndarray] = []
        for start in range(0, len(X_enc), batch):
            end = start + batch
            parts.append(
                np.asarray(self._model.predict(X_enc[start:end]), dtype=np.float64)
            )
            _empty_cache()
        return np.concatenate(parts, axis=0)
