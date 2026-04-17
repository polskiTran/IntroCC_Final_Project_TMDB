"""Overview text embedding used by the Gold builder.

Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim) compressed to
`OVERVIEW_EMBED_DIM` via a deterministic PCA, so the Gold parquet stays small
(~16 floats per row) and the ML layer can consume it without re-running the
encoder.

Offline/fallback behaviour: if torch / sentence-transformers fail to import or
the weights can't be loaded, `embed_overviews` returns a zero matrix and logs
a warning. The pipeline stays idempotent; downstream models just see a
constant column.
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

OVERVIEW_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OVERVIEW_RAW_DIM = 384
OVERVIEW_EMBED_DIM = 16
RANDOM_STATE = 42
PCA_FILENAME = "overview_pca.joblib"
ENV_TOGGLE = "TMDB_ML_EMBEDDINGS"


def _env_enabled() -> bool:
    raw = os.environ.get(ENV_TOGGLE, "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def pca_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.gold_dir / PCA_FILENAME


def _resolve_device() -> str:
    override = os.environ.get("TMDB_ML_DEVICE", "").strip().upper()
    if override == "GPU":
        return "cuda"
    if override == "CPU":
        return "cpu"
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _load_encoder():  # type: ignore[no-untyped-def]
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    device = _resolve_device()
    logger.info("loading %s on %s", OVERVIEW_MODEL_NAME, device)
    return SentenceTransformer(OVERVIEW_MODEL_NAME, device=device)


def _encode(texts: list[str]) -> np.ndarray:
    encoder = _load_encoder()
    return np.asarray(
        encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def _fit_pca(embeddings: np.ndarray, n_components: int) -> Any:
    """Fit a PCA with deterministic axis signs."""
    from sklearn.decomposition import PCA  # noqa: PLC0415

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(embeddings)
    components = pca.components_
    flip = np.ones(components.shape[0], dtype=np.float64)
    for i in range(components.shape[0]):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            flip[i] = -1.0
    pca.components_ = components * flip[:, None]
    return pca


def _hash_fallback(texts: list[str]) -> np.ndarray:
    """Deterministic zero-ish fallback when encoder is unavailable.

    Returns a low-variance signature (small hash-derived floats) so downstream
    variance-based checks don't crash but the model sees essentially noise.
    """
    out = np.zeros((len(texts), OVERVIEW_EMBED_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        if not t:
            continue
        digest = hashlib.sha256(t.encode("utf-8")).digest()
        for j in range(OVERVIEW_EMBED_DIM):
            out[i, j] = ((digest[j % len(digest)] / 255.0) - 0.5) * 1e-3
    return out


def embed_overviews(
    overviews: list[str | None],
    settings: Settings | None = None,
) -> np.ndarray:
    """Return a ``(N, OVERVIEW_EMBED_DIM)`` float32 matrix and persist the PCA.

    The PCA basis is saved to ``settings.gold_dir / PCA_FILENAME`` so the
    Streamlit predict tab can project ad-hoc user-entered overviews with the
    same basis.
    """
    texts = [t if isinstance(t, str) and t.strip() else "" for t in overviews]
    if not texts:
        return np.zeros((0, OVERVIEW_EMBED_DIM), dtype=np.float32)
    if not _env_enabled():
        logger.info("overview embeddings disabled via %s", ENV_TOGGLE)
        return _hash_fallback(texts)
    try:
        raw = _encode(texts)
    except Exception as exc:  # noqa: BLE001 - offline/missing weights path
        logger.warning(
            "overview encoder unavailable (%s); falling back to zero-ish vectors",
            exc,
        )
        return _hash_fallback(texts)

    n = raw.shape[0]
    if n < OVERVIEW_EMBED_DIM + 1:
        reduced = np.zeros((n, OVERVIEW_EMBED_DIM), dtype=np.float32)
        k = min(n, OVERVIEW_EMBED_DIM, raw.shape[1])
        reduced[:, :k] = raw[:, :k]
        return reduced

    pca = _fit_pca(raw, OVERVIEW_EMBED_DIM)
    reduced = pca.transform(raw).astype(np.float32)

    try:
        target = pca_path(settings)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, target)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not persist overview PCA (%s); continuing", exc)

    return reduced


def _load_pca(settings: Settings | None = None) -> Any | None:
    target = pca_path(settings)
    if not target.is_file():
        return None
    try:
        return joblib.load(target)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load overview PCA (%s); using zeros", exc)
        return None


def embed_single_overview(
    overview: str | None,
    settings: Settings | None = None,
) -> np.ndarray:
    """Return a single ``(OVERVIEW_EMBED_DIM,)`` vector using the persisted PCA."""
    if not overview or not overview.strip():
        return np.zeros(OVERVIEW_EMBED_DIM, dtype=np.float32)
    if not _env_enabled():
        return np.zeros(OVERVIEW_EMBED_DIM, dtype=np.float32)
    pca = _load_pca(settings)
    try:
        raw = _encode([overview])
    except Exception as exc:  # noqa: BLE001
        logger.warning("single overview encode failed (%s); returning zeros", exc)
        return np.zeros(OVERVIEW_EMBED_DIM, dtype=np.float32)
    if pca is None:
        out = np.zeros(OVERVIEW_EMBED_DIM, dtype=np.float32)
        k = min(OVERVIEW_EMBED_DIM, raw.shape[1])
        out[:k] = raw[0, :k]
        return out
    return pca.transform(raw).astype(np.float32)[0]
