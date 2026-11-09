"""Emotion-vector space: loading, recomputation from caches, and projection.

Follows gemotions' `extract_vectors.py`:

    raw_e      = mean(stories of e) - mean(all stories)
    noise      = top principal components of neutral-dialogue activations
                 explaining DENOISE_VARIANCE of their variance
    v_e        = raw_e - noise @ (noise.T @ raw_e)

Scores for an activation x are `(x - ref_mean) . v_e / |v_e|`. Because every
v_e is orthogonal to the noise basis, projecting the noise out of x would not
change the score, so it is not done.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DENOISE_VARIANCE = 0.5  # gemotions DENOISING_VARIANCE_THRESHOLD
RESULTS_SUBDIR = "results/gemma4-31b"
NEUTRAL_KEY = "_neutral"
# (positive, negative) emotions that fix the sign of PC1 so that valence
# points the same way at every layer.
PC1_ANCHOR = ("happy", "sad")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def vectors_path(root: str | Path, layer: int) -> Path:
    return Path(root) / RESULTS_SUBDIR / f"emotion_vectors_layer{layer}.npz"


def cache_dir(root: str | Path, layer: int) -> Path:
    return Path(root) / RESULTS_SUBDIR / f"_raw_cache_layer{layer}"


def load_released_vectors(root: str | Path, layer: int) -> dict[str, np.ndarray]:
    """Released (centered, denoised) vectors, keyed by emotion name."""
    with np.load(vectors_path(root, layer), allow_pickle=False) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def cache_emotions(root: str | Path, layer: int) -> list[str]:
    d = cache_dir(root, layer)
    return sorted(p.stem for p in d.glob("*.npy") if p.stem != NEUTRAL_KEY)


def load_cache_array(root: str | Path, layer: int, name: str) -> np.ndarray:
    """Per-story mean activations, [n_stories, hidden], memory-mapped."""
    return np.load(cache_dir(root, layer) / f"{name}.npy", mmap_mode="r", allow_pickle=False)


# --------------------------------------------------------------------------
# Recomputation from the per-story caches
# --------------------------------------------------------------------------

@dataclass
class Recomputed:
    layer: int
    emotions: list[str]
    global_mean: np.ndarray      # [D] mean over all stories of all emotions
    emotion_means: np.ndarray    # [E, D]
    noise_basis: np.ndarray      # [D, k] orthonormal columns
    noise_variance: float        # variance fraction the k components explain
    vectors: np.ndarray          # [E, D] centered and denoised
    counts: np.ndarray           # [E] stories per emotion

    def as_dict(self) -> dict[str, np.ndarray]:
        return {e: self.vectors[i] for i, e in enumerate(self.emotions)}


def noise_basis_from_neutral(neutral: np.ndarray, variance: float = DENOISE_VARIANCE):
    """Top right-singular vectors of centered neutral activations.

    Returns (basis [D, k], explained variance fraction).
    """
    x = np.asarray(neutral, dtype=np.float64)
    x = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    cum = np.cumsum(s**2) / np.sum(s**2)
    k = int(np.searchsorted(cum, variance) + 1)
    return vt[:k].T, float(cum[k - 1])


def recompute_from_cache(root: str | Path, layer: int) -> Recomputed:
    emotions = cache_emotions(root, layer)
    if not emotions:
        raise FileNotFoundError(f"no activation cache for layer {layer} under {root}")

    means, counts = [], []
    total = None
    for e in emotions:
        a = load_cache_array(root, layer, e)
        s = a.sum(axis=0, dtype=np.float64)
        total = s if total is None else total + s
        means.append(s / len(a))
        counts.append(len(a))
    counts_arr = np.array(counts)
    # gemotions averages over the pooled list of stories, so emotions with
    # more stories weigh more. Reproduce that, not a mean of means.
    global_mean = total / counts_arr.sum()
    emotion_means = np.stack(means)

    neutral = load_cache_array(root, layer, NEUTRAL_KEY)
    basis, var = noise_basis_from_neutral(neutral)

    raw = emotion_means - global_mean
    vectors = raw - (raw @ basis) @ basis.T
    return Recomputed(layer, emotions, global_mean, emotion_means, basis, var, vectors, counts_arr)


# --------------------------------------------------------------------------
# The space used for projection
# --------------------------------------------------------------------------

def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-12)


@dataclass
class Reference:
    """Reference statistics for one (prompt, layer, readout position).

    `mean` replaces gemotions' story mean, which is the wrong origin for
    image inputs. The std fields put emotions on a comparable scale.
    """
    mean: np.ndarray        # [D]
    score_std: np.ndarray   # [E]
    pc_std: np.ndarray      # [K]


@dataclass
class EVSpace:
    layer: int
    emotions: list[str]
    vectors: np.ndarray       # [E, D]
    unit: np.ndarray          # [E, D] rows normalized
    components: np.ndarray    # [K, D] fixed PCA basis, sign-anchored
    explained: np.ndarray     # [K]

    @classmethod
    def from_vectors(cls, layer: int, vectors: dict[str, np.ndarray], n_components: int = 5) -> "EVSpace":
        emotions = sorted(vectors)
        m = np.stack([np.asarray(vectors[e], dtype=np.float64) for e in emotions])
        unit = m / np.linalg.norm(m, axis=1, keepdims=True)

        # Same PCA as gemotions: on the raw (not unit-normalized) vectors.
        centered = m - m.mean(axis=0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(n_components, len(s))
        comps = vt[:k].copy()
        coords = centered @ comps.T
        pos, neg = PC1_ANCHOR
        if pos in emotions and neg in emotions:
            if coords[emotions.index(pos), 0] < coords[emotions.index(neg), 0]:
                comps[0] *= -1
        for j in range(1, k):  # deterministic sign: largest loading positive
            col = coords[:, j]
            if col[np.argmax(np.abs(col))] < 0:
                comps[j] *= -1
        explained = (s[:k] ** 2) / np.sum(s**2)
        return cls(layer, emotions, m, unit, comps, explained)

    @classmethod
    def load(cls, root: str | Path, layer: int, n_components: int = 5) -> "EVSpace":
        return cls.from_vectors(layer, load_released_vectors(root, layer), n_components)

    # -- projection ---------------------------------------------------------

    def scores(self, x: np.ndarray, ref: Reference | np.ndarray | None = None) -> np.ndarray:
        """Emotion scores for activations x [..., D] -> [..., E]."""
        x = np.asarray(x, dtype=np.float64)
        if isinstance(ref, Reference):
            return ((x - ref.mean) @ self.unit.T) / ref.score_std
        if ref is not None:
            x = x - ref
        return x @ self.unit.T

    def pcs(self, x: np.ndarray, ref: Reference | np.ndarray | None = None) -> np.ndarray:
        """Coordinates in the fixed PCA basis, [..., K]. PC1 is valence."""
        x = np.asarray(x, dtype=np.float64)
        if isinstance(ref, Reference):
            return ((x - ref.mean) @ self.components.T) / ref.pc_std
        if ref is not None:
            x = x - ref
        return x @ self.components.T

    def cosine_scores(self, x: np.ndarray) -> np.ndarray:
        """gemotions' own scoring: cosine of the uncentered activation."""
        x = np.asarray(x, dtype=np.float64)
        xn = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
        return xn @ self.unit.T

    def fit_reference(self, acts: np.ndarray) -> Reference:
        """Reference statistics from a set of activations [N, D]."""
        acts = np.asarray(acts, dtype=np.float64)
        mean = acts.mean(axis=0)
        c = acts - mean
        score_std = np.maximum((c @ self.unit.T).std(axis=0), 1e-12)
        pc_std = np.maximum((c @ self.components.T).std(axis=0), 1e-12)
        return Reference(mean, score_std, pc_std)

    def emotion_pcs(self) -> np.ndarray:
        """Where each emotion vector sits in the PCA basis, [E, K]."""
        return (self.vectors - self.vectors.mean(axis=0)) @ self.components.T


def compare_to_released(root: str | Path, layer: int) -> dict:
    """Gate 0, check 1: vectors recomputed from the caches vs the released file."""
    rec = recompute_from_cache(root, layer)
    released = load_released_vectors(root, layer)
    missing = sorted(set(released) ^ set(rec.emotions))
    common = [e for e in rec.emotions if e in released]
    a = np.stack([rec.as_dict()[e] for e in common])
    b = np.stack([released[e] for e in common])
    cos = _cosine_rows(a, b)
    worst = int(np.argmin(cos))
    return {
        "layer": layer,
        "n_emotions": len(common),
        "name_mismatches": missing,
        "stories_per_emotion": (int(rec.counts.min()), int(rec.counts.max())),
        "noise_components": int(rec.noise_basis.shape[1]),
        "noise_variance": rec.noise_variance,
        "cosine_min": float(cos.min()),
        "cosine_mean": float(cos.mean()),
        "worst_emotion": common[worst],
        "norm_ratio_range": (
            float((np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).min()),
            float((np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).max()),
        ),
        "released_orthogonal_to_noise": float(np.abs(b @ rec.noise_basis).max()),
    }
