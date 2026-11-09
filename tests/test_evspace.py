import os
from pathlib import Path

import numpy as np
import pytest

from cogmirror import evspace
from cogmirror.evspace import EVSpace

DATA = Path(os.environ.get("GEMOTIONS_DIR", Path(__file__).parents[1] / "data" / "gemotions"))


def _toy_vectors(rng, n=12, d=64):
    names = ["happy", "sad"] + [f"e{i}" for i in range(n - 2)]
    valence = rng.normal(size=d)
    vecs = {}
    for i, name in enumerate(names):
        sign = 1.0 if i % 2 == 0 else -1.0
        vecs[name] = 3 * sign * valence + rng.normal(size=d)
    return vecs


def test_noise_basis_is_orthonormal_and_hits_threshold():
    rng = np.random.default_rng(0)
    neutral = rng.normal(size=(200, 32)) * np.linspace(5, 0.1, 32)
    basis, var = evspace.noise_basis_from_neutral(neutral, 0.5)
    assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8)
    assert var >= 0.5
    if basis.shape[1] > 1:  # one fewer component would fall short
        _, s, _ = np.linalg.svd(neutral - neutral.mean(0), full_matrices=False)
        assert np.cumsum(s**2)[basis.shape[1] - 2] / np.sum(s**2) < 0.5


def test_pc1_sign_is_anchored():
    rng = np.random.default_rng(1)
    space = EVSpace.from_vectors(40, _toy_vectors(rng))
    pcs = space.emotion_pcs()
    assert pcs[space.emotions.index("happy"), 0] > pcs[space.emotions.index("sad"), 0]


def test_scores_ignore_noise_directions():
    """Denoised vectors are orthogonal to the noise basis, so removing the
    noise from an activation must not change its scores."""
    rng = np.random.default_rng(2)
    d = 64
    basis, _ = evspace.noise_basis_from_neutral(rng.normal(size=(100, d)) * np.linspace(4, 0.1, d))
    raw = _toy_vectors(rng, d=d)
    den = {k: v - basis @ (basis.T @ v) for k, v in raw.items()}
    space = EVSpace.from_vectors(40, den)
    x = rng.normal(size=(5, d))
    x_clean = x - (x @ basis) @ basis.T
    assert np.allclose(space.scores(x), space.scores(x_clean), atol=1e-9)


def test_reference_standardizes_scores():
    rng = np.random.default_rng(3)
    space = EVSpace.from_vectors(40, _toy_vectors(rng))
    acts = rng.normal(size=(500, 64)) + 10.0  # large shared offset
    ref = space.fit_reference(acts)
    s = space.scores(acts, ref)
    assert np.allclose(s.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(s.std(axis=0), 1, atol=1e-8)
    assert np.allclose(space.pcs(acts, ref).std(axis=0), 1, atol=1e-8)


def _layers_with_cache():
    base = DATA / evspace.RESULTS_SUBDIR
    if not base.exists():
        return []
    out = []
    for p in sorted(base.glob("_raw_cache_layer*")):
        layer = int(p.name.removeprefix("_raw_cache_layer"))
        if (p / "_neutral.npy").exists() and evspace.vectors_path(DATA, layer).exists():
            out.append(layer)
    return out


@pytest.mark.parametrize("layer", _layers_with_cache() or [pytest.param(None, marks=pytest.mark.skip("no gemotions cache downloaded"))])
def test_gate0_recomputed_vectors_match_released(layer):
    r = evspace.compare_to_released(DATA, layer)
    assert not r["name_mismatches"], r
    assert r["n_emotions"] == 171
    assert r["cosine_min"] >= 0.999, r
