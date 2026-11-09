import numpy as np

from cogmirror import gate0


def _cache(rng, n=200, d=64):
    shared = 50 * rng.normal(size=d)          # large component common to every story
    return (shared + rng.normal(size=(n, d))).astype(np.float32)


def test_matching_row_scores_high_and_ranks_first():
    rng = np.random.default_rng(0)
    cache = _cache(rng)
    ours = cache[17] + 0.01 * rng.normal(size=cache.shape[1])
    r = gate0.compare_row(ours, cache, 17)
    assert r["rank"] == 1 and r["nearest_row"] == 17
    assert r["centered_cosine"] > 0.99


def test_raw_cosine_cannot_tell_stories_apart_but_centered_can():
    rng = np.random.default_rng(1)
    cache = _cache(rng)
    r = gate0.compare_row(cache[3], cache, 17)   # a different story
    assert r["raw_cosine"] > 0.99                # looks like a match
    assert r["centered_cosine"] < 0.5            # is not
    assert r["nearest_row"] == 3 and r["rank"] > 1


def test_passed_uses_worst_layer():
    ok = {"layers": {20: {"centered_cosine_min": 0.999}, 40: {"centered_cosine_min": 0.995}}}
    bad = {"layers": {20: {"centered_cosine_min": 0.999}, 40: {"centered_cosine_min": 0.80}}}
    assert gate0.passed(ok) and not gate0.passed(bad)
