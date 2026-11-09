"""Gate 0, check 2: do our activations reproduce gemotions' cached ones?

Raw cosine similarity between residual activations is dominated by a large
component shared by every text, so two *different* stories already score
close to 1. The checks here therefore use quantities that can tell stories
apart:

- centered cosine: cosine after subtracting the mean of that emotion's
  cached rows from both vectors
- rank: where the expected cache row falls when all 1,000 rows for that
  emotion are ordered by distance to our activation (1 is a match)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from . import evspace

# The exact query gemotions' extract_vectors.py uses. Row order within an
# emotion defines the cache row index, so it must not be changed.
STORY_QUERY = "SELECT emotion, text FROM stories_clean ORDER BY emotion, story_idx"

DEFAULT_EMOTIONS = (
    "happy", "sad", "angry", "afraid", "disgusted", "calm", "excited", "bored",
    "proud", "ashamed", "grateful", "jealous", "nostalgic", "sleepy", "desperate", "playful",
)


def load_stories(root: str | Path) -> dict[str, list[str]]:
    con = sqlite3.connect(f"file:{Path(root) / 'data' / 'stories.db'}?mode=ro", uri=True, timeout=30)
    try:
        rows = con.execute(STORY_QUERY).fetchall()
    finally:
        con.close()
    out: dict[str, list[str]] = {}
    for emotion, text in rows:
        out.setdefault(emotion, []).append(text)
    return out


def pick(stories: dict[str, list[str]], emotions, per_emotion: int, seed: int = 0):
    """(emotion, row index, text) samples, spread across each emotion's rows."""
    rng = np.random.default_rng(seed)
    picks = []
    for e in emotions:
        if e not in stories:
            continue
        idx = sorted(rng.choice(len(stories[e]), size=min(per_emotion, len(stories[e])), replace=False).tolist())
        picks.extend((e, i, stories[e][i]) for i in idx)
    return picks


def compare_row(ours: np.ndarray, cache: np.ndarray, row: int) -> dict:
    """Compare one recomputed activation with its expected cache row."""
    cache = np.asarray(cache, dtype=np.float64)
    ours = np.asarray(ours, dtype=np.float64)
    mu = cache.mean(axis=0)
    c = cache - mu
    o = ours - mu
    cos_all = (c @ o) / np.maximum(np.linalg.norm(c, axis=1) * np.linalg.norm(o), 1e-12)
    dist = np.linalg.norm(cache - ours, axis=1)
    order = np.argsort(dist)
    exp = cache[row]
    return {
        "raw_cosine": float(exp @ ours / (np.linalg.norm(exp) * np.linalg.norm(ours))),
        "centered_cosine": float(cos_all[row]),
        "rank": int(np.nonzero(order == row)[0][0]) + 1,
        "nearest_row": int(order[0]),
        "nearest_centered_cosine": float(cos_all[order[0]]),
        "rel_error": float(dist[row] / max(np.linalg.norm(exp - mu), 1e-12)),
        "median_other_centered_cosine": float(np.median(np.delete(cos_all, row))),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-story comparisons for one layer."""
    cc = np.array([r["centered_cosine"] for r in rows])
    return {
        "n": len(rows),
        "centered_cosine_min": float(cc.min()),
        "centered_cosine_median": float(np.median(cc)),
        "raw_cosine_min": float(min(r["raw_cosine"] for r in rows)),
        "rank1_fraction": float(np.mean([r["rank"] == 1 for r in rows])),
        "nearest_is_other_row_but_matches": int(sum(
            r["rank"] != 1 and r["nearest_centered_cosine"] >= 0.99 for r in rows)),
        "rel_error_median": float(np.median([r["rel_error"] for r in rows])),
        "other_story_centered_cosine_median": float(np.median([r["median_other_centered_cosine"] for r in rows])),
    }


VARIANTS = {"no_bos": False, "bos": True}


def run(model, tokenizer, root: str | Path, layers, emotions=DEFAULT_EMOTIONS, per_emotion: int = 5,
        progress=print) -> dict:
    """Recompute story activations with `model` and compare with the caches.

    Each story is run twice, without and with a leading BOS token. gemotions
    used the bare tokenizer (no BOS for Gemma 4), so `no_bos` is the expected
    match; the other variant is there to make a mismatch diagnosable.
    """
    from .model import story_activation

    layers = [l for l in layers if evspace.cache_dir(root, l).exists()]
    stories = load_stories(root)
    picks = pick(stories, emotions, per_emotion)
    rows: dict[str, dict[int, list[dict]]] = {v: {l: [] for l in layers} for v in VARIANTS}
    caches: dict[tuple[int, str], np.ndarray] = {}
    for n, (emotion, row, text) in enumerate(picks, 1):
        for variant, bos in VARIANTS.items():
            acts = story_activation(model, tokenizer, text, layers, bos=bos)
            for l in layers:
                key = (l, emotion)
                if key not in caches:
                    caches[key] = np.asarray(evspace.load_cache_array(root, l, emotion), dtype=np.float32)
                r = compare_row(acts[l].numpy(), caches[key], row)
                r.update(emotion=emotion, row=row)
                rows[variant][l].append(r)
        if n % 10 == 0:
            progress(f"  {n}/{len(picks)} stories")
    variants = {v: {l: summarize(rs) for l, rs in per.items()} for v, per in rows.items()}
    best = max(variants, key=lambda v: min(s["centered_cosine_min"] for s in variants[v].values()))
    return {
        "best_variant": best,
        "layers": variants[best],
        "variants": variants,
        "detail": rows,
        "n_stories": len(picks),
    }


def passed(summary: dict, min_centered_cosine: float = 0.99) -> bool:
    """Every sampled story matches its expected cache row at every layer."""
    return all(s["centered_cosine_min"] >= min_centered_cosine for s in summary["layers"].values())
