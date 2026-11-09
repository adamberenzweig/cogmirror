"""Gate 1 metrics. Everything here runs offline on stored activations.

Conventions: `y` is an array of emotion labels per clip, `actor` an array of
actor ids, `split` an array of "dev" / "heldout". Settings are chosen on dev
actors and reported on held-out actors.
"""

from __future__ import annotations

import numpy as np

from .datasets import NEGATIVE


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def valence_auc(valence: np.ndarray, y: np.ndarray) -> float:
    """Happy versus the four negative emotions."""
    return auc(valence[y == "happy"], valence[np.isin(y, NEGATIVE)])


def center_by_group(x: np.ndarray, groups: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Subtract each group's mean. With `mask`, the mean is taken over the
    masked rows only (e.g. an actor's neutral clips) but applied to all."""
    out = np.array(x, dtype=np.float64, copy=True)
    for g in np.unique(groups):
        rows = groups == g
        base = rows if mask is None else rows & mask
        if base.any():
            out[rows] -= x[base].mean(axis=0)
    return out


def nearest_vector_accuracy(scores: np.ndarray, y: np.ndarray, emotions: list[str], classes=NEGATIVE) -> dict:
    """Classify each clip as the class whose emotion vector scores highest.
    No training: `scores` are the readout's own per-emotion scores."""
    cols = [emotions.index(c) for c in classes]
    keep = np.isin(y, classes)
    pred = np.array(classes)[scores[keep][:, cols].argmax(axis=1)]
    true = y[keep]
    acc = float((pred == true).mean()) if keep.any() else float("nan")
    recall = {c: float((pred[true == c] == c).mean()) for c in classes if (true == c).any()}
    return {"accuracy": acc, "balanced": float(np.mean(list(recall.values()))) if recall else float("nan"),
            "recall": recall, "n": int(keep.sum())}


def eta_squared(x: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance of x explained by group membership.
    For 2-D x, variance is summed over columns."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    grand = x.mean(axis=0)
    total = ((x - grand) ** 2).sum()
    between = sum((groups == g).sum() * ((x[groups == g].mean(axis=0) - grand) ** 2).sum() for g in np.unique(groups))
    return float(between / total) if total > 0 else float("nan")


def probe_accuracy(x_train, y_train, x_test, y_test, c: float = 0.1) -> dict:
    """A trained linear classifier, for comparison only (not zero-training)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(y_test) == 0 or len(set(y_train)) < 2:
        return {"accuracy": float("nan"), "balanced": float("nan"), "n_train": int(len(y_train)), "n_test": int(len(y_test))}
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    classes = sorted(set(y_test))
    recall = {k: float((pred[y_test == k] == k).mean()) for k in classes}
    return {"accuracy": float((pred == y_test).mean()), "balanced": float(np.mean(list(recall.values()))),
            "n_train": int(len(y_train)), "n_test": int(len(y_test))}


def probe_auc(x_train, y_train_bin, x_test, y_test_bin, c: float = 0.1) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(y_test_bin)) < 2 or len(set(y_train_bin)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    clf.fit(x_train, y_train_bin)
    s = clf.decision_function(x_test)
    return auc(s[y_test_bin == 1], s[y_test_bin == 0])


def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)
