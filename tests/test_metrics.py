import numpy as np

from cogmirror import metrics


def test_auc_basic_and_ties():
    assert metrics.auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert metrics.auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert metrics.auc([1, 1], [1, 1]) == 0.5
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 400), rng.normal(0, 1, 400)
    brute = np.mean(a[:, None] > b[None, :])
    assert abs(metrics.auc(a, b) - brute) < 1e-9


def test_center_by_group_with_mask_uses_only_masked_rows():
    x = np.array([1.0, 3.0, 10.0, 20.0])
    g = np.array(["a", "a", "b", "b"])
    out = metrics.center_by_group(x, g, mask=np.array([True, False, True, False]))
    assert out.tolist() == [0.0, 2.0, 0.0, 10.0]


def test_eta_squared_extremes():
    g = np.array([0, 0, 1, 1])
    assert metrics.eta_squared(np.array([1.0, 1.0, 5.0, 5.0]), g) == 1.0
    assert metrics.eta_squared(np.array([1.0, 5.0, 1.0, 5.0]), g) == 0.0


def test_nearest_vector_accuracy():
    emotions = ["afraid", "angry", "disgusted", "happy", "sad"]
    y = np.array(["angry", "sad", "happy", "afraid"])
    scores = np.array([[0, 5, 0, 9, 0], [0, 0, 0, 9, 4], [0, 0, 0, 9, 0], [0, 0, 3, 9, 0]], float)
    r = metrics.nearest_vector_accuracy(scores, y, emotions)
    assert r["n"] == 3 and abs(r["accuracy"] - 2 / 3) < 1e-9   # happy column and the happy clip are ignored
