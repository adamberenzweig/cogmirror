import json

import numpy as np
import pytest

from cogmirror import store, video
from cogmirror.evspace import EVSpace, Reference


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def test_sample_mask_rate_and_first_frame():
    m = video.sample_mask(300, 30.0, 5.0)
    assert m[0]
    assert m.sum() == 50                      # 10 s at 5 fps
    idx = np.flatnonzero(m)
    assert set(np.diff(idx).tolist()) == {6}  # evenly spaced at 30 / 5


def test_sample_mask_streaming_matches_vector_form():
    n, fps, rate = 109, 29.97, 5.0
    m = video.sample_mask(n, fps, rate)
    bucket, kept = -1, []
    for i in range(n):
        keep, bucket = video.keep_frame(i, fps, rate, bucket)
        kept.append(keep)
    assert np.array_equal(m, np.array(kept))
    assert 17 <= m.sum() <= 19


def test_sample_mask_rate_above_source_keeps_everything():
    assert video.sample_mask(10, 5.0, 30.0).all()
    assert video.sample_mask(0, 30.0, 5.0).size == 0


# --------------------------------------------------------------------------
# box smoothing
# --------------------------------------------------------------------------

def test_one_euro_removes_jitter_but_tracks_motion():
    rng = np.random.default_rng(0)
    f = video.OneEuro(min_cutoff=0.6, beta=0.02)
    t = np.arange(0, 10, 0.2)
    truth = 100 + 20 * (t > 5)                # a step at 5 s
    noisy = truth + rng.normal(0, 3, len(t))
    out = np.array([f.update([x], ti)[0] for x, ti in zip(noisy, t)])
    # less jitter than the input on the flat part
    assert out[5:24].std() < noisy[5:24].std() * 0.8
    # and the step is followed within about a second
    assert abs(out[-10:].mean() - noisy[-10:].mean()) < 1   # the filter is unbiased
    assert out[t.tolist().index(6.0)] > 110


def test_box_smoother_holds_through_short_gaps_and_resets_after_long_ones():
    s = video.BoxSmoother(max_gap=1.0)
    box, held = s.update((100, 100, 50), 0.0)
    assert not held and np.allclose(box, [100, 100, 50])
    box, held = s.update(None, 0.4)
    assert held and np.allclose(box, [100, 100, 50])
    box, held = s.update(None, 1.6)           # gap longer than max_gap
    assert box is None and not held
    box, held = s.update((300, 300, 80), 2.0)  # re-acquired: fresh start, no drag from the old box
    assert not held and np.allclose(box, [300, 300, 80])


def test_box_smoother_size_is_smoothed_harder_than_center():
    s = video.BoxSmoother()
    s.update((0, 0, 100), 0.0)
    box, _ = s.update((20, 0, 120), 0.2)
    center_frac = box[0] / 20
    size_frac = (box[2] - 100) / 20
    assert 0 < size_frac < center_frac < 1


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

def _payload(idx, layers=(20, 35, 50), d=8, seed=0):
    rng = np.random.default_rng(seed)
    n = len(idx)
    return store.chunk_payload(idx, np.asarray(idx) / 30.0, rng.normal(size=(n, len(layers), d)),
                               rng.normal(size=(n, len(layers), d)), layers)


def test_plan_chunks():
    assert store.plan_chunks(7, 3) == [(0, 0, 3), (1, 3, 6), (2, 6, 7)]
    assert store.plan_chunks(0, 3) == []
    with pytest.raises(ValueError):
        store.plan_chunks(3, 0)


def test_store_roundtrip_sorted_and_deduplicated(tmp_path):
    p1 = _payload([30, 0, 12], seed=1)
    p2 = _payload([18, 12], seed=2)           # frame 12 again: the later chunk wins
    store.save_chunk(tmp_path, 1, p2)
    store.save_chunk(tmp_path, 0, p1)
    assert store.chunk_done(tmp_path, 0) and store.chunk_done(tmp_path, 1) and not store.chunk_done(tmp_path, 2)
    acts = store.load_store(tmp_path)
    assert acts["frame_index"].tolist() == [0, 12, 18, 30]
    assert acts["layers"] == [20, 35, 50]
    assert acts["last"].dtype == np.float16
    row12 = acts["last"][acts["frame_index"].tolist().index(12)]
    assert np.array_equal(row12, p2["last"][1])
    assert store.store_frames(tmp_path).tolist() == [0, 12, 18, 30]
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_chunk_payload_validates_shapes():
    with pytest.raises(ValueError):
        store.chunk_payload([0, 1], [0.0, 0.1], np.zeros((2, 3, 4)), np.zeros((2, 3, 5)), [1, 2, 3])
    with pytest.raises(ValueError):
        store.chunk_payload([0], [0.0, 0.1], np.zeros((2, 3, 4)), np.zeros((2, 3, 4)), [1, 2, 3])


# --------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------

def test_reference_roundtrip(tmp_path):
    refs = {("last", 35): Reference(np.arange(4.0), np.array([1.0, 2.0]), np.array([0.5, 0.25, 0.125])),
            ("image", 50): Reference(np.ones(4), np.array([3.0, 4.0]), np.array([2.0, 1.0, 0.5]))}
    path = store.save_reference(tmp_path / "ref.npz", refs, {"layout": "a2_prefill"})
    back, meta = store.load_reference(path)
    assert meta["layout"] == "a2_prefill"
    assert set(back) == set(refs)
    assert np.allclose(back[("last", 35)].mean, np.arange(4.0))
    assert np.allclose(back[("image", 50)].pc_std, [2.0, 1.0, 0.5])
    cut = store.reference_for(back, "last", 35, 2)
    assert cut.pc_std.tolist() == [0.5, 0.25]
    with pytest.raises(ValueError):
        store.reference_for(back, "last", 35, 4)


# --------------------------------------------------------------------------
# projection, baseline, smoothing
# --------------------------------------------------------------------------

def _space(layer, d=6, seed=0):
    rng = np.random.default_rng(seed)
    vecs = {name: rng.normal(size=d) for name in ["happy", "sad", "angry", "calm", "bored"]}
    return EVSpace.from_vectors(layer, vecs, n_components=3)


def test_project_uses_reference_and_layers():
    layers = [20, 35, 50]
    spaces = {35: _space(35, seed=1), 50: _space(50, seed=2)}
    rng = np.random.default_rng(0)
    cal = rng.normal(size=(50, 6))
    refs = {("last", 35): spaces[35].fit_reference(cal), ("last", 50): spaces[50].fit_reference(cal + 1)}
    x = rng.normal(size=(4, 3, 6))
    acts = {"frame_index": np.arange(4), "t": np.arange(4) / 5, "last": x.astype(np.float16), "image": x.astype(np.float16),
            "layers": layers}
    proj = store.project(acts, spaces, refs, store.ProjectionConfig(n_pcs=2))
    assert proj.scores.shape == (4, 5) and proj.pcs.shape == (4, 2)
    expect = spaces[50].pcs(x[:, 2].astype(np.float16).astype(np.float64), store.reference_for(refs, "last", 50, 3))
    assert np.allclose(proj.valence, expect[:, 0])
    assert proj.emotions == spaces[35].emotions
    with pytest.raises(ValueError):
        store.project(acts, spaces, refs, store.ProjectionConfig(score_layer=40))


def test_baseline_fit_apply_roundtrip(tmp_path):
    proj = store.Projected(np.arange(3), np.arange(3) / 5.0, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                           np.array([[1.0], [2.0], [3.0]]), ["a", "b"], store.ProjectionConfig())
    base = store.fit_baseline(proj, source="cal")
    assert np.allclose(base.scores, [3.0, 4.0]) and np.allclose(base.pcs, [2.0]) and base.n_frames == 3
    base.save(tmp_path / "b.json")
    back = store.Baseline.load(tmp_path / "b.json")
    out = store.apply_baseline(proj, back)
    assert np.allclose(out.valence, [-1.0, 0.0, 1.0])
    assert np.allclose(out.scores.mean(axis=0), 0)


def test_window_mean_is_centered_and_nan_aware():
    t = np.arange(0, 2.0, 0.2)
    x = np.zeros(len(t))
    x[5] = 10.0                       # one spike at t = 1.0
    y = store.window_mean(t, x, 1.0)  # +-0.5 s -> 5 or 6 samples
    assert y[5] == pytest.approx(10 / 5, abs=1e-9) or y[5] == pytest.approx(10 / 6, abs=1e-9)
    assert y[3] > 0 and y[7] > 0 and y[2] == 0 and y[8] == 0
    assert y[3] == y[7] and y[4] == y[6]   # symmetric around the spike: no lag
    x[2] = np.nan
    y = store.window_mean(t, x, 1.0)
    assert np.isfinite(y).all()       # NaNs are ignored, not propagated
    assert np.array_equal(store.window_mean(t, x, 0), x, equal_nan=True)
    m = store.window_mean(t, np.stack([x, 2 * x], axis=1), 1.0)
    assert m.shape == (len(t), 2) and np.allclose(m[:, 1], 2 * m[:, 0], equal_nan=True)


def test_ema_and_top_emotions():
    t = np.arange(5) * 0.2
    x = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    y = store.ema(t, x, 0.2)
    assert y[0] == 0 and 0 < y[1] < 1 and y[1] < y[2] < y[3] < 1
    assert np.array_equal(store.ema(t, x, 0), x)
    top = store.top_emotions(np.array([[0.1, 0.9, 0.5], [np.nan, 1, 2]]), ["a", "b", "c"], k=2)
    assert top == [["b", "c"], ["", ""]]


def test_build_series_rows_follow_frame_manifest_and_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    frames = [{"frame_index": i * 6, "t": i * 0.2, "face_found": i != 2} for i in range(5)]
    present = [0, 6, 18, 24]          # frame 12 has no activation
    rng = np.random.default_rng(0)
    proj = store.Projected(np.array(present), np.array(present) / 30.0, rng.normal(size=(4, 3)), rng.normal(size=(4, 2)),
                           ["a", "b", "c"], store.ProjectionConfig(n_pcs=2))
    cols = store.build_series(proj, store.SeriesConfig(window=0.0, top_k=2), frames=frames)
    assert cols["frame_index"].tolist() == [0, 6, 12, 18, 24]
    assert np.isnan(cols["valence_raw"][2]) and cols["top1"][2] == ""
    assert not cols["face_found"][2]
    assert np.allclose(cols["valence"][[0, 1, 3, 4]], proj.pcs[:, 0])
    assert set(c for c in cols if c.startswith("score_")) == {"score_a", "score_b", "score_c"}
    path = store.write_parquet(tmp_path / "s.parquet", cols, {"window_s": "0.0"})
    back, meta = store.read_parquet(path)
    assert meta["window_s"] == "0.0"
    assert back["top1"].tolist() == cols["top1"].tolist()
    assert np.allclose(back["valence"], cols["valence"], equal_nan=True)
    js = store.series_to_json(back, meta)
    json.dumps(js)                    # NaN became null
    assert js["valence_raw"][2] is None and len(js["scores"]) == 5 and js["emotions"] == ["a", "b", "c"]


# --------------------------------------------------------------------------
# onsets
# --------------------------------------------------------------------------

def test_onset_time():
    t = np.arange(10) * 0.2
    x = np.array([0, 0, 0, 0.1, 0.6, 0.9, 1, 1, 1, 1.0])
    assert store is not None
    assert video.onset_time(t, x, margin=0.5) == pytest.approx(0.8)
    assert video.onset_time(t, x, margin=0.5, min_run=2) == pytest.approx(0.8)
    assert np.isnan(video.onset_time(t, x, margin=5.0))
    x[1] = np.nan
    assert video.onset_time(t, x, margin=0.5) == pytest.approx(0.8)
    assert np.isnan(video.onset_time(t[:2], x[:2], margin=0.5))


def test_blendshape_series():
    recs = [{"blendshapes": {"mouthSmileLeft": 0.2, "mouthSmileRight": 0.4}}, {}, {"blendshapes": {"mouthSmileLeft": 1.0}}]
    s = video.blendshape_series(recs, ("mouthSmileLeft", "mouthSmileRight"))
    assert s[0] == pytest.approx(0.3) and np.isnan(s[1]) and s[2] == 1.0


def test_crop_for_model_upper_variant():
    from PIL import Image

    img = Image.new("RGB", (600, 600), (10, 20, 30))
    assert video.crop_for_model(img, "medium", 256).size == (256, 256)
    assert video.crop_for_model(img, "upper", 256).size == (256, 154)
