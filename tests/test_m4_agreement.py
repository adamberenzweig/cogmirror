import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cogmirror import agreement, align, text_ev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m4_synthetic  # noqa: E402


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def test_pearson_spearman_and_small_n():
    x = np.arange(10, dtype=float)
    assert agreement.pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert agreement.spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert np.isnan(agreement.pearson(x[:3], x[:3]))
    assert np.isnan(agreement.pearson(x, np.ones(10)))


def test_zscore_fit_on_dev_only():
    x = np.array([[0.0, 10.0], [2.0, 10.0], [100.0, 10.0]])
    z = agreement.ZScore.fit(x[:2])
    assert np.allclose(z.mean, [1.0, 10.0]) and z.std[1] == 1.0        # constant column keeps scale 1
    assert z(x)[2, 0] == pytest.approx(99.0)


def test_retrieval_and_similarity_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8))
    r = agreement.retrieval(x, x)
    assert r["top1"] == 1.0 and r["mean_norm_rank"] == 0.0 and r["mrr"] == 1.0
    s = agreement.per_segment_similarity(x, x)
    assert s["cosine"] == pytest.approx(1.0) and s["spearman"] == pytest.approx(1.0)
    r2 = agreement.retrieval(x, rng.normal(size=(20, 8)))
    assert 0.2 < r2["mean_norm_rank"] < 0.8


def test_story_permutation_and_circular_shift_structure():
    rng = np.random.default_rng(1)
    story = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 5)
    idx = np.concatenate([np.arange(4), np.arange(3), np.arange(5)])
    p = agreement.story_permutation(story, idx, rng)
    # each face row is paired with a text row of exactly one story, in order within the story
    for s in "abc":
        rows = np.flatnonzero(story == s)
        partner = set(story[p[rows]])
        assert len(partner) == 1
        assert np.all(np.diff(idx[p[rows]]) >= 0)
    q = agreement.circular_shift(story, idx, rng)
    for s in "abc":
        rows = np.flatnonzero(story == s)
        assert set(q[rows]) == set(rows) and not np.array_equal(q[rows], rows)
    assert agreement.p_value(0.5, np.array([0.1, 0.2, 0.6, 0.7])) == pytest.approx(3 / 5)
    assert np.isnan(agreement.p_value(float("nan"), np.array([0.1])))


def test_turn_statistic_sign():
    story = np.array(["a"] * 4 + ["b"] * 4)
    after = np.array([0, 0, 1, 1] * 2, bool)
    has = np.ones(8, bool)
    text_v = np.array([0, 0, 1, 1, 1, 1, 0, 0], float)
    face_v = np.array([0, 0, 2, 2, 1, 1, 0, 0], float)
    t = agreement.turn_statistic(face_v, text_v, story, after, has)
    assert t["n_stories"] == 2 and t["sign_agreement"] == 1.0 and t["mean_signed_step"] == pytest.approx(1.5)


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 6))
    w = rng.normal(size=(6, 2))
    y = x @ w + 0.1 * rng.normal(size=(200, 2))
    m = agreement.fit_ridge(x, y, 0.1)
    assert np.allclose(m.w, w, atol=0.1)
    cv = agreement.ridge_cv(x, y, np.repeat(np.arange(10), 20))
    assert cv["alpha"] in (0.1, 1, 10) and cv["n_groups"] == 10


def test_make_split_is_deterministic_and_stratified():
    stories = [{"id": f"s{i}", "kind": "neutral" if i < 5 else "single"} for i in range(25)]
    takes = [align.Take("A", "t1", "s1", "read", "single", [], [], [], [], None, 1.0, False, extra={"start_wallclock": "2026-01-01"}),
             align.Take("B", "t2", "s2", "read", "single", [], [], [], [], None, 1.0, False, extra={"start_wallclock": "2026-02-01"})]
    a = agreement.make_split(takes, stories, seed=3)
    b = agreement.make_split(takes, stories, seed=3)
    assert a.holdout_session == "B" and a.holdout_stories == b.holdout_stories
    assert sum(s in a.holdout_stories for s in ("s0", "s1", "s2", "s3", "s4")) == 1 and len(a.holdout_stories) == 5
    assert agreement.make_split(takes[:1], stories).holdout_session is None


# --------------------------------------------------------------------------
# end to end on a synthetic session
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    # 2 sessions x 12 takes: enough takes per held-out condition for a story-level p below 0.05
    return m4_synthetic.make_synthetic(out, n_sessions=2, takes_per_session=12, lag=0.5, noise=0.8, seed=5)


def _collect(synth):
    cols, _ = text_ev.read_table(synth["text_ev"])
    ev = agreement.load_text_ev(cols, 50, 35)
    takes = [t for t in align.find_takes(synth["recordings"]) if t.condition in align.CONDITIONS]
    data, skipped = agreement.collect_takes(takes, synth["series_root"], ev)
    return ev, takes, data, skipped


def test_collect_and_assemble(synthetic):
    ev, takes, data, skipped = _collect(synthetic)
    assert len(data) == len(takes) and not skipped
    a = agreement.assemble(data, lag=0.5, ev_emotions=data[0].text_emotions)
    assert len(a) == sum(len(d.segments) for d in data)
    assert a.face_pcs.shape[1] == a.text_pcs.shape[1] == 5
    assert a.face_scores.shape[1] == a.text_scores.shape[1] == len(ev.emotions)
    assert a.blend is not None and a.blend.shape[1] == 4
    assert set(a.condition) <= {"read", "listen"} and a.valid().all()
    # the face follows the text at the planted lag: raw correlation is high even before normalization
    assert agreement.pearson(a.face_pcs[:, 0], a.text_pcs[:, 0]) > 0.8


def test_lag_recovered_and_analysis_passes_gate(synthetic):
    ev, takes, data, _ = _collect(synthetic)
    stories = text_ev.load_stories(synthetic["stories"])
    split = agreement.make_split(takes, stories, seed=0)
    cfg = agreement.AnalysisConfig(lag_grid=(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5), n_perm=60, seed=0)
    R = agreement.run_analysis(data, split, cfg)
    assert abs(R["lag"]["lag"] - 0.5) <= 0.5
    z = R["results"]["heldout"]["both"]["zero_training"]
    assert z["valence_r"] > 0.7 and z["p"]["valence_r_story"] < 0.05 and z["retrieval"]["top5"] > 0.8
    assert abs(z["null_mean"]["valence_r_story"]) < 0.4
    assert R["gate3"]["valence"]["pass"]
    assert "ridge" in R["results"]["heldout"]["both"] and "blendshapes" in R["results"]["heldout"]["both"]
    assert R["results"]["heldout"]["both"]["ridge"]["valence_r"] > 0.5
    # a shuffled pairing should not pass: swap the text of every take with another take's
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(data))
    while np.any(perm == np.arange(len(data))):
        perm = rng.permutation(len(data))
    shuffled = []
    for d, j in zip(data, perm):
        other = data[j]
        n = min(len(d.segments), len(other.segments))
        shuffled.append(agreement.TakeData(d.take, d.segments[:n], d.series, d.series_meta, other.text_pcs[:n], other.text_scores[:n],
                                           d.frames, d.text_emotions))
    R2 = agreement.run_analysis(shuffled, split, agreement.AnalysisConfig(lag_grid=(0.5,), n_perm=40, seed=0, blendshapes=False))
    assert not (R2["results"]["heldout"]["both"]["zero_training"]["valence_r"] > 0.5
                and R2["results"]["heldout"]["both"]["zero_training"]["p"]["valence_r_story"] < 0.05)


def test_report_and_json(synthetic, tmp_path):
    ev, takes, data, _ = _collect(synthetic)
    split = agreement.SplitConfig(None, [takes[0].story_id], note="test")
    R = agreement.run_analysis(data, split, agreement.AnalysisConfig(lag_grid=(0.5,), n_perm=10, ridge=False, blendshapes=False))
    p = agreement.write_report(R, tmp_path / "r.md")
    text = p.read_text()
    assert "## Gate 3 table" in text and "| read |" in text and "| listen |" in text and "Verbal baseline" in text
    json.loads(agreement.to_json(R))


def test_scripts_end_to_end(synthetic, tmp_path):
    py = sys.executable
    out = tmp_path / "m4"
    r = subprocess.run([py, str(ROOT / "scripts/m4_agreement.py"), "--data", str(synthetic["root"]), "--series-root", str(synthetic["series_root"]),
                        "--text-ev", str(synthetic["text_ev"]), "--out-dir", str(out), "--n-perm", "20", "--lag-grid", "0,0.5"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "agreement_report.md").exists() and (out / "agreement_metrics.json").exists() and (out / "split.json").exists()
    assert (out / "segments.parquet").exists()
    # the split file is reused on a second run
    r2 = subprocess.run([py, str(ROOT / "scripts/m4_agreement.py"), "--data", str(synthetic["root"]), "--series-root", str(synthetic["series_root"]),
                         "--text-ev", str(synthetic["text_ev"]), "--out-dir", str(out), "--n-perm", "5", "--lag", "0.5"],
                        capture_output=True, text=True)
    assert r2.returncode == 0 and "split from" in r2.stdout
    cal = next(p for p in (synthetic["recordings"] / "synth1").glob("*calibration*.json"))
    r3 = subprocess.run([py, str(ROOT / "scripts/m4_scripted_events.py"), "--take", str(cal), "--series-root", str(synthetic["series_root"]),
                         "--out-dir", str(out)], capture_output=True, text=True)
    assert r3.returncode == 0, r3.stdout + r3.stderr
    summary = json.loads((out / f"scripted_events_{cal.stem}.json").read_text())
    assert summary["n_scored"] == 3 and summary["n_within_1s"] == 3 and summary["pass"]
    # CSV events and a signal override
    csv = tmp_path / "events.csv"
    csv.write_text("t,label\n10,smile\n")
    r4 = subprocess.run([py, str(ROOT / "scripts/m4_scripted_events.py"), "--take", str(cal), "--events", str(csv), "--signal", "valence:+1",
                         "--series-root", str(synthetic["series_root"]), "--out-dir", str(out)], capture_output=True, text=True)
    assert r4.returncode == 0 and "| smile | 10.00 | valence (+)" in r4.stdout
