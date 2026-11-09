import json

import numpy as np
import pytest

from cogmirror import align
from cogmirror.text_ev import TextEV

SENTENCES = ["One two three.", "Four five.", "Six seven eight nine."]


def _log(condition, session="s1", idx=2, story="f001", turn=None):
    if condition == "listen":
        t = align.LISTEN_LEAD_IN
        times, events = [], []
        for i, d in enumerate((2.0, 1.5, 3.0)):
            events += [{"t": t, "type": "sentence_start", "sentence_idx": i, "audio_duration_s": d},
                       {"t": t + d, "type": "sentence_end", "sentence_idx": i}]
            times.append({"sentence_idx": i, "start": t, "end": t + d})
            t += d + align.LISTEN_GAP
    else:
        bounds = [(0.0, 2.2), (2.2, 3.5), (3.5, 6.0)]
        times = [{"sentence_idx": i, "start": s, "end": e} for i, (s, e) in enumerate(bounds)]
        events = []
        for i, (s, e) in enumerate(bounds):
            events += [{"t": s, "type": "sentence_start", "sentence_idx": i}, {"t": e, "type": "sentence_end", "sentence_idx": i}]
    return {"session_id": session, "take_id": f"{idx:03d}_{story}_{condition}", "take_index": idx, "story_id": story,
            "condition": condition, "kind": "turn" if turn else "single", "emotions": ["sad"], "sentences": SENTENCES,
            "turn": turn, "duration_s": 10.0, "sentence_times": times, "events": events, "bad": False}


def test_load_and_find_takes(tmp_path):
    rec = tmp_path / "rec" / "s1"
    rec.mkdir(parents=True)
    (rec / "002_f001_listen.json").write_text(json.dumps(_log("listen")))
    bad = _log("read", idx=3)
    bad["bad"] = True
    (rec / "003_f001_read.json").write_text(json.dumps(bad))
    takes = align.find_takes(tmp_path / "rec")
    assert [t.take_id for t in takes] == ["002_f001_listen"]
    assert len(align.find_takes(tmp_path / "rec", include_bad=True)) == 2
    t = takes[0]
    assert t.condition == "listen" and t.story_id == "f001" and t.video_path.name == "002_f001_listen.webm"


def test_segments_listening_uses_audio_events(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps(_log("listen", turn={"sentence_idx": 2, "word_idx": 5})))
    segs = align.segments_from_take(align.load_take(p))
    assert [s.sentence_idx for s in segs] == [0, 1, 2]
    assert segs[0].start == pytest.approx(1.5) and segs[0].end == pytest.approx(3.5)
    assert segs[1].start == pytest.approx(4.0)          # 0.5 s gap
    assert segs[2].turn_starts_here and segs[2].after_turn and not segs[1].after_turn
    assert segs[1].text == "Four five." and segs[0].source == "log"


def test_segments_reading_uses_key_presses_and_falls_back_to_events(tmp_path):
    log = _log("read")
    log["sentence_times"] = []          # older log without the summary: rebuild from events
    p = tmp_path / "x.json"
    p.write_text(json.dumps(log))
    segs = align.segments_from_take(align.load_take(p))
    assert [(s.start, s.end) for s in segs] == [(0.0, 2.2), (2.2, 3.5), (3.5, 6.0)]
    # the calibration take yields no segments
    cal = dict(log, story_id=None, condition="calibration", sentences=[], sentence_times=[], events=[])
    p.write_text(json.dumps(cal))
    assert align.segments_from_take(align.load_take(p)) == []


def test_refine_reading_bounds_with_words():
    bounds = [(0, 0.0, 2.2), (1, 2.2, 3.5), (2, 3.5, 6.0)]
    words = [{"word": "One", "start": 0.3, "end": 0.5}, {"word": "two", "start": 0.6, "end": 0.8}, {"word": "three.", "start": 0.9, "end": 1.4},
             {"word": "Four", "start": 2.5, "end": 2.7}, {"word": "five,", "start": 2.8, "end": 3.1},
             {"word": "six", "start": 3.9, "end": 4.1}, {"word": "seven", "start": 4.2, "end": 4.4}, {"word": "eight", "start": 4.5, "end": 4.7},
             {"word": "nine", "start": 4.8, "end": 5.2}]
    out = align.refine_reading_bounds(bounds, words, SENTENCES)
    assert out[0] == (0, 0.3, 1.4) and out[1] == (1, 2.5, 3.1) and out[2] == (2, 3.9, 5.2)
    # a far-off match is ignored (kept within 1.5 s of the key press)
    words2 = [dict(w, start=w["start"] + 5, end=w["end"] + 5) for w in words]
    assert align.refine_reading_bounds(bounds, words2, SENTENCES) is None
    p = {"word": "x"}
    assert align.refine_reading_bounds(bounds, [p], SENTENCES) is None


def test_segment_means_with_lag():
    t = np.arange(0, 10, 0.5)
    x = np.stack([t, np.ones_like(t)], axis=1)
    m, n = align.segment_means(t, x, [1.0, 5.0], [3.0, 6.0])
    assert n.tolist() == [4, 2] and m[0, 0] == pytest.approx(np.mean([1.0, 1.5, 2.0, 2.5]))
    m2, _ = align.segment_means(t, x, [1.0], [3.0], lag=1.0)
    assert m2[0, 0] == pytest.approx(np.mean([2.0, 2.5, 3.0, 3.5]))
    m3, n3 = align.segment_means(t, x, [20.0], [21.0])
    assert n3[0] == 0 and np.isnan(m3).all()


def test_face_and_text_segment_features():
    t = np.arange(0, 8, 0.2)
    series = {"t": t, "pc1": np.where(t < 4, -1.0, 1.0), "pc2": np.zeros_like(t), "valence_raw": np.where(t < 4, -2.0, 2.0),
              "score_a": np.ones_like(t), "score_b": t}
    segs = [align.Segment("s", "k", "f001", "read", 0, 0.0, 4.0), align.Segment("s", "k", "f001", "read", 1, 4.0, 8.0)]
    f = align.face_segment_features(series, segs)
    assert f.pcs[:, 0].tolist() == [-1.0, 1.0] and f.emotions == ["a", "b"] and f.scores[0, 0] == 1.0
    f2 = align.face_segment_features(series, segs, raw_valence=True)
    assert f2.pcs[:, 0].tolist() == [-2.0, 2.0]
    ev = TextEV(50, ["a", "b"], {("f001", 0): 0, ("f001", 1): 1}, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.5], [0.7]]),
                np.array(["f001", "f001"]), np.array([0, 1]))
    pcs, scores, have = align.text_segment_features(ev, segs + [align.Segment("s", "k", "f009", "read", 0, 0.0, 1.0)])
    assert have.tolist() == [True, True, False] and pcs[1, 0] == 0.7 and np.isnan(pcs[2]).all()


def test_scripted_events_from_log_and_csv(tmp_path):
    log = {"events": [{"t": 0.0, "type": "calibration_start"}, {"t": 10.0, "type": "event", "label": "Smile"},
                      {"t": 20.0, "type": "yawn_start"}, {"t": 23.0, "type": "yawn_end"}, {"t": 60.0, "type": "calibration_end"}]}
    ev = align.scripted_events_from_log(log)
    assert [(e.t, e.label, e.end) for e in ev] == [(10.0, "smile", None), (20.0, "yawn", 23.0)]
    csv = tmp_path / "e.csv"
    csv.write_text("t,label,end\n5,frown,\n# comment\n12.5,surprise,14\n")
    ev2 = align.scripted_events_from_csv(csv)
    assert [(e.t, e.label, e.end) for e in ev2] == [(5.0, "frown", None), (12.5, "surprise", 14.0)]
    assert align.event_signal("smile", {"valence": 1}) == ("valence", 1)
    assert align.event_signal("yawn", {"valence": 1}) is None       # score_tired missing from the series
    assert align.event_signal("hum", {"valence": 1}) is None


def test_event_response_finds_onset_within_window():
    t = np.arange(0, 20, 0.2)
    x = np.where(t >= 10.4, 3.0, 0.0) + np.random.default_rng(0).normal(0, 0.05, len(t))
    r = align.event_response(t, x, 10.0)
    assert r["onset"] == pytest.approx(10.4) and r["within_1s"] and r["lag"] == pytest.approx(0.4) and r["peak_z"] > 5
    r2 = align.event_response(t, -x, 10.0)       # wrong direction: no onset
    assert np.isnan(r2["onset"]) and not r2["within_1s"]
    r3 = align.event_response(t, x, 15.0)        # already high: no step within the window
    assert np.isnan(r3["onset"])
