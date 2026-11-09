"""A synthetic recording session with known structure, to exercise the M4 code end to end.

    .venv/bin/python scripts/m4_synthetic.py --out data/synthetic_m4 [--sessions 2 --takes 10 --lag 0.5 --noise 1.0]

Writes, under --out:
    stories/stories.jsonl                  the story set used (the real one if --stories exists, else fabricated)
    results/m4/text_ev.parquet             text EVs (the real table if --text-ev exists, else fabricated)
    recordings/<session>/<take>.json       take logs in the recorder's format (listening and reading takes,
                                           plus a calibration take with marked scripted events)
    video/<session>/<take>/<take>.parquet  face series in the video pipeline's format: the text's EV of the
                                           sentence being read/heard, delayed by --lag, plus Gaussian noise
    video/<session>/<take>/frames/frames.jsonl  tracker records with synthetic blendshapes

The face of a synthetic take follows the text with lag and noise, so the
agreement analysis should recover the lag and find a positive valence
correlation; the calibration take's events (smile, frown, yawn) move the
corresponding signal 0.3 s after the mark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogmirror import align, store, text_ev  # noqa: E402

FAKE_EMOTIONS = ["happy", "sad", "angry", "afraid", "disgusted", "surprised", "calm", "excited", "bored", "tired",
                 "anxious", "proud", "ashamed", "grateful", "lonely", "content", "frustrated", "relieved", "amused", "hopeful"]
N_PCS = 5
FPS = 5.0
WORD_S = 0.38           # seconds per word when reading aloud
EVENT_DELAY = 0.3       # the synthetic face responds this long after a marked event
EVENT_LEN = 2.0


def fake_stories(n: int, rng: np.random.Generator) -> list[dict]:
    out = []
    for i in range(n):
        kind = "neutral" if i % 7 == 6 else ("turn" if i % 3 == 2 else "single")
        n_sent = int(rng.integers(7, 11))
        words_per = rng.integers(8, 16, size=n_sent)
        sentences = [" ".join(f"w{i}{k}{j}" for j in range(w)) + "." for k, w in enumerate(words_per)]
        offsets = np.concatenate([[0], np.cumsum(words_per)[:-1]]).tolist()
        emotions = [] if kind == "neutral" else [FAKE_EMOTIONS[i % len(FAKE_EMOTIONS)]]
        turn = None
        if kind == "turn":
            emotions.append(FAKE_EMOTIONS[(i * 7 + 3) % len(FAKE_EMOTIONS)])
            ti = n_sent // 2
            turn = {"sentence_idx": ti, "word_idx": offsets[ti], "n_sentences_before": ti}
        out.append({"id": f"f{i + 1:03d}", "kind": kind, "emotions": emotions, "clusters": [], "quadrants": [],
                    "topic": f"topic {i}", "text": " ".join(sentences), "sentences": sentences,
                    "sentence_word_offsets": offsets, "n_sentences": n_sent, "word_count": int(words_per.sum()), "turn": turn})
    return out


def fake_text_ev(stories: list[dict], rng: np.random.Generator, layers=(35, 50), emotions=FAKE_EMOTIONS) -> dict[str, np.ndarray]:
    """A text EV table: each story has a random profile (and a second one after its turn);
    sentences are the story profile plus a little sentence-level variation. Valence (pc1) is
    the first score's sign-weighted sum so that neutral stories sit near zero."""
    rows = {"story_id": [], "sentence_idx": [], "layer": [], "origin": []}
    scores, pcs = [], []
    E = len(emotions)
    for s in stories:
        prof = rng.normal(0, 1, E) if s["kind"] != "neutral" else rng.normal(0, 0.15, E)
        prof2 = rng.normal(0, 1, E) if s["kind"] == "turn" else prof
        turn = (s.get("turn") or {}).get("sentence_idx")
        sent = []
        for i in range(len(s["sentences"])):
            base = prof2 if (turn is not None and i >= turn) else prof
            sent.append(base + rng.normal(0, 0.3, E))
        sent = np.stack(sent)
        story_row = sent.mean(axis=0)
        for layer in layers:
            for i, row in enumerate(sent):
                rows["story_id"].append(s["id"]); rows["sentence_idx"].append(i); rows["layer"].append(layer); rows["origin"].append("synthetic")
                scores.append(row); pcs.append(_pcs_of(row))
            rows["story_id"].append(s["id"]); rows["sentence_idx"].append(text_ev.STORY_ROW); rows["layer"].append(layer); rows["origin"].append("synthetic")
            scores.append(story_row); pcs.append(_pcs_of(story_row))
    scores = np.stack(scores)
    pcs = np.stack(pcs)
    cols: dict[str, np.ndarray] = {"story_id": np.asarray(rows["story_id"], dtype=str), "sentence_idx": np.asarray(rows["sentence_idx"], dtype=np.int64),
                                   "layer": np.asarray(rows["layer"], dtype=np.int64), "origin": np.asarray(rows["origin"], dtype=str)}
    for j, e in enumerate(emotions):
        cols[f"score_{e}"] = scores[:, j]
    for j in range(pcs.shape[1]):
        cols[f"pc{j + 1}"] = pcs[:, j]
    return cols


def _pcs_of(row: np.ndarray) -> np.ndarray:
    """Fixed linear 'PCA': pc1 = valence-like contrast of the first scores, others random fixed projections."""
    E = len(row)
    rng = np.random.default_rng(1234)
    basis = rng.normal(0, 1, (N_PCS, E))
    basis[0] = 0
    basis[0, :E // 2] = 1.0      # 'positive' emotions
    basis[0, E // 2:] = -1.0     # 'negative' ones
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    return basis @ row


def take_log(session: str, idx: int, story: dict | None, condition: str, durations: list[float] | None, rng: np.random.Generator,
             events: list[dict] | None = None) -> dict:
    take_id = f"{idx:03d}_{story['id'] if story else 'calibration'}_{condition}"
    evs: list[dict] = []
    sentence_times = []
    if story is None:
        evs = [{"t": 0.0, "type": "calibration_start"}] + list(events or []) + [{"t": 60.0, "type": "calibration_end"}]
        duration = 61.0
    elif condition == "listen":
        t = align.LISTEN_LEAD_IN
        for i, d in enumerate(durations):
            evs.append({"t": round(t, 3), "type": "sentence_start", "sentence_idx": i, "audio_duration_s": d})
            evs.append({"t": round(t + d, 3), "type": "sentence_end", "sentence_idx": i})
            sentence_times.append({"sentence_idx": i, "start": round(t, 3), "end": round(t + d, 3)})
            t += d + align.LISTEN_GAP
        duration = t - align.LISTEN_GAP + align.TAIL
    else:
        t = 0.0
        for i, d in enumerate(durations):
            d = d * float(rng.uniform(0.9, 1.15))
            evs.append({"t": round(t, 3), "type": "sentence_start", "sentence_idx": i})
            evs.append({"t": round(t + d, 3), "type": "sentence_end", "sentence_idx": i})
            sentence_times.append({"sentence_idx": i, "start": round(t, 3), "end": round(t + d, 3)})
            t += d
        duration = t + align.TAIL
    return {
        "session_id": session, "take_id": take_id, "take_index": idx, "story_id": story["id"] if story else None,
        "condition": condition, "kind": story["kind"] if story else "calibration",
        "emotions": story["emotions"] if story else [], "clusters": [], "topic": story["topic"] if story else None,
        "story_text": story["text"] if story else None, "sentences": story["sentences"] if story else [],
        "turn": story.get("turn") if story else None, "start_wallclock": f"2026-11-{10 + int(session[-1]) if session[-1].isdigit() else 10:02d}T10:{idx:02d}:00.000Z",
        "duration_s": round(duration, 3), "timestamps_relative_to": "MediaRecorder start event, seconds",
        "sentence_times": sentence_times, "events": evs,
        "camera": {"video": {"width": 1280, "height": 720, "frameRate": 30}, "fake_stream": True},
        "recorder": {"mimeType": "video/webm;codecs=vp9,opus", "videoBitsPerSecond": 8000000, "timeslice_ms": 1000},
        "video_bytes": 0, "video_file": f"{take_id}.webm", "user_agent": "synthetic", "screen": {}, "bad": False, "note": "synthetic",
    }


def face_series(log: dict, ev: text_ev.TextEV, rng: np.random.Generator, lag: float, noise: float, fps: float = FPS,
                events: list[dict] | None = None, window: float = 1.5) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Per-frame series: the current sentence's text EV (delayed by lag) plus noise, windowed like the pipeline."""
    T = float(log["duration_s"])
    t = np.arange(0, T, 1.0 / fps)
    E, K = len(ev.emotions), ev.pcs.shape[1]
    scores_raw = rng.normal(0, noise, (len(t), E))
    pcs_raw = rng.normal(0, noise, (len(t), K))
    for st in log.get("sentence_times", []):
        r = ev.lookup(log["story_id"], st["sentence_idx"])
        if r is None:
            continue
        m = (t >= st["start"] + lag) & (t < st["end"] + lag)
        scores_raw[m] += ev.scores[r]
        pcs_raw[m] += ev.pcs[r]
    for e in events or []:
        m = (t >= e["t"] + EVENT_DELAY) & (t < e["t"] + EVENT_DELAY + EVENT_LEN)
        col, sign = align.EVENT_SIGNALS.get(e["label"], ("valence", 1))
        if col == "valence":
            pcs_raw[m, 0] += sign * 4.0 * noise
        elif col[len("score_"):] in ev.emotions:
            scores_raw[m, ev.emotions.index(col[len("score_"):])] += sign * 4.0 * noise
    scores = store.window_mean(t, scores_raw, window)
    pcs = store.window_mean(t, pcs_raw, window)
    cols: dict[str, np.ndarray] = {"frame_index": (t * 30).astype(np.int64), "t": t, "face_found": np.ones(len(t), bool),
                                   "valence_raw": pcs_raw[:, 0], "valence": pcs[:, 0]}
    for k in range(K):
        cols[f"pc{k + 1}"] = pcs[:, k]
    for j, e in enumerate(ev.emotions):
        cols[f"score_{e}"] = scores[:, j]
    top = store.top_emotions(scores, ev.emotions, 3)
    for k in range(3):
        cols[f"top{k + 1}"] = np.array([row[k] for row in top], dtype=str)
    # Tracker records with blendshapes: a smile shape that follows valence, the rest noise.
    frames = []
    for i in range(len(t)):
        v = pcs_raw[i, 0]
        bs = {"mouthSmileLeft": float(np.clip(0.2 + 0.1 * v + rng.normal(0, 0.05), 0, 1)),
              "mouthSmileRight": float(np.clip(0.2 + 0.1 * v + rng.normal(0, 0.05), 0, 1)),
              "browDownLeft": float(np.clip(0.3 - 0.05 * v + rng.normal(0, 0.05), 0, 1)),
              "jawOpen": float(np.clip(rng.normal(0.1, 0.05), 0, 1))}
        frames.append({"i": i, "frame_index": int(cols["frame_index"][i]), "t": round(float(t[i]), 4), "face_found": True,
                       "held": False, "blendshapes": bs, "file": f"f{int(cols['frame_index'][i]):06d}.jpg"})
    return cols, frames


def make_synthetic(out: Path, stories: list[dict] | None = None, text_cols: dict[str, np.ndarray] | None = None,
                   n_sessions: int = 2, takes_per_session: int = 10, lag: float = 0.5, noise: float = 1.0, seed: int = 0,
                   pc_layer: int = 50, score_layer: int = 35, durations: dict[str, list[float]] | None = None) -> dict:
    """Write the whole synthetic data tree; returns the paths and the planned takes."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    if stories is None:
        stories = fake_stories(max(12, takes_per_session * n_sessions // 2 + 4), rng)
    if text_cols is None:
        text_cols = fake_text_ev(stories, rng, layers=(score_layer, pc_layer))
    (out / "stories").mkdir(parents=True, exist_ok=True)
    (out / "stories" / "stories.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stories))
    tpath = out / "results" / "m4" / "text_ev.parquet"
    text_ev.write_table(tpath, text_cols, {"synthetic": "1"})
    from cogmirror.agreement import load_text_ev
    ev = load_text_ev(text_cols, pc_layer, score_layer)
    by_id = {s["id"]: s for s in stories}
    pairs = [(s["id"], c) for s in stories for c in align.CONDITIONS]
    rng.shuffle(pairs)
    pairs = pairs[: n_sessions * takes_per_session]
    takes = []
    k = 0
    for si in range(n_sessions):
        session = f"synth{si + 1}"
        rec = out / "recordings" / session
        rec.mkdir(parents=True, exist_ok=True)
        # calibration take with scripted events
        events = [{"t": 10.0, "type": "event", "label": "smile"}, {"t": 25.0, "type": "event", "label": "frown"},
                  {"t": 40.0, "type": "event", "label": "yawn"}]
        log = take_log(session, 1, None, "calibration", None, rng, events=events)
        _write_take(out, log, ev, rng, lag, noise, events)
        takes.append(log["take_id"])
        for ti in range(takes_per_session):
            if k >= len(pairs):
                break
            sid, cond = pairs[k]
            k += 1
            s = by_id[sid]
            dur = (durations or {}).get(sid) or [len(x.split()) * WORD_S for x in s["sentences"]]
            log = take_log(session, ti + 2, s, cond, dur, rng)
            _write_take(out, log, ev, rng, lag, noise, None)
            takes.append(log["take_id"])
    return {"root": out, "recordings": out / "recordings", "series_root": out / "video", "text_ev": tpath,
            "stories": out / "stories" / "stories.jsonl", "takes": takes, "lag": lag, "noise": noise}


def _write_take(out: Path, log: dict, ev, rng, lag, noise, events):
    rec = out / "recordings" / log["session_id"]
    rec.mkdir(parents=True, exist_ok=True)
    (rec / f"{log['take_id']}.json").write_text(json.dumps(log, indent=1))
    cols, frames = face_series(log, ev, rng, lag, noise, events=events)
    vdir = out / "video" / log["session_id"] / log["take_id"]
    (vdir / "frames").mkdir(parents=True, exist_ok=True)
    store.write_parquet(vdir / f"{log['take_id']}.parquet", cols,
                        {"readout": "last", "score_layer": "35", "valence_layer": "50", "n_pcs": str(N_PCS), "window_s": "1.5",
                         "synthetic": "1", "lag_s": str(lag), "noise": str(noise)})
    (vdir / "frames" / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic_m4"))
    ap.add_argument("--stories", default=str(ROOT / "data/stories/stories.jsonl"), help="use the real story set when it exists")
    ap.add_argument("--text-ev", default=str(ROOT / "results/m4/text_ev.parquet"), help="use the real text EVs when they exist")
    ap.add_argument("--fake", action="store_true", help="fabricate stories and text EVs even if the real ones exist")
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--takes", type=int, default=10, help="takes per session (plus one calibration take)")
    ap.add_argument("--lag", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stories = text_cols = None
    durations = None
    if not args.fake and Path(args.stories).exists() and Path(args.text_ev).exists():
        stories = text_ev.load_stories(args.stories)
        text_cols, _ = text_ev.read_table(args.text_ev)
        durations = {}
        for s in stories:
            idx = Path(args.stories).parent / "audio" / s["id"] / "index.json"
            if idx.exists():
                durations[s["id"]] = json.loads(idx.read_text())["durations_s"]
        print(f"using the real story set ({len(stories)} stories) and text EVs from {args.text_ev}")
    r = make_synthetic(Path(args.out), stories, text_cols, args.sessions, args.takes, args.lag, args.noise, args.seed, durations=durations)
    print(f"wrote {len(r['takes'])} takes under {r['root']} (lag {args.lag:+.1f} s, noise {args.noise})")
    print(f"next: .venv/bin/python scripts/m4_agreement.py --data {r['root']} --series-root {r['series_root']} "
          f"--text-ev {r['text_ev']} --out-dir {r['root'] / 'results' / 'm4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
