"""Alignment of recorded takes with the story text and the face series (M4 part 2).

A take (one story in one condition, recorded by tools/recorder) has a JSON
log with per-sentence timestamps on the recording clock. Segments are
sentences:

- listening: the narration's `playing` / `ended` events of each sentence
- reading: the reader's key presses (sentence shown -> sentence finished);
  optionally refined with WhisperX word timestamps from the audio track
  (`refine="whisperx"`, off by default; whisperx is not a dependency)

Each segment is joined to its sentence's text EV (`text_ev.TextEV`) and to
the face series (the Parquet from `scripts/video_run.py`) aggregated over
the segment's span, with an optional lag applied to the face series:
positive lag means the face trails the text, so the face window for a text
segment [s, e] is [s + lag, e + lag].

No torch, no Modal.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .text_ev import TextEV

CONDITIONS = ("read", "listen")
CALIBRATION = "calibration"
# Recorder constants (tools/recorder/static/app.js), for synthetic takes and sanity checks.
LISTEN_LEAD_IN = 1.5
LISTEN_GAP = 0.5
TAIL = 1.5


# --------------------------------------------------------------------------
# Takes
# --------------------------------------------------------------------------

@dataclass
class Take:
    session_id: str
    take_id: str
    story_id: str | None
    condition: str
    kind: str
    emotions: list[str]
    sentences: list[str]
    sentence_times: list[dict]      # [{sentence_idx, start, end}]
    events: list[dict]
    turn: dict | None
    duration_s: float | None
    bad: bool
    path: Path | None = None
    take_index: int | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def video_path(self) -> Path | None:
        return self.path.with_suffix(".webm") if self.path else None

    @property
    def turn_sentence(self) -> int | None:
        return int(self.turn["sentence_idx"]) if self.turn and self.turn.get("sentence_idx") is not None else None


def load_take(path: str | Path) -> Take:
    path = Path(path)
    log = json.loads(path.read_text())
    return Take(
        session_id=log.get("session_id", path.parent.name),
        take_id=log.get("take_id", path.stem),
        story_id=log.get("story_id"),
        condition=log.get("condition", ""),
        kind=log.get("kind", ""),
        emotions=list(log.get("emotions") or []),
        sentences=list(log.get("sentences") or []),
        sentence_times=list(log.get("sentence_times") or []),
        events=list(log.get("events") or []),
        turn=log.get("turn"),
        duration_s=log.get("duration_s"),
        bad=bool(log.get("bad")),
        path=path,
        take_index=log.get("take_index"),
        note=log.get("note", ""),
        extra={k: log.get(k) for k in ("camera", "recorder", "fake_stream", "start_wallclock") if k in log},
    )


def find_takes(recordings: str | Path, include_bad: bool = False, sessions: list[str] | None = None) -> list[Take]:
    """Every take log under recordings/<session>/<take>.json, sorted."""
    out = []
    for p in sorted(Path(recordings).glob("*/*.json")):
        try:
            t = load_take(p)
        except (json.JSONDecodeError, KeyError):
            continue
        if t.bad and not include_bad:
            continue
        if sessions and t.session_id not in sessions:
            continue
        out.append(t)
    return out


# --------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------

@dataclass
class Segment:
    session_id: str
    take_id: str
    story_id: str
    condition: str
    sentence_idx: int
    start: float
    end: float
    text: str = ""
    turn_starts_here: bool = False
    after_turn: bool = False
    kind: str = ""
    source: str = "log"     # "log" | "whisperx"

    @property
    def duration(self) -> float:
        return self.end - self.start


def _bounds_from_log(take: Take) -> list[tuple[int, float, float]]:
    """(sentence_idx, start, end) from the logged sentence_times; falls back
    to the raw events when sentence_times is absent."""
    rows = []
    if take.sentence_times:
        for st in take.sentence_times:
            if st.get("start") is None or st.get("end") is None:
                continue
            rows.append((int(st["sentence_idx"]), float(st["start"]), float(st["end"])))
        return rows
    starts, ends = {}, {}
    for ev in take.events:
        if ev.get("type") == "sentence_start":
            starts.setdefault(int(ev["sentence_idx"]), float(ev["t"]))
        elif ev.get("type") == "sentence_end":
            ends[int(ev["sentence_idx"])] = float(ev["t"])
    for i in sorted(starts):
        if i in ends:
            rows.append((i, starts[i], ends[i]))
    return rows


def segments_from_take(take: Take, refine: str | None = None, words: list[dict] | None = None,
                       min_duration: float = 0.3) -> list[Segment]:
    """Sentence segments of one take.

    `refine="whisperx"` (reading condition only) replaces the key-press
    boundaries by word timestamps: `words` (from `whisperx_words`, or any
    list of {"word", "start", "end"}) or, when not given, a WhisperX run on
    the take's audio. Segments shorter than `min_duration` are dropped.
    """
    if take.story_id is None or take.condition not in CONDITIONS:
        return []
    bounds = _bounds_from_log(take)
    source = "log"
    if refine == "whisperx" and take.condition == "read":
        if words is None:
            words = whisperx_words(take.video_path)
        refined = refine_reading_bounds(bounds, words, take.sentences)
        if refined:
            bounds, source = refined, "whisperx"
    turn = take.turn_sentence
    out = []
    for i, s, e in bounds:
        if e - s < min_duration:
            continue
        out.append(Segment(take.session_id, take.take_id, take.story_id, take.condition, i, s, e,
                           text=take.sentences[i] if i < len(take.sentences) else "",
                           turn_starts_here=turn is not None and i == turn,
                           after_turn=turn is not None and i >= turn, kind=take.kind, source=source))
    return out


# --------------------------------------------------------------------------
# WhisperX refinement (optional)
# --------------------------------------------------------------------------

def _norm_word(w: str) -> str:
    return "".join(ch for ch in w.lower() if ch.isalnum())


def refine_reading_bounds(bounds: list[tuple[int, float, float]], words: list[dict], sentences: list[str],
                          slack: float = 1.5) -> list[tuple[int, float, float]] | None:
    """Sentence bounds from word timestamps, anchored to the key presses.

    Words are assigned to sentences in order by matching their text against
    the sentence words; a sentence's refined start is its first matched
    word's start and its end the last matched word's end, each kept within
    `slack` seconds of the logged boundary (a wrong match further away is
    discarded). Returns None when fewer than half the sentences get both
    bounds, so the caller keeps the log's timing.
    """
    if not words or not bounds:
        return None
    # Flatten the sentences into a word list with sentence ids.
    target = []
    for i, s in enumerate(sentences):
        for w in s.split():
            n = _norm_word(w)
            if n:
                target.append((i, n))
    if not target:
        return None
    # Greedy in-order matching: each transcript word is matched to the next
    # unmatched target word if the text agrees, searching a few words ahead.
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    pos = 0
    for w in words:
        n = _norm_word(str(w.get("word", "")))
        if not n or w.get("start") is None or w.get("end") is None:
            continue
        for look in range(pos, min(pos + 4, len(target))):
            if target[look][1] == n:
                i = target[look][0]
                first.setdefault(i, float(w["start"]))
                last[i] = float(w["end"])
                pos = look + 1
                break
    out = []
    ok = 0
    for i, s, e in bounds:
        ns, ne = s, e
        if i in first and abs(first[i] - s) <= slack:
            ns = first[i]
        if i in last and abs(last[i] - e) <= slack:
            ne = last[i]
        if ns != s and ne != e:
            ok += 1
        if ne <= ns:
            ns, ne = s, e
        out.append((i, ns, ne))
    if ok < max(1, len(bounds) // 2):
        return None
    return out


def whisperx_words(audio_path: str | Path | None, model_name: str = "small", device: str = "cpu",
                   language: str = "en") -> list[dict]:
    """Word timestamps from WhisperX on the take's audio track. Imports
    whisperx lazily; raises ImportError with the install hint when absent."""
    try:
        import whisperx  # type: ignore
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("WhisperX refinement needs `pip install whisperx` (and ffmpeg)") from e
    if audio_path is None or not Path(audio_path).exists():
        raise FileNotFoundError(f"audio/video file not found: {audio_path}")
    audio = whisperx.load_audio(str(audio_path))
    model = whisperx.load_model(model_name, device, compute_type="int8" if device == "cpu" else "float16", language=language)
    result = model.transcribe(audio, batch_size=8)
    align_model, meta = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(result["segments"], align_model, meta, audio, device, return_char_alignments=False)
    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                              "score": float(w.get("score", 0.0))})
    return words


# --------------------------------------------------------------------------
# Face series per segment
# --------------------------------------------------------------------------

def series_columns(cols: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
    """(pc columns in order, score columns in emotion order) of a face series."""
    pcs = sorted((c for c in cols if c.startswith("pc") and c[2:].isdigit()), key=lambda c: int(c[2:]))
    scores = [c for c in cols if c.startswith("score_")]
    return pcs, scores


def segment_means(t: np.ndarray, x: np.ndarray, starts: np.ndarray, ends: np.ndarray, lag: float = 0.0,
                  min_frames: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Mean of x [N, D] over rows with start + lag <= t < end + lag, per
    segment -> ([S, D], frames used [S]); NaN where fewer than `min_frames`
    finite rows fall in the span."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    starts = np.asarray(starts, dtype=np.float64) + lag
    ends = np.asarray(ends, dtype=np.float64) + lag
    out = np.full((len(starts), x.shape[1]), np.nan)
    n = np.zeros(len(starts), dtype=np.int64)
    lo = np.searchsorted(t, starts, side="left")
    hi = np.searchsorted(t, ends, side="left")
    for k in range(len(starts)):
        rows = x[lo[k]:hi[k]]
        if len(rows) == 0:
            continue
        finite = np.all(np.isfinite(rows), axis=1)
        n[k] = int(finite.sum())
        if n[k] >= min_frames:
            out[k] = rows[finite].mean(axis=0)
    return out, n


@dataclass
class FaceSegments:
    pcs: np.ndarray         # [S, K]
    scores: np.ndarray      # [S, E]
    n_frames: np.ndarray    # [S]
    emotions: list[str]
    lag: float


def face_segment_features(series: dict[str, np.ndarray], segments: list[Segment], lag: float = 0.0,
                          raw_valence: bool = False, min_frames: int = 1) -> FaceSegments:
    """Aggregate a take's face series over each segment's span.

    Uses the series' windowed columns (`pc*`, `score_*`); with
    `raw_valence=True` PC1 comes from `valence_raw` (unwindowed) instead.
    """
    pc_cols, score_cols = series_columns(series)
    if not pc_cols:
        raise ValueError("series has no pc columns")
    t = np.asarray(series["t"], dtype=np.float64)
    pcs = np.stack([np.asarray(series[c], dtype=np.float64) for c in pc_cols], axis=1)
    if raw_valence and "valence_raw" in series:
        pcs[:, 0] = np.asarray(series["valence_raw"], dtype=np.float64)
    scores = np.stack([np.asarray(series[c], dtype=np.float64) for c in score_cols], axis=1) if score_cols \
        else np.zeros((len(t), 0))
    starts = np.array([s.start for s in segments])
    ends = np.array([s.end for s in segments])
    x = np.concatenate([pcs, scores], axis=1)
    m, n = segment_means(t, x, starts, ends, lag=lag, min_frames=min_frames)
    return FaceSegments(m[:, : pcs.shape[1]], m[:, pcs.shape[1]:], n, [c[len("score_"):] for c in score_cols], lag)


# --------------------------------------------------------------------------
# Joining text and face
# --------------------------------------------------------------------------

def text_segment_features(ev: TextEV, segments: list[Segment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Text PCs [S, K] and scores [S, E] for each segment's sentence, plus a
    mask of segments whose sentence exists in the table."""
    pcs = np.full((len(segments), ev.pcs.shape[1]), np.nan)
    scores = np.full((len(segments), ev.scores.shape[1]), np.nan)
    have = np.zeros(len(segments), dtype=bool)
    for k, s in enumerate(segments):
        r = ev.lookup(s.story_id, s.sentence_idx)
        if r is not None:
            pcs[k], scores[k], have[k] = ev.pcs[r], ev.scores[r], True
    return pcs, scores, have


def series_path(series_root: str | Path, take: Take) -> Path | None:
    """Where `scripts/video_run.py` leaves a take's series: tries
    <root>/<session>/<take>/<take>.parquet, then <root>/<take>/<take>.parquet."""
    root = Path(series_root)
    for p in (root / take.session_id / take.take_id / f"{take.take_id}.parquet", root / take.take_id / f"{take.take_id}.parquet"):
        if p.exists():
            return p
    return None


def frames_manifest_path(series_root: str | Path, take: Take) -> Path | None:
    """The tracker's frames.jsonl next to the series (holds the blendshapes)."""
    p = series_path(series_root, take)
    if p is None:
        return None
    f = p.parent / "frames" / "frames.jsonl"
    return f if f.exists() else None


def blendshape_matrix(frames: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(t [N], blendshapes [N, B], names) from the tracker's frame records; NaN rows without a detection."""
    names: list[str] = []
    for r in frames:
        if r.get("blendshapes"):
            names = sorted(r["blendshapes"])
            break
    t = np.array([float(r["t"]) for r in frames])
    x = np.full((len(frames), len(names)), np.nan)
    for k, r in enumerate(frames):
        bs = r.get("blendshapes")
        if bs:
            x[k] = [bs.get(n, np.nan) for n in names]
    return t, x, names


def read_frames_manifest(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------
# Scripted events (recorder calibration or a dedicated take)
# --------------------------------------------------------------------------

EVENT_TYPES = ("event", "marker", "scripted_event")
# Which signal in the face series an event should move, and in which direction.
EVENT_SIGNALS: dict[str, tuple[str, int]] = {
    "smile": ("valence", +1),
    "laugh": ("valence", +1),
    "frown": ("valence", -1),
    "sad": ("valence", -1),
    "angry": ("score_angry", +1),
    "surprise": ("score_surprised", +1),
    "surprised": ("score_surprised", +1),
    "yawn": ("score_tired", +1),
    "tired": ("score_tired", +1),
    "fear": ("score_afraid", +1),
    "afraid": ("score_afraid", +1),
    "disgust": ("score_disgusted", +1),
    "disgusted": ("score_disgusted", +1),
}


@dataclass
class ScriptedEvent:
    t: float
    label: str
    end: float | None = None


def scripted_events_from_log(log: dict | Take) -> list[ScriptedEvent]:
    """Marked events in a take log: entries of `events` with type `event`,
    `marker` or `scripted_event` (a `label` and `t`, optional `end`), or
    `<label>_start` / `<label>_end` pairs other than sentence and calibration."""
    events = log.events if isinstance(log, Take) else list(log.get("events") or [])
    out: list[ScriptedEvent] = []
    pending: dict[str, float] = {}
    for ev in events:
        typ = str(ev.get("type", ""))
        if typ in EVENT_TYPES and "label" in ev:
            out.append(ScriptedEvent(float(ev["t"]), str(ev["label"]).strip().lower(), ev.get("end")))
        elif typ.endswith("_start") and not typ.startswith(("sentence", "calibration")):
            pending[typ[: -len("_start")]] = float(ev["t"])
        elif typ.endswith("_end") and not typ.startswith(("sentence", "calibration")):
            label = typ[: -len("_end")]
            if label in pending:
                out.append(ScriptedEvent(pending.pop(label), label.lower(), float(ev["t"])))
    for label, t in pending.items():
        out.append(ScriptedEvent(t, label.lower()))
    return sorted(out, key=lambda e: e.t)


def scripted_events_from_csv(path: str | Path) -> list[ScriptedEvent]:
    """A CSV with columns t,label[,end] (header optional), times in seconds on the recording clock."""
    out = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                t = float(row[0])
            except ValueError:
                continue    # header
            label = row[1].strip().lower() if len(row) > 1 else "event"
            end = float(row[2]) if len(row) > 2 and row[2].strip() else None
            out.append(ScriptedEvent(t, label, end))
    return sorted(out, key=lambda e: e.t)


def event_signal(label: str, series: dict[str, np.ndarray]) -> tuple[str, int] | None:
    """(column, expected sign) for an event label, or None when unknown."""
    key = label.strip().lower()
    for name, sig in EVENT_SIGNALS.items():
        if key == name or key.startswith(name):
            return sig if sig[0] in series else None
    return None


def event_response(t: np.ndarray, x: np.ndarray, t_event: float, sign: int = +1, pre: tuple[float, float] = (-3.0, -0.5),
                   search: tuple[float, float] = (-1.0, 3.0), z_threshold: float = 2.0, noise_sd: float | None = None,
                   min_run: int = 2) -> dict:
    """Did signal x respond to an event at `t_event`?

    Baseline: mean and SD of x over [t_event + pre[0], t_event + pre[1]]
    (`noise_sd` overrides the SD, e.g. from the calibration clip). Onset:
    first time in [t_event + search[0], t_event + search[1]] at which
    sign * (x - baseline) exceeds z_threshold SDs for `min_run` consecutive
    samples. Reports the onset lag relative to the event and the peak z.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    pre_mask = (t >= t_event + pre[0]) & (t < t_event + pre[1]) & np.isfinite(x)
    out = {"t_event": float(t_event), "n_pre": int(pre_mask.sum()), "onset": float("nan"), "lag": float("nan"),
           "peak_z": float("nan"), "peak_t": float("nan"), "within_1s": False, "sign": int(sign)}
    if pre_mask.sum() < 2:
        return out
    base = x[pre_mask].mean()
    sd = float(noise_sd) if noise_sd else float(x[pre_mask].std())
    sd = max(sd, 1e-9)
    out.update(baseline=float(base), sd=sd)
    win = (t >= t_event + search[0]) & (t <= t_event + search[1]) & np.isfinite(x)
    if not win.any():
        return out
    z = sign * (x - base) / sd
    zw = z[win]
    tw = t[win]
    k = int(np.argmax(zw))
    out["peak_z"], out["peak_t"] = float(zw[k]), float(tw[k])
    above = zw > z_threshold
    run = 0
    for i in range(len(zw)):
        run = run + 1 if above[i] else 0
        if run >= min_run:
            onset = float(tw[i - min_run + 1])
            out["onset"], out["lag"] = onset, onset - t_event
            out["within_1s"] = abs(onset - t_event) <= 1.0
            break
    return out
