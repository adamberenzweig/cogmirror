"""Text–face agreement metrics (M4 part 3; design doc, Gate 3 procedure).

Data: per-sentence segments of recorded takes (`align.py`), each with the
face readout aggregated over its span and its sentence's text EV. The
analysis:

1. split: development = every session except the held-out one and every
   story except the held-out set (`SplitConfig`, kept in a JSON file)
2. z-score each modality per dimension on development segments
3. lag: one value chosen on development data from a grid, then fixed
4. metrics on held-out (and, labelled, development) segments per condition:
   - per-component correlation across segments (valence = PC1 first)
   - per-segment cosine and Spearman between the normalized 171-score profiles
   - retrieval rank of the true text segment among the condition's segments
5. nulls that respect autocorrelation: story-level permutation (every face
   segment of story A is paired with the proportional sentence of a
   permuted partner story) and within-story circular shift; one-sided
   p-values
6. baselines: neutral stories; blendshape ridge regression (by story);
   the verbal baseline is not in the video store and is reported as absent
7. the lightly trained variant: a ridge map from face scores to text PCs,
   fitted on development data, reported apart from the zero-training numbers
8. turn check: in stories with a scripted turn, does the face valence step
   with the text's, more than under circular shifts of the face series?

Everything is numpy; scipy is used for Spearman only when present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import align
from .align import Segment, Take
from .text_ev import STORY_ROW, TextEV, table_at_layer

CONDITIONS = align.CONDITIONS
DEFAULT_LAG_GRID = np.round(np.arange(-3.0, 3.0001, 0.5), 2)
DEFAULT_N_PERM = 1000
MIN_SEGMENTS = 4        # below this, correlations are reported as NaN


# --------------------------------------------------------------------------
# Split configuration
# --------------------------------------------------------------------------

@dataclass
class SplitConfig:
    holdout_session: str | None
    holdout_stories: list[str]
    seed: int = 0
    note: str = ""

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"holdout_session": self.holdout_session, "holdout_stories": sorted(self.holdout_stories),
                                    "seed": self.seed, "note": self.note}, indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SplitConfig":
        d = json.loads(Path(path).read_text())
        return cls(d.get("holdout_session"), list(d.get("holdout_stories") or []), int(d.get("seed", 0)), d.get("note", ""))


def make_split(takes: list[Take], stories: list[dict], seed: int = 0, story_fraction: float = 0.2,
               holdout_session: str | None = "latest") -> SplitConfig:
    """A deterministic split: the latest session (by wallclock, else by id)
    and a stratified-by-kind sample of the stories are held out. With a
    single session no session is held out (said in the note)."""
    sessions = sorted({t.session_id for t in takes})
    if holdout_session == "latest":
        if len(sessions) >= 2:
            by_start = {}
            for t in takes:
                by_start[t.session_id] = max(by_start.get(t.session_id, ""), str(t.extra.get("start_wallclock") or ""))
            holdout_session = max(sessions, key=lambda s: (by_start.get(s, ""), s))
        else:
            holdout_session = None
    rng = np.random.default_rng(seed)
    held = []
    by_kind: dict[str, list[str]] = {}
    for s in stories:
        by_kind.setdefault(s.get("kind", ""), []).append(s["id"])
    for kind in sorted(by_kind):
        ids = sorted(by_kind[kind])
        n = int(round(story_fraction * len(ids)))
        held.extend(sorted(rng.choice(ids, size=n, replace=False).tolist()) if n else [])
    note = (f"{len(sessions)} session(s) seen when the split was made; " +
            ("no session held out (only one recorded)" if holdout_session is None else f"session {holdout_session} held out") +
            f"; {len(held)} of {len(stories)} stories held out, stratified by kind, seed {seed}")
    return SplitConfig(holdout_session, sorted(held), seed, note)


# --------------------------------------------------------------------------
# Text features at the face series' layers
# --------------------------------------------------------------------------

def load_text_ev(cols: dict[str, np.ndarray], pc_layer: int, score_layer: int) -> TextEV:
    """Text EVs with PCs from `pc_layer` and scores from `score_layer`
    (the face series uses two layers too: valence at 50, scores at 35)."""
    ev = table_at_layer(cols, pc_layer)
    if score_layer != pc_layer:
        sc = table_at_layer(cols, score_layer)
        scores = np.full_like(ev.scores, np.nan)
        for key, r in ev.index.items():
            r2 = sc.lookup(*key)
            if r2 is not None:
                scores[r] = sc.scores[r2]
        ev = TextEV(ev.layer, sc.emotions, ev.index, scores, ev.pcs, ev.story_ids, ev.sentence_idx)
    return ev


# --------------------------------------------------------------------------
# Collecting takes
# --------------------------------------------------------------------------

@dataclass
class TakeData:
    take: Take
    segments: list[Segment]
    series: dict[str, np.ndarray]
    series_meta: dict[str, str]
    text_pcs: np.ndarray        # [S, K]
    text_scores: np.ndarray     # [S, E]
    frames: list[dict] | None = None    # tracker manifest with blendshapes
    text_emotions: list[str] = field(default_factory=list)


def collect_takes(takes: list[Take], series_root: str | Path, ev: TextEV, refine: str | None = None,
                  blendshapes: bool = True, progress=None) -> tuple[list[TakeData], list[dict]]:
    """Load every take's series and segments; returns (data, skipped notes)."""
    from .store import read_parquet

    data, skipped = [], []
    for t in takes:
        if t.condition not in CONDITIONS or t.story_id is None:
            continue
        p = align.series_path(series_root, t)
        if p is None:
            skipped.append({"take": t.take_id, "reason": "no face series (run scripts/video_run.py / m4_run_videos.py)"})
            continue
        segs = align.segments_from_take(t, refine=refine)
        if not segs:
            skipped.append({"take": t.take_id, "reason": "no segments in the log"})
            continue
        cols, meta = read_parquet(p)
        pcs, scores, have = align.text_segment_features(ev, segs)
        if not have.all():
            segs = [s for s, h in zip(segs, have) if h]
            pcs, scores = pcs[have], scores[have]
            skipped.append({"take": t.take_id, "reason": f"{int((~have).sum())} segments without a text EV row (dropped)"})
        frames = None
        if blendshapes:
            fp = align.frames_manifest_path(series_root, t)
            if fp is not None:
                frames = align.read_frames_manifest(fp)
        data.append(TakeData(t, segs, cols, meta, pcs, scores, frames, list(ev.emotions)))
        if progress:
            progress(f"  {t.take_id}: {len(segs)} segments, {len(cols['t'])} frames")
    return data, skipped


# --------------------------------------------------------------------------
# The assembled dataset at one lag
# --------------------------------------------------------------------------

@dataclass
class Aligned:
    session: np.ndarray         # [S] str
    take: np.ndarray            # [S] str
    story: np.ndarray           # [S] str
    condition: np.ndarray       # [S] str
    kind: np.ndarray            # [S] str
    sentence_idx: np.ndarray    # [S]
    start: np.ndarray           # [S]
    end: np.ndarray             # [S]
    after_turn: np.ndarray      # [S] bool
    has_turn: np.ndarray        # [S] bool
    face_pcs: np.ndarray        # [S, K]
    face_scores: np.ndarray     # [S, E]
    face_n_frames: np.ndarray   # [S]
    text_pcs: np.ndarray        # [S, K]
    text_scores: np.ndarray     # [S, E]
    blend: np.ndarray | None    # [S, B] or None
    blend_names: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    lag: float = 0.0

    def __len__(self) -> int:
        return len(self.story)

    def valid(self) -> np.ndarray:
        return np.isfinite(self.face_pcs).all(axis=1) & np.isfinite(self.text_pcs).all(axis=1)

    def subset(self, mask: np.ndarray) -> "Aligned":
        mask = np.asarray(mask, bool)
        return Aligned(**{k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(self) else v)
                          for k, v in self.__dict__.items()})


def assemble(data: list[TakeData], lag: float = 0.0, raw_valence: bool = False, ev_emotions: list[str] | None = None) -> Aligned:
    """Face features at `lag` for every segment of every take, joined with the
    text features. `ev_emotions` is the text table's emotion order (needed
    when it differs from the face series' column order)."""
    rows: dict[str, list] = {k: [] for k in ("session", "take", "story", "condition", "kind", "sentence_idx", "start", "end",
                                             "after_turn", "has_turn")}
    fp, fs, fn, tp, ts, bl = [], [], [], [], [], []
    emotions: list[str] = []
    blend_names: list[str] = []
    any_blend = any(d.frames for d in data)
    k_pcs = None
    text_order: np.ndarray | None = None
    for d in data:
        f = align.face_segment_features(d.series, d.segments, lag=lag, raw_valence=raw_valence)
        if not emotions and f.emotions:
            emotions = f.emotions
            # Text scores follow the text table's emotion order; put them in the face series' order.
            if ev_emotions:
                text_order = np.array([ev_emotions.index(e) if e in ev_emotions else -1 for e in emotions])
                if (text_order < 0).any():
                    raise ValueError("face series has emotions missing from the text table: "
                                     + ", ".join(e for e, k in zip(emotions, text_order) if k < 0))
        k_pcs = f.pcs.shape[1] if k_pcs is None else min(k_pcs, f.pcs.shape[1])
        for s in d.segments:
            rows["session"].append(s.session_id)
            rows["take"].append(s.take_id)
            rows["story"].append(s.story_id)
            rows["condition"].append(s.condition)
            rows["kind"].append(s.kind)
            rows["sentence_idx"].append(s.sentence_idx)
            rows["start"].append(s.start)
            rows["end"].append(s.end)
            rows["after_turn"].append(s.after_turn)
            rows["has_turn"].append(d.take.turn_sentence is not None)
        fp.append(f.pcs)
        fs.append(f.scores)
        fn.append(f.n_frames)
        tp.append(d.text_pcs)
        ts.append(d.text_scores[:, text_order] if text_order is not None and d.text_scores.shape[1] else d.text_scores)
        if any_blend:
            if d.frames:
                t, x, names = align.blendshape_matrix(d.frames)
                blend_names = blend_names or names
                m, _ = align.segment_means(t, x, [s.start for s in d.segments], [s.end for s in d.segments], lag=lag)
            else:
                m = np.full((len(d.segments), len(blend_names) or 1), np.nan)
            bl.append(m)
    face_pcs = np.concatenate([x[:, :k_pcs] for x in fp]) if fp else np.zeros((0, 0))
    text_pcs = np.concatenate(tp) if tp else np.zeros((0, 0))
    k = min(face_pcs.shape[1], text_pcs.shape[1]) if len(fp) else 0
    blend = None
    if any_blend and bl:
        width = max(b.shape[1] for b in bl)
        blend = np.concatenate([b if b.shape[1] == width else np.full((len(b), width), np.nan) for b in bl])
    return Aligned(
        session=np.asarray(rows["session"], dtype=str), take=np.asarray(rows["take"], dtype=str),
        story=np.asarray(rows["story"], dtype=str), condition=np.asarray(rows["condition"], dtype=str),
        kind=np.asarray(rows["kind"], dtype=str), sentence_idx=np.asarray(rows["sentence_idx"], dtype=np.int64),
        start=np.asarray(rows["start"], dtype=np.float64), end=np.asarray(rows["end"], dtype=np.float64),
        after_turn=np.asarray(rows["after_turn"], dtype=bool), has_turn=np.asarray(rows["has_turn"], dtype=bool),
        face_pcs=face_pcs[:, :k], face_scores=np.concatenate(fs) if fs else np.zeros((0, 0)),
        face_n_frames=np.concatenate(fn) if fn else np.zeros(0, dtype=np.int64),
        text_pcs=text_pcs[:, :k], text_scores=np.concatenate(ts) if ts else np.zeros((0, 0)),
        blend=blend, blend_names=blend_names, emotions=emotions, lag=float(lag))


def split_masks(a: Aligned, cfg: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """(development, held-out) masks. Held-out = held-out session or held-out story."""
    held = np.isin(a.story, np.asarray(cfg.holdout_stories, dtype=str))
    if cfg.holdout_session:
        held |= a.session == cfg.holdout_session
    return ~held, held


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

@dataclass
class ZScore:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "ZScore":
        x = np.asarray(x, dtype=np.float64)
        ok = np.isfinite(x).all(axis=1)
        if ok.sum() < 2:
            return cls(np.zeros(x.shape[1]), np.ones(x.shape[1]))
        m = x[ok].mean(axis=0)
        s = x[ok].std(axis=0)
        return cls(m, np.where(s > 1e-12, s, 1.0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std


@dataclass
class Normalized:
    """Both modalities z-scored with statistics fitted on the development segments."""
    face_pcs: np.ndarray
    text_pcs: np.ndarray
    face_scores: np.ndarray
    text_scores: np.ndarray


def normalize(a: Aligned, dev: np.ndarray) -> Normalized:
    dev = np.asarray(dev, bool)
    return Normalized(
        ZScore.fit(a.face_pcs[dev])(a.face_pcs), ZScore.fit(a.text_pcs[dev])(a.text_pcs),
        ZScore.fit(a.face_scores[dev])(a.face_scores), ZScore.fit(a.text_scores[dev])(a.text_scores))


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_SEGMENTS:
        return float("nan")
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 0 else float("nan")


def _ranks(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata
        return rankdata(x)
    except ImportError:  # pragma: no cover
        order = np.argsort(x)
        r = np.empty(len(x))
        r[order] = np.arange(1, len(x) + 1)
        return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_SEGMENTS:
        return float("nan")
    return pearson(_ranks(x[ok]), _ranks(y[ok]))


def component_correlations(face: np.ndarray, text: np.ndarray) -> dict:
    """Pearson and Spearman r across segments for each component (column)."""
    k = min(face.shape[1], text.shape[1])
    return {"pearson": [pearson(face[:, j], text[:, j]) for j in range(k)],
            "spearman": [spearman(face[:, j], text[:, j]) for j in range(k)],
            "n": int((np.isfinite(face[:, :k]).all(axis=1) & np.isfinite(text[:, :k]).all(axis=1)).sum())}


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def per_segment_similarity(face: np.ndarray, text: np.ndarray) -> dict:
    """Mean cosine and mean Spearman between each segment's normalized face and text profiles."""
    ok = np.isfinite(face).all(axis=1) & np.isfinite(text).all(axis=1)
    if ok.sum() == 0 or face.shape[1] < 3:
        return {"cosine": float("nan"), "spearman": float("nan"), "n": int(ok.sum())}
    f, t = face[ok], text[ok]
    cos = np.sum(_unit(f) * _unit(t), axis=1)
    sp = np.array([pearson(_ranks(a), _ranks(b)) for a, b in zip(f, t)]) if f.shape[1] >= MIN_SEGMENTS else np.full(len(f), np.nan)
    return {"cosine": float(cos.mean()), "spearman": float(np.nanmean(sp)) if np.isfinite(sp).any() else float("nan"),
            "n": int(ok.sum()), "cosine_per_segment": cos}


def retrieval(face: np.ndarray, text: np.ndarray) -> dict:
    """Rank of each face segment's true text among all text segments, by
    cosine between normalized profiles. Ties (identical text rows) count in
    the face segment's favour. Normalized rank is (rank - 1) / (n - 1):
    0.5 is chance."""
    ok = np.isfinite(face).all(axis=1) & np.isfinite(text).all(axis=1)
    n = int(ok.sum())
    if n < 2:
        return {"n": n, "median_rank": float("nan"), "mean_norm_rank": float("nan"), "top1": float("nan"),
                "top5": float("nan"), "mrr": float("nan")}
    sim = _unit(face[ok]) @ _unit(text[ok]).T
    true = np.diag(sim)
    rank = 1 + (sim > true[:, None]).sum(axis=1)
    return {"n": n, "median_rank": float(np.median(rank)), "mean_norm_rank": float(((rank - 1) / (n - 1)).mean()),
            "top1": float((rank == 1).mean()), "top5": float((rank <= 5).mean()), "mrr": float((1.0 / rank).mean()),
            "ranks": rank}


def turn_statistic(face_valence: np.ndarray, text_valence: np.ndarray, story: np.ndarray, after: np.ndarray,
                   has_turn: np.ndarray) -> dict:
    """For each story with a turn: face step (mean after - mean before) times the
    sign of the text step; the statistic is the mean signed face step and the
    fraction of stories where the signs agree."""
    steps, agree = [], []
    for s in np.unique(story[has_turn]):
        m = story == s
        before, aft = m & ~after, m & after
        if before.sum() == 0 or aft.sum() == 0:
            continue
        fv, tv = face_valence, text_valence
        f_step = np.nanmean(fv[aft]) - np.nanmean(fv[before])
        t_step = np.nanmean(tv[aft]) - np.nanmean(tv[before])
        if not (np.isfinite(f_step) and np.isfinite(t_step)) or t_step == 0:
            continue
        steps.append(np.sign(t_step) * f_step)
        agree.append(np.sign(t_step) == np.sign(f_step))
    if not steps:
        return {"n_stories": 0, "mean_signed_step": float("nan"), "sign_agreement": float("nan")}
    return {"n_stories": len(steps), "mean_signed_step": float(np.mean(steps)), "sign_agreement": float(np.mean(agree))}


# --------------------------------------------------------------------------
# Nulls
# --------------------------------------------------------------------------

def _story_rows(story: np.ndarray, sentence_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Row indices of each story, in sentence order."""
    out = {}
    for s in np.unique(story):
        rows = np.flatnonzero(story == s)
        out[s] = rows[np.argsort(sentence_idx[rows], kind="stable")]
    return out


def story_permutation(story: np.ndarray, sentence_idx: np.ndarray, rng: np.random.Generator,
                      take: np.ndarray | None = None) -> np.ndarray:
    """Index array p such that text row p[k] replaces the true text of face
    row k: the stories are permuted among themselves and each face segment
    takes the partner story's sentence at the same relative position. When
    a story has several takes (`take` given), each take is permuted as its
    own unit so that units stay comparable in length."""
    unit = story if take is None else np.char.add(np.char.add(story.astype(str), "|"), take.astype(str))
    rows = _story_rows(unit, sentence_idx)
    units = sorted(rows)
    perm = rng.permutation(len(units))
    p = np.arange(len(story))
    for i, u in enumerate(units):
        src = rows[u]
        dst = rows[units[perm[i]]]
        m, n = len(src), len(dst)
        pos = np.arange(m) * (n - 1) / max(m - 1, 1) if m > 1 else np.zeros(1)
        p[src] = dst[np.round(pos).astype(int)]
    return p


def circular_shift(story: np.ndarray, sentence_idx: np.ndarray, rng: np.random.Generator,
                   take: np.ndarray | None = None, min_shift: int = 1) -> np.ndarray:
    """Index array: within each story (or take), the text rows rotated by a
    random non-zero offset."""
    unit = story if take is None else np.char.add(np.char.add(story.astype(str), "|"), take.astype(str))
    rows = _story_rows(unit, sentence_idx)
    p = np.arange(len(story))
    for u, src in rows.items():
        n = len(src)
        if n <= min_shift:
            continue
        k = int(rng.integers(min_shift, n))
        p[src] = src[(np.arange(n) + k) % n]
    return p


def p_value(observed: float, null: np.ndarray, greater: bool = True) -> float:
    """One-sided permutation p-value with the +1 correction; NaN if the observed value is NaN."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return float("nan")
    hits = (null >= observed).sum() if greater else (null <= observed).sum()
    return float((1 + hits) / (1 + len(null)))


# --------------------------------------------------------------------------
# Lag
# --------------------------------------------------------------------------

def estimate_lag(data: list[TakeData], dev_fn, grid=DEFAULT_LAG_GRID, condition: str | None = None, tol: float = 0.01) -> dict:
    """Valence correlation on development segments at each lag; the chosen
    lag is the middle of the plateau within `tol` of the best r.
    `dev_fn(aligned) -> dev mask`."""
    curve = []
    for lag in grid:
        a = assemble(data, lag=float(lag), ev_emotions=data[0].text_emotions if data else None)
        dev = dev_fn(a)
        if condition:
            dev = dev & (a.condition == condition)
        sub = a.subset(dev)
        if len(sub) < MIN_SEGMENTS:
            curve.append({"lag": float(lag), "r": float("nan"), "n": int(len(sub))})
            continue
        nz = normalize(sub, np.ones(len(sub), bool))
        curve.append({"lag": float(lag), "r": pearson(nz.face_pcs[:, 0], nz.text_pcs[:, 0]), "n": int(len(sub))})
    rs = np.array([c["r"] for c in curve])
    if np.isfinite(rs).any():
        # The curve is flat near its top (segments are seconds long, the series is
        # windowed), so take the middle of the plateau within `tol` of the maximum
        # rather than the exact argmax.
        top = np.nanmax(rs)
        plateau = [c["lag"] for c, r in zip(curve, rs) if np.isfinite(r) and r >= top - tol]
        best = float(np.median(plateau))
        if best not in plateau:        # an even plateau: the grid point nearest the median
            best = min(plateau, key=lambda x: abs(x - best))
    else:
        best = 0.0
    return {"grid": [float(g) for g in grid], "curve": curve, "lag": float(best), "plateau_tol": tol}


# --------------------------------------------------------------------------
# Ridge map (the lightly trained variant, and the blendshape baseline)
# --------------------------------------------------------------------------

@dataclass
class Ridge:
    w: np.ndarray       # [D, K]
    x_mean: np.ndarray
    y_mean: np.ndarray
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, float) - self.x_mean) @ self.w + self.y_mean


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    d = x.shape[1]
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ yc)
    return Ridge(w, xm, ym, float(alpha))


def ridge_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas=(0.1, 1, 10, 100, 1000, 10000), n_folds: int = 5,
             seed: int = 0) -> dict:
    """Group (story) K-fold on the development data; picks the alpha with the
    best mean valence correlation of out-of-fold predictions."""
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y, groups = x[ok], y[ok], groups[ok]
    ug = np.unique(groups)
    if len(ug) < 2 or len(x) < MIN_SEGMENTS:
        return {"alpha": float(alphas[len(alphas) // 2]), "cv": [], "n": int(len(x))}
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(ug)) % min(n_folds, len(ug))
    fold_of = {g: folds[i] for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups])
    out = []
    for alpha in alphas:
        pred = np.full_like(y, np.nan)
        for k in np.unique(f):
            tr, te = f != k, f == k
            if tr.sum() < 2:
                continue
            pred[te] = fit_ridge(x[tr], y[tr], alpha).predict(x[te])
        out.append({"alpha": float(alpha), "r_valence": pearson(pred[:, 0], y[:, 0]),
                    "r_mean": float(np.nanmean([pearson(pred[:, j], y[:, j]) for j in range(y.shape[1])]))})
    rs = np.array([o["r_valence"] for o in out])
    best = out[int(np.nanargmax(rs))]["alpha"] if np.isfinite(rs).any() else float(alphas[len(alphas) // 2])
    return {"alpha": float(best), "cv": out, "n": int(len(x)), "n_groups": int(len(ug))}


# --------------------------------------------------------------------------
# The evaluation of one subset
# --------------------------------------------------------------------------

def evaluate(a: Aligned, nz: Normalized, mask: np.ndarray, n_perm: int = DEFAULT_N_PERM, seed: int = 0,
             pred_pcs: np.ndarray | None = None) -> dict:
    """All metrics and both nulls on the masked segments.

    `pred_pcs` (e.g. a ridge prediction of the text PCs from the face)
    replaces the face PCs and scores in the comparison when given.
    """
    mask = np.asarray(mask, bool) & a.valid()
    sub = a.subset(mask)
    f_pcs = (pred_pcs[mask] if pred_pcs is not None else nz.face_pcs[mask])
    t_pcs = nz.text_pcs[mask]
    f_sc = (pred_pcs[mask] if pred_pcs is not None else nz.face_scores[mask])
    t_sc = (nz.text_pcs[mask] if pred_pcs is not None else nz.text_scores[mask])
    n = int(len(sub))
    out: dict = {"n_segments": n, "n_stories": int(len(np.unique(sub.story))), "n_takes": int(len(np.unique(sub.take))),
                 "n_sessions": int(len(np.unique(sub.session)))}
    if n < MIN_SEGMENTS:
        out["note"] = f"fewer than {MIN_SEGMENTS} segments"
        return out
    cc = component_correlations(f_pcs, t_pcs)
    sim = per_segment_similarity(f_sc, t_sc)
    ret = retrieval(f_sc, t_sc)
    turn = turn_statistic(f_pcs[:, 0], t_pcs[:, 0], sub.story, sub.after_turn, sub.has_turn)
    out.update(valence_r=cc["pearson"][0], valence_rho=cc["spearman"][0], components=cc,
               cosine=sim["cosine"], spearman=sim["spearman"],
               retrieval={k: v for k, v in ret.items() if k != "ranks"}, turn=turn)

    # Nulls. Each permutation re-pairs face rows with other text rows.
    rng = np.random.default_rng(seed)
    nulls = {"story": {"valence_r": [], "cosine": [], "mrr": [], "top5": []},
             "shift": {"valence_r": [], "cosine": [], "mrr": [], "top5": [], "turn": []}}
    for kind in ("story", "shift"):
        for _ in range(n_perm):
            if kind == "story":
                p = story_permutation(sub.story, sub.sentence_idx, rng, take=sub.take)
            else:
                p = circular_shift(sub.story, sub.sentence_idx, rng, take=sub.take)
            nulls[kind]["valence_r"].append(pearson(f_pcs[:, 0], t_pcs[p, 0]))
            nulls[kind]["cosine"].append(per_segment_similarity(f_sc, t_sc[p])["cosine"])
            r = retrieval(f_sc, t_sc[p])
            nulls[kind]["mrr"].append(r["mrr"])
            nulls[kind]["top5"].append(r["top5"])
            if kind == "shift":
                # Shift the face series within each story instead (the text keeps its turn position).
                q = circular_shift(sub.story, sub.sentence_idx, rng, take=sub.take)
                nulls[kind]["turn"].append(turn_statistic(f_pcs[q, 0], t_pcs[:, 0], sub.story, sub.after_turn,
                                                          sub.has_turn)["mean_signed_step"])
    out["p"] = {
        "valence_r_story": p_value(out["valence_r"], nulls["story"]["valence_r"]),
        "valence_r_shift": p_value(out["valence_r"], nulls["shift"]["valence_r"]),
        "cosine_story": p_value(out["cosine"], nulls["story"]["cosine"]),
        "cosine_shift": p_value(out["cosine"], nulls["shift"]["cosine"]),
        "mrr_story": p_value(out["retrieval"]["mrr"], nulls["story"]["mrr"]),
        "mrr_shift": p_value(out["retrieval"]["mrr"], nulls["shift"]["mrr"]),
        "top5_story": p_value(out["retrieval"]["top5"], nulls["story"]["top5"]),
        "turn_shift": p_value(turn["mean_signed_step"], nulls["shift"]["turn"]),
    }
    out["null_mean"] = {f"{stat}_{kind}": float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
                        for kind, d in nulls.items() for stat, v in ((s, np.asarray(x, float)) for s, x in d.items())}
    out["null_sd"] = {f"{stat}_{kind}": float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")
                      for kind, d in nulls.items() for stat, v in ((s, np.asarray(x, float)) for s, x in d.items())}
    out["n_perm"] = int(n_perm)
    return out


def neutral_baseline(a: Aligned, nz: Normalized, mask: np.ndarray) -> dict:
    """Face readout on neutral-text segments against emotional ones: the
    valence spread the face shows when the text carries no emotion, and
    the agreement metrics with the neutral stories left out."""
    mask = np.asarray(mask, bool) & a.valid()
    neutral = mask & (a.kind == "neutral")
    emo = mask & (a.kind != "neutral")
    out = {"n_neutral": int(neutral.sum()), "n_emotional": int(emo.sum())}
    if neutral.sum() >= 2:
        out["face_valence_sd_neutral"] = float(np.nanstd(nz.face_pcs[neutral, 0]))
        out["face_valence_mean_neutral"] = float(np.nanmean(nz.face_pcs[neutral, 0]))
        out["cosine_neutral"] = per_segment_similarity(nz.face_scores[neutral], nz.text_scores[neutral])["cosine"]
    if emo.sum() >= 2:
        out["face_valence_sd_emotional"] = float(np.nanstd(nz.face_pcs[emo, 0]))
        out["text_valence_sd_emotional"] = float(np.nanstd(nz.text_pcs[emo, 0]))
        out["valence_r_emotional_only"] = pearson(nz.face_pcs[emo, 0], nz.text_pcs[emo, 0])
        out["cosine_emotional_only"] = per_segment_similarity(nz.face_scores[emo], nz.text_scores[emo])["cosine"]
    return out


# --------------------------------------------------------------------------
# The whole analysis
# --------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    lag_grid: tuple = tuple(float(x) for x in DEFAULT_LAG_GRID)
    n_perm: int = DEFAULT_N_PERM
    seed: int = 0
    fixed_lag: float | None = None      # skip estimation
    ridge: bool = True
    blendshapes: bool = True


def run_analysis(data: list[TakeData], split: SplitConfig, cfg: AnalysisConfig = AnalysisConfig(), progress=None) -> dict:
    """Lag on development data, then every metric per condition and split."""
    def dev_fn(a: Aligned) -> np.ndarray:
        return split_masks(a, split)[0]

    R: dict = {"split": split.__dict__, "config": cfg.__dict__, "lag": {}}
    if cfg.fixed_lag is None:
        R["lag"] = estimate_lag(data, dev_fn, cfg.lag_grid)
        lag = R["lag"]["lag"]
        # Per-condition curves are reported but the single fixed lag is used.
        R["lag"]["per_condition"] = {c: estimate_lag(data, dev_fn, cfg.lag_grid, condition=c) for c in CONDITIONS}
    else:
        lag = float(cfg.fixed_lag)
        R["lag"] = {"lag": lag, "curve": [], "fixed": True}
    if progress:
        progress(f"lag fixed at {lag:+.1f} s")
    a = assemble(data, lag=lag, ev_emotions=data[0].text_emotions if data else None)
    dev, held = split_masks(a, split)
    if dev.sum() < MIN_SEGMENTS:   # nothing to fit on: use everything, and say so
        R["warning"] = "fewer than 4 development segments; normalization fitted on all segments"
        dev = np.ones(len(a), bool)
    nz = normalize(a, dev)
    R["dataset"] = {
        "n_segments": int(len(a)), "n_takes": int(len(np.unique(a.take))), "n_stories": int(len(np.unique(a.story))),
        "sessions": sorted(np.unique(a.session).tolist()),
        "by_condition": {c: int((a.condition == c).sum()) for c in CONDITIONS},
        "dev_segments": int(dev.sum()), "heldout_segments": int(held.sum()),
        "heldout_by_condition": {c: int((held & (a.condition == c)).sum()) for c in CONDITIONS},
        "segments_without_face": int((~a.valid()).sum()),
        "mean_frames_per_segment": float(np.mean(a.face_n_frames)) if len(a) else 0.0,
        "mean_segment_s": float(np.mean(a.end - a.start)) if len(a) else 0.0,
        "blendshapes": a.blend is not None,
    }

    # Ridge maps fitted on development data (lightly trained variant; blendshape baseline).
    pred_ridge = pred_blend = None
    if cfg.ridge:
        x = np.concatenate([nz.face_scores, nz.face_pcs], axis=1)
        y = nz.text_pcs
        cv = ridge_cv(x[dev], y[dev], a.story[dev], seed=cfg.seed)
        okd = dev & a.valid()
        if okd.sum() >= MIN_SEGMENTS:
            model = fit_ridge(x[okd], y[okd], cv["alpha"])
            pred_ridge = np.full_like(y, np.nan)
            okx = np.isfinite(x).all(axis=1)
            pred_ridge[okx] = model.predict(x[okx])
        R["ridge_cv"] = cv
    if cfg.blendshapes and a.blend is not None:
        okb = np.isfinite(a.blend).all(axis=1)
        zb = ZScore.fit(a.blend[dev & okb])
        xb = zb(a.blend)
        cv = ridge_cv(xb[dev], nz.text_pcs[dev], a.story[dev], seed=cfg.seed)
        okd = dev & okb & a.valid()
        if okd.sum() >= MIN_SEGMENTS:
            model = fit_ridge(xb[okd], nz.text_pcs[okd], cv["alpha"])
            pred_blend = np.full_like(nz.text_pcs, np.nan)
            pred_blend[okb] = model.predict(xb[okb])
        R["blend_cv"] = cv

    R["results"] = {}
    for split_name, m in (("heldout", held), ("dev", dev)):
        R["results"][split_name] = {}
        for cond in (*CONDITIONS, "both"):
            mask = m if cond == "both" else (m & (a.condition == cond))
            if progress:
                progress(f"  {split_name}/{cond}: {int(mask.sum())} segments")
            entry = {"zero_training": evaluate(a, nz, mask, cfg.n_perm, cfg.seed),
                     "neutral_baseline": neutral_baseline(a, nz, mask)}
            if pred_ridge is not None:
                entry["ridge"] = evaluate(a, nz, mask & np.isfinite(pred_ridge).all(axis=1), cfg.n_perm, cfg.seed, pred_pcs=pred_ridge)
            if pred_blend is not None:
                entry["blendshapes"] = evaluate(a, nz, mask & np.isfinite(pred_blend).all(axis=1), cfg.n_perm, cfg.seed,
                                                pred_pcs=pred_blend)
            R["results"][split_name][cond] = entry
    R["verbal_baseline"] = "not available: the video store holds activations only, no next-token logits"
    if a.blend is None:
        R["blendshape_baseline"] = "not available: no frames.jsonl with blendshapes next to the face series"
    R["gate3"] = gate3(R)
    return R


def gate3(R: dict) -> dict:
    """The Gate 3 criteria from the work plan, on held-out segments."""
    held = R["results"]["heldout"]
    best = None         # the passing condition with the highest r; else the highest r overall
    for cond in CONDITIONS:
        z = held[cond]["zero_training"]
        if "valence_r" not in z or not np.isfinite(z["valence_r"]):
            continue
        cand = (z["valence_r"], z["p"]["valence_r_story"], cond)
        ok = cand[0] >= 0.2 and np.isfinite(cand[1]) and cand[1] < 0.05
        if best is None:
            best = cand
            continue
        best_ok = best[0] >= 0.2 and np.isfinite(best[1]) and best[1] < 0.05
        if (ok and not best_ok) or (ok == best_ok and cand[0] > best[0]):
            best = cand
    valence_pass = bool(best and best[0] >= 0.2 and np.isfinite(best[1]) and best[1] < 0.05)
    turns = {c: held[c]["zero_training"].get("turn") for c in CONDITIONS}
    turn_p = {c: held[c]["zero_training"].get("p", {}).get("turn_shift") for c in CONDITIONS}
    turn_pass = any(p is not None and np.isfinite(p) and p < 0.05 and turns[c]["mean_signed_step"] > 0 for c, p in turn_p.items())
    return {"valence": {"pass": valence_pass, "best_condition": best[2] if best else None,
                        "r": best[0] if best else float("nan"), "p_story": best[1] if best else float("nan"),
                        "criterion": "r >= 0.2 and p < 0.05 (story-level permutation) in at least one condition"},
            "turn": {"pass": turn_pass, "per_condition": {c: {"stat": turns[c], "p_shift": turn_p[c]} for c in CONDITIONS},
                     "criterion": "face valence steps with the text at scripted turns more than under circular shifts"},
            "by_condition": {c: {"valence_r": held[c]["zero_training"].get("valence_r"),
                                 "p_story": held[c]["zero_training"].get("p", {}).get("valence_r_story"),
                                 "n": held[c]["zero_training"].get("n_segments")} for c in CONDITIONS}}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _f(x, nd=2) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "—"
    return f"{xf:.{nd}f}"


def _p(x) -> str:
    return "—" if x is None or not np.isfinite(float(x)) else (f"{float(x):.3f}" if float(x) >= 0.001 else "<0.001")


def _row(label: str, e: dict) -> str:
    if "valence_r" not in e:
        return f"| {label} | {e.get('n_segments', 0)} | {e.get('n_stories', 0)} | — | — | — | — | — | — | — | — |"
    p = e["p"]
    r = e["retrieval"]
    comps = e["components"]["pearson"]
    others = ", ".join(_f(c) for c in comps[1:4]) if len(comps) > 1 else "—"
    return (f"| {label} | {e['n_segments']} | {e['n_stories']} | {_f(e['valence_r'])} | {_p(p['valence_r_story'])} / {_p(p['valence_r_shift'])} "
            f"| {others} | {_f(e['cosine'])} ({_p(p['cosine_story'])}) | {_f(e['spearman'])} "
            f"| {_f(r['mean_norm_rank'])} | {_f(r['top1'])} / {_f(r['top5'])} ({_p(p['top5_story'])}) "
            f"| {_f(e['turn']['mean_signed_step'])}, {e['turn']['n_stories']} st. ({_p(p['turn_shift'])}) |")


def write_report(R: dict, path: str | Path, title: str = "M4 text–face agreement (Gate 3)") -> Path:
    L: list[str] = [f"# {title}", ""]
    d = R["dataset"]
    L.append(f"Segments: {d['n_segments']} from {d['n_takes']} takes of {d['n_stories']} stories in {len(d['sessions'])} session(s) "
             f"({', '.join(d['sessions'])}); reading {d['by_condition'].get('read', 0)}, listening {d['by_condition'].get('listen', 0)}. "
             f"Development {d['dev_segments']}, held-out {d['heldout_segments']} "
             f"(reading {d['heldout_by_condition'].get('read', 0)}, listening {d['heldout_by_condition'].get('listen', 0)}). "
             f"Mean segment {d['mean_segment_s']:.1f} s, {d['mean_frames_per_segment']:.1f} frames; "
             f"{d['segments_without_face']} segments without a face readout.")
    L.append("")
    L.append(f"Split: {R['split'].get('note', '')}. Held-out session: {R['split'].get('holdout_session') or 'none'}; "
             f"held-out stories: {len(R['split'].get('holdout_stories', []))}.")
    if R.get("warning"):
        L.append(f"\n**Warning:** {R['warning']}")
    lag = R["lag"]
    L.append("")
    L.append(f"Lag: {lag['lag']:+.1f} s (positive = face trails text), " +
             ("fixed by configuration." if lag.get("fixed") else
              "chosen on development segments by valence r over the grid " +
              ", ".join(f"{c['lag']:+.1f}: {_f(c['r'])}" for c in lag["curve"]) + "."))
    if lag.get("per_condition"):
        for c, e in lag["per_condition"].items():
            L.append(f"  - {c}: best {e['lag']:+.1f} s (" + ", ".join(f"{x['lag']:+.1f}: {_f(x['r'])}" for x in e["curve"]) + ")")
    L.append("")
    g = R["gate3"]
    L.append("## Gate 3 table")
    L.append("")
    L.append("| Check | Criterion | Result | Pass? |")
    L.append("|---|---|---|---|")
    bc = g["by_condition"]
    L.append(f"| Valence correlation, held-out segments | {g['valence']['criterion']} | "
             + "; ".join(f"{c}: r {_f(v['valence_r'])}, p {_p(v['p_story'])} (n {v['n']})" for c, v in bc.items())
             + f" | {'Yes' if g['valence']['pass'] else 'No'} |")
    tp = g["turn"]["per_condition"]
    L.append(f"| Scripted turns | {g['turn']['criterion']} | "
             + "; ".join(f"{c}: signed step {_f((v['stat'] or {}).get('mean_signed_step'))}, "
                         f"{(v['stat'] or {}).get('n_stories', 0)} stories, p {_p(v['p_shift'])}" for c, v in tp.items())
             + f" | {'Yes' if g['turn']['pass'] else 'No'} |")
    L.append("| Reported by condition against every baseline | reading and listening; nulls, neutral text, blendshapes, ridge | "
             "see tables below; verbal baseline absent | Yes |")
    L.append("")
    cols = ("| Subset | Segments | Stories | Valence r | p story / shift | r hostility / arousal / detachment (PC2–4) | Cosine (p story) | Spearman "
            "| Norm. rank | Top-1 / top-5 (p) | Turn step, n (p shift) |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    for split_name, heading in (("heldout", "Held-out segments (the reported result)"), ("dev", "Development segments (tuning data; optimistic)")):
        L.append(f"## {heading}")
        L.append("")
        L.append("Zero-training readout: each modality z-scored on development data, no map fitted.")
        L.append("")
        L.extend([cols, sep])
        for cond in (*CONDITIONS, "both"):
            L.append(_row(cond, R["results"][split_name][cond]["zero_training"]))
        L.append("")
        L.append("Lightly trained variant: ridge map from the face profile (171 scores + PCs) to the text PCs, fitted on development segments"
                 f" (alpha {_f(R.get('ridge_cv', {}).get('alpha'), 1)}, chosen by story-grouped CV). Cosine/retrieval are on the predicted PCs.")
        L.append("")
        L.extend([cols, sep])
        for cond in (*CONDITIONS, "both"):
            e = R["results"][split_name][cond].get("ridge")
            L.append(_row(f"{cond} (ridge)", e) if e else f"| {cond} (ridge) | — | — | not fitted | | | | | | | |")
        L.append("")
        L.append("Blendshape baseline: ridge from the segment-mean MediaPipe blendshapes to the text PCs, fitted on development segments.")
        L.append("")
        if any("blendshapes" in R["results"][split_name][c] for c in CONDITIONS):
            L.extend([cols, sep])
            for cond in (*CONDITIONS, "both"):
                e = R["results"][split_name][cond].get("blendshapes")
                L.append(_row(f"{cond} (blendshapes)", e) if e else f"| {cond} (blendshapes) | — | — | not fitted | | | | | | | |")
        else:
            L.append(R.get("blendshape_baseline", "not computed"))
        L.append("")
        L.append("Neutral-text baseline: the face readout on neutral stories, where the text carries no emotion.")
        L.append("")
        L.append("| Subset | Neutral segments | Emotional segments | Face valence SD, neutral | Face valence SD, emotional | Cosine, neutral | Valence r, emotional only | Cosine, emotional only |")
        L.append("|---|---|---|---|---|---|---|---|")
        for cond in (*CONDITIONS, "both"):
            nb = R["results"][split_name][cond]["neutral_baseline"]
            L.append(f"| {cond} | {nb['n_neutral']} | {nb['n_emotional']} | {_f(nb.get('face_valence_sd_neutral'))} | "
                     f"{_f(nb.get('face_valence_sd_emotional'))} | {_f(nb.get('cosine_neutral'))} | "
                     f"{_f(nb.get('valence_r_emotional_only'))} | {_f(nb.get('cosine_emotional_only'))} |")
        L.append("")
    L.append("## Nulls")
    L.append("")
    L.append(f"{R['config']['n_perm']} permutations each. Story-level permutation re-pairs every story's face segments with the "
             "proportional sentences of a permuted partner story (each take is one unit). Within-story circular shift rotates the "
             "text sentences inside each take. p-values are one-sided with the +1 correction. Null means (held-out, both conditions):")
    nm = R["results"]["heldout"]["both"]["zero_training"].get("null_mean", {})
    if nm:
        L.append("")
        L.append("| Statistic | Story null mean | Shift null mean | Observed |")
        L.append("|---|---|---|---|")
        z = R["results"]["heldout"]["both"]["zero_training"]
        for stat, obs in (("valence_r", z.get("valence_r")), ("cosine", z.get("cosine")), ("mrr", z.get("retrieval", {}).get("mrr")),
                          ("top5", z.get("retrieval", {}).get("top5"))):
            L.append(f"| {stat} | {_f(nm.get(f'{stat}_story'), 3)} | {_f(nm.get(f'{stat}_shift'), 3)} | {_f(obs, 3)} |")
    L.append("")
    L.append("## Notes and limits")
    L.append("")
    L.append(f"- Verbal baseline: {R['verbal_baseline']}.")
    L.append("- One subject, performed expression: the agreement ceiling is well below 1 and the result does not generalize across people. "
             "Read text is reported apart from listening, where there is no mouth motion.")
    L.append("- Segments within a story are not independent; the story-level null is the one to read. Held-out numbers are the result; "
             "development numbers were used to pick the lag, the normalization and the ridge alpha.")
    L.append("- Retrieval ranks are among the segments of the same subset (condition and split), so the chance normalized rank is 0.5 regardless of n.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return path


def to_json(R: dict) -> str:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    return json.dumps(R, indent=1, default=default)
