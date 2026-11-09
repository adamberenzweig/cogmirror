"""Per-video activation store, the projection reference, and the time series.

Layout of one video's store (a directory):

    chunk_0000.npz   frame_index [n], t [n], last [n, L, D] f16, image [n, L, D] f16, layers [L]
    chunk_0001.npz   ...

Chunks are written atomically (temp file, then rename), so a rerun of a
preempted job skips the chunks that exist and recomputes the rest.

Projection follows the design doc: scores are `(x - mean) . v_e / |v_e|`
divided by the per-emotion scale, and PCs are coordinates in the fixed PCA
basis of the 171 vectors, both relative to a `Reference` fitted once on the
M1 dev actors (results/m1/reference_a2_prefill.npz). Per-person baselines
subtract a mean taken over a calibration clip or over the whole session.

Nothing here needs torch. pyarrow is imported only when Parquet is read or
written.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .evspace import EVSpace, Reference

CHUNK_GLOB = "chunk_*.npz"
READOUTS = ("last", "image")
DEFAULT_SCORE_LAYER = 35     # categories (M1: best 4-way among negatives)
DEFAULT_VALENCE_LAYER = 50   # valence (M1: best held-out AUC)
DEFAULT_N_PCS = 5
DEFAULT_WINDOW = 1.5         # seconds, centered
TOP_K = 3


# --------------------------------------------------------------------------
# Chunked activation store
# --------------------------------------------------------------------------

def chunk_path(store_dir: str | Path, chunk: int) -> Path:
    return Path(store_dir) / f"chunk_{chunk:04d}.npz"


def chunk_done(store_dir: str | Path, chunk: int) -> bool:
    p = chunk_path(store_dir, chunk)
    return p.exists() and p.stat().st_size > 0


def plan_chunks(n_frames: int, chunk_size: int) -> list[tuple[int, int, int]]:
    """(chunk id, start, stop) over `n_frames` in pieces of `chunk_size`."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [(k, s, min(s + chunk_size, n_frames)) for k, s in enumerate(range(0, n_frames, chunk_size))]


def chunk_payload(frame_index, t, last, image, layers) -> dict[str, np.ndarray]:
    last = np.asarray(last)
    image = np.asarray(image)
    if last.shape != image.shape or last.ndim != 3:
        raise ValueError(f"expected [n, L, D] arrays, got {last.shape} and {image.shape}")
    if len(frame_index) != last.shape[0] or len(t) != last.shape[0]:
        raise ValueError("frame_index, t and activations disagree on the number of frames")
    return {
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "t": np.asarray(t, dtype=np.float64),
        "last": last.astype(np.float16),
        "image": image.astype(np.float16),
        "layers": np.asarray(sorted(int(l) for l in layers), dtype=np.int64),
    }


def save_chunk(store_dir: str | Path, chunk: int, payload: dict[str, np.ndarray]) -> Path:
    """Write one chunk atomically."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    final = chunk_path(store_dir, chunk)
    tmp = final.with_suffix(".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
    os.replace(tmp, final)
    return final


def load_store(store_dir: str | Path) -> dict:
    """Merge every chunk of a store, sorted by frame index (duplicates keep
    the later chunk). Activations stay float16."""
    files = sorted(Path(store_dir).glob(CHUNK_GLOB))
    if not files:
        raise FileNotFoundError(f"no chunks under {store_dir}")
    parts = []
    layers = None
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            if layers is None:
                layers = z["layers"].tolist()
            elif z["layers"].tolist() != layers:
                raise ValueError(f"{f} has layers {z['layers'].tolist()}, expected {layers}")
            parts.append({k: z[k] for k in ("frame_index", "t", "last", "image")})
    idx = np.concatenate([p["frame_index"] for p in parts])
    # Stable sort keeps the later chunk's copy of a duplicated frame last; unique() then keeps the first
    # of each run, so reverse before deduplicating.
    order = np.argsort(idx, kind="stable")[::-1]
    _, first = np.unique(idx[order], return_index=True)
    keep = order[first]
    out = {k: np.concatenate([p[k] for p in parts])[keep] for k in ("frame_index", "t", "last", "image")}
    out["layers"] = layers
    return out


def store_frames(store_dir: str | Path) -> np.ndarray:
    """Frame indices present in a store (empty if none)."""
    files = sorted(Path(store_dir).glob(CHUNK_GLOB))
    if not files:
        return np.zeros(0, dtype=np.int64)
    idx = [np.load(f, allow_pickle=False)["frame_index"] for f in files]
    return np.unique(np.concatenate(idx))


# --------------------------------------------------------------------------
# Reference statistics
# --------------------------------------------------------------------------

def _ref_key(readout: str, layer: int) -> str:
    return f"{readout}_L{int(layer)}"


def save_reference(path: str | Path, refs: dict[tuple[str, int], Reference], meta: dict | None = None) -> Path:
    """One file for every (readout, layer): `<readout>_L<layer>_{mean,score_std,pc_std}`."""
    arrays: dict[str, np.ndarray] = {}
    for (readout, layer), ref in refs.items():
        k = _ref_key(readout, layer)
        arrays[f"{k}_mean"] = np.asarray(ref.mean, dtype=np.float32)
        arrays[f"{k}_score_std"] = np.asarray(ref.score_std, dtype=np.float32)
        arrays[f"{k}_pc_std"] = np.asarray(ref.pc_std, dtype=np.float32)
    arrays["keys"] = np.array([f"{r}:{l}" for r, l in refs], dtype=str)
    arrays["meta"] = np.array(json.dumps(meta or {}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    return path


def load_reference(path: str | Path) -> tuple[dict[tuple[str, int], Reference], dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        refs = {}
        for key in z["keys"].tolist():
            readout, layer = key.split(":")
            k = _ref_key(readout, int(layer))
            refs[(readout, int(layer))] = Reference(
                z[f"{k}_mean"].astype(np.float64), z[f"{k}_score_std"].astype(np.float64), z[f"{k}_pc_std"].astype(np.float64))
    return refs, meta


def reference_for(refs: dict, readout: str, layer: int, n_components: int) -> Reference:
    """The stored reference, with the PC scale cut to `n_components`."""
    ref = refs[(readout, int(layer))]
    if len(ref.pc_std) < n_components:
        raise ValueError(f"reference holds {len(ref.pc_std)} PC scales, {n_components} requested")
    return Reference(ref.mean, ref.score_std, ref.pc_std[:n_components])


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------

@dataclass
class ProjectionConfig:
    readout: str = "last"
    score_layer: int = DEFAULT_SCORE_LAYER
    valence_layer: int = DEFAULT_VALENCE_LAYER
    n_pcs: int = DEFAULT_N_PCS


@dataclass
class Projected:
    """Per-frame projections of one store. Rows follow the store's frames."""
    frame_index: np.ndarray     # [N]
    t: np.ndarray               # [N]
    scores: np.ndarray          # [N, E] at score_layer, in reference units
    pcs: np.ndarray             # [N, K] at valence_layer; column 0 is valence
    emotions: list[str]
    config: ProjectionConfig

    @property
    def valence(self) -> np.ndarray:
        return self.pcs[:, 0]


def load_spaces(gemotions_root: str | Path, layers, n_components: int = DEFAULT_N_PCS) -> dict[int, EVSpace]:
    return {int(l): EVSpace.load(gemotions_root, int(l), n_components=n_components) for l in sorted(set(layers))}


def project(acts: dict, spaces: dict[int, EVSpace], refs: dict, cfg: ProjectionConfig = ProjectionConfig()) -> Projected:
    """Project a loaded store (`load_store` output) with the saved reference."""
    layers = list(acts["layers"])
    for layer in (cfg.score_layer, cfg.valence_layer):
        if layer not in layers:
            raise ValueError(f"layer {layer} not in store layers {layers}")
    x = acts[cfg.readout]
    xs = x[:, layers.index(cfg.score_layer)].astype(np.float64)
    xv = x[:, layers.index(cfg.valence_layer)].astype(np.float64)
    sp_s, sp_v = spaces[cfg.score_layer], spaces[cfg.valence_layer]
    scores = sp_s.scores(xs, reference_for(refs, cfg.readout, cfg.score_layer, len(sp_s.explained)))
    if len(sp_v.explained) < cfg.n_pcs:
        raise ValueError(f"space at layer {cfg.valence_layer} has {len(sp_v.explained)} components, {cfg.n_pcs} requested")
    pcs = sp_v.pcs(xv, reference_for(refs, cfg.readout, cfg.valence_layer, len(sp_v.explained)))
    return Projected(np.asarray(acts["frame_index"]), np.asarray(acts["t"], dtype=np.float64), scores, pcs[:, : cfg.n_pcs],
                     list(sp_s.emotions), cfg)


# --------------------------------------------------------------------------
# Baselines and smoothing
# --------------------------------------------------------------------------

@dataclass
class Baseline:
    """Per-person offset in projected units: subtracted from every frame."""
    scores: np.ndarray      # [E]
    pcs: np.ndarray         # [K]
    n_frames: int = 0
    source: str = ""

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"scores": self.scores.tolist(), "pcs": self.pcs.tolist(),
                                          "n_frames": self.n_frames, "source": self.source}))

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        d = json.loads(Path(path).read_text())
        return cls(np.asarray(d["scores"], float), np.asarray(d["pcs"], float), int(d.get("n_frames", 0)), d.get("source", ""))


def fit_baseline(proj: Projected, mask: np.ndarray | None = None, source: str = "") -> Baseline:
    """Mean over the (masked) frames of a calibration clip or a whole session."""
    rows = np.ones(len(proj.t), bool) if mask is None else np.asarray(mask, bool)
    rows &= np.isfinite(proj.valence)
    if not rows.any():
        raise ValueError("no frames to fit a baseline on")
    return Baseline(proj.scores[rows].mean(axis=0), proj.pcs[rows].mean(axis=0), int(rows.sum()), source)


def apply_baseline(proj: Projected, base: Baseline) -> Projected:
    return Projected(proj.frame_index, proj.t, proj.scores - base.scores, proj.pcs - base.pcs, proj.emotions, proj.config)


def window_mean(t: np.ndarray, x: np.ndarray, width: float) -> np.ndarray:
    """Centered moving average over time: for each row, the mean of rows
    within `width / 2` seconds, ignoring NaNs. `width <= 0` returns x."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if width <= 0 or len(t) == 0:
        return x.copy()
    flat = x.ndim == 1
    x2 = x[:, None] if flat else x
    order = np.argsort(t, kind="stable")
    ts, xs = t[order], x2[order]
    valid = np.isfinite(xs)
    filled = np.where(valid, xs, 0.0)
    cs = np.vstack([np.zeros((1, xs.shape[1])), np.cumsum(filled, axis=0)])
    cn = np.vstack([np.zeros((1, xs.shape[1])), np.cumsum(valid, axis=0)])
    half = width / 2
    lo = np.searchsorted(ts, ts - half, side="left")
    hi = np.searchsorted(ts, ts + half, side="right")
    s = cs[hi] - cs[lo]
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out_sorted = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out[:, 0] if flat else out


def ema(t: np.ndarray, x: np.ndarray, tau: float) -> np.ndarray:
    """Causal exponential smoothing with time constant `tau` seconds. NaN
    rows are skipped (carry the previous state). `tau <= 0` returns x."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if tau <= 0 or len(t) == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    state = None
    t_prev = None
    for i in np.argsort(t, kind="stable"):
        xi = x[i]
        if not np.all(np.isfinite(xi)):
            out[i] = state if state is not None else np.nan
            continue
        if state is None:
            state = xi.astype(np.float64)
        else:
            a = float(np.exp(-(t[i] - t_prev) / tau))
            state = a * state + (1 - a) * xi
        t_prev = t[i]
        out[i] = state
    return out


def top_emotions(scores: np.ndarray, emotions: list[str], k: int = TOP_K) -> list[list[str]]:
    """The k highest-scoring emotions per row ("" when the row is NaN)."""
    out = []
    names = np.asarray(emotions)
    for row in np.asarray(scores):
        if not np.all(np.isfinite(row)):
            out.append([""] * k)
            continue
        out.append(names[np.argsort(row)[::-1][:k]].tolist())
    return out


# --------------------------------------------------------------------------
# Time series table
# --------------------------------------------------------------------------

@dataclass
class SeriesConfig:
    window: float = DEFAULT_WINDOW
    ema_tau: float = 0.0
    top_k: int = TOP_K
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)


def _align(proj: Projected, frame_index: np.ndarray, t: np.ndarray) -> Projected:
    """Expand a projection to a full frame list, NaN where a frame is missing."""
    pos = {int(i): k for k, i in enumerate(proj.frame_index)}
    rows = np.array([pos.get(int(i), -1) for i in frame_index])
    have = rows >= 0
    scores = np.full((len(frame_index), proj.scores.shape[1]), np.nan)
    pcs = np.full((len(frame_index), proj.pcs.shape[1]), np.nan)
    scores[have] = proj.scores[rows[have]]
    pcs[have] = proj.pcs[rows[have]]
    return Projected(np.asarray(frame_index), np.asarray(t, dtype=np.float64), scores, pcs, proj.emotions, proj.config)


def build_series(proj: Projected, cfg: SeriesConfig = SeriesConfig(), frames: list[dict] | None = None) -> dict:
    """Columns of the output table, one row per frame.

    `frames` is the tracker's frame manifest (dicts with `frame_index`, `t`,
    `face_found`); when given, every sampled frame gets a row, NaN where no
    activation exists. Columns: frame_index, t, face_found, valence_raw,
    valence, pc1..pcK (windowed), score_<emotion> (windowed), top1..topk.
    """
    if frames is not None:
        fi = np.array([f["frame_index"] for f in frames], dtype=np.int64)
        t = np.array([f["t"] for f in frames], dtype=np.float64)
        face = np.array([bool(f.get("face_found", True)) for f in frames])
        proj = _align(proj, fi, t)
    else:
        face = np.isfinite(proj.valence)
    order = np.argsort(proj.t, kind="stable")
    t = proj.t[order]
    scores_raw = proj.scores[order]
    pcs_raw = proj.pcs[order]

    def smooth(x):
        y = window_mean(t, x, cfg.window)
        return ema(t, y, cfg.ema_tau) if cfg.ema_tau > 0 else y

    scores = smooth(scores_raw)
    pcs = smooth(pcs_raw)
    top = top_emotions(scores, proj.emotions, cfg.top_k)
    cols: dict[str, np.ndarray] = {
        "frame_index": proj.frame_index[order],
        "t": t,
        "face_found": face[order],
        "valence_raw": pcs_raw[:, 0],
        "valence": pcs[:, 0],
    }
    for k in range(pcs.shape[1]):
        cols[f"pc{k + 1}"] = pcs[:, k]
    for j, e in enumerate(proj.emotions):
        cols[f"score_{e}"] = scores[:, j]
    for k in range(cfg.top_k):
        cols[f"top{k + 1}"] = np.array([row[k] for row in top], dtype=str)
    return cols


def series_metadata(cfg: SeriesConfig, extra: dict | None = None) -> dict[str, str]:
    meta = {
        "readout": cfg.projection.readout,
        "score_layer": str(cfg.projection.score_layer),
        "valence_layer": str(cfg.projection.valence_layer),
        "n_pcs": str(cfg.projection.n_pcs),
        "window_s": str(cfg.window),
        "ema_tau_s": str(cfg.ema_tau),
    }
    meta.update({k: str(v) for k, v in (extra or {}).items()})
    return meta


def write_parquet(path: str | Path, cols: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({k: pa.array(v.tolist() if v.dtype.kind == "U" else v) for k, v in cols.items()})
    if metadata:
        table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                               **{k.encode(): v.encode() for k, v in metadata.items()}})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def read_parquet(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items() if not k.startswith(b"pandas")}
    return cols, meta


def series_to_json(cols: dict[str, np.ndarray], metadata: dict[str, str] | None = None, emotions: list[str] | None = None) -> dict:
    """Compact JSON for the viewer: time, valence, PCs, top emotions, and the
    full score matrix as a list of rows."""
    def clean(a):
        return [None if (isinstance(x, float) and not np.isfinite(x)) else x for x in np.asarray(a, dtype=float).tolist()]

    score_cols = [c for c in cols if c.startswith("score_")]
    emotions = emotions or [c[len("score_"):] for c in score_cols]
    out = {
        "meta": metadata or {},
        "t": clean(cols["t"]),
        "face_found": [bool(x) for x in cols["face_found"]],
        "valence_raw": clean(cols["valence_raw"]),
        "valence": clean(cols["valence"]),
        "pcs": {c: clean(cols[c]) for c in cols if c.startswith("pc")},
        "top": [[str(cols[f"top{k}"][i]) for k in range(1, TOP_K + 1) if f"top{k}" in cols] for i in range(len(cols["t"]))],
        "emotions": emotions,
        "scores": [clean([cols[f"score_{e}"][i] for e in emotions]) for i in range(len(cols["t"]))],
    }
    return out
