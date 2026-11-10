"""One video to an emotion time series: the steps behind scripts/video_run.py
and the hosted recorder's processing function.

    1. sample frames, track the face, save loose crops          (CPU)
    2. activations, a few hundred frames per call, chunked store (GPU, via `run_jobs`)
    3. projection with the saved reference, baseline, windowing, Parquet
    4. optionally a Rerun recording

`run_jobs(jobs, batch_size)` is whatever executes the activation calls: the
deployed Modal class, an ephemeral app, or, inside Modal, the class itself.
Each job is `([spec], payload)` as `modal_app.video.Infer.run` takes them.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from cogmirror import prep, store, video

RunJobs = Callable[[list, int], list]


@dataclass
class Settings:
    fps: float = video.DEFAULT_RATE
    max_gap: float = video.DEFAULT_MAX_GAP
    crop: str = "medium"
    budget: int = 280
    batch_size: int = 8
    chunk_frames: int = 300
    readout: str = "last"
    score_layer: int = store.DEFAULT_SCORE_LAYER
    valence_layer: int = store.DEFAULT_VALENCE_LAYER
    n_pcs: int = store.DEFAULT_N_PCS
    window: float = store.DEFAULT_WINDOW
    ema: float = 0.0
    landmarker: str = prep.LANDMARKER_PATH
    gemotions: str = "data/gemotions"
    reference: str = "results/m1/reference_a2_prefill.npz"

    def as_dict(self) -> dict:
        return asdict(self)


def track(video_path: str | Path, work: Path, s: Settings, force: bool = False, log=print) -> tuple[list[dict], dict]:
    frames_dir = work / "frames"
    if (frames_dir / "frames.jsonl").exists() and not force:
        recs, info = video.read_manifest(frames_dir)
        log(f"frames: {len(recs)} already tracked under {frames_dir}")
        return recs, info
    lm = video.make_landmarker(s.landmarker)
    t0 = time.time()
    recs, info = video.track_video(str(video_path), lm, frames_dir, rate=s.fps,
                                   smoother=video.BoxSmoother(max_gap=s.max_gap), progress=log)
    log(f"frames: {info['n_sampled']} sampled at {s.fps} fps, {info['n_faces']} detections, {info['n_crops']} crops, "
        f"{time.time() - t0:.1f}s")
    return recs, info


def infer(work: Path, recs: list[dict], run_jobs: RunJobs, s: Settings, force: bool = False, log=print) -> Path:
    frames = video.frames_with_crops(recs)
    acts_dir = work / "acts"
    chunks = store.plan_chunks(len(frames), s.chunk_frames)
    todo = [c for c in chunks if force or not store.chunk_done(acts_dir, c[0])]
    log(f"acts: {len(chunks)} chunks, {len(todo)} to compute")
    if not todo:
        return acts_dir
    frames_dir = work / "frames"
    jobs = []
    for k, a, b in todo:
        part = frames[a:b]
        spec = {"video_id": work.name, "chunk": k, "files": [f["file"] for f in part],
                "frame_index": [f["frame_index"] for f in part], "t": [f["t"] for f in part],
                "store": "", "crop": s.crop, "budget": s.budget}
        payload = {f["file"]: (frames_dir / f["file"]).read_bytes() for f in part}
        jobs.append(([spec], payload))
    t0 = time.time()
    results = run_jobs(jobs, s.batch_size)
    gpu_s, gpu = 0.0, ""
    for res in results:
        for r in res:
            with np.load(io.BytesIO(r["npz"]), allow_pickle=False) as z:
                store.save_chunk(acts_dir, r["chunk"], {k: z[k] for k in z.files})
            gpu_s += r["seconds"]
            gpu = r.get("gpu", gpu)
    log(f"acts: {sum(len(j[0][0]['files']) for j in jobs)} frames in {time.time() - t0:.0f}s wall, "
        f"{gpu_s:.0f}s GPU compute (plus model load), {gpu}")
    return acts_dir


def project(video_path: str | Path, work: Path, recs: list[dict], info: dict, acts_dir: Path, s: Settings,
            baseline: str = "none", save_baseline: str = "", log=print) -> Path:
    """Projection, baseline, windowing; writes <work>/<name>.parquet.

    baseline: "none" | "session" (this video's own mean) | path to a baseline json.
    save_baseline: write this video's mean to that path (e.g. from a calibration clip).
    """
    acts = store.load_store(acts_dir)
    cfg = store.ProjectionConfig(readout=s.readout, score_layer=s.score_layer, valence_layer=s.valence_layer, n_pcs=s.n_pcs)
    spaces = store.load_spaces(s.gemotions, [cfg.score_layer, cfg.valence_layer], cfg.n_pcs)
    refs, _ = store.load_reference(s.reference)
    proj = store.project(acts, spaces, refs, cfg)
    if save_baseline:
        store.fit_baseline(proj, source=str(video_path)).save(save_baseline)
        log(f"baseline: saved {save_baseline}")
    if baseline == "session":
        proj = store.apply_baseline(proj, store.fit_baseline(proj, source="session"))
    elif baseline not in ("", "none"):
        proj = store.apply_baseline(proj, store.Baseline.load(baseline))
    scfg = store.SeriesConfig(window=s.window, ema_tau=s.ema, projection=cfg)
    cols = store.build_series(proj, scfg, frames=recs)
    meta = store.series_metadata(scfg, {"video": str(video_path), "fps_sampled": s.fps, "crop": s.crop,
                                        "budget": s.budget, "baseline": baseline, "reference": s.reference,
                                        "duration_s": info.get("duration"), "source_fps": info.get("fps")})
    out = work / f"{work.name}.parquet"
    store.write_parquet(out, cols, meta)
    v = cols["valence"][np.isfinite(cols["valence"])]
    if len(v):
        log(f"series: {len(cols['t'])} rows -> {out}; valence mean {v.mean():+.2f} sd {v.std():.2f} "
            f"(range {v.min():+.2f} .. {v.max():+.2f}); top at mid-clip: {cols['top1'][len(v) // 2]}, {cols['top2'][len(v) // 2]}")
    else:
        log(f"series: {len(cols['t'])} rows -> {out}; no frames with a face")
    return out


def summary_of(parquet: Path) -> dict:
    cols, _ = store.read_parquet(parquet)
    v = cols["valence"][np.isfinite(cols["valence"])]
    out = {"frames": int(len(cols["t"])), "faces": int(np.isfinite(cols["valence"]).sum())}
    if len(v):
        out.update({"valence_mean": round(float(v.mean()), 3), "valence_sd": round(float(v.std()), 3),
                    "top_mid": [str(cols[k][len(v) // 2]) for k in ("top1", "top2", "top3") if k in cols]})
    return out


def process(video_path: str | Path, work: Path, run_jobs: RunJobs, s: Settings = Settings(), baseline: str = "none",
            save_baseline: str = "", rrd: Path | None = None, traces: str = "default", force: bool = False,
            log=print, stage: Callable[[str], None] | None = None) -> dict:
    """The whole pipeline for one video. `stage(name)` is called as each step starts."""
    timings: dict[str, float] = {}
    work.mkdir(parents=True, exist_ok=True)

    def step(name):
        timings[name] = time.time()
        if stage:
            stage(name)

    step("tracking")
    recs, info = track(video_path, work, s, force, log)
    step("model")
    acts_dir = infer(work, recs, run_jobs, s, force, log)
    step("series")
    parquet = project(video_path, work, recs, info, acts_dir, s, baseline, save_baseline, log)
    result = {"parquet": str(parquet), "summary": summary_of(parquet)}
    if rrd is not None:
        step("export")
        from cogmirror import rerun_export

        rerun_export.export(work, rrd, traces=traces)
        result["rrd"] = str(rrd)
        log(f"rerun -> {rrd}")
    end = time.time()
    names = list(timings)
    result["timings"] = {n: round((timings[names[i + 1]] if i + 1 < len(names) else end) - timings[n], 1) for i, n in enumerate(names)}
    return result
