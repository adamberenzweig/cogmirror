"""One video to an emotion time series.

    python scripts/video_run.py path/to/video.mp4 [--fps 5] [--window 1.5] [--rrd]

Steps (each skipped when its output exists, unless --force):
  1. sample frames at --fps, track the face, save loose crops (local CPU)
  2. activations on Modal (L40S), a few hundred frames per call, into a
     chunked store under --work/<name>/acts (idempotent per chunk)
  3. projection with the saved reference, baseline, windowing, Parquet

Outputs under --work/<name>/ (default data/video/<name>/):
    frames/            crops, frames.jsonl, video.json
    acts/chunk_*.npz   activation store
    <name>.parquet     one row per sampled frame
    <name>.rrd         Rerun recording for the viewer (with --rrd; see scripts/rerun_export.py)

Baselines: --baseline none | session | <baseline.json>. `session` subtracts
this video's own mean. --save-baseline writes the mean of this video as a
baseline file for later runs (e.g. from a calibration clip).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cogmirror import store, video  # noqa: E402

LANDMARKER = "data/models/face_landmarker.task"
REFERENCE = "results/m1/reference_a2_prefill.npz"
GEMOTIONS = "data/gemotions"


def track(args, work: Path) -> tuple[list[dict], dict]:
    frames_dir = work / "frames"
    if (frames_dir / "frames.jsonl").exists() and not args.force:
        recs, info = video.read_manifest(frames_dir)
        print(f"frames: {len(recs)} already tracked under {frames_dir}")
        return recs, info
    lm = video.make_landmarker(args.landmarker)
    t0 = time.time()
    recs, info = video.track_video(args.video, lm, frames_dir, rate=args.fps,
                                   smoother=video.BoxSmoother(max_gap=args.max_gap), progress=print)
    print(f"frames: {info['n_sampled']} sampled at {args.fps} fps, {info['n_faces']} detections, {info['n_crops']} crops, "
          f"{time.time() - t0:.1f}s")
    return recs, info


def infer(args, work: Path, recs: list[dict]) -> Path:
    frames = video.frames_with_crops(recs)
    acts_dir = work / "acts"
    chunks = store.plan_chunks(len(frames), args.chunk_frames)
    todo = [c for c in chunks if args.force or not store.chunk_done(acts_dir, c[0])]
    print(f"acts: {len(chunks)} chunks, {len(todo)} to compute")
    if not todo:
        return acts_dir
    from modal_app import video as mv

    frames_dir = work / "frames"
    specs, payloads = [], []
    for k, s, e in todo:
        part = frames[s:e]
        specs.append({"video_id": work.name, "chunk": k, "files": [f["file"] for f in part],
                      "frame_index": [f["frame_index"] for f in part], "t": [f["t"] for f in part],
                      "store": "", "crop": args.crop, "budget": args.budget})
        payloads.append({f["file"]: (frames_dir / f["file"]).read_bytes() for f in part})
    t0 = time.time()
    jobs = [([s], p) for s, p in zip(specs, payloads)]
    deployed = None if args.ephemeral else deployed_infer(mv.APP, args.env)
    if deployed is not None:
        print(f"acts: using the deployed {mv.APP} app (its containers keep the model loaded between takes; "
              "redeploy after code changes: modal deploy modal_app/video.py --env <env>)")
        results = list(deployed().run.starmap(jobs, kwargs={"batch_size": args.batch_size}))
    else:
        with mv.app.run(environment_name=args.env):
            results = list(mv.Infer().run.starmap(jobs, kwargs={"batch_size": args.batch_size}))
    gpu_s = 0.0
    for res in results:
        for r in res:
            with np.load(io.BytesIO(r["npz"]), allow_pickle=False) as z:
                store.save_chunk(acts_dir, r["chunk"], {k: z[k] for k in z.files})
            gpu_s += r["seconds"]
    print(f"acts: {sum(len(s['files']) for s in specs)} frames in {time.time() - t0:.0f}s wall, "
          f"{gpu_s:.0f}s GPU compute (plus model load), {results[0][0]['gpu']}")
    return acts_dir


def deployed_infer(app_name: str, env: str):
    """The Infer class of a deployed app (`modal deploy modal_app/video.py --env <env>`), or None.

    An ephemeral `app.run()` tears its containers down when the call returns, so
    every take pays the model load. A deployed app keeps a container for
    SCALEDOWN_S after the last call, which is what makes per-take processing
    during a recording session cheap. The deployed code is whatever was last
    deployed, not the working tree.
    """
    import modal
    try:
        cls = modal.Cls.from_name(app_name, "Infer", environment_name=env)
        cls.hydrate()
        return cls
    except modal.exception.NotFoundError:
        return None


def project(args, work: Path, recs: list[dict], acts_dir: Path, info: dict) -> Path:
    acts = store.load_store(acts_dir)
    cfg = store.ProjectionConfig(readout=args.readout, score_layer=args.score_layer, valence_layer=args.valence_layer,
                                 n_pcs=args.pcs)
    spaces = store.load_spaces(args.gemotions, [cfg.score_layer, cfg.valence_layer], cfg.n_pcs)
    refs, _ = store.load_reference(args.reference)
    proj = store.project(acts, spaces, refs, cfg)
    if args.save_baseline:
        store.fit_baseline(proj, source=str(args.video)).save(args.save_baseline)
        print("baseline: saved", args.save_baseline)
    if args.baseline == "session":
        proj = store.apply_baseline(proj, store.fit_baseline(proj, source="session"))
    elif args.baseline not in ("", "none"):
        proj = store.apply_baseline(proj, store.Baseline.load(args.baseline))
    scfg = store.SeriesConfig(window=args.window, ema_tau=args.ema, projection=cfg)
    cols = store.build_series(proj, scfg, frames=recs)
    meta = store.series_metadata(scfg, {"video": str(args.video), "fps_sampled": args.fps, "crop": args.crop,
                                        "budget": args.budget, "baseline": args.baseline, "reference": args.reference,
                                        "duration_s": info.get("duration"), "source_fps": info.get("fps")})
    out = work / f"{work.name}.parquet"
    store.write_parquet(out, cols, meta)
    v = cols["valence"][np.isfinite(cols["valence"])]
    print(f"series: {len(cols['t'])} rows -> {out}; valence mean {v.mean():+.2f} sd {v.std():.2f} "
          f"(range {v.min():+.2f} .. {v.max():+.2f}); top at mid-clip: {cols['top1'][len(v) // 2]}, {cols['top2'][len(v) // 2]}")
    if args.rrd:
        from cogmirror import rerun_export

        rrd = rerun_export.rrd_path(args.rrd_dir, work)
        rerun_export.export(work, rrd, traces=args.traces)
        print(f"rerun -> {rrd}  (open with: .venv/bin/python -m rerun {rrd})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--work", default="data/video", help="working directory root")
    ap.add_argument("--name", default="", help="video id (default: file stem)")
    ap.add_argument("--fps", type=float, default=video.DEFAULT_RATE)
    ap.add_argument("--max-gap", type=float, default=video.DEFAULT_MAX_GAP, help="seconds to hold the box through missed detections")
    ap.add_argument("--crop", default="medium", choices=video.CROP_VARIANTS)
    ap.add_argument("--budget", type=int, default=280)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-frames", type=int, default=300)
    ap.add_argument("--readout", default="last", choices=store.READOUTS)
    ap.add_argument("--score-layer", type=int, default=store.DEFAULT_SCORE_LAYER)
    ap.add_argument("--valence-layer", type=int, default=store.DEFAULT_VALENCE_LAYER)
    ap.add_argument("--pcs", type=int, default=store.DEFAULT_N_PCS)
    ap.add_argument("--window", type=float, default=store.DEFAULT_WINDOW, help="centered averaging window, seconds")
    ap.add_argument("--ema", type=float, default=0.0, help="exponential smoothing time constant, seconds (0 = off)")
    ap.add_argument("--baseline", default="none", help="none | session | path to a baseline json")
    ap.add_argument("--save-baseline", default="", help="write this video's mean as a baseline json")
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--gemotions", default=GEMOTIONS)
    ap.add_argument("--landmarker", default=LANDMARKER)
    ap.add_argument("--env", default="cogmirror", help="Modal environment")
    ap.add_argument("--ephemeral", action="store_true", help="run the Modal app ephemerally from the working tree even if one is deployed")
    ap.add_argument("--rrd", action="store_true", help="also export a Rerun recording")
    ap.add_argument("--rrd-dir", default="data/rerun")
    ap.add_argument("--traces", default="default", help="traces for the Rerun blueprint (see scripts/rerun_export.py)")
    ap.add_argument("--track-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute every step")
    args = ap.parse_args()

    name = args.name or Path(args.video).stem
    work = Path(args.work) / name
    work.mkdir(parents=True, exist_ok=True)
    recs, info = track(args, work)
    if args.track_only:
        return
    acts_dir = infer(args, work, recs)
    project(args, work, recs, acts_dir, info)


if __name__ == "__main__":
    main()
