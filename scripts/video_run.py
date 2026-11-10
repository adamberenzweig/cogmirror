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

The steps themselves are in cogmirror.pipeline; this is the local entry point
that runs the activations on Modal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cogmirror import pipeline, rerun_export, store, video  # noqa: E402

LANDMARKER = "data/models/face_landmarker.task"
REFERENCE = "results/m1/reference_a2_prefill.npz"
GEMOTIONS = "data/gemotions"


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


def modal_run_jobs(env: str, ephemeral: bool):
    """`run_jobs` for the pipeline: the deployed Infer class when there is one, else an ephemeral app."""
    from modal_app import video as mv

    deployed = None if ephemeral else deployed_infer(mv.APP, env)

    def run_jobs(jobs, batch_size):
        if deployed is not None:
            print(f"acts: using the deployed {mv.APP} app (its containers keep the model loaded between takes; "
                  "redeploy after code changes: modal deploy modal_app/video.py --env <env>)")
            return list(deployed().run.starmap(jobs, kwargs={"batch_size": batch_size}))
        with mv.app.run(environment_name=env):
            return list(mv.Infer().run.starmap(jobs, kwargs={"batch_size": batch_size}))

    return run_jobs


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
    s = pipeline.Settings(fps=args.fps, max_gap=args.max_gap, crop=args.crop, budget=args.budget, batch_size=args.batch_size,
                          chunk_frames=args.chunk_frames, readout=args.readout, score_layer=args.score_layer,
                          valence_layer=args.valence_layer, n_pcs=args.pcs, window=args.window, ema=args.ema,
                          landmarker=args.landmarker, gemotions=args.gemotions, reference=args.reference)
    if args.track_only:
        work.mkdir(parents=True, exist_ok=True)
        pipeline.track(args.video, work, s, args.force)
        return
    rrd = rerun_export.rrd_path(args.rrd_dir, work) if args.rrd else None
    pipeline.process(args.video, work, modal_run_jobs(args.env, args.ephemeral), s, baseline=args.baseline,
                     save_baseline=args.save_baseline, rrd=rrd, traces=args.traces, force=args.force)
    if rrd is not None:
        print(f"(open with: .venv/bin/python -m rerun {rrd})")


if __name__ == "__main__":
    main()
