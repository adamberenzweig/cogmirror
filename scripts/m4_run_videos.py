"""Run the video pipeline on every recorded take (glue around scripts/video_run.py).

    .venv/bin/python scripts/m4_run_videos.py [--sessions s1,s2] [--dry-run] [--baseline calibration|session|none] [--rrd]
    .venv/bin/python scripts/m4_run_videos.py --watch --rrd     # during a recording session

For each take log under data/recordings/<session>/ with a video:
  1. remux the WebM with ffmpeg (`-c copy`) into data/video/<session>/<take>/<take>.webm,
     because Chromium writes no duration header (see tools/recorder/README.md)
  2. scripts/video_run.py on it with --work data/video/<session> --name <take>, so the
     series lands at data/video/<session>/<take>/<take>.parquet where m4_agreement.py looks
  3. with --baseline calibration (default), the session's calibration take runs first and
     its mean is saved as data/video/<session>/baseline.json, then subtracted from every take

Takes whose series already exists are skipped. Pass --force to redo. --rrd exports a
Rerun recording per take to --rrd-dir once the takes are processed. Every other
option is forwarded to video_run.py (fps, crop, window, env, ...).

--watch keeps polling the recordings folder and processes each take as soon as the
recorder has saved it as kept (the log is written after the review, so a take marked
bad is never picked up). Local work runs at low CPU priority so it does not compete
with the recording, and the Rerun exports (which transcode the video) wait until no
take is pending. Ctrl-C stops it after finishing the exports.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogmirror import align  # noqa: E402


def remux(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    if shutil.which("ffmpeg") is None:
        print("  ffmpeg not found; using the original WebM (duration header missing, frame timing still fine)")
        return src
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c", "copy", str(dst)], check=True)
    return dst


def series_path(work: Path, t: align.Take) -> Path:
    return work / t.session_id / t.take_id / f"{t.take_id}.parquet"


def pending_takes(args, work: Path) -> dict[str, list[align.Take]]:
    """Kept takes with a video and no series yet, by session, calibration first."""
    takes = align.find_takes(Path(args.data) / "recordings", include_bad=args.include_bad,
                             sessions=args.sessions.split(",") if args.sessions else None)
    by_session: dict[str, list[align.Take]] = {}
    for t in takes:
        if not (t.video_path and t.video_path.exists()):
            continue
        if series_path(work, t).exists() and not args.force:
            continue
        by_session.setdefault(t.session_id, []).append(t)
    for ts in by_session.values():
        ts.sort(key=lambda t: (t.condition != align.CALIBRATION, t.take_index or 0))
    return by_session


def process(t: align.Take, args, work: Path, forward: list[str], nice: int) -> Path | None:
    """Run video_run.py on one take; returns its work dir (None on a dry run)."""
    session_dir = work / t.session_id
    base_file = session_dir / "baseline.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "video_run.py"), "--work", str(session_dir), "--name", t.take_id]
    if t.condition == align.CALIBRATION and args.baseline == "calibration":
        cmd += ["--save-baseline", str(base_file), "--baseline", "none"]
    elif args.baseline == "calibration" and base_file.exists():
        cmd += ["--baseline", str(base_file)]
    elif args.baseline == "session":
        cmd += ["--baseline", "session"]
    else:
        cmd += ["--baseline", "none"]
    if args.force:
        cmd.append("--force")
    cmd += forward
    if args.dry_run:
        print("  " + " ".join(cmd + ["<remuxed " + t.video_path.name + ">"]))
        return None
    video = remux(t.video_path, session_dir / t.take_id / t.video_path.name)
    print(f"  {t.take_id}: {video.name}", flush=True)
    subprocess.run(cmd + [str(video)], check=True, preexec_fn=(lambda: os.nice(nice)) if nice else None)
    return session_dir / t.take_id


def export_rrd(work_dirs: list[Path], args) -> None:
    from cogmirror import rerun_export
    out_dir = Path(args.rrd_dir)
    for d in work_dirs:
        out = rerun_export.rrd_path(out_dir, d)
        rerun_export.export(d, out, traces=args.traces)
        print(f"  rerun -> {out}", flush=True)


def run_pass(args, work: Path, forward: list[str], nice: int) -> list[Path]:
    done = []
    for session, ts in pending_takes(args, work).items():
        print(f"session {session}: {len(ts)} take(s) to process", flush=True)
        for t in ts:
            d = process(t, args, work, forward, nice)
            if d is not None:
                done.append(d)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--work", default="", help="series root (default <data>/video)")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--baseline", default="calibration", choices=("calibration", "session", "none"))
    ap.add_argument("--include-bad", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rrd", action="store_true", help="export a Rerun recording per processed take")
    ap.add_argument("--rrd-dir", default=str(ROOT / "data" / "rerun"))
    ap.add_argument("--traces", default="default", help="traces for the Rerun blueprint (see scripts/rerun_export.py)")
    ap.add_argument("--watch", action="store_true", help="keep polling for new kept takes and process each as it arrives")
    ap.add_argument("--interval", type=float, default=10.0, help="--watch: seconds between polls")
    ap.add_argument("--nice", type=int, default=10, help="CPU priority reduction for the local steps (0 = normal)")
    args, forward = ap.parse_known_args()

    work = Path(args.work) if args.work else Path(args.data) / "video"
    pending_rrd: list[Path] = []

    if not args.watch:
        done = run_pass(args, work, forward, args.nice)
        if not done and not args.dry_run:
            print(f"nothing to process under {Path(args.data) / 'recordings'} (every kept take has a series)")
        if args.rrd and done:
            export_rrd(done, args)
        return 0

    print(f"watching {Path(args.data) / 'recordings'} every {args.interval:g} s; Ctrl-C to stop", flush=True)
    try:
        while True:
            done = run_pass(args, work, forward, args.nice)
            pending_rrd += done
            if done:
                continue  # look again at once: more may have arrived meanwhile
            if args.rrd and pending_rrd:
                print(f"idle: exporting {len(pending_rrd)} Rerun recording(s)", flush=True)
                export_rrd(pending_rrd, args)
                pending_rrd = []
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if args.rrd and pending_rrd:
            print(f"\nstopping: exporting {len(pending_rrd)} Rerun recording(s) first", flush=True)
            export_rrd(pending_rrd, args)
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
