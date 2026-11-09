"""Run the video pipeline on every recorded take (glue around scripts/video_run.py).

    .venv/bin/python scripts/m4_run_videos.py [--sessions s1,s2] [--dry-run] [--baseline calibration|session|none]

For each take log under data/recordings/<session>/ with a video:
  1. remux the WebM with ffmpeg (`-c copy`) into data/video/<session>/<take>/<take>.webm,
     because Chromium writes no duration header (see tools/recorder/README.md)
  2. scripts/video_run.py on it with --work data/video/<session> --name <take>, so the
     series lands at data/video/<session>/<take>/<take>.parquet where m4_agreement.py looks
  3. with --baseline calibration (default), the session's calibration take runs first and
     its mean is saved as data/video/<session>/baseline.json, then subtracted from every take

Takes whose series already exists are skipped. Pass --force to redo. Every
other option is forwarded to video_run.py (fps, crop, window, env, ...).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--work", default="", help="series root (default <data>/video)")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--baseline", default="calibration", choices=("calibration", "session", "none"))
    ap.add_argument("--include-bad", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args, forward = ap.parse_known_args()

    data = Path(args.data)
    work = Path(args.work) if args.work else data / "video"
    takes = align.find_takes(data / "recordings", include_bad=args.include_bad,
                             sessions=args.sessions.split(",") if args.sessions else None)
    takes = [t for t in takes if t.video_path and t.video_path.exists()]
    if not takes:
        print(f"no takes with video under {data / 'recordings'}")
        return 1
    py = sys.executable
    by_session: dict[str, list[align.Take]] = {}
    for t in takes:
        by_session.setdefault(t.session_id, []).append(t)
    for session, ts in by_session.items():
        # calibration first (its mean is the per-person baseline for the session)
        ts.sort(key=lambda t: (t.condition != align.CALIBRATION, t.take_index or 0))
        base_file = work / session / "baseline.json"
        print(f"session {session}: {len(ts)} takes")
        for t in ts:
            out = work / session / t.take_id / f"{t.take_id}.parquet"
            if out.exists() and not args.force:
                print(f"  {t.take_id}: series exists, skipped")
                continue
            cmd = [py, str(ROOT / "scripts" / "video_run.py"), "--work", str(work / session), "--name", t.take_id]
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
                continue
            video = remux(t.video_path, work / session / t.take_id / t.video_path.name)
            print(f"  {t.take_id}: {video.name}")
            subprocess.run(cmd + [str(video)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
