"""Export processed videos as Rerun recordings and open them in the viewer.

    python scripts/rerun_export.py data/video/<id>            # one video
    python scripts/rerun_export.py --all                       # every work dir under data/video
    python scripts/rerun_export.py --all --open                # ... and open the viewer on all of them
    python scripts/rerun_export.py --all --traces valence,pc2,happy,sad,angry --y-range 4

In the viewer, the Recordings panel (top left) lists every exported video;
pick one to switch. The blueprint panel toggles traces and views. Traces not
in the stacked layout are still in the recording under `traces/`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cogmirror import rerun_export as RE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dirs", nargs="*", help="work directories written by scripts/video_run.py")
    ap.add_argument("--all", action="store_true", help="every work dir under --work")
    ap.add_argument("--work", default="data/video")
    ap.add_argument("--out", default="data/rerun")
    ap.add_argument("--traces", default="default",
                    help="comma list of trace names, or: default (valence, PCs, the most varying emotions), all, pcs, emotions")
    ap.add_argument("--top-k", type=int, default=RE.DEFAULT_TOP_K, help="emotions picked by variance for 'default'")
    ap.add_argument("--y-range", type=float, default=RE.DEFAULT_Y_RANGE, help="shared y-range (reference SD units)")
    ap.add_argument("--no-video", action="store_true", help="skip the source video (crops are always included)")
    ap.add_argument("--take-log", default="", help="recorder take log JSON, for sentence boundaries (single video)")
    ap.add_argument("--force", action="store_true", help="re-export existing .rrd files")
    ap.add_argument("--open", action="store_true", help="open the Rerun viewer on the exported files")
    ap.add_argument("--web", action="store_true", help="with --open: serve the web viewer instead of the native app")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.work_dirs]
    if args.all:
        dirs += sorted(p for p in Path(args.work).iterdir() if p.is_dir() and (p / f"{p.name}.parquet").exists())
    if not dirs:
        ap.error("give work directories or --all")
    out_dir = Path(args.out)
    outputs = []
    for d in dirs:
        out = out_dir / f"{d.name}.rrd"
        if out.exists() and not args.force:
            print(f"kept    {out}")
        else:
            s = RE.export(d, out, traces=args.traces, top_k=args.top_k, y_range=args.y_range,
                          include_video=not args.no_video, take_log=args.take_log or None)
            print(f"wrote   {out}  ({s['frames']} frames, video {'yes' if s['video'] else 'no'}, traces: {', '.join(s['traces'])})")
        outputs.append(out)
    if args.open:
        cmd = [sys.executable, "-m", "rerun"] + (["--web-viewer"] if args.web else []) + [str(o) for o in outputs]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
