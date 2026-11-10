"""Export processed videos as Rerun recordings and open them in the viewer.

    python scripts/rerun_export.py data/video/<id>            # one video
    python scripts/rerun_export.py --all                       # every work dir under data/video
    python scripts/rerun_export.py --all --open                # ... and open them in the browser viewer
    python scripts/rerun_export.py --all --open --native       # ... or in the native app (needs ffmpeg for video)
    python scripts/rerun_export.py --all --traces valence,pc2,happy,sad,angry --y-range 4

In the viewer, the Recordings panel (top left) lists every exported video;
pick one to switch. The blueprint panel toggles traces and views. Traces not
in the stacked layout are still in the recording under `traces/`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote

from cogmirror import rerun_export as RE

WEB_PORT = 9090
GRPC_PORT = 9876


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dirs", nargs="*", help="work directories written by scripts/video_run.py")
    ap.add_argument("--all", action="store_true", help="every processed video under --work (single runs and recorded sessions)")
    ap.add_argument("--work", default="data/video")
    ap.add_argument("--out", default="data/rerun")
    ap.add_argument("--traces", default="default",
                    help="comma list of trace names, or: default (valence, PCs, the most varying emotions), all, pcs, emotions")
    ap.add_argument("--top-k", type=int, default=RE.DEFAULT_TOP_K, help="emotions in the default selection: the most frequent top-3 labels over the clip (plus the story targets)")
    ap.add_argument("--y-range", type=float, default=RE.DEFAULT_Y_RANGE, help="shared y-range (reference SD units)")
    ap.add_argument("--no-video", action="store_true", help="skip the source video (crops are always included)")
    ap.add_argument("--take-log", default="", help="recorder take log JSON, for sentence boundaries (single video)")
    ap.add_argument("--force", action="store_true", help="re-export existing .rrd files")
    ap.add_argument("--open", action="store_true", help="open the exported files in the Rerun viewer (browser)")
    ap.add_argument("--native", action="store_true", help="with --open: use the native Rerun app instead of the browser")
    ap.add_argument("--web-port", type=int, default=WEB_PORT, help="port for the browser viewer")
    ap.add_argument("--grpc-port", type=int, default=GRPC_PORT, help="port of the Rerun gRPC server the browser connects to")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.work_dirs]
    if args.all:
        # data/video/<id>/<id>.parquet (single runs) and data/video/<session>/<take>/<take>.parquet (recorded sessions)
        dirs += sorted({f.parent for pat in ("*/*.parquet", "*/*/*.parquet") for f in Path(args.work).glob(pat) if f.stem == f.parent.name})
    if not dirs:
        ap.error("give work directories or --all")
    out_dir = Path(args.out)
    outputs = []
    for d in dirs:
        out = RE.rrd_path(out_dir, d)
        if out.exists() and not args.force:
            print(f"kept    {out}")
        else:
            s = RE.export(d, out, traces=args.traces, top_k=args.top_k, y_range=args.y_range,
                          include_video=not args.no_video, take_log=args.take_log or None)
            print(f"wrote   {out}  ({s['frames']} frames, video {'yes' if s['video'] else 'no'}, traces: {', '.join(s['traces'])})")
        outputs.append(out)
    if args.open:
        open_viewer(outputs, native=args.native, web_port=args.web_port, grpc_port=args.grpc_port)
    return 0


def open_viewer(files: list[Path], native: bool = False, web_port: int = WEB_PORT, grpc_port: int = GRPC_PORT) -> None:
    """Open the recordings in the Rerun viewer.

    The browser viewer is the default: the native app needs an ffmpeg executable to
    decode the embedded video, the browser does not. `python -m rerun --serve-web`
    serves the viewer over HTTP and the recordings over gRPC; the browser must be
    pointed at the gRPC proxy explicitly, it does not find it on its own. If a server
    is already running on the gRPC port, rerun streams the files to it and exits, and
    the tab already open shows them.
    """
    cmd = [sys.executable, "-m", "rerun"]
    if native:
        cmd += [str(f) for f in files]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=False)
        return
    cmd += ["--serve-web", "--web-viewer-port", str(web_port), "--port", str(grpc_port)] + [str(f) for f in files]
    url = f"http://localhost:{web_port}/?url=" + quote(f"rerun+http://localhost:{grpc_port}/proxy", safe="")
    # A server started by an earlier --open does not pick up new files reliably, so replace it.
    replaced = stop_rerun_server(grpc_port)
    print("$", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    time.sleep(1.5)
    if proc.poll() is None:
        print(f"viewer: {url}  (Ctrl-C stops the server)", flush=True)
        if replaced:
            print("viewer: replaced the previous server; refresh the tab that was already open", flush=True)
        else:
            webbrowser.open(url)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        print(f"viewer: could not start a server on port {grpc_port} (something else is listening); "
              f"stop it and rerun, or pass --grpc-port/--web-port", flush=True)


def stop_rerun_server(grpc_port: int) -> bool:
    """Terminate a Rerun server from an earlier --open that still holds the gRPC port.

    Only a process whose command line is a Rerun `--serve-web` is touched; anything
    else on the port is left alone (and the new server will then fail to bind).
    """
    try:
        pids = subprocess.run(["lsof", "-nP", f"-iTCP:{grpc_port}", "-sTCP:LISTEN", "-t"],
                              capture_output=True, text=True).stdout.split()
    except FileNotFoundError:
        return False
    stopped = False
    for pid in pids:
        cmdline = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
        if "rerun" in cmdline.lower() and "--serve-web" in cmdline:
            subprocess.run(["kill", pid], check=False)
            stopped = True
    if stopped:
        for _ in range(20):
            time.sleep(0.25)
            if not subprocess.run(["lsof", "-nP", f"-iTCP:{grpc_port}", "-sTCP:LISTEN", "-t"],
                                  capture_output=True, text=True).stdout.strip():
                break
    return stopped


if __name__ == "__main__":
    sys.exit(main())
