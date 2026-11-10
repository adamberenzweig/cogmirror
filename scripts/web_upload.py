"""Upload locally recorded takes to the hosted recorder (modal_app/web.py), which processes them.

    .venv/bin/python scripts/web_upload.py --url https://<app>.modal.run --session 20261109-2055
    .venv/bin/python scripts/web_upload.py --url ... --session 20261109-2055 --as test-20261109 --takes 003_s027_read,004_s006_read

Kept takes only (the log is uploaded after the video, as the recorder does, so
processing starts on the log). The calibration take goes first. The access key
is read from data/web_access_key.txt or --key. Takes that are already on the
server with a status are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogmirror import align  # noqa: E402


def request(url: str, key: str, data: bytes | None = None, ctype: str = "application/octet-stream") -> dict:
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(f"{url}{sep}key={key}", data=data, method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("content-type", ctype)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="the deployed app's URL")
    ap.add_argument("--session", required=True, help="local session id under data/recordings")
    ap.add_argument("--as", dest="remote", default="", help="session id on the server (default: same)")
    ap.add_argument("--takes", default="", help="comma list of take ids (default: every kept take)")
    ap.add_argument("--key", default="", help="access key (default: data/web_access_key.txt)")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--force", action="store_true", help="re-upload takes the server already has")
    ap.add_argument("--wait", action="store_true", help="poll /api/status until the uploaded takes are done")
    args = ap.parse_args()

    key = args.key or Path(args.data, "web_access_key.txt").read_text().strip()
    url = args.url.rstrip("/")
    remote = args.remote or args.session
    takes = align.find_takes(Path(args.data) / "recordings", sessions=[args.session])
    takes = [t for t in takes if t.video_path and t.video_path.exists()]
    if args.takes:
        wanted = set(args.takes.split(","))
        takes = [t for t in takes if t.take_id in wanted]
    takes.sort(key=lambda t: (t.condition != align.CALIBRATION, t.take_index or 0))
    if not takes:
        print("nothing to upload")
        return 1
    have = {r["take_id"] for r in request(f"{url}/api/status", key)["rows"] if r["session_id"] == remote and r["stage"] != "no video"}
    uploaded = []
    for t in takes:
        if t.take_id in have and not args.force:
            print(f"  {t.take_id}: already on the server, skipped")
            continue
        log = json.loads(t.path.read_text())
        log["session_id"] = remote
        print(f"  {t.take_id}: {t.video_path.stat().st_size / 1e6:.1f} MB", end=" ", flush=True)
        r1 = request(f"{url}/api/take/{remote}/{t.take_id}.webm", key, t.video_path.read_bytes())
        r2 = request(f"{url}/api/take/{remote}/{t.take_id}.json", key, json.dumps(log).encode(), "application/json")
        print(f"-> {r1.get('bytes')} bytes, processing={r2.get('processing')}")
        uploaded.append(t.take_id)
    print(f"{len(uploaded)} take(s) uploaded to session {remote}; status: {url}/status")
    if args.wait and uploaded:
        while True:
            rows = {r["take_id"]: r for r in request(f"{url}/api/status", key)["rows"] if r["session_id"] == remote}
            line = "; ".join(f"{k}={rows.get(k, {}).get('stage', '?')}" for k in uploaded)
            print(time.strftime("%H:%M:%S"), line, flush=True)
            if all(rows.get(k, {}).get("stage") in ("done", "failed") for k in uploaded):
                break
            time.sleep(15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
