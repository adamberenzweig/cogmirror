"""Local launcher for the M3 webcam recorder.

    .venv/bin/python tools/recorder/serve.py            # http://127.0.0.1:8770
    .venv/bin/python tools/recorder/serve.py --port 9000 --data /some/other/data

Serves the static app, the story set (data/stories), the narration audio, and
accepts uploaded takes into data/recordings/<session_id>/<take_id>.{webm,json}.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATIC = HERE / "static"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def scan_recordings(rec_dir: Path) -> dict:
    """Every take log under data/recordings, plus the set of completed (story, condition) pairs."""
    takes = []
    for p in sorted(rec_dir.glob("*/*.json")):
        try:
            log = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        webm = p.with_suffix(".webm")
        takes.append({
            "session_id": log.get("session_id", p.parent.name),
            "take_id": log.get("take_id", p.stem),
            "story_id": log.get("story_id"),
            "condition": log.get("condition"),
            "bad": bool(log.get("bad")),
            "duration_s": log.get("duration_s"),
            "has_video": webm.exists(),
            "video_bytes": webm.stat().st_size if webm.exists() else 0,
            "start_wallclock": log.get("start_wallclock"),
        })
    done = sorted({(t["story_id"], t["condition"]) for t in takes
                   if t["story_id"] and not t["bad"] and t["has_video"] and t["condition"] in ("read", "listen")})
    sessions = sorted({t["session_id"] for t in takes})
    return {"takes": takes, "done": [list(d) for d in done], "sessions": sessions}


def load_stories(stories_dir: Path) -> list[dict]:
    path = stories_dir / "stories.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        idx = stories_dir / "audio" / r["id"] / "index.json"
        r["audio"] = json.loads(idx.read_text()) if idx.exists() else None
        out.append(r)
    return out


class Handler(BaseHTTPRequestHandler):
    data: Path  # set on the class by main()
    protocol_version = "HTTP/1.1"

    # -- helpers ---------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _file(self, path: Path) -> None:
        if not path.is_file():
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".m4a":
            ctype = "audio/mp4"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "none")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # quieter log
        if self.command == "POST" or "/api/" in str(args[0]):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        if path.startswith("/static/"):
            rel = Path(path[len("/static/"):])
            if ".." in rel.parts:
                return self._send(HTTPStatus.FORBIDDEN, b"", "text/plain")
            return self._file(STATIC / rel)
        if path == "/api/stories":
            return self._json(load_stories(self.data / "stories"))
        if path == "/api/progress":
            rec = self.data / "recordings"
            rec.mkdir(parents=True, exist_ok=True)
            return self._json(scan_recordings(rec))
        m = re.match(r"^/audio/([^/]+)/(\d+)\.m4a$", path)
        if m and SAFE_ID.match(m.group(1)):
            return self._file(self.data / "stories" / "audio" / m.group(1) / f"{m.group(2)}.m4a")
        return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        m = re.match(r"^/api/take/([^/]+)/([^/]+)\.(webm|json)$", path)
        if not m or not SAFE_ID.match(m.group(1)) or not SAFE_ID.match(m.group(2)):
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        session, take, ext = m.groups()
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n) if n else b""
        d = self.data / "recordings" / session
        d.mkdir(parents=True, exist_ok=True)
        target = d / f"{take}.{ext}"
        if ext == "json":
            try:
                obj = json.loads(body.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._json({"ok": False, "error": "bad json"}, 400)
            target.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        else:
            target.write_bytes(body)
        self._json({"ok": True, "path": str(target.relative_to(self.data.parent) if self.data.parent in target.parents else target),
                    "bytes": len(body)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data", default=str(REPO / "data"), help="folder holding stories/ and recordings/")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    Handler.data = Path(args.data).resolve()
    (Handler.data / "recordings").mkdir(parents=True, exist_ok=True)
    n = len(load_stories(Handler.data / "stories"))
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"recorder at {url}  ({n} stories from {Handler.data / 'stories'}; takes -> {Handler.data / 'recordings'})")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
