"""Local launcher for the M3 webcam recorder.

    .venv/bin/python tools/recorder/serve.py            # http://127.0.0.1:8770
    .venv/bin/python tools/recorder/serve.py --port 9000 --data /some/other/data

Serves the static app, the story set (data/stories), the narration audio, and
accepts uploaded takes into data/recordings/<session_id>/<take_id>.{webm,json}.
Standard library only (plus cogmirror.recorder for the take-log helpers,
which the hosted version in modal_app/web.py shares).
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
sys.path.insert(0, str(REPO / "src"))

from cogmirror.recorder import SAFE_ID, load_stories, scan_recordings  # noqa: E402


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
