"""Take logs on disk: what the recorder server and the hosted status page read.

A take is `<recordings>/<session>/<take>.webm` plus `<take>.json` (the log,
written after the review, so `bad` is final). Processing status, when a take
is processed by the hosted pipeline, is `<status>/<session>/<take>.json`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
CONDITIONS = ("read", "listen")
STAGES = ("uploaded", "queued", "tracking", "model", "series", "export", "done", "failed")


def scan_recordings(rec_dir: str | Path) -> dict:
    """Every take log under recordings/, plus the set of completed (story, condition) pairs."""
    takes = []
    for p in sorted(Path(rec_dir).glob("*/*.json")):
        try:
            log = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        webm = p.with_suffix(".webm")
        takes.append({
            "session_id": log.get("session_id", p.parent.name),
            "take_id": log.get("take_id", p.stem),
            "story_id": log.get("story_id"),
            "condition": log.get("condition"),
            "kind": log.get("kind"),
            "emotions": list(log.get("emotions") or []),
            "bad": bool(log.get("bad")),
            "duration_s": log.get("duration_s"),
            "has_video": webm.exists(),
            "video_bytes": webm.stat().st_size if webm.exists() else 0,
            "start_wallclock": log.get("start_wallclock"),
            "note": log.get("note", ""),
        })
    done = sorted({(t["story_id"], t["condition"]) for t in takes
                   if t["story_id"] and not t["bad"] and t["has_video"] and t["condition"] in CONDITIONS})
    sessions = sorted({t["session_id"] for t in takes})
    return {"takes": takes, "done": [list(d) for d in done], "sessions": sessions}


def load_stories(stories_dir: str | Path) -> list[dict]:
    stories_dir = Path(stories_dir)
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


# --------------------------------------------------------------------------
# Processing status (hosted pipeline)
# --------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def status_path(status_dir: str | Path, session: str, take: str) -> Path:
    return Path(status_dir) / session / f"{take}.json"


def write_status(status_dir: str | Path, session: str, take: str, stage: str, **fields) -> dict:
    """Advance a take's status file; earlier fields are kept, `stage` and `updated` replaced."""
    p = status_path(status_dir, session, take)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = read_status(p) or {"session_id": session, "take_id": take, "created": now_iso()}
    cur.update(fields)
    cur["stage"] = stage
    cur["updated"] = now_iso()
    if stage == "done" or stage == "failed":
        cur["finished"] = cur["updated"]
    p.write_text(json.dumps(cur, indent=1))
    return cur


def read_status(path: str | Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def scan_status(status_dir: str | Path) -> dict[tuple[str, str], dict]:
    out = {}
    for p in sorted(Path(status_dir).glob("*/*.json")):
        s = read_status(p)
        if s:
            out[(p.parent.name, p.stem)] = s
    return out


def status_rows(progress: dict, status: dict[tuple[str, str], dict]) -> list[dict]:
    """One row per take, newest session first: the log's facts plus the processing stage."""
    rows = []
    for t in progress["takes"]:
        key = (t["session_id"], t["take_id"])
        s = status.get(key, {})
        stage = s.get("stage") or ("uploaded" if t["has_video"] else "no video")
        if t["bad"]:
            stage = "bad (not processed)"
        rows.append({**t, "stage": stage, "error": s.get("error"), "timings": s.get("timings", {}),
                     "updated": s.get("updated"), "rrd": s.get("rrd"), "summary": s.get("summary")})
    rows.sort(key=lambda r: (r["session_id"], r["take_id"]))
    rows.reverse()
    return rows
