"""List every recorded take as one JSON line, with its per-sentence timestamps.

    .venv/bin/python scripts/recordings_manifest.py                       # -> results/recordings_manifest.jsonl
    .venv/bin/python scripts/recordings_manifest.py --data other/data -o - # to stdout

One line per take (calibration takes included), in the spirit of results/m1/manifest.jsonl:
session_id, take_id, story_id, condition, kind, emotions, clusters, bad, duration_s,
video file (relative to --data), camera settings, and `segments`: one entry per sentence
with start/end seconds relative to the recording start, the sentence text, and whether the
story's scripted turn begins at that sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def take_row(log_path: Path, data: Path) -> dict:
    log = json.loads(log_path.read_text())
    webm = log_path.with_suffix(".webm")
    turn = log.get("turn") or {}
    segments = []
    for st in log.get("sentence_times", []):
        i = st["sentence_idx"]
        segments.append({
            "sentence_idx": i,
            "start": st["start"],
            "end": st["end"],
            "text": log["sentences"][i] if i < len(log.get("sentences", [])) else None,
            "turn_starts_here": bool(turn) and turn.get("sentence_idx") == i,
        })
    cam = (log.get("camera") or {}).get("video") or {}
    return {
        "session_id": log.get("session_id", log_path.parent.name),
        "take_id": log.get("take_id", log_path.stem),
        "take_index": log.get("take_index"),
        "story_id": log.get("story_id"),
        "condition": log.get("condition"),
        "kind": log.get("kind"),
        "emotions": log.get("emotions", []),
        "clusters": log.get("clusters", []),
        "bad": bool(log.get("bad")),
        "note": log.get("note", ""),
        "start_wallclock": log.get("start_wallclock"),
        "duration_s": log.get("duration_s"),
        "n_sentences": len(log.get("sentences", [])),
        "turn": log.get("turn"),
        "video": str(webm.relative_to(data)) if webm.exists() else None,
        "video_bytes": webm.stat().st_size if webm.exists() else 0,
        "log": str(log_path.relative_to(data)),
        "frame_w": cam.get("width"), "frame_h": cam.get("height"), "fps": cam.get("frameRate"),
        "mime_type": (log.get("recorder") or {}).get("mimeType"),
        "fake_stream": bool((log.get("camera") or {}).get("fake_stream")),
        "segments": segments,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("-o", "--out", default=str(ROOT / "results" / "recordings_manifest.jsonl"), help="'-' for stdout")
    ap.add_argument("--include-bad", action="store_true", help="also list takes marked bad")
    args = ap.parse_args()
    data = Path(args.data).resolve()
    rows = [take_row(p, data) for p in sorted((data / "recordings").glob("*/*.json"))]
    if not args.include_bad:
        rows = [r for r in rows if not r["bad"]]
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        good = sum(1 for r in rows if r["video"])
        by_cond = {}
        for r in rows:
            by_cond[r["condition"]] = by_cond.get(r["condition"], 0) + 1
        print(f"{len(rows)} takes ({good} with video) -> {out}; by condition: {by_cond}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
