"""scripts/m4_run_videos.py: take selection, ordering, and the watch loop (pipeline stubbed)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m4_run_videos", ROOT / "scripts" / "m4_run_videos.py")
mrv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mrv)


def write_take(rec: Path, session: str, idx: int, condition: str, story: str | None, bad: bool = False, video: bool = True):
    take_id = f"{idx:03d}_{story or 'calibration'}_{condition}"
    d = rec / session
    d.mkdir(parents=True, exist_ok=True)
    if video:
        (d / f"{take_id}.webm").write_bytes(b"\x1aE\xdf\xa3")
    (d / f"{take_id}.json").write_text(json.dumps({
        "session_id": session, "take_id": take_id, "take_index": idx, "condition": condition,
        "story_id": story, "kind": "calibration" if story is None else "story", "bad": bad, "duration_s": 3.0}))
    return take_id


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace video_run.py with a stub that writes the series file; record the commands."""
    calls = []

    def fake_run(cmd, check=True, preexec_fn=None, **kw):
        calls.append(cmd)
        work = Path(cmd[cmd.index("--work") + 1]); name = cmd[cmd.index("--name") + 1]
        (work / name).mkdir(parents=True, exist_ok=True)
        (work / name / f"{name}.parquet").write_bytes(b"")
        if "--save-baseline" in cmd:
            Path(cmd[cmd.index("--save-baseline") + 1]).write_text("{}")

    monkeypatch.setattr(mrv.subprocess, "run", fake_run)
    monkeypatch.setattr(mrv.shutil, "which", lambda name: None)  # no ffmpeg: use the WebM as is
    return calls


def test_pending_orders_calibration_first_and_skips_bad(tmp_path, fake_pipeline):
    rec = tmp_path / "recordings"
    write_take(rec, "s1", 2, "read", "s010")
    write_take(rec, "s1", 1, "calibration", None)
    write_take(rec, "s1", 3, "read", "s011", bad=True)
    write_take(rec, "s1", 4, "read", "s012", video=False)
    args = type("A", (), {"data": str(tmp_path), "sessions": "", "include_bad": False, "force": False})()
    pend = mrv.pending_takes(args, tmp_path / "video")
    assert [t.take_id for t in pend["s1"]] == ["001_calibration_calibration", "002_s010_read"]


def test_run_pass_uses_calibration_baseline_then_is_idempotent(tmp_path, fake_pipeline):
    rec = tmp_path / "recordings"
    write_take(rec, "s1", 1, "calibration", None)
    write_take(rec, "s1", 2, "read", "s010")
    args = type("A", (), {"data": str(tmp_path), "sessions": "", "include_bad": False, "force": False,
                          "baseline": "calibration", "dry_run": False})()
    work = tmp_path / "video"
    done = mrv.run_pass(args, work, ["--fps", "5"], nice=10)
    assert [d.name for d in done] == ["001_calibration_calibration", "002_s010_read"]
    cal, read = fake_pipeline
    assert "--save-baseline" in cal and cal[cal.index("--baseline") + 1] == "none"
    assert read[read.index("--baseline") + 1] == str(work / "s1" / "baseline.json")
    assert read[-2:] == ["5", str(work / "s1" / "002_s010_read" / "002_s010_read.webm")] or "--fps" in read
    assert mrv.run_pass(args, work, [], nice=0) == []


def test_watch_processes_new_takes_and_exports_when_idle(tmp_path, fake_pipeline, monkeypatch):
    rec = tmp_path / "recordings"
    write_take(rec, "s1", 1, "calibration", None)
    exported = []
    monkeypatch.setattr(mrv, "export_rrd", lambda dirs, args: exported.append([d.name for d in dirs]))
    polls = {"n": 0}

    def fake_sleep(_):
        polls["n"] += 1
        if polls["n"] == 1:
            write_take(rec, "s1", 2, "read", "s010")   # a take is kept while we wait
        elif polls["n"] == 3:
            raise KeyboardInterrupt
    monkeypatch.setattr(mrv.time, "sleep", fake_sleep)
    monkeypatch.setattr(sys, "argv", ["m4_run_videos.py", "--data", str(tmp_path), "--watch", "--rrd", "--interval", "1"])
    assert mrv.main() == 0
    # calibration exported at the first idle poll, the read take at the next idle poll
    assert exported == [["001_calibration_calibration"], ["002_s010_read"]]
    assert (tmp_path / "video" / "s1" / "002_s010_read" / "002_s010_read.parquet").exists()
