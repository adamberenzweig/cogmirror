import json

from cogmirror import recorder as R


def write_take(rec, session, take, bad=False, video=True, **log):
    d = rec / session
    d.mkdir(parents=True, exist_ok=True)
    if video:
        (d / f"{take}.webm").write_bytes(b"\x1aE\xdf\xa3")
    (d / f"{take}.json").write_text(json.dumps({"session_id": session, "take_id": take, "bad": bad, **log}))


def test_scan_recordings_done_pairs_and_sessions(tmp_path):
    rec = tmp_path / "recordings"
    write_take(rec, "s1", "001_calibration_calibration", condition="calibration", story_id=None, kind="calibration")
    write_take(rec, "s1", "002_s010_read", condition="read", story_id="s010", emotions=["calm"], duration_s=40.0)
    write_take(rec, "s1", "003_s011_read", condition="read", story_id="s011", bad=True)
    write_take(rec, "s2", "002_s012_read", condition="read", story_id="s012", video=False)
    (rec / "s2" / "broken.json").write_text("{not json")
    p = R.scan_recordings(rec)
    assert p["sessions"] == ["s1", "s2"]
    assert p["done"] == [["s010", "read"]]
    assert [t["take_id"] for t in p["takes"]] == ["001_calibration_calibration", "002_s010_read", "003_s011_read", "002_s012_read"]
    assert p["takes"][1]["emotions"] == ["calm"] and p["takes"][1]["video_bytes"] == 4


def test_status_lifecycle_and_rows(tmp_path):
    rec = tmp_path / "recordings"
    st = tmp_path / "status"
    write_take(rec, "s1", "001_calibration_calibration", condition="calibration", story_id=None, kind="calibration")
    write_take(rec, "s1", "002_s010_read", condition="read", story_id="s010")
    write_take(rec, "s1", "003_s011_read", condition="read", story_id="s011", bad=True)
    R.write_status(st, "s1", "001_calibration_calibration", "queued")
    R.write_status(st, "s1", "001_calibration_calibration", "tracking")
    s = R.write_status(st, "s1", "001_calibration_calibration", "done", rrd="s1_001.rrd", timings={"tracking": 3.0})
    assert s["stage"] == "done" and s["finished"] and s["rrd"] == "s1_001.rrd" and "created" in s
    R.write_status(st, "s1", "002_s010_read", "failed", error="boom")
    rows = R.status_rows(R.scan_recordings(rec), R.scan_status(st))
    by = {r["take_id"]: r for r in rows}
    assert by["001_calibration_calibration"]["stage"] == "done" and by["001_calibration_calibration"]["timings"] == {"tracking": 3.0}
    assert by["002_s010_read"]["stage"] == "failed" and by["002_s010_read"]["error"] == "boom"
    assert by["003_s011_read"]["stage"] == "bad (not processed)"
    assert [r["take_id"] for r in rows] == ["003_s011_read", "002_s010_read", "001_calibration_calibration"]  # newest first
