import json
from pathlib import Path

import numpy as np
import pytest

from cogmirror import rerun_export as RE


def _cols(n=20):
    rng = np.random.default_rng(0)
    cols = {"frame_index": np.arange(n) * 6, "t": np.arange(n) * 0.2, "face_found": np.ones(n, bool),
            "valence": rng.normal(size=n), "valence_raw": rng.normal(size=n), "pc1": rng.normal(size=n),
            "pc2": rng.normal(size=n), "pc3": rng.normal(size=n), "top1": np.array(["happy"] * n)}
    for e, scale in (("happy", 3), ("sad", 1), ("at ease", 0.1), ("angry", 2)):
        cols[f"score_{e}"] = rng.normal(size=n) * scale
    return cols


def test_catalog_and_default_selection():
    cols = _cols()
    cat = RE.trace_catalog(cols)
    assert cat["valence"] == "traces/valence/windowed" and "pc1" not in cat and cat["pc2"] == "traces/pc/pc2"
    assert cat["at ease"] == "traces/emotion/at_ease"
    sel = RE.resolve_traces("default", cols, top_k=2)
    assert sel[:3] == ["valence", "pc2", "pc3"] and set(sel[3:]) == {"happy", "angry"}
    assert RE.resolve_traces("valence,happy,valence", cols) == ["valence", "happy"]
    with pytest.raises(KeyError):
        RE.resolve_traces("nope", cols)


def test_colors_distinct_for_selected():
    cs = {RE.color_for(f"t{i}", i) for i in range(10)}
    assert len(cs) == 10
    assert RE.color_for("sad", None) == RE.color_for("sad", None)


def test_export_writes_rrd(tmp_path):
    pq = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pqt

    work = tmp_path / "clip1"
    (work / "frames").mkdir(parents=True)
    cols = _cols()
    table = pq.table({k: v for k, v in cols.items()})
    table = table.replace_schema_metadata({"video": "none.mp4", "crop": "medium", "window_s": "1.5"})
    pqt.write_table(table, work / "clip1.parquet")
    from PIL import Image
    for i in cols["frame_index"][:3]:
        Image.new("RGB", (32, 32), (120, 90, 80)).save(work / "frames" / f"f{int(i):06d}.jpg")
    log = {"sentences": ["One.", "Two."], "sentence_times": [{"sentence_idx": 0, "start": 0.0, "end": 1.5}, {"sentence_idx": 1, "start": 2.0, "end": 3.5}]}
    (tmp_path / "take.json").write_text(json.dumps(log))
    out = tmp_path / "clip1.rrd"
    s = RE.export(work, out, traces="valence,pc2,happy", take_log=tmp_path / "take.json", include_video=False)
    assert out.exists() and out.stat().st_size > 1000
    assert s["traces"] == ["valence", "pc2", "happy"] and s["video"] is None
