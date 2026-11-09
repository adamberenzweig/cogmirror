"""Export a processed video (the `data/video/<id>/` work directory written by
`scripts/video_run.py`) as a Rerun recording (.rrd).

The recording holds, on one timeline called `time` (seconds):

    video                 the source video, when Rerun can decode it (MP4)
    face/crop             the loose crop the model saw, at the sampled frames
    labels/top            the top-3 emotions at the current time
    traces/valence        PC1 at the valence layer (windowed) and the raw value
    traces/pc/pc2..       the other principal components
    traces/emotion/<e>    all 171 emotion scores
    traces/segment/*      sentence boundaries, when a recorder take log is given

Every trace is in reference-SD units, so one y-range fits them all. The
default blueprint stacks the selected traces vertically, EKG style, with the
same y-range on each, a legend, and distinct colors; an "overlay" tab shows
them on one plot. All other traces stay in the recording, so any of them can
be added in the viewer.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

TIMELINE = "time"
APP_ID = "cogmirror"
DEFAULT_Y_RANGE = 3.0
DEFAULT_TOP_K = 6

# Distinct, legible colors for the selected traces (RGB).
PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    (230, 85, 13), (49, 130, 189), (117, 107, 177), (99, 121, 57),
]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def read_series(work_dir: str | Path):
    """(table as dict of numpy columns, metadata dict) from the work dir's Parquet."""
    import pyarrow.parquet as pq

    work = Path(work_dir)
    t = pq.read_table(work / f"{work.name}.parquet")
    cols = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}
    meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    return cols, meta


def _path(name: str) -> str:
    return name.replace(" ", "_")


def trace_catalog(cols: dict) -> dict[str, str]:
    """{trace name: entity path} for every scalar column that is a trace."""
    out = {}
    if "valence" in cols:
        out["valence"] = "traces/valence/windowed"
    if "valence_raw" in cols:
        out["valence_raw"] = "traces/valence/raw"
    for c in cols:
        if c.startswith("pc") and c[2:].isdigit() and c != "pc1":
            out[c] = f"traces/pc/{c}"
        elif c.startswith("score_"):
            out[c[len("score_"):]] = f"traces/emotion/{_path(c[len('score_'):])}"
        elif c.startswith("blend_"):
            out[c] = f"traces/blendshape/{_path(c[len('blend_'):])}"
    return out


def default_traces(cols: dict, top_k: int = DEFAULT_TOP_K) -> list[str]:
    """valence, the other PCs, and the k emotions that vary most in this video."""
    names = ["valence"] + [c for c in cols if c.startswith("pc") and c[2:].isdigit() and c != "pc1"]
    emos = [c for c in cols if c.startswith("score_")]
    if emos and top_k:
        var = {c: float(np.nanvar(cols[c].astype(float))) for c in emos}
        names += [c[len("score_"):] for c in sorted(emos, key=lambda c: -var[c])[:top_k]]
    return names


def resolve_traces(spec: str | list[str], cols: dict, top_k: int = DEFAULT_TOP_K) -> list[str]:
    """`default`, `all`, `pcs`, `emotions`, or a comma-separated list of names."""
    names = spec if isinstance(spec, list) else [s.strip() for s in spec.split(",") if s.strip()]
    cat = trace_catalog(cols)
    out: list[str] = []
    for n in names:
        if n == "default":
            out += default_traces(cols, top_k)
        elif n == "all":
            out += list(cat)
        elif n == "pcs":
            out += [k for k in cat if k.startswith("pc")]
        elif n == "emotions":
            out += [k for k in cat if cat[k].startswith("traces/emotion/")]
        elif n in cat:
            out.append(n)
        else:
            raise KeyError(f"unknown trace {n!r}; known: default, all, pcs, emotions, {', '.join(list(cat)[:12])}, ...")
    seen: list[str] = []
    for n in out:
        if n not in seen:
            seen.append(n)
    return seen


def color_for(name: str, index: int | None) -> tuple[int, int, int]:
    """Palette color for selected traces; a muted hashed hue for the rest."""
    if index is not None and index < len(PALETTE):
        return PALETTE[index]
    h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.45, 0.75)
    return int(r * 255), int(g * 255), int(b * 255)


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------

def find_video(work: Path, meta: dict) -> Path | None:
    for cand in [work / f"{work.name}.mp4", work / f"{work.name}.webm", Path(meta.get("video", ""))]:
        if cand and cand.exists():
            return cand
    return None


def ffmpeg_exe() -> str | None:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def ensure_mp4(video: Path, cache_dir: Path) -> Path | None:
    """Rerun decodes MP4 (H.264/H.265/AV1). WebM from the recorder is transcoded
    once, next to the recording, if an ffmpeg binary is available."""
    if video.suffix.lower() == ".mp4":
        return video
    out = cache_dir / f"{video.stem}.mp4"
    if out.exists():
        return out
    exe = ffmpeg_exe()
    if exe is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-an", "-movflags", "+faststart", str(out)], check=True)
    return out


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export(work_dir: str | Path, out_path: str | Path, traces: str | list[str] = "default", top_k: int = DEFAULT_TOP_K,
           y_range: float = DEFAULT_Y_RANGE, include_video: bool = True, take_log: str | Path | None = None,
           name: str | None = None) -> dict:
    """Write one .rrd for a processed video. Returns a small summary."""
    import rerun as rr
    import rerun.blueprint as rrb

    work = Path(work_dir)
    cols, meta = read_series(work)
    name = name or work.name
    t_all = cols["t"].astype(float)
    face = cols["face_found"].astype(bool) if "face_found" in cols else np.ones(len(t_all), bool)
    catalog = trace_catalog(cols)
    selected = resolve_traces(traces, cols, top_k)

    rec = rr.RecordingStream(APP_ID, recording_id=name)
    rr.send_recording_name(name, recording=rec)
    props = {k: meta[k] for k in ("video", "crop", "budget", "readout", "score_layer", "valence_layer", "window_s", "baseline") if k in meta}
    rec.log("meta", rr.TextDocument("\n".join(f"{k}: {v}" for k, v in props.items())), static=True)

    # -- source video, as frame references on the timeline
    summary = {"name": name, "frames": int(len(t_all)), "video": None, "traces": selected}
    video = find_video(work, meta) if include_video else None
    if video is not None:
        mp4 = ensure_mp4(video, work / "rerun")
        if mp4 is not None:
            asset = rr.AssetVideo(path=mp4)
            rec.log("video", asset, static=True)
            ts = asset.read_frame_timestamps_nanos()
            rec.send_columns("video", indexes=[rr.TimeColumn(TIMELINE, duration=ts / 1e9)],
                             columns=rr.VideoFrameReference.columns_nanos(ts))
            summary["video"] = str(mp4)

    # -- the crops the model saw, and the top emotions, per sampled frame
    frames_dir = work / "frames"
    for i in np.nonzero(face)[0]:
        rec.set_time(TIMELINE, duration=float(t_all[i]))
        jpg = frames_dir / f"f{int(cols['frame_index'][i]):06d}.jpg"
        if jpg.exists():
            rec.log("face/crop", rr.EncodedImage(path=jpg))
        tops = [str(cols[k][i]) for k in ("top1", "top2", "top3") if k in cols]
        if tops:
            rec.log("labels/top", rr.TextDocument("  ·  ".join(tops)))

    # -- every trace, as columns; style per entity
    def column_name(trace: str) -> str:
        if trace in ("valence", "valence_raw"):
            return trace
        if trace.startswith("pc"):
            return trace
        if trace.startswith("blend_"):
            return trace
        return f"score_{trace}"

    for trace, path in catalog.items():
        vals = cols[column_name(trace)].astype(float)
        ok = np.isfinite(vals) & face
        if not ok.any():
            continue
        idx = selected.index(trace) if trace in selected else None
        rec.log(path, rr.SeriesLines(colors=[color_for(trace, idx)], names=[trace], widths=[1.5 if idx is not None else 1.0]), static=True)
        rec.send_columns(path, indexes=[rr.TimeColumn(TIMELINE, duration=t_all[ok])], columns=rr.Scalars.columns(scalars=vals[ok]))

    # -- sentence boundaries from a recorder take log
    if take_log:
        log = json.loads(Path(take_log).read_text())
        times = log.get("sentence_times") or []
        sentences = log.get("sentences") or []
        for k, st in enumerate(times):
            start = st.get("start") if isinstance(st, dict) else st[0]
            end = st.get("end") if isinstance(st, dict) else st[1]
            text = sentences[k] if k < len(sentences) else f"sentence {k}"
            if start is not None:
                rec.set_time(TIMELINE, duration=float(start))
                rec.log("traces/segment/sentence", rr.Scalars([k]))
                rec.log("events/sentence", rr.TextLog(f"[{k}] {text}"))
            if end is not None:
                rec.set_time(TIMELINE, duration=float(end))
                rec.log("traces/segment/sentence", rr.Scalars([k]))
        rec.log("traces/segment/sentence", rr.SeriesLines(names=["sentence"], colors=[(90, 90, 90)]), static=True)

    # -- blueprint: video + crop + labels on the left, stacked traces on the right
    axis = rrb.ScalarAxis(range=(-y_range, y_range), zoom_lock=True)
    legend = rrb.PlotLegend(corner=rrb.Corner2D.LeftTop, visible=True)
    stacked = [rrb.TimeSeriesView(origin=catalog[n], name=n, axis_y=axis, plot_legend=legend, titlebar=False)
               for n in selected if n in catalog]
    overlay = rrb.TimeSeriesView(name="overlay", contents=[catalog[n] for n in selected if n in catalog], axis_y=axis, plot_legend=legend)
    left = [rrb.Spatial2DView(origin="face/crop", name="model input")]
    if summary["video"]:
        left.insert(0, rrb.Spatial2DView(origin="video", name="video"))
    left.append(rrb.TextDocumentView(origin="labels/top", name="top emotions"))
    left.append(rrb.TextDocumentView(origin="meta", name="settings"))
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*left, row_shares=[3, 3, 1, 1][: len(left)]),
            rrb.Tabs(rrb.Vertical(*stacked, name="stacked"), overlay),
            column_shares=[1, 2],
        ),
        rrb.TimePanel(timeline=TIMELINE),
        collapse_panels=False,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(out, default_blueprint=blueprint, recording=rec)
    summary["rrd"] = str(out)
    return summary
