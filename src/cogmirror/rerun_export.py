"""Export a processed video (the `data/video/<id>/` work directory written by
`scripts/video_run.py`) as a Rerun recording (.rrd).

The recording holds, on one timeline called `time` (seconds):

    video                 the source video, transcoded small (MP4, 640 px wide)
    audio                 its audio track as mono MP3 (AssetAudio; plays while the time cursor moves)
    face/crop             the loose crop the model saw, at the sampled frames
    labels/top            the top-3 emotions at the current time
    story                 the story text with the sentence being read in bold (recorded takes)
    events/sentence       one text log per sentence start
    traces/valence        PC1 at the valence layer (windowed) and the raw value
    traces/pc/pc2..       the other principal components
    traces/emotion/<e>    all 171 emotion scores
    traces/segment/*      sentence index over time

Sentence timing comes from the recorder's take log (the reader's key presses),
found automatically for data/video/<session>/<take>/ work dirs.

Every trace is in reference-SD units, so one y-range fits them all. The
default blueprint shows the selected traces on one "overlay" plot with a
legend and distinct colors; a "stacked" tab behind it draws them one strip
each, EKG style, with the same y-range on every strip. All other traces stay in the recording, so any of them can
be added in the viewer.

The default selection is per recording: valence, the PCs, the story's target
emotions from the take log (drawn thicker, named in the story header), and the emotions
that appear most often in the frame-level top three over the clip (the "top
emotions" pane aggregated), `top_k` of them.
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
VIDEO_WIDTH = 640       # embedded video width; the viewer holds every recording in memory
CROP_VIEW_PX = 384      # face crops re-encoded to this size for the viewer

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


def top_label_emotions(cols: dict) -> list[str]:
    """Emotions ranked by how often they are in a frame's top three (weighted 3/2/1),
    i.e. the "top emotions" pane aggregated over the clip."""
    weight: dict[str, float] = {}
    for key, w in (("top1", 3), ("top2", 2), ("top3", 1)):
        if key not in cols:
            continue
        for v in cols[key]:
            v = str(v)
            if v and v != "nan":
                weight[v] = weight.get(v, 0.0) + w
    return sorted(weight, key=lambda e: -weight[e])


def default_traces(cols: dict, top_k: int = DEFAULT_TOP_K, targets: list[str] | tuple[str, ...] = ()) -> list[str]:
    """valence, the other PCs, the story's target emotions (when known), and the k
    emotions most often in the top three over this video; by variance if the series
    has no top-label columns."""
    names = ["valence"] + [c for c in cols if c.startswith("pc") and c[2:].isdigit() and c != "pc1"]
    have = {c[len("score_"):] for c in cols if c.startswith("score_")}
    emos = [t for t in targets if t in have]
    ranked = top_label_emotions(cols)
    if not ranked and have and top_k:
        var = {e: float(np.nanvar(cols[f"score_{e}"].astype(float))) for e in have}
        ranked = sorted(have, key=lambda e: -var[e])
    for e in ranked:
        if len(emos) >= len([t for t in targets if t in have]) + top_k:
            break
        if e in have and e not in emos:
            emos.append(e)
    return names + emos


def resolve_traces(spec: str | list[str], cols: dict, top_k: int = DEFAULT_TOP_K,
                   targets: list[str] | tuple[str, ...] = ()) -> list[str]:
    """`default`, `all`, `pcs`, `emotions`, or a comma-separated list of names."""
    names = spec if isinstance(spec, list) else [s.strip() for s in spec.split(",") if s.strip()]
    cat = trace_catalog(cols)
    out: list[str] = []
    for n in names:
        if n == "default":
            out += default_traces(cols, top_k, targets)
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
    out = cache_dir / f"{video.stem}_{VIDEO_WIDTH}.mp4"
    if out.exists():
        return out
    exe = ffmpeg_exe()
    if exe is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Small: the viewer loads every recording into memory, and the model's crop is logged separately at full size.
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(video), "-vf", f"scale={VIDEO_WIDTH}:-2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-an", "-movflags", "+faststart", str(out)], check=True)
    return out


def ensure_audio(video: Path, cache_dir: Path) -> tuple[Path, str] | None:
    """The video's audio track for Rerun's AssetAudio: mono MP3 (the viewer does not
    decode the recorder's Opus), or 16 kHz WAV if this ffmpeg has no MP3 encoder."""
    for suffix, media_type in ((".mp3", "audio/mpeg"), (".wav", "audio/wav")):
        out = cache_dir / f"{video.stem}{suffix}"
        if out.exists():
            return out, media_type
    exe = ffmpeg_exe()
    if exe is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    attempts = [(".mp3", "audio/mpeg", ["-c:a", "libmp3lame", "-b:a", "48k", "-ac", "1"]),
                (".wav", "audio/wav", ["-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1"])]
    for suffix, media_type, codec in attempts:
        out = cache_dir / f"{video.stem}{suffix}"
        r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(video), "-vn", *codec, str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out, media_type
        out.unlink(missing_ok=True)
    return None


def small_jpeg(path: Path, size: int = CROP_VIEW_PX, quality: int = 80) -> bytes:
    """The model's 768 px crop re-encoded for the viewer: about 130 KB -> 15 KB per frame,
    which is what decides the recording's size (300 frames per minute of video)."""
    from io import BytesIO

    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > size:
            im.thumbnail((size, size))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def find_take_log(work: Path) -> Path | None:
    """The recorder's take log for a session take: data/recordings/<session>/<take>.json,
    given the work dir data/video/<session>/<take>."""
    work = work.resolve()
    if work.parent.name == "video":
        return None
    cand = work.parents[2] / "recordings" / work.parent.name / f"{work.name}.json"
    return cand if cand.exists() else None


def story_markdown(sentences: list[str], current: int | None, header: str = "") -> str:
    parts = [header] if header else []
    for k, text in enumerate(sentences):
        parts.append(f"**▶ {text}**" if k == current else text)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def recording_label(work_dir: str | Path) -> str:
    """Recording id and display name for a processed video.

    Single runs live at data/video/<id>/; recorded takes at data/video/<session>/<take>/,
    and take ids repeat across sessions (every session has 001_calibration_calibration),
    so a take is labelled <session>/<take>.
    """
    work = Path(work_dir).resolve()
    return work.name if work.parent.name == "video" else f"{work.parent.name}/{work.name}"


def rrd_path(out_dir: str | Path, work_dir: str | Path) -> Path:
    return Path(out_dir) / (recording_label(work_dir).replace("/", "_") + ".rrd")


def export(work_dir: str | Path, out_path: str | Path, traces: str | list[str] = "default", top_k: int = DEFAULT_TOP_K,
           y_range: float = DEFAULT_Y_RANGE, include_video: bool = True, take_log: str | Path | None = None,
           name: str | None = None) -> dict:
    """Write one .rrd for a processed video. Returns a small summary."""
    import rerun as rr
    import rerun.blueprint as rrb

    work = Path(work_dir)
    cols, meta = read_series(work)
    name = name or recording_label(work)
    t_all = cols["t"].astype(float)
    face = cols["face_found"].astype(bool) if "face_found" in cols else np.ones(len(t_all), bool)
    catalog = trace_catalog(cols)
    take_log = Path(take_log) if take_log else find_take_log(work)
    log = json.loads(Path(take_log).read_text()) if take_log else {}
    targets = [str(e) for e in (log.get("emotions") or [])]
    selected = resolve_traces(traces, cols, top_k, targets)

    # One application id per recording: the viewer keys blueprints by application id,
    # and the default layout (which traces are stacked) is chosen per recording.
    rec = rr.RecordingStream(f"{APP_ID}/{name}", recording_id=name)
    rr.send_recording_name(name, recording=rec)
    props = {k: meta[k] for k in ("video", "crop", "budget", "readout", "score_layer", "valence_layer", "window_s", "baseline") if k in meta}
    if targets:
        props["story_target"] = ", ".join(targets)
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
        audio = ensure_audio(video, work / "rerun")
        if audio is not None:
            rec.set_time(TIMELINE, duration=0.0)
            rec.log("audio", rr.AssetAudio(path=audio[0], media_type=audio[1]))
            summary["audio"] = str(audio[0])

    # -- the crops the model saw, and the top emotions, per sampled frame
    frames_dir = work / "frames"
    for i in np.nonzero(face)[0]:
        rec.set_time(TIMELINE, duration=float(t_all[i]))
        jpg = frames_dir / f"f{int(cols['frame_index'][i]):06d}.jpg"
        if jpg.exists():
            rec.log("face/crop", rr.EncodedImage(contents=small_jpeg(jpg), media_type="image/jpeg"))
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
        # The legend labels a series by its entity path; story targets are drawn thicker.
        width = 2.5 if trace in targets else (1.5 if idx is not None else 1.0)
        rec.log(path, rr.SeriesLines(colors=[color_for(trace, idx)], names=[trace], widths=[width]), static=True)
        rec.send_columns(path, indexes=[rr.TimeColumn(TIMELINE, duration=t_all[ok])], columns=rr.Scalars.columns(scalars=vals[ok]))

    # -- sentence boundaries and the story text from a recorder take log
    if take_log:
        times = log.get("sentence_times") or []
        sentences = log.get("sentences") or []
        summary["take_log"] = str(take_log)
        header = f"*story {log.get('story_id', '')}, target: {', '.join(targets) or 'none'}*" if log else ""
        rec.set_time(TIMELINE, duration=0.0)
        rec.log("story", rr.TextDocument(story_markdown(sentences, None, header), media_type=rr.MediaType.MARKDOWN))
        last_end = 0.0
        for k, st in enumerate(times):
            start = st.get("start") if isinstance(st, dict) else st[0]
            end = st.get("end") if isinstance(st, dict) else st[1]
            text = sentences[k] if k < len(sentences) else f"sentence {k}"
            if start is not None:
                rec.set_time(TIMELINE, duration=float(start))
                rec.log("traces/segment/sentence", rr.Scalars([k]))
                rec.log("events/sentence", rr.TextLog(f"[{k}] {text}"))
                rec.log("story", rr.TextDocument(story_markdown(sentences, k, header), media_type=rr.MediaType.MARKDOWN))
            if end is not None:
                rec.set_time(TIMELINE, duration=float(end))
                rec.log("traces/segment/sentence", rr.Scalars([k]))
                last_end = max(last_end, float(end))
        if times:
            rec.set_time(TIMELINE, duration=last_end + 1e-3)
            rec.log("story", rr.TextDocument(story_markdown(sentences, None, header), media_type=rr.MediaType.MARKDOWN))
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
    shares = [3, 3, 1]
    if summary.get("take_log"):
        left.append(rrb.TextDocumentView(origin="story", name="story"))
        shares.append(3)
    if summary.get("audio"):
        left.append(rrb.AudioView(origin="audio", name="audio"))
        shares.append(1)
    left.append(rrb.TextDocumentView(origin="meta", name="settings"))
    shares.append(1)
    if not summary["video"]:
        shares = shares[1:]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*left, row_shares=shares),
            rrb.Tabs(overlay, rrb.Vertical(*stacked, name="stacked"), active_tab="overlay"),
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
