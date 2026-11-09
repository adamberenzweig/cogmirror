"""Video to face crops: frame sampling, face tracking with a smoothed crop
box, and the per-video frame manifest. CPU only (OpenCV + MediaPipe); runs
on the laptop or in the Modal `prep` image.

For each sampled frame the tracker detects the face with MediaPipe's
FaceLandmarker (`cogmirror.prep.detect`), smooths the box center and size
with a one-euro filter, holds the last box through short detection gaps, and
saves one loose square crop (`cogmirror.prep.loose_crop`, 2.2x the face,
<= 768 px). Tighter named crops are taken from the loose crop at inference
time with `prep.apply_crop`, exactly as in M1.

Frame manifest: one dict per sampled frame with the sample index `i`, the
source `frame_index`, time `t` in seconds, `face_found` (a detection on this
frame), `held` (box carried over from an earlier frame), the raw detection,
the smoothed box, the 52 blendshapes, and the crop `file`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import prep

DEFAULT_RATE = 5.0          # sampled frames per second
DEFAULT_MAX_GAP = 1.0       # seconds to hold the last box through missed detections
UPPER_FRACTION = 0.6        # "upper" crop variant keeps this fraction of the medium crop's height
CROP_VARIANTS = ("medium", "upper", "tight", "loose")

# Blendshapes that mark the onset of an expression, per emotion (for Gate 2's temporal check).
ONSET_BLENDSHAPES = {
    "happy": ("mouthSmileLeft", "mouthSmileRight"),
    "angry": ("browDownLeft", "browDownRight"),
    "surprised": ("browInnerUp", "jawOpen"),
    "disgusted": ("noseSneerLeft", "noseSneerRight"),
    "sad": ("mouthFrownLeft", "mouthFrownRight"),
    "afraid": ("browInnerUp", "eyeWideLeft", "eyeWideRight"),
}


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def sample_mask(n_frames: int, src_fps: float, rate: float) -> np.ndarray:
    """Which source frames to keep for a target rate: frame i is kept when
    floor(i * rate / src_fps) advances, so samples are spaced as evenly as
    the source rate allows and the first frame is always kept."""
    if n_frames <= 0:
        return np.zeros(0, dtype=bool)
    if rate <= 0 or src_fps <= 0 or rate >= src_fps:
        return np.ones(n_frames, dtype=bool)
    bucket = np.floor(np.arange(n_frames) * rate / src_fps).astype(np.int64)
    keep = np.ones(n_frames, dtype=bool)
    keep[1:] = bucket[1:] != bucket[:-1]
    return keep


def keep_frame(i: int, src_fps: float, rate: float, last_bucket: int) -> tuple[bool, int]:
    """Streaming form of `sample_mask`: (keep, new bucket)."""
    if rate <= 0 or src_fps <= 0 or rate >= src_fps:
        return True, i
    b = int(math.floor(i * rate / src_fps))
    return (b != last_bucket), b


def probe_video(path: str) -> dict:
    """fps, width, height, and the container's frame count (unreliable for .flv)."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames_container": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    return info


def iter_sampled_frames(path: str, rate: float = DEFAULT_RATE, fps_override: float = 0.0):
    """Yield (frame_index, t_seconds, BGR frame) for the sampled frames,
    decoding sequentially (seeking is unreliable for .flv)."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = fps_override or float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    i, bucket = 0, -1
    try:
        while cap.grab():
            keep, bucket = keep_frame(i, fps, rate, bucket)
            if keep:
                ok, frame = cap.retrieve()
                if ok:
                    yield i, i / fps, frame
            i += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Box smoothing
# --------------------------------------------------------------------------

class OneEuro:
    """One-euro filter on a vector signal (Casiez et al. 2012).

    `min_cutoff` (Hz) sets the smoothing at rest; `beta` raises the cutoff
    with speed so fast motion is followed with little lag.
    """

    def __init__(self, min_cutoff: float = 0.6, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    def update(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), float(t)
            return self.x.copy()
        dt = float(t) - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = float(t)
        return self.x.copy()


@dataclass
class BoxSmoother:
    """Smoothed (cx, cy, size) with hold-through for detection gaps.

    Center and size get their own filters: the size changes slowly, so it is
    smoothed harder (lower cutoff) than the center.
    """
    min_cutoff: float = 0.6
    beta: float = 0.02
    size_cutoff: float = 0.3
    max_gap: float = DEFAULT_MAX_GAP
    center: OneEuro = field(init=False)
    size: OneEuro = field(init=False)
    last_box: np.ndarray | None = field(init=False, default=None)
    last_t: float | None = field(init=False, default=None)

    def __post_init__(self):
        self.center = OneEuro(self.min_cutoff, self.beta)
        self.size = OneEuro(self.size_cutoff, self.beta / 4)

    def reset(self) -> None:
        self.center.reset()
        self.size.reset()
        self.last_box = None
        self.last_t = None

    def update(self, det: tuple[float, float, float] | None, t: float) -> tuple[np.ndarray | None, bool]:
        """Returns (box or None, held). `det` is (cx, cy, size) or None."""
        if det is not None:
            if self.last_t is not None and t - self.last_t > self.max_gap:
                self.reset()                      # a long gap: start fresh rather than drag the old box
            c = self.center.update(det[:2], t)
            s = self.size.update([det[2]], t)
            self.last_box = np.array([c[0], c[1], s[0]])
            self.last_t = float(t)
            return self.last_box.copy(), False
        if self.last_box is not None and self.last_t is not None and t - self.last_t <= self.max_gap:
            return self.last_box.copy(), True
        return None, False


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------

def make_landmarker(model_path: str = prep.LANDMARKER_PATH):
    """MediaPipe FaceLandmarker in IMAGE mode with blendshapes, from a given
    model file (the Modal image keeps it at `prep.LANDMARKER_PATH`)."""
    import mediapipe as mp

    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def crop_name(frame_index: int) -> str:
    return f"f{frame_index:06d}.jpg"


def track_video(path: str, landmarker, out_dir: str | Path, rate: float = DEFAULT_RATE,
                smoother: BoxSmoother | None = None, jpeg_quality: int = 95, fps_override: float = 0.0,
                progress=None) -> tuple[list[dict], dict]:
    """Sample, detect, smooth, and save one loose crop per sampled frame.

    Returns (frame records, video info). Also writes `frames.jsonl` and
    `video.json` into `out_dir`.
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    smoother = smoother or BoxSmoother()
    info = probe_video(path)
    records: list[dict] = []
    n_src = 0
    for k, (idx, t, frame) in enumerate(iter_sampled_frames(path, rate, fps_override)):
        n_src = idx + 1
        det = prep.detect(landmarker, frame)
        raw = None if det is None else (det[0], det[1], det[2])
        box, held = smoother.update(raw, t)
        rec = {"i": k, "frame_index": idx, "t": round(t, 4), "face_found": det is not None, "held": held,
               "frame_h": frame.shape[0], "frame_w": frame.shape[1]}
        if raw is not None:
            rec["det"] = [round(v, 2) for v in raw]
            rec["blendshapes"] = {n: round(v, 5) for n, v in det[3].items()}
        if box is not None:
            rec["box"] = [round(float(v), 2) for v in box]
            name = crop_name(idx)
            cv2.imwrite(str(out_dir / name), prep.loose_crop(frame, float(box[0]), float(box[1]), float(box[2])),
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            rec["file"] = name
        records.append(rec)
        if progress and k % 50 == 0:
            progress(f"  {Path(path).name}: {k} frames sampled at t={t:.1f}s")
    info.update(rate=rate, n_frames_decoded=n_src, n_sampled=len(records),
                n_faces=sum(r["face_found"] for r in records), n_crops=sum("file" in r for r in records),
                duration=(n_src / (fps_override or info["fps"])) if info["fps"] or fps_override else None,
                source=str(path))
    write_manifest(out_dir, records, info)
    return records, info


def write_manifest(out_dir: str | Path, records: list[dict], info: dict) -> None:
    out_dir = Path(out_dir)
    (out_dir / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (out_dir / "video.json").write_text(json.dumps(info, indent=1))


def read_manifest(out_dir: str | Path) -> tuple[list[dict], dict]:
    out_dir = Path(out_dir)
    records = [json.loads(l) for l in (out_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
    info_path = out_dir / "video.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return records, info


def frames_with_crops(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("file")]


# --------------------------------------------------------------------------
# Crops at inference time
# --------------------------------------------------------------------------

def crop_for_model(img, variant: str = "medium", size: int = 768):
    """Named crop from a saved loose crop. `upper` keeps the top 60% of the
    medium crop (eyes and brows, no mouth), as a non-square image."""
    if variant == "upper":
        med = prep.apply_crop(img, "medium", size)
        return med.crop((0, 0, size, int(round(size * UPPER_FRACTION))))
    return prep.apply_crop(img, variant, size)


# --------------------------------------------------------------------------
# Onsets (Gate 2 temporal check)
# --------------------------------------------------------------------------

def onset_time(t: np.ndarray, x: np.ndarray, margin: float, baseline_frames: int = 2,
               min_run: int = 1) -> float:
    """First time at which x exceeds its own early baseline (mean of the
    first `baseline_frames` finite values) by `margin`, for `min_run`
    consecutive samples. NaN when it never does."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < baseline_frames + 1:
        return float("nan")
    base = x[finite][:baseline_frames].mean()
    above = (x - base > margin) & finite
    run = 0
    for i in range(len(x)):
        run = run + 1 if above[i] else 0
        if run >= min_run:
            return float(t[i - min_run + 1])
    return float("nan")


def blendshape_series(records: list[dict], names) -> np.ndarray:
    """Mean of the named blendshapes per frame record (NaN without a detection)."""
    out = np.full(len(records), np.nan)
    for k, r in enumerate(records):
        bs = r.get("blendshapes")
        if bs:
            vals = [bs[n] for n in names if n in bs]
            if vals:
                out[k] = float(np.mean(vals))
    return out
