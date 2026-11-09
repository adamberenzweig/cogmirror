"""Frame extraction and face cropping. Runs in the Modal `prep` image
(OpenCV + MediaPipe); nothing here needs a GPU.

For each clip: take K frames evenly spaced over the middle third, find the
face with MediaPipe's FaceLandmarker, and save one loose square crop per
frame. Tighter crops are center crops of the loose one, so they can be
chosen later without touching the video again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FRAMES_PER_CLIP = 5
LOOSE_SCALE = 2.2          # loose crop side = 2.2 x face size (head and shoulders)
CROPS = {"tight": 1.2, "medium": 1.6, "loose": LOOSE_SCALE}   # side as a multiple of face size
SAVE_SIZE = 768            # loose crops are saved at up to this size, never upscaled
PAD_VALUE = 114
LANDMARKER_PATH = "/root/face_landmarker.task"


def frame_indices(n_frames: int, k: int = FRAMES_PER_CLIP) -> list[int]:
    """k indices evenly spaced over the middle third of the clip."""
    if n_frames <= 0:
        return []
    lo, hi = n_frames / 3, 2 * n_frames / 3
    idx = np.linspace(lo, hi, k).round().astype(int)
    return np.clip(idx, 0, n_frames - 1).tolist()


def center_fraction(crop: str) -> float:
    """Fraction of the saved loose crop's side that a named crop keeps."""
    return CROPS[crop] / LOOSE_SCALE


def apply_crop(img, crop: str, out_size: int = 768):
    """Named crop from a saved loose crop (PIL image), resized to a square."""
    from PIL import Image

    f = center_fraction(crop)
    w, h = img.size
    side = round(min(w, h) * f)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((out_size, out_size), Image.BICUBIC)


def loose_crop(frame: np.ndarray, cx: float, cy: float, face: float) -> np.ndarray:
    """Square crop of side LOOSE_SCALE * face centered on (cx, cy), padded with
    gray where it leaves the frame, downscaled to SAVE_SIZE if larger."""
    import cv2

    side = int(round(LOOSE_SCALE * face))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    h, w = frame.shape[:2]
    out = np.full((side, side, 3), PAD_VALUE, dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    if side > SAVE_SIZE:
        out = cv2.resize(out, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    return out


def make_landmarker():
    import mediapipe as mp

    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def detect(landmarker, bgr: np.ndarray):
    """(cx, cy, face size in px, blendshape scores dict) or None."""
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = (lo + hi) / 2
    face = float(max(hi - lo))
    shapes = {c.category_name: float(c.score) for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
    return float(cx), float(cy), face, shapes


def read_frames(path: str, k: int = FRAMES_PER_CLIP):
    """(frames [BGR arrays], their indices, total frame count, fps)."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = 0
    while cap.grab():       # container frame counts are unreliable for .flv
        n += 1
    cap.release()
    want = frame_indices(n, k)
    frames, got = [], []
    cap = cv2.VideoCapture(path)
    i, targets = 0, sorted(set(want))
    while targets and cap.grab():
        if i == targets[0]:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
                got.append(i)
            targets.pop(0)
        i += 1
    cap.release()
    return frames, got, n, float(fps)


def process_clip(landmarker, video_path: str, out_dir: str, meta: dict) -> list[dict]:
    """Save loose crops for one clip; return one record per frame."""
    import cv2

    frames, idxs, n, fps = read_frames(video_path)
    records = []
    for k, (frame, idx) in enumerate(zip(frames, idxs)):
        rec = {**meta, "k": k, "frame_index": idx, "n_frames": n, "fps": fps,
               "frame_h": frame.shape[0], "frame_w": frame.shape[1]}
        det = detect(landmarker, frame)
        if det is None:
            rec.update(face_found=False)
        else:
            cx, cy, face, shapes = det
            name = f"{meta['clip']}_{k}.jpg"
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(out_dir) / name), loose_crop(frame, cx, cy, face), [cv2.IMWRITE_JPEG_QUALITY, 95])
            rec.update(face_found=True, face_px=round(face, 1), file=name, blendshapes=shapes)
        records.append(rec)
    if not records:
        records.append({**meta, "k": -1, "face_found": False, "n_frames": n, "fps": fps, "error": "no frames decoded"})
    return records
