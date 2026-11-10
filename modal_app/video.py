"""Modal app for the M2 video pipeline (separate from modal_app/app.py).

    modal run --env cogmirror modal_app/video.py::track --datasets ravdess,crema   # CPU: sample, track, crop
    modal run --env cogmirror modal_app/video.py::run --datasets ravdess,crema     # GPU: activations per chunk
    modal run --env cogmirror modal_app/video.py::run --run upper --crop upper --datasets ravdess --subset upper
    modal run --env cogmirror modal_app/video.py::bench                            # throughput at batch 8

On the Volume:
    /vol/video/frames/<dataset>/<clip>/   f000123.jpg crops, frames.jsonl, video.json
    /vol/video/acts/<run>/<dataset>/<clip>/chunk_0000.npz

`scripts/video_run.py` drives the same GPU class for a local video, sending
the crops as bytes and keeping the store on the laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP = "cogmirror-video"
VOL = "/vol"
GEMOTIONS = f"{VOL}/gemotions"
DATASETS = f"{VOL}/datasets"
M1_FRAMES = f"{VOL}/frames"
VIDEO = f"{VOL}/video"
FRAMES = f"{VIDEO}/frames"
ACTS = f"{VIDEO}/acts"
GPU = "L40S"            # one GPU type only: deep-layer activations shift with arithmetic details
MAX_GPUS = 7            # leave room under the 10-GPU workspace cap for other jobs
GPU_HOUR_USD = 1.95
CHUNK_FRAMES = 300      # frames per store chunk and per GPU call
SCALEDOWN_S = 600       # keep the loaded model around between takes of a recording session (~$0.03 per idle minute)
DEFAULT_RATE = 5.0

app = modal.App(APP)
volume = modal.Volume.from_name("cogmirror", create_if_missing=False)

# Same pins as modal_app/app.py (activations must match M1's).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.15.0",
        "torchvision",
        "transformers==5.22.2",
        "accelerate>=1.0",
        "bitsandbytes==0.50.2",
        "huggingface_hub>=1.0",
        "numpy>=1.26",
        "pillow>=10",
    )
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("cogmirror")
)
# CPU image for decoding and face tracking (OpenCV comes with mediapipe).
prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libegl1", "libgles2", "wget")
    .pip_install("mediapipe==1.2.2", "numpy>=1.26", "pillow>=10")
    .run_commands(
        "wget -q -O /root/face_landmarker.task https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    .add_local_python_source("cogmirror")
)

prep_fn = dict(image=prep_image, volumes={VOL: volume}, timeout=60 * 60, cpu=2)


# --------------------------------------------------------------------------
# Clip selection for the Gate 2 runs
# --------------------------------------------------------------------------

def crema_subset(stems: list[str], n_actors: int = 8) -> list[str]:
    """Held-out CREMA-D actors (every (30 / n_actors)-th), their unspecified-
    and high-intensity clips: the full sentence x emotion crossing per actor."""
    from cogmirror.datasets import parse_crema

    clips = [parse_crema(s) for s in stems]
    held = sorted({c.actor for c in clips if c.split == "heldout"})
    step = max(1, len(held) // n_actors)
    chosen = set(held[::step][:n_actors])
    return [c.clip for c in clips if c.actor in chosen and c.intensity in ("unspecified", "high")]


def ravdess_upper_subset(stems: list[str]) -> list[str]:
    """Held-out RAVDESS actors: every neutral clip plus the strong-intensity
    happy, angry and sad clips (for the upper-face speech-motion comparison)."""
    from cogmirror.datasets import parse_ravdess

    out = []
    for s in stems:
        c = parse_ravdess(s)
        if c is None or c.split != "heldout":
            continue
        if c.emotion == "neutral" or (c.emotion in ("happy", "angry", "sad") and c.intensity == "strong"):
            out.append(c.clip)
    return out


def _source(dataset: str) -> tuple[Path, str]:
    return (Path(DATASETS, "crema", "VideoFlash"), ".flv") if dataset == "crema" else (Path(DATASETS, "ravdess"), ".mp4")


def pack_jobs(specs: list[dict], target: int = CHUNK_FRAMES) -> list[list[dict]]:
    """Group chunk specs into calls of about `target` frames."""
    jobs, cur, n = [], [], 0
    for s in specs:
        cur.append(s)
        n += len(s["files"])
        if n >= target:
            jobs.append(cur)
            cur, n = [], 0
    if cur:
        jobs.append(cur)
    return jobs


# --------------------------------------------------------------------------
# CPU: tracking
# --------------------------------------------------------------------------

@app.function(**prep_fn, max_containers=40)
def track_remote(dataset: str, stems: list[str], rate: float = DEFAULT_RATE, force: bool = False) -> list[dict]:
    """Sample and track a list of clips; crops and manifests go to the Volume.
    Returns one summary per clip with its frame records."""
    from cogmirror import datasets, video

    volume.reload()
    src, ext = _source(dataset)
    lm = video.make_landmarker()
    out = []
    for stem in stems:
        meta = datasets.parse(dataset, stem).as_dict()
        out_dir = Path(FRAMES, dataset, stem)
        try:
            if not force and (out_dir / "frames.jsonl").exists():
                records, info = video.read_manifest(out_dir)
            else:
                records, info = video.track_video(str(src / f"{stem}{ext}"), lm, out_dir, rate=rate)
            out.append({**meta, "info": info, "frames": records})
        except Exception as e:  # noqa: BLE001
            out.append({**meta, "error": repr(e), "frames": []})
    volume.commit()
    return out


@app.function(**prep_fn)
def list_clips_remote() -> dict:
    volume.reload()
    return {"crema": sorted(p.stem for p in Path(DATASETS, "crema", "VideoFlash").glob("*.flv")),
            "ravdess": sorted(p.stem for p in Path(DATASETS, "ravdess").glob("*.mp4"))}


@app.function(**prep_fn)
def list_stores_remote(run: str) -> dict[str, list[int]]:
    """{video_id: [chunk ids present]} for one run."""
    volume.reload()
    root = Path(ACTS, run)
    out: dict[str, list[int]] = {}
    for p in root.glob("*/*/chunk_*.npz"):
        out.setdefault(f"{p.parent.parent.name}/{p.parent.name}", []).append(int(p.stem.split("_")[1]))
    return out


# --------------------------------------------------------------------------
# GPU: activations
# --------------------------------------------------------------------------

@app.cls(image=image, volumes={VOL: volume}, gpu=GPU, memory=65536, timeout=90 * 60,
         max_containers=MAX_GPUS, scaledown_window=SCALEDOWN_S)
class Infer:
    """One model per container; `run` processes a list of chunk specs.

    A chunk spec: {"video_id", "chunk", "files": [paths], "frame_index": [...],
    "t": [...], "store": "<dir on the Volume>" | "", "crop": "medium",
    "budget": 280}. Files are read from the Volume unless `images` (bytes by
    file name) is given. With `store` set, a chunk whose file exists is
    skipped and the result is written to the Volume (committed once per
    call); otherwise (or with "return_bytes": true) the chunk's npz bytes
    come back to the caller.
    """

    @modal.enter()
    def load(self):
        import time

        from cogmirror import model as M

        t0 = time.time()
        self.model, self.processor = M.load_model()
        self.layers = sorted(M.CAPTURE_LAYERS)
        self.load_seconds = round(time.time() - t0, 1)
        self.budget = None

    def _batch_acts(self, images: list, layout: str, prompt: str):
        import numpy as np

        from cogmirror import readouts as R

        batch = R.make_batch(self.model, self.processor, images, layout, prompt)
        acts = R.read(self.model, batch, self.layers)
        return acts["last"].numpy().astype(np.float16), acts["image"].numpy().astype(np.float16)

    @modal.method()
    def run(self, specs: list[dict], images: dict[str, bytes] | None = None, batch_size: int = 8,
            layout: str = "a2_prefill", prompt_key: str = "feeling") -> list[dict]:
        import io
        import time
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        import torch
        from PIL import Image

        from cogmirror import readouts as R, store, video

        prompt = R.PROMPTS[prompt_key]
        results = []
        wrote = False
        if images is None:
            volume.reload()
        for spec in specs:
            t0 = time.time()
            store_dir = spec.get("store") or ""
            res = {"video_id": spec["video_id"], "chunk": spec["chunk"], "n": len(spec["files"]), "skipped": False}
            if store_dir and store.chunk_done(store_dir, spec["chunk"]) and not spec.get("return_bytes"):
                res["skipped"] = True
                results.append(res)
                continue
            budget = int(spec.get("budget", 280))
            if budget != self.budget:
                R.set_token_budget(self.processor, budget)
                self.budget = budget
            crop = spec.get("crop", "medium")

            def load(f):
                raw = Image.open(io.BytesIO(images[f])) if images is not None else Image.open(f)
                return video.crop_for_model(raw.convert("RGB"), crop)

            # Volume reads have per-file latency, so decode the chunk's frames in parallel.
            with ThreadPoolExecutor(8) as pool:
                pil_all = list(pool.map(load, spec["files"]))
            last, image = [], []
            for start in range(0, len(pil_all), batch_size):
                pil = pil_all[start:start + batch_size]
                l, im = self._batch_acts(pil, layout, prompt)
                last.append(l)
                image.append(im)
            payload = store.chunk_payload(spec["frame_index"], spec["t"], np.concatenate(last), np.concatenate(image), self.layers)
            if store_dir:
                store.save_chunk(store_dir, spec["chunk"], payload)
                wrote = True
            if not store_dir or spec.get("return_bytes"):
                buf = io.BytesIO()
                np.savez(buf, **payload)
                res["npz"] = buf.getvalue()
            res.update(seconds=round(time.time() - t0, 1), fps=round(len(spec["files"]) / max(time.time() - t0, 1e-6), 2),
                       gpu=torch.cuda.get_device_name(0), load_seconds=self.load_seconds)
            results.append(res)
        if wrote:
            volume.commit()     # once per call: a commit per short clip costs more than the inference
        return results

    @modal.method()
    def bench(self, budgets: list[int] = [280, 560], batch_sizes: list[int] = [8], n_batches: int = 6,
              layout: str = "a2_prefill") -> dict:
        """Frames per second through the capture pass, on real face crops."""
        import time

        import torch
        from PIL import Image

        from cogmirror import prep, readouts as R

        volume.reload()
        files = sorted(Path(M1_FRAMES, "ravdess").glob("*.jpg"))[:16]
        crops = [prep.apply_crop(Image.open(f).convert("RGB"), "medium", 768) for f in files]
        prompt = R.PROMPTS["feeling"]
        out = {"gpu": torch.cuda.get_device_name(0), "load_seconds": self.load_seconds, "layout": layout,
               "n_layers": len(self.layers), "deepest_layer": max(self.layers), "runs": []}
        for budget in budgets:
            R.set_token_budget(self.processor, budget)
            self.budget = budget
            for bs in batch_sizes:
                imgs = [crops[i % len(crops)] for i in range(bs)]
                batch = R.make_batch(self.model, self.processor, imgs, layout, prompt)
                R.read(self.model, batch, self.layers)       # warm-up
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(n_batches):
                    R.read(self.model, batch, self.layers)
                torch.cuda.synchronize()
                dt = time.time() - t0
                out["runs"].append({"budget": budget, "batch_size": bs, "seq_len": int(batch.inputs["input_ids"].shape[1]),
                                    "image_tokens": batch.n_image_tokens, "fps": round(bs * n_batches / dt, 2),
                                    "ms_per_frame": round(1000 * dt / (bs * n_batches), 1)})
        out["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        return out


# --------------------------------------------------------------------------
# Local entry points
# --------------------------------------------------------------------------

def _results_dir() -> Path:
    p = Path("results/m2")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def _select(clips: dict, datasets: list[str], subset: str, limit: int) -> dict[str, list[str]]:
    chosen = {}
    for ds in datasets:
        stems = clips[ds]
        if ds == "crema" and subset in ("", "gate2"):
            stems = crema_subset(stems)
        elif subset == "upper":
            stems = ravdess_upper_subset(stems) if ds == "ravdess" else []
        elif subset == "all":
            pass
        if limit:
            step = max(1, len(stems) // limit)
            stems = stems[::step][:limit]
        chosen[ds] = stems
    return chosen


@app.local_entrypoint()
def track(datasets: str = "ravdess,crema", subset: str = "gate2", limit: int = 0, rate: float = DEFAULT_RATE,
          force: bool = False):
    """Sample and track the selected clips; writes results/m2/frames_<dataset>.jsonl
    (one line per clip: metadata, video info, frame records)."""
    import collections

    clips = list_clips_remote.remote()
    chosen = _select(clips, datasets.split(","), subset, limit)
    jobs = [(ds, chunk, rate, force) for ds, stems in chosen.items() for chunk in _chunks(stems, 25)]
    print({ds: len(s) for ds, s in chosen.items()}, f"-> {len(jobs)} tracking jobs")
    by_ds: dict[str, list[dict]] = collections.defaultdict(list)
    for res in track_remote.starmap(jobs):
        for r in res:
            by_ds[r["dataset"]].append(r)
    for ds, rows in by_ds.items():
        suffix = f"_{subset}" if subset not in ("", "gate2") else ""
        path = _results_dir() / f"frames_{ds}{suffix}.jsonl"
        # Keep earlier clips of the same dataset when adding a subset.
        old = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    old[rec["clip"]] = line
        for r in rows:
            old[r["clip"]] = json.dumps(r)
        path.write_text("\n".join(old[k] for k in sorted(old)) + "\n")
        ok = [r for r in rows if not r.get("error")]
        n_frames = sum(len(r["frames"]) for r in ok)
        n_faces = sum(r["info"]["n_faces"] for r in ok)
        n_crops = sum(r["info"]["n_crops"] for r in ok)
        dur = sum(r["info"]["duration"] or 0 for r in ok)
        print(f"{ds}: {len(ok)} clips ({len(rows) - len(ok)} errors), {n_frames} sampled frames, {n_faces} detections, "
              f"{n_crops} crops, {dur / 60:.1f} min of video -> {path}")
        errs = collections.Counter(r.get("error", "")[:80] for r in rows if r.get("error"))
        if errs:
            print("  errors:", errs.most_common(3))


def _specs_for(frames_file: Path, run: str, crop: str, budget: int, chunk_frames: int) -> list[dict]:
    from cogmirror import store, video

    specs = []
    for line in frames_file.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            continue
        frames = video.frames_with_crops(rec["frames"])
        if not frames:
            continue
        vid = f"{rec['dataset']}/{rec['clip']}"
        for k, s, e in store.plan_chunks(len(frames), chunk_frames):
            part = frames[s:e]
            specs.append({"video_id": vid, "chunk": k, "files": [f"{FRAMES}/{vid}/{f['file']}" for f in part],
                          "frame_index": [f["frame_index"] for f in part], "t": [f["t"] for f in part],
                          "store": f"{ACTS}/{run}/{vid}", "crop": crop, "budget": budget})
    return specs


@app.local_entrypoint()
def run(datasets: str = "ravdess,crema", run: str = "medium", crop: str = "medium", budget: int = 280,
        subset: str = "gate2", limit: int = 0, batch_size: int = 8, chunk_frames: int = CHUNK_FRAMES):
    """Activations for every tracked clip listed in results/m2/frames_<dataset>[_<subset>].jsonl.
    Idempotent: chunks already on the Volume are skipped."""
    import time

    suffix = f"_{subset}" if subset not in ("", "gate2") else ""
    specs = []
    for ds in datasets.split(","):
        f = _results_dir() / f"frames_{ds}{suffix}.jsonl"
        if not f.exists():
            print(f"missing {f}; run ::track first")
            continue
        specs.extend(_specs_for(f, run, crop, budget, chunk_frames))
    if limit:
        specs = specs[:limit]
    done = list_stores_remote.remote(run)
    todo = [s for s in specs if s["chunk"] not in done.get(s["video_id"], [])]
    n_frames = sum(len(s["files"]) for s in todo)
    jobs = pack_jobs(todo, chunk_frames)
    print(f"{len(specs)} chunks, {len(todo)} to do ({n_frames} frames) in {len(jobs)} GPU calls on up to {MAX_GPUS} x {GPU}")
    if not jobs:
        return
    t0 = time.time()
    rows = []
    infer = Infer()
    for res in infer.run.map(jobs, kwargs={"batch_size": batch_size}):
        rows.extend(res)
        done_n = sum(r["n"] for r in rows if not r["skipped"])
        if len(rows) % 20 == 0:
            print(f"  {done_n}/{n_frames} frames, {time.time() - t0:.0f}s")
    wall = time.time() - t0
    fps = [r["fps"] for r in rows if not r["skipped"] and r.get("fps")]
    gpu_s = sum(r["seconds"] for r in rows if not r["skipped"])
    summary = {"run": run, "crop": crop, "budget": budget, "datasets": datasets, "subset": subset,
               "chunks": len(rows), "frames": sum(r["n"] for r in rows if not r["skipped"]),
               "wall_seconds": round(wall), "gpu_busy_seconds": round(gpu_s),
               "fps_per_gpu_median": round(float(sorted(fps)[len(fps) // 2]), 2) if fps else None,
               "load_seconds": sorted({r.get("load_seconds") for r in rows if r.get("load_seconds")}),
               "gpus": sorted({r.get("gpu") for r in rows if r.get("gpu")})}
    path = _results_dir() / f"run_{run}{suffix}.json"
    hist = json.loads(path.read_text()) if path.exists() else []
    hist.append(summary)
    path.write_text(json.dumps(hist, indent=1))
    print(summary)


@app.local_entrypoint()
def bench(budgets: str = "280,560", batch_sizes: str = "8", n_batches: int = 6):
    """Throughput per L40S and the cost per hour of video at 2, 5 and 10 fps."""
    r = Infer().bench.remote([int(b) for b in budgets.split(",")], [int(b) for b in batch_sizes.split(",")], n_batches)
    table = []
    for run in r["runs"]:
        row = {**run, "usd_per_hour_video": {}}
        for rate in (2, 5, 10):
            # GPU-hours per hour of video = rate / fps; cost at $1.95 per GPU-hour.
            row["usd_per_hour_video"][str(rate)] = round(GPU_HOUR_USD * rate / run["fps"], 2)
        table.append(row)
    r["cost"] = table
    r["gpu_hour_usd"] = GPU_HOUR_USD
    (_results_dir() / "benchmark.json").write_text(json.dumps(r, indent=1))
    print(f"{r['gpu']} (model load {r['load_seconds']} s, peak {r['gpu_mem_gb']} GB), layers to {r['deepest_layer']}")
    for row in table:
        print(f"  budget {row['budget']:4d} batch {row['batch_size']:2d}: {row['fps']:.2f} frames/s "
              f"({row['image_tokens']} image tokens, seq {row['seq_len']}) -> $/h of video at 2/5/10 fps: "
              f"{row['usd_per_hour_video']['2']} / {row['usd_per_hour_video']['5']} / {row['usd_per_hour_video']['10']}")
