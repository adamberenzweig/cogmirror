"""Modal app for the 31B work.

    modal run --env cogmirror modal_app/app.py::download          # weights + gemotions data -> Volume (CPU, once)
    modal run --env cogmirror modal_app/app.py::gate0             # reproduction check, both loads (2 GPUs)
    modal run --env cogmirror modal_app/app.py::smoke             # a few images through every readout layout

Everything persistent lives on the `cogmirror` Volume, mounted at /vol:
    /vol/hf          Hugging Face cache (model weights)
    /vol/gemotions   vectors, stories, per-story activation caches
    /vol/results     JSON results (Modal keeps logs for one day only)
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP = "cogmirror"
VOL = "/vol"
GEMOTIONS = f"{VOL}/gemotions"
RESULTS = f"{VOL}/results"
CACHE_LAYERS = [20, 30, 40, 50]
# One GPU type only. Deep-layer activations shift with arithmetic details
# (see Gate 0), so mixing GPU models across shards would add noise.
GPU = "L40S"

app = modal.App(APP)
volume = modal.Volume.from_name(APP, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.15.0",
        "torchvision",  # required by the Gemma 4 processor
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

# CPU image for video decoding and face landmarks (M1 data preparation).
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
# CPU image for the off-the-shelf expression model used as a baseline.
fer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("hsemotion-onnx==0.3.1", "onnx", "onnxruntime", "opencv-python-headless", "numpy>=1.26", "pillow>=10")
    .add_local_python_source("cogmirror")
)
# CPU image with ffmpeg, for transcoding sample clips to browser-playable MP4.
video_image = modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg").add_local_python_source("cogmirror")
DATASETS = f"{VOL}/datasets"
FRAMES = f"{VOL}/frames"

common = dict(image=image, volumes={VOL: volume})
prep_fn = dict(image=prep_image, volumes={VOL: volume}, timeout=60 * 60)
# Loading reads ~62 GB of weights and quantizes them, so allow generous
# memory and time. Modal's default timeout is 5 minutes.
gpu_fn = dict(**common, gpu=GPU, memory=65536, timeout=2 * 60 * 60)


def _save(name: str, payload: dict) -> None:
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    Path(RESULTS, name).write_text(json.dumps(payload, indent=1, default=str))
    volume.commit()


@app.function(**common, timeout=4 * 60 * 60, memory=8192)
def download_remote(cache_layers: list[int] = CACHE_LAYERS) -> dict:
    from huggingface_hub import snapshot_download

    from cogmirror.fetch import fetch_gemotions
    from cogmirror.model import MODEL_31B

    weights = snapshot_download(MODEL_31B)
    volume.commit()
    data = fetch_gemotions(GEMOTIONS, cache_layers)
    volume.commit()
    return {"weights": weights, "gemotions": data}


@app.function(**gpu_fn)
def gate0_remote(mode: str = "gemotions", per_emotion: int = 5) -> dict:
    """Recompute story activations and compare with the gemotions caches.

    mode="gemotions"   gemotions' exact load (everything 4-bit by default)
    mode="vision_fp"   our load (vision path left unquantized)
    """
    import time

    import torch

    from cogmirror import evspace, gate0, model as M

    t0 = time.time()
    if mode == "gemotions":
        mdl, tok = M.load_gemotions_model()
    else:
        mdl, proc = M.load_model()
        tok = proc.tokenizer
    load_s = time.time() - t0

    out = gate0.run(mdl, tok, GEMOTIONS, CACHE_LAYERS, per_emotion=per_emotion)
    out.update(
        mode=mode,
        passed=gate0.passed(out),
        load_seconds=round(load_s, 1),
        n_blocks=len(M.decoder_layers(mdl)),
        quantization=M.quantization_report(mdl),
        gpu=torch.cuda.get_device_name(0),
        gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
        vectors_vs_cache={l: evspace.compare_to_released(GEMOTIONS, l) for l in CACHE_LAYERS},
    )
    _save(f"gate0_{mode}.json", out)
    out.pop("detail")
    return out


@app.function(**gpu_fn)
def smoke_remote(images: dict[str, bytes] | None = None, budgets: list[int] = [280, 560]) -> dict:
    """A few images through every layout: token layout, the model's own
    next-word guesses, and emotion scores relative to the first image.

    Without `images`, uses schematic drawn faces (neutral, happy, sad, angry).
    The first image is the reference that the others are compared against.
    Also times a batch of 8 for a first throughput figure.
    """
    import io
    import time

    import numpy as np
    import torch
    from PIL import Image

    from cogmirror import model as M, readouts as R, toyfaces
    from cogmirror.evspace import EVSpace

    if images:
        pil = {}
        for name, data in images.items():
            im = Image.open(io.BytesIO(data)).convert("RGB")
            side = min(im.size)
            left, top = (im.size[0] - side) // 2, (im.size[1] - side) // 2
            pil[name] = im.crop((left, top, left + side, top + side)).resize((768, 768))
    else:
        pil = {e: toyfaces.draw(e) for e in toyfaces.EXPRESSIONS}
    names = list(pil)
    imgs = [pil[n] for n in names]

    mdl, proc = M.load_model()
    layers = sorted(M.CAPTURE_LAYERS)
    spaces = {l: EVSpace.load(GEMOTIONS, l) for l in layers}
    out: dict = {"quantization": M.quantization_report(mdl), "images": names, "toy": not images, "runs": []}
    for budget in budgets:
        R.set_token_budget(proc, budget)
        for layout in R.LAYOUTS:
            prompt = (R.QUESTIONS if layout == "a2_turn" else R.PROMPTS)["feeling"]
            batch = R.make_batch(mdl, proc, imgs, layout, prompt)
            acts = R.read(mdl, batch, layers)
            logits = R.next_token_logits(mdl, batch)
            run = {"budget": budget, "layout": layout, **R.describe_layout(proc, batch), "per_image": {}}
            for i, name in enumerate(names):
                entry = {"next_tokens": [proc.tokenizer.decode([t]) for t in logits[i].topk(6).indices.tolist()]}
                if i > 0:
                    for readout in ("last", "image"):
                        delta = (acts[readout][i] - acts[readout][0]).numpy()   # [L, D], vs the reference image
                        # PC1 shift, in units of the spread (std) of the 171 emotion vectors on PC1.
                        entry[f"valence_{readout}"] = {
                            l: round(float(spaces[l].pcs(delta[k])[0] / spaces[l].emotion_pcs()[:, 0].std()), 2)
                            for k, l in enumerate(layers)}
                        k40 = layers.index(40)
                        sc = spaces[40].scores(delta[k40])
                        entry[f"top_{readout}_l40"] = [spaces[40].emotions[j] for j in np.argsort(sc)[::-1][:6]]
                        entry[f"bottom_{readout}_l40"] = [spaces[40].emotions[j] for j in np.argsort(sc)[:4]]
                run["per_image"][name] = entry
            out["runs"].append(run)

    # Throughput: 8 identical-size frames per batch, read up to the deepest captured layer.
    timing = {}
    for budget in budgets:
        R.set_token_budget(proc, budget)
        batch = R.make_batch(mdl, proc, [imgs[i % len(imgs)] for i in range(8)], "a1_raw", R.PROMPTS["feeling"])
        R.read(mdl, batch, layers)  # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            R.read(mdl, batch, layers)
        torch.cuda.synchronize()
        timing[budget] = round(8 * 3 / (time.time() - t0), 2)
    out["frames_per_second_batch8"] = timing
    out["gpu"] = torch.cuda.get_device_name(0)
    out["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    _save("smoke.json", out)
    return out


@app.function(**gpu_fn)
def gate0_diag_remote(per_emotion: int = 40, subset: int = 24) -> dict:
    """Follow-up to Gate 0: save our story activations for offline comparison
    with the caches, and rerun a subset under different numerical settings to
    see how sensitive deep-layer activations are to them."""
    import gc
    import importlib.metadata as md

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from cogmirror import gate0, model as M

    emotions = ("happy", "sad", "angry", "afraid", "calm", "excited")
    stories = gate0.load_stories(GEMOTIONS)
    picks = gate0.pick(stories, emotions, per_emotion, seed=1)
    # Stories that matched worst in the first run, plus a spread of others.
    worst = [("angry", 557), ("excited", 950), ("desperate", 897), ("afraid", 175), ("sleepy", 404), ("ashamed", 718)]
    sub = [(e, r, stories[e][r]) for e, r in worst] + picks[:: max(1, len(picks) // (subset - len(worst)))][: subset - len(worst)]

    def acts_for(mdl, tok, items):
        out = np.zeros((len(items), len(CACHE_LAYERS), 5376), dtype=np.float32)
        lens = []
        for i, (_, _, text) in enumerate(items):
            a = M.story_activation(mdl, tok, text, CACHE_LAYERS)
            out[i] = np.stack([a[l].numpy() for l in CACHE_LAYERS])
            lens.append(len(tok(text, truncation=True, max_length=M.MAX_TEXT_TOKENS)["input_ids"]))
        return out, lens

    def load(compute="bfloat16", **kw):
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute)
        mdl = AutoModelForCausalLM.from_pretrained(M.MODEL_31B, device_map="auto", quantization_config=q, **kw)
        return mdl.eval()

    def free(mdl):
        del mdl
        gc.collect()
        torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(M.MODEL_31B)
    save = {}
    mdl = load()
    attn_default = getattr(mdl.config, "_attn_implementation", None)
    save["main"], main_lens = acts_for(mdl, tok, picks)
    save["sub_default"], sub_lens = acts_for(mdl, tok, sub)
    save["sub_default_repeat"], _ = acts_for(mdl, tok, sub)
    free(mdl)

    mdl = load(attn_implementation="eager")
    save["sub_eager"], _ = acts_for(mdl, tok, sub)
    free(mdl)

    mdl = load(compute="float32")
    save["sub_fp32compute"], _ = acts_for(mdl, tok, sub)
    free(mdl)

    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(RESULTS, "gate0_diag.npz"),
        layers=np.array(CACHE_LAYERS),
        main_emotion=np.array([e for e, _, _ in picks]), main_row=np.array([r for _, r, _ in picks]),
        main_len=np.array(main_lens),
        sub_emotion=np.array([e for e, _, _ in sub]), sub_row=np.array([r for _, r, _ in sub]),
        sub_len=np.array(sub_lens),
        **save,
    )
    volume.commit()
    versions = {p: md.version(p) for p in ("torch", "transformers", "bitsandbytes", "accelerate")}
    return {"n_main": len(picks), "n_sub": len(sub), "attn_default": attn_default, "versions": versions,
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}


# --------------------------------------------------------------------------
# M1 data: download the acted-emotion corpora, extract frames and face crops
# --------------------------------------------------------------------------

@app.function(**prep_fn, max_containers=24)
def fetch_crema_remote(stems: list[str]) -> dict:
    """Download CREMA-D videos (Git LFS objects served over HTTPS)."""
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from cogmirror.datasets import CREMA_MEDIA

    out = Path(DATASETS, "crema", "VideoFlash")
    out.mkdir(parents=True, exist_ok=True)

    def get(stem: str) -> str:
        dest = out / f"{stem}.flv"
        if dest.exists() and dest.stat().st_size > 10_000:
            return "skipped"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(CREMA_MEDIA.format(name=stem), timeout=60) as r:
                    data = r.read()
                if len(data) < 10_000:      # an LFS pointer or an error page, not a video
                    raise IOError(f"only {len(data)} bytes")
                dest.write_bytes(data)
                return "ok"
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                time.sleep(2 * (attempt + 1))
        return f"failed: {err}"

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(get, stems))
    volume.commit()
    return {"ok": res.count("ok"), "skipped": res.count("skipped"),
            "failed": [(s, r) for s, r in zip(stems, res) if r.startswith("failed")]}


@app.function(**prep_fn, max_containers=6)
def fetch_ravdess_remote(actor: int) -> dict:
    """Download one actor's zip from Zenodo; keep the video-only speech clips."""
    import urllib.request
    import zipfile

    from cogmirror.datasets import RAVDESS_ZIP, parse_ravdess

    out = Path(DATASETS, "ravdess")
    out.mkdir(parents=True, exist_ok=True)
    have = list(out.glob(f"02-01-*-{actor:02d}.mp4"))
    if len(have) >= 60:
        return {"actor": actor, "kept": len(have), "skipped": True}
    tmp = f"/tmp/actor_{actor:02d}.zip"
    urllib.request.urlretrieve(RAVDESS_ZIP.format(actor=actor), tmp)
    kept = 0
    with zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            stem = Path(name).stem
            if name.endswith(".mp4") and parse_ravdess(stem) is not None:
                (out / f"{stem}.mp4").write_bytes(z.read(name))
                kept += 1
    volume.commit()
    return {"actor": actor, "kept": kept, "skipped": False}


@app.function(**prep_fn, max_containers=40)
def prep_remote(dataset: str, stems: list[str]) -> list[dict]:
    """Frames, face crops and blendshapes for a list of clips."""
    from cogmirror import datasets, prep

    volume.reload()
    src = Path(DATASETS, "crema", "VideoFlash") if dataset == "crema" else Path(DATASETS, "ravdess")
    ext = ".flv" if dataset == "crema" else ".mp4"
    out = Path(FRAMES, dataset)
    lm = prep.make_landmarker()
    records = []
    for stem in stems:
        meta = datasets.parse(dataset, stem).as_dict()
        try:
            records.extend(prep.process_clip(lm, str(src / f"{stem}{ext}"), str(out), meta))
        except Exception as e:  # noqa: BLE001
            records.append({**meta, "k": -1, "face_found": False, "error": repr(e)})
    volume.commit()
    return records


@app.function(**prep_fn)
def list_clips_remote() -> dict:
    volume.reload()
    crema = sorted(p.stem for p in Path(DATASETS, "crema", "VideoFlash").glob("*.flv"))
    rav = sorted(p.stem for p in Path(DATASETS, "ravdess").glob("*.mp4"))
    return {"crema": crema, "ravdess": rav}


@app.function(**prep_fn)
def put_text_remote(path: str, text: str) -> int:
    dest = Path(VOL, path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    volume.commit()
    return len(text)


@app.function(image=fer_image, volumes={VOL: volume}, timeout=60 * 60, max_containers=16, cpu=4)
def fer_remote(shard: int, n_shards: int, frames: list[int] = [1, 3], limit: int = 0) -> list[dict]:
    """Baseline: HSEmotion (EfficientNet-B0 trained on AffectNet), which
    outputs 8 expression logits plus valence and arousal."""
    import urllib.request  # noqa: F401  (the package downloads its model but forgets this import)

    import numpy as np
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    from PIL import Image

    from cogmirror import m1, prep

    volume.reload()
    recs = m1.select(m1.load_manifest(f"{FRAMES}/manifest.jsonl"), tuple(frames), shard, n_shards, limit)
    fer = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl")
    out = []
    for r in recs:
        img = prep.apply_crop(Image.open(Path(FRAMES, r["dataset"], r["file"])).convert("RGB"), "tight", 224)
        _, scores = fer.predict_emotions(np.asarray(img), logits=True)
        out.append({"dataset": r["dataset"], "clip": r["clip"], "k": r["k"], "scores": [float(x) for x in scores]})
    return out


@app.function(image=video_image, volumes={VOL: volume}, timeout=30 * 60)
def sample_videos_remote(per_emotion: int = 3) -> dict:
    """A few clips per dataset and emotion, transcoded to H.264 MP4 so they
    play in a browser. Returns {relative path: bytes} plus labels."""
    import subprocess

    from cogmirror import datasets

    volume.reload()
    src = {"crema": (Path(DATASETS, "crema", "VideoFlash"), ".flv"), "ravdess": (Path(DATASETS, "ravdess"), ".mp4")}
    files, labels = {}, []
    for ds, (folder, ext) in src.items():
        chosen: dict[str, list] = {}
        for p in sorted(folder.glob(f"*{ext}")):
            c = datasets.parse(ds, p.stem)
            if c is None:
                continue
            got = chosen.setdefault(c.emotion, [])
            # Different actors per emotion; prefer the stronger-intensity takes.
            if len(got) < per_emotion and c.actor not in [g.actor for g in got] and c.intensity in ("high", "strong", "unspecified"):
                got.append(c)
        for emotion, clips in sorted(chosen.items()):
            for c in clips:
                out = f"/tmp/{c.clip}.mp4"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(folder / f"{c.clip}{ext}"),
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac",
                                "-movflags", "+faststart", out], check=True)
                rel = f"{ds}/{c.clip}.mp4"
                files[rel] = Path(out).read_bytes()
                labels.append({**c.as_dict(), "file": rel})
    return {"files": files, "labels": labels}


# --------------------------------------------------------------------------
# M1 inference
# --------------------------------------------------------------------------

ACTS = f"{VOL}/acts"


@app.function(**gpu_fn)
def m1_acts_remote(run: str, layout: str, shard: int, n_shards: int, crop: str = "medium", budget: int = 280,
                   limit: int = 0, frames: list[int] = [1, 3], perturb: str = "", batch_size: int = 8,
                   tag: str = "", prompt_key: str = "feeling", split: str = "") -> dict:
    """Activations and verbal log-probabilities for one shard of frames."""
    import time

    import numpy as np
    import torch

    from cogmirror import evspace, m1, model as M, perturb as P

    volume.reload()
    recs = m1.select(m1.load_manifest(f"{FRAMES}/manifest.jsonl"), tuple(frames), shard, n_shards, limit, split)
    emotions = sorted(evspace.load_released_vectors(GEMOTIONS, 40))
    mdl, proc = M.load_model()
    t0 = time.time()
    out = m1.run_activations(mdl, proc, recs, FRAMES, layout, emotions, crop=crop, budget=budget,
                             transform=P.get(perturb) if perturb else None, batch_size=batch_size,
                             prompt_key=prompt_key)
    dt = time.time() - t0
    dest = Path(ACTS, run)
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{layout}_{shard}.npz" if not tag else f"{layout}_{tag}_{shard}.npz"
    np.savez(dest / name, **out)
    volume.commit()
    return {"run": run, "layout": layout, "shard": shard, "n": len(recs), "seconds": round(dt, 1),
            "fps": round(len(recs) / dt, 2), "gpu": torch.cuda.get_device_name(0)}


@app.function(**gpu_fn)
def m1_choices_remote(run: str, shard: int, n_shards: int, crop: str = "medium", budget: int = 280,
                      limit: int = 0, frames: list[int] = [2]) -> dict:
    """Zero-shot forced-choice labels by generation (the perception ceiling)."""
    import time

    from cogmirror import m1, model as M

    volume.reload()
    recs = m1.select(m1.load_manifest(f"{FRAMES}/manifest.jsonl"), tuple(frames), shard, n_shards, limit)
    mdl, proc = M.load_model()
    t0 = time.time()
    rows = m1.run_choices(mdl, proc, recs, FRAMES, crop=crop, budget=budget)
    dest = Path(ACTS, run)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"choices_{shard}.json").write_text(json.dumps(rows))
    volume.commit()
    return {"run": run, "shard": shard, "n": len(rows), "seconds": round(time.time() - t0, 1),
            "unparsed": sum(not r["pred"] for r in rows), "examples": [r["text"] for r in rows[:5]]}


# --------------------------------------------------------------------------
# Local entry points
# --------------------------------------------------------------------------

def _write_local(name: str, payload: dict) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("results", name).write_text(json.dumps(payload, indent=1, default=str))


@app.local_entrypoint()
def download():
    print(download_remote.remote())


@app.local_entrypoint()
def gate0(per_emotion: int = 5):
    calls = {m: gate0_remote.spawn(m, per_emotion) for m in ("gemotions", "vision_fp")}
    ok = True
    for mode, call in calls.items():
        r = call.get()
        _write_local(f"gate0_{mode}.json", r)
        ok &= r["passed"]
        print(f"\n== {mode} load ({r['gpu']}, load {r['load_seconds']} s, {r['gpu_mem_gb']} GB) ==")
        print("quantization:", r["quantization"])
        print("best-matching tokenization:", r["best_variant"])
        for name, per in r["variants"].items():
            worst = min(s["centered_cosine_min"] for s in per.values())
            print(f"  variant {name}: worst centered cosine {worst:.4f}")
        for layer, s in r["layers"].items():
            print(f"  layer {layer}: centered cos min {s['centered_cosine_min']:.4f} "
                  f"median {s['centered_cosine_median']:.4f} | rank-1 {s['rank1_fraction']:.0%} "
                  f"| other stories {s['other_story_centered_cosine_median']:.3f}")
        for layer, v in r["vectors_vs_cache"].items():
            print(f"  layer {layer}: recomputed vs released vectors, cos min {v['cosine_min']:.6f}")
    print("\nGATE 0:", "PASS" if ok else "FAIL")


@app.local_entrypoint()
def smoke(images: str = ""):
    """`--images neutral.jpg,smile.jpg,...`; the first is the reference."""
    data = {Path(p).stem: Path(p).read_bytes() for p in images.split(",") if p} or None
    r = smoke_remote.remote(data)
    _write_local("smoke.json", r)
    print("quantization:", r["quantization"])
    print("frames/s at batch 8:", r["frames_per_second_batch8"], "on", r["gpu"], f"({r['gpu_mem_gb']} GB peak)")
    for run in r["runs"]:
        print(f"\n[{run['layout']} @ {run['budget']}] seq {run['seq_len']}, {run['n_image_tokens']} image tokens, "
              f"last token {run['last_token']!r}")
        for name, e in run["per_image"].items():
            print(f"  {name:8s} next: {e['next_tokens']}")
            if "valence_last" in e:
                print(f"           valence vs reference, last token by layer: {e['valence_last']}")
                print(f"           valence vs reference, image tokens by layer: {e['valence_image']}")
                print(f"           top (last, L40): {e['top_last_l40']} | bottom: {e['bottom_last_l40']}")
                print(f"           top (image, L40): {e['top_image_l40']}")


@app.local_entrypoint()
def gate0diag():
    print(gate0_diag_remote.remote())


def _chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


@app.local_entrypoint()
def m1data(skip_download: bool = False, limit: int = 0):
    """Download CREMA-D and RAVDESS to the Volume, extract face crops, and
    write the frame manifest (results/m1/manifest.jsonl and on the Volume)."""
    import collections
    import urllib.request

    from cogmirror.datasets import CREMA_RAW, CREMA_TREE

    if not skip_download:
        tree = json.load(urllib.request.urlopen(CREMA_TREE, timeout=60))
        stems = sorted(Path(t["path"]).stem for t in tree["tree"] if t["path"].startswith("VideoFlash/"))
        if limit:
            stems = stems[:limit]
        print(f"CREMA-D: {len(stems)} clips to fetch")
        ok = skipped = 0
        failed = []
        for r in fetch_crema_remote.map(_chunks(stems, 200)):
            ok += r["ok"]; skipped += r["skipped"]; failed += r["failed"]
        print(f"  downloaded {ok}, already present {skipped}, failed {len(failed)}", failed[:5])
        actors = list(range(1, 25))[: (2 if limit else 24)]
        for r in fetch_ravdess_remote.map(actors):
            print("  RAVDESS", r)
        for name in ("processedResults/tabulatedVotes.csv", "VideoDemographics.csv", "SentenceFilenames.csv"):
            text = urllib.request.urlopen(CREMA_RAW.format(path=name), timeout=60).read().decode()
            Path("data/crema").mkdir(parents=True, exist_ok=True)
            Path("data/crema", Path(name).name).write_text(text)

    clips = list_clips_remote.remote()
    print({k: len(v) for k, v in clips.items()})
    jobs = [(ds, chunk) for ds, stems in clips.items() for chunk in _chunks(stems[: limit or None], 100 if ds == "crema" else 30)]
    records = []
    for recs in prep_remote.starmap(jobs):
        records.extend(recs)
    Path("results/m1").mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in records) + "\n"
    Path("results/m1/manifest.jsonl").write_text(text)
    put_text_remote.remote("frames/manifest.jsonl", text)

    found = [r for r in records if r.get("face_found")]
    print(f"{len(records)} frame records, {len(found)} with a face, "
          f"{len({(r['dataset'], r['clip']) for r in found})} clips with at least one face")
    print("errors:", collections.Counter(r.get("error", "")[:60] for r in records if r.get("error")).most_common(5))
    for ds in clips:
        sub = [r for r in found if r["dataset"] == ds]
        if sub:
            px = sorted(r["face_px"] for r in sub)
            print(f"  {ds}: frames {len(sub)}, emotions {dict(collections.Counter(r['emotion'] for r in sub if r['k'] == 0))}, "
                  f"face size px median {px[len(px) // 2]:.0f} (5th pct {px[len(px) // 20]:.0f}), "
                  f"frame {sub[0]['frame_w']}x{sub[0]['frame_h']}")


@app.local_entrypoint()
def m1run(run: str = "pass1", limit: int = 0, shards: int = 2, crop: str = "medium", budget: int = 280,
          layouts: str = "a1_raw,a2_prefill,a2_turn", choices: bool = True, perturb: str = ""):
    """First-pass M1 inference: every layout over the selected frames, plus
    forced-choice labels. With the defaults this uses 10 GPUs at once."""
    calls = [m1_acts_remote.spawn(run, layout, i, shards, crop, budget, limit, [1, 3], perturb)
             for layout in layouts.split(",") if layout for i in range(shards)]
    if choices:
        calls.append(m1_choices_remote.spawn(run, 0, 1, crop, budget, limit))
    results = [c.get() for c in calls]
    _write_local(f"m1_{run}_runs.json", {"runs": results})
    for r in results:
        print(r)


@app.local_entrypoint()
def m1fer(limit: int = 0, shards: int = 16):
    """Off-the-shelf expression model on the same frames (CPU)."""
    rows = []
    for part in fer_remote.starmap([(i, shards, [1, 3], limit) for i in range(shards)]):
        rows.extend(part)
    Path("results/m1").mkdir(parents=True, exist_ok=True)
    Path("results/m1/fer.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(len(rows), "frames scored")


@app.local_entrypoint()
def samples(per_emotion: int = 3):
    """Download a few clips per emotion and write data/samples/index.html."""
    import html

    r = sample_videos_remote.remote(per_emotion)
    root = Path("data/samples")
    for rel, data in r["files"].items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    names = {"crema": "CREMA-D (480x360, with audio)", "ravdess": "RAVDESS (1280x720, video-only files)"}
    parts = ["<!doctype html><meta charset=utf-8><title>Sample clips</title>",
             "<style>body{font:14px system-ui;margin:24px;background:#fff;color:#111}"
             "h2{margin-top:32px}.row{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:8px}"
             "figure{margin:0;width:300px}video{width:300px;background:#000}"
             "figcaption{font-size:12px;color:#555}h3{margin:16px 0 6px}</style>",
             "<h1>Sample clips</h1><p>A few clips per emotion from each corpus, transcoded for the browser. "
             "The pipeline uses frames from the middle third of each clip.</p>"]
    for ds in ("crema", "ravdess"):
        parts.append(f"<h2>{html.escape(names[ds])}</h2>")
        for emotion in sorted({l["emotion"] for l in r["labels"] if l["dataset"] == ds}):
            parts.append(f"<h3>{emotion}</h3><div class=row>")
            for l in r["labels"]:
                if l["dataset"] == ds and l["emotion"] == emotion:
                    parts.append(f"<figure><video src='{l['file']}' controls loop muted playsinline preload=metadata></video>"
                                 f"<figcaption>{l['clip']} · {l['actor']} · intensity {l['intensity']} · sentence {l['sentence']}</figcaption></figure>")
            parts.append("</div>")
    (root / "index.html").write_text("\n".join(parts))
    print(len(r["files"]), "clips written to", root, "- open", root / "index.html")


@app.local_entrypoint()
def m1noise(run: str = "noise", clips: int = 600, layout: str = "a2_prefill", crop: str = "medium", budget: int = 280):
    """Noise floor: the same frames under nuisance perturbations and a
    different batch size (8 GPUs at once)."""
    from cogmirror.perturb import PERTURBATIONS

    variants = [("base", "", 8), ("batch3", "", 3)] + [(name, name, 8) for name in PERTURBATIONS]
    calls = [m1_acts_remote.spawn(run, layout, 0, 1, crop, budget, clips, [1], perturb, bs, tag)
             for tag, perturb, bs in variants]
    for c in calls:
        print(c.get())


@app.local_entrypoint()
def m1prompts(run: str = "prompts", layout: str = "a2_prefill", shards: int = 2):
    """Prompt robustness: every paraphrase over the held-out clips (one frame each)."""
    from cogmirror.readouts import PROMPTS

    calls = [m1_acts_remote.spawn(run, layout, i, shards, "medium", 280, 0, [1], "", 8, key, key, "heldout")
             for key in PROMPTS for i in range(shards)]
    for c in calls:
        print(c.get())
