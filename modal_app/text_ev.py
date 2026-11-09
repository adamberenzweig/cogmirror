"""Modal app for the M4 text EVs (separate from modal_app/app.py and video.py).

Driven by `scripts/m4_text_ev.py`, which opens the app with `app.run(...)`,
calls the remote functions, and writes results/m4/. Nothing here is a local
entrypoint on purpose: the driver keeps the raw activations on the laptop.

    text_ev_remote(stories, layers)   GPU: one raw-text pass per story -> npz bytes
    origins_remote(layers)            CPU: gemotions story mean per layer from the Volume caches

Same image pins as modal_app/app.py so the activations match Gate 0.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP = "cogmirror-text-ev"
VOL = "/vol"
GEMOTIONS = f"{VOL}/gemotions"
GPU = "L40S"            # one GPU type only (deep-layer activations shift with arithmetic details)
MAX_GPUS = 3            # the M4 text job is small; leave the 10-GPU cap to the video runs

app = modal.App(APP)
volume = modal.Volume.from_name("cogmirror", create_if_missing=False)

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
cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy>=1.26").add_local_python_source("cogmirror")


@app.function(image=image, volumes={VOL: volume}, gpu=GPU, memory=65536, timeout=60 * 60, max_containers=MAX_GPUS)
def text_ev_remote(stories: list[dict], layers: list[int]) -> dict:
    """Raw-text activations for a batch of stories (gemotions' exact model load).

    Returns {"npz": bytes, "ids": [...], "load_seconds", "seconds", "gpu"}.
    The npz holds, per story, `sentence_<i>` [S_i, L, D], `story_<i>` [L, D],
    `story_full_<i>` [L, D], `n_tokens` [N], `token_spans_<i>` [S_i, 2].
    """
    import io
    import time

    import numpy as np
    import torch

    from cogmirror import model as M, text_ev

    t0 = time.time()
    mdl, tok = M.load_gemotions_model()
    load_s = time.time() - t0
    layers = [int(l) for l in layers]
    t1 = time.time()
    arrays: dict[str, np.ndarray] = {"n_tokens": np.zeros(len(stories), dtype=np.int64)}
    truncated = []
    for i, s in enumerate(stories):
        r = text_ev.story_text_activations(mdl, tok, s, layers)
        arrays[f"sentence_{i}"] = r["sentence"].astype(np.float16)
        arrays[f"story_{i}"] = r["story"].astype(np.float16)
        arrays[f"story_full_{i}"] = r["story_full"].astype(np.float16)
        arrays[f"token_spans_{i}"] = np.asarray(r["token_spans"], dtype=np.int64)
        arrays["n_tokens"][i] = r["n_tokens"]
        if r["truncated_sentences"]:
            truncated.append((s["id"], r["truncated_sentences"]))
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return {
        "npz": buf.getvalue(),
        "ids": [s["id"] for s in stories],
        "truncated": truncated,
        "layers": layers,
        "load_seconds": round(load_s, 1),
        "seconds": round(time.time() - t1, 1),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "quantization": M.quantization_report(mdl),
    }


@app.function(image=cpu_image, volumes={VOL: volume}, memory=24576, cpu=4, timeout=60 * 60)
def origins_remote(layers: list[int]) -> dict:
    """gemotions' global story mean per layer, recomputed from the per-story
    caches on the Volume (`evspace.recompute_from_cache`). Layers without a
    cache are left out; the driver fits those on the story set instead.
    Returns {"origins": {layer: [D] floats}, "n_stories": {layer: int}, "missing": [...]}.
    """
    import time

    from cogmirror import evspace

    volume.reload()
    out: dict = {"origins": {}, "n_stories": {}, "noise_components": {}, "seconds": {}, "missing": []}
    for l in [int(l) for l in layers]:
        if not evspace.cache_dir(GEMOTIONS, l).exists():
            out["missing"].append(l)
            continue
        t0 = time.time()
        rec = evspace.recompute_from_cache(GEMOTIONS, l)
        out["origins"][str(l)] = rec.global_mean.astype("float32").tolist()
        out["n_stories"][str(l)] = int(rec.counts.sum())
        out["noise_components"][str(l)] = int(rec.noise_basis.shape[1])
        out["seconds"][str(l)] = round(time.time() - t0, 1)
    return out


@app.local_entrypoint()
def check(layers: str = "20,25,30,35,40,45,50"):
    """Which layers have a per-story cache on the Volume (no GPU)."""
    from cogmirror import evspace  # noqa: F401  (import check only)

    r = origins_remote.remote([int(l) for l in layers.split(",")])
    print("origins from cache:", sorted(int(k) for k in r["origins"]), "missing:", r["missing"])
    print("stories per layer:", r["n_stories"], "seconds:", r["seconds"])
    out = Path("results/m4")
    out.mkdir(parents=True, exist_ok=True)
    import json

    (out / "origins_check.json").write_text(json.dumps({k: v for k, v in r.items() if k != "origins"}, indent=1))
