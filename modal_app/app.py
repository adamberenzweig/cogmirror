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
# L40S first; A100 40 GB costs about the same and also fits the 4-bit model.
GPU = ["L40S", "A100-40GB"]

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

common = dict(image=image, volumes={VOL: volume})
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
