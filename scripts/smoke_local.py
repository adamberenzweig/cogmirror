"""Local smoke test of the hook and readout code on a small Gemma 4.

    HF_HOME=data/hf python scripts/smoke_local.py

No emotion vectors are involved. This checks mechanics that are the same on
31B: the layer indexing, early exit, both load paths giving the same text
activations, and the token layout of each image readout.
"""

import argparse
import gc
import json

import numpy as np
import torch
from PIL import Image

from cogmirror import model as M, readouts as R

SMALL = "google/gemma-4-E2B-it"
TEXT = (
    "The kettle clicked off in the next room. Mara stood at the window with the letter "
    "still folded in her hand, watching the street lights come on one by one. She had "
    "read it three times on the train and once more on the stairs. Now she set it on the "
    "sill, smoothed the crease with her thumb, and went to pour the water before it cooled. "
) * 2


def cos(a, b):
    a, b = a.detach().cpu().flatten().double(), b.detach().cpu().flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SMALL)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="auto", help='"auto", "mps", or "cpu"')
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    # On a 24 GB Mac, pinning the whole model to MPS runs out of memory during
    # load; "auto" spills part of it to disk, which is slow but works.
    place = "auto" if args.device == "auto" else {"": args.device}
    report = {}

    # ---- text-only load: layer indexing and early exit -------------------
    mdl, tok = M.load_gemotions_model(args.model, four_bit=False, dtype=dtype, device_map=place)
    n = len(M.decoder_layers(mdl))
    layers = [n // 3, n // 2, (2 * n) // 3]
    inputs = tok(TEXT, return_tensors="pt", truncation=True, max_length=M.MAX_TEXT_TOKENS)
    inputs = {k: v.to(M.input_device(mdl)) for k, v in inputs.items()}
    with torch.no_grad():
        full = mdl(**inputs, output_hidden_states=True)
    with M.ResidualCapture(mdl, layers) as cap:
        acts = cap.run(**inputs)
    report["text"] = {
        "device": str(M.input_device(mdl)),
        "n_blocks": n,
        "n_tokens": int(inputs["input_ids"].shape[1]),
        "first_token": tok.convert_ids_to_tokens(inputs["input_ids"][0, :1].tolist()),
        "hook_N_equals_hidden_states_N_plus_1": {
            l: bool(torch.equal(acts[l], full.hidden_states[l + 1])) for l in layers},
        "hook_N_vs_hidden_states_N_cos": {l: cos(acts[l], full.hidden_states[l]) for l in layers},
        "finite": bool(all(torch.isfinite(a).all() for a in acts.values())),
        "quantization": M.quantization_report(mdl),
    }
    text_pooled = M.story_activation(mdl, tok, TEXT, layers)
    del mdl, full, acts
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ---- multimodal load: same text activations, image layouts -----------
    mdl, proc = M.load_model(args.model, four_bit=False, dtype=dtype, device_map=place)
    mm_pooled = M.story_activation(mdl, proc.tokenizer, TEXT, layers)
    report["multimodal_text_matches_text_only"] = {l: cos(text_pooled[l], mm_pooled[l]) for l in layers}
    report["multimodal_quantization"] = M.quantization_report(mdl)

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8))
    runs = []
    for budget in (70, 280):
        R.set_token_budget(proc, budget)
        for layout in R.LAYOUTS:
            prompt = (R.QUESTIONS if layout == "a2_turn" else R.PROMPTS)["feeling"]
            batch = R.make_batch(mdl, proc, [img, img], layout, prompt)
            out = R.read(mdl, batch, layers)
            logits = R.next_token_logits(mdl, batch)
            runs.append({
                "budget": budget,
                "layout": layout,
                **R.describe_layout(proc, batch),
                "shapes": {k: list(v.shape) for k, v in out.items()},
                "finite": bool(all(torch.isfinite(v).all() for v in out.values())),
                "batch_rows_equal": cos(out["last"][0], out["last"][1]),
                "next_tokens": [proc.tokenizer.decode([i]) for i in logits[0].topk(5).indices.tolist()],
                "input_keys": sorted(batch.inputs),
            })
    report["image_runs"] = runs
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
