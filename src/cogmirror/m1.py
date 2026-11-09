"""M1 inference: activations, verbal logits and zero-shot labels for the
face-crop frames. Runs in the GPU image on Modal.

One activation store is one (run name, layout): pooled residuals at every
captured layer for the `last` and `image` readouts, plus log-probabilities of
emotion words at the final position (the verbal baseline, from the same
forward pass).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import prep

# torch-dependent modules (readouts, model) are imported inside the functions
# that need them, so the manifest helpers work in CPU-only containers.

FIRST_PASS_FRAMES = (1, 3)   # of the 5 saved per clip

# Words scored by the verbal baseline in addition to the 171 emotion names.
EXTRA_WORDS = ("neutral", "fearful", "scared", "surprised", "disgust", "angry", "fear", "anger", "sadness", "happiness")

# Forced-choice options, in each dataset's own vocabulary.
CHOICES = {
    "crema": ["angry", "disgusted", "fearful", "happy", "neutral", "sad"],
    "ravdess": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
}
CHOICE_TO_LABEL = {"fearful": "afraid"}


def load_manifest(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select(records: list[dict], frames=FIRST_PASS_FRAMES, shard: int = 0, n_shards: int = 1,
           limit: int = 0, split: str = "") -> list[dict]:
    """Frames with a detected face, in a fixed order, for one shard.

    Shards are assigned by clip, so all frames of a clip land together.
    `limit` caps the number of clips (taken evenly across the sorted list, so
    a small limit still covers both datasets).
    """
    recs = [r for r in records if r.get("face_found") and r["k"] in frames and (not split or r["split"] == split)]
    recs.sort(key=lambda r: (r["dataset"], r["clip"], r["k"]))
    clips = sorted({(r["dataset"], r["clip"]) for r in recs})
    if limit and limit < len(clips):
        step = len(clips) / limit
        clips = [clips[int(i * step)] for i in range(limit)]
    index = {c: i for i, c in enumerate(clips)}
    return [r for r in recs if (r["dataset"], r["clip"]) in index and index[(r["dataset"], r["clip"])] % n_shards == shard]


def word_token_ids(tokenizer, words) -> np.ndarray:
    """First token of each word as it appears mid-sentence (leading space)."""
    return np.array([tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in words])


def load_images(frames_dir: str | Path, recs: list[dict], crop: str, size: int = 768):
    from PIL import Image

    return [prep.apply_crop(Image.open(Path(frames_dir, r["dataset"], r["file"])).convert("RGB"), crop, size)
            for r in recs]


def run_activations(model, processor, recs: list[dict], frames_dir: str, layout: str, emotions: list[str],
                    crop: str = "medium", budget: int = 280, prompt_key: str = "feeling",
                    batch_size: int = 8, progress=print, transform=None) -> dict:
    """Activations (and word log-probabilities) for a list of frame records.

    `transform`, if given, maps a PIL image to a PIL image after cropping; it
    is used by the noise-floor runs.
    """
    import torch

    from . import readouts as R
    from .model import CAPTURE_LAYERS, ResidualCapture

    R.set_token_budget(processor, budget)
    prompt = (R.QUESTIONS if layout == "a2_turn" else R.PROMPTS)[prompt_key]
    want_logits = layout != "a2_turn"       # the turn-start position predicts sentence openers, not emotions
    words = list(emotions) + [w for w in EXTRA_WORDS if w not in emotions]
    ids = torch.tensor(word_token_ids(processor.tokenizer, words))
    layers = sorted(CAPTURE_LAYERS)

    last, image, logp, top = [], [], [], []
    for start in range(0, len(recs), batch_size):
        chunk = recs[start:start + batch_size]
        imgs = load_images(frames_dir, chunk, crop)
        if transform is not None:
            imgs = [transform(im) for im in imgs]
        batch = R.make_batch(model, processor, imgs, layout, prompt)
        with ResidualCapture(model, layers, early_exit=not want_logits) as cap, torch.no_grad():
            if want_logits:
                out = model(**batch.inputs)
                acts = cap.acts
                lp = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
                logp.append(lp[:, ids.to(lp.device)].cpu().numpy())
                top.append(lp.argmax(dim=-1).cpu().numpy())
            else:
                acts = cap.run(**batch.inputs)
        ls, im = [], []
        for i in layers:
            h = acts[i].float()
            m = batch.image_mask.to(h.device)
            ls.append(h[:, -1])
            im.append((h * m.unsqueeze(-1)).sum(dim=1) / m.sum(dim=1, keepdim=True))
        last.append(torch.stack(ls, dim=1).cpu().numpy().astype(np.float16))
        image.append(torch.stack(im, dim=1).cpu().numpy().astype(np.float16))
        if (start // batch_size) % 50 == 0:
            progress(f"  {layout}: {start + len(chunk)}/{len(recs)}")
    out = {
        "dataset": np.array([r["dataset"] for r in recs]),
        "clip": np.array([r["clip"] for r in recs]),
        "k": np.array([r["k"] for r in recs]),
        "layers": np.array(layers),
        "last": np.concatenate(last),
        "image": np.concatenate(image),
    }
    if want_logits:
        out.update(words=np.array(words), word_ids=ids.numpy(), logp=np.concatenate(logp).astype(np.float32),
                   top_id=np.concatenate(top))
    return out


def choice_prompt(dataset: str) -> str:
    opts = ", ".join(CHOICES[dataset])
    return f"Which emotion is this person expressing? Answer with exactly one word from this list: {opts}."


def parse_choice(text: str, dataset: str) -> str:
    """Map generated text to a label, or "" when no option appears."""
    t = text.lower()
    hits = [(t.find(c), c) for c in CHOICES[dataset] if c in t]
    for alias, canon in (("fear", "fearful"), ("anger", "angry"), ("disgust", "disgusted"), ("happiness", "happy"),
                         ("sadness", "sad"), ("surprise", "surprised"), ("afraid", "fearful"), ("scared", "fearful")):
        if alias in t and canon in CHOICES[dataset]:
            hits.append((t.find(alias), canon))
    if not hits:
        return ""
    choice = min(hits)[1]
    return CHOICE_TO_LABEL.get(choice, choice)


def run_choices(model, processor, recs: list[dict], frames_dir: str, crop: str = "medium", budget: int = 280,
                batch_size: int = 8, max_new_tokens: int = 8, progress=print) -> list[dict]:
    """Zero-shot forced-choice labels by generation: the perception ceiling."""
    import torch

    from . import readouts as R

    R.set_token_budget(processor, budget)
    out = []
    by_ds: dict[str, list[dict]] = {}
    for r in recs:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, rs in by_ds.items():
        prompt = choice_prompt(ds)
        for start in range(0, len(rs), batch_size):
            chunk = rs[start:start + batch_size]
            batch = R.make_batch(model, processor, load_images(frames_dir, chunk, crop), "a2_turn", prompt)
            n_in = batch.inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**batch.inputs, max_new_tokens=max_new_tokens, do_sample=False)
            texts = processor.tokenizer.batch_decode(gen[:, n_in:], skip_special_tokens=True)
            for r, t in zip(chunk, texts):
                out.append({"dataset": ds, "clip": r["clip"], "k": r["k"], "text": t.strip(),
                            "pred": parse_choice(t, ds)})
            if (start // batch_size) % 50 == 0:
                progress(f"  choices {ds}: {start + len(chunk)}/{len(rs)}")
    return out
