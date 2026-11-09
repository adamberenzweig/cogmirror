"""Prompt layouts and readout positions for image inputs.

Every layout puts one image first and reads two things from the same pass:

- `last`:  the residual at the final token (strategy A)
- `image`: the mean residual over the image's soft tokens (strategy B)

Layouts (see the design doc, "Readout strategies"):

- `a1_raw`      no chat template: <image>{prompt}  (optionally with <bos> first)
- `a2_prefill`  chat template; the prompt is the start of the model's turn
- `a2_turn`     chat template; a user question, read where the model's turn begins

Images in a batch must share one size so that every sequence has the same
length and no padding is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ResidualCapture, input_device

LAYOUTS = ("a1_raw", "a2_prefill", "a2_turn")

PROMPTS = {
    "feeling": "The person in this picture is feeling",
    "feels": "The person in this photo feels",
    "emotion": "The emotion on this person's face is",
    "looks": "Right now, this person looks",
    "mood": "This person's mood could be described as",
}
QUESTIONS = {
    "feeling": "How is the person in this picture feeling?",
    "emotion": "What emotion does this person's face show?",
}


@dataclass
class Batch:
    inputs: dict            # processor output, on the model's device
    image_mask: torch.Tensor  # [B, T] bool: image soft-token positions
    text: str               # the final prompt string, for inspection
    n_image_tokens: int


def set_token_budget(processor, budget: int) -> None:
    """Visual token budget per image: 70, 140, 280, 560, or 1120.

    This is an upper bound. The actual count depends on aspect ratio; a
    square image at 280 yields 256 tokens (768 px on a side).
    """
    processor.image_processor.max_soft_tokens = budget
    processor.image_seq_length = budget


def build_text(processor, layout: str, prompt: str, raw_bos: bool = False) -> str:
    tok = processor.image_token
    if layout == "a1_raw":
        return f"{processor.tokenizer.bos_token if raw_bos else ''}{tok}{prompt}"
    if layout == "a2_prefill":
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe how the person in this picture is feeling."},
        ]}]
        head = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return head + prompt
    if layout == "a2_turn":
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise ValueError(f"unknown layout {layout!r}")


def make_batch(model, processor, images: list, layout: str, prompt: str, raw_bos: bool = False) -> Batch:
    """Tokenize `images` (PIL, all the same size) under one layout and prompt."""
    sizes = {im.size for im in images}
    if len(sizes) != 1:
        raise ValueError(f"images in a batch must share one size, got {sizes}")
    text = build_text(processor, layout, prompt, raw_bos)
    # The Gemma 4 tokenizer adds no BOS by itself. Chat templates include one;
    # the raw layout gets one only when the prompt text carries it. That
    # mirrors gemotions, whose stories went through the bare tokenizer.
    enc = processor(text=[text] * len(images), images=[[im] for im in images], return_tensors="pt")
    ids = enc["input_ids"]
    if "attention_mask" in enc and not bool(enc["attention_mask"].all()):
        raise RuntimeError("unexpected padding in batch")
    image_mask = ids == processor.image_token_id
    n_img = int(image_mask[0].sum())
    if n_img == 0:
        raise RuntimeError("no image tokens found in the tokenized prompt")
    device = input_device(model)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
    for key in ("pixel_values",):
        if key in inputs and inputs[key].is_floating_point():
            inputs[key] = inputs[key].to(model.dtype)
    return Batch(inputs, image_mask, text, n_img)


def describe_layout(processor, batch: Batch, tail: int = 12) -> dict:
    """Token layout of the first sequence, for checking readout positions."""
    ids = batch.inputs["input_ids"][0].cpu()
    mask = batch.image_mask[0]
    pos = torch.nonzero(mask).flatten()
    tk = processor.tokenizer
    return {
        "seq_len": int(ids.shape[0]),
        "starts_with_bos": bool(ids[0] == tk.bos_token_id),
        "n_image_tokens": batch.n_image_tokens,
        "image_span": (int(pos[0]), int(pos[-1])),
        "image_tokens_contiguous": bool(pos[-1] - pos[0] + 1 == len(pos)),
        "head": tk.convert_ids_to_tokens(ids[: int(pos[0]) + 1].tolist()),
        "tail": tk.convert_ids_to_tokens(ids[-tail:].tolist()),
        "last_token": tk.convert_ids_to_tokens([int(ids[-1])])[0],
    }


@torch.no_grad()
def read(model, batch: Batch, layers) -> dict[str, torch.Tensor]:
    """Run one batch. Returns {"last": [B, L, D], "image": [B, L, D]} float32
    on CPU, with L indexing `sorted(layers)`."""
    with ResidualCapture(model, layers) as cap:
        acts = cap.run(**batch.inputs)
    last, image = [], []
    for i in sorted(acts):
        h = acts[i].float()
        m = batch.image_mask.to(h.device)
        last.append(h[:, -1])
        image.append((h * m.unsqueeze(-1)).sum(dim=1) / m.sum(dim=1, keepdim=True))
    return {"last": torch.stack(last, dim=1).cpu(), "image": torch.stack(image, dim=1).cpu()}


@torch.no_grad()
def next_token_logits(model, batch: Batch) -> torch.Tensor:
    """Full forward pass; logits at the final position, [B, vocab], on CPU.
    Used by the verbal baseline."""
    out = model(**batch.inputs)
    return out.logits[:, -1].float().cpu()
