"""Model loading and residual-stream capture for Gemma 4.

Conventions match gemotions' `extract_vectors.py`:

- "Layer N" is the *output* of decoder block N (`hidden_states[N + 1]` in
  `output_hidden_states` indexing).
- gemotions loads with `AutoModelForCausalLM`. For Gemma 4 checkpoints that
  resolves to the full multimodal class (`Gemma4ForConditionalGeneration`),
  the same class `AutoModelForMultimodalLM` gives. Decoder blocks live at
  `model.model.language_model.layers`.
- Its 4-bit load is `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_compute_dtype="bfloat16")` with every other field at its default
  (FP4, no double quantization). By default that quantizes the vision
  encoder as well as the language model.

So there are two loads, differing only in what is quantized:

- `load_gemotions_model`  gemotions' exact load
- `load_model`            the same, with the vision path left in bfloat16
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

MODEL_31B = "google/gemma-4-31B-it"
START_TOKEN = 50        # gemotions: pool story activations from this token on
MAX_TEXT_TOKENS = 512   # gemotions: truncation length for stories
CAPTURE_LAYERS = (20, 25, 30, 35, 40, 45, 50)

# Submodules of the multimodal model that are not the language model.
NON_TEXT_MODULES = ("vision_tower", "embed_vision", "audio_tower", "embed_audio")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def bnb_4bit_config(skip_modules: list[str] | None = None):
    from transformers import BitsAndBytesConfig

    kwargs = dict(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
    if skip_modules is not None:
        kwargs["llm_int8_skip_modules"] = skip_modules
    return BitsAndBytesConfig(**kwargs)


def load_gemotions_model(model_id: str = MODEL_31B, four_bit: bool = True, **kwargs):
    """gemotions' exact load. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    load = {"device_map": "auto", **kwargs}
    if four_bit:
        load["quantization_config"] = bnb_4bit_config()
    else:
        load.setdefault("dtype", torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load)
    model.eval()
    return model, tokenizer


def load_model(
    model_id: str = MODEL_31B,
    four_bit: bool = True,
    quantize_vision: bool = False,
    **kwargs,
):
    """Load for image work. Returns (model, processor).

    With `quantize_vision=False` the vision encoder and its projector stay in
    bfloat16; the language model is quantized exactly as in gemotions' load.
    Passing a skip list replaces the library's default one, so `lm_head`
    (which the default skips) is re-added. Check the outcome with
    `quantization_report`.
    """
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    load = {"device_map": "auto", **kwargs}
    if four_bit:
        # Skip patterns match from the start of the full module name.
        skip = None if quantize_vision else [*(f"model.{m}" for m in NON_TEXT_MODULES), "lm_head"]
        load["quantization_config"] = bnb_4bit_config(skip)
        load.setdefault("dtype", torch.bfloat16)
    else:
        load.setdefault("dtype", torch.bfloat16)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **load)
    model.eval()
    return model, processor


def decoder_layers(model) -> torch.nn.ModuleList:
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model.layers
    if hasattr(inner, "layers"):
        return inner.layers
    raise RuntimeError("cannot find decoder layers on this model")


def quantization_report(model) -> dict[str, dict[str, int]]:
    """Count linear-layer types per top-level component.

    Used to confirm which parts were quantized: the language model should be
    all `Linear4bit`, and the vision encoder none when it was skipped.
    """
    report: dict[str, Counter] = {}
    for name, module in model.named_modules():
        cls = type(module).__name__
        if "Linear" not in cls or cls == "Gemma4ClippableLinear":
            continue
        parts = name.split(".")
        if parts[0] == "model" and len(parts) > 1 and parts[1] in (*NON_TEXT_MODULES, "language_model"):
            top = parts[1]
        elif parts[0] == "model":
            top = "language_model"  # text-only class: blocks sit directly under .model
        else:
            top = parts[0]
        report.setdefault(top, Counter())[cls] += 1
    return {k: dict(v) for k, v in report.items()}


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

class _StopForward(Exception):
    pass


class ResidualCapture:
    """Capture block outputs at the given layers during a forward pass.

        with ResidualCapture(model, [20, 40]) as cap:
            cap.run(**inputs)
        cap.acts[40]   # [batch, seq, hidden], on the layer's device

    With `early_exit=True` the forward pass is aborted after the deepest
    captured layer, so later layers and the LM head are never computed.
    """

    def __init__(self, model, layers: Iterable[int], early_exit: bool = True):
        self.model = model
        self.layers = sorted(set(int(i) for i in layers))
        self.early_exit = early_exit
        self.acts: dict[int, torch.Tensor] = {}
        self._handles = []
        blocks = decoder_layers(model)
        if self.layers[-1] >= len(blocks):
            raise ValueError(f"layer {self.layers[-1]} requested but model has {len(blocks)} blocks")
        self._blocks = blocks

    def __enter__(self):
        last = self.layers[-1]
        for i in self.layers:
            self._handles.append(self._blocks[i].register_forward_hook(self._make_hook(i, i == last)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def _make_hook(self, index: int, is_last: bool):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.acts[index] = hidden.detach()
            if is_last and self.early_exit:
                raise _StopForward
        return hook

    @torch.no_grad()
    def run(self, **inputs) -> dict[int, torch.Tensor]:
        self.acts = {}
        try:
            self.model(**inputs)
        except _StopForward:
            pass
        return self.acts


# --------------------------------------------------------------------------
# Text activations, gemotions-style
# --------------------------------------------------------------------------

def pool_story(hidden: torch.Tensor) -> torch.Tensor:
    """gemotions pooling for one sequence [seq, hidden]: mean from token 50,
    or over the whole sequence when it is no longer than that."""
    if hidden.shape[0] <= START_TOKEN:
        return hidden.float().mean(dim=0)
    return hidden[START_TOKEN:].float().mean(dim=0)


def story_activation(model, tokenizer, text: str, layers: Iterable[int], bos: bool = False) -> dict[int, "torch.Tensor"]:
    """Pooled activation of one raw (untemplated) text at each layer, on CPU.

    One text per forward pass with no padding, exactly as gemotions ran it.
    gemotions calls the bare tokenizer, which for Gemma 4 adds no BOS token;
    `bos=True` prepends one (and shifts the pooling window with it) so the
    reproduction check can tell which the caches were made with.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TEXT_TOKENS)
    if bos and inputs["input_ids"][0, 0] != tokenizer.bos_token_id:
        ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), inputs["input_ids"]], dim=1)[:, :MAX_TEXT_TOKENS]
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    inputs = {k: v.to(input_device(model)) for k, v in inputs.items()}
    with ResidualCapture(model, layers) as cap:
        acts = cap.run(**inputs)
    return {i: pool_story(h[0]).cpu() for i, h in acts.items()}
