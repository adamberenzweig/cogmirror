"""Text emotion vectors per sentence and per story (M3 task 5 / M4 part 1).

One forward pass per story over the raw story text (no chat template, no
BOS, as in gemotions' extraction), capturing the residual stream at every
layer in `model.CAPTURE_LAYERS`. The story so far is each sentence's context
by construction. Per sentence, the activation is the mean over that
sentence's tokens; per story, gemotions' pooling (mean from token 50) plus
a plain mean over all tokens.

Projection uses the gemotions story mean as the origin, recomputed from the
per-story caches where a layer's cache exists and otherwise fitted on the
story set itself (`origins_from_stories`). Scores are `(x - origin) . v_e /
|v_e|` and PCs are coordinates in the fixed PCA basis of the 171 vectors.
Nothing is scaled here: M4 z-scores each modality on development data.

Nothing in this module needs torch except `story_text_activations`.

Files (results/m4/):
    text_ev_acts.npz    raw activations: sentence [S, L, D] f16, story [N, L, D] f16,
                        story_full [N, L, D] f16, index arrays, layers, origins
    text_ev.parquet     one row per (story, sentence, layer) with scores and PCs;
                        sentence_idx -1 is the whole story (gemotions pooling)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import evspace
from .evspace import EVSpace

STORY_ROW = -1          # sentence_idx of a whole-story row in the table
ORIGIN_CACHE = "cache"  # origin recomputed from the gemotions per-story caches
ORIGIN_STORIES = "stories"  # origin fitted on the story set itself


# --------------------------------------------------------------------------
# Stories
# --------------------------------------------------------------------------

def load_stories(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sentence_char_spans(text: str, sentences: list[str]) -> list[tuple[int, int]]:
    """Character span of each sentence inside `text`, found in order.

    The story text is the sentences joined by single spaces, but the search
    is sequential and whitespace-tolerant so a story whose text was edited
    still maps as long as every sentence occurs in order.
    """
    spans = []
    pos = 0
    for s in sentences:
        s = s.strip()
        if not s:
            raise ValueError("empty sentence")
        i = text.find(s, pos)
        if i < 0:
            raise ValueError(f"sentence not found in text from position {pos}: {s[:40]!r}")
        spans.append((i, i + len(s)))
        pos = i + len(s)
    return spans


def token_spans(offsets, char_spans: list[tuple[int, int]]) -> np.ndarray:
    """Token [start, stop) per sentence from the tokenizer's character
    offsets (`return_offsets_mapping`), [n_sentences, 2].

    A token belongs to the sentence whose character span it overlaps; a
    token straddling a boundary (rare: a space-prefixed word piece) goes to
    the sentence that holds its last character. Every sentence gets at
    least one token unless the text was truncated before it, in which case
    its span is empty (start == stop) and the caller should drop it.
    """
    offsets = [(int(a), int(b)) for a, b in offsets]
    out = np.zeros((len(char_spans), 2), dtype=np.int64)
    starts = np.array([a for a, _ in char_spans])
    ends = np.array([b for _, b in char_spans])
    owner = np.full(len(offsets), -1, dtype=np.int64)
    for k, (a, b) in enumerate(offsets):
        if b <= a:          # special or empty token
            continue
        last = b - 1
        hit = np.flatnonzero((starts <= last) & (last < ends))
        if len(hit):
            owner[k] = hit[0]
            continue
        # inside a gap between sentences (the joining space): give it to the next sentence
        nxt = np.flatnonzero(starts > last)
        if len(nxt):
            owner[k] = nxt[0]
    for s in range(len(char_spans)):
        idx = np.flatnonzero(owner == s)
        if len(idx):
            out[s] = (idx[0], idx[-1] + 1)
        else:
            out[s] = (0, 0)
    # spans must be contiguous and ordered; a tokenizer that reorders would break pooling
    for s in range(1, len(char_spans)):
        if out[s, 1] > out[s, 0] and out[s - 1, 1] > out[s - 1, 0] and out[s, 0] < out[s - 1, 1]:
            raise ValueError(f"token spans of sentences {s - 1} and {s} overlap: {out[s - 1]} {out[s]}")
    return out


def pool_spans(hidden: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Mean of hidden [seq, D] over each [start, stop) span -> [n, D]; NaN for empty spans."""
    hidden = np.asarray(hidden)
    out = np.full((len(spans), hidden.shape[-1]), np.nan, dtype=np.float32)
    for k, (a, b) in enumerate(spans):
        if b > a:
            out[k] = hidden[a:b].astype(np.float32).mean(axis=0)
    return out


# --------------------------------------------------------------------------
# The forward pass (needs torch and a loaded model)
# --------------------------------------------------------------------------

def story_text_activations(model, tokenizer, story: dict, layers) -> dict:
    """One raw-text forward pass over a story; per-sentence and per-story activations.

    Returns {"sentence": [S, L, D], "story": [L, D] (gemotions pooling from
    token 50), "story_full": [L, D] (mean over every token), "n_tokens",
    "token_spans": [S, 2], "truncated_sentences": [...]} with float32 arrays
    on CPU. Layers are in the order given.
    """
    import torch

    from .model import MAX_TEXT_TOKENS, ResidualCapture, input_device, pool_story

    layers = [int(l) for l in layers]
    text = story["text"]
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TEXT_TOKENS, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    spans = token_spans(offsets, sentence_char_spans(text, story["sentences"]))
    inputs = {k: v.to(input_device(model)) for k, v in enc.items()}
    with ResidualCapture(model, layers) as cap:
        acts = cap.run(**inputs)
    n_tok = int(inputs["input_ids"].shape[1])
    sent = np.zeros((len(spans), len(layers), 0), dtype=np.float32)
    per_layer_sent, per_layer_story, per_layer_full = [], [], []
    for l in layers:
        h = acts[l][0].float().cpu().numpy()            # [seq, D]
        per_layer_sent.append(pool_spans(h, spans))     # [S, D]
        per_layer_story.append(pool_story(torch.from_numpy(h)).numpy())
        per_layer_full.append(h.mean(axis=0))
    sent = np.stack(per_layer_sent, axis=1)             # [S, L, D]
    return {
        "sentence": sent,
        "story": np.stack(per_layer_story),             # [L, D]
        "story_full": np.stack(per_layer_full),         # [L, D]
        "n_tokens": n_tok,
        "token_spans": spans,
        "truncated_sentences": [int(k) for k, (a, b) in enumerate(spans) if b <= a],
    }


# --------------------------------------------------------------------------
# Raw activation store
# --------------------------------------------------------------------------

@dataclass
class TextActs:
    """Raw per-sentence and per-story activations for a story set."""
    layers: list[int]
    story_ids: list[str]            # [N]
    sentence_story: np.ndarray      # [S] index into story_ids
    sentence_idx: np.ndarray        # [S]
    sentence: np.ndarray            # [S, L, D]
    story: np.ndarray               # [N, L, D] gemotions pooling
    story_full: np.ndarray          # [N, L, D] mean over all tokens
    n_tokens: np.ndarray            # [N]
    origins: dict[int, np.ndarray]  # layer -> [D]
    origin_source: dict[int, str]   # layer -> "cache" | "stories"

    @property
    def sentence_story_ids(self) -> np.ndarray:
        return np.asarray(self.story_ids)[self.sentence_story]

    def layer_index(self, layer: int) -> int:
        if int(layer) not in self.layers:
            raise ValueError(f"layer {layer} not in {self.layers}")
        return self.layers.index(int(layer))


def stack_results(story_ids: list[str], results: list[dict], layers) -> dict:
    """Combine per-story `story_text_activations` outputs into one set of arrays."""
    sent = np.concatenate([r["sentence"] for r in results]).astype(np.float16)
    sentence_story = np.concatenate([np.full(len(r["sentence"]), i) for i, r in enumerate(results)]).astype(np.int64)
    sentence_idx = np.concatenate([np.arange(len(r["sentence"])) for r in results]).astype(np.int64)
    return {
        "layers": np.asarray([int(l) for l in layers], dtype=np.int64),
        "story_ids": np.asarray(story_ids, dtype=str),
        "sentence_story": sentence_story,
        "sentence_idx": sentence_idx,
        "sentence": sent,
        "story": np.stack([r["story"] for r in results]).astype(np.float16),
        "story_full": np.stack([r["story_full"] for r in results]).astype(np.float16),
        "n_tokens": np.asarray([r["n_tokens"] for r in results], dtype=np.int64),
    }


def save_acts(path: str | Path, arrays: dict, origins: dict[int, np.ndarray], origin_source: dict[int, str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(arrays)
    for l, o in origins.items():
        out[f"origin_L{int(l)}"] = np.asarray(o, dtype=np.float32)
    out["origin_source"] = np.array(json.dumps({str(k): v for k, v in origin_source.items()}))
    with open(path, "wb") as f:
        np.savez(f, **out)
    return path


def load_acts(path: str | Path) -> TextActs:
    with np.load(path, allow_pickle=False) as z:
        layers = [int(l) for l in z["layers"].tolist()]
        src = json.loads(str(z["origin_source"])) if "origin_source" in z.files else {}
        origins = {l: z[f"origin_L{l}"].astype(np.float64) for l in layers if f"origin_L{l}" in z.files}
        return TextActs(
            layers=layers,
            story_ids=z["story_ids"].tolist(),
            sentence_story=z["sentence_story"],
            sentence_idx=z["sentence_idx"],
            sentence=z["sentence"],
            story=z["story"],
            story_full=z["story_full"],
            n_tokens=z["n_tokens"],
            origins=origins,
            origin_source={int(k): v for k, v in src.items()},
        )


# --------------------------------------------------------------------------
# Origins and projection
# --------------------------------------------------------------------------

def origin_from_cache(gemotions_root: str | Path, layer: int) -> np.ndarray:
    """gemotions' global story mean at `layer`, recomputed from the per-story caches."""
    return evspace.recompute_from_cache(gemotions_root, layer).global_mean


def origins_from_stories(story_acts: np.ndarray, layers) -> dict[int, np.ndarray]:
    """Fallback origin: the mean of the whole-story activations, per layer."""
    a = np.asarray(story_acts, dtype=np.float64)
    return {int(l): a[:, k].mean(axis=0) for k, l in enumerate(layers)}


def resolve_origins(gemotions_root: str | Path, layers, story_acts: np.ndarray,
                    cached: dict[int, np.ndarray] | None = None, progress=None) -> tuple[dict, dict]:
    """Origin per layer: from the caches where present, else from the stories."""
    origins, source = {}, {}
    fallback = origins_from_stories(story_acts, layers)
    for l in layers:
        l = int(l)
        if cached and l in cached:
            origins[l], source[l] = np.asarray(cached[l], dtype=np.float64), ORIGIN_CACHE
            continue
        if evspace.cache_dir(gemotions_root, l).exists():
            if progress:
                progress(f"  origin for layer {l} from the gemotions cache")
            origins[l], source[l] = origin_from_cache(gemotions_root, l), ORIGIN_CACHE
        else:
            origins[l], source[l] = fallback[l], ORIGIN_STORIES
    return origins, source


def project_text(acts: TextActs, spaces: dict[int, EVSpace], layers=None) -> dict[str, np.ndarray]:
    """Scores and PCs per sentence and per story at each layer, with the stored origins.

    Returns table columns: story_id, sentence_idx (-1 = whole story), layer,
    n_tokens, origin, score_<emotion>..., pc1..pcK.
    """
    layers = [int(l) for l in (layers or acts.layers)]
    rows_story, rows_sent, rows_layer, scores, pcs, origin_src = [], [], [], [], [], []
    for l in layers:
        k = acts.layer_index(l)
        sp = spaces[l]
        origin = acts.origins[l]
        x_sent = acts.sentence[:, k].astype(np.float64)
        x_story = acts.story[:, k].astype(np.float64)
        for x, sid, sidx in ((x_sent, acts.sentence_story_ids, acts.sentence_idx),
                             (x_story, np.asarray(acts.story_ids), np.full(len(acts.story_ids), STORY_ROW))):
            scores.append(sp.scores(x, origin))
            pcs.append(sp.pcs(x, origin))
            rows_story.extend(sid.tolist())
            rows_sent.extend(int(i) for i in sidx)
            rows_layer.extend([l] * len(x))
            origin_src.extend([acts.origin_source.get(l, "")] * len(x))
    scores = np.concatenate(scores)
    pcs = np.concatenate(pcs)
    emotions = spaces[layers[0]].emotions
    cols: dict[str, np.ndarray] = {
        "story_id": np.asarray(rows_story, dtype=str),
        "sentence_idx": np.asarray(rows_sent, dtype=np.int64),
        "layer": np.asarray(rows_layer, dtype=np.int64),
        "origin": np.asarray(origin_src, dtype=str),
    }
    for j, e in enumerate(emotions):
        cols[f"score_{e}"] = scores[:, j]
    for j in range(pcs.shape[1]):
        cols[f"pc{j + 1}"] = pcs[:, j]
    return cols


# --------------------------------------------------------------------------
# The projected table
# --------------------------------------------------------------------------

@dataclass
class TextEV:
    """Projected text EVs at one layer, indexed by (story_id, sentence_idx)."""
    layer: int
    emotions: list[str]
    index: dict[tuple[str, int], int]
    scores: np.ndarray      # [R, E]
    pcs: np.ndarray         # [R, K]
    story_ids: np.ndarray   # [R]
    sentence_idx: np.ndarray  # [R]

    def lookup(self, story_id: str, sentence_idx: int) -> int | None:
        return self.index.get((str(story_id), int(sentence_idx)))

    def n_sentences(self, story_id: str) -> int:
        return int(sum(1 for (s, i) in self.index if s == story_id and i >= 0))


def table_at_layer(cols: dict[str, np.ndarray], layer: int) -> TextEV:
    sel = np.asarray(cols["layer"]) == int(layer)
    if not sel.any():
        raise ValueError(f"no rows at layer {layer}; layers present: {sorted(set(np.asarray(cols['layer']).tolist()))}")
    emotions = [c[len("score_"):] for c in cols if c.startswith("score_")]
    pc_cols = sorted((c for c in cols if c.startswith("pc") and c[2:].isdigit()), key=lambda c: int(c[2:]))
    scores = np.stack([np.asarray(cols[f"score_{e}"], dtype=np.float64)[sel] for e in emotions], axis=1)
    pcs = np.stack([np.asarray(cols[c], dtype=np.float64)[sel] for c in pc_cols], axis=1)
    sid = np.asarray(cols["story_id"]).astype(str)[sel]
    sidx = np.asarray(cols["sentence_idx"], dtype=np.int64)[sel]
    index = {(str(s), int(i)): r for r, (s, i) in enumerate(zip(sid, sidx))}
    return TextEV(int(layer), emotions, index, scores, pcs, sid, sidx)


def write_table(path: str | Path, cols: dict[str, np.ndarray], meta: dict | None = None) -> Path:
    from .store import write_parquet

    return write_parquet(path, cols, {k: str(v) for k, v in (meta or {}).items()})


def read_table(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    from .store import read_parquet

    return read_parquet(path)


def summarize(cols: dict[str, np.ndarray], stories: list[dict], layer: int, top_k: int = 3) -> list[dict]:
    """Per story at one layer: its target emotions, the top-scoring emotions
    of the whole-story row, and whether a target is among them (the M3
    screening check)."""
    ev = table_at_layer(cols, layer)
    out = []
    for s in stories:
        r = ev.lookup(s["id"], STORY_ROW)
        if r is None:
            continue
        order = np.argsort(ev.scores[r])[::-1]
        top = [ev.emotions[j] for j in order[:top_k]]
        targets = list(s.get("emotions") or [])
        ranks = {e: int(np.flatnonzero(order == ev.emotions.index(e))[0]) + 1 for e in targets if e in ev.emotions}
        out.append({"id": s["id"], "kind": s.get("kind"), "targets": targets, "top": top, "target_ranks": ranks,
                    "valence": float(ev.pcs[r, 0]), "hit": any(v <= top_k for v in ranks.values()) if ranks else None})
    return out
