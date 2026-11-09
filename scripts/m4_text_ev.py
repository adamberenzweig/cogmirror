"""Text EVs for the story set: per sentence and per story, every captured layer.

    .venv/bin/python scripts/m4_text_ev.py                 # Modal (3 x L40S at most), then project locally
    .venv/bin/python scripts/m4_text_ev.py --project-only  # re-project results/m4/text_ev_acts.npz offline

Steps:
  1. activations on Modal: one raw-text forward pass per story with gemotions'
     model load (no chat template, no BOS); mean over each sentence's tokens
     and gemotions' story pooling, at --layers. Saved to results/m4/text_ev_acts.npz
     (float16, with the origins). Skipped when the file exists, unless --force.
  2. origins: the gemotions story mean per layer, recomputed from the caches on
     the Volume where one exists; otherwise the mean over the 70 stories.
  3. projection with the released vectors (data/gemotions): scores for the 171
     emotions and the top PCs per (story, sentence, layer) -> results/m4/text_ev.parquet,
     plus results/m4/text_ev_summary.json (the M3 screening check: is a target
     emotion among the top-3 scores of the whole-story row at --screen-layer).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cogmirror import text_ev  # noqa: E402
from cogmirror.evspace import EVSpace  # noqa: E402

CAPTURE_LAYERS = (20, 25, 30, 35, 40, 45, 50)   # cogmirror.model.CAPTURE_LAYERS, without importing torch
N_PCS = 10


def compute(args, stories: list[dict], layers: list[int], acts_path: Path) -> None:
    from modal_app import text_ev as mt

    batches = [stories[i:i + args.batch] for i in range(0, len(stories), args.batch)]
    print(f"{len(stories)} stories in {len(batches)} GPU calls (up to {mt.MAX_GPUS} x {mt.GPU}); origins on CPU")
    t0 = time.time()
    with mt.app.run(environment_name=args.env):
        origins_call = mt.origins_remote.spawn(layers)
        outs = list(mt.text_ev_remote.map(batches, kwargs={"layers": layers}))
        origins = origins_call.get()
    print(f"remote done in {time.time() - t0:.0f}s; model load {sorted({o['load_seconds'] for o in outs})}s, "
          f"compute {sum(o['seconds'] for o in outs):.0f}s GPU, {outs[0]['gpu']}")
    results, ids = [], []
    for o in outs:
        with np.load(io.BytesIO(o["npz"]), allow_pickle=False) as z:
            for i, sid in enumerate(o["ids"]):
                results.append({"sentence": z[f"sentence_{i}"], "story": z[f"story_{i}"], "story_full": z[f"story_full_{i}"],
                                "n_tokens": int(z["n_tokens"][i])})
                ids.append(sid)
        for sid, trunc in o.get("truncated", []):
            print(f"  warning: {sid}: sentences {trunc} fell beyond the 512-token truncation")
    cached = {int(k): np.asarray(v, dtype=np.float64) for k, v in origins["origins"].items()}
    print(f"origins from the Volume caches: {sorted(cached)}; missing: {origins['missing']}")
    arrays = text_ev.stack_results(ids, results, layers)
    o, src = text_ev.resolve_origins(args.gemotions, layers, arrays["story"], cached=cached, progress=print)
    text_ev.save_acts(acts_path, arrays, o, src)
    n_tok = arrays["n_tokens"]
    print(f"saved {acts_path}: {len(ids)} stories, {len(arrays['sentence'])} sentences, "
          f"tokens per story {n_tok.min()}..{n_tok.max()} (median {int(np.median(n_tok))})")
    meta = {"stories": str(args.stories), "layers": layers, "n_stories": len(ids), "n_sentences": int(len(arrays["sentence"])),
            "origin_source": {str(k): v for k, v in src.items()}, "origins_n_stories": origins.get("n_stories"),
            "gpu": outs[0]["gpu"], "load_seconds": [o["load_seconds"] for o in outs], "seconds": [o["seconds"] for o in outs],
            "quantization": outs[0].get("quantization"), "wall_seconds": round(time.time() - t0)}
    (acts_path.parent / "text_ev_run.json").write_text(json.dumps(meta, indent=1))


def project(args, stories: list[dict], acts_path: Path, out_dir: Path) -> None:
    acts = text_ev.load_acts(acts_path)
    missing = [l for l in acts.layers if l not in acts.origins]
    if missing:
        fallback = text_ev.origins_from_stories(acts.story, acts.layers)
        for l in missing:
            acts.origins[l], acts.origin_source[l] = fallback[l], text_ev.ORIGIN_STORIES
    spaces = {l: EVSpace.load(args.gemotions, l, n_components=args.pcs) for l in acts.layers}
    cols = text_ev.project_text(acts, spaces)
    table = out_dir / "text_ev.parquet"
    text_ev.write_table(table, cols, {"acts": str(acts_path), "layers": acts.layers, "n_pcs": args.pcs,
                                      "origin_source": json.dumps({str(k): v for k, v in acts.origin_source.items()})})
    print(f"wrote {table}: {len(cols['story_id'])} rows ({len(acts.story_ids)} stories x {len(acts.layers)} layers, "
          f"sentence rows + whole-story rows); origins: {acts.origin_source}")
    # Screening with two origins: the gemotions story mean (the stored one) and the
    # mean of this story set. The 70 stories sit about half a spread-SD away from the
    # gemotions mean, which colours every whole-story ranking with the same emotions;
    # M4 z-scores per dimension on development data, so the choice does not affect it.
    summaries = {"stored": text_ev.summarize(cols, stories, args.screen_layer)}
    within = text_ev.TextActs(**{**acts.__dict__, "origins": text_ev.origins_from_stories(acts.story, acts.layers),
                                 "origin_source": {l: text_ev.ORIGIN_STORIES for l in acts.layers}})
    summaries["story_set_mean"] = text_ev.summarize(text_ev.project_text(within, spaces, [args.screen_layer]), stories, args.screen_layer)
    (out_dir / "text_ev_summary.json").write_text(json.dumps({"screen_layer": args.screen_layer, "summaries": summaries}, indent=1))
    for name, summary in summaries.items():
        hits = [s for s in summary if s["hit"] is True]
        scored = [s for s in summary if s["hit"] is not None]
        ranks = [min(s["target_ranks"].values()) for s in scored if s["target_ranks"]]
        print(f"screening at layer {args.screen_layer}, origin {name}: a target emotion is in the top 3 for {len(hits)}/{len(scored)} "
              f"emotional stories; median best target rank {int(np.median(ranks)) if ranks else '-'} of {len(spaces[args.screen_layer].emotions)}")
    for s in summaries["story_set_mean"]:
        if s["hit"] is False and min(s["target_ranks"].values(), default=0) > 20:
            print(f"  miss (origin story_set_mean): {s['id']} targets {s['targets']} ranks {s['target_ranks']} top {s['top']}")
    ev = text_ev.table_at_layer(cols, args.screen_layer)
    v = np.array([ev.pcs[ev.lookup(s["id"], text_ev.STORY_ROW), 0] for s in stories if ev.lookup(s["id"], text_ev.STORY_ROW) is not None])
    kinds = np.array([s["kind"] for s in stories])
    for k in sorted(set(kinds)):
        print(f"  whole-story valence (PC1, raw units), {k}: mean {v[kinds == k].mean():+.2f} sd {v[kinds == k].std():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stories", default=str(ROOT / "data/stories/stories.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "results/m4"))
    ap.add_argument("--layers", default=",".join(str(l) for l in CAPTURE_LAYERS))
    ap.add_argument("--gemotions", default=str(ROOT / "data/gemotions"))
    ap.add_argument("--pcs", type=int, default=N_PCS)
    ap.add_argument("--screen-layer", type=int, default=40, help="layer for the story screening summary")
    ap.add_argument("--batch", type=int, default=24, help="stories per GPU call")
    ap.add_argument("--env", default="cogmirror", help="Modal environment")
    ap.add_argument("--project-only", action="store_true", help="skip Modal; re-project the saved activations")
    ap.add_argument("--force", action="store_true", help="recompute the activations even if the npz exists")
    args = ap.parse_args()

    stories = text_ev.load_stories(args.stories)
    layers = [int(l) for l in args.layers.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acts_path = out_dir / "text_ev_acts.npz"
    if args.project_only or (acts_path.exists() and not args.force):
        print(f"activations: using {acts_path}")
    else:
        compute(args, stories, layers, acts_path)
    project(args, stories, acts_path, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
