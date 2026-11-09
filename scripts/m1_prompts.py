"""Prompt robustness for M1: the same held-out frames under each paraphrase.

    python scripts/m1_prompts.py --run prompts --layout a2_prefill
"""

from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from cogmirror import metrics as M
from cogmirror.datasets import NEGATIVE
from cogmirror.evspace import EVSpace
from cogmirror.readouts import PROMPTS

GEM = "data/gemotions"
FOUR = list(NEGATIVE)


def load_prompt(run_dir: Path, layout: str, key: str):
    parts = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(str(run_dir / f"{layout}_{key}_*.npz")))]
    if not parts:
        return None
    ids = np.concatenate([np.char.add(np.char.add(p["dataset"].astype(str), "/"), p["clip"].astype(str)) for p in parts])
    order = np.argsort(ids)
    return {"ids": ids[order], "layers": parts[0]["layers"].tolist(),
            "last": np.concatenate([p["last"] for p in parts])[order],
            "image": np.concatenate([p["image"] for p in parts])[order]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="prompts")
    ap.add_argument("--layout", default="a2_prefill")
    ap.add_argument("--acts", default="data/acts")
    args = ap.parse_args()
    run_dir = Path(args.acts, args.run)

    manifest = {}
    for line in open("results/m1/manifest.jsonl"):
        r = json.loads(line)
        if r.get("face_found"):
            manifest[f"{r['dataset']}/{r['clip']}"] = r
    data = {k: d for k in PROMPTS if (d := load_prompt(run_dir, args.layout, k)) is not None}
    keys = list(data)
    ids = data[keys[0]]["ids"]
    for k in keys[1:]:
        assert (data[k]["ids"] == ids).all(), "prompt runs cover different clips"
    y = np.array([manifest[i]["emotion"] for i in ids])
    layers = data[keys[0]]["layers"]
    report = {"run": args.run, "layout": args.layout, "prompts": {k: PROMPTS[k] for k in keys}, "n_clips": len(ids), "results": []}
    print(f"{len(ids)} held-out clips, prompts: {keys}\n")
    for layer in (20, 35, 50):
        li = layers.index(layer)
        sp = EVSpace.load(GEM, layer)
        cols = [sp.emotions.index(c) for c in FOUR]
        neg = np.isin(y, FOUR)
        val, lab, sc = {}, {}, {}
        row = {"layer": layer, "per_prompt": {}}
        for k in keys:
            x = data[k]["last"][:, li].astype(np.float64)
            ref = sp.fit_reference(x)
            s = sp.scores(x, ref)
            v = sp.pcs(x, ref)[:, 0]
            val[k], sc[k] = v, s
            lab[k] = s[:, cols].argmax(1)
            row["per_prompt"][k] = {"valence_auc": round(M.valence_auc(v, y), 3),
                                    "four_way": round(float((np.array(FOUR)[lab[k][neg]] == y[neg]).mean()), 3)}
        pairs = list(combinations(keys, 2))
        row["valence_corr_between_prompts"] = {"min": round(min(np.corrcoef(val[a], val[b])[0, 1] for a, b in pairs), 3),
                                               "mean": round(float(np.mean([np.corrcoef(val[a], val[b])[0, 1] for a, b in pairs])), 3)}
        prof = [float(np.median(np.sum(sc[a] * sc[b], 1) / (np.linalg.norm(sc[a], axis=1) * np.linalg.norm(sc[b], axis=1)))) for a, b in pairs]
        row["profile_cosine_between_prompts_median"] = {"min": round(min(prof), 3), "mean": round(float(np.mean(prof)), 3)}
        row["four_way_label_agreement"] = {"min": round(min(float((lab[a][neg] == lab[b][neg]).mean()) for a, b in pairs), 3),
                                           "mean": round(float(np.mean([(lab[a][neg] == lab[b][neg]).mean() for a, b in pairs])), 3)}
        report["results"].append(row)
        print(f"[last L{layer}]")
        for k, v in row["per_prompt"].items():
            print(f"   {k:8s} valence AUC {v['valence_auc']:.3f}  4-way {v['four_way']:.3f}   \"{PROMPTS[k]}\"")
        print(f"   between prompts: valence r {row['valence_corr_between_prompts']}, profile cosine {row['profile_cosine_between_prompts_median']}, "
              f"4-way agreement {row['four_way_label_agreement']}\n")
    Path(f"results/m1/{args.run}_prompts.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
