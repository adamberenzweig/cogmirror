"""Noise-floor analysis for M1.

    python scripts/m1_noise.py --run noise --layout a2_prefill

Compares the base run with each variant (nuisance perturbations, a different
batch size) on the same frames: how far does the valence readout and the
4-way nearest-vector label move, against how far expressions differ?
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from cogmirror import metrics as M
from cogmirror.datasets import NEGATIVE
from cogmirror.evspace import EVSpace

GEM = "data/gemotions"
FOUR = list(NEGATIVE)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("dataset", "clip", "k", "layers", "last", "image")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="noise")
    ap.add_argument("--layout", default="a2_prefill")
    ap.add_argument("--acts", default="data/acts")
    args = ap.parse_args()
    run_dir = Path(args.acts, args.run)

    manifest = {}
    for line in open("results/m1/manifest.jsonl"):
        r = json.loads(line)
        if r.get("face_found"):
            manifest[(r["dataset"], r["clip"])] = r
    base = load(run_dir / f"{args.layout}_base_0.npz")
    keys = list(zip(base["dataset"].tolist(), base["clip"].tolist()))
    y = np.array([manifest[k]["emotion"] for k in keys])
    layers = base["layers"].tolist()
    variants = {}
    for f in sorted(glob.glob(str(run_dir / f"{args.layout}_*_0.npz"))):
        tag = Path(f).name[len(args.layout) + 1:-len("_0.npz")]
        if tag != "base":
            v = load(f)
            assert v["clip"].tolist() == base["clip"].tolist()
            variants[tag] = v

    report = {"run": args.run, "layout": args.layout, "n_frames": len(keys), "variants": list(variants), "results": []}
    happy, neg = y == "happy", np.isin(y, FOUR)
    for readout in ("last", "image"):
        for li, layer in enumerate(layers):
            sp = EVSpace.load(GEM, layer)
            xb = base[readout][:, li].astype(np.float64)
            ref = sp.fit_reference(xb)
            vb = sp.pcs(xb, ref)[:, 0]
            sb = sp.scores(xb, ref)
            cols = [sp.emotions.index(c) for c in FOUR]
            lab_b = sb[:, cols].argmax(1)
            expression_gap = float(vb[happy].mean() - vb[neg].mean())
            # Within-expression spread of valence across different clips: what "different clip, same label" looks like.
            within = float(np.mean([vb[y == e].std() for e in ["happy", *FOUR] if (y == e).sum() > 5]))
            row = {"readout": readout, "layer": layer, "expression_gap": round(expression_gap, 3),
                   "within_expression_sd": round(within, 3), "variants": {}}
            for tag, v in variants.items():
                xv = v[readout][:, li].astype(np.float64)
                vv = sp.pcs(xv, ref)[:, 0]
                sv = sp.scores(xv, ref)
                d = vv - vb
                rms = float(np.sqrt(np.mean(d**2)))
                row["variants"][tag] = {
                    "valence_rms_shift": round(rms, 3),
                    "gap_over_shift": round(expression_gap / rms, 2) if rms > 0 else None,
                    "valence_corr": round(float(np.corrcoef(vb, vv)[0, 1]), 4),
                    "four_way_label_flips": round(float((sv[neg][:, cols].argmax(1) != lab_b[neg]).mean()), 3),
                    "profile_cosine_median": round(float(np.median(np.sum(sb * sv, 1) / (np.linalg.norm(sb, axis=1) * np.linalg.norm(sv, axis=1) + 1e-12))), 4),
                }
            report["results"].append(row)

    Path("results/m1").mkdir(parents=True, exist_ok=True)
    Path(f"results/m1/{args.run}_noise.json").write_text(json.dumps(report, indent=1))
    print(f"{len(keys)} frames; variants: {', '.join(variants)}")
    print("gap = mean valence(happy) - mean valence(negatives); shift = RMS change in valence under the variant\n")
    for row in report["results"]:
        if row["layer"] not in (20, 35, 50):
            continue
        print(f"[{row['readout']} L{row['layer']}] gap {row['expression_gap']:.2f}, within-expression sd {row['within_expression_sd']:.2f}")
        for tag, v in row["variants"].items():
            print(f"   {tag:11s} shift {v['valence_rms_shift']:.3f}  gap/shift {v['gap_over_shift']:>6}  r {v['valence_corr']:.3f}"
                  f"  4-way flips {v['four_way_label_flips']:.1%}  profile cos {v['profile_cosine_median']:.3f}")


if __name__ == "__main__":
    main()
