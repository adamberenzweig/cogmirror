"""Fit the projection reference from the M1 activations.

    python scripts/video_reference.py

Reads data/acts/pass1/a2_prefill_*.npz (downloaded from the Volume) and
results/m1/manifest.jsonl, averages the two frames per clip, keeps the clips
of dev actors, and fits `EVSpace.fit_reference` for both readouts at every
captured layer. Writes results/m1/reference_a2_prefill.npz, which every
video run uses as its origin and scale (see cogmirror.store.load_reference).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from cogmirror import store
from cogmirror.evspace import EVSpace

GEM = "data/gemotions"
N_COMPONENTS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/acts/pass1")
    ap.add_argument("--layout", default="a2_prefill")
    ap.add_argument("--manifest", default="results/m1/manifest.jsonl")
    ap.add_argument("--out", default="results/m1/reference_a2_prefill.npz")
    ap.add_argument("--gemotions", default=GEM)
    args = ap.parse_args()

    split = {}
    for line in open(args.manifest):
        if line.strip():
            r = json.loads(line)
            split[(r["dataset"], r["clip"])] = r["split"]

    files = sorted(glob.glob(str(Path(args.acts, f"{args.layout}_*.npz"))))
    if not files:
        raise SystemExit(f"no {args.layout}_*.npz under {args.acts}")

    # Clip means over frames, dev clips only, accumulated shard by shard.
    sums: dict[str, dict] = {"last": {}, "image": {}}
    counts: dict[str, int] = {}
    layers = None
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            layers = z["layers"].tolist()
            keys = [(str(d), str(c)) for d, c in zip(z["dataset"], z["clip"])]
            dev = np.array([split.get(k) == "dev" for k in keys])
            for readout in ("last", "image"):
                a = z[readout]
                for i in np.flatnonzero(dev):
                    k = keys[i]
                    acc = sums[readout].get(k)
                    sums[readout][k] = a[i].astype(np.float32) if acc is None else acc + a[i].astype(np.float32)
            for i in np.flatnonzero(dev):
                counts[keys[i]] = counts.get(keys[i], 0) + 1
        print(f"{f}: {int(dev.sum())} dev frames")

    clips = sorted(counts)
    print(f"{len(clips)} dev clips, frames per clip: {sorted(set(counts.values()))}")
    refs = {}
    for layer_i, layer in enumerate(layers):
        space = EVSpace.load(args.gemotions, layer, n_components=N_COMPONENTS)
        for readout in ("last", "image"):
            x = np.stack([sums[readout][c][layer_i] / counts[c] for c in clips]).astype(np.float64)
            refs[(readout, layer)] = space.fit_reference(x)
            print(f"  {readout} L{layer}: mean norm {np.linalg.norm(refs[(readout, layer)].mean):.1f}, "
                  f"PC1 std {refs[(readout, layer)].pc_std[0]:.2f}, score std median {np.median(refs[(readout, layer)].score_std):.2f}")
    meta = {"layout": args.layout, "n_dev_clips": len(clips), "source": args.acts, "n_components": N_COMPONENTS,
            "emotions": EVSpace.load(args.gemotions, layers[0]).emotions}
    store.save_reference(args.out, refs, meta)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
