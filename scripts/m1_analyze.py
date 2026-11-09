"""Gate 1 analysis on stored activations.

    python scripts/m1_analyze.py --run pass1

Reads results/m1/manifest.jsonl and data/acts/<run>/ (downloaded from the
Volume), writes results/m1/<run>_metrics.json, and prints the gate table.

Settings (layout, readout, layer) are chosen on dev actors; reported numbers
are for held-out actors unless stated.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import warnings

import numpy as np

from cogmirror import metrics as M
from cogmirror.datasets import NEGATIVE, SIX
from cogmirror.evspace import EVSpace

GEM = "data/gemotions"
READOUTS = ("last", "image")
FOUR = list(NEGATIVE)
FIVE = FOUR + ["happy"]
# Words whose log-probabilities stand for each label in the verbal baseline.
LABEL_WORDS = {
    "angry": ["angry", "mad"], "disgusted": ["disgusted", "disgust"], "afraid": ["afraid", "scared", "fearful"],
    "sad": ["sad"], "happy": ["happy"],
}


def load_layout(run_dir: Path, layout: str, clip_index: dict):
    """Clip-level means over frames: {"last": [C, L, D], "image": ..., "logp": [C, W]}."""
    files = sorted(glob.glob(str(run_dir / f"{layout}_*.npz")))
    if not files:
        return None
    out, counts, words, layers = {}, np.zeros(len(clip_index)), None, None
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            idx = np.array([clip_index[(d, c)] for d, c in zip(z["dataset"], z["clip"])])
            layers = z["layers"].tolist()
            for key in ("last", "image", "logp"):
                if key not in z.files:
                    continue
                a = z[key].astype(np.float32)
                if key not in out:
                    out[key] = np.zeros((len(clip_index),) + a.shape[1:], dtype=np.float32)
                np.add.at(out[key], idx, a)
            np.add.at(counts, idx, 1)
            if "words" in z.files:
                words = [str(w) for w in z["words"]]
    for key in out:
        out[key] /= np.maximum(counts, 1).reshape((-1,) + (1,) * (out[key].ndim - 1))
    out.update(counts=counts, words=words, layers=layers)
    return out


def blendshape_table(manifest, clips, frames):
    names = None
    acc = {}
    for r in manifest:
        if not r.get("face_found") or r["k"] not in frames:
            continue
        if names is None:
            names = sorted(r["blendshapes"])
        acc.setdefault((r["dataset"], r["clip"]), []).append([r["blendshapes"][n] for n in names])
    return np.array([np.mean(acc[c], axis=0) if c in acc else np.zeros(len(names)) for c in clips])


def crema_human(clip_meta):
    """Majority-vote accuracy of CREMA-D's face-only raters, on the same clips."""
    path = Path("data/crema/tabulatedVotes.csv")
    if not path.exists():
        return None
    code = {"A": "angry", "D": "disgusted", "F": "afraid", "H": "happy", "N": "neutral", "S": "sad"}
    votes = {}
    for row in csv.DictReader(open(path)):
        if row[""].startswith("2"):       # 1xxxxx voice only, 2xxxxx face only, 3xxxxx audio-visual
            votes[row["fileName"]] = row["emoVote"]
    res = {}
    for split in ("dev", "heldout"):
        hits = [code.get(votes[m["clip"]], "tie") == m["emotion"] for m in clip_meta
                if m["dataset"] == "crema" and m["split"] == split and m["clip"] in votes]
        res[split] = {"accuracy": float(np.mean(hits)), "n": len(hits)}
    return res


def main():
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pass1")
    ap.add_argument("--acts", default="data/acts")
    ap.add_argument("--frames", type=int, nargs="*", default=[1, 3])
    ap.add_argument("--debug-split", action="store_true", help="split clips by parity (for tiny trial runs only)")
    args = ap.parse_args()
    run_dir = Path(args.acts, args.run)

    manifest = [json.loads(l) for l in open("results/m1/manifest.jsonl") if l.strip()]
    meta_by_clip = {}
    for r in manifest:
        if r.get("face_found") and r["k"] in args.frames:
            meta_by_clip.setdefault((r["dataset"], r["clip"]), r)
    clips = sorted(meta_by_clip)
    clip_index = {c: i for i, c in enumerate(clips)}
    meta = [meta_by_clip[c] for c in clips]
    ds = np.array([m["dataset"] for m in meta])
    y = np.array([m["emotion"] for m in meta])
    actor = np.array([m["actor"] for m in meta])
    split = np.array([m["split"] for m in meta])
    if args.debug_split:
        split = np.array(["dev" if i % 2 else "heldout" for i in range(len(meta))])
        for m, sp_ in zip(meta, split):
            m["split"] = sp_
    intensity = np.array([m["intensity"] for m in meta])
    dev, held = split == "dev", split == "heldout"
    datasets = sorted(set(ds.tolist()))
    report = {"run": args.run, "n_clips": len(clips),
              "counts": {d: {s: int(((ds == d) & (split == s)).sum()) for s in ("dev", "heldout")} for d in datasets},
              "actors": {d: {s: int(len(set(actor[(ds == d) & (split == s)]))) for s in ("dev", "heldout")} for d in datasets},
              "configs": []}

    def per_dataset(fn, mask):
        vals = {d: fn((ds == d) & mask) for d in datasets}
        ok = [v for v in vals.values() if v == v]
        vals["mean"] = float(np.mean(ok)) if ok else float("nan")
        return vals

    spaces = {}
    layouts = sorted({Path(f).name.rsplit("_", 1)[0] for f in glob.glob(str(run_dir / "a*_*.npz"))})
    verbal = {}
    for layout in layouts:
        data = load_layout(run_dir, layout, clip_index)
        have = data["counts"] > 0
        print(f"{layout}: {int(have.sum())}/{len(clips)} clips with activations")
        layers = data["layers"]

        # ---- verbal baseline: the model's own next-word probabilities
        if data.get("logp") is not None and data["words"]:
            w, lp = data["words"], data["logp"]
            grp = {k: M.logsumexp(lp[:, [w.index(x) for x in v if x in w]]) for k, v in LABEL_WORDS.items()}
            val = grp["happy"] - M.logsumexp(np.stack([grp[k] for k in FOUR], axis=1))
            four = np.stack([grp[k] for k in FOUR], axis=1)
            five = np.stack([grp[k] for k in FIVE], axis=1)
            verbal[layout] = {
                "valence_auc": per_dataset(lambda m: M.valence_auc(val[m], y[m]), held & have),
                "four_way": per_dataset(lambda m: float((np.array(FOUR)[four[m & np.isin(y, FOUR)].argmax(1)] == y[m & np.isin(y, FOUR)]).mean()), held & have),
                "five_way": per_dataset(lambda m: float((np.array(FIVE)[five[m & np.isin(y, FIVE)].argmax(1)] == y[m & np.isin(y, FIVE)]).mean()), held & have),
            }

        for readout in READOUTS:
            acts = data[readout]
            for li, layer in enumerate(layers):
                if layer not in spaces:
                    spaces[layer] = EVSpace.load(GEM, layer, n_components=10)
                sp = spaces[layer]
                x = acts[:, li].astype(np.float64)
                ref = sp.fit_reference(x[dev & have])           # origin and scale from dev actors, all expressions
                scores = sp.scores(x, ref)
                pcs = sp.pcs(x, ref)
                val = pcs[:, 0]
                neutral = y == "neutral"
                scores_b = M.center_by_group(scores, actor, neutral)   # per-person baseline: own neutral clips
                val_b = M.center_by_group(val, actor, neutral)
                cfg = {"layout": layout, "readout": readout, "layer": layer}
                for name, mask in (("dev", dev & have), ("heldout", held & have)):
                    cfg[name] = {
                        "valence_auc": per_dataset(lambda m: M.valence_auc(val[m], y[m]), mask),
                        "valence_auc_baselined": per_dataset(lambda m: M.valence_auc(val_b[m], y[m]), mask),
                        "four_way": per_dataset(lambda m: M.nearest_vector_accuracy(scores[m], y[m], sp.emotions, FOUR)["accuracy"], mask),
                        "four_way_baselined": per_dataset(lambda m: M.nearest_vector_accuracy(scores_b[m], y[m], sp.emotions, FOUR)["accuracy"], mask),
                        "five_way_baselined": per_dataset(lambda m: M.nearest_vector_accuracy(scores_b[m], y[m], sp.emotions, FIVE)["accuracy"], mask),
                    }
                # Remove the valence direction (PC1) from the activations, then repeat the test.
                xc = x - ref.mean
                p1 = sp.components[0]
                s_res = (xc - np.outer(xc @ p1, p1)) @ sp.unit.T
                s_res = s_res / np.maximum(s_res[dev & have].std(axis=0), 1e-12)
                resid = M.center_by_group(s_res, actor, neutral)
                cfg["heldout"]["four_way_valence_removed"] = per_dataset(
                    lambda m: M.nearest_vector_accuracy(resid[m], y[m], sp.emotions, FOUR)["accuracy"], held & have)
                six = np.isin(y, SIX)
                cfg["heldout"]["pc_eta2_emotion"] = [round(M.eta_squared(pcs[held & have & six, k], y[held & have & six]), 3)
                                                     for k in range(pcs.shape[1])]
                cfg["heldout"]["identity"] = {
                    "valence_eta2_actor_neutral_clips": per_dataset(lambda m: M.eta_squared(val[m & neutral], actor[m & neutral]), held & have),
                    "valence_eta2_actor": per_dataset(lambda m: M.eta_squared(val[m & six], actor[m & six]), held & have),
                    "valence_eta2_emotion": per_dataset(lambda m: M.eta_squared(val[m & six], y[m & six]), held & have),
                    "valence_eta2_actor_baselined": per_dataset(lambda m: M.eta_squared(val_b[m & six], actor[m & six]), held & have),
                    "valence_eta2_emotion_baselined": per_dataset(lambda m: M.eta_squared(val_b[m & six], y[m & six]), held & have),
                    "scores_eta2_actor": per_dataset(lambda m: M.eta_squared(scores[m & six], actor[m & six]), held & have),
                    "scores_eta2_emotion": per_dataset(lambda m: M.eta_squared(scores[m & six], y[m & six]), held & have),
                    "scores_eta2_actor_baselined": per_dataset(lambda m: M.eta_squared(scores_b[m & six], actor[m & six]), held & have),
                    "scores_eta2_emotion_baselined": per_dataset(lambda m: M.eta_squared(scores_b[m & six], y[m & six]), held & have),
                }
                # Trained comparison: a linear probe on the 171 scores (dev -> held-out).
                tr, te = dev & have & six, held & have & six
                cfg["heldout"]["probe_six_way_on_scores"] = M.probe_accuracy(scores[tr], y[tr], scores[te], y[te])
                report["configs"].append(cfg)
        del data
    report["verbal"] = verbal

    # ---- zero-shot forced choice by generation (perception ceiling)
    rows = []
    for f in sorted(glob.glob(str(run_dir / "choices_*.json"))):
        rows.extend(json.load(open(f)))
    if rows:
        pred = {(r["dataset"], r["clip"]): r["pred"] for r in rows}
        p = np.array([pred.get(c, "") for c in clips])
        got = np.array([c in pred for c in clips])
        report["choices"] = {
            "unparsed": float(np.mean(p[got] == "")),
            "accuracy_own_labels": {s: per_dataset(lambda m: float((p[m] == y[m]).mean()), got & (split == s)) for s in ("dev", "heldout")},
            "accuracy_six": {s: per_dataset(lambda m: float((p[m & np.isin(y, SIX)] == y[m & np.isin(y, SIX)]).mean()), got & (split == s)) for s in ("dev", "heldout")},
            "recall_heldout": {d: {e: round(float((p[(ds == d) & held & got & (y == e)] == e).mean()), 3)
                                   for e in sorted(set(y[ds == d].tolist()))} for d in datasets},
            "by_intensity_heldout": {d: {i: round(float((p[(ds == d) & held & got & (intensity == i)] == y[(ds == d) & held & got & (intensity == i)]).mean()), 3)
                                         for i in sorted(set(intensity[ds == d].tolist()))} for d in datasets},
            "four_way_among_negatives": per_dataset(lambda m: float((p[m & np.isin(y, FOUR)] == y[m & np.isin(y, FOUR)]).mean()), got & held),
        }

    # ---- blendshape baseline (trained on dev actors)
    bs = blendshape_table(manifest, clips, args.frames)
    six = np.isin(y, SIX)
    vmask = np.isin(y, FIVE)
    report["blendshapes"] = {
        "valence_auc": per_dataset(lambda m: M.probe_auc(bs[dev & vmask], (y[dev & vmask] == "happy").astype(int),
                                                         bs[m & vmask], (y[m & vmask] == "happy").astype(int)), held),
        "four_way": per_dataset(lambda m: M.probe_accuracy(bs[dev & np.isin(y, FOUR)], y[dev & np.isin(y, FOUR)],
                                                           bs[m & np.isin(y, FOUR)], y[m & np.isin(y, FOUR)])["accuracy"], held),
        "six_way": per_dataset(lambda m: M.probe_accuracy(bs[dev & six], y[dev & six], bs[m & six], y[m & six])["accuracy"], held),
    }
    report["crema_human_face_only"] = crema_human(meta)

    # ---- off-the-shelf expression model (HSEmotion; not trained on these corpora)
    fer_path = Path("results/m1/fer.jsonl")
    if fer_path.exists():
        acc = {}
        for line in open(fer_path):
            r = json.loads(line)
            if r["k"] in args.frames:
                acc.setdefault((r["dataset"], r["clip"]), []).append(r["scores"])
        got_f = np.array([c in acc for c in clips])
        f = np.array([np.mean(acc[c], axis=0) if c in acc else np.zeros(10) for c in clips])
        # outputs: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise, valence, arousal
        col = {"angry": 0, "disgusted": 2, "afraid": 3, "happy": 4, "neutral": 5, "sad": 6}
        pick = lambda labels, m: np.array(labels)[f[m][:, [col[l] for l in labels]].argmax(1)]
        report["fer"] = {
            "valence_auc": per_dataset(lambda m: M.valence_auc(f[m, 8], y[m]), held & got_f),
            "four_way": per_dataset(lambda m: float((pick(FOUR, m & np.isin(y, FOUR)) == y[m & np.isin(y, FOUR)]).mean()), held & got_f),
            "six_way": per_dataset(lambda m: float((pick(list(SIX), m & np.isin(y, SIX)) == y[m & np.isin(y, SIX)]).mean()), held & got_f),
        }

    # ---- choose settings on dev, report held-out
    def best(key, readout=None):
        cands = [c for c in report["configs"] if readout is None or c["readout"] == readout]
        return max(cands, key=lambda c: c["dev"][key]["mean"])

    report["selected"] = {}
    for name, key in (("valence", "valence_auc_baselined"), ("four_way", "four_way_baselined")):
        for readout in READOUTS:
            c = best(key, readout)
            report["selected"][f"{name}_{readout}"] = {
                "layout": c["layout"], "layer": c["layer"], "dev": c["dev"][key], "heldout": c["heldout"][key],
                "heldout_unbaselined": c["heldout"][key.replace("_baselined", "")],
                "heldout_valence_removed": c["heldout"].get("four_way_valence_removed") if name == "four_way" else None,
                "pc_eta2_emotion": c["heldout"]["pc_eta2_emotion"], "identity": c["heldout"]["identity"],
                "probe_six_way_on_scores": c["heldout"]["probe_six_way_on_scores"],
            }

    Path("results/m1").mkdir(parents=True, exist_ok=True)
    Path(f"results/m1/{args.run}_metrics.json").write_text(json.dumps(report, indent=1))

    # ---- print
    f3 = lambda d: "  ".join(f"{k} {v:.3f}" for k, v in d.items())
    print("\nclips:", report["counts"], "actors:", report["actors"])
    if "choices" in report:
        ch = report["choices"]
        print("\nPERCEPTION CEILING (zero-shot forced choice, held-out actors)")
        print("  own label set:", f3(ch["accuracy_own_labels"]["heldout"]), "| six shared labels:", f3(ch["accuracy_six"]["heldout"]))
        print("  among the four negatives:", f3(ch["four_way_among_negatives"]), "| unparsed:", round(ch["unparsed"], 4))
        print("  recall:", ch["recall_heldout"])
        print("  by intensity:", ch["by_intensity_heldout"])
    if report["crema_human_face_only"]:
        print("  CREMA-D human raters, face only:", report["crema_human_face_only"])
    print("\nVALENCE AUC, held-out (happy vs four negatives), by layout/readout/layer [per-person baseline]")
    for c in report["configs"]:
        print(f"  {c['layout']:11s} {c['readout']:5s} L{c['layer']}: {f3(c['heldout']['valence_auc_baselined'])}"
              f"   | no baseline: {c['heldout']['valence_auc']['mean']:.3f}"
              f" | 4-way {c['heldout']['four_way_baselined']['mean']:.3f} (no baseline {c['heldout']['four_way']['mean']:.3f},"
              f" valence removed {c['heldout']['four_way_valence_removed']['mean']:.3f})"
              f" | probe6 {c['heldout']['probe_six_way_on_scores']['accuracy']:.3f}")
    print("\nSELECTED ON DEV")
    for k, v in report["selected"].items():
        print(f"  {k}: {v['layout']} L{v['layer']} dev {v['dev']['mean']:.3f} -> held-out {f3(v['heldout'])} (no baseline {v['heldout_unbaselined']['mean']:.3f})")
        print(f"      eta2(emotion) by PC: {v['pc_eta2_emotion']}")
        idn = v["identity"]
        print(f"      valence variance: actor {idn['valence_eta2_actor']['mean']:.2f} emotion {idn['valence_eta2_emotion']['mean']:.2f}"
              f" -> baselined: actor {idn['valence_eta2_actor_baselined']['mean']:.2f} emotion {idn['valence_eta2_emotion_baselined']['mean']:.2f};"
              f" 171 scores: actor {idn['scores_eta2_actor']['mean']:.2f} emotion {idn['scores_eta2_emotion']['mean']:.2f}"
              f" -> baselined: actor {idn['scores_eta2_actor_baselined']['mean']:.2f} emotion {idn['scores_eta2_emotion_baselined']['mean']:.2f}")
    print("\nVERBAL BASELINE (next-word probabilities), held-out")
    for layout, v in verbal.items():
        print(f"  {layout}: valence AUC {f3(v['valence_auc'])} | 4-way {f3(v['four_way'])} | 5-way {f3(v['five_way'])}")
    b = report["blendshapes"]
    print("\nBLENDSHAPE BASELINE (trained on dev actors), held-out")
    print(f"  valence AUC {f3(b['valence_auc'])} | 4-way {f3(b['four_way'])} | 6-way {f3(b['six_way'])}")
    if "fer" in report:
        b = report["fer"]
        print("\nOFF-THE-SHELF EXPRESSION MODEL (HSEmotion, no training on these corpora), held-out")
        print(f"  valence AUC {f3(b['valence_auc'])} | 4-way {f3(b['four_way'])} | 6-way {f3(b['six_way'])}")


if __name__ == "__main__":
    main()
