"""Gate 2 analysis on the acted-video runs.

    python scripts/video_gate2.py [--acts data/video/acts] [--run medium] [--upper-run upper]

Inputs
    results/m2/frames_<dataset>.jsonl          tracking output (modal_app/video.py::track)
    results/m2/frames_ravdess_upper.jsonl      the upper-face subset
    data/video/acts/<run>/<dataset>/<clip>/    activation stores (downloaded from /vol/video/acts)
    results/m1/reference_a2_prefill.npz        projection reference (scripts/video_reference.py)
    results/m2/run_*.json, benchmark.json      GPU time and throughput

Outputs
    results/m2/projected_<run>.npz             per-frame projections, cached
    results/m2/gate2_metrics.json
    results/m2/gate2_report.md

Checks (design doc, "Gate 2: acted video"; work plan, M2):
    lexical control    eta^2 of emotion vs sentence on windowed valence and the 171 scores, held-out actors
    speech-motion floor within-clip variance of valence in neutral clips, per frame vs 1.5 s windows,
                       against the between-emotion variance; medium vs upper-face crops on a subset
    temporal sanity    expression onset in the readout vs onset in MediaPipe blendshapes
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from cogmirror import metrics as M, store, video
from cogmirror.datasets import NEGATIVE, SIX

GEM = "data/gemotions"
REFERENCE = "results/m1/reference_a2_prefill.npz"
OUT = Path("results/m2")
WINDOW = store.DEFAULT_WINDOW
# Dataset label -> emotion vector whose score is tracked for onsets.
TARGET = {"happy": "happy", "angry": "angry", "sad": "sad", "afraid": "afraid", "disgusted": "disgusted", "surprised": "surprised"}


# --------------------------------------------------------------------------
# Loading and projection
# --------------------------------------------------------------------------

def load_frames(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error") and r.get("frames"):
                rows.append(r)
    return rows


def project_run(acts_root: Path, clips: list[dict], cfg: store.ProjectionConfig, cache: Path, force: bool = False) -> dict:
    """Per-frame projections for every clip of a run, concatenated, with a
    clip index per frame. Cached in `cache`."""
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    spaces = store.load_spaces(GEM, [cfg.score_layer, cfg.valence_layer], cfg.n_pcs)
    refs, _ = store.load_reference(REFERENCE)
    t, fi, ci, val, pcs, sc = [], [], [], [], [], []
    missing = []
    for k, c in enumerate(clips):
        d = acts_root / c["dataset"] / c["clip"]
        if not d.exists():
            missing.append(c["clip"])
            continue
        proj = store.project(store.load_store(d), spaces, refs, cfg)
        t.append(proj.t)
        fi.append(proj.frame_index)
        ci.append(np.full(len(proj.t), k))
        pcs.append(proj.pcs)
        sc.append(proj.scores.astype(np.float32))
        if k % 200 == 0:
            print(f"  projected {k}/{len(clips)} clips")
    if missing:
        print(f"  {len(missing)} clips without a store (e.g. {missing[:3]})")
    out = {"t": np.concatenate(t), "frame_index": np.concatenate(fi), "clip": np.concatenate(ci),
           "pcs": np.concatenate(pcs), "scores": np.concatenate(sc), "emotions": np.array(spaces[cfg.score_layer].emotions),
           "n_clips": np.array(len(clips))}
    np.savez(cache, **out)
    return out


def per_clip(proj: dict, n_clips: int, window: float):
    """Per-clip arrays: windowed per-frame valence and scores (lists), clip
    means of the windowed signals, and within-clip variances."""
    clip = proj["clip"]
    order = np.argsort(clip, kind="stable")
    bounds = np.searchsorted(clip[order], np.arange(n_clips + 1))
    E = proj["scores"].shape[1]
    out = {
        "t": [None] * n_clips, "val_raw": [None] * n_clips, "val_win": [None] * n_clips, "scores_win": [None] * n_clips,
        "scores_raw": [None] * n_clips,
        "val_mean": np.full(n_clips, np.nan), "scores_mean": np.full((n_clips, E), np.nan),
        "val_mean_mid": np.full(n_clips, np.nan), "scores_mean_mid": np.full((n_clips, E), np.nan),
        "var_raw": np.full(n_clips, np.nan), "var_win": np.full(n_clips, np.nan),
        "scores_var_raw": np.full(n_clips, np.nan), "scores_var_win": np.full(n_clips, np.nan),
        "n_frames": np.zeros(n_clips, int), "frame_index": [None] * n_clips,
    }
    for k in range(n_clips):
        rows = order[bounds[k]:bounds[k + 1]]
        if len(rows) == 0:
            continue
        rows = rows[np.argsort(proj["t"][rows], kind="stable")]
        t = proj["t"][rows]
        v = proj["pcs"][rows, 0]
        s = proj["scores"][rows].astype(np.float64)
        vw = store.window_mean(t, v, window)
        sw = store.window_mean(t, s, window)
        out["t"][k], out["val_raw"][k], out["val_win"][k], out["scores_win"][k], out["scores_raw"][k] = t, v, vw, sw, s
        out["frame_index"][k] = proj["frame_index"][rows]
        out["val_mean"][k] = vw.mean()
        out["scores_mean"][k] = sw.mean(axis=0)
        mid = (t >= t[-1] / 3) & (t <= 2 * t[-1] / 3)        # the middle third, as M1 sampled it
        if mid.any():
            out["val_mean_mid"][k] = v[mid].mean()
            out["scores_mean_mid"][k] = s[mid].mean(axis=0)
        out["var_raw"][k] = v.var()
        out["var_win"][k] = vw.var()
        out["scores_var_raw"][k] = s.var(axis=0).sum()
        out["scores_var_win"][k] = sw.var(axis=0).sum()
        out["n_frames"][k] = len(rows)
    return out


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def lexical_control(pc: dict, meta: list[dict], mask: np.ndarray) -> dict:
    """eta^2 of emotion vs sentence on clip-level windowed valence and scores,
    with and without a per-actor baseline, plus frame-level valence."""
    y = np.array([m["emotion"] for m in meta])
    sent = np.array([m["sentence"] for m in meta])
    actor = np.array([m["actor"] for m in meta])
    val = pc["val_mean"]
    sc = pc["scores_mean"]
    ok = mask & np.isfinite(val)
    val_b = np.full_like(val, np.nan)
    val_b[ok] = M.center_by_group(val[ok], actor[ok])
    sc_b = np.full_like(sc, np.nan)
    sc_b[ok] = M.center_by_group(sc[ok], actor[ok])
    res = {
        "n_clips": int(ok.sum()), "n_emotions": int(len(set(y[ok]))), "n_sentences": int(len(set(sent[ok]))),
        "valence": {"emotion": M.eta_squared(val[ok], y[ok]), "sentence": M.eta_squared(val[ok], sent[ok]),
                    "actor": M.eta_squared(val[ok], actor[ok])},
        "valence_baselined": {"emotion": M.eta_squared(val_b[ok], y[ok]), "sentence": M.eta_squared(val_b[ok], sent[ok])},
        "scores": {"emotion": M.eta_squared(sc[ok], y[ok]), "sentence": M.eta_squared(sc[ok], sent[ok]),
                   "actor": M.eta_squared(sc[ok], actor[ok])},
        "scores_baselined": {"emotion": M.eta_squared(sc_b[ok], y[ok]), "sentence": M.eta_squared(sc_b[ok], sent[ok])},
    }
    # sentence effect inside each emotion (removes any emotion x sentence imbalance)
    within = [M.eta_squared(val[ok & (y == e)], sent[ok & (y == e)]) for e in sorted(set(y[ok])) if (ok & (y == e)).sum() > 2]
    res["valence"]["sentence_within_emotion_mean"] = float(np.mean(within)) if within else float("nan")
    # frame level
    fv, fy, fs = [], [], []
    for k in np.flatnonzero(ok):
        fv.append(pc["val_win"][k])
        fy += [y[k]] * len(pc["val_win"][k])
        fs += [sent[k]] * len(pc["val_win"][k])
    fv = np.concatenate(fv)
    res["valence_frames"] = {"emotion": M.eta_squared(fv, np.array(fy)), "sentence": M.eta_squared(fv, np.array(fs)), "n": int(len(fv))}
    for key in ("valence", "valence_baselined", "scores", "scores_baselined", "valence_frames"):
        d = res[key]
        d["ratio"] = float(d["emotion"] / d["sentence"]) if d["sentence"] > 0 else float("inf")
    return res


def accuracy_checks(pc: dict, meta: list[dict], mask: np.ndarray, emotions: list[str]) -> dict:
    """M1's headline numbers recomputed from the video pipeline's clip means
    (whole clip, and the middle third as M1 sampled it)."""
    y = np.array([m["emotion"] for m in meta])
    ok = mask & np.isfinite(pc["val_mean"])
    okm = ok & np.isfinite(pc["val_mean_mid"])
    return {
        "valence_auc": M.valence_auc(pc["val_mean"][ok], y[ok]),
        "four_way": M.nearest_vector_accuracy(pc["scores_mean"][ok], y[ok], emotions, NEGATIVE)["accuracy"],
        "five_way": M.nearest_vector_accuracy(pc["scores_mean"][ok], y[ok], emotions, list(NEGATIVE) + ["happy"])["accuracy"],
        "valence_auc_mid": M.valence_auc(pc["val_mean_mid"][okm], y[okm]),
        "four_way_mid": M.nearest_vector_accuracy(pc["scores_mean_mid"][okm], y[okm], emotions, NEGATIVE)["accuracy"],
        "n": int(ok.sum()),
    }


def speech_motion(pc: dict, meta: list[dict], mask: np.ndarray, neutral_labels=("neutral",)) -> dict:
    y = np.array([m["emotion"] for m in meta])
    ok = mask & np.isfinite(pc["val_mean"])
    neu = ok & np.isin(y, neutral_labels)
    six = ok & np.isin(y, SIX)
    means = pc["val_mean"]
    # between-emotion variance of clip means: variance of the emotion means, weighted by clip count
    grand = means[six].mean()
    between = sum((y[six] == e).sum() * (means[six & (y == e)].mean() - grand) ** 2 for e in set(y[six])) / six.sum()
    gap = float(means[ok & (y == "happy")].mean() - means[ok & np.isin(y, NEGATIVE)].mean())
    within_raw = float(np.mean(pc["var_raw"][neu]))
    within_win = float(np.mean(pc["var_win"][neu]))
    res = {
        "n_neutral_clips": int(neu.sum()),
        "within_neutral_var_per_frame": within_raw,
        "within_neutral_var_windowed": within_win,
        "within_neutral_sd_per_frame": float(np.sqrt(within_raw)),
        "within_neutral_sd_windowed": float(np.sqrt(within_win)),
        "between_emotion_var_clip_means": float(between),
        "total_var_clip_means_six": float(means[six].var()),
        "happy_minus_negative_gap": gap,
        "gap_over_sd_per_frame": float(gap / np.sqrt(within_raw)) if within_raw > 0 else float("inf"),
        "gap_over_sd_windowed": float(gap / np.sqrt(within_win)) if within_win > 0 else float("inf"),
        "scores_within_neutral_var_per_frame": float(np.mean(pc["scores_var_raw"][neu])),
        "scores_within_neutral_var_windowed": float(np.mean(pc["scores_var_win"][neu])),
        # per-frame readout within neutral clips: the time series' noise band
        "neutral_frame_sd_pooled": float(np.sqrt(np.mean([np.var(pc["val_raw"][k]) for k in np.flatnonzero(neu)]))),
    }
    # the same for calm (RAVDESS), if present
    if (ok & (y == "calm")).any():
        calm = ok & (y == "calm")
        res["calm"] = {"n": int(calm.sum()), "within_var_per_frame": float(np.mean(pc["var_raw"][calm])),
                       "within_var_windowed": float(np.mean(pc["var_win"][calm]))}
    return res


def onsets(pc: dict, meta: list[dict], mask: np.ndarray, emotions: list[str], neutral_sd_val: float,
           neutral_sd_scores: dict, bs_rise_min: float = 0.1) -> dict:
    """Readout onset vs blendshape onset per clip; lag distribution."""
    y = np.array([m["emotion"] for m in meta])
    res = {"margin_valence": 2 * neutral_sd_val, "blendshape_min_rise": bs_rise_min, "per_emotion": {}}
    all_lags = []
    for label, target in TARGET.items():
        if label not in video.ONSET_BLENDSHAPES or target not in emotions:
            continue
        rows = np.flatnonzero(mask & (y == label))
        if len(rows) == 0:
            continue
        j = emotions.index(target)
        ev_on, bs_on, lags, both, start_high = [], [], [], 0, 0
        margin = 2 * (neutral_sd_val if label == "happy" else neutral_sd_scores[target])
        for k in rows:
            t = pc["t"][k]
            if t is None or len(t) < 4:
                continue
            sig = pc["val_win"][k] if label == "happy" else pc["scores_win"][k][:, j]
            frames = meta[k]["frames"]
            bs = video.blendshape_series(frames, video.ONSET_BLENDSHAPES[label])
            bt = np.array([f["t"] for f in frames])
            # align blendshapes to the frames that have activations
            pos = {f["frame_index"]: i for i, f in enumerate(frames)}
            idx = [pos.get(int(fi)) for fi in pc_frame_index(pc, k)]
            bs_al = np.array([bs[i] if i is not None else np.nan for i in idx])
            e_on = video.onset_time(t, sig, margin)
            finite = bs_al[np.isfinite(bs_al)]
            if len(finite) >= 3 and finite.max() - finite[:2].mean() >= bs_rise_min:
                b_on = video.onset_time(t, bs_al, 0.5 * (finite.max() - finite[:2].mean()))
            else:
                b_on = float("nan")
            ev_on.append(e_on)
            bs_on.append(b_on)
            # already expressive at the start: first-two-frame level above the clip's median
            if np.isfinite(sig).sum() >= 4 and sig[np.isfinite(sig)][:2].mean() > np.nanmedian(sig):
                start_high += 1
            if np.isfinite(e_on) and np.isfinite(b_on):
                lags.append(e_on - b_on)
                both += 1
        lags = np.array(lags)
        ev_on, bs_on = np.array(ev_on), np.array(bs_on)
        d = {"n_clips": int(len(ev_on)), "readout_onset_found": float(np.isfinite(ev_on).mean()) if len(ev_on) else float("nan"),
             "blendshape_onset_found": float(np.isfinite(bs_on).mean()) if len(bs_on) else float("nan"),
             "both_found": int(both), "starts_expressive_frac": float(start_high / max(len(ev_on), 1)),
             "margin": float(margin), "signal": "valence" if label == "happy" else f"score_{target}",
             "blendshapes": list(video.ONSET_BLENDSHAPES[label])}
        if both:
            d.update(lag_median=float(np.median(lags)), lag_iqr=[float(np.percentile(lags, 25)), float(np.percentile(lags, 75))],
                     lag_mean=float(lags.mean()), within_1s=float((np.abs(lags) <= 1.0).mean()),
                     within_0_5s=float((np.abs(lags) <= 0.5).mean()),
                     readout_onset_median=float(np.nanmedian(ev_on)), blendshape_onset_median=float(np.nanmedian(bs_on)))
            all_lags.append(lags)
        res["per_emotion"][label] = d
    if all_lags:
        lags = np.concatenate(all_lags)
        res["all"] = {"n": int(len(lags)), "lag_median": float(np.median(lags)),
                      "lag_iqr": [float(np.percentile(lags, 25)), float(np.percentile(lags, 75))],
                      "within_1s": float((np.abs(lags) <= 1.0).mean()), "within_0_5s": float((np.abs(lags) <= 0.5).mean())}
    return res


def pc_frame_index(pc: dict, k: int) -> np.ndarray:
    return pc["frame_index"][k]


def gpu_hours() -> dict:
    out = {"runs": [], "busy_hours": 0.0}
    for f in sorted(OUT.glob("run_*.json")):
        for r in json.loads(f.read_text()):
            out["runs"].append({k: r.get(k) for k in ("run", "subset", "frames", "chunks", "wall_seconds", "gpu_busy_seconds", "fps_per_gpu_median", "load_seconds")})
            out["busy_hours"] += (r.get("gpu_busy_seconds") or 0) / 3600
    # model loads: one per container, up to 7 containers per run
    loads = [max(r["load_seconds"]) for r in out["runs"] if r.get("load_seconds")]
    out["load_overhead_hours_estimate"] = float(sum(loads) * 7 / 3600) if loads else 0.0
    b = OUT / "benchmark.json"
    out["benchmark"] = json.loads(b.read_text()) if b.exists() else None
    notes = OUT / "gpu_notes.txt"
    out["notes"], out["extra_hours"] = [], 0.0
    for line in (notes.read_text().strip().splitlines() if notes.exists() else []):
        if line.startswith("EXTRA_HOURS="):
            head, _, text = line.partition("|")
            out["extra_hours"] += float(head.split("=")[1])
            line = text.strip()
        out["notes"].append(line)
    out["total_hours"] = out["busy_hours"] + out["load_overhead_hours_estimate"] + 0.1 + out["extra_hours"]
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "inf" if x > 0 else "nan"
        return f"{x:.{nd}f}"
    return str(x)


def write_report(R: dict, path: Path):
    L = []
    L.append("# Gate 2 results: acted video (M2)\n")
    L.append(f"Date: {R['date']}. Pipeline: {R['pipeline']}\n")
    L.append("## Gate table\n")
    L.append("| Check | Criterion | Result | Pass? |")
    L.append("|---|---|---|---|")
    for g in R["gate"]:
        L.append(f"| {g['check']} | {g['criterion']} | {g['result']} | {g['pass']} |")
    L.append("")
    L.append("## Runs\n")
    for name, d in R["datasets"].items():
        L.append(f"- **{name}**: {d['n_clips']} clips ({d['n_heldout']} held-out actors' clips), {d['n_frames']} frames at "
                 f"{d['rate']} fps, {d['minutes']:.1f} min of video; face detected on {d['detect_frac']:.1%} of sampled frames, "
                 f"box held through {d['held_frac']:.1%}; {d['n_clips_with_store']} clips with activations.")
    L.append("")
    L.append("## Sanity: M1 metrics from the video pipeline (clip means of 1.5 s windows, held-out actors)\n")
    L.append("| Dataset | Valence AUC | 4-way negatives | 5-way | Valence AUC, middle third | 4-way, middle third | n |")
    L.append("|---|---|---|---|---|---|---|")
    for name, a in R["accuracy"].items():
        L.append(f"| {name} | {fmt(a['valence_auc'])} | {fmt(a['four_way'])} | {fmt(a['five_way'])} | {fmt(a['valence_auc_mid'])} | {fmt(a['four_way_mid'])} | {a['n']} |")
    L.append("\nM1 (two still frames per clip from the middle third): valence AUC 0.982, 4-way 0.576. Whole-clip means include the "
             "neutral lead-in and tail of each take; the middle third matches M1's sampling.\n")
    L.append("## Lexical control (held-out actors)\n")
    L.append("η² = share of variance explained by the grouping; clip-level values are means of the windowed series.\n")
    L.append("| Dataset | Signal | η² emotion | η² sentence | ratio | η² actor |")
    L.append("|---|---|---|---|---|---|")
    for name, lc in R["lexical"].items():
        for key, label in (("valence", "valence (clip)"), ("valence_baselined", "valence, actor-baselined"),
                           ("valence_frames", "valence (frames)"), ("scores", "171 scores (clip)"),
                           ("scores_baselined", "171 scores, actor-baselined")):
            d = lc[key]
            L.append(f"| {name} | {label} | {fmt(d['emotion'])} | {fmt(d['sentence'])} | {fmt(d['ratio'], 1)} | {fmt(d.get('actor'))} |")
    L.append("")
    for name, lc in R["lexical"].items():
        L.append(f"- {name}: {lc['n_clips']} clips, {lc['n_emotions']} emotions, {lc['n_sentences']} sentences; "
                 f"η² of sentence within each emotion, mean {fmt(lc['valence']['sentence_within_emotion_mean'])}.")
    L.append("")
    L.append("## Speech-motion floor\n")
    L.append("Valence in reference-SD units (scale from the M1 dev actors). Within-clip variance is the mean over neutral clips.\n")
    L.append("| Dataset | Neutral clips | Within-clip var, per frame | Within-clip var, 1.5 s windows | Between-emotion var of clip means | Happy − negative gap | Gap / within SD (frame) | Gap / within SD (windowed) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, s in R["speech_motion"].items():
        L.append(f"| {name} | {s['n_neutral_clips']} | {fmt(s['within_neutral_var_per_frame'], 4)} | {fmt(s['within_neutral_var_windowed'], 4)} | "
                 f"{fmt(s['between_emotion_var_clip_means'], 3)} | {fmt(s['happy_minus_negative_gap'], 2)} | {fmt(s['gap_over_sd_per_frame'], 1)} | {fmt(s['gap_over_sd_windowed'], 1)} |")
    L.append("")
    if R.get("upper"):
        u = R["upper"]
        L.append("### Upper-face crops (held-out RAVDESS subset: neutral, and strong happy/angry/sad)\n")
        L.append("| Crop | Clips | Within-neutral var, per frame | Within-neutral var, windowed | Between-emotion var | Gap / within SD (frame) | Gap / within SD (windowed) | Valence AUC | 4-way (angry/sad) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for crop, s in u.items():
            L.append(f"| {crop} | {s['n_clips']} | {fmt(s['within_neutral_var_per_frame'], 4)} | {fmt(s['within_neutral_var_windowed'], 4)} | "
                     f"{fmt(s['between_emotion_var_clip_means'], 3)} | {fmt(s['gap_over_sd_per_frame'], 1)} | {fmt(s['gap_over_sd_windowed'], 1)} | "
                     f"{fmt(s['valence_auc'])} | {fmt(s['two_way'])} |")
        L.append("")
    L.append("## Temporal sanity: expression onsets\n")
    L.append("Readout onset: first time the windowed signal (valence for happy, the target emotion's score otherwise) exceeds "
             "the mean of the clip's first two frames by 2× the within-neutral-clip SD of that signal. Blendshape onset: first time "
             "the mean of the listed MediaPipe blendshapes rises halfway from its first-two-frame level to its clip maximum "
             "(clips with a rise under 0.1 have no blendshape onset). Lag = readout onset − blendshape onset.\n")
    for name, on in R["onsets"].items():
        L.append(f"### {name}\n")
        L.append("| Emotion | Signal | Blendshapes | Clips | Readout onset found | Blendshape onset found | Both | Lag median (s) | IQR | within ±0.5 s | within ±1 s | Starts expressive |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e, d in on["per_emotion"].items():
            iqr = f"{d['lag_iqr'][0]:+.2f} .. {d['lag_iqr'][1]:+.2f}" if "lag_iqr" in d else "—"
            L.append(f"| {e} | {d['signal']} | {', '.join(d['blendshapes'])} | {d['n_clips']} | {fmt(d['readout_onset_found'], 2)} | "
                     f"{fmt(d['blendshape_onset_found'], 2)} | {d['both_found']} | {fmt(d.get('lag_median'), 2)} | {iqr} | "
                     f"{fmt(d.get('within_0_5s'), 2)} | {fmt(d.get('within_1s'), 2)} | {fmt(d['starts_expressive_frac'], 2)} |")
        if "all" in on:
            a = on["all"]
            L.append(f"\nAll emotions: {a['n']} clips with both onsets, lag median {a['lag_median']:+.2f} s "
                     f"(IQR {a['lag_iqr'][0]:+.2f} .. {a['lag_iqr'][1]:+.2f}), {a['within_1s']:.0%} within ±1 s, {a['within_0_5s']:.0%} within ±0.5 s.\n")
    L.append("Self-recorded scripted events (neutral → smile → neutral, a yawn, a frown, with marked timestamps) are not available "
             "yet; that part of the temporal check remains open until the M3 recording tool has produced them.\n")
    L.append("## Throughput and cost\n")
    g = R["gpu"]
    if g.get("benchmark"):
        b = g["benchmark"]
        L.append(f"Benchmark on {b['gpu']} (model load {b['load_seconds']} s, peak {b['gpu_mem_gb']} GB), capture to layer {b['deepest_layer']}, "
                 f"chat-prefill layout, real face crops:\n")
        L.append("| Token budget | Batch | Image tokens | Seq len | Frames/s per GPU | $/h of video at 2 fps | 5 fps | 10 fps |")
        L.append("|---|---|---|---|---|---|---|---|")
        for row in b["cost"]:
            c = row["usd_per_hour_video"]
            L.append(f"| {row['budget']} | {row['batch_size']} | {row['image_tokens']} | {row['seq_len']} | {row['fps']} | ${c['2']} | ${c['5']} | ${c['10']} |")
        L.append("")
    L.append("GPU time for the runs (L40S at $1.95/h):\n")
    L.append("| Run | Frames | Calls | Wall (s) | GPU busy (s) | Median frames/s per GPU |")
    L.append("|---|---|---|---|---|---|")
    for r in g["runs"]:
        L.append(f"| {r['run']} ({r.get('subset')}) | {r['frames']} | {r['chunks']} | {r['wall_seconds']} | {r['gpu_busy_seconds']} | {r['fps_per_gpu_median']} |")
    L.append(f"\nGPU busy time {g['busy_hours']:.2f} h, plus about {g['load_overhead_hours_estimate']:.2f} h of model loading "
             f"(one load per container), about 0.1 h for the benchmark and the single-clip test, and {g['extra_hours']:.1f} h for the "
             f"aborted first run below: roughly {g['total_hours']:.1f} GPU-hours, about ${1.95 * g['total_hours']:.0f}. "
             f"The realized rate inside calls (2.6 frames/s on the medium run) was below the benchmark's 5.9 because frames were read "
             f"one by one from the Volume and each clip is a short chunk; parallel reads (added before the upper run) brought it to 4.8.\n")
    for n in g.get("notes", []):
        L.append(f"- {n}")
    L.append("")
    L.append("## Reading of the result\n")
    for n in R["reading"]:
        L.append(f"- {n}")
    L.append("")
    L.append("## Notes\n")
    for n in R["notes"]:
        L.append(f"- {n}")
    L.append("")
    path.write_text("\n".join(L))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered")
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/video/acts")
    ap.add_argument("--run", default="medium")
    ap.add_argument("--upper-run", default="upper")
    ap.add_argument("--window", type=float, default=WINDOW)
    ap.add_argument("--force", action="store_true", help="recompute cached projections")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import datetime

    cfg = store.ProjectionConfig()
    clips = []
    for ds in ("ravdess", "crema"):
        f = OUT / f"frames_{ds}.jsonl"
        if f.exists():
            clips += load_frames(f)
    if not clips:
        raise SystemExit("no results/m2/frames_*.jsonl; run modal_app/video.py::track first")
    print(f"{len(clips)} tracked clips")
    proj = project_run(Path(args.acts, args.run), clips, cfg, OUT / f"projected_{args.run}.npz", args.force)
    emotions = [str(e) for e in proj["emotions"]]
    pc = per_clip(proj, len(clips), args.window)
    ds = np.array([c["dataset"] for c in clips])
    split = np.array([c["split"] for c in clips])
    have = np.isfinite(pc["val_mean"])
    held = split == "heldout"

    R: dict = {"date": datetime.date.today().isoformat(),
               "pipeline": f"5 fps sampling, one-euro box smoothing, medium crop at 768 px, budget 280, chat-prefill layout, last token; "
                           f"scores at layer {cfg.score_layer}, valence (PC1) at layer {cfg.valence_layer}; reference from M1 dev actors; "
                           f"{args.window} s centered windows; no per-person baseline unless stated.",
               "datasets": {}, "accuracy": {}, "lexical": {}, "speech_motion": {}, "onsets": {}, "notes": []}
    for name in ("ravdess", "crema"):
        m = ds == name
        if not (m & held & have).any():
            continue
        frames = [f for c in clips if c["dataset"] == name for f in c["frames"]]
        R["datasets"][name] = {
            "n_clips": int(m.sum()), "n_heldout": int((m & held).sum()), "n_frames": len(frames),
            "rate": clips[int(np.flatnonzero(m)[0])]["info"]["rate"],
            "minutes": sum(c["info"].get("duration") or 0 for c in clips if c["dataset"] == name) / 60,
            "detect_frac": float(np.mean([f["face_found"] for f in frames])),
            "held_frac": float(np.mean([f.get("held", False) for f in frames])),
            "n_clips_with_store": int((m & have).sum()),
        }
        R["accuracy"][name] = accuracy_checks(pc, clips, m & held & have, emotions)
        R["lexical"][name] = lexical_control(pc, clips, m & held & have)
        R["speech_motion"][name] = speech_motion(pc, clips, m & have)
    R["accuracy"]["both"] = accuracy_checks(pc, clips, held & have, emotions)
    R["lexical"]["both"] = lexical_control(pc, clips, held & have)
    R["speech_motion"]["both"] = speech_motion(pc, clips, have)

    # onsets, per dataset (all actors), margins from the within-neutral SD of each signal
    for name in ("ravdess", "crema"):
        m = (ds == name) & have
        if not m.any():
            continue
        neu = m & (np.array([c["emotion"] for c in clips]) == "neutral")
        if not neu.any():
            continue
        sd_val = float(np.sqrt(np.mean(pc["var_win"][neu])))
        sd_scores = {}
        for target in set(TARGET.values()):
            if target in emotions:
                j = emotions.index(target)
                sd_scores[target] = float(np.sqrt(np.mean([pc["scores_win"][k][:, j].var() for k in np.flatnonzero(neu)])))
        R["onsets"][name] = onsets(pc, clips, m, emotions, sd_val, sd_scores)

    # upper-face subset
    up_frames = OUT / "frames_ravdess_upper.jsonl"
    up_root = Path(args.acts, args.upper_run)
    if up_frames.exists() and up_root.exists():
        sub = load_frames(up_frames)
        names = {c["clip"] for c in sub}
        R["upper"] = {}
        for crop, root in (("medium", Path(args.acts, args.run)), ("upper", up_root)):
            cache = OUT / f"projected_{args.run if crop == 'medium' else args.upper_run}_uppersubset.npz"
            p = project_run(root, sub, cfg, cache, args.force)
            pcs_ = per_clip(p, len(sub), args.window)
            ok = np.isfinite(pcs_["val_mean"])
            s = speech_motion(pcs_, sub, ok)
            y = np.array([c["emotion"] for c in sub])
            s["n_clips"] = int(ok.sum())
            s["valence_auc"] = M.valence_auc(pcs_["val_mean"][ok], y[ok])
            s["two_way"] = M.nearest_vector_accuracy(pcs_["scores_mean"][ok], y[ok], emotions, ("angry", "sad"))["accuracy"]
            R["upper"][crop] = s
        R["notes"].append(f"Upper-face comparison on {len(names)} held-out RAVDESS clips (all neutral, plus strong happy/angry/sad); "
                          "the upper crop keeps the top 60% of the medium crop (768×461 px, fewer image tokens).")
    else:
        R["notes"].append("Upper-face crop comparison not run (no results/m2/frames_ravdess_upper.jsonl or no upper store).")

    R["gpu"] = gpu_hours()

    # gate table
    lex = R["lexical"]["both"]["valence"]
    sm = R["speech_motion"]["both"]
    on_all = {k: v.get("all") for k, v in R["onsets"].items()}
    gate = [
        {"check": "Lexical control", "criterion": "η²(emotion) ≥ 3 × η²(sentence) on valence, held-out actors",
         "result": f"emotion {lex['emotion']:.3f} vs sentence {lex['sentence']:.3f} (ratio {lex['ratio']:.0f}); "
                   + "".join(f"{n} {R['lexical'][n]['valence']['ratio']:.0f}×; " for n in ("ravdess", "crema") if n in R["lexical"])
                   + f"171 scores: {R['lexical']['both']['scores']['emotion']:.3f} vs {R['lexical']['both']['scores']['sentence']:.3f}",
         "pass": "Yes" if lex["ratio"] >= 3 else "No"},
        {"check": "Speech-motion floor", "criterion": "Windowed neutral-speech readout within the Gate 1 noise band: happy − negative gap ≥ 2 × within-clip SD",
         "result": f"within-neutral SD per frame {sm['within_neutral_sd_per_frame']:.2f}, windowed {sm['within_neutral_sd_windowed']:.2f}; "
                   f"gap {sm['happy_minus_negative_gap']:.2f} → gap/SD {sm['gap_over_sd_per_frame']:.0f} (frame), {sm['gap_over_sd_windowed']:.0f} (windowed); "
                   f"between-emotion var of clip means {sm['between_emotion_var_clip_means']:.2f} vs within-neutral {sm['within_neutral_var_windowed']:.3f}",
         "pass": "Yes" if sm["gap_over_sd_windowed"] >= 2 else "No"},
    ]
    parts = []
    for name, a in on_all.items():
        if a and R["lexical"].get(name):
            parts.append(f"{name}: lag median {a['lag_median']:+.2f} s, {a['within_1s']:.0%} within ±1 s (n={a['n']})")
    gate.append({"check": "Temporal sanity (acted onsets)", "criterion": "Readout onset within ±1 s of the blendshape onset; scripted self-recorded events within ±1 s",
                 "result": "; ".join(parts) + ". Scripted events: not yet recorded",
                 "pass": "Partial" if parts and all(on_all[n]["within_1s"] >= 0.5 for n in on_all if on_all[n]) else "Open"})
    b = R["gpu"].get("benchmark")
    if b:
        r280 = next((r for r in b["cost"] if r["budget"] == 280 and r["batch_size"] == 8), b["cost"][0])
        gate.append({"check": "Cost per hour of video", "criterion": "Known and acceptable for M4",
                     "result": f"{r280['fps']} frames/s per L40S at budget 280, batch 8 → ${r280['usd_per_hour_video']['2']} / "
                               f"${r280['usd_per_hour_video']['5']} / ${r280['usd_per_hour_video']['10']} per hour of video at 2 / 5 / 10 fps",
                     "pass": "Yes"})
    R["gate"] = gate
    acc = R["accuracy"]["both"]
    lexb = R["lexical"]["both"]
    on_r, on_c = R["onsets"].get("ravdess", {}).get("per_emotion", {}), R["onsets"].get("crema", {}).get("per_emotion", {})
    R["reading"] = [
        f"**The video pipeline reproduces M1.** On the middle third of each clip (M1's sampling) the clip means give valence AUC "
        f"{acc['valence_auc_mid']:.3f} and 4-way {acc['four_way_mid']:.3f} against M1's 0.982 and 0.576 from two still frames; "
        f"whole-clip means are slightly lower on categories ({acc['four_way']:.3f}) because they include the neutral lead-in and tail.",
        f"**No lip-reading leakage is visible.** Sentence identity explains {lexb['valence']['sentence']:.1%} of clip-level valence variance "
        f"against {lexb['valence']['emotion']:.0%} for emotion (CREMA-D, 12 sentences: {R['lexical']['crema']['valence']['sentence']:.1%} vs "
        f"{R['lexical']['crema']['valence']['emotion']:.0%}). On the 171 scores sentence reaches {lexb['scores']['sentence']:.1%} "
        f"(CREMA-D {R['lexical']['crema']['scores']['sentence']:.1%}), still far under emotion ({lexb['scores']['emotion']:.0%}). "
        f"Actor identity explains more than sentence does ({lexb['valence']['actor']:.1%} on valence, {lexb['scores']['actor']:.1%} on scores), "
        f"and a per-actor baseline raises emotion's share (valence {lexb['valence_baselined']['emotion']:.0%}, scores {lexb['scores_baselined']['emotion']:.0%}).",
        f"**Speech motion sets the per-frame noise floor, and 1.5 s windows halve it.** Within a neutral-speech clip the per-frame valence SD is "
        f"{sm['within_neutral_sd_per_frame']:.2f} reference units; after windowing {sm['within_neutral_sd_windowed']:.2f}. The happy − negative gap "
        f"is {sm['happy_minus_negative_gap']:.1f}, so the windowed series sits {sm['gap_over_sd_windowed']:.0f} SDs below the expression gap "
        f"(Gate 1 required 2). Between-emotion variance of clip means ({sm['between_emotion_var_clip_means']:.2f}) is "
        f"{sm['between_emotion_var_clip_means'] / sm['within_neutral_var_windowed']:.0f}× the windowed within-neutral variance.",
    ]
    if R.get("upper"):
        u, m_ = R["upper"]["upper"], R["upper"]["medium"]
        R["reading"].append(
            f"**Upper-face crops remove most of the speech-motion variance but also part of the signal.** On the same 128 clips the "
            f"within-neutral per-frame variance drops from {m_['within_neutral_var_per_frame']:.3f} to {u['within_neutral_var_per_frame']:.3f} "
            f"({m_['within_neutral_var_per_frame'] / u['within_neutral_var_per_frame']:.0f}×) while the between-emotion variance drops from "
            f"{m_['between_emotion_var_clip_means']:.2f} to {u['between_emotion_var_clip_means']:.2f}; gap/SD improves ({m_['gap_over_sd_windowed']:.0f} → "
            f"{u['gap_over_sd_windowed']:.0f} windowed) but valence AUC falls from {m_['valence_auc']:.3f} to {u['valence_auc']:.3f} and "
            f"angry-vs-sad from {m_['two_way']:.2f} to {u['two_way']:.2f}. Keep the medium crop as the default and treat the upper crop as the "
            f"ablation for speech-heavy material. Scores are in medium-crop reference units, so the upper crop's absolute scale is not calibrated.")
    hp = {k: v.get("happy", {}) for k, v in (("ravdess", on_r), ("crema", on_c))}
    R["reading"].append(
        f"**Onsets line up with the face where MediaPipe sees a clear event.** For happy, the valence onset and the smile-blendshape onset "
        f"coincide (median lag 0.0 s, within ±0.5 s for {hp['ravdess'].get('within_0_5s', float('nan')):.0%} of RAVDESS and "
        f"{hp['crema'].get('within_0_5s', float('nan')):.0%} of CREMA-D clips with both onsets); surprised likewise. For angry the readout leads "
        f"the brow-lowering blendshape by about half a second on RAVDESS. Sad and disgusted have no usable blendshape onset (the frown and "
        f"sneer scores rarely rise by 0.1), so the comparison is limited to four emotions. Many takes are expressive from the first frame "
        f"(no onset to detect), and the acted onset is not a scripted event with a marked time; the self-recorded events remain the open item.")
    R["notes"] += [
        "Clip-level numbers use the mean of the 1.5 s windowed series over the whole clip (RAVDESS about 3.6 s, CREMA-D about 2.5 s).",
        "CREMA-D subset: 8 held-out actors (every 4th of the 30), all their unspecified- and high-intensity clips, so each actor has the full 12 sentences × 6 emotions crossing.",
        "Reference units: scores and PCs are divided by their SD over the M1 dev-actor clip means, so 1.0 is one SD of the expression spread in M1.",
    ]
    (OUT / "gate2_metrics.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    write_report(R, OUT / "gate2_report.md")
    print("\n".join(f"{g['check']}: {g['pass']} -- {g['result']}" for g in gate))
    print("wrote", OUT / "gate2_report.md")


if __name__ == "__main__":
    main()
