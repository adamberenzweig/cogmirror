"""M4: text–face agreement on the recorded takes (Gate 3).

    .venv/bin/python scripts/m4_agreement.py                       # data/recordings + data/video + results/m4/text_ev.parquet
    .venv/bin/python scripts/m4_agreement.py --data data/synthetic_m4 --series-root data/synthetic_m4/video \\
        --text-ev data/synthetic_m4/results/m4/text_ev.parquet --out-dir data/synthetic_m4/results/m4

Inputs: take logs under <data>/recordings/<session>/*.json, one face series per
take under <series-root> (from scripts/video_run.py or m4_run_videos.py:
<series-root>/<session>/<take>/<take>.parquet, or <series-root>/<take>/<take>.parquet),
and the text EV table. The split (held-out session and stories) is read from
--config (default <out-dir>/split.json), created deterministically when absent.

Outputs in --out-dir: agreement_report.md (the Gate 3 table and every metric by
condition and split), agreement_metrics.json, and segments.parquet (one row per
segment with its face and text valence, for plots).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogmirror import agreement, align, text_ev  # noqa: E402
from cogmirror.store import DEFAULT_SCORE_LAYER, DEFAULT_VALENCE_LAYER, read_parquet, write_parquet  # noqa: E402


def parse_grid(spec: str) -> tuple[float, ...]:
    """'-3:3:0.5' -> (-3.0, -2.5, ..., 3.0); or a comma list."""
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        return tuple(float(x) for x in np.round(np.arange(lo, hi + step / 2, step), 3))
    return tuple(float(x) for x in spec.split(","))


def series_layers(data_root: Path, series_root: Path, takes: list[align.Take]) -> tuple[int, int]:
    """Valence and score layers from the first face series' metadata."""
    for t in takes:
        p = align.series_path(series_root, t)
        if p is not None:
            _, meta = read_parquet(p)
            return int(meta.get("valence_layer", DEFAULT_VALENCE_LAYER)), int(meta.get("score_layer", DEFAULT_SCORE_LAYER))
    return DEFAULT_VALENCE_LAYER, DEFAULT_SCORE_LAYER


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"), help="folder with recordings/ and stories/")
    ap.add_argument("--series-root", default="", help="face series root (default <data>/video)")
    ap.add_argument("--text-ev", default=str(ROOT / "results/m4/text_ev.parquet"))
    ap.add_argument("--stories", default="", help="stories.jsonl (default <data>/stories/stories.jsonl)")
    ap.add_argument("--out-dir", default=str(ROOT / "results/m4"))
    ap.add_argument("--config", default="", help="split config JSON (default <out-dir>/split.json; created if absent)")
    ap.add_argument("--pc-layer", type=int, default=0, help="text layer for PCs (default: the face series' valence layer)")
    ap.add_argument("--score-layer", type=int, default=0, help="text layer for scores (default: the face series' score layer)")
    ap.add_argument("--lag-grid", default="-3:3:0.5", help="seconds; positive = face trails text")
    ap.add_argument("--lag", default="", help="fix the lag instead of estimating it")
    ap.add_argument("--n-perm", type=int, default=agreement.DEFAULT_N_PERM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refine", default="", choices=("", "whisperx"), help="refine reading-condition boundaries with WhisperX")
    ap.add_argument("--no-ridge", action="store_true")
    ap.add_argument("--no-blendshapes", action="store_true")
    ap.add_argument("--include-bad", action="store_true")
    ap.add_argument("--sessions", default="", help="comma list of session ids to use (default all)")
    args = ap.parse_args()

    data = Path(args.data)
    series_root = Path(args.series_root) if args.series_root else data / "video"
    stories_path = Path(args.stories) if args.stories else data / "stories" / "stories.jsonl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    takes = align.find_takes(data / "recordings", include_bad=args.include_bad,
                             sessions=args.sessions.split(",") if args.sessions else None)
    story_takes = [t for t in takes if t.condition in align.CONDITIONS]
    print(f"{len(takes)} takes ({len(story_takes)} story takes) in {len({t.session_id for t in takes})} session(s) under {data / 'recordings'}")
    if not story_takes:
        print("nothing to analyse")
        return 1
    stories = text_ev.load_stories(stories_path) if stories_path.exists() else []

    pc_layer, score_layer = series_layers(data, series_root, story_takes)
    pc_layer = args.pc_layer or pc_layer
    score_layer = args.score_layer or score_layer
    cols, _ = text_ev.read_table(args.text_ev)
    ev = agreement.load_text_ev(cols, pc_layer, score_layer)
    print(f"text EVs: PCs at layer {pc_layer}, scores at layer {score_layer}, {len(ev.index)} rows")

    cfg_path = Path(args.config) if args.config else out_dir / "split.json"
    if cfg_path.exists():
        split = agreement.SplitConfig.load(cfg_path)
        print(f"split from {cfg_path}: {split.note or ''}")
    else:
        split = agreement.make_split(story_takes, stories or [{"id": s, "kind": ""} for s in sorted({t.story_id for t in story_takes})],
                                     seed=args.seed)
        split.save(cfg_path)
        print(f"split written to {cfg_path}: {split.note}")

    tdata, skipped = agreement.collect_takes(story_takes, series_root, ev, refine=args.refine or None,
                                             blendshapes=not args.no_blendshapes, progress=print)
    for s in skipped:
        print(f"  skipped/partial {s['take']}: {s['reason']}")
    if not tdata:
        print("no take has a face series; run scripts/m4_run_videos.py first")
        return 1

    cfg = agreement.AnalysisConfig(lag_grid=parse_grid(args.lag_grid), n_perm=args.n_perm, seed=args.seed,
                                   fixed_lag=float(args.lag) if args.lag else None, ridge=not args.no_ridge,
                                   blendshapes=not args.no_blendshapes)
    R = agreement.run_analysis(tdata, split, cfg, progress=print)
    R["inputs"] = {"data": str(data), "series_root": str(series_root), "text_ev": str(args.text_ev), "pc_layer": pc_layer,
                   "score_layer": score_layer, "skipped": skipped, "refine": args.refine or None}
    (out_dir / "agreement_metrics.json").write_text(agreement.to_json(R))
    report = agreement.write_report(R, out_dir / "agreement_report.md")

    # Per-segment table for plots.
    a = agreement.assemble(tdata, lag=R["lag"]["lag"], ev_emotions=tdata[0].text_emotions)
    dev, held = agreement.split_masks(a, split)
    nz = agreement.normalize(a, dev)
    seg_cols = {"session": a.session, "take": a.take, "story": a.story, "condition": a.condition, "kind": a.kind,
                "sentence_idx": a.sentence_idx, "start": a.start, "end": a.end, "after_turn": a.after_turn,
                "split": np.where(held, "heldout", "dev").astype(str), "n_frames": a.face_n_frames,
                "face_valence_z": nz.face_pcs[:, 0], "text_valence_z": nz.text_pcs[:, 0]}
    for k in range(1, min(a.face_pcs.shape[1], a.text_pcs.shape[1])):
        seg_cols[f"face_pc{k + 1}_z"] = nz.face_pcs[:, k]
        seg_cols[f"text_pc{k + 1}_z"] = nz.text_pcs[:, k]
    write_parquet(out_dir / "segments.parquet", seg_cols, {"lag_s": str(R["lag"]["lag"])})

    g = R["gate3"]
    print(f"\nlag {R['lag']['lag']:+.1f} s; held-out valence r by condition: "
          + ", ".join(f"{c}: {agreement._f(v['valence_r'])} (p {agreement._p(v['p_story'])}, n {v['n']})" for c, v in g["by_condition"].items()))
    print(f"Gate 3 valence: {'PASS' if g['valence']['pass'] else 'no'}; turns: {'PASS' if g['turn']['pass'] else 'no'}")
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
