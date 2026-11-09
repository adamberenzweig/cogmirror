"""Scripted-event check (Gate 2's open item): does the readout respond within ±1 s of marked events?

    .venv/bin/python scripts/m4_scripted_events.py --take data/recordings/<session>/001_calibration_calibration.json
    .venv/bin/python scripts/m4_scripted_events.py --take <log.json> --events events.csv [--series <take>.parquet]

Events come from the take log (entries of `events` with type `event`,
`marker` or `scripted_event` and a `label`, or `<label>_start`/`<label>_end`
pairs; the recorder's calibration take can be annotated by hand) or from a
CSV with columns `t,label[,end]` in seconds on the recording clock. Labels
map to signals (smile/laugh -> valence up, frown/sad -> valence down,
yawn/tired -> score_tired, surprise -> score_surprised, angry, afraid,
disgusted); --signal overrides with `<column>[:+1|-1]` for every event.

The face series is found like the agreement script does (<series-root>/<session>/
<take>/<take>.parquet or <series-root>/<take>/<take>.parquet) unless --series is
given. For each event: baseline mean/SD over [t-3, t-0.5] s (or --noise-sd), the
first time in [t-1, t+3] s at which the signal moves ≥ 2 SD in the expected
direction for 2 samples, the lag of that onset, whether it is within ±1 s, and
the peak z within 3 s. Writes results/m4/scripted_events_<take>.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogmirror import align  # noqa: E402
from cogmirror.store import read_parquet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True, help="take log JSON")
    ap.add_argument("--events", default="", help="CSV of t,label[,end] (overrides the log's events)")
    ap.add_argument("--series", default="", help="face series Parquet (default: found under --series-root)")
    ap.add_argument("--series-root", default=str(ROOT / "data/video"))
    ap.add_argument("--signal", default="", help="<column>[:sign] to use for every event, e.g. valence:+1")
    ap.add_argument("--noise-sd", type=float, default=0.0, help="SD to use instead of the pre-event SD (e.g. from the calibration clip)")
    ap.add_argument("--z", type=float, default=2.0, help="onset threshold in SDs")
    ap.add_argument("--out-dir", default=str(ROOT / "results/m4"))
    args = ap.parse_args()

    take = align.load_take(args.take)
    events = align.scripted_events_from_csv(args.events) if args.events else align.scripted_events_from_log(take)
    if not events:
        print("no scripted events found: mark them in the log (type 'event', fields t and label) or pass --events CSV")
        return 1
    sp = Path(args.series) if args.series else align.series_path(args.series_root, take)
    if sp is None or not sp.exists():
        print(f"no face series for {take.take_id}; run scripts/video_run.py on {take.video_path} first (or pass --series)")
        return 1
    cols, meta = read_parquet(sp)
    t = np.asarray(cols["t"], dtype=np.float64)

    override = None
    if args.signal:
        name, _, sgn = args.signal.partition(":")
        override = (name, int(sgn) if sgn else +1)
        if name not in cols:
            print(f"column {name} not in the series; columns: {[c for c in cols if c.startswith(('pc', 'valence', 'score_'))][:12]}...")
            return 1
    rows = []
    for e in events:
        sig = override or align.event_signal(e.label, cols)
        if sig is None:
            rows.append({"t": e.t, "label": e.label, "signal": None, "note": "no signal mapped for this label (use --signal)"})
            continue
        col, sign = sig
        r = align.event_response(t, np.asarray(cols[col], dtype=np.float64), e.t, sign=sign, z_threshold=args.z,
                                 noise_sd=args.noise_sd or None)
        # Also the unsigned response of valence, as a second opinion on events that are not about valence.
        rv = align.event_response(t, np.abs(np.asarray(cols["valence"], dtype=np.float64) - np.nanmean(cols["valence"])), e.t, sign=+1,
                                  z_threshold=args.z, noise_sd=args.noise_sd or None) if "valence" in cols and col != "valence" else None
        rows.append({"t": e.t, "label": e.label, "signal": col, "sign": sign, **{k: v for k, v in r.items() if k not in ("t_event", "sign")},
                     "abs_valence_lag": rv["lag"] if rv else None})
    scored = [r for r in rows if r.get("signal")]
    n_within = sum(1 for r in scored if r.get("within_1s"))
    n_onset = sum(1 for r in scored if np.isfinite(r.get("lag", float("nan"))))
    lags = [r["lag"] for r in scored if np.isfinite(r.get("lag", float("nan")))]
    summary = {"take": take.take_id, "session": take.session_id, "series": str(sp), "n_events": len(events), "n_scored": len(scored),
               "n_onset_found": n_onset, "n_within_1s": n_within, "median_lag": float(np.median(lags)) if lags else None,
               "pass": bool(scored) and n_within == len(scored), "z_threshold": args.z, "noise_sd": args.noise_sd or None,
               "series_meta": meta, "events": rows}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"scripted_events_{take.take_id}.json").write_text(json.dumps(summary, indent=1, default=float))
    L = [f"# Scripted events: {take.take_id} ({take.session_id})", "",
         f"Series: `{sp}`. Onset = first time in [t−1, t+3] s at which the signal exceeds its pre-event baseline "
         f"([t−3, t−0.5] s) by {args.z} SD in the expected direction for two samples"
         + (f"; SD fixed at {args.noise_sd}" if args.noise_sd else "") + ".", "",
         "| Event | t (s) | Signal | Baseline | SD | Onset (s) | Lag (s) | Within ±1 s | Peak z | Peak t (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("signal"):
            L.append(f"| {r['label']} | {r['t']:.2f} | — | | | | | | | {r.get('note', '')} |")
            continue
        f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.2f}"  # noqa: E731
        L.append(f"| {r['label']} | {r['t']:.2f} | {r['signal']} ({'+' if r['sign'] > 0 else '−'}) | {f(r.get('baseline'))} | {f(r.get('sd'))} "
                 f"| {f(r['onset'])} | {f(r['lag'])} | {'yes' if r['within_1s'] else 'no'} | {f(r['peak_z'])} | {f(r['peak_t'])} |")
    L += ["", f"{n_within}/{len(scored)} events respond within ±1 s ({n_onset} with an onset found; median lag "
          f"{summary['median_lag']:+.2f} s)." if lags else f"{n_within}/{len(scored)} events respond within ±1 s (no onset found).",
          "", f"Gate 2 scripted-event criterion (every marked event within ±1 s): {'PASS' if summary['pass'] else 'not met'}."]
    (out_dir / f"scripted_events_{take.take_id}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
