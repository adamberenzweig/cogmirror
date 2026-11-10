# cogmirror

Reading LLM emotion vectors from facial video. See the [brief](cogmirror-brief.md), the [design](face-emotion-vectors-design.md), and the [work plan](face-emotion-vectors-workplan.md). Day-to-day instructions for recording and viewing are in [docs/runbook.md](docs/runbook.md).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Add `".[dev,model]"` to also install torch and transformers locally. That is only needed for `scripts/smoke_local.py`; the 31B model runs on Modal.

```bash
.venv/bin/python scripts/fetch_gemotions.py --cache-layers 40
```

This downloads the gemotions vectors, stories, and the layer-40 activation cache (about 4.5 GB) to `data/gemotions`.

```bash
.venv/bin/python -m pytest -q
```

## Running on Modal

Authenticate once (opens a browser):

```bash
.venv/bin/modal setup
```

All Modal commands use the `cogmirror` environment in the workspace (`--env cogmirror`).

Copy the model weights and gemotions data to the `cogmirror` Volume. CPU only; about 62 GB of weights plus 15 GB of caches:

```bash
.venv/bin/modal run --env cogmirror modal_app/app.py::download
```

The video pipeline (M2 onward) should be deployed once so its containers keep the model loaded between calls; redeploy after changing anything under `src/` or `modal_app/`:

```bash
.venv/bin/modal deploy --env cogmirror modal_app/video.py
```

Gate 0, the reproduction check. Uses two GPUs for a few minutes each, and writes `results/gate0_*.json`:

```bash
.venv/bin/modal run --env cogmirror modal_app/app.py::gate0
```

Image smoke test: a few images through every readout layout, plus a throughput figure. Without `--images` it uses schematic drawn faces. With it, the first image is the reference the others are compared against.

```bash
.venv/bin/modal run --env cogmirror modal_app/app.py::smoke --images neutral.jpg,smile.jpg
```

Gate 0 diagnostic: saves our story activations and reruns a subset under different numerical settings.

```bash
.venv/bin/modal run --env cogmirror modal_app/app.py::gate0diag
```

## M1: static-image feasibility

Data (both corpora to the Volume, face crops, the frame manifest), then the first-pass inference, baselines and analysis:

```bash
.venv/bin/modal run --env cogmirror modal_app/app.py::m1data
```

```bash
.venv/bin/modal run --env cogmirror modal_app/app.py::m1run --run pass1
```

```bash
.venv/bin/modal run --env cogmirror modal_app/app.py::m1fer
```

Then download `/acts/pass1` from the Volume to `data/acts/pass1` and run `scripts/m1_analyze.py --run pass1`. `m1noise` + `scripts/m1_noise.py` give the noise floor; `m1prompts` + `scripts/m1_prompts.py` the prompt robustness; `samples` writes a browsable gallery of clips to `data/samples/index.html`. Results and the written-up gate table are in the work plan.

## Viewer (Rerun)

Time series are inspected in [Rerun](https://rerun.io): video, the crop the model saw, and the traces overlaid on one plot, with a stacked EKG-style tab behind it. Install the extra once (`.venv/bin/pip install -e ".[viz]"`), then:

```bash
.venv/bin/python scripts/rerun_export.py --all --open
```

That exports every processed video under `data/video/` to `data/rerun/<id>.rrd` and opens the browser viewer on all of them; the Sources panel (top left) switches between videos. `--traces valence,pc2,happy,sad,angry` picks the stacked traces (default, per video: valence, the PCs, the story's target emotions, and the six emotions most often in the frame-level top three over the clip); every other trace is still in the recording under `traces/` and can be added from the blueprint panel. `--y-range` sets the shared scale (reference-SD units, default ±3). `scripts/video_run.py ... --rrd` exports as part of a run; `--native` opens the native app instead of the browser (it needs `ffmpeg` to play the video).

## Layout

| Path | Contents |
|---|---|
| `src/cogmirror/evspace.py` | Vectors, recomputation from caches, PCA basis, projection, reference statistics |
| `src/cogmirror/model.py` | Model loads, residual-stream capture, gemotions-style story pooling |
| `src/cogmirror/readouts.py` | Image prompt layouts and readout positions |
| `src/cogmirror/gate0.py` | Reproduction check against the gemotions activation caches |
| `src/cogmirror/toyfaces.py` | Schematic drawn faces for smoke tests |
| `src/cogmirror/datasets.py` | CREMA-D and RAVDESS metadata, actor splits |
| `src/cogmirror/prep.py` | Frame extraction, face detection, crops, blendshapes (runs on Modal CPU) |
| `src/cogmirror/m1.py` | M1 inference: activations, verbal logits, forced-choice labels |
| `src/cogmirror/perturb.py` | Nuisance perturbations for the noise floor |
| `src/cogmirror/metrics.py` | Gate metrics |
| `src/cogmirror/rerun_export.py` | Rerun recording export (video, audio, crops, story, traces, blueprint) |
| `modal_app/app.py` | Modal image, Volume, and remote functions |
| `scripts/` | Local entry points |
| `data/`, `results/` | Not in git |
