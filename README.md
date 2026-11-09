# cogmirror

Reading LLM emotion vectors from facial video. See the [brief](cogmirror-brief.md), the [design](face-emotion-vectors-design.md), and the [work plan](face-emotion-vectors-workplan.md).

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

## Layout

| Path | Contents |
|---|---|
| `src/cogmirror/evspace.py` | Vectors, recomputation from caches, PCA basis, projection, reference statistics |
| `src/cogmirror/model.py` | Model loads, residual-stream capture, gemotions-style story pooling |
| `src/cogmirror/readouts.py` | Image prompt layouts and readout positions |
| `src/cogmirror/gate0.py` | Reproduction check against the gemotions activation caches |
| `src/cogmirror/toyfaces.py` | Schematic drawn faces for smoke tests |
| `modal_app/app.py` | Modal image, Volume, and remote functions |
| `scripts/` | Local entry points |
| `data/`, `results/` | Not in git |
