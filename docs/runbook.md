# Runbook: recording and viewing

Short instructions for the two tools you use by hand. Everything runs locally except the model, which runs on Modal (`--env cogmirror`). Full details: `tools/recorder/README.md` and the "Viewer (Rerun)" section of the top-level `README.md`.

## 1. Record a session

Prerequisites: the story set and narration exist (`data/stories/stories.jsonl`, `data/stories/audio/`), and the venv is set up (`.venv/bin/pip install -e ".[dev,analysis,video,viz]"`).

Start the recorder (it opens http://127.0.0.1:8770/ in your default browser; use Chrome or Edge):

```bash
.venv/bin/python tools/recorder/serve.py
```

In the app:

1. **Setup:** tick the four reminders (same lighting and framing as every session, camera at eye height, text at the top of the screen right under the lens, mic on). Start the camera and check it reports 1920×1080 or at least 1280×720. Choose the number of takes (12 is about 20 minutes). Leave *Allow repeats* off. The condition filter defaults to *reading only*; pick *both, interleaved* to include listening takes.
2. **Calibration:** 15 s looking at the lens with a relaxed face (its mean becomes the session baseline). For the first session, also do the scripted events here or as a separate take: neutral → smile → neutral, a yawn, a frown, noting the times (this closes Gate 2's open item).
3. **Takes:** the queue alternates stories (and conditions, if listening is on).
   - *Reading:* read the story silently first, press space, then read each sentence aloud and press space the moment you finish it.
   - *Listening* (optional): look at the lens; the narration plays one sentence at a time. Press nothing.
4. **Review:** Enter keeps the take, R marks it bad and re-queues the pair. *End session after this take* stops early.

Files land in `data/recordings/<session_id>/<take_id>.webm` with a `.json` log of per-sentence timestamps beside each. Plan at least three sessions on different days.

To list the good takes with their segments:

```bash
.venv/bin/python scripts/recordings_manifest.py
```

## 2. Process the takes

Recording does not trigger processing by itself. The usual way is to leave a watcher running in a second terminal during the session:

```bash
.venv/bin/python scripts/m4_run_videos.py --watch --rrd
```

It polls `data/recordings/` every 10 s and processes each take as soon as the recorder has saved it as kept (takes marked bad are never picked up): remux, face tracking locally at low CPU priority, the model on Modal, the time series. The calibration take runs first and its mean becomes the session baseline (`data/video/<session>/baseline.json`), subtracted from the other takes, so start the watcher before or during calibration. The Rerun exports wait until no take is pending, since they transcode the video. Ctrl-C stops it after finishing the exports. For the model to stay loaded between takes, the video app must be deployed once (and again after any code change under `src/` or `modal_app/`):

```bash
.venv/bin/modal deploy --env cogmirror modal_app/video.py
```

`video_run.py` then uses the deployed app, whose container keeps the model for 10 minutes after the last call, so takes a few minutes apart skip the model load (about $0.03 per idle minute while it waits). Without a deployment each take runs an ephemeral app and pays the load, roughly a minute of wall time; `--ephemeral` forces that, for example to test working-tree changes without redeploying.

Without `--watch` the same command processes every kept take that has no series yet and exits; use that after a session if the watcher was not running:

```bash
.venv/bin/python scripts/m4_run_videos.py --rrd
```

Outputs go to `data/video/<session>/<take>/` (frames, activation store, `<take>.parquet`), which is where `scripts/m4_agreement.py` looks, plus `data/rerun/<session>_<take>.rrd` with `--rrd`. `--force` redoes existing series; `--sessions <id>` limits it to one session; `--dry-run` prints the commands. Other options are forwarded to `scripts/video_run.py` (`--fps`, `--crop`, `--window`, ...). Cost is about $1.65 per hour of video at the default 5 fps.

A single video outside a session (a dataset clip, a stray file) goes through `scripts/video_run.py <video> --rrd` directly; that writes to `data/video/<stem>/`.

## 3. View

Export every processed video and open them in the Rerun viewer, in a browser tab:

```bash
.venv/bin/python scripts/rerun_export.py --all --open
```

In the viewer:

- **Switch videos** in the *Sources* panel at the top left; each recording sits under its own `cogmirror/<session>/<take>` entry (one application id per recording, so each keeps its own default layout).
- **Scrub** with the timeline at the bottom; the video, the model's crop, the top emotions, the story (the sentence being read in bold, from the recorder's key presses) and every trace follow the cursor. Space plays, with the take's audio.
- **Traces** are overlaid on one plot on the right, with a colored legend; the *stacked* tab behind it draws them one strip each with the same y-scale (±3 reference SDs). The default set is chosen per take: valence, the PCs, the story's target emotions (drawn thicker; the story panel's header names them), and the six emotions that appear most often in the top-three labels over the clip. The blueprint panel on the left hides or shows strips. Every emotion is in the recording under `traces/emotion/`, so any can be dragged into a view.

Options: `--traces valence,pc2,happy,sad,angry` chooses the plotted traces (presets: `default`, `pcs`, `emotions`, `all`); `--y-range 4` changes the shared scale; `--take-log <take>.json` adds the sentence boundaries; `--native` opens the native Rerun app instead of the browser (it needs an `ffmpeg` executable to play the video; the browser does not); `--force` re-exports existing files.

`--open` keeps the server running until Ctrl-C. A later `--open` replaces that server with one holding every recording; refresh the tab to see them. The embedded video is transcoded small (640 px wide, a few MB per take) because the browser viewer loads every recording into memory; the model's crop is kept at full size. To open existing recordings without re-exporting:

```bash
.venv/bin/python -m rerun --serve-web data/rerun/*.rrd
```

then browse to `http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy` (the browser viewer does not find the server on its own).

## 4. Hosted version (share with others)

The recorder, the processing and the viewer links also run as a Modal web app (`modal_app/web.py`), so other people can record takes in their browser without installing anything. Visitors' takes live under `/public` on the Volume, apart from the research recordings.

One-time setup: create the access key secret and upload the projection reference (the stories and narration audio are baked into the image at deploy time). The key is the only protection on the page, so pick something long; keep a copy in `data/web_access_key.txt` (git-ignored):

```bash
.venv/bin/modal secret create --env cogmirror --force cogmirror-web ACCESS_KEY="$(cat data/web_access_key.txt)"
```

```bash
.venv/bin/modal run --env cogmirror modal_app/web.py::setup
```

Deploy (and redeploy after any change under `src/`, `modal_app/web.py`, `tools/recorder/static/` or `data/stories/`); the model app must be deployed too (`modal deploy --env cogmirror modal_app/video.py`):

```bash
.venv/bin/modal deploy --env cogmirror modal_app/web.py
```

The deploy prints the URL. Share it with `?key=<the key>` appended; the first visit sets a cookie, after which `/`, `/status` and the view links work without it. Rotating the key is `modal secret create --force` with a new value plus a redeploy.

How it runs: each kept take is uploaded to the Volume and a Modal function processes it at once (face tracking on CPU, activations through the deployed `cogmirror-video` app, projection against the session's calibration baseline, Rerun export). A story take waits for its session's calibration take to finish first. `/status` lists every take with its stage, timings and a *view* link that opens Rerun's hosted web viewer (app.rerun.io) on that recording; `/api/status` is the same as JSON. Cost per take is a few cents of GPU plus the model container's 10 minutes of warm idle after the last take.

To put locally recorded takes on the hosted version (they are processed there like a visitor's), `scripts/web_upload.py --url <app url> --session <local session> [--as <remote session>] --wait`. To analyse visitor takes locally, download them from the Volume (`modal volume get --env cogmirror cogmirror /public/recordings data/public/recordings`, likewise `/public/video`).

