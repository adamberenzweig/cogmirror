# Runbook: recording and viewing

Short instructions for the two tools you use by hand. Everything runs locally except the model, which runs on Modal (`--env cogmirror`). Full details: `tools/recorder/README.md` and the "Viewer (Rerun)" section of the top-level `README.md`.

## 1. Record a session

Prerequisites: the story set and narration exist (`data/stories/stories.jsonl`, `data/stories/audio/`), and the venv is set up (`.venv/bin/pip install -e ".[dev,analysis,video,viz]"`).

Start the recorder (it opens http://127.0.0.1:8770/ in your default browser; use Chrome or Edge):

```bash
.venv/bin/python tools/recorder/serve.py
```

In the app:

1. **Setup:** tick the four reminders (same lighting and framing as every session, camera at eye height, text at the top of the screen right under the lens, mic and speakers on). Start the camera and check it reports 1920×1080 or at least 1280×720. Choose the number of takes (12 is about 20 minutes). Leave *Allow repeats* off.
2. **Calibration:** 60 s looking at the lens with a relaxed face. For the first session, also do the scripted events here or as a separate take: neutral → smile → neutral, a yawn, a frown, noting the times (this closes Gate 2's open item).
3. **Takes:** the queue alternates stories and conditions.
   - *Reading:* read the story silently first, press space, then read each sentence aloud and press space the moment you finish it.
   - *Listening:* look at the lens; the narration plays one sentence at a time. Press nothing.
4. **Review:** Enter keeps the take, R marks it bad and re-queues the pair. *End session after this take* stops early.

Files land in `data/recordings/<session_id>/<take_id>.webm` with a `.json` log of per-sentence timestamps beside each. Plan at least three sessions on different days.

To list the good takes with their segments:

```bash
.venv/bin/python scripts/recordings_manifest.py
```

## 2. Process a take

One command per video; it tracks the face locally, runs the model on Modal, writes the time series, and (with `--rrd`) the Rerun recording:

```bash
.venv/bin/python scripts/video_run.py data/recordings/<session>/<take>.webm --rrd
```

Outputs go to `data/video/<take>/` (frames, activation store, `<take>.parquet`) and `data/rerun/<take>.rrd`. Add `--baseline session` to subtract the take's own mean, or `--baseline <calibration.json>` after `--save-baseline` on the calibration take. Cost is about $1.65 per hour of video at the default 5 fps.

## 3. View

Export every processed video and open the Rerun viewer on all of them:

```bash
.venv/bin/python scripts/rerun_export.py --all --open
```

In the viewer:

- **Switch videos** in the *Recordings* panel at the top left; each is listed by its id.
- **Scrub** with the timeline at the bottom; the video, the model's crop, the top emotions and every trace follow the cursor. Space plays.
- **Traces** are stacked on the right, one strip each with the same y-scale (±3 reference SDs) and a colored legend; the *overlay* tab draws them on one plot. The blueprint panel on the left hides or shows strips. Every emotion is in the recording under `traces/emotion/`, so any can be dragged into a view.

Options: `--traces valence,pc2,happy,sad,angry` chooses the stacked strips (presets: `default`, `pcs`, `emotions`, `all`); `--y-range 4` changes the shared scale; `--take-log <take>.json` adds the sentence boundaries; `--web` serves the viewer in a browser tab instead of the native app; `--force` re-exports existing files.

To open an existing recording without re-exporting:

```bash
.venv/bin/python -m rerun data/rerun/*.rrd
```
