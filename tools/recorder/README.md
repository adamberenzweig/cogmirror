# M3 recorder

A local web app for the text–face recordings: webcam takes of one subject reading
stories aloud (teleprompter, sentence by sentence) or listening to a per-sentence
narration, with per-sentence timestamps logged against the recording clock. No
build step; a standard-library Python server plus static HTML/JS. Nothing leaves
the machine.

## Start it

Stories and narration must exist first (`data/stories/stories.jsonl` and
`data/stories/audio/<id>/<k>.m4a`, made by `scripts/generate_stories.py`). Then:

```bash
.venv/bin/python tools/recorder/serve.py
```

This opens http://127.0.0.1:8770/ in the default browser. Use Chrome, Edge or
Safari 17+; the app needs `getUserMedia` and `MediaRecorder` (WebM VP9 in
Chromium, which is what the analysis expects). Grant camera and microphone
access when asked. Options: `--port`, `--data` (a folder with `stories/` and
`recordings/`, default `data/`), `--no-browser`. The desktop app's
`.claude/launch.json` has a `recorder` entry for the same thing.

For a dry run without a camera, open `/?fake=1&calib=3`: a synthetic video
stream and a 3-second calibration. Takes made that way carry `fake_stream: true`.

## Run a session

1. **Setup screen.** Tick the four reminders (same lighting and framing as every
   other session, camera at eye height, text at the top of the screen right under
   the lens, mic on and speakers on). Pick the camera and resolution (1920×1080 by
   default; 1280×720 is the minimum), press *Start camera*, check the reported
   resolution and frame rate. Set the number of takes for this session (12 is
   about 20 minutes). Leave *Allow repeats* off: the tool reads every log under
   `data/recordings/` and never offers a (story, condition) pair that already
   has a good take.
2. **Calibration.** Every session starts with a 15 s neutral take: look at the
   lens, relaxed face, no talking. Press space to start. Its mean is the
   session baseline that `scripts/m4_run_videos.py` subtracts from the other
   takes, so 15 s (75 frames at 5 fps) is plenty; `?calib=60` on the URL makes
   it longer.
3. **Takes.** The queue is a random order of (story, condition) pairs, seeded by
   the session id, arranged so that consecutive takes differ in story, emotion
   cluster and condition. *Only this condition* defaults to reading only: the
   Gate 2 speech-motion check showed speech adds noise but no emotion-specific
   content, and the narration audio is too flat to carry the stories, so
   listening takes are off unless selected. The screen stays the same grey throughout so the light
   on the face does not change between screens.
   - **Reading.** The whole story appears first; read it silently. Press space
     when ready: recording starts immediately and the first sentence appears
     large at the top of the screen. Read it aloud, expressively, and press
     space the moment you finish it; the next sentence appears. After the last
     sentence, hold still for about a second. Every press is logged.
   - **Listening.** Look at the lens. Press space: after 1.5 s the narration
     plays one sentence at a time with half-second gaps. The audio start
     (`playing` event) and end (`ended` event) of each sentence are logged.
     Keep looking at the lens; there is nothing to press.
4. **Review.** After each take: *Enter* keeps it, *R* marks it bad and queues
   the same pair again immediately. Bad takes are saved too (flagged
   `"bad": true`) and ignored by the progress tracking and the manifest.
   *End session after this take* stops early; the pairs left over are simply
   not recorded and will come up in a later session.
5. **Done.** A summary lists the takes. Plan at least three sessions on
   different days; the hold-out session and stories are chosen at analysis
   time (`split` is assigned in M4, not here).

The subject is never shown the target emotion during a take; it is only in the logs.

## Where files land

```
data/recordings/<session_id>/<take_id>.webm   video+audio, WebM (VP9/Opus), one file per take
data/recordings/<session_id>/<take_id>.json   the take log
```

`take_id` is `<take index>_<story id>_<condition>`, e.g. `003_s017_read`,
`001_calibration_calibration`. The log holds: session id, story id, condition,
kind, target emotion(s) and clusters, the story text and sentences, the turn
position, `start_wallclock`, `duration_s`, `sentence_times` (start/end in
seconds since the `MediaRecorder` start event), the raw `events`, camera
settings (`getSettings()` of the video and audio tracks), recorder settings
(MIME type, bitrate), screen size, user agent, the `bad` flag and an optional
note.

Timing notes for the analysis: the clock is `performance.now()` at the
recorder's `start` event, which is within a frame or two of the first encoded
frame. In the reading condition the sentence boundaries are the reader's key
presses; WhisperX word timestamps on the audio track refine them. In the
listening condition they are the narration's own play/end events. Chromium
writes WebM without a duration header; `ffmpeg -i in.webm -c copy out.webm`
or `ffprobe -count_frames` fixes or measures it.

## Manifest

```bash
.venv/bin/python scripts/recordings_manifest.py
```

writes `results/recordings_manifest.jsonl`, one line per good take (add
`--include-bad` to list all) with its per-sentence segments, in the spirit of
`results/m1/manifest.jsonl`. `--data` and `-o` change the folders.
