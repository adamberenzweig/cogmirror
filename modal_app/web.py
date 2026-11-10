"""Hosted recorder: record takes in a browser, process them on Modal, view them in Rerun.

    modal run    --env cogmirror modal_app/web.py::setup     # once: the projection reference to the Volume
    modal deploy --env cogmirror modal_app/web.py            # the web app; prints its URL

Needs the Modal Secret `cogmirror-web` with ACCESS_KEY=<shared key>; every page
and file needs `?key=<key>` once (it sets a cookie) and the model app
`cogmirror-video` deployed (modal deploy modal_app/video.py).

Layout on the Volume, under /vol/public (separate from the research data):
    reference.npz            the M1 projection reference (stories and audio are baked into the image)
    recordings/<session>/    uploaded takes, <take>.webm + <take>.json (as tools/recorder/serve.py writes them)
    video/<session>/<take>/  frames, activation store, series (as scripts/video_run.py writes them), baseline.json
    rerun/<session>_<take>.rrd
    status/<session>/<take>.json

Routes: `/` the recorder, `/status` the processing table (`/api/status` as JSON),
`/view/<session>/<take>` opens Rerun's hosted web viewer on the take's recording,
`/rrd/<name>.rrd` serves recordings (CORS open, key in the query, for that viewer).
"""


import json
import os
import re
import traceback
from pathlib import Path
from urllib.parse import quote

import modal

APP = "cogmirror-web"
MODEL_APP = "cogmirror-video"
VOL = "/vol"
PUB = Path(VOL) / "public"
STATIC = "/app/static"
STORIES = "/app/stories"        # stories + narration audio travel with the image
REPO = Path(__file__).resolve().parents[1]
COOKIE = "cogmirror_key"
RERUN_VERSION = "0.42.0"       # must match the rerun-sdk below; the hosted viewer is version-specific

app = modal.App(APP)
volume = modal.Volume.from_name("cogmirror", create_if_missing=False)
secret = modal.Secret.from_name("cogmirror-web")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libegl1", "libgles2", "wget")
    .pip_install(
        "fastapi[standard]>=0.115",
        "mediapipe==1.2.2",
        "numpy>=1.26",
        "pillow>=10",
        "pyarrow>=15",
        f"rerun-sdk=={RERUN_VERSION}",
        "imageio-ffmpeg",
    )
    .run_commands(
        "wget -q -O /root/face_landmarker.task https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    .add_local_python_source("cogmirror")
    .add_local_dir(str(REPO / "tools" / "recorder" / "static"), remote_path=STATIC)
    .add_local_dir(str(REPO / "data" / "stories"), remote_path=STORIES, ignore=["raw/*", "text/*", "plan.json"])
)

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


# --------------------------------------------------------------------------
# Processing: one take, spawned by the upload handler
# --------------------------------------------------------------------------

@app.function(image=image, volumes={VOL: volume}, cpu=2, memory=8192, timeout=30 * 60)
def process_take(session: str, take: str) -> dict:
    from cogmirror import pipeline, recorder, rerun_export

    volume.reload()
    status = PUB / "status"
    rec_dir = PUB / "recordings" / session
    video = rec_dir / f"{take}.webm"
    log = recorder.read_status(rec_dir / f"{take}.json") or {}
    work = PUB / "video" / session / take
    base_file = PUB / "video" / session / "baseline.json"
    is_calibration = log.get("condition") == "calibration"
    if not video.exists():
        recorder.write_status(status, session, take, "failed", error="no video file")
        volume.commit()
        return {"ok": False}

    def stage(name):
        recorder.write_status(status, session, take, name)
        volume.commit()

    if not is_calibration:
        wait_for_calibration(session, take)
        volume.reload()

    def run_jobs(jobs, batch_size):
        infer = modal.Cls.from_name(MODEL_APP, "Infer")
        return list(infer().run.starmap(jobs, kwargs={"batch_size": batch_size}))

    s = pipeline.Settings(gemotions=str(Path(VOL) / "gemotions"), reference=str(PUB / "reference.npz"))
    rrd = rerun_export.rrd_path(PUB / "rerun", work)
    baseline = "none" if is_calibration or not base_file.exists() else str(base_file)
    try:
        result = pipeline.process(
            video, work, run_jobs, s, baseline=baseline,
            save_baseline=str(base_file) if is_calibration else "",
            rrd=rrd, stage=stage, log=print,
        )
        recorder.write_status(status, session, take, "done", rrd=rrd.name, timings=result["timings"], summary=result["summary"],
                              baseline="none" if baseline == "none" else "calibration")
        volume.commit()
        return {"ok": True, **result}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        recorder.write_status(status, session, take, "failed", error=f"{type(e).__name__}: {e}"[:500])
        volume.commit()
        return {"ok": False, "error": str(e)}


def wait_for_calibration(session: str, take: str, timeout_s: float = 15 * 60) -> None:
    """Takes of a session are spawned as they are kept, so a story take can start before the
    calibration take has finished; wait for it, so the session baseline exists (as the local
    watcher's ordering guarantees). Gives up quietly after `timeout_s` or if it failed."""
    import time

    from cogmirror import recorder

    status = PUB / "status"
    rec_dir = PUB / "recordings" / session
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        volume.reload()
        cals = [p for p in rec_dir.glob("*_calibration_calibration.json")
                if not (recorder.read_status(p) or {}).get("bad")]
        if not cals:
            return
        states = [(recorder.read_status(recorder.status_path(status, session, p.stem)) or {}).get("stage") for p in cals]
        if all(st in ("done", "failed") for st in states):
            return
        if time.time() - t0 < 5:
            recorder.write_status(status, session, take, "waiting for calibration")
            volume.commit()
        time.sleep(5)


# --------------------------------------------------------------------------
# Web app
# --------------------------------------------------------------------------

@app.function(image=image, volumes={VOL: volume}, secrets=[secret], cpu=1, memory=2048, timeout=600,
              scaledown_window=300)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="cogmirror")
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles

    from cogmirror import recorder

    import threading

    api = FastAPI()
    api.mount("/static", StaticFiles(directory=STATIC), name="static")
    static = Path(STATIC)
    reload_lock = threading.Lock()

    def refresh():
        """See other containers' writes. A reload fails while a file on the Volume is open in
        this container (another request streaming a recording); then the view is a little stale."""
        with reload_lock:
            try:
                volume.reload()
            except Exception as e:  # noqa: BLE001
                print(f"volume.reload skipped: {e}")
    key = os.environ.get("ACCESS_KEY", "")
    denied = HTMLResponse("<p style='font-family:sans-serif'>This page needs an access key: open the link you were given.</p>", 401)

    @api.middleware("http")
    async def gate(request: Request, call_next):
        q = request.query_params.get("key")
        if not key or not (q == key or request.cookies.get(COOKIE) == key):
            return denied
        resp = await call_next(request)
        if q == key and request.cookies.get(COOKIE) != key:
            resp.set_cookie(COOKIE, key, max_age=90 * 24 * 3600, httponly=True, samesite="lax", secure=True)
        return resp

    def safe(*ids: str) -> bool:
        return all(SAFE_ID.match(i) for i in ids)

    @api.get("/")
    def index():
        return FileResponse(static / "index.html")

    @api.get("/status")
    def status_page():
        return FileResponse(static / "status.html")

    story_list = recorder.load_stories(STORIES)

    @api.get("/api/stories")
    def stories():
        return JSONResponse(story_list)

    @api.get("/audio/{story}/{k}.m4a")
    def audio(story: str, k: int):
        if not safe(story):
            return Response("not found", 404)
        p = Path(STORIES) / "audio" / story / f"{k}.m4a"
        return FileResponse(p, media_type="audio/mp4") if p.exists() else Response("not found", 404)

    @api.get("/api/progress")
    def progress():
        refresh()
        (PUB / "recordings").mkdir(parents=True, exist_ok=True)
        return JSONResponse(recorder.scan_recordings(PUB / "recordings"))

    @api.get("/api/status")
    def status_json():
        refresh()
        rows = recorder.status_rows(recorder.scan_recordings(PUB / "recordings"), recorder.scan_status(PUB / "status"))
        return JSONResponse({"rows": rows})

    @api.post("/api/take/{session}/{take}.{ext}")
    async def upload(session: str, take: str, ext: str, request: Request):
        if not safe(session, take) or ext not in ("webm", "json"):
            return JSONResponse({"ok": False, "error": "bad name"}, 404)
        body = await request.body()
        d = PUB / "recordings" / session
        try:
            d.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # A container's view of the Volume can be behind another container's commit; mkdir then
            # fails with EPERM instead of EEXIST. Reload and try again.
            await volume.reload.aio()
            d.mkdir(parents=True, exist_ok=True)
        target = d / f"{take}.{ext}"
        spawned = False
        if ext == "json":
            try:
                log = json.loads(body.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                return JSONResponse({"ok": False, "error": "bad json"}, 400)
            target.write_text(json.dumps(log, indent=1, ensure_ascii=False))
            if not log.get("bad") and (d / f"{take}.webm").exists():
                recorder.write_status(PUB / "status", session, take, "queued")
                await volume.commit.aio()
                await process_take.spawn.aio(session, take)
                spawned = True
            else:
                await volume.commit.aio()
        else:
            target.write_bytes(body)
            await volume.commit.aio()
        return JSONResponse({"ok": True, "path": f"public/recordings/{session}/{take}.{ext}", "bytes": len(body), "processing": spawned})

    @api.get("/rrd/{name}")
    def rrd(name: str):
        if not safe(name) or not name.endswith(".rrd"):
            return Response("not found", 404)
        p = PUB / "rerun" / name
        if not p.exists():
            refresh()
        if not p.exists():
            return Response("not found", 404)
        return FileResponse(p, media_type="application/octet-stream",
                            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"})

    @api.get("/view/{session}/{take}")
    def view(session: str, take: str, request: Request):
        if not safe(session, take):
            return Response("not found", 404)
        name = f"{session}_{take}.rrd"
        base = str(request.base_url).rstrip("/")
        if base.startswith("http://") and "localhost" not in base:
            base = "https://" + base[len("http://"):]
        rrd_url = f"{base}/rrd/{name}?key={quote(key, safe='')}"
        return RedirectResponse(f"https://app.rerun.io/version/{RERUN_VERSION}/?url={quote(rrd_url, safe='')}")

    @api.get("/healthz")
    def healthz():
        return {"ok": True}

    return api


# --------------------------------------------------------------------------
# One-time setup: the projection reference
# --------------------------------------------------------------------------

@app.local_entrypoint()
def setup(reference: str = "results/m1/reference_a2_prefill.npz"):
    ref = Path(reference)
    assert ref.exists(), f"no {ref}"
    with volume.batch_upload(force=True) as b:
        b.put_file(str(ref), "/public/reference.npz")
    print("uploaded the projection reference to /public/reference.npz on the Volume")
