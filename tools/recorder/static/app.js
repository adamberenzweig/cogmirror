/* cogmirror M3 recorder. Plain JS, no build step.
 *
 * Flow: setup -> calibration take -> [preview -> take -> review] x N -> done.
 * Every take uploads <take_id>.webm plus <take_id>.json to the local server.
 * All timestamps in the log are seconds since MediaRecorder's `start` event.
 */
"use strict";

const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const params = new URLSearchParams(location.search);
const FAKE = params.get("fake") === "1";           // synthetic stream for testing without a camera
const CALIBRATION_S = Number(params.get("calib") || 15);  // ?calib=60 for a longer one
const HOSTED = !["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);  // modal_app/web.py vs tools/recorder/serve.py
const LEAD_IN_S = 1.5, TAIL_S = 1.5, LISTEN_GAP_MS = 500;

const state = {
  stories: [], progress: null, stream: null, cameraSettings: null, mimeType: null,
  sessionId: null, queue: [], takeIndex: 0, takes: [], current: null, endAfter: false, selfView: false,
};

// ------------------------------------------------------------------ helpers
function show(id) {
  for (const s of document.querySelectorAll(".screen")) s.classList.add("hidden");
  $("#" + id).classList.remove("hidden");
}
function defaultSessionId() {
  const d = new Date(), p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}
function rngFromString(s) {            // mulberry32 seeded from a string hash
  let h = 1779033703 ^ s.length;
  for (let i = 0; i < s.length; i++) { h = Math.imul(h ^ s.charCodeAt(i), 3432918353); h = (h << 13) | (h >>> 19); }
  let a = h >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function shuffle(arr, rnd) {
  for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
  return arr;
}
function waitKey(keys) {
  return new Promise((resolve) => {
    const h = (e) => {
      if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      const k = e.key === " " ? "space" : e.key.toLowerCase();
      if (keys.includes(k)) { e.preventDefault(); window.removeEventListener("keydown", h); resolve(k); }
    };
    window.addEventListener("keydown", h);
  });
}
function fmtClock(s) { s = Math.max(0, Math.floor(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }
function storyLabel(story) {
  // The subject never sees the target emotion during a take; it is in the logs.
  return story ? story.id : "calibration";
}

// ------------------------------------------------------------------- camera
async function startCamera() {
  $("#setup-error").textContent = "";
  if (state.stream) for (const t of state.stream.getTracks()) t.stop();
  const [w, h] = $("#resolution").value.split("x").map(Number);
  let stream;
  if (FAKE) {
    stream = fakeStream(w, h);
  } else {
    const deviceId = $("#camera").value || undefined;
    const constraints = {
      video: { width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: 30 }, deviceId: deviceId ? { exact: deviceId } : undefined },
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    };
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      $("#setup-error").textContent = `Camera failed: ${e.name}: ${e.message}`;
      return;
    }
  }
  state.stream = stream;
  $("#preview").srcObject = stream;
  const vt = stream.getVideoTracks()[0];
  const at = stream.getAudioTracks()[0];
  state.cameraSettings = {
    video: vt ? vt.getSettings() : null, video_label: vt ? vt.label : null,
    audio: at ? at.getSettings() : null, audio_label: at ? at.label : null, fake_stream: FAKE,
  };
  state.mimeType = pickMimeType();
  const vs = state.cameraSettings.video || {};
  $("#camera-info").textContent =
    `${vs.width}×${vs.height} @ ${vs.frameRate ? vs.frameRate.toFixed(1) : "?"} fps   ${state.mimeType}\n` +
    `video: ${state.cameraSettings.video_label || "(synthetic)"}\naudio: ${state.cameraSettings.audio_label || "(none)"}`;
  if (vs.width && vs.width < 1280) $("#setup-error").textContent = `Warning: camera delivers ${vs.width}×${vs.height}; the protocol asks for 1280×720 or better.`;
  await listCameras();
  updateStartButton();
}
function pickMimeType() {
  for (const m of ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp9", "video/webm;codecs=vp8,opus", "video/webm;codecs=vp8", "video/webm", "video/mp4"])
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  return "";
}
async function listCameras() {
  if (FAKE || !navigator.mediaDevices.enumerateDevices) return;
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const sel = $("#camera"), prev = sel.value;
  sel.innerHTML = "";
  for (const d of devs) { const o = document.createElement("option"); o.value = d.deviceId; o.textContent = d.label || `camera ${sel.length + 1}`; sel.appendChild(o); }
  if (prev) sel.value = prev;
}
function fakeStream(w, h) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d"); const t0 = performance.now();
  (function draw() {
    const t = (performance.now() - t0) / 1000;
    ctx.fillStyle = "#334"; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#fc6"; ctx.beginPath(); ctx.arc(w / 2 + Math.sin(t) * w / 4, h / 2, h / 6, 0, 7); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "48px sans-serif"; ctx.fillText(`synthetic ${t.toFixed(1)} s`, 40, 80);
    requestAnimationFrame(draw);
  })();
  return c.captureStream(30);
}

// ------------------------------------------------------------------ queue
function buildQueue() {
  const done = new Set((state.progress ? state.progress.done : []).map(([s, c]) => `${s}|${c}`));
  const allowRepeat = $("#allow-repeat").checked;
  const only = $("#only-condition").value;
  const n = Number($("#n-takes").value) || 12;
  const rnd = rngFromString(state.sessionId);
  let pairs = [];
  for (const story of state.stories) for (const condition of ["read", "listen"]) {
    if (only && condition !== only) continue;
    if (condition === "listen" && !story.audio) continue;
    if (!allowRepeat && done.has(`${story.id}|${condition}`)) continue;
    pairs.push({ story, condition });
  }
  shuffle(pairs, rnd);
  const cluster = (p) => p.story.clusters[0] || "neutral";
  const out = [];
  while (pairs.length && out.length < n) {
    const prev = out[out.length - 1];
    let i = pairs.findIndex((p) => !prev || (p.story.id !== prev.story.id && cluster(p) !== cluster(prev) && p.condition !== prev.condition));
    if (i < 0) i = pairs.findIndex((p) => !prev || p.story.id !== prev.story.id);
    if (i < 0) i = 0;
    out.push(pairs.splice(i, 1)[0]);
  }
  return out;
}
function updateStartButton() {
  const must = [...document.querySelectorAll(".must")].every((c) => c.checked);
  $("#start-session").disabled = !(must && state.stream && state.stories.length);
}

// --------------------------------------------------------------- recording
class Take {
  constructor({ story, condition, takeIndex }) {
    this.story = story; this.condition = condition; this.takeIndex = takeIndex;
    this.takeId = `${String(takeIndex).padStart(3, "0")}_${story ? story.id : "calibration"}_${condition}`;
    this.events = []; this.chunks = []; this.t0 = null; this.blob = null; this.duration_s = null;
  }
  async start() {
    const opts = { videoBitsPerSecond: 8_000_000 };
    if (state.mimeType) opts.mimeType = state.mimeType;
    this.rec = new MediaRecorder(state.stream, opts);
    this.rec.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };
    const started = new Promise((res) => { this.rec.onstart = res; });
    this.startWallclock = new Date().toISOString();
    this.rec.start(1000);
    await started;
    this.t0 = performance.now();
    this.startWallclock = new Date().toISOString();
    $("#rec-indicator").classList.remove("hidden");
    this.clock = setInterval(() => { $("#rec-clock").textContent = fmtClock(this.now()); }, 250);
  }
  now() { return (performance.now() - this.t0) / 1000; }
  stamp(type, extra = {}) { const ev = { t: Math.round(this.now() * 1000) / 1000, type, ...extra }; this.events.push(ev); return ev; }
  async stop() {
    const stopped = new Promise((res) => { this.rec.onstop = res; });
    this.rec.stop();
    await stopped;
    this.duration_s = Math.round(this.now() * 1000) / 1000;
    clearInterval(this.clock);
    $("#rec-indicator").classList.add("hidden");
    this.blob = new Blob(this.chunks, { type: this.rec.mimeType || state.mimeType });
  }
  sentenceTimes() {
    const n = this.story ? this.story.sentences.length : 0;
    const out = [];
    for (let i = 0; i < n; i++) {
      const s = this.events.find((e) => e.type === "sentence_start" && e.sentence_idx === i);
      const e = this.events.find((e) => e.type === "sentence_end" && e.sentence_idx === i);
      out.push({ sentence_idx: i, start: s ? s.t : null, end: e ? e.t : null });
    }
    return out;
  }
  log(extra) {
    const s = this.story;
    return {
      session_id: state.sessionId, take_id: this.takeId, take_index: this.takeIndex,
      story_id: s ? s.id : null, condition: this.condition, kind: s ? s.kind : "calibration",
      emotions: s ? s.emotions : [], clusters: s ? s.clusters : [], topic: s ? s.topic : null,
      story_text: s ? s.text : null, sentences: s ? s.sentences : [], turn: s ? s.turn : null,
      start_wallclock: this.startWallclock, duration_s: this.duration_s,
      timestamps_relative_to: "MediaRecorder start event, seconds",
      sentence_times: this.sentenceTimes(), events: this.events,
      camera: state.cameraSettings, recorder: { mimeType: this.rec.mimeType, videoBitsPerSecond: this.rec.videoBitsPerSecond, timeslice_ms: 1000 },
      video_bytes: this.blob ? this.blob.size : 0, video_file: `${this.takeId}.webm`,
      user_agent: navigator.userAgent, screen: { w: screen.width, h: screen.height, inner_w: innerWidth, inner_h: innerHeight },
      bad: false, note: "", ...extra,
    };
  }
}

function setPrompter(hint, text, progress, small = false) {
  $("#prompter-hint").textContent = hint || "";
  $("#prompter-text").textContent = text || "";
  $("#prompter-text").classList.toggle("small", small);
  $("#prompter-progress").textContent = progress || "";
}

async function runCalibration(take) {
  show("take");
  $("#take-meta").textContent = `${state.sessionId} · take ${take.takeIndex} · calibration`;
  $("#preview-text").classList.add("hidden");
  setPrompter("Calibration", "Look at the lens with a relaxed, neutral face.", `${CALIBRATION_S} seconds. Press SPACE to start.`, true);
  $("#take-instructions").innerHTML = "Breathe normally, blink normally, do not talk. Keep looking at the lens itself, not the screen.";
  await waitKey(["space"]);
  await take.start();
  take.stamp("calibration_start");
  const end = take.now() + CALIBRATION_S;
  while (take.now() < end) { setPrompter("Calibration", "Look at the lens.", `${fmtClock(end - take.now())} remaining`, true); await sleep(200); }
  take.stamp("calibration_end");
  setPrompter("Calibration", "Done.", "", true);
  await take.stop();
}

async function runReading(take) {
  const s = take.story;
  show("take");
  $("#take-meta").textContent = `${state.sessionId} · take ${take.takeIndex} · reading · ${storyLabel(s)}`;
  setPrompter("Reading, silent preview", "Read the whole story below to yourself.", "Press SPACE when you are ready to read it aloud. Recording starts at once with the first sentence.", true);
  $("#preview-text").textContent = s.text;
  $("#preview-text").classList.remove("hidden");
  $("#take-instructions").innerHTML = "Then read aloud, expressively, one sentence at a time. <kbd>space</kbd> shows the next sentence; press it the moment you finish the sentence you are on. Keep your eyes near the lens.";
  await waitKey(["space"]);
  $("#preview-text").classList.add("hidden");
  $("#take-instructions").innerHTML = "<kbd>space</kbd> = next sentence.";
  await take.start();
  const n = s.sentences.length;
  for (let i = 0; i < n; i++) {
    take.stamp("sentence_start", { sentence_idx: i });
    setPrompter("Read aloud", s.sentences[i], `${i + 1} / ${n}`);
    await waitKey(["space"]);
    take.stamp("sentence_end", { sentence_idx: i });
  }
  setPrompter("", "Done. Hold still a moment.", "", true);
  await sleep(TAIL_S * 1000);
  await take.stop();
}

function preloadAudio(story) {
  return story.sentences.map((_, k) => {
    const a = new Audio(`/audio/${story.id}/${k}.m4a`);
    a.preload = "auto";
    const ready = new Promise((res) => { a.addEventListener("canplaythrough", res, { once: true }); a.addEventListener("error", res, { once: true }); setTimeout(res, 4000); });
    a.load();
    return { audio: a, ready };
  });
}

async function runListening(take) {
  const s = take.story;
  show("take");
  $("#take-meta").textContent = `${state.sessionId} · take ${take.takeIndex} · listening · ${storyLabel(s)}`;
  $("#preview-text").classList.add("hidden");
  setPrompter("Listening", "Look at the lens.", "Loading narration…", true);
  $("#take-instructions").innerHTML = "You will hear the story one sentence at a time. Keep looking at the lens the whole time and react naturally. Nothing to press until it ends.";
  const clips = preloadAudio(s);
  await Promise.all(clips.map((c) => c.ready));
  setPrompter("Listening", "Look at the lens.", "Press SPACE to start.", true);
  await waitKey(["space"]);
  await take.start();
  setPrompter("Listening", "Look at the lens.", "", true);
  await sleep(LEAD_IN_S * 1000);
  const n = s.sentences.length;
  for (let i = 0; i < n; i++) {
    const a = clips[i].audio;
    const ended = new Promise((res) => { a.addEventListener("ended", res, { once: true }); a.addEventListener("error", res, { once: true }); });
    const playing = new Promise((res) => a.addEventListener("playing", res, { once: true }));
    try { await a.play(); } catch (e) { take.stamp("audio_error", { sentence_idx: i, error: String(e) }); continue; }
    await playing;
    take.stamp("sentence_start", { sentence_idx: i, audio_duration_s: a.duration });
    $("#prompter-progress").textContent = `${i + 1} / ${n}`;
    await ended;
    take.stamp("sentence_end", { sentence_idx: i });
    if (i < n - 1) await sleep(LISTEN_GAP_MS);
  }
  setPrompter("", "Done. Hold still a moment.", "", true);
  await sleep(TAIL_S * 1000);
  await take.stop();
}

// ------------------------------------------------------------------ review
async function review(take) {
  show("review");
  $("#take-note").value = "";
  $("#upload-status").textContent = "";
  const mb = (take.blob.size / 1e6).toFixed(1);
  $("#review-info").textContent = `${take.takeId}: ${take.duration_s.toFixed(1)} s, ${mb} MB, ${take.events.length} events.` +
    (take.story ? ` Press Enter to keep it, R to mark it bad and redo it.` : ` Enter to keep, R to redo.`);
  $("#keep").focus();
  const choice = await new Promise((resolve) => {
    const onKeep = () => done("keep"), onRedo = () => done("redo");
    const onKey = (e) => { if (e.target.tagName === "INPUT") { if (e.key === "Enter") done("keep"); return; } if (e.key === "Enter") done("keep"); if (e.key.toLowerCase() === "r") done("redo"); };
    function done(v) { $("#keep").removeEventListener("click", onKeep); $("#redo").removeEventListener("click", onRedo); window.removeEventListener("keydown", onKey); resolve(v); }
    $("#keep").addEventListener("click", onKeep); $("#redo").addEventListener("click", onRedo); window.addEventListener("keydown", onKey);
  });
  const bad = choice === "redo";
  const log = take.log({ bad, note: $("#take-note").value.trim() });
  $("#upload-status").textContent = "saving…";
  try {
    const r1 = await fetch(`/api/take/${state.sessionId}/${take.takeId}.webm`, { method: "POST", body: take.blob });
    const r2 = await fetch(`/api/take/${state.sessionId}/${take.takeId}.json`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(log) });
    if (!r1.ok || !r2.ok) throw new Error(`${r1.status} / ${r2.status}`);
    const j = await r1.json();
    $("#upload-status").textContent = `saved ${j.path} (${j.bytes} bytes)`;
  } catch (e) {
    $("#upload-status").textContent = `UPLOAD FAILED: ${e}. The take is lost unless the server comes back; check the terminal.`;
    await sleep(2500);
  }
  state.takes.push({ take_id: take.takeId, story_id: take.story ? take.story.id : null, condition: take.condition, bad, duration_s: take.duration_s });
  return bad;
}

// ---------------------------------------------------------------- session
async function runSession() {
  state.sessionId = ($("#session-id").value.trim() || defaultSessionId()).replace(/[^A-Za-z0-9_.-]/g, "-");
  state.selfView = $("#self-view").checked;
  $("#selfview").classList.toggle("hidden", !state.selfView);
  if (state.selfView) $("#selfview").srcObject = state.stream;
  state.queue = buildQueue();
  state.takeIndex = 0;
  state.takes = [];
  if (!state.queue.length) { $("#setup-error").textContent = "Nothing left to record with these settings (every pair is done; tick 'allow repeats' or change the condition filter)."; return; }

  // Calibration, with redo.
  for (;;) {
    const take = new Take({ story: null, condition: "calibration", takeIndex: ++state.takeIndex });
    await runCalibration(take);
    if (!(await review(take))) break;
  }
  // Stories.
  while (state.queue.length && !state.endAfter) {
    const pair = state.queue.shift();
    const take = new Take({ story: pair.story, condition: pair.condition, takeIndex: ++state.takeIndex });
    if (pair.condition === "read") await runReading(take); else await runListening(take);
    const bad = await review(take);
    if (bad) state.queue.unshift(pair);
  }
  show("done");
  const kept = state.takes.filter((t) => !t.bad);
  $("#done-summary").textContent =
    `session ${state.sessionId}\n${state.takes.length} takes recorded, ${kept.length} kept, ${state.takes.length - kept.length} marked bad\n` +
    `${state.queue.length} queued pairs not recorded\n\n` + state.takes.map((t) => `${t.take_id}  ${t.duration_s.toFixed(1)} s${t.bad ? "  BAD" : ""}`).join("\n") +
    (HOSTED ? `\n\nEach kept take is being processed on the server; results appear on the status page.` : `\n\nfiles: data/recordings/${state.sessionId}/`);
  for (const t of state.stream.getTracks()) t.stop();
}

// ------------------------------------------------------------------- boot
async function boot() {
  if (HOSTED) {
    $("#where").textContent = "Takes are uploaded to the cogmirror server, stored there, and processed by the model; nothing is public.";
    $("#status-link").classList.remove("hidden");
    const done = document.querySelector("#done");
    if (done) done.insertAdjacentHTML("beforeend", '<p><a href="/status">Processing status</a></p>');
  }
  $("#session-id").value = defaultSessionId();
  for (const c of document.querySelectorAll(".must")) c.addEventListener("change", updateStartButton);
  $("#start-camera").addEventListener("click", startCamera);
  $("#camera").addEventListener("change", () => { if (state.stream) startCamera(); });
  $("#resolution").addEventListener("change", () => { if (state.stream) startCamera(); });
  $("#start-session").textContent = `Start session (${CALIBRATION_S} s calibration first)`;
  $("#start-session").addEventListener("click", runSession);
  $("#end-session").addEventListener("click", () => { state.endAfter = true; $("#end-session").textContent = "Ending after this take"; });
  if (FAKE) { for (const c of document.querySelectorAll(".must")) c.checked = true; }
  try {
    const [stories, progress] = await Promise.all([fetch("/api/stories").then((r) => r.json()), fetch("/api/progress").then((r) => r.json())]);
    state.stories = stories; state.progress = progress;
    const total = stories.length * 2, done = progress.done.length;
    const withAudio = stories.filter((s) => s.audio).length;
    $("#progress-summary").textContent = `${stories.length} stories (${withAudio} with narration audio); ${done} of ${total} (story, condition) pairs recorded so far across ${progress.sessions.length} session(s).`;
    if (!stories.length) $("#setup-error").textContent = "No stories found in data/stories/stories.jsonl. Run scripts/generate_stories.py first.";
  } catch (e) {
    $("#setup-error").textContent = `Could not load stories/progress from the server: ${e}`;
  }
  updateStartButton();
  if (FAKE) await startCamera();
}
boot();
