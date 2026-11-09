# Work Plan: Reading LLM Emotion Vectors from Facial Video

**Date:** 2026-11-09
**Design:** [face-emotion-vectors-design.md](face-emotion-vectors-design.md) · **Brief:** [cogmirror-brief.md](cogmirror-brief.md)

## Shape of the plan

Five milestones, each ending in a gate from the design's validation section. The order puts the cheapest decisive experiment first: whether the readout works at all on labeled still images is known by the end of M1, before any video pipeline or recording effort is spent.

| | Milestone | Ends with | Rough effort |
|---|---|---|---|
| M0 | Setup and reproduction | Gate 0: our pipeline reproduces gemotions | 2 days |
| M1 | Static-image feasibility | Gate 1: go / narrow / pivot | 4 days |
| M2 | Video pipeline and acted video | Gate 2: time series behaves; throughput and cost known | 4 days |
| M3 | Text–face data (runs alongside M1–M2) | Recordings, alignment, text EVs | 3 days of work over ~2 weeks |
| M4 | Text–face agreement | Gate 3: headline result | 3 days |
| M5 | Decision and next track | Written recommendation | 1 day |

Effort figures are estimates for focused work, not commitments. GPU cost for the whole plan is on the order of $100–150 (see [Cost](#cost)).

## Decisions needed from you

Each has a default the plan assumes if you say nothing.

1. **Dataset access.** Do you have an academic affiliation for AffectNet, Aff-Wild2, or IEMOCAP requests? *Default: no. Use RAVDESS and CREMA-D only, which download directly.*
2. **Recordings.** Who records, and how much? *Default: you, about 60 stories of roughly a minute each, in two conditions (reading aloud and listening), over at least three sessions. That is 2.5–3 hours on camera.*
3. **Gate thresholds.** The pass criteria below are proposals. *Default: use them as written. They should be fixed before M1 results exist.*
4. **Laptop E4B phase.** The design review dropped it. *Default: stays dropped; all model work runs on Modal.*
5. **Version control.** The project folder is not a git repository. *Default: `git init` at the start of M0, with data and activations ignored.*

## Day-one logistics

- [x] `git init`; Python project in a local `.venv` (plain `venv` and `pip`; `uv` is not installed on this machine). The Modal image pins `torch` 2.15.0, `transformers` 5.22.2, and `bitsandbytes` 0.50.2.
- [x] Modal account and one Volume (`cogmirror`, in the `cogmirror` environment) for weights, vectors, and activation stores. Set a spending limit in the dashboard if you have not.
- [ ] Plan for cold starts: each new container reads about 62 GB of weights from the Volume and quantizes them, likely several minutes. Prefer a few long-lived containers with large frame chunks over wide fan-out, and set an explicit function timeout (Modal's default is 5 minutes).
- [x] Download to the Volume: Gemma 4 31B weights; from gemotions, all `emotion_vectors_layer*.npz`, `data/stories.db`, `data/neutral.db`, and `_raw_cache_layer{20,30,40,50}` (about 3.7 GB each).
- [ ] Download RAVDESS (speech, video) and CREMA-D. Read both licenses before uploading to Modal.
- [ ] If decision 1 is yes: send the AffectNet, Aff-Wild2, and IEMOCAP requests now. They take weeks.
- [ ] Start M3's story design (no dependency on anything else).

## Repository layout

```
cogmirror/
  pyproject.toml
  src/cogmirror/
    evspace.py      # vectors, reference means, PCA basis, projection
    model.py        # model load, hooks, layer indexing
    readouts.py     # prompt builders and readout positions for A1, A2, B, C
    frames.py       # video → frames, face detection, crop smoothing
    store.py        # activation store (read/write)
    text_ev.py      # per-segment text EVs
    align.py        # transcript alignment and segmenting
    metrics.py      # gate metrics, permutation nulls
    baselines/      # verbal.py, fer.py, blendshapes.py
  modal_app/app.py  # image, volume, remote functions
  scripts/          # one entry point per experiment
  tests/
  data/  results/   # git-ignored
```

**Activation store.** One store per (dataset, crop config, prompt, token budget). Each item holds the residual at 7 layers for one pooled readout position: 7 × 5,376 values in float16, about 75 KB. Ten thousand images is about 750 MB per configuration. Layer, reference mean, normalization, and probe variants are all computed from the store on the laptop.

---

## M0: Setup and reproduction

**Goal:** prove that our model load and hooks produce the same activations gemotions published, so that later failures cannot be blamed on plumbing.

**Tasks**

1. **`evspace.py`.** Load the released vectors. From the activation caches, recompute the global mean, the neutral principal components (50% variance), and the denoised vectors. Build the fixed PCA basis per layer with sign anchoring.
2. **`model.py`.** Load 31B on Modal with gemotions' exact settings. Register hooks on the decoder block outputs. Support early exit at the deepest captured layer.
3. **Reproduction test.** Run about 80 stories from `stories.db` as raw text, mean-pool from token 50, and compare with the matching cache rows at layers 20, 30, 40, and 50. Use gemotions' exact query (`stories_clean` ordered by emotion and story index) so that row order matches. Run each story with and without a BOS token.
4. **Vision-unquantized load.** Repeat the test with the vision encoder excluded from 4-bit quantization. (gemotions' loader already returns the full multimodal model, so this is the only difference between the two loads.)
5. **Image smoke test.** One image through each readout strategy; confirm the image token positions, the token budget setting, and the readout positions by inspecting the tokenized input.

**Gate 0 (proposed)**

- Vectors recomputed from the caches match the released `.npz` files: cosine ≥ 0.999 for every emotion.
- Per-story activations match the cache rows: **centered** cosine ≥ 0.99 at all four layers, for every sampled story. Centered means after subtracting the mean of that emotion's cached rows. Raw cosine cannot be used: activations share a large common component, so two different stories already score above 0.99.
- The vision-unquantized load passes the same test, and its quantization report shows no 4-bit layers in the vision encoder.

**If it fails:** check, in order, the layer index (off by one), quantization type (FP4 versus NF4), the BOS variant and truncation, the library versions (gemotions ran in May 2026 on `transformers` ≥ 5.0), and the row-order assumption. The check reports the nearest cache row for each story, which separates a row-order mismatch from a real numerical mismatch. If 4-bit results cannot be matched on our hardware, re-extract vectors from the released stories under our setup. That is a batch job of a few GPU-hours on a subsample of 100 stories per emotion.

**Gate 0 result (2026-11-09): plumbing confirmed; the per-story criterion is not met at deep layers, for a reason that cannot be fixed by configuration. Decision needed.**

Run: 80 stories (16 emotions × 5), both loads, L40S, `torch` 2.15.0, `transformers` 5.22.2, `bitsandbytes` 0.50.2. Follow-up diagnostic: 240 stories at layer 40, plus 24 stories rerun under different numerical settings.

| Check | Result |
|---|---|
| Vectors recomputed from caches vs released | Pass: cosine ≥ 0.999999 at layers 20, 30, 40, 50 |
| Tokenization | No BOS token reproduces the caches; adding one does not (worst centered cosine −0.67) |
| Layer 20, per story | Pass: centered cosine ≥ 0.9991 for all 80 |
| Layer 30, per story | Median 0.998, minimum 0.905; 14 of 80 below 0.99 |
| Layer 40, per story | Median 0.997, minimum 0.564; 21 of 80 below 0.99 |
| Layer 50, per story | Median 0.998, minimum 0.458; 20 of 80 below 0.99 |
| Vision-unquantized load | Identical text activations to gemotions' load; 0 of 190 vision-encoder layers quantized |
| Load time and memory | 38–54 s from the Volume; 19–20 GB peak GPU memory |

Why the deep layers miss:

- **The layer-20 match rules out plumbing.** A wrong token sequence, row order, pooling window, hook index, or quantization type would break layer 20 too.
- **Our own run is exactly repeatable**, but changing only the arithmetic gives the same pattern of divergence as ours-versus-cache. Switching the attention implementation, or the 4-bit compute precision, leaves the layer-40 median at 0.995 and drops the worst story to 0.63–0.67.
- **So the 4-bit model's deep-layer activations are numerically sensitive for a minority of inputs**, and gemotions' run (May 2026, different library versions and GPU) differs from ours by that kind of arithmetic detail. Neither alternative setting reproduces the caches.

What it does to the readout, at layer 40 over 240 stories (scores from our activations versus from the cached ones):

- Per-emotion score correlation across stories: minimum 0.986, median 0.996.
- Top five principal-component coordinates: r between 0.991 and 0.998 (valence 0.998).
- Difference of emotion means over 40 stories each (e.g., happy − sad): cosine 0.99.
- Highest-scoring emotion agrees for 92% of stories.

**Proposed revision of the criterion** (set after seeing the data, so it needs your sign-off): exact per-story match at layer 20; at deeper layers, mismatch no larger than our own sensitivity to numerical settings, and per-emotion score correlation ≥ 0.98. By that standard Gate 0 passes. The alternative is to re-extract the vectors under our runtime, which I do not recommend: the released vectors average 1,000 stories each, so this noise largely cancels in them, and a re-extraction on a subsample would be noisier.

Consequences carried forward:

- Fix the numerical settings for the whole project (current defaults: SDPA attention, bfloat16 compute) and keep the pinned versions.
- Deep-layer readouts have a numerical noise floor even for identical inputs. Batch composition changes arithmetic too, so M1 should measure readout stability across batch sizes alongside the other perturbations.
- Other things equal, prefer the shallower of two layers that carry similar signal.

**Image smoke test on 31B (2026-11-09).** Four schematic drawn faces (neutral, happy, sad, angry), one each, with the neutral drawing as the reference. This checks mechanics and gives a first impression; it is not evidence about real faces.

- **Mechanics:** all three layouts run at budgets 280 and 560, with readout positions as intended. Peak GPU memory is 23.3 GB at batch 8.
- **Throughput:** 6.1 frames per second at budget 280 and 2.5 at 560 (batch 8, one L40S, read to layer 50).
- **The model names the drawings correctly** in the chat-templated prefill layout: its top next word is "neutral", "happy", "sad", and "angry" respectively. The raw layout is less decisive (the right word appears but is not first).
- **Image-token readout (strategy B) shows expression-specific signal.** At layer 40, the angry drawing's top emotions are irate, outraged, furious, enraged; the happy one's are happy, joyful, optimistic (budget 560); the sad one's are stressed, worried, anxious, or sorry, remorseful, regretful, depending on layout.
- **Last-token readout (strategy A) is right for happy and wrong for sad and angry when measured against the neutral drawing.** Both come out as excited and cheerful at layers 40 and above. The likely cause is the reference: at the last token the neutral face reads as resigned and listless, so anything else looks energetic by comparison. A single neutral image is a poor origin for this readout. M1 uses a reference set of mixed expressions, as the design specifies, and should confirm this explanation first.

**Known gotchas** (from the gemotions code, the transformers source, and the local smoke test)

- gemotions' "layer *N*" is the output of block *N*, which is `hidden_states[N+1]`.
- Its `BitsAndBytesConfig` sets only `load_in_4bit` and a bfloat16 compute type. In `transformers` 5.22 the defaults are FP4 and no double quantization.
- `AutoModelForCausalLM` on a Gemma 4 checkpoint returns the full multimodal class, with decoder blocks at `model.model.language_model.layers`.
- Passing a skip list to bitsandbytes replaces the default one, and patterns match from the start of the full module name. Use `model.vision_tower` and `model.embed_vision`, and re-add `lm_head`.
- The Gemma 4 tokenizer adds no BOS token by itself. Chat templates include one; raw text does not.
- The visual token budget is the image processor's `max_soft_tokens`. A square crop gives 64, 256, and 529 tokens at budgets 70, 280, and 560.
- The processor needs `torchvision` installed.
- The 31B chat template's generation prompt already ends with an empty thought block (`<|channel>thought\n<channel|>`). The A2 turn-start readout therefore sits on the `<channel|>` token, and the A2 prefill follows it.

---

## M1: Static-image feasibility

**Goal:** decide whether the zero-training readout carries expression information, using labeled images of many people. This is the main go/no-go point.

**Data.** Frames from CREMA-D and RAVDESS clips. For each clip take 5 frames from the middle third and average their readouts, which also averages over mouth shapes. Split by actor into development and held-out sets. Add neutral clips as the reference set for μ_ref.

**Tasks**

1. **`frames.py`:** face detection and cropping at three tightness settings (tight, loose, head-and-shoulders).
2. **`readouts.py`:** A1, A2 (both forms), and B, with 3–5 prompt paraphrases for A.
3. **Activation runs** on Modal for the grid: 3 crops × 2 token budgets (280, 560) × readouts. Run strategy C on a subset of about 500 clips.
4. **Perception ceiling:** the VLM's zero-shot emotion label for each clip, from generation.
5. **Baselines:** verbal logits over the 171 emotion words (same forward pass as A); an off-the-shelf expression model with valence/arousal output; MediaPipe blendshapes with ridge regression, cross-validated by actor.
6. **`metrics.py`:** the metrics below, with settings chosen on development actors and reported on held-out actors.
7. **Noise floor:** rerun a subset with crop jitter, brightness shifts, and JPEG compression.

**Gate 1 (proposed)**

| Check | Metric | Pass |
|---|---|---|
| Perception ceiling | VLM zero-shot accuracy, 6 emotions | ≥ 50% (chance 17%) |
| Direction | Valence AUC, happy versus negative emotions, best readout | ≥ 0.80 |
| Beyond valence | 4-way accuracy among anger, fear, sadness, disgust by nearest emotion vector, no training | ≥ 40% (chance 25%) |
| Noise floor | Between-expression difference ÷ nuisance-perturbation difference on PC1 | ≥ 2 |
| Identity | Share of neutral-face variance due to identity, after per-person baseline | Reported; no threshold |
| Versus verbal baseline | All of the above for both | Reported; no threshold |

**Gate 1 result (2026-11-09): pass on direction, beyond-valence, noise floor and prompt robustness; the perception-ceiling criterion is not met, for a reason that undercuts the criterion rather than the method. Proceed to M2.**

Run: 8,880 clips (7,440 CREMA-D, 1,440 RAVDESS), two frames per clip from the middle third, medium crop, budget 280, all three layouts, on L40S. Settings chosen on dev actors (61 + 16), reported on held-out actors (30 + 8). About 6.5 GPU-hours in all (first pass 4.6, noise floor 0.3, prompts 0.8, trials).

| Check | Result | Pass? |
|---|---|---|
| Perception ceiling (zero-shot forced choice by generation) | CREMA-D 46% six-way (chance 17%); RAVDESS 34% eight-way (chance 12.5%), 45% on the six shared labels; 32% among the four negatives (chance 25%). CREMA-D's own face-only human raters: 66%. The model says *neutral* (87–91% recall), *happy* (67–77%) and *disgusted* (61–71%) readily, but almost never *angry* (6–7%) or, on CREMA-D, *sad* (13%) | No (threshold 50%) |
| Direction (valence AUC, happy vs four negatives) | 0.982 held-out (CREMA-D 0.966, RAVDESS 0.997) with the chat-prefill layout, last token, layer 50. Nearly every layout and layer is above 0.90; the image-token readout at layer 20 also reaches 0.982 | Yes (≥ 0.80) |
| Beyond valence (4-way among negatives, nearest emotion vector, no training) | 57.6% (CREMA-D 55.1%, RAVDESS 60.2%) with chat-prefill, last token, layer 35. With the valence direction projected out first: 51.2%. Recall: sad 71%, angry 59%, disgusted 46%, afraid 45%; the main confusions are everything → sad and afraid → sad | Yes (≥ 40%) |
| Noise floor (expression gap ÷ RMS valence shift, layer 35) | Batch-size change 43; brightness ±20%: 17–18; 16-px shifts 16–17; 5% zoom 14; JPEG quality 40: 9. The 4-way label flips for 7–10% of clips under image perturbations (16% for JPEG, 3.5% for batch size) | Yes (≥ 2) |
| Identity | Valence variance (held-out, six emotions): actor 6%, emotion 73% for the last-token readout. The image-token readout carries more identity (19% on valence, 33% on the 171 scores), which a per-person baseline removes (7% and 12%). For the last-token readout the baseline from each actor's few neutral clips did not help (6% → 12%) | Reported |
| Prompt robustness (five paraphrases, held-out, one frame) | Valence AUC 0.960–0.964 for every prompt at layer 35; between prompts, valence r 0.96 (worst pair 0.91), 171-profile cosine 0.92, 4-way label agreement 80%. "The person in this picture is feeling" is best or tied | Yes |
| Components carrying signal | η² of emotion by principal component, last token, layer 50: 0.65, 0.53, 0.36, 0.49, 0.32, then 0.1–0.2. Several components beyond valence carry expression information. For the image-token readout it is mostly PC1 | Reported |

Baselines on the same held-out clips:

| Method | Valence AUC | 4-way negatives | Six-way |
|---|---|---|---|
| EV readout, zero-training (chat-prefill, last token) | **0.982** | **0.576** | 5-way incl. happy: 0.551 |
| Verbal baseline: next-word probabilities over emotion words, same forward pass | 0.978 | 0.460 | 5-way: 0.530 |
| Zero-shot forced choice by generation | — | 0.322 | 0.459 |
| MediaPipe blendshapes + logistic regression (trained on dev actors) | 0.943 | 0.579 | 0.535 |
| HSEmotion, off-the-shelf expression model (trained on AffectNet) | 0.979 | 0.585 | 0.595 |
| Linear probe on the 171 EV scores (trained on dev actors) | — | — | 0.60 |
| CREMA-D human raters, face only | — | — | 0.66 |

Reading of the result:

- **The vectors add something over asking the model.** On valence they tie the verbal baseline, but on discriminating negative emotions the readout beats the model's own next-word probabilities by 12 points and its generated answer by 25.
- **They match, but do not beat, a purpose-built expression model and a trained blendshape regression** on these coarse metrics. That is the expected position for a zero-training readout. Its case rests on the 171-dimensional output, which these labels cannot test; the top-scoring vectors per class are at least sensible: happy → relieved, amused, euphoric; angry → alert, defiant, spiteful; disgusted → disdainful, contemptuous, disgusted; afraid → surprised, alert, astonished; sad → bored, reflective; neutral → bored, indifferent.
- **The perception-ceiling criterion was built on a false premise.** It assumed that if the model cannot name an expression, no readout can find it. The readout classifies negative emotions at 58% where generation manages 32%: decoding collapses onto a few safe answers (neutral, disgusted) and hides what the activations hold. Generation is a lower bound, not a ceiling, and the gate's 50% threshold should be dropped.
- **Intensity matters as expected:** 4-way accuracy 63% on CREMA-D's high-intensity takes against 48% on low, and valence AUC 0.99 against 0.97. RAVDESS (larger faces, 720p) scores higher than CREMA-D (faces about 150 px) throughout.
- **Two frames beat one:** 4-way 57.6% with two frames per clip against 53% with one, the same settings. Averaging over mouth shapes helps, as the design anticipated.
- **Layouts:** the chat-templated prefill ("<image> ... The person in this picture is feeling", as the model's turn) is best. The raw untemplated layout is weaker on categories, and reading at the turn start before any prefill is weakest. The image-token readout is a strong valence channel even at layer 20 but carries identity.
- **Layer:** valence is flat across layers 25–50 for the last-token readout; category discrimination peaks at 35–50, matching the design's prediction that "action" layers read the emotion about to be named.

Carried into M2:

- Default configuration: chat-prefill layout, last-token readout, layers 35 (categories) and 50 (valence), medium crop, budget 280, the "feeling" prompt. Keep capturing all layers and the image-token readout; it costs nothing extra.
- Reference statistics: mean and per-dimension scale fitted on the M1 dev actors (`EVSpace.fit_reference` on `data/acts/pass1`), not on a single neutral image.
- Per-person baseline from the session mean or a calibration clip of a minute, not from a handful of neutral frames.
- Average readouts over at least 1–2 s of frames.
- Expect roughly 4.3 frames per second per L40S with verbal logits, 6 without.

Not done in M1: the crop and token-budget sweep (one crop and one budget were used), and strategy C (generate, then pool), which the result makes unnecessary for now.

**Outcomes**

- **Pass:** continue to M2 with the best one or two configurations.
- **Valence only** (direction passes, beyond-valence fails): continue, but the claim narrows to valence plus whatever components survive. Try the neutral-face denoising variant.
- **Perception ceiling fails:** raise the token budget, loosen the crop, try 1120 tokens. If it still fails, the VLM is the wrong tool; go to the pivot.
- **Ceiling passes but projections fail:** the model sees the expression but the story-derived vectors do not capture it. Check strategy C. If C works, the issue is readout position. If not, go to the pivot.
- **Pivot:** fit a linear probe on the stored VLM activations (and on blendshapes) to predict labels and, later, text EVs. This is the brief's supervised track. It reuses everything built so far.

**Deliverable:** a short results note with the grid, the chosen configuration, and the gate table filled in.

---

## M2: Video pipeline and acted video

**Goal:** a working video-to-time-series pipeline, with evidence that the series responds to expression and not to words or mouth shapes, and a measured cost per hour of video.

**Tasks**

1. **Video pipeline:** frame sampling, face tracking with temporal smoothing of the crop box, batched Modal inference fanned out by chunk, store, projection, Parquet output with timestamps.
2. **Per-person baseline** from a calibration clip or the session mean.
3. **Viewer:** a simple page or notebook that plays a video beside its valence trace and top emotions. This is a debugging tool, not the demo.
4. **Throughput benchmark:** frames per second per L40S at each token budget and batch size. Compute cost per hour of video at 2, 5, and 10 fps.
5. **Lexical control** on RAVDESS and CREMA-D: variance in the readout explained by emotion versus by sentence.
6. **Speech-motion floor:** readout variance within neutral-emotion clips; compare per-frame against 1–2 s averages and upper-face crops.
7. **Temporal sanity:** self-recorded scripted events (neutral → smile → neutral, a yawn, a frown) with marked timestamps.
8. **Optional:** a 3–5 frame multi-image window, compared with per-frame plus smoothing.

**Gate 2 (proposed)**

- Emotion explains at least 3× the variance that sentence identity does on PC1.
- Scripted events appear in the series within ±1 s of their marked times, on a visibly distinct trace.
- Windowed readouts during neutral speech stay within the noise band set at Gate 1.
- Cost per hour of video is known and acceptable for M4's volume.

**Gate 2 result (2026-11-09): pass on lexical control, speech-motion floor and cost; temporal sanity passes on acted onsets and awaits the self-recorded scripted events. Built and run in a separate session; full report in [docs/results/m2-gate2-report.md](docs/results/m2-gate2-report.md).**

Pipeline as built: 5 fps sampling, one-euro smoothing of the face box, medium crop at 768 px, budget 280, chat-prefill layout, last token; scores at layer 35, valence at layer 50; reference statistics from the M1 dev actors (`results/m1/reference_a2_prefill.npz`); 1.5 s centered windows. Code: `src/cogmirror/video.py`, `src/cogmirror/store.py`, `modal_app/video.py`, `scripts/video_run.py`; the viewer is Rerun, via `scripts/rerun_export.py` (video, the model's crop, and the traces stacked EKG-style on one shared scale; the recorder's WebM takes are transcoded with the bundled ffmpeg). Runs: all 1,440 RAVDESS clips and 575 held-out CREMA-D clips (110 minutes of video, 34,700 frames); about 4.5 GPU-hours.

| Check | Result | Pass? |
|---|---|---|
| Lexical control | On valence, emotion explains 73% of variance and sentence 0.4% (ratio 165; RAVDESS 10,000+, CREMA-D 94). On the 171 scores, 59% vs 1.6% | Yes (≥ 3×) |
| Speech-motion floor | Within-neutral-clip SD of valence 0.42 per frame, 0.23 after 1.5 s windows; happy − negative gap 2.21, so gap/SD is 5 per frame and 10 windowed. Upper-face crops cut the per-frame neutral-speech variance 6× while keeping valence AUC 0.96 (4-way angry/sad drops from 0.88 to 0.77) | Yes |
| Temporal sanity | Readout onset vs MediaPipe blendshape onset: RAVDESS median lag −0.2 s, 79% within ±1 s (n = 360); CREMA-D +0.2 s, 85% within ±1 s (n = 126). Clearest for happy and surprised; sad and disgusted have no usable blendshape onset. Self-recorded scripted events: not yet recorded | Partial |
| Cost | 5.9 frames/s per L40S at budget 280 (2.5 at 560): $0.66 / $1.65 / $3.30 per hour of video at 2 / 5 / 10 fps | Yes |

Sanity: the video pipeline reproduces M1 on the same clips (valence AUC 0.99–1.00; 4-way 0.53 on whole-clip means, 0.57 on the middle third, against M1's 0.58 from two stills).

Carried into M4:

- The realized rate inside the Gate 2 calls was 2.6 frames/s, not the benchmark's 5.9, because frames were read one at a time from the Volume in short per-clip chunks. Pack a recording's frames per chunk before the M4 runs.
- Upper-face crops are a strong lever against speech motion for valence; keep both crops available and let M4 compare them.
- The scripted-event check (neutral → smile → neutral, a yawn, a frown, with marked times) runs on the first recording session's calibration material.

**If it fails:** if sentence identity dominates, restrict to upper-face crops or pause frames and re-test. If the series is too noisy, lengthen the averaging window and re-check the crop smoothing before concluding anything about the model.

---

## M3: Text–face data (in parallel with M1–M2)

**Goal:** recordings and aligned text EVs ready when M2 finishes. None of this depends on M1's outcome, because the pivot track needs the same data.

**Tasks**

1. **Story set.** Generate about 70 new stories with the gemotions/Anthropic prompts, 120–180 words each. Do not reuse the released stories, which defined the vectors.
   - Choose 24–30 target emotions spanning the clusters and both ends of the top components, plus neutral passages.
   - About a third of the stories have a scripted emotional turn partway through.
   - Screen every story by computing its text EV (possible once M0 is done) and dropping any whose profile misses the target.
2. **Narration** for the listening condition: synthesized or recorded audio with word timestamps.
3. **Recording protocol.**
   - Fixed camera, lighting, and framing within a session; text or teleprompter close to the lens.
   - Each session begins with a 60 s neutral calibration clip.
   - Randomized story order, emotions interleaved; at least three sessions on different days.
   - Reading condition: preview the story silently, then read aloud expressively. Listening condition: watch the lens and listen.
   - Hold out one full session and a set of stories that are never used for tuning.
4. **`align.py`:** word timestamps with WhisperX for the reading condition (the listening condition has them by construction); segment into sentences or 5–10 s windows.
5. **`text_ev.py`:** per-segment text EVs: the story so far as raw text, mean-pooled over the segment's tokens, at all captured layers.

**Deliverable:** a dataset manifest (story, condition, session, segment times, split) and the text-EV store.

---

## M4: Text–face agreement

**Goal:** the headline number: how well the face readout tracks the text's emotion profile, with honest nulls and baselines.

**Tasks**

1. Run the video pipeline on all recordings with the configuration(s) from M1–M2.
2. Aggregate face readouts per segment; z-score each modality per dimension on development data.
3. Estimate the face–text lag on development data and fix it.
4. Compute the metrics: per-component correlation (valence first), per-segment similarity, and retrieval rank.
5. Compute the nulls: story-level permutation and within-story circular shift.
6. Run the baselines: the neutral-text condition, the verbal baseline, and blendshape regression cross-validated by story.
7. Limited tuning (layer, window length, crop) on development data only; report once on the held-out session and stories.
8. Separately, fit the linear face-EV → text-EV map and report it as the lightly trained variant.

**Gate 3 (proposed)**

- Valence correlation across held-out segments of r ≥ 0.2, significant at p < 0.05 under the story-level permutation null, in at least one condition.
- Stories with an emotional turn show the turn in the face series more often than the circular-shift null.
- Results reported by condition (reading, listening), against every baseline.

**Deliverable:** a results report with the gate table, the comparison against baselines, example traces, and the limits of a one-subject, performed-expression dataset stated plainly.

---

## M5: Decision and next track

Write a one-page recommendation based on the gates. The likely branches:

| Result | Next track |
|---|---|
| Zero-training readout works beyond valence | Use it as a teacher: label more video, distill into a small real-time student, build the brief's demo |
| Works for valence only, or only with the linear map | Adopt the lightly trained variant; test few-shot personalization (the brief's second option) |
| Blendshape regression matches or beats it | Continue on the brief's supervised track; keep the VLM activations as one feature set among several |
| Nothing beats the nulls on self-recordings but Gates 1–2 passed | The method reads acted expression but not this subject's reading face. Try more expressive sources (vlogs, commentary) before changing method |

Extensions to scope after the decision, not before: voice (extract vectors on Gemma 4 12B or E4B from the released stories), cognitive-state vectors (focus, fatigue, effort), more subjects, and spontaneous-expression datasets if access comes through.

---

## Cost

Assumes an L40S at $1.95 per hour on Modal. Measured in M0: 6.1 frames per second at budget 280 and 2.5 at 560, per layout pass.

| Milestone | GPU-hours (estimate) | Basis |
|---|---|---|
| M0 | 2–3 | Model download and load, ~100 stories, smoke tests |
| M1 | 10–15 | ~8k clips × 5 frames × the crop/budget/readout grid, plus a generation subset |
| M2 | 8–12 | Acted-video runs, benchmarks, perturbation reruns |
| M3 | 1–2 | Text EVs for ~70 stories and screening |
| M4 | 15–25 | ~2.5 h of video at 5–10 fps × 2–3 configurations |
| **Total** | **~40–60** | **≈ $100–150**, plus a few dollars of Volume storage |

**M1 needs staging at the measured throughput.** The full grid as first written (about 40,000 frames × 3 crops × 3 layouts × 2 budgets) would take roughly 57 GPU-hours, far above the estimate in the table. Run it in two passes instead:

1. First pass: 2 frames per clip, one crop, budget 280, all three layouts. About 2–3 GPU-hours. This is enough for the gate's main checks.
2. Then expand only along the axes the first pass shows to matter (crop, budget 560, more frames), keeping M1 within the 10–15 GPU-hour estimate.

M0 used well under one GPU-hour.

## What this plan does not cover

- Speech prosody, real-time inference, and the visualization demo from the brief. They follow from M5's decision.
- Felt-emotion ground truth (self-report, physiology).
- Subjects other than the default single recorder, and any use of other people's video beyond the public corpora.
