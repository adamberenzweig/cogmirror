# Design: Reading LLM Emotion Vectors from Facial Video

**Status:** Draft for prototype, reviewed
**Date:** 2026-11-09 (reviewed same day; see [Review notes](#review-notes-2026-11-09) for what changed)
**Work plan:** [face-emotion-vectors-workplan.md](face-emotion-vectors-workplan.md)

## Goal

Produce a high-dimensional, per-frame time series describing a person's apparent affective and cognitive state from video. The method feeds face frames to an open vision-language model (VLM) and projects its internal activations onto pre-extracted emotion vectors. No new model is trained.

## Scope relative to the brief

The [brief](../cogmirror-brief.md) proposes training a model to predict the text's emotion vector (EV) from facial and vocal features. This design tests a cheaper route first, the brief's third architecture option: read the EVs straight out of an existing VLM, with no training.

- The brief's core idea survives here as the **evaluation** (text–face agreement) and as the **fallback**. The blendshape-regression "baseline" below is the brief's original proposal in its simplest form. If it beats the zero-training readout, that is a result, and the project continues on that track with the same data and metrics.
- Out of scope for this design: speech prosody, real-time inference, and the demo. Two consequences are worth knowing now:
  - Gemma 4 31B has no audio input. Only E2B, E4B, and 12B do. Adding voice later means extracting vectors on one of those models.
  - A 31B model will not run in real time on a laptop. The real-time path is to use this readout as a teacher and distill it into a small model.

## Non-goals

- Deception or microexpression detection.
- Claims about felt emotion. Like any annotator-style signal, the readout reflects how the expression would be *described*. Linking it to felt state requires separate ground truth (self-report, physiology, task performance).
- Real-time performance in the first prototype.

## Background

**Anthropic, *Emotion Concepts and their Function in a Large Language Model* (April 2026).** Found 171 linear emotion representations in Claude Sonnet 4.5, organized mainly along a valence axis (PC1, 26% of variance), which causally influence behavior. No code or vectors were released. Three findings shape this design:

- The vectors are **locally scoped**. They track the "operative" emotion at a token position (whatever is relevant to the local context and to predicting the next tokens), not a persistent state of a speaker.
- **Layer roles differ.** Early-middle layers encode the emotional connotation of the present content ("sensory"). Middle-late layers encode the emotion relevant to upcoming tokens ("action").
- The model keeps **distinct representations for the present speaker's and the other speaker's emotions**. The story-derived vectors are closest to the former.

**gemotions** (dejanai/gemotions) replicates the method on `google/gemma-4-31B-it`, 60 layers, hidden size 5,376. Details verified from the released code:

- **Extraction input is raw text.** Stories are tokenized with no chat template, truncated to 512 tokens, and the activation is the mean over token positions from the 50th onward.
- **No BOS token.** The code calls the bare tokenizer, which for Gemma 4 adds none. The reproduction check confirmed it: only the no-BOS variant matches the caches.
- **Hook point** is the output of decoder block *N* (`model.model.language_model.layers[N]`). In `output_hidden_states` indexing that is `hidden_states[N+1]` (confirmed on E2B).
- **Model load** is `AutoModelForCausalLM` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")`. The quantization type is left at the library default (FP4, not the more common NF4) with no double quantization.
- **That load is already the full multimodal model.** For Gemma 4 checkpoints `AutoModelForCausalLM` resolves to the same class as `AutoModelForMultimodalLM`, so there is one load path, not two. By default the 4-bit conversion covers the vision encoder too.
- **Vector math:** per-emotion mean minus the global mean over all stories, then project out the top principal components of neutral-dialogue activations that explain 50% of variance.
- **Released:** final vectors for layers 5–55 in steps of 5, the 171,000 stories, the 1,200 neutral dialogues, and the **per-story activation caches** (1,000 × 5,376 per emotion) plus neutral activations for each layer. Layer 15's cache is incomplete. The caches let us recompute the global mean and the neutral components exactly, fit alternative probes without rerunning stories, and test our own pipeline against known activations.
- **Not released:** the global mean and neutral components as files (recompute from the caches), and any vectors for E4B (the code supports E4B, but only 31B results are published).
- Their own validation scores new text by cosine similarity between the uncentered activation and each vector.
- Valence (PC1) explains 32–39% of variance at every released layer. PC2 does not map cleanly to arousal.

**Gemma 4.** 31B IT accepts text and images, and video as sampled frames (up to 60 s at 1 fps). The visual token budget per image is configurable: 70, 140, 280, 560, or 1120. Video frames appear to default to 70 tokens each, which is coarse for a face. Weights are Apache 2.0 and not gated.

**Gemma-4-E4B** (42 layers, about 8.0B parameters). *Where Do Models Find Happiness?* (June 2026) extracts emotion vectors and releases code. Whether the vectors themselves are released is unconfirmed. Valence peaks at layer 16 and collapses in later layers, and the valence direction rotates across layers. Layer choices do not transfer between E4B and 31B.

## Approach

### Pipeline

0. **Reproduction check (once).** Run a sample of the released stories through our own model load and compare with the released activation caches. This verifies the layer index, the module path, and the quantization settings before any image work.
1. **Frame extraction.** Sample video at a target rate. When the subject is speaking, sample densely enough to average over mouth shapes (see [Speech motion](#caveats)).
2. **Face crop.** Detect and crop the face (e.g., MediaPipe face detector) with a fixed margin, then resize. Smooth the crop box over time: detector jitter otherwise shows up as noise in the readout. Crop tightness is a tuned parameter, because tightly aligned crops may be less natural for the VLM than head-and-shoulders.
3. **Forward pass.** Run each crop through Gemma 4 31B IT with a short text prompt. Generation is not needed. Stop at the deepest layer being read.
4. **Activation capture.** Hook the residual stream at **all candidate layers in one pass** (20–50 in steps of 5) and store the activations at the readout positions. Layer choice, projection variants, and normalization are then swept offline at no GPU cost.
5. **Projection.** See [Projection and normalization](#projection-and-normalization).
6. **Output.** A per-frame 171-dimensional vector plus coordinates in a fixed PCA basis. Store as Parquet or NPZ with timestamps.

### Projection and normalization

The score for emotion *e* on activation *x* is

`score_e(x) = (x − μ_ref) · v_e / ‖v_e‖`

where *v_e* is the released (denoised) vector.

- **The neutral-component projection needs no extra step.** Each *v_e* is already orthogonal to those components, so projecting them out of *x* changes nothing.
- **The reference mean μ_ref is the real design choice.** gemotions uses the mean over story activations. A face-frame activation sits far from that mean (different modality, prompt, and token position), so with the story mean every frame gets the same large per-emotion offset. Absolute scores and "top emotion" rankings would reflect the offset, not the face.
  - **Default:** μ_ref is the mean activation over a reference set of faces (many identities, mixed expressions) under the same prompt, layer, and readout position. Divide each dimension by its standard deviation on that set.
  - **Per-person baseline:** for time series, also subtract the person's own mean over a calibration clip or the session. A resting face that reads as "sad" is otherwise a constant error.
- **PCA basis.** Compute it once from the 171 vectors at each layer and keep it fixed. Anchor the sign of each component with known emotions (e.g., `happy` positive on PC1). "PC1 = valence" was checked by gemotions at layer 40; verify it at other layers.
- **Variant, reported separately: neutral-face denoising.** Project out the top components of activations on neutral faces across identities. This is the image-domain analogue of the neutral-dialogue step. It uses no emotion labels, but it is an adaptation.

### Readout strategies

The vectors were built from raw narrative text, mean-pooled from token 50. Every readout here differs from that in some way, so the comparison is which differences matter.

| Strategy | Where to read | Pros | Cons |
|---|---|---|---|
| **A1. Raw completion** | Last tokens of an untemplated prompt after the image, e.g. "The person in this picture is feeling" | No chat template, as in extraction; the next token is an emotion word | Prompt wording may bias the readout; untemplated image input is unusual for an instruction-tuned model |
| **A2. Chat-templated** | The same text as a prefilled model turn, or a user question read at the start of the model turn | The model's native multimodal format | The chat format was never seen during extraction |
| **B. Image-token** | Mean over image token positions | Direct test of a shared vision–language emotion subspace; no prompt bias | Image tokens may not fall in the text-derived subspace |
| **C. Generate, then pool** (diagnostic) | Generate a short description of the person's state; mean-pool over the generated tokens | Closest to the extraction distribution | Slow; describes rather than reads. Use on subsets to separate "wrong readout position" from "the VLM does not see the expression" |

Expectations from the layer findings above: A-style readouts should peak in middle-late layers, where the operative emotion is the one about to be named. B, if it works at all, should peak earlier. Sweep layers per strategy.

A-style readouts are close to a vector-space form of the model's zero-shot verbal answer. Two consequences:

- The **verbal baseline** (next-token logits over the 171 emotion words at the same position, from the same forward pass) is the comparison that shows whether the vectors add anything over asking the model.
- Agreement between the projection and the model's own description is expected. It is a sanity check, not evidence.

Start with A1 and A2; run B as an experiment and C as a diagnostic.

*Result (M1, acted corpora): A2 with the prompt prefilled as the model's turn is best (valence AUC 0.98, 58% among four negative emotions with no training). A1 is weaker on categories; reading at the turn start before any prefill is weakest. B gives valence as strong as A2 even at layer 20 but carries more identity information, which a per-person baseline removes. Category discrimination peaks at layers 35–50, as predicted for "action" layers.*

### Frame vs. clip mode

- **Per-frame (default):** independent passes produce a clean time series and parallelize trivially.
- **Clip mode:** use the model's native video input (≤ 60 s at 1 fps). Frames inform each other, so dynamics are captured, but the readout reflects the whole clip, and the low per-frame token budget loses facial detail. A middle option is a short multi-image window (3–5 frames at the full budget).

## Configuration defaults

| Parameter | Default | Notes |
|---|---|---|
| Model | `google/gemma-4-31B-it`, bitsandbytes 4-bit | Match gemotions exactly: default quant type (FP4), bfloat16 compute, no double quantization |
| Vision encoder | Unquantized | gemotions never ran the vision path, so nothing needs matching. Keep the vision encoder and projector out of 4-bit quantization and confirm that it helps |
| Layers captured | 20–50 in steps of 5 | Block outputs, as in gemotions. Choose per strategy offline |
| Visual token budget | 280 | Set with the image processor's `max_soft_tokens`. It is an upper bound: a square crop gives 256 tokens (768 px) at 280 and 529 tokens (1,104 px) at 560. Smaller crops are upscaled, so 560 only helps when the source face is larger than 768 px |
| Frame rate | 2–5 fps silent; denser when speaking | Bounded by throughput |
| Prompt (A) | "The person in this picture is feeling" | Test 3–5 paraphrases |
| Reference mean | Face reference set, plus per-person baseline | Not the story mean |

## Performance

- One partial forward pass per frame, with no decoding.
- Batch frames, and run multiple model replicas for higher frame rates.
- Cache activations once per (crop, prompt, token budget). Everything downstream is offline.
- Benchmark frames per second on the target GPU before choosing a sampling rate. A rough expectation for 31B at 4-bit with about 300 tokens per frame is a few frames per second per L40S. That is an estimate, not a measurement.

## Compute plan

Hosted Gemma APIs don't expose activations, so we run the weights ourselves.

**Laptop (MacBook M4, 24 GB): everything that is not the 31B forward pass.**
- Frame extraction, face cropping, transcript alignment, and all offline analysis on cached activations (projection, sweeps, metrics, plots).
- A smoke test of the hook code on a small Gemma 4 (E2B). This does not need emotion vectors. E2B does not fit on this laptop's GPU in half precision, so it runs on CPU, which is enough for a handful of forward passes.
- The earlier plan to iterate on E4B first is dropped as a science step, for three reasons:
  - No E4B vectors are published, so we would have to extract them.
  - E4B's emotion geometry differs by layer from 31B's, so the best layer, prompt, and readout would not transfer.
  - At about 8.0B parameters it needs roughly 16 GB in half precision, which is tight on 24 GB.
- E4B or 12B becomes relevant again for audio.

**Modal: all 31B work, from the first milestone.**
- Model: `google/gemma-4-31B-it`, bitsandbytes 4-bit, matching gemotions.
- GPU: L40S (48 GB) as the default, with room for batching. Use A100/H100 80 GB only if running unquantized (which requires re-extracting the vectors).
- Package the pipeline as a Modal function: one video (or chunk of frames) per call, fanned out across containers. Cache model weights, gemotions vectors, and the activation caches for the layers in use (about 3.7 GB per layer) in a Modal Volume.
- Batch jobs: readout and crop sweeps, the validation sets, and, if needed, re-extracting vectors from the released stories.
- Setup: new Modal account for this project. The Gemma weights are not gated; a Hugging Face token only helps with rate limits.

**Numerical sensitivity (found in the reproduction check):** our activations match the released caches exactly at layer 20, but at layers 30–50 a minority of stories diverge. The same divergence appears between two of our own runs that differ only in attention implementation or compute precision. Emotion scores are barely affected (per-emotion correlation ≥ 0.986 at layer 40), but deep-layer readouts carry a numerical noise floor. Keep numerical settings and library versions fixed, and prefer the shallower of two equally informative layers. Details are in the work plan.

**Quantization consistency:** the reproduction check (pipeline step 0) decides this empirically. If our activations do not match the released caches, fix the load configuration before anything else. If the setup must differ (e.g., unquantized), re-extract the vectors under that setup.

## Validation

Validation runs as a sequence of gates, cheapest first. Each gate has a pass criterion fixed before the results are seen; the criteria are in the work plan.

### Gate 0: reproduction

Our pipeline, on released stories, reproduces the released activation caches and vectors.

### Gate 1: static images (decides whether the approach is viable)

On labeled expression images covering many identities:

1. **Perception ceiling.** The VLM's zero-shot expression recognition (by generation or logits). *Result (M1): this is a lower bound, not a ceiling. Generation named negative expressions at 32% where the readout reached 58%; decoding collapses onto a few safe answers.*
2. **Direction check.** PC1 (valence) moves in the expected direction; report AUC for positive versus negative expressions.
3. **Beyond valence.** After removing valence, does the readout separate same-valence emotions (anger, fear, sadness, disgust)? This tests the claim that the output is usefully high-dimensional. Report how many components carry signal.
4. **Noise floor.** Readout change under nuisance perturbations (crop jitter, brightness, compression) compared with the change between expressions. If jitter moves the readout as much as a smile does, the time series will be noise.
5. **Identity confound.** Neutral faces from different people should give similar readouts. Decompose variance into identity (and age, gender, skin tone where labels exist) versus expression, before and after per-person baselining.
6. **Prompt robustness.** Readouts are stable across prompt paraphrases (strategy A).
7. **Baselines on the same images:** the verbal baseline, an off-the-shelf facial-expression model with valence/arousal outputs, and blendshape regression.

### Gate 2: acted video

On acted corpora with fixed sentences (RAVDESS, CREMA-D):

- **Temporal sanity:** the time series responds at expression onsets and on self-recorded known events (a smile, a yawn).
- **Lexical control:** the same sentence spoken with different emotions gives different readouts; different sentences with the same emotion give similar ones. This is the direct test for lip-reading leakage.
- **Speech-motion floor:** variance of the readout during neutral speech.
- **Human ceiling:** CREMA-D includes crowd ratings of perceived emotion from video alone.

### Gate 3: text–face agreement (the project's headline metric)

Record or collect video of people encountering a known text. Compare the EV readout from the **face** with the EV readout from the model processing the **text itself**, segment by segment. Matching the text's emotion profile from the face alone gives a quantitative target that is specific to this project.

Use it to choose among settings that already passed gates 1 and 2, not as the only tuning signal: the labeled corpora have far more identities and independent samples.

**Data sources**
- **Self-recordings.** Controlled conditions, known text, and no consent issues, but one subject and performed rather than spontaneous expression. Use for development. Protocol points:
  - **Stories:** freshly generated with the gemotions/Anthropic prompts (not the released stories, which defined the vectors). Include stories with a scripted emotional turn partway through, so the target varies within a take.
  - **Conditions:** *reading aloud* (needed later for prosody; preview each story silently first) and *listening* to a narration of the story (no mouth motion, exact timing, more reactive expression).
  - **Design:** about 60 or more stories spread over valence and arousal plus neutral, in randomized order, interleaved across several sessions. A weak correlation needs many independent units, and the independent unit is the story, not the segment.
  - **Setup:** text near the camera; fixed lighting and framing within a session.
- **Existing corpora with transcripts:** CMU-MOSEI (YouTube monologues), IEMOCAP (actors, scripted and improvised), MELD (TV dialogue). Many speakers, plus emotion and sentiment labels for a secondary check. Access takes time: IEMOCAP needs a signed license, and several are academic-only.
- **Web video** (game commentary, unboxings, drama): the most naturalistic, but needs transcription, speaker detection, and handling of shot cuts.

**Procedure**
1. Get word-level timestamps for the transcript (forced alignment, e.g. WhisperX; exact by construction in the listening condition). Split into segments (sentences or ~5–10 s windows).
2. **Text EV:** run each segment's text, with the preceding story as context, as raw text through the model. Mean-pool over the segment's tokens at the same layers. This matches the extraction setup.
3. **Face EV:** aggregate per-frame face readouts over the segment's time span. The face-only input must contain no transcript in the prompt and no audio.
4. **Normalize each modality separately** (z-score per dimension on development data) before comparing. The two readouts have different offsets and scales, and unnormalized cosine similarity mostly measures those.
5. **Lag:** expression can lead or trail the words. Estimate one lag on development data and fix it.
6. **Metrics:**
   - Per-dimension correlation across segments along the top principal components (valence first). This is the primary number.
   - Per-segment similarity between the normalized face and text EVs (cosine or Spearman), averaged.
   - Retrieval: given a face segment, rank its true text segment among the others.
7. **Null distributions** must respect autocorrelation. Segments within a story are not independent, and anything that drifts across a take (lighting, posture, fatigue) is confounded with that story's emotion. Use story-level permutation and within-story circular shifts. A naive segment shuffle overstates significance.
8. **Baselines:** the nulls above, the same speaker reading neutral text, the verbal baseline, and blendshape regression (cross-validated by story).

### Later

Relate readouts to ground truth (experience sampling, task performance, physiology) using per-person baselines.

### Caveats

- **The text's emotion is not the speaker's emotion.** A reader can enjoy narrating a sad story, and an actor portrays a character. The agreement ceiling is well below 1, and lower for read text than for self-expression (e.g., commentary or vlogs). Report results by source type.
- **Speech motion confound:** mouth movements from talking dominate the lower face, and a single frame captures a mouth shape, not an expression (an "ee" resembles a smile). Aggregate over at least 1–2 s, ablate with upper-face-only crops and with frames from pauses, and compare against the listening condition.
- **Semantic leakage:** visible mouth shapes could in principle carry word information. The lexical control in gate 2 tests this.
- **Overfitting:** tuning on any one metric will overfit the chosen settings. Hold out speakers, stories, and sessions, and confirm improvements across gates.
- **Linear map variant:** a cheap linear map from face EVs to text EVs, fit on held-in data, would likely raise agreement substantially. It is a light form of training; keep it separate from the zero-training result.

## Risks and caveats

- **The VLM may not see subtle expressions.** Vision encoders trained for semantic alignment are coarse on small facial movements. Acted, high-intensity expressions may read well while the low-intensity expressions of someone at a screen (the brief's use case) do not. Gate 1's perception ceiling and noise floor measure this directly. This is the main scientific risk.
- **Subspace mismatch:** the text-derived vectors may not transfer to image inputs. Strategy B tests this directly; strategy C separates it from readout-position effects.
- **The vectors may add nothing over asking the model.** If the verbal baseline matches the projection on every metric, the method reduces to zero-shot classification with extra steps. *M1: they tie on valence and beat the verbal baseline by 12 points on negative-emotion discrimination; they match, but do not beat, an off-the-shelf expression model and a trained blendshape regression on coarse labels.*
- **Readout offset:** without a face-domain reference mean, scores reflect the prompt and modality, not the face.
- **Perceived vs. felt:** the output is an observer-style description, not internal state.
- **Low expressivity:** people alone at a screen show little expression, so expect weak signals and the need for per-person baselines.
- **Demographic bias:** both the face detector and the VLM may perform unevenly across groups.
- **Engineering:** 4-bit loading quantizes the vision encoder unless it is explicitly skipped, and skip patterns must be full module paths (`model.vision_tower`, not `vision_tower`). The reproduction check confirms the language model is unaffected by the skip.
- **Dataset access:** the best spontaneous-expression and valence-annotated datasets (AffectNet, Aff-Wild2, IEMOCAP) need signed or academic requests that can take weeks. RAVDESS and CREMA-D are directly downloadable; plan the first gates around them.
- **Licensing, consent, and regulation:** gemotions uses a link-attribution license. Obtain consent for any video of people. Check dataset terms before uploading third-party video to cloud compute. Emotion inference from faces is regulated in some settings (e.g., workplace and education under the EU AI Act), which matters for any use beyond self-directed research.

## Open questions

- Which layer gives the best face signal for each readout strategy? Is it the same as for text?
- Does clip mode or a short multi-frame window add information beyond per-frame readouts plus temporal smoothing?
- Would additional vectors for cognitive and energy states (focus, fatigue, effort) extracted with the same story method help? The Fatigue and Alertness clusters already exist.
- Is 31B the right long-term model, given that voice needs E4B or 12B? Extracting vectors on 12B from the released stories is a modest batch job.
- If the readout works, what is the smallest student model that reproduces it in real time?

## Review notes (2026-11-09)

Changes from the first draft, most consequential first:

1. **Validation reordered into gates.** A static-image feasibility gate now comes before any video or recording work. Text–face agreement stays the headline metric but is no longer the first or only tuning signal.
2. **Reference mean.** "Apply the gemotions preprocessing" was not well defined for image inputs. The projection is now written out, with a face-domain reference mean and a per-person baseline.
3. **Missing baseline added.** The model's own next-token logits over emotion words, which shows whether the vectors add anything. Also an off-the-shelf expression model.
4. **Text–face agreement statistics fixed.** Per-modality normalization, autocorrelation-aware nulls, lag handling, stories with emotional turns, a listening condition, and a story count sized to the effect.
5. **Readout strategies expanded.** The vectors come from raw untemplated text, mean-pooled from token 50, so the chat-template choice is now explicit (A1 versus A2), with a generate-then-pool diagnostic (C).
6. **Reproduction check added**, made possible by the released per-story activation caches.
7. **Laptop E4B phase dropped as a science step** (no published vectors, different layer behavior, tight memory). All 31B work moves to Modal from the start.
8. **Exact gemotions settings recorded** (FP4 default quantization, block-output hook index, raw text input), along with what is and is not released.
9. **New tests:** beyond-valence, noise floor, and a lexical control using RAVDESS/CREMA-D in place of custom mismatched readings.
10. **Scope section added**, tying the design to the brief and noting the consequences for audio and real time.
11. **Corrections:** Gemma weights are not gated; 31B has no audio input.

## References

- Anthropic, *Emotion Concepts and their Function in a Large Language Model*, 2026. https://transformer-circuits.pub/2026/emotions/index.html
- gemotions. https://huggingface.co/dejanai/gemotions
- Gemma 4 31B IT model card. https://huggingface.co/google/gemma-4-31B-it (also NVIDIA: https://docs.api.nvidia.com/nim/reference/google-gemma-4-31b-it)
- *Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs*, 2026. https://arxiv.org/abs/2606.26987
- RAVDESS (24 actors, fixed sentences, 8 emotions; CC BY-NC-SA). https://zenodo.org/record/1188976
- CREMA-D (91 actors, 12 sentences, 6 emotions, crowd ratings by modality). https://github.com/CheyneyComputerScience/CREMA-D
