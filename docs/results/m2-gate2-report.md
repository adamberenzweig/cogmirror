# Gate 2 results: acted video (M2)

Date: 2026-11-09. Pipeline: 5 fps sampling, one-euro box smoothing, medium crop at 768 px, budget 280, chat-prefill layout, last token; scores at layer 35, valence (PC1) at layer 50; reference from M1 dev actors; 1.5 s centered windows; no per-person baseline unless stated.

## Gate table

| Check | Criterion | Result | Pass? |
|---|---|---|---|
| Lexical control | η²(emotion) ≥ 3 × η²(sentence) on valence, held-out actors | emotion 0.729 vs sentence 0.004 (ratio 165); ravdess 10512×; crema 94×; 171 scores: 0.591 vs 0.016 | Yes |
| Speech-motion floor | Windowed neutral-speech readout within the Gate 1 noise band: happy − negative gap ≥ 2 × within-clip SD | within-neutral SD per frame 0.42, windowed 0.23; gap 2.21 → gap/SD 5 (frame), 10 (windowed); between-emotion var of clip means 0.71 vs within-neutral 0.053 | Yes |
| Temporal sanity (acted onsets) | Readout onset within ±1 s of the blendshape onset; scripted self-recorded events within ±1 s | ravdess: lag median -0.20 s, 79% within ±1 s (n=360); crema: lag median +0.20 s, 85% within ±1 s (n=126). Scripted events: not yet recorded | Partial |
| Cost per hour of video | Known and acceptable for M4 | 5.91 frames/s per L40S at budget 280, batch 8 → $0.66 / $1.65 / $3.3 per hour of video at 2 / 5 / 10 fps | Yes |

## Runs

- **ravdess**: 1440 clips (480 held-out actors' clips), 27148 frames at 5.0 fps, 86.6 min of video; face detected on 100.0% of sampled frames, box held through 0.0%; 1440 clips with activations.
- **crema**: 575 clips (575 held-out actors' clips), 7536 frames at 5.0 fps, 23.5 min of video; face detected on 100.0% of sampled frames, box held through 0.0%; 575 clips with activations.

## Sanity: M1 metrics from the video pipeline (clip means of 1.5 s windows, held-out actors)

| Dataset | Valence AUC | 4-way negatives | 5-way | Valence AUC, middle third | 4-way, middle third | n |
|---|---|---|---|---|---|---|
| ravdess | 1.000 | 0.535 | 0.619 | 0.999 | 0.594 | 480 |
| crema | 0.991 | 0.521 | 0.571 | 0.978 | 0.557 | 575 |
| both | 0.996 | 0.527 | 0.590 | 0.989 | 0.572 | 1055 |

M1 (two still frames per clip from the middle third): valence AUC 0.982, 4-way 0.576. Whole-clip means include the neutral lead-in and tail of each take; the middle third matches M1's sampling.

## Lexical control (held-out actors)

η² = share of variance explained by the grouping; clip-level values are means of the windowed series.

| Dataset | Signal | η² emotion | η² sentence | ratio | η² actor |
|---|---|---|---|---|---|
| ravdess | valence (clip) | 0.801 | 0.000 | 10511.7 | 0.037 |
| ravdess | valence, actor-baselined | 0.831 | 0.000 | 10511.7 | — |
| ravdess | valence (frames) | 0.708 | 0.000 | 22995.8 | — |
| ravdess | 171 scores (clip) | 0.687 | 0.000 | 3815.4 | 0.085 |
| ravdess | 171 scores, actor-baselined | 0.751 | 0.000 | 3815.4 | — |
| crema | valence (clip) | 0.688 | 0.007 | 93.8 | 0.103 |
| crema | valence, actor-baselined | 0.767 | 0.008 | 90.4 | — |
| crema | valence (frames) | 0.592 | 0.006 | 98.1 | — |
| crema | 171 scores (clip) | 0.557 | 0.012 | 47.1 | 0.179 |
| crema | 171 scores, actor-baselined | 0.680 | 0.015 | 46.3 | — |
| both | valence (clip) | 0.729 | 0.004 | 165.0 | 0.068 |
| both | valence, actor-baselined | 0.777 | 0.004 | 208.4 | — |
| both | valence (frames) | 0.649 | 0.004 | 180.4 | — |
| both | 171 scores (clip) | 0.591 | 0.016 | 37.0 | 0.135 |
| both | 171 scores, actor-baselined | 0.685 | 0.006 | 113.2 | — |

- ravdess: 480 clips, 8 emotions, 2 sentences; η² of sentence within each emotion, mean 0.006.
- crema: 575 clips, 6 emotions, 12 sentences; η² of sentence within each emotion, mean 0.064.
- both: 1055 clips, 8 emotions, 14 sentences; η² of sentence within each emotion, mean 0.105.

## Speech-motion floor

Valence in reference-SD units (scale from the M1 dev actors). Within-clip variance is the mean over neutral clips.

| Dataset | Neutral clips | Within-clip var, per frame | Within-clip var, 1.5 s windows | Between-emotion var of clip means | Happy − negative gap | Gap / within SD (frame) | Gap / within SD (windowed) |
|---|---|---|---|---|---|---|---|
| ravdess | 96 | 0.1647 | 0.0593 | 0.874 | 2.41 | 5.9 | 9.9 |
| crema | 95 | 0.1964 | 0.0472 | 0.454 | 1.80 | 4.1 | 8.3 |
| both | 191 | 0.1805 | 0.0533 | 0.713 | 2.21 | 5.2 | 9.6 |

### Upper-face crops (held-out RAVDESS subset: neutral, and strong happy/angry/sad)

| Crop | Clips | Within-neutral var, per frame | Within-neutral var, windowed | Between-emotion var | Gap / within SD (frame) | Gap / within SD (windowed) | Valence AUC | 4-way (angry/sad) |
|---|---|---|---|---|---|---|---|---|
| medium | 128 | 0.1495 | 0.0480 | 1.136 | 6.4 | 11.3 | 1.000 | 0.875 |
| upper | 128 | 0.0234 | 0.0056 | 0.437 | 10.0 | 20.3 | 0.961 | 0.766 |

## Temporal sanity: expression onsets

Readout onset: first time the windowed signal (valence for happy, the target emotion's score otherwise) exceeds the mean of the clip's first two frames by 2× the within-neutral-clip SD of that signal. Blendshape onset: first time the mean of the listed MediaPipe blendshapes rises halfway from its first-two-frame level to its clip maximum (clips with a rise under 0.1 have no blendshape onset). Lag = readout onset − blendshape onset.

### ravdess

| Emotion | Signal | Blendshapes | Clips | Readout onset found | Blendshape onset found | Both | Lag median (s) | IQR | within ±0.5 s | within ±1 s | Starts expressive |
|---|---|---|---|---|---|---|---|---|---|---|---|
| happy | valence | mouthSmileLeft, mouthSmileRight | 192 | 0.57 | 0.73 | 105 | 0.00 | -0.40 .. +0.40 | 0.62 | 0.76 | 0.22 |
| angry | score_angry | browDownLeft, browDownRight | 192 | 0.69 | 0.55 | 85 | -0.60 | -1.00 .. -0.40 | 0.31 | 0.74 | 0.35 |
| sad | score_sad | mouthFrownLeft, mouthFrownRight | 192 | 0.50 | 0.03 | 4 | -0.70 | -1.10 .. -0.25 | 0.50 | 0.50 | 0.43 |
| afraid | score_afraid | browInnerUp, eyeWideLeft, eyeWideRight | 192 | 0.52 | 0.33 | 47 | 0.60 | +0.20 .. +1.40 | 0.34 | 0.55 | 0.40 |
| disgusted | score_disgusted | noseSneerLeft, noseSneerRight | 192 | 0.77 | 0.00 | 0 | — | — | — | — | 0.18 |
| surprised | score_surprised | browInnerUp, jawOpen | 192 | 0.73 | 0.76 | 119 | 0.00 | -0.40 .. +0.20 | 0.79 | 0.96 | 0.09 |

All emotions: 360 clips with both onsets, lag median -0.20 s (IQR -0.60 .. +0.20), 79% within ±1 s, 56% within ±0.5 s.

### crema

| Emotion | Signal | Blendshapes | Clips | Readout onset found | Blendshape onset found | Both | Lag median (s) | IQR | within ±0.5 s | within ±1 s | Starts expressive |
|---|---|---|---|---|---|---|---|---|---|---|---|
| happy | valence | mouthSmileLeft, mouthSmileRight | 96 | 0.84 | 0.97 | 81 | 0.00 | -0.40 .. +0.40 | 0.69 | 0.88 | 0.07 |
| angry | score_angry | browDownLeft, browDownRight | 96 | 0.42 | 0.32 | 19 | -0.20 | -0.50 .. +0.30 | 0.58 | 0.84 | 0.55 |
| sad | score_sad | mouthFrownLeft, mouthFrownRight | 96 | 0.26 | 0.01 | 0 | — | — | — | — | 0.44 |
| afraid | score_afraid | browInnerUp, eyeWideLeft, eyeWideRight | 96 | 0.52 | 0.43 | 26 | 0.40 | +0.20 .. +0.75 | 0.54 | 0.77 | 0.35 |
| disgusted | score_disgusted | noseSneerLeft, noseSneerRight | 96 | 0.80 | 0.00 | 0 | — | — | — | — | 0.15 |

All emotions: 126 clips with both onsets, lag median +0.20 s (IQR -0.40 .. +0.40), 85% within ±1 s, 64% within ±0.5 s.

Self-recorded scripted events (neutral → smile → neutral, a yawn, a frown, with marked timestamps) are not available yet; that part of the temporal check remains open until the M3 recording tool has produced them.

## Throughput and cost

Benchmark on NVIDIA L40S (model load 35.7 s, peak 23.5 GB), capture to layer 50, chat-prefill layout, real face crops:

| Token budget | Batch | Image tokens | Seq len | Frames/s per GPU | $/h of video at 2 fps | 5 fps | 10 fps |
|---|---|---|---|---|---|---|---|
| 280 | 8 | 256 | 288 | 5.91 | $0.66 | $1.65 | $3.3 |
| 560 | 8 | 529 | 561 | 2.47 | $1.58 | $3.95 | $7.89 |

GPU time for the runs (L40S at $1.95/h):

| Run | Frames | Calls | Wall (s) | GPU busy (s) | Median frames/s per GPU |
|---|---|---|---|---|---|
| medium (gate2) | 23164 | 1399 | 1506 | 9266 | 2.57 |
| upper (upper) | 2497 | 128 | 269 | 531 | 4.8 |

GPU busy time 2.72 h, plus about 0.25 h of model loading (one load per container), about 0.1 h for the benchmark and the single-clip test, and 1.4 h for the aborted first run below: roughly 4.5 GPU-hours, about $9. The realized rate inside calls (2.6 frames/s on the medium run) was below the benchmark's 5.9 because frames were read one by one from the Volume and each clip is a short chunk; parallel reads (added before the upper run) brought it to 4.8.

- First medium run stopped at 15:58 after 603 RAVDESS + 3 CREMA-D clips (about 12 min on 7 L40S, ~1.4 GPU-h): it committed the Volume after every clip and ran at ~14 frames/s aggregate; restarted with one commit per call, which skipped the finished chunks.

## Reading of the result

- **The video pipeline reproduces M1.** On the middle third of each clip (M1's sampling) the clip means give valence AUC 0.989 and 4-way 0.572 against M1's 0.982 and 0.576 from two still frames; whole-clip means are slightly lower on categories (0.527) because they include the neutral lead-in and tail.
- **No lip-reading leakage is visible.** Sentence identity explains 0.4% of clip-level valence variance against 73% for emotion (CREMA-D, 12 sentences: 0.7% vs 69%). On the 171 scores sentence reaches 1.6% (CREMA-D 1.2%), still far under emotion (59%). Actor identity explains more than sentence does (6.8% on valence, 13.5% on scores), and a per-actor baseline raises emotion's share (valence 78%, scores 68%).
- **Speech motion sets the per-frame noise floor, and 1.5 s windows halve it.** Within a neutral-speech clip the per-frame valence SD is 0.42 reference units; after windowing 0.23. The happy − negative gap is 2.2, so the windowed series sits 10 SDs below the expression gap (Gate 1 required 2). Between-emotion variance of clip means (0.71) is 13× the windowed within-neutral variance.
- **Upper-face crops remove most of the speech-motion variance but also part of the signal.** On the same 128 clips the within-neutral per-frame variance drops from 0.150 to 0.023 (6×) while the between-emotion variance drops from 1.14 to 0.44; gap/SD improves (11 → 20 windowed) but valence AUC falls from 1.000 to 0.961 and angry-vs-sad from 0.88 to 0.77. Keep the medium crop as the default and treat the upper crop as the ablation for speech-heavy material. Scores are in medium-crop reference units, so the upper crop's absolute scale is not calibrated.
- **Onsets line up with the face where MediaPipe sees a clear event.** For happy, the valence onset and the smile-blendshape onset coincide (median lag 0.0 s, within ±0.5 s for 62% of RAVDESS and 69% of CREMA-D clips with both onsets); surprised likewise. For angry the readout leads the brow-lowering blendshape by about half a second on RAVDESS. Sad and disgusted have no usable blendshape onset (the frown and sneer scores rarely rise by 0.1), so the comparison is limited to four emotions. Many takes are expressive from the first frame (no onset to detect), and the acted onset is not a scripted event with a marked time; the self-recorded events remain the open item.

## Notes

- Upper-face comparison on 128 held-out RAVDESS clips (all neutral, plus strong happy/angry/sad); the upper crop keeps the top 60% of the medium crop (768×461 px, fewer image tokens).
- Clip-level numbers use the mean of the 1.5 s windowed series over the whole clip (RAVDESS about 3.6 s, CREMA-D about 2.5 s).
- CREMA-D subset: 8 held-out actors (every 4th of the 30), all their unspecified- and high-intensity clips, so each actor has the full 12 sentences × 6 emotions crossing.
- Reference units: scores and PCs are divided by their SD over the M1 dev-actor clip means, so 1.0 is one SD of the expression spread in M1.
