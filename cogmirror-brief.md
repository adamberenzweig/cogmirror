# The Cognitive Mirror

The Cognitive Mirror is a tool to help humans understand ourselves better. Not everyone is equally good at knowing how they feel [CITE]. But the signals are there:

1. Emotions can be read from faces, although it requires a lot of
context and knowledge of the individual.  Cognitive states that are
the most directly tied to physiology (pain, fatigue, hunger, and arousal) have well-established correlates in facial features. Commercial fatigue-detection systems for drivers are now required in the EU.

1.  The mind and the body are not distinct, biosignals (e.g. heart
rate, blood pressure, pupil dilation)  carry information about
cognitive states.

1.  Speech prosody and intonation carry much more information than a transcript. Recent systems (e.g. EmotionThinker) are moving from simple classification to reasoning and predicting emotion with interpretable explanations, grounded in prosody.

This prototype attempts to tie emotion vectors \[[Sofroniew 2026](https://transformer-circuits.pub/2026/emotions/index.html)\] to facial expressions and tone of voice.

The basic idea is to predict the emotion vector associated with
the text as a user reads a passage designed to carry emotional weight.

The resulting model could then be used to provide rich labels about the speaker's
cognitive and affective states. 


## Technical Sketch

1. Generate synthetic stories designed to carry emotional weight, then extract the emotion
vectors at each token, as with Sofroniew 2026.

1. Record videos of the user reading the stories.

1. Extract facial expression features (and eventually speech prosody features), aligned with the text (and therefore with the emotion vectors).

1. Train a model to predict the emotion vector from the expressive features.

1. Build a real-time pipeline to predict EVs from a live camera/audio feed.

1. Build a demo to visualize the EVs in various ways, e.g. a word cloud, bar graph or EKG-like sweeping chart, or a cartoon face animated with blendshapes.

## Model architecture

What model architecture to use, and how to train it?

Two high level options:

1. Train on a corpus of speech/video with transcription. Aim for emotion-rich transcripts, not
someone reading the news. dramas, youtubers excited about the game their streaming.

1. Few-shot personalization. Build and train a new model for this task. 
we want something that can show results with as little data as possible, e.g. a few minutes
of video.

1. Use an existing vision-language model, e.g. Gemma.

https://huggingface.co/dejanai/gemotions



## Open Questions

1. The original work from Anthropic used a closed model. There is some work to reproduce it
with open models, are these good enough and which one should we use?

1. What face features to use? Do we need an open model so that we can extract embeddings, or 
is a vector of posteriors or predicted features enough?

1. What speech features to use? We need a model that does well on prosody tasks like 
in the SUPERB benchmark.

1. Can this run in realtime? We can back off and annotate recordings if not.

1. Time dynamics -- video vs sequence of frames

1. Personalization

## Logistics

1. Where to run models? AWS, Google, Modal, etc.
1. Cost