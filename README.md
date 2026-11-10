# Cognitive Mirror

Reading LLM emotion vectors from facial video.

- [Project brief](cogmirror-brief.md) -- My original prompt, everything else is with Claude.
- Demo: [Viewer](https://adam-b-11275-cogmirror--cogmirror.us-east.modal.run/status?key=MqijQ1eG17xsG2v1c7s0ipxa) and [Recorder](https://adam-b-11275-cogmirror--cogmirror.us-east.modal.run/?key=MqijQ1eG17xsG2v1c7s0ipxa)

## Walkthrough

1. Read the [project brief](cogmirror-brief.md) to understand the goal, and skim the [Design doc](docs/face-emotion-vectors-design.md) to
get a sense of the approach.

2. Watch a few example recordings using the [Viewer](https://adam-b-11275-cogmirror--cogmirror.us-east.modal.run/status?key=MqijQ1eG17xsG2v1c7s0ipxa).

3. Skim the [Notebook](notebooks/m4_results.ipynb) to get a sense of the validation and evaluation results. There are also results in the [work plan](docs/face-emotion-vectors-workplan.md).

3. If you're game, use the [Recorder](https://adam-b-11275-cogmirror--cogmirror.us-east.modal.run/?key=MqijQ1eG17xsG2v1c7s0ipxa) to record 
a few takes of yourself reading some stories. Set the number of takes to 2-3, assuming you don't want to spend a half hour recording 12 stories. Then revisit the Viewer, the new recordings should be processed with a minute or two of completing each take.


## Results

See the [Notebook](notebooks/m4_results.ipynb) for detailed results.

* Gemma 4’s emotion directions, which were trained on text, do read facial expression from images and video without any training on faces. This is evidence for shared emotion axes across modalities.
* The evidence is pretty weak beyond the first PC of emotion space (valence, ie. pleasure-displeasure). More work would be needed to say whether it's because of the 
limitations of this little prototype and tiny corpus, or a real result.

### Future Work

- more data from more conditions, people. youtube, meetings.
- Run in realtime (runs at 4 frames/sec, should be possible)
- word-level alignment with WhisperX
- per-user normalization?
- Labels or eval other than synthetic stories. sentiment tags from youtube comments?


## Setup and Runbook 

- [Runbook](docs/runbook.md): setup, running locally or hosted on Modal, recording, processing, viewing, analysis