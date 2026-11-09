"""Clip metadata for the acted-emotion corpora used in M1 and M2.

CREMA-D   91 actors, 12 sentences, 6 emotions, 7,442 clips. Open Database License.
          File names: {actor}_{sentence}_{emotion}_{level}.flv
RAVDESS   24 actors, 2 sentences, 8 emotions. CC BY-NC-SA 4.0 (non-commercial).
          File names: {modality}-{channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.mp4
          Only modality 02 (video without audio), channel 01 (speech) is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

CREMA_EMOTIONS = {"ANG": "angry", "DIS": "disgusted", "FEA": "afraid", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
CREMA_LEVELS = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "afraid", "07": "disgusted", "08": "surprised",
}
RAVDESS_INTENSITY = {"01": "normal", "02": "strong"}

# Labels shared by both corpora, and the valence sign used for the direction check.
SIX = ("angry", "disgusted", "afraid", "happy", "neutral", "sad")
NEGATIVE = ("angry", "disgusted", "afraid", "sad")
VALENCE = {"happy": 1, "angry": -1, "disgusted": -1, "afraid": -1, "sad": -1}

CREMA_MEDIA = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/VideoFlash/{name}.flv"
CREMA_RAW = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/{path}"
CREMA_TREE = "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/master?recursive=1"
RAVDESS_ZIP = "https://zenodo.org/api/records/1188976/files/Video_Speech_Actor_{actor:02d}.zip/content"


@dataclass
class Clip:
    dataset: str
    clip: str        # file stem
    actor: str       # unique across datasets, e.g. "crema-1001", "ravdess-07"
    emotion: str
    intensity: str
    sentence: str
    split: str       # "dev" or "heldout", by actor

    def as_dict(self) -> dict:
        return asdict(self)


def split_for(actor_number: int) -> str:
    """Every third actor is held out. Depends only on the actor id."""
    return "heldout" if actor_number % 3 == 0 else "dev"


def parse_crema(stem: str) -> Clip:
    actor, sentence, emotion, level = stem.split("_")
    return Clip("crema", stem, f"crema-{actor}", CREMA_EMOTIONS[emotion], CREMA_LEVELS[level], sentence,
                split_for(int(actor)))


def parse_ravdess(stem: str) -> Clip | None:
    """None for files that are not video-only speech."""
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    modality, channel, emotion, intensity, statement, _rep, actor = parts
    if modality != "02" or channel != "01":
        return None
    return Clip("ravdess", stem, f"ravdess-{actor}", RAVDESS_EMOTIONS[emotion], RAVDESS_INTENSITY[intensity],
                f"s{statement}", split_for(int(actor)))


def parse(dataset: str, stem: str) -> Clip | None:
    return parse_crema(stem) if dataset == "crema" else parse_ravdess(stem)
