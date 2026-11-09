"""Download the parts of dejanai/gemotions this project uses.

Always fetched: the final vectors and per-layer result JSONs for every layer,
and the neutral dialogues. The stories (434 MB) are fetched unless disabled.
The per-story activation caches are about 3.7 GB per layer, so they are
fetched only for the layers named.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download

REPO = "dejanai/gemotions"
RESULTS = "results/gemma4-31b"


def fetch_gemotions(dest: str, cache_layers=(), stories: bool = True) -> str:
    patterns = [
        f"{RESULTS}/emotion_vectors_layer*.npz",
        f"{RESULTS}/experiment_results_layer*.json",
        "data/neutral.db",
        "README.md",
    ]
    if stories:
        patterns.append("data/stories.db")
    for layer in cache_layers:
        patterns.append(f"{RESULTS}/_raw_cache_layer{layer}/*.npy")
    return snapshot_download(REPO, local_dir=dest, allow_patterns=patterns)
