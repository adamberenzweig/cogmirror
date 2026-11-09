"""M3 story set: fresh emotion stories for the text–face recordings.

The prompts are the ones in the appendix of Anthropic's "Emotion Concepts
and their Function in a Large Language Model" (the gemotions replication
used the same prompts), with two additions: a length/format addendum so the
stories split cleanly into sentences for the teleprompter and narration, and
a variant for stories with a scripted emotional turn. The released gemotions
stories are deliberately not reused; they defined the vectors.

Layout under data/stories/:
    plan.json             the deterministic story plan (id, emotions, topic)
    raw/<id>.json         raw LLM responses, kept so reruns are free
    stories.jsonl         one record per accepted story
    text/<id>.txt         plain text for the teleprompter
    audio/<id>/<k>.m4a    per-sentence narration, plus index.json
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paper appendix material
# --------------------------------------------------------------------------

EMOTIONS_171 = [
    "afraid", "alarmed", "alert", "amazed", "amused", "angry", "annoyed", "anxious", "aroused",
    "ashamed", "astonished", "at ease", "awestruck", "bewildered", "bitter", "blissful", "bored",
    "brooding", "calm", "cheerful", "compassionate", "contemptuous", "content", "defiant",
    "delighted", "dependent", "depressed", "desperate", "disdainful", "disgusted", "disoriented",
    "dispirited", "distressed", "disturbed", "docile", "droopy", "dumbstruck", "eager", "ecstatic",
    "elated", "embarrassed", "empathetic", "energized", "enraged", "enthusiastic", "envious",
    "euphoric", "exasperated", "excited", "exuberant", "frightened", "frustrated", "fulfilled",
    "furious", "gloomy", "grateful", "greedy", "grief-stricken", "grumpy", "guilty", "happy",
    "hateful", "heartbroken", "hope", "hopeful", "horrified", "hostile", "humiliated", "hurt",
    "hysterical", "impatient", "indifferent", "indignant", "infatuated", "inspired", "insulted",
    "invigorated", "irate", "irritated", "jealous", "joyful", "jubilant", "kind", "lazy", "listless",
    "lonely", "loving", "mad", "melancholy", "miserable", "mortified", "mystified", "nervous",
    "nostalgic", "obstinate", "offended", "on edge", "optimistic", "outraged", "overwhelmed",
    "panicked", "paranoid", "patient", "peaceful", "perplexed", "playful", "pleased", "proud",
    "puzzled", "rattled", "reflective", "refreshed", "regretful", "rejuvenated", "relaxed",
    "relieved", "remorseful", "resentful", "resigned", "restless", "sad", "safe", "satisfied",
    "scared", "scornful", "self-confident", "self-conscious", "self-critical", "sensitive",
    "sentimental", "serene", "shaken", "shocked", "skeptical", "sleepy", "sluggish", "smug", "sorry",
    "spiteful", "stimulated", "stressed", "stubborn", "stuck", "sullen", "surprised", "suspicious",
    "sympathetic", "tense", "terrified", "thankful", "thrilled", "tired", "tormented", "trapped",
    "triumphant", "troubled", "uneasy", "unhappy", "unnerved", "unsettled", "upset", "valiant",
    "vengeful", "vibrant", "vigilant", "vindictive", "vulnerable", "weary", "worn out", "worried",
    "worthless",
]

# Table 12 of the paper: k-means clusters (k=10), ordered by valence.
CLUSTERS = {
    "Exuberant Joy": ["blissful", "cheerful", "delighted", "eager", "ecstatic", "elated", "energized",
                      "enthusiastic", "euphoric", "excited", "exuberant", "happy", "invigorated", "joyful",
                      "jubilant", "optimistic", "pleased", "stimulated", "thrilled", "vibrant"],
    "Peaceful Contentment": ["at ease", "calm", "content", "patient", "peaceful", "refreshed", "relaxed",
                             "safe", "serene"],
    "Compassionate Gratitude": ["compassionate", "empathetic", "fulfilled", "grateful", "hope", "hopeful",
                                "inspired", "kind", "loving", "rejuvenated", "relieved", "satisfied",
                                "sentimental", "sympathetic", "thankful"],
    "Competitive Pride": ["greedy", "proud", "self-confident", "smug", "spiteful", "triumphant", "valiant",
                          "vengeful", "vindictive"],
    "Playful Amusement": ["amused", "playful"],
    "Depleted Disengagement": ["bored", "depressed", "docile", "droopy", "indifferent", "lazy", "listless",
                               "resigned", "restless", "sleepy", "sluggish", "sullen", "tired", "weary",
                               "worn out"],
    "Vigilant Suspicion": ["paranoid", "suspicious", "vigilant"],
    "Hostile Anger": ["angry", "annoyed", "contemptuous", "defiant", "disdainful", "enraged", "exasperated",
                      "frustrated", "furious", "grumpy", "hateful", "hostile", "impatient", "indignant",
                      "insulted", "irate", "irritated", "mad", "obstinate", "offended", "outraged",
                      "resentful", "scornful", "skeptical", "stubborn"],
    "Fear and Overwhelm": ["afraid", "alarmed", "alert", "amazed", "anxious", "aroused", "astonished",
                           "awestruck", "bewildered", "disgusted", "disoriented", "distressed", "disturbed",
                           "dumbstruck", "embarrassed", "frightened", "horrified", "hysterical", "mortified",
                           "mystified", "nervous", "on edge", "overwhelmed", "panicked", "perplexed",
                           "puzzled", "rattled", "scared", "self-conscious", "sensitive", "shaken", "shocked",
                           "stressed", "surprised", "tense", "terrified", "uneasy", "unnerved", "unsettled",
                           "upset", "worried"],
    "Despair and Shame": ["ashamed", "bitter", "brooding", "dependent", "desperate", "dispirited", "envious",
                          "gloomy", "grief-stricken", "guilty", "heartbroken", "humiliated", "hurt",
                          "infatuated", "jealous", "lonely", "melancholy", "miserable", "nostalgic",
                          "reflective", "regretful", "remorseful", "sad", "self-critical", "sorry", "stuck",
                          "tormented", "trapped", "troubled", "unhappy", "vulnerable", "worthless"],
}
CLUSTER_OF = {e: c for c, es in CLUSTERS.items() for e in es}

TOPICS = [
    "An artist discovers someone has tattooed their work",
    "A family member announces they're converting to a different religion",
    "Someone's childhood imaginary friend appears in their niece's drawings",
    "A person finds out their biography was written without their knowledge",
    "A neighbor starts a renovation project",
    "Someone finds their grandmother's engagement ring in a pawn shop",
    "A student learns their scholarship application was denied",
    "A person's online friend turns out to live in the same city",
    "A neighbor wants to install a fence",
    "An adult child moves back in with their parents",
    "An employee is asked to train their replacement",
    "An athlete is asked to switch positions",
    "A traveler's flight is delayed, causing them to miss an important event",
    "A patient receives an unexpected medical diagnosis",
    "A professional is passed over for a promotion",
    "Someone receives a mysterious package with no return address",
    "A neighbor's dog keeps barking at night",
    "A family member borrows money and doesn't pay it back",
    "A person discovers their partner has been reading their messages",
    "A pet owner's cat brings home an unusual gift",
    "A commuter encounters a street performer",
    "A job candidate learns the position has been filled internally",
    "A person's roommate is hosting loud parties",
    "A hiker encounters wildlife on a trail",
    "A teenager's parents decide to downsize to a smaller home",
    "A coworker takes credit for someone's idea",
    "A person's favorite restaurant closes",
    "A gardener finds an unexpected plant growing in their yard",
    "A commuter's regular route is blocked by construction",
    "A person finds out their partner has a secret bank account",
    "A manager must deliver layoff news",
    "A couple disagrees about how to spend their tax refund",
    "A person's grandmother passes down a family recipe",
    "A volunteer is asked to lead the organization",
    "A parent finds their child's diary",
    "A person's coworker keeps eating their lunch",
    "A person discovers an unknown relative through a DNA test",
    "A teacher receives an anonymous gift",
    "A person's childhood home is being demolished",
    "A diner finds a foreign object in their food",
    "A person's partner announces they want to adopt a child",
    "A hobbyist's collection is damaged in a flood",
    "A person learns their favorite teacher has retired",
    "A couple's wedding venue cancels at the last minute",
    "A person learns a close friend has been talking behind their back",
    "A commuter's car breaks down on the highway",
    "A homeowner discovers a hidden room",
    "A person's social media post goes unexpectedly viral",
    "A parent finds an old letter they wrote but never sent",
    "A person is asked to give a eulogy",
    "A student's roommate drops out of school",
    "Someone receives a letter from their younger self",
    "A person's favorite band announces a reunion tour",
    "A neighbor keeps parking in someone's spot",
    "A person's pet needs expensive surgery",
    "A couple finds out they're having twins",
    "A professional is asked to work on a holiday",
    "A homeowner's basement floods",
    "Someone finds out they have a half-sibling they never knew about",
    "A person's landlord raises the rent significantly",
    "A worker's company is acquired by a competitor",
    "A traveler loses their luggage",
    "A person's partner wants to move to another country",
    "A student is accused of cheating",
    "A person's parents announce they're getting divorced",
    "A neighbor's tree falls on someone's property",
    "A person is invited to speak at a conference",
    "A person's friend asks them to be a kidney donor",
    "A customer receives the wrong order",
    "A person's elderly parent needs full-time care",
    "A driver is involved in a minor accident",
    "A person's partner proposes in public",
    "A homeowner finds termites",
    "A person's coworker is promoted over them",
    "A tenant's heat stops working in winter",
    "A person's childhood best friend reaches out after years of silence",
    "An employee is asked to relocate for work",
    "A person's sibling announces an engagement to someone the family dislikes",
    "A homeowner receives a notice of eminent domain",
    "A shopper finds a wallet in a parking lot",
    "A person's car is towed",
    "A parent's child is bullied at school",
    "A person's ex moves in next door",
    "A worker's hours are cut",
    "A person discovers their favorite book is being adapted into a film",
    "A neighbor builds a treehouse that overlooks someone's yard",
    "A person's partner forgets their anniversary",
    "A professional makes a costly mistake at work",
    "A person's teenager gets a tattoo without permission",
    "A student receives a much lower grade than expected",
    "A person receives an invitation to their high school reunion",
    "A person's partner wants to open their relationship",
    "A homeowner's property value drops",
    "A person's friend borrows their car and gets a ticket",
    "A person's identity is stolen",
    "Someone finds a time capsule they don't remember burying",
    "Someone finds their partner's bucket list",
    "A neighbor asks to use part of the yard for a garden",
    "A person learns their apartment building is going condo",
    "Someone finds their college application essay published as an example",
]

STORY_PROMPT = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]

[story 2]

[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:

- The character's actions and behaviors

- Physical sensations and body language

- Dialogue and tone of voice

- Thoughts and internal reactions

- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""

TURN_MARKER = "[TURN]"

TURN_PROMPT = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion_a} for the first part of the story. Partway through, roughly halfway, something happens, and from that point on the character is feeling {emotion_b} instead. The change should be clear and should happen at one identifiable moment.

Mark that moment: put the exact marker {marker} immediately before the first sentence in which the character's state has changed. Use the marker exactly once in each story, and nowhere else.

Format the stories like so:

[story 1]

[story 2]

[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the words '{emotion_a}' or '{emotion_b}' or any direct synonyms of them in the stories. Instead, convey each emotion ONLY through:

- The character's actions and behaviors

- Physical sensations and body language

- Dialogue and tone of voice

- Thoughts and internal reactions

- Situational context and environmental descriptions

Both emotions should be clearly conveyed to the reader through these indirect means, but never explicitly named."""

# The neutral-dialogue prompt from the appendix, adapted to prose passages.
NEUTRAL_PROMPT = """Write {n_stories} different short passages loosely connected to the following topic.

Topic: {topic}

Each passage is matter-of-fact expository or instructional prose, like a reference article, a manual, or a how-to guide that happens to touch on something from the topic (a procedure, a definition, a piece of background information, a practical set of steps). It is not a story and has no characters with feelings.

Format the passages like so:

[story 1]

[story 2]

[story 3]

etc.

Generate a diverse mix across the {n_stories} passages: some factual background (science, history, law, geography), some practical how-to instructions, some work-related explanations (procedures, definitions, summaries).

CRITICAL REQUIREMENT: These passages must be completely neutral and emotionless.

- NO emotional content whatsoever - not explicit, not implied, not subtle

- No person in the passage expresses or is described as having any feelings

- Avoid emotionally-charged topics and emotionally-charged language entirely

- Use matter-of-fact, neutral language throughout

- No pleasantries, no encouragement, no evaluative words like "great" or "unfortunately"

- Focus purely on information"""

# Added to every prompt so the text splits cleanly into sentences and fits the recordings.
FORMAT_ADDENDUM = """

Length and form: each one is 120 to 180 words, written as a single paragraph of complete sentences that read naturally aloud. Use only periods, question marks and exclamation marks to end sentences. Do not use semicolons, dashes, parentheses, ellipses, bullet points, headings, lists, code, numerals with decimal points, or abbreviations with periods (write "Mister" or avoid titles). Do not include a title."""


# --------------------------------------------------------------------------
# Target emotions and plan
# --------------------------------------------------------------------------

# 28 target emotions spanning the ten clusters and the four valence x arousal
# quadrants. Arousal is judged by the human circumplex, not by the gemotions
# PC2, which at layer 40 separates dominance-like emotions from calm ones.
TARGETS: dict[str, tuple[str, str]] = {
    # positive, high arousal
    "excited": ("positive", "high"), "joyful": ("positive", "high"), "thrilled": ("positive", "high"),
    "amused": ("positive", "high"), "proud": ("positive", "high"), "triumphant": ("positive", "high"),
    # positive, low arousal
    "calm": ("positive", "low"), "content": ("positive", "low"), "serene": ("positive", "low"),
    "relieved": ("positive", "low"), "grateful": ("positive", "low"), "loving": ("positive", "low"),
    "hopeful": ("positive", "low"), "nostalgic": ("positive", "low"),
    # negative, high arousal
    "angry": ("negative", "high"), "frustrated": ("negative", "high"), "disgusted": ("negative", "high"),
    "afraid": ("negative", "high"), "anxious": ("negative", "high"), "surprised": ("negative", "high"),
    "embarrassed": ("negative", "high"), "suspicious": ("negative", "high"),
    # negative, low arousal
    "sad": ("negative", "low"), "lonely": ("negative", "low"), "ashamed": ("negative", "low"),
    "bored": ("negative", "low"), "tired": ("negative", "low"), "resigned": ("negative", "low"),
}
assert all(e in CLUSTER_OF for e in TARGETS)

# Emotions that get a second single-emotion story (the most central ones).
SECOND_SINGLE = ["joyful", "calm", "sad", "angry", "afraid", "anxious", "content", "excited", "lonely", "proud"]

# Scripted turns: mostly valence flips, a few arousal shifts within a valence.
TURN_PAIRS: list[tuple[str, str]] = [
    ("calm", "afraid"), ("anxious", "relieved"), ("sad", "joyful"), ("excited", "sad"),
    ("bored", "surprised"), ("proud", "ashamed"), ("lonely", "loving"), ("content", "angry"),
    ("hopeful", "resigned"), ("angry", "sad"), ("excited", "calm"), ("afraid", "relieved"),
    ("grateful", "disgusted"), ("serene", "frustrated"), ("tired", "thrilled"), ("suspicious", "grateful"),
    ("joyful", "embarrassed"), ("amused", "anxious"), ("nostalgic", "lonely"), ("frustrated", "triumphant"),
    ("ashamed", "proud"), ("calm", "surprised"), ("sad", "content"), ("angry", "amused"),
]

# Roots that must not appear in a story about the emotion (the word itself,
# its inflections, and the plainest synonyms). Checked as word prefixes.
BANNED_ROOTS: dict[str, list[str]] = {
    "excited": ["excit", "thrill", "exhilarat"],
    "joyful": ["joy", "happi", "happy", "delight", "elat"],
    "thrilled": ["thrill", "excit", "exhilarat"],
    "amused": ["amus", "funny", "hilarious", "laughable"],
    "proud": ["proud", "pride"],
    "triumphant": ["triumph", "victorious", "victory"],
    "calm": ["calm", "tranquil", "serene", "serenity"],
    "content": ["content", "satisf"],
    "serene": ["seren", "tranquil", "calm"],
    "relieved": ["reliev", "relief"],
    "grateful": ["grateful", "gratitude", "thankful"],
    "loving": ["loving", "lovingly", "affection", "tender"],
    "hopeful": ["hope", "optimis"],
    "nostalgic": ["nostalg", "wistful"],
    "angry": ["angr", "anger", "furious", "fury", "rage", "enrage", "mad "],
    "frustrated": ["frustrat", "exasperat"],
    "disgusted": ["disgust", "revolt", "repuls", "nauseat"],
    "afraid": ["afraid", "fear", "scare", "scary", "frighten", "terrif", "terror", "dread"],
    "anxious": ["anxi", "nervous", "worr", "uneas"],
    "surprised": ["surpris", "astonish", "amaze", "shock", "startl"],
    "embarrassed": ["embarrass", "humiliat", "mortif"],
    "suspicious": ["suspic", "suspect", "distrust", "mistrust", "wary"],
    "sad": ["sad", "sorrow", "grief", "griev", "unhapp", "melanchol"],
    "lonely": ["lonel", "loneliness", "alone", "isolat"],
    "ashamed": ["asham", "shame", "humiliat", "guilt"],
    "bored": ["bored", "boredom", "boring", "tedious", "tedium", "dull"],
    "tired": ["tired", "exhaust", "weary", "wearin", "fatigue"],
    "resigned": ["resign", "give up", "gave up", "giving up", "acceptance"],
}
assert set(BANNED_ROOTS) == set(TARGETS)


@dataclass
class StorySpec:
    id: str
    kind: str                       # "single" | "turn" | "neutral"
    emotions: list[str]             # [] for neutral, [a] for single, [a, b] for turn
    topic: str
    topic_idx: int

    @property
    def clusters(self) -> list[str]:
        return [CLUSTER_OF[e] for e in self.emotions]


def build_plan(seed: int = 20261109, n_neutral: int = 8) -> list[StorySpec]:
    """Deterministic plan: 38 single + 24 turn + 8 neutral = 70 stories, all on distinct topics."""
    rng = random.Random(seed)
    topics = list(range(len(TOPICS)))
    rng.shuffle(topics)
    specs: list[StorySpec] = []

    def take_topic() -> int:
        return topics.pop()

    singles = list(TARGETS) + SECOND_SINGLE
    for e in singles:
        t = take_topic()
        specs.append(StorySpec(id="", kind="single", emotions=[e], topic=TOPICS[t], topic_idx=t))
    for a, b in TURN_PAIRS:
        t = take_topic()
        specs.append(StorySpec(id="", kind="turn", emotions=[a, b], topic=TOPICS[t], topic_idx=t))
    for _ in range(n_neutral):
        t = take_topic()
        specs.append(StorySpec(id="", kind="neutral", emotions=[], topic=TOPICS[t], topic_idx=t))
    rng.shuffle(specs)
    for i, s in enumerate(specs, 1):
        s.id = f"s{i:03d}"
    return specs


def prompt_for(spec: StorySpec, n_stories: int = 3) -> str:
    if spec.kind == "single":
        body = STORY_PROMPT.format(n_stories=n_stories, topic=spec.topic, emotion=spec.emotions[0])
    elif spec.kind == "turn":
        a, b = spec.emotions
        body = TURN_PROMPT.format(n_stories=n_stories, topic=spec.topic, emotion_a=a, emotion_b=b,
                                  marker=TURN_MARKER)
    elif spec.kind == "neutral":
        body = NEUTRAL_PROMPT.format(n_stories=n_stories, topic=spec.topic)
    else:
        raise ValueError(spec.kind)
    return body + FORMAT_ADDENDUM


# --------------------------------------------------------------------------
# Text processing and checks
# --------------------------------------------------------------------------

_SENT_RE = re.compile(r'\S.*?[.!?]+["”’\')\]]*(?=\s+|$)', re.S)


def split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation; merge fragments of fewer than three words into the previous sentence."""
    text = " ".join(text.split())
    parts = [m.group(0).strip() for m in _SENT_RE.finditer(text)]
    consumed = sum(len(p) for p in parts) + max(len(parts) - 1, 0)
    if not parts or consumed < len(text) - 2 * len(parts) - 2:
        # Something did not end with punctuation; keep the tail as its own sentence.
        tail = text[len(" ".join(parts)):].strip()
        if tail:
            parts.append(tail)
    merged: list[str] = []
    for p in parts:
        if merged and len(p.split()) < 3:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def word_count(text: str) -> int:
    return len(text.split())


def banned_hits(text: str, emotion: str) -> list[str]:
    """Words in `text` that start with a banned root for `emotion`."""
    roots = BANNED_ROOTS.get(emotion, [emotion.lower()])
    low = text.lower()
    hits = []
    for root in roots:
        if " " in root:
            if root in low:
                hits.append(root)
            continue
        for m in re.finditer(r"\b" + re.escape(root) + r"\w*", low):
            hits.append(m.group(0))
    return sorted(set(hits))


def parse_stories(raw: str) -> list[str]:
    """Split an LLM response on [story N] markers."""
    parts = re.split(r"\[\s*(?:story|passage)\s*\d+\s*\]", raw, flags=re.I)
    out = []
    for p in parts:
        p = p.strip().strip("*").strip()
        if word_count(p) >= 40:
            out.append(" ".join(p.split()))
    return out


@dataclass
class Check:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def check_story(text: str, spec: StorySpec, lo: int = 110, hi: int = 190) -> Check:
    reasons = []
    body = text.replace(TURN_MARKER, " ")
    n = word_count(body)
    if not lo <= n <= hi:
        reasons.append(f"{n} words")
    for e in spec.emotions:
        hits = banned_hits(body, e)
        if hits:
            reasons.append(f"names {e}: {', '.join(hits)}")
    if spec.kind == "turn":
        k = text.count(TURN_MARKER)
        if k != 1:
            reasons.append(f"{k} turn markers")
        else:
            before = word_count(text.split(TURN_MARKER)[0])
            frac = before / max(n, 1)
            if not 0.25 <= frac <= 0.75:
                reasons.append(f"turn at {frac:.0%}")
    else:
        if TURN_MARKER in text:
            reasons.append("unexpected turn marker")
    if re.search(r"[;()\[\]—–]|\.\.\.|\b(?:Mr|Mrs|Ms|Dr|St|etc|e\.g|i\.e)\.", body):
        reasons.append("disallowed punctuation")
    return Check(ok=not reasons, reasons=reasons)


def make_record(spec: StorySpec, text: str, meta: dict) -> dict:
    """Turn accepted text into the stories.jsonl record (marker removed, sentences and turn located)."""
    turn = None
    if spec.kind == "turn":
        head, tail = text.split(TURN_MARKER, 1)
        head_sents = split_sentences(head)
        clean = " ".join((head.strip() + " " + tail.strip()).split())
        sents = split_sentences(clean)
        # The turn sentence is the first sentence not fully inside `head`.
        head_words = word_count(head)
        acc = 0
        turn_idx = len(sents) - 1
        for i, s in enumerate(sents):
            if acc >= head_words:
                turn_idx = i
                break
            acc += word_count(s)
        turn = {"sentence_idx": turn_idx, "word_idx": head_words, "n_sentences_before": len(head_sents)}
    else:
        clean = " ".join(text.split())
        sents = split_sentences(clean)
    offsets, acc = [], 0
    for s in sents:
        offsets.append(acc)
        acc += word_count(s)
    return {
        "id": spec.id,
        "kind": spec.kind,
        "emotions": spec.emotions,
        "clusters": spec.clusters,
        "quadrants": [list(TARGETS[e]) for e in spec.emotions],
        "topic": spec.topic,
        "topic_idx": spec.topic_idx,
        "text": clean,
        "sentences": sents,
        "sentence_word_offsets": offsets,
        "n_sentences": len(sents),
        "word_count": acc,
        "turn": turn,
        **meta,
    }


# --------------------------------------------------------------------------
# LLM access (plain HTTP, no SDK)
# --------------------------------------------------------------------------

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=value lines, no expansion; does not override existing variables."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.startswith("export "):
            k = k[7:].strip()
        os.environ.setdefault(k, v)


def detect_provider() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _post_json(url: str, headers: dict, payload: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < 4:
                time.sleep(2 ** attempt + 1)
                continue
            raise RuntimeError(f"HTTP {e.code} from {url}: {body[:800]}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < 4:
                time.sleep(2 ** attempt + 1)
                continue
            raise
    raise RuntimeError("unreachable")


class LLM:
    """One `complete(prompt) -> text` call against Anthropic, Gemini or OpenAI, chosen by the key present."""

    def __init__(self, provider: str | None = None, model: str | None = None, max_tokens: int = 4000):
        self.provider = provider or detect_provider()
        if self.provider is None:
            raise RuntimeError("no LLM API key found (ANTHROPIC_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY)")
        self.model = model or DEFAULT_MODELS[self.provider]
        self.max_tokens = max_tokens
        self.usage: dict[str, int] = {"input": 0, "output": 0, "calls": 0}

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.usage["calls"] += 1
        if self.provider == "anthropic":
            return self._anthropic(prompt, system)
        if self.provider == "gemini":
            return self._gemini(prompt, system)
        return self._openai(prompt, system)

    def _anthropic(self, prompt: str, system: str | None) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        headers = {"anthropic-version": "2023-06-01"}
        betas = ["server-side-fallback-2026-07-01"]
        if key:
            headers["x-api-key"] = key
        elif token:
            headers["authorization"] = f"Bearer {token}"
            betas.append("oauth-2025-04-20")
        headers["anthropic-beta"] = ",".join(betas)
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "fallbacks": "default",
        }
        if system:
            payload["system"] = system
        out = _post_json("https://api.anthropic.com/v1/messages", headers, payload)
        if out.get("stop_reason") == "refusal":
            raise RuntimeError(f"refusal: {out.get('stop_details')}")
        u = out.get("usage", {})
        self.usage["input"] += u.get("input_tokens", 0)
        self.usage["output"] += u.get("output_tokens", 0)
        return "".join(b.get("text", "") for b in out["content"] if b.get("type") == "text")

    def _gemini(self, prompt: str, system: str | None) -> str:
        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        out = _post_json(url, {"x-goog-api-key": key}, payload)
        u = out.get("usageMetadata", {})
        self.usage["input"] += u.get("promptTokenCount", 0)
        self.usage["output"] += u.get("candidatesTokenCount", 0)
        parts = out["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)

    def _openai(self, prompt: str, system: str | None) -> str:
        key = os.environ["OPENAI_API_KEY"]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "max_completion_tokens": self.max_tokens}
        out = _post_json("https://api.openai.com/v1/chat/completions", {"authorization": f"Bearer {key}"}, payload)
        u = out.get("usage", {})
        self.usage["input"] += u.get("prompt_tokens", 0)
        self.usage["output"] += u.get("completion_tokens", 0)
        return out["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# Generation (idempotent)
# --------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def generate(root: str | Path, llm: LLM | None = None, n_stories: int = 3, max_attempts: int = 2,
             seed: int = 20261109, only: set[str] | None = None, log=print) -> list[dict]:
    """Generate every story in the plan that is not already in stories.jsonl.

    Raw responses are cached under raw/<id>.<attempt>.json, so a rerun that only
    changes the checks costs nothing. Returns all records, in plan order.
    """
    root = Path(root)
    plan = build_plan(seed=seed)
    (root / "plan.json").write_text(json.dumps([asdict(s) for s in plan], indent=1))
    out_path = root / "stories.jsonl"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    have = {r["id"]: r for r in read_jsonl(out_path)}
    rejected: dict[str, list[str]] = {}

    for spec in plan:
        if spec.id in have or (only and spec.id not in only):
            continue
        prompt = prompt_for(spec, n_stories)
        accepted = None
        for attempt in range(max_attempts):
            raw_path = raw_dir / f"{spec.id}.{attempt}.json"
            if raw_path.exists():
                cached = json.loads(raw_path.read_text())
                raw, meta = cached["response"], cached["meta"]
            else:
                if llm is None:
                    llm = LLM()
                t0 = time.time()
                raw = llm.complete(prompt)
                meta = {"provider": llm.provider, "model": llm.model, "seconds": round(time.time() - t0, 1),
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                raw_path.write_text(json.dumps({"prompt": prompt, "response": raw, "meta": meta}, indent=1,
                                               ensure_ascii=False))
            candidates = parse_stories(raw)
            reasons = []
            for text in candidates:
                c = check_story(text, spec)
                if c.ok:
                    accepted = (text, meta)
                    break
                reasons.append("; ".join(c.reasons))
            if accepted:
                break
            rejected[spec.id] = reasons
            log(f"{spec.id}: no candidate passed on attempt {attempt + 1}: {reasons}")
        if accepted is None:
            log(f"{spec.id}: FAILED ({spec.kind} {spec.emotions} / {spec.topic})")
            continue
        rec = make_record(spec, accepted[0], accepted[1])
        have[spec.id] = rec
        write_jsonl(out_path, [have[s.id] for s in plan if s.id in have])
        log(f"{spec.id}: ok, {rec['word_count']} words, {rec['n_sentences']} sentences, "
            f"{spec.kind} {'/'.join(spec.emotions) or 'neutral'}")

    records = [have[s.id] for s in plan if s.id in have]
    write_jsonl(out_path, records)
    write_plaintext(root, records)
    return records


def write_plaintext(root: str | Path, records: list[dict]) -> None:
    d = Path(root) / "text"
    d.mkdir(parents=True, exist_ok=True)
    for r in records:
        (d / f"{r['id']}.txt").write_text(r["text"] + "\n")


def verify(records: list[dict]) -> list[str]:
    """Re-run the content checks on final records; returns a list of problems (empty when clean)."""
    problems = []
    for r in records:
        spec = StorySpec(r["id"], r["kind"], r["emotions"], r["topic"], r["topic_idx"])
        for e in spec.emotions:
            hits = banned_hits(r["text"], e)
            if hits:
                problems.append(f"{r['id']} names {e}: {hits}")
        if not 110 <= r["word_count"] <= 190:
            problems.append(f"{r['id']} has {r['word_count']} words")
        if spec.kind == "turn" and r["turn"] is None:
            problems.append(f"{r['id']} has no turn")
        if " ".join(r["sentences"]) != r["text"]:
            problems.append(f"{r['id']} sentences do not reassemble to text")
    return problems


# --------------------------------------------------------------------------
# Narration audio (macOS `say`)
# --------------------------------------------------------------------------

def render_audio(root: str | Path, records: list[dict], voice: str = "Samantha", rate: int = 170,
                 force: bool = False, log=print) -> None:
    """One AAC .m4a per sentence under audio/<id>/<k>.m4a, with index.json holding durations."""
    for r in records:
        d = Path(root) / "audio" / r["id"]
        d.mkdir(parents=True, exist_ok=True)
        index_path = d / "index.json"
        if index_path.exists() and not force:
            idx = json.loads(index_path.read_text())
            if idx.get("sentences") == r["sentences"] and idx.get("voice") == voice and idx.get("rate") == rate:
                continue
        durations = []
        for k, s in enumerate(r["sentences"]):
            aiff = d / f"{k}.aiff"
            m4a = d / f"{k}.m4a"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), s], check=True)
            subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(aiff), str(m4a)], check=True,
                           capture_output=True)
            aiff.unlink()
            durations.append(audio_duration(m4a))
        index_path.write_text(json.dumps({
            "id": r["id"], "voice": voice, "rate": rate, "sentences": r["sentences"],
            "files": [f"{k}.m4a" for k in range(len(r["sentences"]))], "durations_s": durations,
            "total_s": round(sum(durations), 3),
        }, indent=1, ensure_ascii=False))
        log(f"{r['id']}: {len(durations)} sentences, {sum(durations):.1f} s")


def audio_duration(path: Path) -> float:
    out = subprocess.run(["afinfo", str(path)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"estimated duration:\s*([\d.]+)", out)
    return round(float(m.group(1)), 3) if m else 0.0
