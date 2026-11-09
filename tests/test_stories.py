"""Offline tests for the M3 story set: plan, parsing, checks, record construction, and the idempotent generate loop."""

import json

from cogmirror import stories as S


def test_plan_shape_and_coverage():
    plan = S.build_plan()
    kinds = [s.kind for s in plan]
    assert len(plan) == 70
    assert kinds.count("turn") == 24 and kinds.count("neutral") == 8 and kinds.count("single") == 38
    assert len({s.topic_idx for s in plan}) == 70            # distinct topics
    assert len({s.id for s in plan}) == 70
    used = {e for s in plan for e in s.emotions}
    assert used == set(S.TARGETS)                             # every target emotion appears
    clusters = {S.CLUSTER_OF[e] for e in used}
    assert len(clusters) == 10                                # all ten gemotions clusters
    quads = {S.TARGETS[e] for e in used}
    assert len(quads) == 4                                    # both ends of valence and arousal
    assert S.build_plan() == plan                             # deterministic


def test_paper_lists():
    assert len(S.EMOTIONS_171) == 171 and len(set(S.EMOTIONS_171)) == 171
    assert len(S.TOPICS) == 100 and len(set(S.TOPICS)) == 100
    assert sorted(S.CLUSTER_OF) == sorted(S.EMOTIONS_171)


def test_split_sentences_and_merge():
    t = '"Are you sure?" he asked. She nodded slowly. The rain fell on the roof. Then nothing.'
    sents = S.split_sentences(t)
    assert sents == ['"Are you sure?" he asked.', "She nodded slowly.", "The rain fell on the roof. Then nothing."]
    assert " ".join(sents) == t


def test_banned_words_check():
    spec = S.StorySpec("x", "single", ["afraid"], "t", 0)
    ok = " ".join(["word"] * 130) + "."
    assert S.check_story(ok, spec).ok
    bad = S.check_story("She was scared. " + ok, spec)
    assert not bad.ok and any("names afraid" in r for r in bad.reasons)
    short = S.check_story("Too short.", spec)
    assert "2 words" in short.reasons[0]


def test_turn_record_locates_turn():
    spec = S.StorySpec("t1", "turn", ["calm", "afraid"], "topic", 3)
    head = "He sat by the window and watched the rain for a long while. The tea was warm in his hands. " * 2
    tail = "Then the lights went out all at once. He froze in the chair. " * 3
    text = head.strip() + " " + S.TURN_MARKER + " " + tail.strip()
    words = S.word_count(text.replace(S.TURN_MARKER, " "))
    chk = S.check_story(text, spec, lo=10, hi=400)
    assert chk.ok, chk.reasons
    rec = S.make_record(spec, text, {"provider": "test"})
    assert S.TURN_MARKER not in rec["text"]
    assert rec["word_count"] == words
    assert rec["turn"]["sentence_idx"] == 4                   # four sentences before the turn
    assert rec["turn"]["word_idx"] == S.word_count(head)
    assert rec["sentences"][rec["turn"]["sentence_idx"]].startswith("Then the lights")
    assert rec["sentence_word_offsets"][4] == rec["turn"]["word_idx"]


class FakeLLM:
    provider, model = "fake", "fake-1"

    def __init__(self):
        self.calls = 0
        self.usage = {}

    def complete(self, prompt):
        self.calls += 1
        n_stories = 3
        body = ("The kettle clicked off and she poured without looking. " * 16).strip()  # 144 words
        if S.TURN_MARKER in prompt:
            half = ("The kettle clicked off and she poured without looking. " * 8).strip()
            body = half + " " + S.TURN_MARKER + " " + half
        return "\n\n".join(f"[story {i + 1}]\n\n{body}" for i in range(n_stories))


def test_generate_is_idempotent(tmp_path):
    llm = FakeLLM()
    recs = S.generate(tmp_path, llm, log=lambda *_: None)
    assert len(recs) == 70 and llm.calls == 70
    assert (tmp_path / "stories.jsonl").exists() and (tmp_path / "text" / "s001.txt").exists()
    assert len(list((tmp_path / "raw").glob("*.json"))) == 70
    # Second run: nothing to do, no calls.
    recs2 = S.generate(tmp_path, llm, log=lambda *_: None)
    assert llm.calls == 70 and [r["id"] for r in recs2] == [r["id"] for r in recs]
    # Delete one record: only that one is regenerated, from the raw cache (still no call).
    rows = [json.loads(l) for l in (tmp_path / "stories.jsonl").read_text().splitlines()]
    (tmp_path / "stories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[1:]))
    recs3 = S.generate(tmp_path, llm, log=lambda *_: None)
    assert llm.calls == 70 and len(recs3) == 70
    turn_recs = [r for r in recs3 if r["kind"] == "turn"]
    assert all(r["turn"] is not None for r in turn_recs)
    assert S.verify(recs3) == []
