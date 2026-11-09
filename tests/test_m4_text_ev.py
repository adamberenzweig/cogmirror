import numpy as np
import pytest

from cogmirror import text_ev
from cogmirror.evspace import EVSpace


def _offsets(tokens):
    """Character offsets of a token list that concatenates to the text."""
    out, i = [], 0
    for w in tokens:
        out.append((i, i + len(w)))
        i += len(w)
    return out


def test_sentence_char_spans_sequential():
    text = "A b. C d e. F."
    spans = text_ev.sentence_char_spans(text, ["A b.", "C d e.", "F."])
    assert spans == [(0, 4), (5, 11), (12, 14)]
    with pytest.raises(ValueError):
        text_ev.sentence_char_spans(text, ["C d e.", "A b."])    # out of order


def test_token_spans_contiguous_and_cover_every_token():
    text = "Hello there. How are you?"
    toks = ["Hello", " there", ".", " How", " are", " you", "?"]
    spans = text_ev.token_spans(_offsets(toks), text_ev.sentence_char_spans(text, ["Hello there.", "How are you?"]))
    assert spans.tolist() == [[0, 3], [3, 7]]
    # a special token with an empty span is ignored; a leading-space token goes to the sentence it ends in
    toks2 = ["<s>", "Hello", " there", ".", " How", " are", " you", "?"]
    offs = [(0, 0)] + _offsets(toks2[1:])
    spans2 = text_ev.token_spans(offs, text_ev.sentence_char_spans(text, ["Hello there.", "How are you?"]))
    assert spans2.tolist() == [[1, 4], [4, 8]]


def test_token_spans_truncated_sentence_is_empty():
    text = "One two. Three four."
    toks = ["One", " two", "."]       # truncated before the second sentence
    spans = text_ev.token_spans(_offsets(toks), text_ev.sentence_char_spans(text, ["One two.", "Three four."]))
    assert spans[0].tolist() == [0, 3] and spans[1].tolist() == [0, 0]
    pooled = text_ev.pool_spans(np.ones((3, 4)), spans)
    assert np.allclose(pooled[0], 1) and np.isnan(pooled[1]).all()


def test_pool_spans_means_over_tokens():
    h = np.arange(12, dtype=float).reshape(6, 2)
    out = text_ev.pool_spans(h, np.array([[0, 2], [2, 6]]))
    assert np.allclose(out[0], h[:2].mean(axis=0)) and np.allclose(out[1], h[2:].mean(axis=0))


def _fake_space(layer, rng, E=6, D=16):
    vecs = {f"e{i}": rng.normal(size=D) for i in range(E)}
    vecs["happy"], vecs["sad"] = rng.normal(size=D), rng.normal(size=D)
    return EVSpace.from_vectors(layer, vecs, n_components=3)


def test_project_round_trip_and_table(tmp_path):
    rng = np.random.default_rng(0)
    layers = [20, 30]
    D = 16
    results = []
    ids = ["s1", "s2"]
    for n_sent in (3, 2):
        results.append({"sentence": rng.normal(size=(n_sent, len(layers), D)).astype(np.float32),
                        "story": rng.normal(size=(len(layers), D)).astype(np.float32),
                        "story_full": rng.normal(size=(len(layers), D)).astype(np.float32), "n_tokens": 40})
    arrays = text_ev.stack_results(ids, results, layers)
    assert arrays["sentence"].shape == (5, 2, D) and arrays["sentence_story"].tolist() == [0, 0, 0, 1, 1]
    origins = {20: rng.normal(size=D), 30: rng.normal(size=D)}
    p = text_ev.save_acts(tmp_path / "acts.npz", arrays, origins, {20: "cache", 30: "stories"})
    acts = text_ev.load_acts(p)
    assert acts.layers == layers and acts.origin_source == {20: "cache", 30: "stories"}
    spaces = {l: _fake_space(l, rng) for l in layers}
    cols = text_ev.project_text(acts, spaces)
    assert len(cols["story_id"]) == (5 + 2) * 2
    # the whole-story row of s1 at layer 30 equals the direct projection
    ev = text_ev.table_at_layer(cols, 30)
    r = ev.lookup("s1", text_ev.STORY_ROW)
    x = acts.story[0, 1].astype(np.float64)
    assert np.allclose(ev.scores[r], spaces[30].scores(x, origins[30]))
    assert np.allclose(ev.pcs[r], spaces[30].pcs(x, origins[30]))
    assert ev.n_sentences("s1") == 3
    # parquet round trip keeps the index
    t = text_ev.write_table(tmp_path / "t.parquet", cols, {"layers": layers})
    cols2, meta = text_ev.read_table(t)
    ev2 = text_ev.table_at_layer(cols2, 20)
    assert ev2.lookup("s2", 1) is not None and ev2.emotions == ev.emotions
    assert meta["layers"] == str(layers)


def test_resolve_origins_falls_back_to_stories(tmp_path):
    rng = np.random.default_rng(1)
    story = rng.normal(size=(4, 2, 8))
    origins, src = text_ev.resolve_origins(tmp_path, [20, 30], story, cached={20: np.zeros(8)})
    assert src == {20: text_ev.ORIGIN_CACHE, 30: text_ev.ORIGIN_STORIES}
    assert np.allclose(origins[30], story[:, 1].mean(axis=0))


def test_summarize_reports_target_rank():
    rng = np.random.default_rng(2)
    space = _fake_space(20, rng)
    layers = [20]
    acts = text_ev.TextActs(layers, ["s1"], np.array([0]), np.array([0]), np.zeros((1, 1, 16), np.float16),
                            (space.vectors[space.emotions.index("happy")] * 10)[None, None].astype(np.float16),
                            np.zeros((1, 1, 16), np.float16), np.array([10]), {20: np.zeros(16)}, {20: "stories"})
    cols = text_ev.project_text(acts, {20: space})
    s = text_ev.summarize(cols, [{"id": "s1", "kind": "single", "emotions": ["happy"]}], 20)
    assert s[0]["target_ranks"]["happy"] == 1 and s[0]["hit"] is True
