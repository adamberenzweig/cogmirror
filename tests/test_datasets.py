from cogmirror import datasets, prep


def test_parse_crema():
    c = datasets.parse_crema("1002_IEO_HAP_HI")
    assert (c.actor, c.emotion, c.intensity, c.sentence, c.split) == ("crema-1002", "happy", "high", "IEO", "heldout")
    assert datasets.parse_crema("1001_DFA_ANG_XX").split == "dev"


def test_parse_ravdess_keeps_only_video_only_speech():
    c = datasets.parse_ravdess("02-01-06-02-01-02-12")
    assert (c.actor, c.emotion, c.intensity, c.sentence, c.split) == ("ravdess-12", "afraid", "strong", "s01", "heldout")
    assert datasets.parse_ravdess("01-01-06-02-01-02-12") is None   # has audio
    assert datasets.parse_ravdess("02-02-06-02-01-02-12") is None   # song


def test_frame_indices_stay_in_middle_third():
    idx = prep.frame_indices(90)
    assert len(idx) == 5 and idx[0] == 30 and idx[-1] == 60
    assert prep.frame_indices(0) == []
    assert all(0 <= i < 3 for i in prep.frame_indices(3))


def test_named_crops_are_center_crops_of_the_loose_one():
    from PIL import Image

    img = Image.new("RGB", (660, 660))
    assert prep.apply_crop(img, "loose").size == (768, 768)
    assert abs(prep.center_fraction("medium") - 1.6 / 2.2) < 1e-9


def test_select_keeps_a_clips_frames_in_one_shard():
    from cogmirror import m1

    recs = [{"dataset": "crema", "clip": f"c{i}", "k": k, "face_found": True} for i in range(7) for k in range(5)]
    shards = [m1.select(recs, (1, 3), s, 3) for s in range(3)]
    assert sum(len(s) for s in shards) == 14
    for s in shards:
        clips = [r["clip"] for r in s]
        assert all(clips.count(c) == 2 for c in set(clips))
    assert len({r["clip"] for r in m1.select(recs, (1, 3), 0, 1, limit=3)}) == 3
