import numpy as np
import pytest

from qaqc_gist.gist import select


def test_exactly_k_distinct_and_complete(clusters):
    E, _ = clusters(n_clusters=6, per=5, tight=False)
    q = np.random.default_rng(1).random(len(E))
    r = select(E, 10, quality=q)
    assert len(r["selected"]) == 10 == len(set(r["selected"]))
    assert sorted(r["selected"] + r["not_selected"]) == list(range(len(E)))
    assert set(r["reasons"]) == set(r["not_selected"])


def test_equal_quality_covers_every_group_first(clusters):
    E, lab = clusters(n_clusters=5, per=6, tight=False)
    r = select(E, 5)                      # no quality signal: diversity decides
    assert sorted(lab[r["selected"]]) == [0, 1, 2, 3, 4]


def test_alpha_zero_is_max_min_spread(clusters):
    E, lab = clusters(n_clusters=4, per=6, tight=False)
    q = np.random.default_rng(2).random(len(E))
    r = select(E, 4, quality=q, alpha=0.0)
    assert sorted(lab[r["selected"]]) == [0, 1, 2, 3]


def test_alpha_one_is_quality_order():
    E = np.random.default_rng(3).normal(size=(30, 256))   # nearly orthogonal: no suppression
    q = np.random.default_rng(4).random(30)
    r = select(E, 8, quality=q, alpha=1.0)
    assert r["selected"] == list(np.argsort(-q)[:8])


def test_near_identical_suppressed_then_filled_to_k(clusters):
    E, lab = clusters(n_clusters=5, per=6, tight=True)    # within a group: > 0.99 alike
    q = np.random.default_rng(5).random(len(E))
    r = select(E, 8, quality=q)
    assert len(r["selected"]) == 8
    assert sorted(lab[r["selected"][:5]]) == [0, 1, 2, 3, 4]   # stage B: one per group first
    first = r["selected"][0]
    assert q[first] == q[lab == lab[first]].max()               # each group's sharpest
    for iid in r["not_selected"]:
        assert r["reasons"][iid].startswith("near-identical to #")
        assert lab[r["duplicate_of"][iid]["anchor"]] == lab[iid]


def test_dislikes_exclude_the_image_and_its_near_copies(clusters):
    E, lab = clusters(n_clusters=4, per=5, tight=True)
    r = select(E, 15, disliked=[0])
    group0 = set(np.flatnonzero(lab == 0))
    assert not group0 & set(r["selected"])
    assert r["reasons"][0] == "disliked by a reviewer"
    assert all(r["reasons"][i].startswith("too similar to disliked image #0") for i in group0 - {0})
    assert len(r["selected"]) == 15                         # every other image


def test_liked_image_is_picked_first(clusters):
    E, _ = clusters(n_clusters=5, per=6, tight=False)
    q = np.arange(len(E), dtype=float)                      # image 0 is the least sharp
    r = select(E, 3, quality=q, liked=[0])
    assert r["selected"][0] == 0 and r["liked_used"] == 1


def test_small_and_degenerate_cases():
    E = np.random.default_rng(6).normal(size=(5, 16))
    assert select(E, 0)["selected"] and len(select(E, 0)["selected"]) == 5
    assert len(select(E, 10)["selected"]) == 5
    assert len(select(E[:2], 1)["selected"]) == 2           # under 3 images: keep them all
    assert select(np.zeros((0, 16)), 3)["selected"] == []


def test_string_ids_and_determinism(clusters):
    E, _ = clusters(n_clusters=3, per=4, tight=False)
    ids = [f"img_{i}.jpg" for i in range(len(E))]
    a = select(E, 5, ids=ids, liked=["img_3.jpg"])
    b = select(E, 5, ids=ids, liked=["img_3.jpg"])
    assert a["selected"] == b["selected"] and "img_3.jpg" in a["selected"]


def test_bad_input():
    with pytest.raises(ValueError):
        select(np.ones((4, 3)), 2, ids=[1, 2])
    with pytest.raises(ValueError):
        select(np.ones((4, 3)), 2, alpha=1.5)
