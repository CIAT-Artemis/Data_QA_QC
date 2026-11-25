import pytest

from qaqc_gist.search import search, typical_similarity


def test_finds_the_examples_group_most_similar_first(clusters):
    E, lab = clusters(n_clusters=4, per=6, tight=False)
    res = search(E, [0], threshold=0.5)
    found = [h["id"] for h in res["results"]]
    assert set(found) == set(range(1, 6))                     # its group, not itself
    sims = [h["similarity"] for h in res["results"]]
    assert sims == sorted(sims, reverse=True)
    assert all(0 <= h["match"] <= 100 for h in res["results"])
    assert res["typical"] < 0.5


def test_empty_result_names_the_closest(clusters):
    E, _ = clusters(n_clusters=4, per=6, tight=False)
    res = search(E, [0], threshold=0.9999)
    assert res["results"] == []
    assert res["closest"]["id"] in range(1, 6)


def test_several_examples_and_candidates(clusters):
    E, lab = clusters(n_clusters=4, per=6, tight=False)
    res = search(E, [0, 6], threshold=0.5, candidates=[1, 2, 7, 13])
    assert {h["id"] for h in res["results"]} == {1, 2, 7}
    assert res["compared"] == 4


def test_unknown_example():
    import numpy as np
    with pytest.raises(KeyError):
        search(np.eye(3), ["nope"], ids=["a", "b", "c"])


def test_typical_similarity_small_inputs():
    import numpy as np
    assert typical_similarity(np.ones((1, 4))) is None
