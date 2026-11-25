"""Similarity search: find every image that looks like one or more examples.

In QA/QC this is how one bad frame becomes a fix for all of them (find every
image like this thumb-over-the-lens shot, then reject them together), and how
a rare but valuable case is found again in a large collection.

Raw cosine similarity is hard to read on its own: two unrelated photos from
the same field campaign can score 0.85 or more. So each result also gets a
`match` score relative to this dataset: 0 = as similar as a typical random
pair here, 100 = identical.
"""
from __future__ import annotations

import numpy as np


def typical_similarity(E, pairs=4000, seed=0):
    """Median cosine similarity of random pairs of distinct images: what
    "unrelated" looks like in this dataset."""
    n = E.shape[0]
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    m = min(pairs, n * 4)
    a, b = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = a != b
    if not keep.any():
        return None
    return float(np.median(np.einsum("ij,ij->i", E[a[keep]], E[b[keep]])))


def search(E, examples, threshold=0.85, max_results=500, ids=None, candidates=None, typical=None):
    """Images similar to any of the examples.

    E            (n, d) unit-length embeddings
    examples     ids of the query images (rows of E when ids is None)
    threshold    minimum cosine similarity to report
    ids          labels for the rows (default 0..n-1)
    candidates   only search among these ids (e.g. the keep bucket)
    typical      precomputed typical_similarity(E), to save time

    An image's score is its highest similarity to any example. Returns:
        results    [{"id", "similarity", "match"}], most similar first
        typical    the dataset's typical pair similarity
        closest    when nothing reaches the threshold, the nearest image
                   ({"id", "similarity"}), so the threshold can be lowered
                   to fit the dataset instead of assuming search is broken
        compared   how many images were compared
    """
    E = np.asarray(E, dtype=np.float32)
    n = E.shape[0]
    ids = list(range(n)) if ids is None else list(ids)
    row = {iid: i for i, iid in enumerate(ids)}
    q = sorted({row[e] for e in examples if e in row})
    if not q:
        raise KeyError("none of the example images has an embedding")
    score = (E[q] @ E.T).max(axis=0).astype(np.float64)
    mask = np.ones(n, dtype=bool)
    if candidates is not None:
        mask[:] = False
        mask[[row[c] for c in candidates if c in row]] = True
    mask[q] = False
    typ = typical_similarity(E) if typical is None else typical

    def match(s):
        if typ is None or typ >= 1.0:
            return None
        return round(100.0 * min(1.0, max(0.0, (s - typ) / (1.0 - typ))), 1)

    hits = np.flatnonzero(mask & (score >= threshold))
    hits = hits[np.argsort(-score[hits], kind="stable")][:max_results]
    results = [{"id": ids[r], "similarity": round(float(score[r]), 4), "match": match(score[r])}
               for r in hits]
    closest = None
    if not results and mask.any():
        r = int(np.flatnonzero(mask)[np.argmax(score[mask])])
        closest = {"id": ids[r], "similarity": round(float(score[r]), 4)}
    return {"results": results, "typical": None if typ is None else round(typ, 4),
            "closest": closest, "compared": int(mask.sum())}
