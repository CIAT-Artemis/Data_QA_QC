"""GIST: pick exactly k images that are both good and different.

Two goals pull against each other:

    utility    each picked image should be worth having on its own.
               Here: sharpness, rescaled to 0..1 over the candidates, plus
               reviewer preferences (liked / disliked examples) when given.
    diversity  picked images should differ from each other: an image adds
               little if something very similar is already picked.

alpha sets the balance: 1.0 = utility only, 0.0 = diversity only.

The selection runs in three stages:

    A  near-identical suppression. Visit images from most to least useful;
       each image that survives suppresses every other image at or above
       duplicate_threshold cosine similarity to it. Non-transitive: a
       suppressed image never suppresses others. This is a greedy
       independent set on the "near-identical" graph.
    B  useful-and-diverse greedy. Start with the most useful survivor, then
       repeatedly add the survivor with the highest
           alpha * utility + (1 - alpha) * (1 - max similarity to the picked set)
    C  exact-k fill. If stage A left fewer than k survivors, fill up from the
       suppressed images with the same score, so the result has exactly k
       images (unless reviewer dislikes rule out too many).

This follows the max-min diversification with utility objective of
Fahrbach et al., "GIST: Greedy Independent Set Thresholding for Max-Min
Diversification with Submodular Utility", NeurIPS 2025 (arXiv:2405.18754),
in a simplified form built for interactive QA/QC; see docs/METHOD.md for how
it differs from the paper's algorithm.
"""
from __future__ import annotations

import numpy as np


def _unit(E):
    E = np.asarray(E, dtype=np.float32)
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-8, None)


def select(E, k, quality=None, alpha=0.75, duplicate_threshold=0.98, ids=None,
           liked=(), disliked=(), dislike_radius=0.95, preference_weight=0.7):
    """Choose k of the n images whose embeddings are the rows of E.

    E                  (n, d) embeddings of the candidate images
    k                  how many to pick; k <= 0 or k >= n picks every eligible image
    quality            (n,) higher = better, e.g. sharpness. None: all equal
    alpha              utility vs diversity balance, 0..1
    duplicate_threshold cosine similarity treated as "the same picture" in stage A
    ids                labels for the rows (default 0..n-1); liked, disliked
                       and the results use these labels
    liked              images to steer towards: their utility is set to the
                       maximum, and images similar to them gain utility
    disliked           images to steer away from: they, and every image more
                       than dislike_radius similar to one of them, are never picked
    preference_weight  share of utility that comes from preferences (rest: quality)

    Returns a dict:
        selected           ids in the order they were picked
        not_selected       the other ids, in input order
        reasons            {id: why it was not selected}
        duplicate_of       {id: {"anchor": id, "similarity": s}} from stage A
        utility            {id: utility used}
        plus the settings and summary counts.
    """
    E = _unit(E)
    n = E.shape[0]
    ids = list(range(n)) if ids is None else list(ids)
    if len(ids) != n:
        raise ValueError(f"{len(ids)} ids for {n} embeddings")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    out = {"k": k, "alpha": alpha, "duplicate_threshold": duplicate_threshold,
           "candidates": n, "selected": [], "not_selected": [], "reasons": {},
           "duplicate_of": {}, "utility": {}, "near_identical_suppressed": 0,
           "excluded_by_dislikes": 0, "liked_used": 0, "disliked_used": 0}
    if n == 0:
        return out

    # ── utility ──
    q = np.zeros(n) if quality is None else np.nan_to_num(np.asarray(quality, dtype=np.float64))
    if len(q) != n:
        raise ValueError(f"{len(q)} quality values for {n} embeddings")
    util = (q - q.min()) / (q.max() - q.min()) if q.max() > q.min() else np.ones(n)
    row = {iid: i for i, iid in enumerate(ids)}
    liked_rows = sorted({row[i] for i in liked if i in row})
    disliked_rows = sorted({row[i] for i in disliked if i in row})
    excluded = {}                       # row -> reason; never selected
    if liked_rows or disliked_rows:
        pos = (E @ E[liked_rows].T).max(axis=1).astype(np.float64) if liked_rows else np.zeros(n)
        neg = np.zeros(n)
        if disliked_rows:
            S = E @ E[disliked_rows].T
            neg = S.max(axis=1).astype(np.float64)
            near = S.argmax(axis=1)
            for r in np.flatnonzero(neg > dislike_radius):
                excluded[int(r)] = (f"too similar to disliked image #{ids[disliked_rows[near[r]]]} "
                                    f"(similarity {neg[r]:.3f})")
            for r in disliked_rows:
                excluded[r] = "disliked by a reviewer"
        pref = pos - neg
        pref = (pref - pref.min()) / (pref.max() - pref.min()) if pref.max() > pref.min() else np.zeros(n)
        util = (1.0 - preference_weight) * util + preference_weight * pref
        util[liked_rows] = 1.0
    out.update(liked_used=len(liked_rows), disliked_used=len(disliked_rows),
               excluded_by_dislikes=len(excluded),
               utility={ids[i]: round(float(util[i]), 4) for i in range(n)})

    eligible = [i for i in range(n) if i not in excluded]
    order = sorted(eligible, key=lambda i: (-util[i], i))   # most useful first

    def finish(selected_rows, dup, mx):
        sel = set(selected_rows)
        out["selected"] = [ids[r] for r in selected_rows]
        out["not_selected"] = [ids[r] for r in range(n) if r not in sel]
        out["duplicate_of"] = {ids[j]: {"anchor": ids[a], "similarity": round(s, 4)}
                               for j, (a, s) in dup.items() if j not in sel}
        out["near_identical_suppressed"] = len(out["duplicate_of"])
        for r in range(n):
            if r in sel:
                continue
            if r in excluded:
                why = excluded[r]
            elif r in dup:
                a, s = dup[r]
                why = f"near-identical to #{ids[a]} (similarity {s:.3f})"
            elif mx is not None:
                why = (f"ranked below the {len(sel)} picked images "
                       f"(most similar picked image: {mx[r]:.2f})")
            else:
                why = "ranked below the picked images"
            out["reasons"][ids[r]] = why
        return out

    if k <= 0 or k >= len(eligible) or n < 3:
        return finish(order, {}, None)

    # ── stage A: near-identical suppression (non-transitive) ──
    suppressed, dup, reps = set(excluded), {}, []
    for i in order:
        if i in suppressed:
            continue
        reps.append(i)
        sims = (E @ E[i]).astype(np.float64)
        for j in np.flatnonzero(sims >= duplicate_threshold):
            j = int(j)
            if j != i and j not in suppressed:
                suppressed.add(j)
                dup[j] = (i, float(sims[j]))

    def best(rows, mx):
        """Row with the highest alpha * utility + (1 - alpha) * (1 - max_sim);
        ties go to the lower row (rows are sorted)."""
        sc = alpha * util[rows] + (1.0 - alpha) * (1.0 - mx[rows])
        ok = ~np.isnan(sc)
        if not ok.any():
            return None
        return int(rows[np.flatnonzero(sc == sc[ok].max())[0]])

    # ── stage B: useful-and-diverse greedy over the survivors ──
    selected = [reps[0]]                    # the most useful survivor
    mx = (E @ E[reps[0]]).astype(np.float64)   # each image's max similarity to the picked set
    pool = np.array(sorted(reps[1:]), dtype=np.int64)
    while len(selected) < k and len(pool):
        b = best(pool, mx)
        if b is None:
            break
        selected.append(b)
        pool = pool[pool != b]
        mx = np.maximum(mx, E @ E[b])

    # ── stage C: fill to exactly k from the suppressed (never the excluded) ──
    if len(selected) < k:
        sel = set(selected)
        pool = np.array(sorted(r for r in suppressed if r not in excluded and r not in sel), dtype=np.int64)
        while len(selected) < k and len(pool):
            b = best(pool, mx)
            if b is None:
                break
            selected.append(b)
            pool = pool[pool != b]
            mx = np.maximum(mx, E @ E[b])

    return finish(selected, dup, mx)
