"""QA/QC health check: sort every image into one of four buckets.

    keep      passes every check; the pool GIST selects from
    hidden    usable but redundant or too small: near-duplicates of a
              sharper image, and tiny images. Easy to restore.
    outlier   unusual for this dataset (isolated, off-topic or low-signal).
              Often a problem frame, sometimes a rare and valuable one:
              review these.
    rejected  unusable: cannot be opened, blank, or an exact copy of
              another file.

Images also get informational tags (blurry, dark, bright, low_contrast,
aspect) that do not move them between buckets. A reviewer's decision for an
image always wins over every automatic rule, and survives reruns.

Every image gets a plain-language `reason`, so each decision can be audited.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

KEEP, HIDDEN, OUTLIER, REJECTED = "keep", "hidden", "outlier", "rejected"
BUCKETS = (KEEP, HIDDEN, OUTLIER, REJECTED)

DEFAULT_PARAMS = {
    # tags only (the image stays where it is)
    "min_sharpness": 25.0,      # below: "blurry"
    "min_brightness": 40.0,     # mean pixel value below: "dark"
    "max_brightness": 230.0,    # above: "bright" (washed out)
    "min_contrast": 20.0,       # below: "low_contrast"
    "min_aspect": 0.2,          # width / height outside [min, max]: "aspect"
    "max_aspect": 5.0,
    # rules that move images
    "blank_contrast": 5.0,      # contrast below: rejected as blank
    "min_pixels": 2500,         # width x height below: hidden as tiny
    "near_duplicate_similarity": 0.97,  # at or above, to a sharper image: hidden
    "outlier_mad_multiplier": 2.0,      # outlier cut: median + m x MAD
    "neighbours": 10,           # k for the nearest-neighbour graph
}

_ALIASES = {"keep": KEEP, "kept": KEEP, "restore": KEEP, "hidden": HIDDEN, "hide": HIDDEN,
            "outlier": OUTLIER, "outliers": OUTLIER, "rejected": REJECTED, "reject": REJECTED}


def as_bucket(decision):
    """Map a reviewer's word (keep, restore, hide, reject, ...) to a bucket."""
    b = _ALIASES.get(str(decision).strip().lower())
    if b is None:
        raise ValueError(f"unknown decision {decision!r}; use one of {', '.join(BUCKETS)}")
    return b


def nearest_neighbours(E, k=10):
    """Cosine distance to, and row index of, each row's k nearest rows.
    Column 0 is normally the row itself. Returns (None, None) below 3 rows."""
    from sklearn.neighbors import NearestNeighbors
    n = 0 if E is None else E.shape[0]
    if n < 3:
        return None, None
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(E)
    return nn.kneighbors(E)


def _mad_cut(x, m):
    """median + m * MAD. The median absolute deviation has a 50% breakdown
    point, so the outliers themselves cannot inflate the cut."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med + m * max(mad, 1e-9), med


def counts(images):
    c = {b: 0 for b in BUCKETS}
    for r in images:
        c[r["bucket"]] = c.get(r["bucket"], 0) + 1
    return c


def assess(images, E=None, norms=None, rows=None, params=None, decisions=None, knn=None):
    """Put every image record in a bucket, with a reason.

    images     records from scan.scan()
    E          unit-length embeddings, one row per embedded image (optional:
               without them only the file and pixel checks run)
    norms      the embeddings' lengths before scaling (for the signal check)
    rows       rows[i] = id of the image in row i of E
    params     overrides for DEFAULT_PARAMS
    decisions  {image id: bucket or word} set by a reviewer; always respected
    knn        a precomputed nearest_neighbours(E) result, to save time

    Changes the records in place and returns the bucket counts.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    decisions = {k: as_bucket(v) for k, v in (decisions or {}).items()}
    by_id = {r["id"]: r for r in images}

    # 0. start clean; reviewer decisions first
    for r in images:
        r.update(bucket=KEEP, reason="", issues=[], manual=False, duplicate_of=None,
                 outlier_votes=0, outlier_checks=[])
        if r["id"] in decisions:
            r.update(bucket=decisions[r["id"]], reason="set by a reviewer", manual=True)

    # 1. file and pixel checks
    for r in images:
        if not r["readable"]:
            if not r["manual"]:
                r.update(bucket=REJECTED, reason="could not be opened as an image")
            continue
        iss = []
        if r["width"] * r["height"] < p["min_pixels"]:
            iss.append("tiny")
        if r["aspect"] and not (p["min_aspect"] <= r["aspect"] <= p["max_aspect"]):
            iss.append("aspect")
        # a readable image is always measured, so 0 is a real value here
        # (a perfectly flat frame has contrast 0 and must count as blank)
        if r["sharpness"] < p["min_sharpness"]:
            iss.append("blurry")
        if r["brightness"] < p["min_brightness"]:
            iss.append("dark")
        if r["brightness"] > p["max_brightness"]:
            iss.append("bright")
        if r["contrast"] < p["min_contrast"]:
            iss.append("low_contrast")
        blank = r["contrast"] < p["blank_contrast"]
        if blank:
            iss.append("blank")
        r["issues"] = iss
        if r["manual"]:
            continue
        if blank:
            r.update(bucket=REJECTED, reason=f"blank: almost no detail (contrast {r['contrast']})")
        elif "tiny" in iss:
            r.update(bucket=HIDDEN, reason=f"tiny: {r['width']} x {r['height']} px, "
                                           f"below the {p['min_pixels']:g}-pixel minimum")

    # 2. exact copies (same file bytes): keep the first, reject the rest
    groups = defaultdict(list)
    for r in images:
        if r["readable"] and r["sha256"]:
            groups[r["sha256"]].append(r["id"])
    for ids in groups.values():
        first = by_id[ids[0]]
        for i in ids[1:]:
            r = by_id[i]
            r["duplicate_of"] = ids[0]
            if not r["manual"]:
                r.update(bucket=REJECTED, reason=f"exact copy of #{ids[0]} ({first['rel_path']})")

    if E is None or rows is None or len(rows) < 3:
        return counts(images)
    E = np.asarray(E, dtype=np.float32)
    if knn is None:
        knn = nearest_neighbours(E, p["neighbours"])
    kd, ki = knn
    nv = len(rows)
    thr = p["near_duplicate_similarity"]

    # 3. near-duplicates, non-transitive: each image points at its most similar
    #    neighbour that is sharper, and chains are followed to the sharpest
    #    image. A~B and B~C never make A a duplicate of C unless they are
    #    actually linked through sharper images.
    def better(a, b):   # is image a sharper than image b? (ties: lower id)
        qa, qb = by_id[a]["sharpness"], by_id[b]["sharpness"]
        return qa > qb or (qa == qb and a < b)

    link = {}
    for ri in range(nv):
        r = by_id[rows[ri]]
        if r["manual"] or r["bucket"] == REJECTED:
            continue
        best, best_sim = None, -1.0
        for j in range(kd.shape[1]):
            nri = int(ki[ri, j])
            if nri == ri:
                continue
            sim = 1.0 - float(kd[ri, j])
            if sim < thr:
                break       # neighbours are sorted: the rest are less similar
            if by_id[rows[nri]]["bucket"] == REJECTED:
                continue
            if better(rows[nri], rows[ri]) and sim > best_sim:
                best, best_sim = nri, sim
        if best is not None:
            link[ri] = (best, best_sim)

    for ri, (nri, sim) in link.items():
        a, seen = nri, {ri}
        while a in link and a not in seen:   # follow to the sharpest copy
            seen.add(a)
            a = link[a][0]
        r = by_id[rows[ri]]
        if r["bucket"] == REJECTED:
            continue
        kept = by_id[rows[a]]
        reason = f"near-duplicate of #{rows[nri]} (similarity {sim:.3f})"
        if a != nri:
            reason += f"; the sharpest copy, #{rows[a]} ({kept['rel_path']}), is kept"
        else:
            reason += f", which is sharper and is kept ({kept['rel_path']})"
        r.update(bucket=HIDDEN, reason=reason, duplicate_of=rows[a])

    # 4. outliers: three detectors, each with a robust median + m * MAD cut.
    #    An image is an outlier if any of them flags it; more votes = surer.
    m = p["outlier_mad_multiplier"]
    votes = np.zeros(nv, dtype=int)
    why = [[] for _ in range(nv)]
    kth = kd.shape[1] - 1
    dk = kd[:, -1]                                   # isolated: no close look-alikes
    cut, med = _mad_cut(dk, m)
    for ri in np.flatnonzero(dk > cut):
        votes[ri] += 1
        why[ri].append(f"isolated: even its {kth}th most similar image is only "
                       f"{1 - dk[ri]:.2f} alike (typical {1 - med:.2f})")
    c = E.mean(axis=0)                               # off-topic: far from the whole set
    c = c / max(1e-9, float(np.linalg.norm(c)))
    dc = 1.0 - E @ c
    cut, med = _mad_cut(dc, m)
    for ri in np.flatnonzero(dc > cut):
        votes[ri] += 1
        why[ri].append(f"off-topic: similarity to the dataset as a whole is "
                       f"{1 - dc[ri]:.2f} (typical {1 - med:.2f})")
    if norms is not None and len(norms) == nv and float(np.ptp(norms)) > 1e-3:
        nrm = np.asarray(norms, dtype=np.float64)  # low-signal: washed-out, odd frames
        nmed = float(np.median(nrm))
        nd = np.abs(nrm - nmed)
        cut, _ = _mad_cut(nd, m)
        for ri in np.flatnonzero(nd > cut):
            votes[ri] += 1
            why[ri].append(f"unusual signal strength (embedding length {nrm[ri]:.1f}, "
                           f"typical {nmed:.1f})")
    for ri in range(nv):
        r = by_id[rows[ri]]
        r["outlier_votes"] = int(votes[ri])
        r["outlier_checks"] = why[ri]
        if votes[ri] and not r["manual"] and r["bucket"] == KEEP:
            r.update(bucket=OUTLIER, reason=f"outlier ({votes[ri]} of 3 checks): " + "; ".join(why[ri]))
    return counts(images)
