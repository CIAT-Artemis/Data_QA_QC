# The QA/QC GIST method

This document describes every step of the method, with the formulas and
default settings, so results can be understood, audited and reproduced. The
code for each step is named in brackets.

1. [Scan: open, measure, hash](#1-scan-open-measure-hash)
2. [Embeddings](#2-embeddings)
3. [QA/QC health check](#3-qaqc-health-check)
4. [Reviewer decisions](#4-reviewer-decisions)
5. [Similarity search](#5-similarity-search)
6. [GIST selection](#6-gist-selection)
7. [How it compares to the GIST paper](#how-it-compares-to-the-gist-paper)
8. [Settings](#settings)
9. [Limitations](#limitations)

---

## 1. Scan: open, measure, hash

[`scan.py`] Every `.jpg`, `.jpeg`, `.png` and `.webp` file under the input
folder is found (recursively; hidden folders, `__MACOSX` and `._` resource
forks are skipped) and inspected once:

| Field | Definition |
|---|---|
| `readable` | the file decodes as an image |
| `width`, `height`, `aspect` | after applying the EXIF orientation tag; aspect = width / height |
| `sharpness` | variance of the 4-neighbour Laplacian of a 224 × 224 greyscale copy. Edges give large Laplacian values, so sharp images score high and blurred ones low |
| `brightness` | mean pixel value over the R, G, B channels (0 = black, 255 = white) |
| `contrast` | standard deviation of pixel values, averaged over R, G, B |
| `sha256` | hash of the file bytes |

## 2. Embeddings

[`embed.py`] Each readable image is turned into an embedding with DINOv3
(default ViT-L/16, `vit_large_patch16_dinov3.lvd1689m` from timm), using the
model's own preprocessing. DINOv3 is a self-supervised vision model: it was
not trained on any label set, so its embeddings describe what an image looks
like in general (scene, objects, texture, framing), which is what QA/QC
needs.

Each embedding **e** is scaled to unit length, **ê** = **e** / ‖**e**‖, so the
similarity of two images is a dot product:

    sim(i, j) = ê_i · ê_j        (cosine similarity: 1 = identical, lower = more different)

The original length ‖**e**‖ is kept: it is one of the outlier signals (step 3).

On a GPU the model runs in mixed precision (fp16 autocast). Large vision
transformers have a few very large activations that can overflow fp16; any
batch that comes back with NaN or inf is redone in fp32, and fp32 is used
from then on. A run never continues with invalid embeddings.

## 3. QA/QC health check

[`health.py`] Every image ends up in exactly one bucket, with a reason.

| Bucket | Meaning | What to do with it |
|---|---|---|
| **keep** | passed every check | the pool GIST selects from |
| **hidden** | usable but redundant or too small | restore any that are wanted |
| **outlier** | unusual for this dataset | review: often a problem frame, sometimes a rare, valuable one |
| **rejected** | unusable | set aside |

The checks run in this order.

### 3.1 File and pixel checks

| Rule | Condition (defaults) | Result |
|---|---|---|
| unreadable | the file does not decode | rejected |
| blank | contrast < 5 | rejected |
| tiny | width × height < 2,500 | hidden |
| blurry | sharpness < 25 | tag |
| dark / bright | brightness < 40 / > 230 | tag |
| low contrast | contrast < 20 | tag |
| odd shape | aspect outside [0.2, 5] | tag |

Tags are informational: they are recorded in `results.csv` and in the
`issues` field but do not move an image. Sharpness still matters later: it is
the utility in GIST, and near-duplicates keep the sharpest copy.

### 3.2 Exact copies

Images with the same SHA-256 hash are byte-for-byte copies. The first (in
path order) is kept; the others are rejected with
`exact copy of #<id> (<path>)`.

### 3.3 Near-duplicates (non-transitive)

Burst shots, re-shoots of the same plot and lightly edited copies are not
byte-identical, but their embeddings are nearly the same.

For each image *i*, look at its 10 nearest neighbours (cosine distance).
Among those with sim(i, j) ≥ τ (default τ = 0.97) that are **sharper** than *i*
(ties: lower id), link *i* to the most similar one. Then follow the links to
the end of the chain: that image is the sharpest copy and is kept; every
image that links into it is hidden, with the reason naming both its most
similar sharper neighbour and the copy that is kept.

This is deliberately **non-transitive**. Grouping all images connected by
similarity (A ~ B and B ~ C ⇒ A, B, C one group) collapses gradual series,
such as a walk along a row of plants, into one image. Here an image is only
hidden when it has a sharper, near-identical neighbour of its own.

### 3.4 Outliers

Three independent detectors, each with a robust cut:

    cut(x) = median(x) + m × MAD(x),     MAD(x) = median(|x − median(x)|)

The median absolute deviation has a 50% breakdown point, so the outliers
themselves cannot drag the cut upwards (a mean and standard deviation would
be pulled by them). Default m = 2; raise it to flag fewer images.

| Detector | Signal | Catches |
|---|---|---|
| isolated | distance to the 10th nearest neighbour, 1 − sim | one-of-a-kind frames: a hand over the lens, the sky, a misfire |
| off-topic | distance to the dataset's mean direction, 1 − ê_i · **c**, with **c** = mean(ê) / ‖mean(ê)‖ | a whole cluster of the wrong subject (e.g. maize photos in a bean dataset). Such a cluster passes the isolation test, because its members are close to each other, but the whole cluster is far from the centre |
| signal strength | \|‖**e**_i‖ − median‖**e**‖\| | washed-out, low-information frames that cosine similarity alone does not see (skipped when all lengths are equal, e.g. pre-normalised embeddings) |

An image flagged by at least one detector moves from keep to outlier. The
number of detectors that flagged it (1 to 3) is recorded as `outlier_votes`;
review the most-flagged first. Outliers are a review queue, not a reject
list: in field data, rare growth stages, diseases or lighting conditions are
outliers too, and they are often exactly the images a model needs.

## 4. Reviewer decisions

[`health.assess(decisions=...)`, `Dataset.assess`] A reviewer can put any
image in any bucket. Decisions are applied first and are never overridden by
automatic rules, on this run or any rerun with other settings. They are
written to `decisions.csv` (relative path, decision) and can be passed back
with `--decisions`, which makes the review itself part of the record.

## 5. Similarity search

[`search.py`] Given one or more example images, every other image is scored
by its highest similarity to any example, and those at or above a threshold
are returned, most similar first. Search can be limited to some buckets.

Raw cosine similarity is hard to read on its own: in a field campaign, two
unrelated photos of the same crop can score 0.85 or more. So each result also
gets a **match** score relative to the dataset:

    t      = median similarity of 4,000 random pairs of different images   ("typical")
    match  = 100 × clip((sim − t) / (1 − t), 0, 1)

0 means "as similar as a typical random pair here", 100 means identical. When
nothing reaches the threshold, the closest image and its similarity are
reported, so the threshold can be set for the dataset rather than assuming
search is broken.

In QA/QC, search turns one example into a fix for every case: find all
images like this thumb-over-the-lens frame and reject them together, or find
more of a rare case to make sure it survives into the selection.

## 6. GIST selection

[`gist.py`] GIST picks exactly *k* images from the keep bucket (or any
candidate set) that are both good and different from each other.

### Utility

Without preferences, utility is sharpness rescaled to [0, 1] over the
candidates:

    u_i = (s_i − min s) / (max s − min s)          (all 1 when every s is equal)

With reviewer preferences (liked set L, disliked set D):

    p⁺_i = max_{l ∈ L} sim(i, l)        p⁻_i = max_{d ∈ D} sim(i, d)
    p_i  = rescale(p⁺_i − p⁻_i) to [0, 1]
    u_i  = (1 − w) × sharpness_i + w × p_i,      w = 0.7
    u_l  = 1 for every liked image

Disliked images, and every image more than 0.95 similar to a disliked one,
are excluded outright.

### Stage A: near-identical suppression

Visit images from most to least useful. Each image not yet suppressed becomes
a **representative** and suppresses every other image with
sim ≥ θ (default θ = 0.98). Suppressed images never suppress others
(non-transitive). This is a greedy independent set on the graph that links
near-identical images, taken in utility order, so the representative of each
near-identical group is its most useful member.

### Stage B: useful and diverse

Start with the most useful representative. Then repeatedly add the
representative *i* with the highest score

    score(i) = α × u_i + (1 − α) × (1 − max_{j ∈ S} sim(i, j))

where *S* is the set picked so far. The second term is the diversity bonus:
an image very similar to something already picked earns almost nothing, an
image unlike anything picked earns the most. α = 1 picks by utility alone;
α = 0 is farthest-point (max-min) sampling; the default α = 0.75 favours good
images but still spreads the selection. Ties go to the lower id, so a run is
deterministic.

### Stage C: exact-k fill

If Stage A left fewer than *k* representatives, the remaining places are
filled from the suppressed images with the same score, so the result has
exactly *k* images. Excluded (disliked) images are never used, so with many
dislikes a selection can be smaller than *k*.

### Output

The picked ids in pick order (the first picks are the most useful and most
different), and, for every candidate left out, why: near-identical to a named
image (with the similarity), excluded by a dislike, or ranked below the
picked set (with its similarity to the most similar picked image).

### Cost

Stage A and B use one matrix-vector product per representative and per pick,
so time grows with n × k × d. On a laptop CPU, a few thousand images take
well under a second; tens of thousands take seconds to minutes. Embedding is
by far the slowest step, and it is cached.

## How it compares to the GIST paper

Fahrbach et al. (2025) study max-min diversification with monotone
submodular utility:

    maximise  f(S) = g(S) + λ × div(S)   subject to |S| ≤ k,
    div(S) = min distance between two distinct members of S

Their GIST algorithm sweeps a grid of distance thresholds *d*; for each, it
greedily builds an independent set (members at least *d* apart) by marginal
utility gain, and returns the best solution found. It achieves a
(1/2 − ε)-approximation, and (2/3 − ε) for linear utility; they show the
problem is NP-hard to approximate beyond 0.5584.

QA/QC GIST uses the same ingredients in a simplified form built for
interactive review:

| | GIST paper | QA/QC GIST |
|---|---|---|
| Utility | monotone submodular g(S) | linear: sharpness and reviewer preferences |
| Diversity | min pairwise distance, weighted by λ | per-step bonus 1 − max similarity to the picked set, weighted by 1 − α |
| Thresholds | geometric sweep of *d*; best solution kept | one fixed near-identical threshold θ (Stage A) |
| Independent set | at every threshold | once, at θ, to remove near-identical images |
| Size | at most *k* | exactly *k* (Stage C fill) |
| Guarantee | 1/2 − ε approximation | none claimed |

The simplification makes a run fast enough to repeat while reviewing and
makes every decision explainable in one sentence. We do not claim the
paper's approximation guarantee for it.

## Settings

| Setting | Default | Step |
|---|---|---|
| `min_sharpness` | 25 | tag blurry |
| `min_brightness` / `max_brightness` | 40 / 230 | tag dark / bright |
| `min_contrast` | 20 | tag low contrast |
| `min_aspect` / `max_aspect` | 0.2 / 5 | tag odd shape |
| `blank_contrast` | 5 | reject blank |
| `min_pixels` | 2,500 | hide tiny |
| `near_duplicate_similarity` | 0.97 | hide near-duplicates |
| `outlier_mad_multiplier` | 2.0 | outlier cut |
| `neighbours` | 10 | nearest-neighbour graph |
| `k` | 100 | GIST size |
| `alpha` | 0.75 | GIST quality vs diversity |
| `duplicate_threshold` | 0.98 | GIST Stage A |
| `dislike_radius` | 0.95 | exclusion around disliked images |
| `preference_weight` | 0.7 | share of utility from preferences |

Pass health-check settings with `--params '{"outlier_mad_multiplier": 3}'` or
`analyse(params={...})`.

## Limitations

- **Thresholds depend on the embedding model.** They were tuned on DINOv3
  ViT-L/16. Smaller models and other models spread similarities differently;
  check the near-duplicate pairs and the outliers on a first run.
- **Sharpness is a proxy for quality.** The variance of the Laplacian also
  rises with texture and noise: a sharp image of a smooth leaf can score
  lower than a noisy image of soil. Reviewer preferences exist to correct
  this where it matters.
- **Outliers are relative to the dataset.** In a collection that is mostly
  wrong, the wrong images are the norm. Outliers are a review queue, not a
  reject list.
- **One crop or campaign at a time works best.** The off-topic check measures
  distance from the dataset's centre; mixing very different collections in
  one run makes the smaller ones look off-topic.
- **Similarity is visual.** Two images of different varieties that look alike
  are near-duplicates to the method. Where such differences matter, run QA/QC
  per variety or treatment.

## References

- M. Fahrbach, S. Ramalingam, M. Zadimoghaddam, S. Ahmadian, G. Citovsky,
  G. DeSalvo. GIST: Greedy Independent Set Thresholding for Max-Min
  Diversification with Submodular Utility. NeurIPS 2025. arXiv:2405.18754.
- O. Siméoni et al. DINOv3. 2025. arXiv:2508.10104.
- P. J. Rousseeuw, C. Croux. Alternatives to the Median Absolute Deviation.
  Journal of the American Statistical Association, 1993.
