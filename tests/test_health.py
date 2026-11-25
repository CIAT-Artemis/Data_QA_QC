import numpy as np

from qaqc_gist.health import HIDDEN, KEEP, OUTLIER, REJECTED, assess


def rec(i, **kw):
    r = {"id": i, "path": f"/data/{i}.jpg", "rel_path": f"{i}.jpg", "filename": f"{i}.jpg",
         "folder": "data", "readable": True, "width": 640, "height": 480, "aspect": 1.333,
         "sharpness": 100.0, "brightness": 120.0, "contrast": 50.0, "sha256": f"h{i}"}
    r.update(kw)
    return r


def test_file_and_pixel_checks():
    imgs = [rec(0, readable=False), rec(1, contrast=0.0), rec(2, width=40, height=40),
            rec(3, sharpness=10.0, brightness=20.0, contrast=15.0), rec(4, brightness=240.0, aspect=6.0),
            rec(5)]
    c = assess(imgs)
    assert [r["bucket"] for r in imgs] == [REJECTED, REJECTED, HIDDEN, KEEP, KEEP, KEEP]
    assert imgs[0]["reason"] == "could not be opened as an image"
    assert imgs[1]["reason"].startswith("blank")
    assert imgs[2]["reason"].startswith("tiny: 40 x 40 px")
    assert set(imgs[3]["issues"]) == {"blurry", "dark", "low_contrast"}
    assert set(imgs[4]["issues"]) == {"bright", "aspect"}
    assert c == {KEEP: 3, HIDDEN: 1, OUTLIER: 0, REJECTED: 2}


def test_exact_copies_keep_the_first():
    imgs = [rec(0, sha256="same"), rec(1), rec(2, sha256="same")]
    assess(imgs)
    assert imgs[0]["bucket"] == KEEP
    assert imgs[2]["bucket"] == REJECTED and imgs[2]["duplicate_of"] == 0
    assert imgs[2]["reason"] == "exact copy of #0 (0.jpg)"


def _background(rng, n, d):
    """n ordinary images: around 0.8 alike, no near-duplicates."""
    c = rng.normal(size=d)
    c /= np.linalg.norm(c)
    X = c + 0.06 * rng.normal(size=(n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_near_duplicates_point_at_the_sharpest_copy():
    rng = np.random.default_rng(0)
    d = 64
    bg = _background(rng, 20, d)
    v = rng.normal(size=d); v /= np.linalg.norm(v)
    step = rng.normal(size=d); step -= (step @ v) * v; step /= np.linalg.norm(step)
    chain = np.array([v, v + 0.12 * step, v + 0.24 * step])   # A~B, B~C (> 0.97); A~C a little less
    chain /= np.linalg.norm(chain, axis=1, keepdims=True)
    E = np.vstack([chain, bg])
    imgs = [rec(0, sharpness=300.0), rec(1, sharpness=200.0), rec(2, sharpness=100.0)] + \
           [rec(i) for i in range(3, len(E))]
    assess(imgs, E, np.ones(len(E)), list(range(len(E))), params={"outlier_mad_multiplier": 1e9})
    assert imgs[0]["bucket"] == KEEP
    assert imgs[1]["bucket"] == HIDDEN and imgs[1]["duplicate_of"] == 0
    assert imgs[2]["bucket"] == HIDDEN and imgs[2]["duplicate_of"] == 0   # followed B to A
    assert "the sharpest copy, #0" in imgs[2]["reason"]


def test_an_off_topic_image_is_an_outlier():
    rng = np.random.default_rng(1)
    d = 64
    E = np.vstack([_background(rng, 40, d), rng.normal(size=(1, d))])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    imgs = [rec(i) for i in range(len(E))]
    norms = np.full(len(E), 50.0)
    norms[-1] = 20.0
    assess(imgs, E, norms, list(range(len(E))))
    odd = imgs[-1]
    assert odd["bucket"] == OUTLIER and odd["outlier_votes"] == 3
    assert odd["outlier_votes"] == max(r["outlier_votes"] for r in imgs)
    assert "isolated" in odd["reason"] and "off-topic" in odd["reason"] and "signal" in odd["reason"]


def test_constant_norms_skip_the_signal_check():
    rng = np.random.default_rng(2)
    E = np.vstack([_background(rng, 30, 32), rng.normal(size=(1, 32))])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    imgs = [rec(i) for i in range(len(E))]
    assess(imgs, E, np.ones(len(E)), list(range(len(E))))
    assert not any("signal" in c for r in imgs for c in r["outlier_checks"])


def test_reviewer_decisions_always_win():
    imgs = [rec(0, contrast=3.0), rec(1), rec(2, sha256="x"), rec(3, sha256="x")]
    assess(imgs, decisions={0: "keep", 1: "reject", 3: "restore"})
    assert imgs[0]["bucket"] == KEEP and imgs[0]["manual"] and "blank" in imgs[0]["issues"]
    assert imgs[1]["bucket"] == REJECTED and imgs[1]["reason"] == "set by a reviewer"
    assert imgs[3]["bucket"] == KEEP and imgs[3]["duplicate_of"] == 2
