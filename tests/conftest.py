import numpy as np
import pytest
from PIL import Image


def unit(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture
def clusters():
    """n_clusters groups of points; within a group every pair is above 0.99
    cosine similarity when tight=True, around 0.8 otherwise."""
    def make(n_clusters=5, per=6, d=64, tight=True, seed=0):
        rng = np.random.default_rng(seed)
        centres = unit(rng.normal(size=(n_clusters, d)))
        noise = 0.005 if tight else 0.06
        E = np.vstack([c + noise * rng.normal(size=(per, d)) for c in centres])
        return unit(E), np.repeat(np.arange(n_clusters), per)
    return make


class PixelEmbedder:
    """Stand-in for DINOv3 in tests: 8 x 8 colour thumbnail, centred.
    Shifted crops of a smooth image stay very similar; different smooth
    random images do not."""
    name = "pixel-test-embedder"

    def __init__(self):
        self.calls = 0
        self.embedded = []
        self.failed = []

    def __call__(self, paths):
        self.calls += 1
        self.embedded += list(paths)
        out = []
        for p in paths:
            a = np.asarray(Image.open(p).convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
            out.append((a - 0.5).ravel())
        return np.array(out)


@pytest.fixture
def embedder():
    return PixelEmbedder()


def smooth_image(seed, size=(160, 120)):
    rng = np.random.default_rng(seed)
    grid = (rng.random((3, 4, 3)) * 255).astype(np.uint8)
    return Image.fromarray(grid).resize(size, Image.BICUBIC)


@pytest.fixture
def image_folder(tmp_path):
    """12 different images in two sub-folders, plus planted problems."""
    root = tmp_path / "images"
    (root / "plot_a").mkdir(parents=True)
    (root / "plot_b").mkdir()
    for i in range(12):
        sub = "plot_a" if i < 6 else "plot_b"
        smooth_image(i).save(root / sub / f"img_{i:02d}.jpg", quality=95)
    img = smooth_image(0)
    w, h = img.size
    img.crop((3, 2, w - 1, h)).resize((w, h)).save(root / "plot_a" / "img_00_burst.jpg", quality=95)
    (root / "plot_b" / "img_06_copy.jpg").write_bytes((root / "plot_b" / "img_06.jpg").read_bytes())
    Image.new("RGB", (200, 200), (128, 128, 128)).save(root / "blank.jpg")
    smooth_image(3, (40, 40)).save(root / "tiny.jpg")
    (root / "corrupt.jpg").write_bytes(b"not an image")
    (root / ".hidden").mkdir()
    smooth_image(99).save(root / ".hidden" / "skip_me.jpg")
    return root
