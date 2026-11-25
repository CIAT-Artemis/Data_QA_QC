"""Find images, open each one, and measure simple quality signals.

Every image gets a record (a plain dict) that later stages add to:

    id, path, rel_path, filename, folder
    readable      the file opened and decoded as an image
    width, height, aspect (width / height)
    sharpness     variance of the Laplacian of a 224 x 224 grey copy;
                  sharp images have strong edges, so higher is sharper
    brightness    mean pixel value, 0 (black) to 255 (white)
    contrast      standard deviation of pixel values
    sha256        hash of the file bytes, used to find exact copies
"""
from __future__ import annotations

import hashlib
import io
import os
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps, ImageStat

from .util import progress

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(root):
    """Image files under root, recursively, in a stable order. Hidden
    folders, macOS resource forks and __MACOSX folders are skipped."""
    paths = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if not d.startswith(".") and d != "__MACOSX")
        for f in sorted(fns):
            if os.path.splitext(f)[1].lower() in VALID_EXT and not f.startswith("._"):
                paths.append(os.path.join(dp, f))
    return paths


def resolve_input(path, work_dir):
    """A folder is used as it is; a .zip file is extracted once into work_dir.
    (zipfile drops absolute paths and '..' parts when extracting.)"""
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"image path not found: {path}")
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        dest = os.path.join(work_dir, "unzipped_" + hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:8])
        if not os.path.isdir(dest):
            with zipfile.ZipFile(path) as z:
                z.extractall(dest)
        return dest
    if not os.path.isdir(path):
        raise NotADirectoryError(f"expected a folder or a .zip file: {path}")
    return path


def open_image(src):
    """Open a path or bytes as an RGB image, turned the way the camera's
    EXIF orientation tag says (as a person would see it)."""
    img = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    img.load()
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass   # a broken EXIF block must not make a good image unreadable
    return img.convert("RGB")


def laplacian_variance(gray):
    """Variance of the 4-neighbour Laplacian, with mirrored edges (the same
    result as scipy.ndimage.laplace(gray).var())."""
    g = np.pad(np.asarray(gray, dtype=np.float64), 1, mode="symmetric")
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def inspect_image(path):
    """Hash, open and measure one image. Never raises: a file that cannot be
    read or decoded comes back with readable=False."""
    rec = {"readable": False, "width": 0, "height": 0, "aspect": 0.0,
           "sharpness": 0.0, "brightness": 0.0, "contrast": 0.0, "sha256": ""}
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return rec
    rec["sha256"] = hashlib.sha256(data).hexdigest()
    try:
        img = open_image(data)
    except Exception:
        return rec
    w, h = img.size
    gray = np.asarray(img.convert("L").resize((224, 224)), dtype=np.float64)
    stat = ImageStat.Stat(img)
    rec.update(readable=True, width=w, height=h, aspect=round(w / max(h, 1), 3),
               sharpness=round(laplacian_variance(gray), 2),
               brightness=round(sum(stat.mean[:3]) / 3.0, 1),
               contrast=round(sum(stat.stddev[:3]) / 3.0, 1))
    return rec


def sample_paths(paths, max_images=0, seed=0):
    """A fixed random sample of max_images paths (all of them if 0)."""
    if max_images and 0 < max_images < len(paths):
        return sorted(random.Random(seed).sample(paths, max_images))
    return list(paths)


def inspect_many(paths, workers=None, show_progress=True):
    """inspect_image for many paths, in parallel, in order."""
    workers = workers or max(2, min(8, (os.cpu_count() or 2) * 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(progress(pool.map(inspect_image, paths), len(paths), "inspect", show_progress))


def make_record(i, path, root, info):
    return {"id": i, "path": path, "rel_path": os.path.relpath(path, root),
            "filename": os.path.basename(path),
            "folder": os.path.basename(os.path.dirname(path)), **info}


def scan(root, paths=None, workers=None, show_progress=True):
    """Inspect every image under root (or the given paths). Returns one
    record per image, with ids 0..n-1 in path order."""
    paths = collect_images(root) if paths is None else list(paths)
    infos = inspect_many(paths, workers, show_progress)
    return [make_record(i, p, root, info) for i, (p, info) in enumerate(zip(paths, infos))]
