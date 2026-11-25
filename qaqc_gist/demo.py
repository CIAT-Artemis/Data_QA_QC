"""A small test set with the problems QA/QC GIST is built to catch.

Built from the public sample photographs that ship with scikit-image, plus
planted problems: burst shots (shifted crops), a mirrored view, an exact
copy, a blurred copy, a blank frame, a tiny image and a corrupt file.

Real photographs matter: an image model sees simple synthetic drawings as
near-identical, so a synthetic demo would say nothing about the method.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
from PIL import Image, ImageFilter

PHOTOS = ("astronaut", "chelsea", "coffee", "rocket", "hubble_deep_field",
          "immunohistochemistry", "camera", "coins", "moon", "text", "clock",
          "brick", "grass", "gravel", "logo", "retina", "colorwheel", "page", "horse")


def make_demo_images(dest):
    try:
        from skimage import data as skdata
    except ImportError as e:
        raise ImportError("the demo needs scikit-image: pip install 'qaqc-gist[demo]'") from e
    os.makedirs(dest, exist_ok=True)
    photos = {}
    for name in PHOTOS:
        try:
            arr = getattr(skdata, name)()
        except Exception:
            continue          # a few photos need a download; skip any that fail
        if arr.dtype == bool:
            arr = arr.astype(np.uint8) * 255
        img = Image.fromarray(np.asarray(arr)).convert("RGB")
        photos[name] = img
        img.save(os.path.join(dest, f"{name}.jpg"), quality=92)
    if len(photos) < 6:
        raise RuntimeError("could not build the demo set from scikit-image's sample photos")

    def burst(img, shift):    # the same shot a moment later: a 3% shifted crop
        w, h = img.size
        dx, dy = int(w * 0.03 * shift), int(h * 0.03 * shift)
        return img.crop((dx, dy, w - int(w * 0.03) + dx, h - int(h * 0.03) + dy)).resize((w, h))

    names = list(photos)
    for name in names[:3]:
        burst(photos[name], 1).save(os.path.join(dest, f"{name}_burst.jpg"), quality=92)
    photos[names[1]].transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(dest, f"{names[1]}_mirrored.jpg"), quality=92)
    shutil.copyfile(os.path.join(dest, f"{names[3]}.jpg"), os.path.join(dest, f"{names[3]}_exact_copy.jpg"))
    photos[names[2]].filter(ImageFilter.GaussianBlur(8)).save(os.path.join(dest, f"{names[2]}_blurred.jpg"), quality=92)
    Image.new("RGB", (512, 512), (128, 128, 128)).save(os.path.join(dest, "blank.jpg"))
    photos[names[0]].resize((40, 40)).save(os.path.join(dest, "tiny.jpg"))
    with open(os.path.join(dest, "corrupt.jpg"), "wb") as f:
        f.write(b"this is not a jpeg")
    return dest
