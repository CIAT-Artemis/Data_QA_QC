"""Image embeddings with DINOv3 (through timm).

An embedding is a list of numbers that describes what an image shows. Two
photos of the same scene get similar embeddings, so the cosine similarity of
two embeddings (1 = same, lower = more different) is how this package
compares images.

The rest of the package works with any embedder: a callable that takes a
list of image paths and returns an (n, d) array. DinoV3Embedder is the one we
use. The weights are downloaded from Hugging Face on first use and are
released by Meta under their own license; read it on the model page.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .scan import open_image
from .util import log, progress

DEFAULT_MODEL = "vit_large_patch16_dinov3.lvd1689m"
# Smaller, faster alternatives (the default thresholds were tuned on the large one):
#   vit_base_patch16_dinov3.lvd1689m, vit_small_patch16_dinov3.lvd1689m


def normalize(raw):
    """Scale every embedding to length 1, so a dot product is a cosine
    similarity. Returns (unit_vectors, original_lengths)."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"expected an (n, d) array of embeddings, got shape {raw.shape}")
    if not np.isfinite(raw).all():
        bad = int((~np.isfinite(raw).all(axis=1)).sum())
        raise ValueError(f"{bad} embedding(s) contain NaN or inf; similarities would be meaningless")
    norms = np.linalg.norm(raw, axis=1)
    return raw / np.clip(norms, 1e-8, None)[:, None], norms


class DinoV3Embedder:
    """Embed images with a DINOv3 vision transformer.

    model_name  any timm DINOv3 model (DEFAULT_MODEL is ViT-L/16)
    weights     optional local .safetensors file, for machines without internet
    device      "cuda" or "cpu" (default: cuda when available)
    batch_size  default 64 on a GPU, 16 on a CPU

    On a GPU it runs in mixed precision (fp16 autocast). Large vision
    transformers carry a few very large activations that can overflow fp16;
    if a batch comes back with NaN or inf, it is redone in fp32 and fp32 is
    used for the rest of the run.
    """

    def __init__(self, model_name=DEFAULT_MODEL, weights=None, device=None,
                 batch_size=None, workers=4, show_progress=True):
        try:
            import torch
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as e:
            raise ImportError("DINOv3 embeddings need torch and timm>=1.0.20: "
                              "pip install 'qaqc-gist[embed]'") from e
        self.torch = torch
        self.name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            log("no GPU found: embedding will be slow" + (
                "" if "small" in model_name else
                ". A smaller model (vit_small_patch16_dinov3.lvd1689m) is much faster on a CPU"))
        kw = {"pretrained": True, "num_classes": 0}
        if weights:
            kw["pretrained_cfg_overlay"] = dict(file=str(weights))
        log(f"loading {model_name}")
        self.model = timm.create_model(model_name, **kw).eval().to(self.device)
        cfg = resolve_model_data_config(self.model)
        self.transform = create_transform(**cfg, is_training=False)
        self.input_size = tuple(cfg.get("input_size", (3, 224, 224)))
        self.batch_size = batch_size or (64 if self.device.type == "cuda" else 16)
        self.workers = workers
        self.show_progress = show_progress
        self.use_amp = self.device.type == "cuda"
        self.failed = []   # paths that could not be read at embedding time

    def _load(self, path):
        try:
            return self.transform(open_image(path)), True
        except Exception:
            return self.torch.zeros(*self.input_size), False

    def _forward(self, batch):
        torch = self.torch
        if self.use_amp:
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(batch).float()
                if bool(torch.isfinite(out).all()):
                    return out
                log("fp16 gave non-finite embeddings; using fp32 from now on")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise
                log(f"fp16 failed ({str(e)[:120]}); using fp32 from now on")
            self.use_amp = False
        return self.model(batch).float()

    def __call__(self, paths):
        torch, bs, out = self.torch, self.batch_size, []
        self.failed = []
        starts = range(0, len(paths), bs)
        with ThreadPoolExecutor(max_workers=self.workers) as pool, torch.inference_mode():
            for s in progress(starts, len(starts), "embed", self.show_progress):
                chunk = paths[s:s + bs]
                loaded = list(pool.map(self._load, chunk))
                self.failed += [p for p, (_, ok) in zip(chunk, loaded) if not ok]
                batch = torch.stack([t for t, _ in loaded]).to(self.device, non_blocking=True)
                try:
                    feats = self._forward(batch)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        raise RuntimeError(f"GPU out of memory at batch size {bs}; "
                                           "pass a smaller batch_size") from e
                    raise
                out.append(feats.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)
