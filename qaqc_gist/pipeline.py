"""The QA/QC GIST pipeline: scan -> embed -> health check -> GIST -> outputs.

    ds = analyse("my_images/", work_dir="out")       # slow part, cached
    picked = ds.select(k=200)                         # seconds; rerun freely
    hits = ds.search(["IMG_0042.jpg"], threshold=0.9)
    write_outputs(ds, picked, "out")

The expensive work (opening every image, DINOv3 embeddings) is cached in
work_dir, so trying other settings, recording reviewer decisions or
searching costs seconds.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from collections import Counter

import numpy as np

from . import __version__
from .embed import DEFAULT_MODEL, DinoV3Embedder, normalize
from .gist import select as gist_select
from .health import BUCKETS, DEFAULT_PARAMS, KEEP, REJECTED, as_bucket, assess, counts, nearest_neighbours
from .scan import collect_images, inspect_many, make_record, resolve_input, sample_paths
from .search import search as sim_search
from .search import typical_similarity
from .util import log

PAPER = ("Fahrbach, Ramalingam, Zadimoghaddam, Ahmadian, Citovsky, DeSalvo. GIST: Greedy "
         "Independent Set Thresholding for Max-Min Diversification with Submodular Utility. "
         "NeurIPS 2025. arXiv:2405.18754")
_KEEP_FIELDS = ("readable", "width", "height", "aspect", "sharpness", "brightness", "contrast", "sha256")


class Dataset:
    """One analysed image collection: the image records (with buckets and
    reasons), unit-length embeddings, and which image each row belongs to."""

    def __init__(self, root, images, E, norms, rows, model, params=None, decisions=None):
        self.root, self.images, self.model = root, images, model
        self.E = np.asarray(E, dtype=np.float32)
        self.norms = np.asarray(norms, dtype=np.float32)
        self.rows = [int(i) for i in rows]
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.decisions = dict(decisions or {})
        self._knn = None
        self._typical = None

    # ── lookup ──
    @property
    def row_of(self):
        return {iid: r for r, iid in enumerate(self.rows)}

    def find(self, name):
        """Image id for a relative path, file name or id."""
        if isinstance(name, (int, np.integer)):
            return int(name)
        name = str(name).replace("\\", "/")
        hits = [r["id"] for r in self.images if r["rel_path"].replace("\\", "/") == name]
        if not hits:
            hits = [r["id"] for r in self.images if r["filename"] == os.path.basename(name)]
        if not hits:
            raise KeyError(f"no image called {name!r}")
        if len(hits) > 1:
            raise KeyError(f"{len(hits)} images are called {name!r}; use the path relative to the image folder")
        return hits[0]

    def bucket(self, *names):
        """Ids of the images in the given buckets."""
        return [r["id"] for r in self.images if r["bucket"] in names]

    # ── QA/QC ──
    def assess(self, params=None, decisions=None, health=True):
        """(Re)run the health check. Reviewer decisions (by path, file name
        or id) are added to the ones already recorded and always win."""
        if params:
            self.params.update(params)
        for name, d in (decisions or {}).items():
            try:
                self.decisions[self.find(name)] = as_bucket(d)
            except KeyError as e:
                log(f"reviewer decision ignored: {e.args[0]}")
        if not health:
            for r in self.images:
                r.update(bucket=KEEP if r["readable"] else REJECTED, issues=[], manual=False,
                         reason="" if r["readable"] else "could not be opened as an image",
                         duplicate_of=None, outlier_votes=0, outlier_checks=[])
            return counts(self.images)
        if self._knn is None and len(self.rows) >= 3:
            self._knn = nearest_neighbours(self.E, int(self.params["neighbours"]))
        return assess(self.images, self.E, self.norms, self.rows, self.params,
                      self.decisions, knn=self._knn)

    def select(self, k, alpha=0.75, duplicate_threshold=0.98, liked=(), disliked=(), candidates=None, **kw):
        """GIST over the keep bucket (or the given candidate images)."""
        pool = self.bucket(KEEP) if candidates is None else [self.find(c) for c in candidates]
        row_of = self.row_of
        pool = [i for i in pool if i in row_of]
        by_id = {r["id"]: r for r in self.images}
        return gist_select(self.E[[row_of[i] for i in pool]] if pool else np.zeros((0, 1)), k,
                           quality=[by_id[i]["sharpness"] for i in pool], alpha=alpha,
                           duplicate_threshold=duplicate_threshold, ids=pool,
                           liked=[self.find(x) for x in liked], disliked=[self.find(x) for x in disliked], **kw)

    def search(self, examples, threshold=0.85, max_results=500, buckets=None):
        """Images similar to the examples (paths, file names or ids),
        optionally only within some buckets."""
        if self._typical is None:
            self._typical = typical_similarity(self.E)
        return sim_search(self.E, [self.find(x) for x in examples], threshold, max_results,
                          ids=self.rows, candidates=self.bucket(*buckets) if buckets else None,
                          typical=self._typical)

    # ── persistence ──
    def save(self, path):
        meta = {"root": self.root, "model": self.model, "images": self.images, "params": self.params,
                "decisions": {str(k): v for k, v in self.decisions.items()}, "version": __version__}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, E=self.E, norms=self.norms, rows=np.asarray(self.rows, dtype=np.int64),
                            meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(meta["root"], meta["images"], z["E"], z["norms"], z["rows"].tolist(), meta["model"],
                   meta.get("params"), {int(k): v for k, v in meta.get("decisions", {}).items()})


# ── cache: per image, so adding images to a folder only processes the new ones ──

def _sig(root, path):
    """Identifies one version of one file: path, size, modification time."""
    try:
        st = os.stat(path)
        return f"{os.path.relpath(path, root)}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return None


def _cache_path(work_dir, model_name):
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_name)
    return os.path.join(work_dir, "cache", f"features_{safe}.npz")


def _load_cache(path):
    """{signature: (scan fields, unit embedding or None, length or None)}"""
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
    except Exception:
        return {}
    emb = {s: i for i, s in enumerate(meta["emb_sigs"])}
    return {s: (f, z["E"][emb[s]] if s in emb else None, float(z["norms"][emb[s]]) if s in emb else None)
            for s, f in meta["entries"].items()}


def _save_cache(path, entries):
    sigs = [s for s, (_, e, _) in entries.items() if e is not None]
    E = np.array([entries[s][1] for s in sigs], dtype=np.float32) if sigs else np.zeros((0, 1), np.float32)
    meta = {"version": 2, "entries": {s: f for s, (f, _, _) in entries.items()}, "emb_sigs": sigs}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, E=E, norms=np.array([entries[s][2] for s in sigs], dtype=np.float32),
                        meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def read_decisions(path):
    """A reviewer's CSV: columns rel_path (or filename) and decision
    (keep / hidden / outlier / rejected)."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("rel_path") or row.get("filename") or row.get("path") or "").strip()
            if name and (row.get("decision") or "").strip():
                out[name] = as_bucket(row["decision"])
    return out


def analyse(image_dir, work_dir=None, embedder=None, model=DEFAULT_MODEL, weights=None,
            batch_size=None, params=None, decisions=None, max_images=0, health=True,
            cache=True, show_progress=True):
    """Scan, embed and health-check a folder (or .zip) of images.

    embedder   any callable(list of paths) -> (n, d) array; default DINOv3
    decisions  {path, file name or id: bucket} from a reviewer. When not
               given, work_dir/decisions.csv from an earlier run is used,
               so a review carries over to the next run.
    max_images >0: work on a fixed random sample of this many images
    cache      keep each image's checks and embedding in work_dir/cache;
               a rerun only opens and embeds new or changed files
    """
    work_dir = work_dir or tempfile.mkdtemp(prefix="qaqc_gist_")
    os.makedirs(work_dir, exist_ok=True)
    root = resolve_input(image_dir, work_dir)
    paths = sample_paths(collect_images(root), max_images)
    if not paths:
        raise FileNotFoundError(f"no .jpg, .jpeg, .png or .webp images under {root}")
    log(f"{len(paths)} images under {root}")
    name = getattr(embedder, "name", None) or (model if embedder is None else type(embedder).__name__)
    cache_file = _cache_path(work_dir, name)
    cached = _load_cache(cache_file) if cache else {}
    sigs = [_sig(root, p) for p in paths]

    # 1. open and measure only files not seen before (or changed since)
    new = [i for i, s in enumerate(sigs) if s is None or s not in cached]
    t = time.time()
    infos = dict(zip(new, inspect_many([paths[i] for i in new], show_progress=show_progress))) if new else {}
    images = [make_record(i, p, root, infos[i] if i in infos else cached[sigs[i]][0])
              for i, p in enumerate(paths)]
    if new:
        log(f"opened and measured {len(new)} new or changed images ({time.time() - t:.0f} s)")

    # 2. embed only readable images without a saved embedding
    vecs, lens = {}, {}
    for i, s in enumerate(sigs):
        if s in cached and cached[s][1] is not None and images[i]["readable"]:
            vecs[i], lens[i] = cached[s][1], cached[s][2]
    todo = [r for r in images if r["readable"] and r["id"] not in vecs]
    if len(paths) > len(new) or len(vecs):
        log(f"reused saved checks for {len(paths) - len(new)} images and embeddings for {len(vecs)}")
    failed = set()
    if todo:
        if embedder is None:
            embedder = DinoV3Embedder(model, weights=weights, batch_size=batch_size,
                                      show_progress=show_progress)
        t = time.time()
        raw = np.asarray(embedder([r["path"] for r in todo]))
        failed = set(getattr(embedder, "failed", []) or [])
        ok = [j for j, r in enumerate(todo) if r["path"] not in failed]
        if ok:
            unit, norms = normalize(raw[ok])
            for j, u, n in zip(ok, unit, norms):
                vecs[todo[j]["id"]], lens[todo[j]["id"]] = u, float(n)
        for r in todo:
            if r["path"] in failed:
                r["readable"] = False     # could not be read again at embedding time
        log(f"embedded {len(ok)} images ({time.time() - t:.0f} s)")

    rows = sorted(vecs)
    E = np.array([vecs[i] for i in rows], dtype=np.float32) if rows else np.zeros((0, 1), np.float32)
    ds = Dataset(root, images, E, np.array([lens[i] for i in rows], dtype=np.float32), rows, name)
    if cache:   # failed reads are not saved, so the next run tries them again
        for i, s in enumerate(sigs):
            if s is not None and paths[i] not in failed:
                keep = {k: images[i][k] for k in _KEEP_FIELDS}
                cached[s] = (keep, vecs.get(i), lens.get(i))
        _save_cache(cache_file, cached)
    if decisions is None and os.path.exists(os.path.join(work_dir, "decisions.csv")):
        decisions = read_decisions(os.path.join(work_dir, "decisions.csv"))
        if decisions:
            log(f"re-applying {len(decisions)} reviewer decisions from decisions.csv")
    ds.assess(params, decisions, health=health)
    log("health check: " + ", ".join(f"{b} {n}" for b, n in counts(ds.images).items()))
    return ds


# ── outputs ──

def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def results_table(ds, picked=None):
    """One row per image: measurements, QA/QC bucket and reason, GIST decision."""
    rank = {iid: i + 1 for i, iid in enumerate(picked["selected"])} if picked else {}
    cand = (set(picked["selected"]) | set(picked["not_selected"])) if picked else set()
    rows = []
    for r in ds.images:
        iid = r["id"]
        rows.append({
            "id": iid, "rel_path": r["rel_path"], "filename": r["filename"], "folder": r["folder"],
            "readable": r["readable"], "width": r["width"], "height": r["height"],
            "sharpness": r["sharpness"], "brightness": r["brightness"], "contrast": r["contrast"],
            "issues": ",".join(r.get("issues", [])), "qaqc_bucket": r["bucket"],
            "qaqc_reason": r.get("reason", ""), "reviewer_decision": bool(r.get("manual")),
            "duplicate_of": "" if r.get("duplicate_of") is None else r["duplicate_of"],
            "outlier_votes": r.get("outlier_votes", 0),
            "gist_candidate": iid in cand, "gist_selected": iid in rank, "gist_rank": rank.get(iid, ""),
            "gist_reason": (picked or {}).get("reasons", {}).get(iid, ""),
            "path": r["path"],
        })
    return rows


def write_outputs(ds, picked, out_dir, copy=True, make_zip=True, plots=True):
    """Write the audit trail and the selection:

        results.csv        every image: measurements, bucket, reason, GIST decision
        selected.csv       the picked images, in pick order
        decisions.csv      reviewer decisions so far (edit and pass back in)
        summary.json       settings, counts, versions: enough to reproduce the run
        dataset.npz        embeddings + records, for `qaqc-gist select/search`
        selected_images/   copies of the picked images (+ .zip)
        plots/             2-D map, thumbnail sheets, near-duplicate pairs
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = results_table(ds, picked)
    fields = list(rows[0].keys()) if rows else ["id"]
    _write_csv(os.path.join(out_dir, "results.csv"), rows, fields)
    sel = sorted((r for r in rows if r["gist_selected"]), key=lambda r: r["gist_rank"])
    _write_csv(os.path.join(out_dir, "selected.csv"), sel, fields)
    by_id = {r["id"]: r for r in ds.images}
    _write_csv(os.path.join(out_dir, "decisions.csv"),
               [{"rel_path": by_id[i]["rel_path"], "decision": d} for i, d in sorted(ds.decisions.items())],
               ["rel_path", "decision"])
    ds.save(os.path.join(out_dir, "dataset.npz"))

    zip_path = None
    if copy and picked and picked["selected"]:
        dest = os.path.join(out_dir, "selected_images")
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest)
        for iid in picked["selected"]:
            r = by_id[iid]
            target = os.path.join(dest, r["rel_path"])     # keep sub-folders: no name clashes
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(r["path"], target)
        if make_zip:
            zip_path = shutil.make_archive(dest, "zip", dest)

    plot_paths = []
    if plots and picked is not None:
        from .report import make_plots
        plot_paths = make_plots(ds, picked, os.path.join(out_dir, "plots"))

    reasons = Counter()
    for why in (picked or {}).get("reasons", {}).values():
        reasons[why.split(" #")[0].split(" (")[0]] += 1
    summary = {
        "tool": f"qaqc-gist {__version__}", "image_dir": ds.root, "model": ds.model,
        "images": len(ds.images), "readable": sum(1 for r in ds.images if r["readable"]),
        "embedded": len(ds.rows), "qaqc_buckets": counts(ds.images), "qaqc_params": ds.params,
        "reviewer_decisions": len(ds.decisions),
        "gist": None if picked is None else {
            k: picked[k] for k in ("k", "alpha", "duplicate_threshold", "candidates",
                                   "near_identical_suppressed", "excluded_by_dislikes",
                                   "liked_used", "disliked_used")}
               | {"selected": len(picked["selected"]), "not_selected_reasons": dict(reasons)},
        "outputs": {"zip": zip_path, "plots": plot_paths},
        "reference": PAPER,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def print_summary(ds, picked, out_dir):
    c = counts(ds.images)
    print("\n" + "=" * 70)
    print(f" Images            {len(ds.images)}  (readable {sum(r['readable'] for r in ds.images)})")
    print(" QA/QC buckets     " + ", ".join(f"{b} {c.get(b, 0)}" for b in BUCKETS))
    if picked is not None:
        print(f" GIST              {len(picked['selected'])} selected of {picked['candidates']} "
              f"candidates (k={picked['k']}, alpha={picked['alpha']})")
        if picked["candidates"] and picked["candidates"] <= picked["k"]:
            print("   note: every candidate was kept, so GIST had nothing to choose between.\n"
                  "   Lower k, or let more images through the health check.")
    print(f" Outputs           {out_dir}")
    print("=" * 70)


def run(image_dir, out_dir, k=100, alpha=0.75, duplicate_threshold=0.98, liked=(), disliked=(),
        copy=True, make_zip=True, plots=True, **analyse_kw):
    """analyse + select + write_outputs, in one call."""
    ds = analyse(image_dir, work_dir=out_dir, **analyse_kw)
    picked = ds.select(k, alpha=alpha, duplicate_threshold=duplicate_threshold, liked=liked, disliked=disliked)
    write_outputs(ds, picked, out_dir, copy=copy, make_zip=make_zip, plots=plots)
    print_summary(ds, picked, out_dir)
    return ds, picked
