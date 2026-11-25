"""Command line: qaqc-gist run | select | search | demo.

    qaqc-gist run   IMAGES --out OUT --k 200       analyse a folder and pick k images
    qaqc-gist select OUT --k 100 --alpha 0.5       pick again from a saved run (seconds)
    qaqc-gist search OUT --example IMG_0042.jpg    find look-alikes in a saved run
    qaqc-gist demo  --out OUT                      try it on a small built-in set
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from .embed import DEFAULT_MODEL
from .pipeline import Dataset, analyse, print_summary, read_decisions, write_outputs


def _split(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _gist_args(ap):
    ap.add_argument("--k", type=int, default=100, help="how many images to pick (default 100)")
    ap.add_argument("--alpha", type=float, default=0.75, help="1 = quality only, 0 = diversity only (default 0.75)")
    ap.add_argument("--dup-threshold", type=float, default=0.98,
                    help="similarity GIST treats as the same picture (default 0.98)")
    ap.add_argument("--liked", default="", help="comma-separated images to steer towards")
    ap.add_argument("--disliked", default="", help="comma-separated images to steer away from")
    ap.add_argument("--decisions", help="CSV of reviewer decisions (rel_path, decision); "
                    "by default the run folder's decisions.csv is re-applied")
    ap.add_argument("--no-copy", action="store_true", help="do not copy the picked images")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-plots", action="store_true")


def _pick_and_write(ds, args, out):
    if args.decisions:
        ds.assess(decisions=read_decisions(args.decisions))
    picked = ds.select(args.k, alpha=args.alpha, duplicate_threshold=args.dup_threshold,
                       liked=_split(args.liked), disliked=_split(args.disliked))
    write_outputs(ds, picked, out, copy=not args.no_copy, make_zip=not args.no_zip, plots=not args.no_plots)
    print_summary(ds, picked, out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qaqc-gist", description="QA/QC and diverse selection for image datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse a folder or .zip of images and pick k of them")
    r.add_argument("images")
    r.add_argument("--out", default="qaqc_gist_output")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--weights", help="local .safetensors file for the model (offline use)")
    r.add_argument("--batch-size", type=int)
    r.add_argument("--max-images", type=int, default=0, help="work on a fixed random sample")
    r.add_argument("--params", help="JSON of health-check settings, e.g. '{\"outlier_mad_multiplier\": 3}'")
    r.add_argument("--no-health", action="store_true", help="skip QA/QC: every readable image is a candidate")
    _gist_args(r)

    s = sub.add_parser("select", help="pick again from a saved run, with other settings")
    s.add_argument("run_dir")
    s.add_argument("--out", help="where to write (default: the run folder)")
    s.add_argument("--params", help="JSON of health-check settings")
    _gist_args(s)

    q = sub.add_parser("search", help="find images similar to one or more examples in a saved run")
    q.add_argument("run_dir")
    q.add_argument("--example", action="append", required=True, help="image path or file name (repeatable)")
    q.add_argument("--threshold", type=float, default=0.85)
    q.add_argument("--max", type=int, default=50)
    q.add_argument("--bucket", action="append", help="only search these buckets (repeatable)")
    q.add_argument("--csv", help="also write the results to this CSV")

    d = sub.add_parser("demo", help="build a small test set with planted problems and run on it")
    d.add_argument("--out", default="qaqc_gist_demo")
    d.add_argument("--model", default="vit_small_patch16_dinov3.lvd1689m")
    d.add_argument("--k", type=int, default=12)

    a = ap.parse_args(argv)
    if a.cmd == "run":
        ds = analyse(a.images, work_dir=a.out, model=a.model, weights=a.weights, batch_size=a.batch_size,
                     params=json.loads(a.params) if a.params else None,
                     max_images=a.max_images, health=not a.no_health)
        _pick_and_write(ds, a, a.out)
    elif a.cmd == "select":
        ds = Dataset.load(os.path.join(a.run_dir, "dataset.npz"))
        if a.params:
            ds.assess(params=json.loads(a.params))
        _pick_and_write(ds, a, a.out or a.run_dir)
    elif a.cmd == "search":
        ds = Dataset.load(os.path.join(a.run_dir, "dataset.npz"))
        res = ds.search(a.example, threshold=a.threshold, max_results=a.max, buckets=a.bucket)
        by_id = {r["id"]: r for r in ds.images}
        print(f"typical similarity in this dataset: {res['typical']}  (compared {res['compared']} images)")
        for h in res["results"]:
            r = by_id[h["id"]]
            print(f"{h['similarity']:.3f}  match {h['match']:>5}%  {r['bucket']:<8}  {r['rel_path']}")
        if not res["results"]:
            c = res["closest"]
            print("nothing at or above the threshold" + (
                f"; the closest is {by_id[c['id']]['rel_path']} at {c['similarity']:.3f}" if c else ""))
        if a.csv:
            with open(a.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["rel_path", "similarity", "match", "bucket"])
                for h in res["results"]:
                    r = by_id[h["id"]]
                    w.writerow([r["rel_path"], h["similarity"], h["match"], r["bucket"]])
    elif a.cmd == "demo":
        from .demo import make_demo_images
        imgs = os.path.join(a.out, "demo_images")
        if not os.path.isdir(imgs):          # build once: rebuilt files would count as changed
            make_demo_images(imgs)
        ds = analyse(imgs, work_dir=a.out, model=a.model)
        a.alpha, a.dup_threshold, a.liked, a.disliked, a.decisions = 0.75, 0.98, "", "", None
        a.no_copy = a.no_zip = a.no_plots = False
        _pick_and_write(ds, a, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
