"""QA/QC GIST from Python: check a folder of images, review, search, select.

    python examples/quickstart.py path/to/images [k]

Without arguments it builds the small demo set (needs scikit-image) and uses
the small DINOv3 model, so it runs in a few minutes on a laptop CPU.
"""
import os
import sys

from qaqc_gist import KEEP, OUTLIER, analyse, write_outputs

OUT = "quickstart_out"

if len(sys.argv) > 1:
    images, model = sys.argv[1], "vit_large_patch16_dinov3.lvd1689m"
else:
    from qaqc_gist.demo import make_demo_images
    images, model = f"{OUT}/demo_images", "vit_small_patch16_dinov3.lvd1689m"
    if not os.path.isdir(images):          # build once: rebuilt files would count as changed
        make_demo_images(images)
k = int(sys.argv[2]) if len(sys.argv) > 2 else 12

# 1. automatic checks: open, measure, embed, sort into buckets (cached in OUT)
ds = analyse(images, work_dir=OUT, model=model)

# 2. review: what was flagged, and why
for r in ds.images:
    if r["bucket"] != KEEP:
        print(f"{r['bucket']:<9} {r['rel_path']:<40} {r['reason']}")

# 3. search: everything that looks like the first outlier
outliers = ds.bucket(OUTLIER)
if outliers:
    hits = ds.search([outliers[0]], threshold=0.8)
    print(f"\nlook-alikes of #{outliers[0]} (typical similarity here {hits['typical']}):")
    for h in hits["results"][:10]:
        print(f"  {h['similarity']:.3f}  match {h['match']}%  #{h['id']}")
    if not hits["results"] and hits["closest"]:
        c = hits["closest"]
        print(f"  none at 0.8 or above; the closest is #{c['id']} at {c['similarity']:.3f}")

# 4. a reviewer decision: restore the first outlier (decisions always win)
if outliers:
    ds.assess(decisions={outliers[0]: "keep"})

# 5. GIST: k good, varied images from the keep bucket
picked = ds.select(k, alpha=0.75)
print(f"\npicked {len(picked['selected'])} of {picked['candidates']} candidates")
for iid in picked["not_selected"][:5]:
    print(f"  left out #{iid}: {picked['reasons'][iid]}")

# 6. the audit trail: results.csv, selected.csv, decisions.csv, summary.json, plots
write_outputs(ds, picked, OUT)
print(f"\noutputs in {OUT}/")
