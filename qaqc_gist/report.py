"""Pictures for reviewing a run: a 2-D map and labelled thumbnail sheets.

    map_2d.png                 every embedded image; picked ones highlighted
    selected_grid.png          the picked images, in pick order
    not_selected_grid.png      candidates left out, with the reason
    near_identical_pairs.png   kept image (left) next to the one set aside
    outliers_grid.png          the outlier bucket, most-flagged first
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

from .health import KEEP, OUTLIER
from .scan import open_image
from .util import log


def _thumb(path, size):
    try:
        img = open_image(path)
        img.thumbnail((size, size))
        return img
    except Exception:
        return Image.new("RGB", (size, size), (220, 220, 220))


def _wrap(text, width):
    lines, cur = [], ""
    for w in str(text).split():
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else lines or [""]


def contact_sheet(items, out_path, title, cols=6, size=200, max_items=36):
    """items: (path, label) pairs. Writes a labelled thumbnail grid."""
    items = items[:max_items]
    if not items:
        return None
    label_h, pad, head = 34, 8, 36
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (size + pad) + pad, head + rows * (size + label_h + pad) + pad), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), title, fill="black")
    for n, (path, label) in enumerate(items):
        x = pad + (n % cols) * (size + pad)
        y = head + (n // cols) * (size + label_h + pad)
        t = _thumb(path, size)
        sheet.paste(t, (x + (size - t.width) // 2, y + (size - t.height) // 2))
        for li, line in enumerate(_wrap(label, 32)[:2]):
            d.text((x, y + size + 3 + li * 14), line, fill="black")
    sheet.save(out_path)
    return out_path


def pairs_sheet(pairs, out_path, size=200):
    """pairs: (kept_path, set_aside_path, label)."""
    if not pairs:
        return None
    pad, label_h, head = 8, 20, 36
    sheet = Image.new("RGB", (2 * (size + pad) + pad, head + len(pairs) * (size + label_h + pad) + pad), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), "Near-identical pairs: kept (left), set aside (right)", fill="black")
    for n, (a, b, label) in enumerate(pairs):
        y = head + n * (size + label_h + pad)
        for col, p in enumerate((a, b)):
            t = _thumb(p, size)
            x = pad + col * (size + pad)
            sheet.paste(t, (x + (size - t.width) // 2, y + (size - t.height) // 2))
        d.text((pad, y + size + 3), label, fill="black")
    sheet.save(out_path)
    return out_path


def map_2d(ds, picked, out_path):
    """2-D map of all embedded images: UMAP when installed, else PCA."""
    try:
        from matplotlib.figure import Figure
    except ImportError:
        log("matplotlib not installed: 2-D map skipped (pip install 'qaqc-gist[plots]')")
        return None
    E = ds.E
    if E.shape[0] < 3:
        return None
    method = "PCA"
    try:
        import umap
        xy = umap.UMAP(n_neighbors=min(15, E.shape[0] - 1), min_dist=0.1, metric="cosine",
                       random_state=42).fit_transform(E)
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=42).fit_transform(E)
    by_id = {r["id"]: r for r in ds.images}
    sel, cand = set(picked["selected"]), set(picked["selected"]) | set(picked["not_selected"])
    groups = {"selected by GIST": [], "candidate, not selected": [], "set aside by QA/QC": []}
    for row, iid in enumerate(ds.rows):
        g = ("selected by GIST" if iid in sel else "candidate, not selected" if iid in cand
             else "set aside by QA/QC")
        groups[g].append(row)
    style = {"selected by GIST": dict(s=28, c="#d95f02", zorder=3),
             "candidate, not selected": dict(s=12, c="#9e9e9e", zorder=2),
             "set aside by QA/QC": dict(s=14, c="#1b9e77", marker="x", zorder=1)}
    fig = Figure(figsize=(9, 7))
    ax = fig.subplots()
    for name, rows in groups.items():
        if rows:
            ax.scatter(xy[rows, 0], xy[rows, 1], label=f"{name} ({len(rows)})", **style[name])
    ax.set_title(f"{method} of image embeddings: k={picked['k']}, alpha={picked['alpha']}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    return out_path


def make_plots(ds, picked, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    by_id = {r["id"]: r for r in ds.images}
    made = []
    try:
        made.append(map_2d(ds, picked, os.path.join(plot_dir, "map_2d.png")))
    except Exception as e:
        log(f"2-D map skipped: {e}")
    sel = picked["selected"]
    made.append(contact_sheet(
        [(by_id[i]["path"], f"#{n + 1} {by_id[i]['filename']}") for n, i in enumerate(sel)],
        os.path.join(plot_dir, "selected_grid.png"),
        f"Selected by GIST, in pick order (first {min(36, len(sel))} of {len(sel)})"))
    ns = picked["not_selected"]
    made.append(contact_sheet(
        [(by_id[i]["path"], f"{by_id[i]['filename']}: {picked['reasons'].get(i, '')}") for i in ns],
        os.path.join(plot_dir, "not_selected_grid.png"),
        f"Candidates not selected, with reasons (first {min(36, len(ns))} of {len(ns)})"))
    outl = sorted((r for r in ds.images if r["bucket"] == OUTLIER), key=lambda r: -r.get("outlier_votes", 0))
    made.append(contact_sheet(
        [(r["path"], f"{r['filename']}: {r['outlier_votes']} of 3 checks") for r in outl],
        os.path.join(plot_dir, "outliers_grid.png"),
        f"Outliers to review, most-flagged first (first {min(36, len(outl))} of {len(outl)})"))
    row_of = ds.row_of
    pairs = []
    for iid, d in list(picked["duplicate_of"].items())[:6]:
        pairs.append((by_id[d["anchor"]]["path"], by_id[iid]["path"],
                      f"GIST: #{iid} near-identical to #{d['anchor']} ({d['similarity']})"))
    for r in ds.images:
        if len(pairs) >= 12:
            break
        a = r.get("duplicate_of")
        if a is None or r["bucket"] == KEEP or r.get("manual"):
            continue
        sim = ""
        if r["id"] in row_of and a in row_of:
            sim = f" ({float(ds.E[row_of[r['id']]] @ ds.E[row_of[a]]):.3f})"
        pairs.append((by_id[a]["path"], r["path"], f"QA/QC: #{r['id']} set aside, copy of #{a}{sim}"))
    made.append(pairs_sheet(pairs, os.path.join(plot_dir, "near_identical_pairs.png")))
    return [m for m in made if m]
