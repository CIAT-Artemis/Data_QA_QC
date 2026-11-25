import csv
import json
import os

import numpy as np
import pytest

from qaqc_gist import Dataset, analyse, write_outputs
from qaqc_gist.cli import main
from qaqc_gist.health import HIDDEN, KEEP, REJECTED
from qaqc_gist.scan import collect_images, inspect_image, laplacian_variance

from conftest import smooth_image


def by_name(ds, name):
    return next(r for r in ds.images if r["filename"] == name)


def test_scan_rules(image_folder):
    names = [os.path.basename(p) for p in collect_images(image_folder)]
    assert "skip_me.jpg" not in names and len(names) == 17
    assert inspect_image(str(image_folder / "corrupt.jpg"))["readable"] is False


def test_laplacian_matches_scipy():
    ndimage = pytest.importorskip("scipy.ndimage")
    g = np.random.default_rng(0).random((50, 40)) * 255
    assert abs(laplacian_variance(g) - float(ndimage.laplace(g).var())) < 1e-6


def test_end_to_end(image_folder, embedder, tmp_path):
    out = tmp_path / "out"
    ds = analyse(image_folder, work_dir=out, embedder=embedder, show_progress=False)
    assert by_name(ds, "corrupt.jpg")["bucket"] == REJECTED
    assert by_name(ds, "blank.jpg")["bucket"] == REJECTED
    assert by_name(ds, "tiny.jpg")["bucket"] == HIDDEN
    copy = by_name(ds, "img_06_copy.jpg")
    assert copy["bucket"] == REJECTED and copy["reason"].startswith("exact copy of")
    a, burst = by_name(ds, "img_00.jpg"), by_name(ds, "img_00_burst.jpg")
    pair = {a["bucket"], burst["bucket"]}
    assert pair == {KEEP, HIDDEN}, (a["reason"], burst["reason"])    # one copy is kept

    picked = ds.select(5)
    assert len(picked["selected"]) == 5
    assert set(picked["selected"]) <= set(ds.bucket(KEEP))
    summary = write_outputs(ds, picked, out, plots=True)
    for f in ("results.csv", "selected.csv", "decisions.csv", "summary.json", "dataset.npz", "selected_images.zip"):
        assert (out / f).exists(), f
    with open(out / "selected.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["gist_rank"]) for r in rows] == [1, 2, 3, 4, 5]
    assert summary["gist"]["selected"] == 5
    copied = [p for p in collect_images(out / "selected_images")]
    assert len(copied) == 5
    assert (out / "plots" / "selected_grid.png").exists()

    # the slow part is cached: same folder again does not embed again
    again = analyse(image_folder, work_dir=out, embedder=embedder, show_progress=False)
    assert embedder.calls == 1 and len(again.rows) == len(ds.rows)
    assert np.allclose(again.E, ds.E)


def test_adding_images_only_processes_the_new_ones(image_folder, embedder, tmp_path):
    out = tmp_path / "out"
    first = analyse(image_folder, work_dir=out, embedder=embedder, show_progress=False)
    n_first = len(embedder.embedded)
    write_outputs(first, first.select(4), out, plots=False)
    (out / "decisions.csv").write_text("rel_path,decision\nplot_b/img_07.jpg,reject\nmissing.jpg,keep\n")
    smooth_image(50).save(image_folder / "plot_b" / "late_arrival.jpg", quality=95)
    second = analyse(image_folder, work_dir=out, embedder=embedder, show_progress=False)
    assert [os.path.basename(p) for p in embedder.embedded[n_first:]] == ["late_arrival.jpg"]
    assert len(second.rows) == len(first.rows) + 1
    assert by_name(second, "late_arrival.jpg")["id"] in second.row_of
    # the earlier review is re-applied; a decision for a vanished file is skipped
    assert by_name(second, "img_07.jpg")["bucket"] == REJECTED and by_name(second, "img_07.jpg")["manual"]
    # unchanged images keep exactly the same embeddings
    a = first.E[first.row_of[by_name(first, "img_03.jpg")["id"]]]
    b = second.E[second.row_of[by_name(second, "img_03.jpg")["id"]]]
    assert np.array_equal(a, b)


def test_saved_run_select_search_and_decisions(image_folder, embedder, tmp_path, capsys):
    out = tmp_path / "out"
    ds = analyse(image_folder, work_dir=out, embedder=embedder, show_progress=False)
    write_outputs(ds, ds.select(4), out, plots=False)
    loaded = Dataset.load(out / "dataset.npz")
    assert loaded.rows == ds.rows and np.allclose(loaded.E, ds.E)

    hits = loaded.search(["img_00.jpg"], threshold=0.5)
    burst_id = by_name(loaded, "img_00_burst.jpg")["id"]
    assert hits["results"][0]["id"] == burst_id

    # a reviewer rejects one image and restores the blank one, in a CSV
    dec = tmp_path / "review.csv"
    dec.write_text("rel_path,decision\nplot_a/img_01.jpg,reject\nblank.jpg,keep\n")
    assert main(["select", str(out), "--k", "3", "--decisions", str(dec), "--no-plots"]) == 0
    with open(out / "results.csv", newline="") as f:
        rows = {r["rel_path"]: r for r in csv.DictReader(f)}
    assert rows["plot_a/img_01.jpg"]["qaqc_bucket"] == REJECTED
    assert rows["plot_a/img_01.jpg"]["reviewer_decision"] == "True"
    assert rows["blank.jpg"]["qaqc_bucket"] == KEEP
    with open(out / "decisions.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2
    assert json.loads((out / "summary.json").read_text())["reviewer_decisions"] == 2

    assert main(["search", str(out), "--example", "img_00.jpg", "--threshold", "0.5"]) == 0
    assert "img_00_burst.jpg" in capsys.readouterr().out


def test_no_health_makes_every_readable_image_a_candidate(image_folder, embedder, tmp_path):
    ds = analyse(image_folder, work_dir=tmp_path / "o", embedder=embedder, health=False, show_progress=False)
    readable = [r["id"] for r in ds.images if r["readable"]]
    assert sorted(ds.bucket(KEEP)) == sorted(readable)
