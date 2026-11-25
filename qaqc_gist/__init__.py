"""QA/QC GIST: quality control and diverse subset selection for image datasets.

    from qaqc_gist import analyse, write_outputs

    ds = analyse("field_images/", work_dir="out")     # checks + DINOv3 embeddings
    picked = ds.select(k=200)                          # GIST: 200 good, varied images
    write_outputs(ds, picked, "out")                   # audit trail + copies
"""
__version__ = "0.1.0"

from .gist import select  # noqa: E402
from .health import BUCKETS, DEFAULT_PARAMS, HIDDEN, KEEP, OUTLIER, REJECTED, assess  # noqa: E402
from .pipeline import Dataset, analyse, results_table, run, write_outputs  # noqa: E402
from .search import search, typical_similarity  # noqa: E402

__all__ = ["analyse", "run", "write_outputs", "results_table", "Dataset", "select", "assess",
           "search", "typical_similarity", "DEFAULT_PARAMS", "BUCKETS", "KEEP", "HIDDEN",
           "OUTLIER", "REJECTED", "__version__"]
