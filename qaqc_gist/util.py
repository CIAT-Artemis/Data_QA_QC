"""Small shared helpers: logging and progress bars."""
from __future__ import annotations

import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def progress(iterable, total, desc, enabled=True):
    """Wrap an iterable in a tqdm bar when tqdm is installed and enabled."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)
