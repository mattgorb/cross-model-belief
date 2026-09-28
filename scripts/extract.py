#!/usr/bin/env python3
"""Warm the activation cache — the expensive pass, run once, on purpose.

Every experiment calls `get_activations` and will extract on demand, but on
real hardware you want the forward passes done deliberately and up front:
they are the only expensive part, the 32B pass is the one you cannot afford to
repeat, and a spot instance can disappear between experiments. Run this first,
confirm the files are on persistent storage, then iterate on the analysis for
free.

    export CMB_CACHE=/mnt/data/activations
    scripts/extract.py --models qwen3-8b,gemma4-12b --datasets truthfulqa --n 2000
    scripts/extract.py --models qwen3-32b --datasets truthfulqa --n 2000
    scripts/extract.py --list                 # what is cached, and how big

All swept layers come out of one forward pass, so extracting the whole sweep
costs the same as extracting one layer. Extract once, sweep later.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmb.cache import cache_path, load
from cmb.config import CACHE_DIR, DATASETS, LAYER_SWEEP, MODELS
from cmb.extract import get_activations
from cmb.models import load_backend


def _release(backend) -> None:
    """Drop a model's weights before loading the next one.

    Without this the next `from_pretrained` finds the card already full, silently
    offloads layers to CPU ("Some parameters are on the meta device"), and then
    dies part-way through the pass.
    """
    if backend is None:
        return
    lm = getattr(backend, "lm", None)
    if lm is not None:
        del lm
        del backend.lm
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"


def list_cache() -> int:
    if not CACHE_DIR.exists():
        print(f"no cache at {CACHE_DIR}")
        return 1
    files = sorted(CACHE_DIR.rglob("*.npz"))
    if not files:
        print(f"cache at {CACHE_DIR} is empty")
        return 1
    print(f"cache: {CACHE_DIR}\n")
    print(f"  {'model':<22}{'dataset':<20}{'items':>7}{'layers':>8}{'size':>10}")
    total = 0
    for f in files:
        total += f.stat().st_size
        try:
            a = load(f)
            print(f"  {a.model:<22}{a.dataset:<20}{len(a.item_ids):>7}"
                  f"{len(a.layers):>8}{human(f.stat().st_size):>10}")
        except Exception as e:                      # a truncated/partial file
            print(f"  {f.relative_to(CACHE_DIR)}  UNREADABLE ({e})")
    print(f"\n  total {human(total)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="qwen3-8b,gemma4-12b",
                    help=f"comma-separated keys from {sorted(MODELS)}, or any "
                         "HuggingFace repo id")
    ap.add_argument("--datasets", default="truthfulqa",
                    help=f"comma-separated, from {list(DATASETS)}")
    ap.add_argument("--n", default="all",
                    help="items per dataset, or 'all' (the default) for every "
                         "item the dataset has. The train/test split happens "
                         "later, locally, so extract everything once.")
    ap.add_argument("--layers", default="",
                    help="extra layer specs beyond the standing sweep "
                         f"{LAYER_SWEEP} (comma-separated)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--refresh", action="store_true",
                    help="re-extract even if a cache file already exists")
    ap.add_argument("--list", action="store_true", help="show the cache and exit")
    args = ap.parse_args()

    if args.list:
        return list_cache()

    args.n = None if str(args.n).lower() in ("all", "none", "-1") else int(args.n)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    extra = tuple(s.strip() for s in args.layers.split(",") if s.strip())

    print(f"cache -> {CACHE_DIR}")
    if not args.synthetic and str(CACHE_DIR).startswith("/tmp"):
        print("  WARNING: the cache is under /tmp. A spot interruption will "
              "take it with the instance — point CMB_CACHE at the EBS volume.")

    failures = []
    for model in models:
        # One backend per model, reused across its datasets. Loading it per
        # dataset leaked the previous copy's VRAM: with a 27B across four
        # datasets the second load has no room left, falls back to CPU offload,
        # and then OOMs mid-pass. It also pays the load cost once instead of
        # once per dataset.
        backend = None
        for ds in datasets:
            key = f"synthetic-{model}" if args.synthetic else model
            path = cache_path(key, ds, args.n)
            if path.exists() and not args.refresh:
                print(f"  [skip] {model}/{ds} already cached "
                      f"({human(path.stat().st_size)})")
                continue
            print(f"\n=== {model} / {ds} (n={args.n}) ===")
            t0 = time.time()
            try:
                if backend is None:
                    backend = load_backend(model, synthetic=args.synthetic)
                acts = get_activations(model, ds, args.n, synthetic=args.synthetic,
                                       refresh=args.refresh, extra_layers=extra,
                                       backend=backend)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                failures.append((model, ds, f"{type(e).__name__}: {e}"))
                continue
            dt = time.time() - t0
            d = acts.pos[acts.layers[0]].shape[1]
            print(f"  {len(acts.item_ids)} items, {len(acts.layers)} layers, "
                  f"d={d}, {dt:.0f}s ({dt / max(len(acts.item_ids), 1):.2f}s/item)")
            print(f"  -> {path}  ({human(path.stat().st_size)})")

        _release(backend)
        backend = None

    if failures:
        print(f"\n{len(failures)} extraction(s) failed:")
        for m, d, e in failures:
            print(f"  {m}/{d}: {e}")
        return 1
    print("\ncache warm. Experiments now run without touching a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
