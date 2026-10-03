#!/usr/bin/env python3
"""Check NDIF: which models are up, then time a real trace on one of them.

Llama-3.1-8B-Instruct is the right model to test on, because it is also in the
local cache -- so the same probe fitted on remotely-extracted activations can be
compared against the local extraction, and the remote path validated before any
large model is trusted.

    export NDIF_API_KEY=...
    scripts/ndif_check.py                       # list + time 8B
    scripts/ndif_check.py --model meta-llama/Llama-3.1-70B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def running():
    from nnsight.ndif import NdifStatus
    out = []
    for v in NdifStatus.request_status().get("deployments", {}).values():
        try:
            hf = json.loads(v["model_key"].split(":", 1)[-1])
            if v.get("application_state") == "RUNNING":
                out.append(hf["repo_id"])
        except Exception:
            pass
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=8, help="claims to time")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    up = running()
    print(f"{len(up)} models running on NDIF:")
    for r in up:
        print(f"   {'* ' if r == args.model else '  '}{r}")
    if args.model not in up:
        print(f"\n{args.model} is not running; pick one above.")
        return 1

    from cmb.data import load_items
    from cmb.ndif import NDIFModel

    claims = [it.claim for it in load_items("truthfulqa", None)[:args.n]]
    print(f"\ntracing {len(claims)} claims through {args.model} "
          f"(batch {args.batch_size}) ...")
    t0 = time.time()
    m = NDIFModel(args.model, batch_size=args.batch_size)
    t1 = time.time()
    feats = m.features_batch(claims)
    dt = time.time() - t1
    layer = max(feats[0].pos)
    print(f"  load {t1 - t0:.1f}s   trace {dt:.1f}s "
          f"({dt / len(claims):.2f}s/item, 3 passes each)")
    print(f"  layers {m.n_layers}  hidden {feats[0].pos[layer].shape[0]}")
    print(f"  p_yes range [{min(f.p_yes for f in feats):.3f}, "
          f"{max(f.p_yes for f in feats):.3f}]")
    full = 1634 * dt / len(claims)
    print(f"\n  => TruthfulQA (1634 items) would take ~{full / 60:.0f} min")
    print(f"  => all five datasets (~15k items) ~{15000 * dt / len(claims) / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
