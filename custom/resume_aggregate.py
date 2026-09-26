"""Offline FedAvg/FedProx re-aggregation for crash recovery.

The server deletes its transient run dir on completion, so a crashed run loses the
aggregated global. But every client persists its per-round deployed model as
`{site_id}_gmic_model_round_{x}.pth` (a flat state_dict, same format as sample_model_*.p).

With `epochs: 1` per round, the deployed model (best-val-of-round) IS the final-epoch
SENT weights the server aggregates -- they are captured at the same epoch boundary. So a
sample-weighted mean of all clients' round-x files reproduces EXACTLY the global the
server would have broadcast at the start of round x+1. This module computes that mean.

Aggregation is FedAvg's weighted mean (FedProx aggregates identically -- the proximal
term is purely client-side and stateless), so this reconstruction is method-agnostic.

Use it two ways:
  - Standalone CLI: run anywhere the round-x files live, then point the server persistor's
    `source_ckpt_file_full_name` at the output .pth (the proven warm-start path).
  - Imported by warmstart_persistor.WarmStartAggregatePersistor for in-job automation.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import OrderedDict

import torch


def _unwrap_state_dict(obj):
    """Accept a flat state_dict OR a {'model'/'model_dict'/'state_dict': sd} wrapper."""
    if isinstance(obj, dict):
        for wrap_key in ("model", "model_dict", "state_dict"):
            inner = obj.get(wrap_key)
            if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
                return inner
    return obj


def aggregate_round_checkpoints(
    directory: str,
    round_idx: int,
    weights: dict | None = None,
    pattern: str = "*_gmic_model_round_{round}.pth",
    logger=print,
):
    """Weighted-average every client's round-`round_idx` checkpoint in `directory`.

    Args:
        directory: folder holding `{site_id}_gmic_model_round_{x}.pth` files.
        round_idx: the round number x to aggregate.
        weights: optional {site_id: float} sample-count weights (the server weights each
            client by its train-sample count). If None, all clients are weighted equally
            -- only correct when the sites have equal training-set sizes.
        pattern: filename glob with a `{round}` placeholder.
        logger: callable(str) for progress (print, or an FL log_info shim).

    Returns:
        (agg: OrderedDict flat state_dict, info: dict with sites/raw_weights/norm_weights/n_keys).

    Raises:
        FileNotFoundError if no files match; ValueError on key mismatch or non-positive weight sum.
    """
    glob_pat = os.path.join(directory, pattern.format(round=round_idx))
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No checkpoints match {glob_pat}")

    weights = weights or {}
    states, sites, raw_w = [], [], []
    for p in paths:
        base = os.path.basename(p)
        # site id = everything before the '_gmic_model_round_' marker
        marker = "_gmic_model_round_"
        site = base.split(marker)[0] if marker in base else base
        sd = _unwrap_state_dict(torch.load(p, map_location="cpu", weights_only=False))
        if not isinstance(sd, dict) or not any(torch.is_tensor(v) for v in sd.values()):
            raise ValueError(f"{p} is not a state_dict (got {type(sd)})")
        states.append(sd)
        sites.append(site)
        raw_w.append(float(weights.get(site, 1.0)))
        logger(f"[aggregate] loaded {site}: {len(sd)} keys, weight={raw_w[-1]} <- {base}")

    # Every client must expose the same parameter keys, else the average is ill-defined.
    ref_keys = list(states[0].keys())
    ref_set = set(ref_keys)
    for site, sd in zip(sites[1:], states[1:]):
        if set(sd.keys()) != ref_set:
            missing = ref_set - set(sd.keys())
            extra = set(sd.keys()) - ref_set
            raise ValueError(
                f"key mismatch: {site} differs from {sites[0]} "
                f"(missing={list(missing)[:5]} extra={list(extra)[:5]})"
            )

    total = sum(raw_w)
    if total <= 0:
        raise ValueError(f"weights sum to {total} (<= 0); supply positive sample counts")
    norm_w = [w / total for w in raw_w]

    # Index of the highest-weight client -- its values are used verbatim for non-float
    # buffers (e.g. num_batches_tracked ints) that cannot be meaningfully averaged.
    dominant = max(range(len(norm_w)), key=lambda i: norm_w[i])

    agg = OrderedDict()
    for k in ref_keys:
        ref = states[0][k]
        if torch.is_floating_point(ref):
            # accumulate in float64 for numerical stability, cast back to the original dtype
            acc = torch.zeros_like(ref, dtype=torch.float64)
            for w, sd in zip(norm_w, states):
                acc += sd[k].to(torch.float64) * w
            agg[k] = acc.to(ref.dtype)
        else:
            agg[k] = states[dominant][k].clone()

    info = {
        "sites": sites,
        "raw_weights": raw_w,
        "norm_weights": norm_w,
        "n_keys": len(ref_keys),
        "dominant_site": sites[dominant],
        "round": int(round_idx),
    }
    logger(
        f"[aggregate] round {round_idx}: averaged {len(sites)} clients over {len(ref_keys)} keys; "
        f"weights={dict(zip(sites, [round(w, 4) for w in norm_w]))}"
    )
    return agg, info


def _parse_weights(s: str | None) -> dict | None:
    """Parse 'siteA=1234,siteB=5678' into {'siteA':1234.0,'siteB':5678.0}."""
    if not s:
        return None
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        site, _, val = pair.partition("=")
        out[site.strip()] = float(val)
    return out


def main():
    ap = argparse.ArgumentParser(description="Re-aggregate per-client round-x checkpoints into a warm-start global.")
    ap.add_argument("--dir", required=True, help="folder with {site}_gmic_model_round_{x}.pth files")
    ap.add_argument("--round", type=int, required=True, help="round number to aggregate")
    ap.add_argument("--out", required=True, help="output .pth path for the aggregated global")
    ap.add_argument("--weights", default=None,
                    help="optional 'siteA=Ntrain,siteB=Ntrain' sample-count weights (equal if omitted)")
    ap.add_argument("--pattern", default="*_gmic_model_round_{round}.pth",
                    help="filename glob with a {round} placeholder")
    args = ap.parse_args()

    agg, info = aggregate_round_checkpoints(
        args.dir, args.round, _parse_weights(args.weights), args.pattern
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(agg, args.out)
    print(f"\nWrote aggregated global -> {os.path.abspath(args.out)}")
    print(f"  sites={info['sites']}  norm_weights={[round(w, 4) for w in info['norm_weights']]}")
    if not args.weights:
        print("  NOTE: equal weighting used (no --weights). Correct only if sites have equal train-set sizes.")
    print("\nTo resume, set in config_fed_server.json (persistor args):")
    print(f'  "source_ckpt_file_full_name": "{os.path.abspath(args.out)}"')


if __name__ == "__main__":
    main()
