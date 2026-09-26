#!/usr/bin/env python
# Part of the federated-learning extension of GMIC.
# Copyright (C) 2026 Shepherd Research Lab, University of Hawaiʻi Cancer Center.
# Licensed under GNU AGPLv3 (see LICENSE and NOTICE).
"""Run per-race subgroup fairness (subgroup_fairness.py) for EVERY method, one site.

For each method it finds the deployed model's best-VALIDATION round (breast-level HIPIMR val
AUC, the same operating point the paper selects), then runs subgroup_fairness.run() on that
round's test preds with the matching val preds for the Youden threshold. Writes one CSV per
method plus a combined table, and prints a group x method pivot of sens@spec (the equity
headline). Pretrained baseline is fixed at round 0.

Run from /workspace (the repo root inside the container):
    python scripts/run_all_subgroups.py
Edit METHODS below if a folder name differs, or pin a round by replacing None with an int.

FedBN note: 'incoming_global' is the shared aggregate (FedBN's global part). FedBN's actual
per-site deployment also carries LOCAL BatchNorm, which is only dumped at the FINAL round under
tag 'fedbn' (no per-round series). If the paper's Table 2 FedBN row used that BN-adapted model,
change the fedbn row's tag to 'fedbn' and its round to that final round.
"""
import glob, os, re, sys
from sklearn.metrics import roc_auc_score
import subgroup_fairness as sf

SITE = "HIPIMR"                 # client name: both the prediction-file prefix and the output label
META = "data/gmic_df_HIPIMR_full_20260604_113112.csv"
PROC = "data/processed"
OUTDIR = "subgroup_out"
ETH_COL = "ETH_DESCR"         # registry column with the race/ethnicity description
ETH_CODE_COL = "ETH"          # registry column with the race/ethnicity code
ETH_MAP_FILE = None           # path to a JSON {code: group} map, or None for subgroup_fairness's built-in

# label, run_dir, file tag, round (None = auto best-val; int = pin, e.g. 0 for pretrained)
METHODS = [
    ("pretrained", f"{PROC}/fedavg_3client_20260611",                      "pretrained_baseline",        0),
    ("fedavg",     f"{PROC}/fedavg_3client_20260611",                      "incoming_global",            None),
    ("fedprox",    f"{PROC}/fedprox_mu0.1_3client_20260623",               "incoming_global",            None),
    ("fedbn",      f"{PROC}/fedbn_3client_20260614",                       "incoming_global",            None),  # see FedBN note
    ("ditto",      f"{PROC}/ditto_l0.05_3client_20260623",                 "ditto_perround",             None),
    ("ditto_mw",   f"{PROC}/ditto_mw_g0.01_l0.5_f0_amp_3client_20260724",  "ditto_modulewise_perround",  None),
]


def _round_of(path):
    m = re.search(r"_round(\d+)_val\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def best_val_round(run_dir, tag, meta):
    """Highest breast-level HIPIMR val AUC across all dumped rounds for this tag."""
    best_r, best_auc = None, -1.0
    for vp in sorted(glob.glob(os.path.join(run_dir, f"{SITE}_predictions_{tag}_round*_val.csv"))):
        r = _round_of(vp)
        if r is None:
            continue
        probs, labs, _info, _un = sf.breast_aggregate(sf.load(vp), meta)
        y = [labs[k] for k in probs]; s = [probs[k] for k in probs]
        if len(set(y)) < 2:
            continue
        auc = roc_auc_score(y, s)
        if auc > best_auc:
            best_auc, best_r = auc, r
    return best_r, best_auc


def main():
    if not os.path.isfile(META):
        sys.exit(f"[fatal] meta CSV not found: {META} (run from /workspace)")
    eth_map = None
    if ETH_MAP_FILE:
        import json
        with open(ETH_MAP_FILE, encoding="utf-8") as f:
            eth_map = json.load(f)
    meta = sf.load_meta(META, eth_col=ETH_COL, eth_code_col=ETH_CODE_COL)
    os.makedirs(OUTDIR, exist_ok=True)

    combined = []          # (method, row-tuple) for the merged CSV
    pivot = {}             # group -> {method -> sens@spec}
    for label, run_dir, tag, pinned in METHODS:
        print("\n" + "#" * 70)
        if not os.path.isdir(run_dir):
            print(f"### {label}: folder missing ({run_dir}) -- skipping")
            continue
        if pinned is not None:
            rnd, vauc = pinned, float("nan")
        else:
            rnd, vauc = best_val_round(run_dir, tag, meta)
            if rnd is None:
                print(f"### {label}: no usable val files for tag '{tag}' in {run_dir} -- skipping")
                continue
        test = os.path.join(run_dir, f"{SITE}_predictions_{tag}_round{rnd}_test.csv")
        val  = os.path.join(run_dir, f"{SITE}_predictions_{tag}_round{rnd}_val.csv")
        if not os.path.isfile(test):
            print(f"### {label}: test file missing ({test}) -- skipping")
            continue
        vtxt = "" if pinned is not None else f" (val AUC {vauc:.4f})"
        print(f"### {label}: tag={tag} round={rnd}{vtxt}")
        out = os.path.join(OUTDIR, f"subgroup_{label}.csv")
        try:
            rows = sf.run(test, META, val_csv=(val if os.path.isfile(val) else None), out_csv=out,
                          site=SITE, eth_col=ETH_COL, eth_code_col=ETH_CODE_COL, eth_map=eth_map)
        except SystemExit as e:               # subgroup_fairness aborts on an unmatched row
            print(f"### {label}: ABORTED -- {e}")
            continue
        for row in rows:
            combined.append((label,) + tuple(row))
            g, N, npos = row[0], row[1], row[2]
            sfs = row[12]                     # sens_at_spec column
            pivot.setdefault(g, {})[label] = (sfs, N, npos)

    # combined long-format CSV
    if combined:
        import csv
        allout = os.path.join(OUTDIR, "ALL_methods_subgroups.csv")
        with open(allout, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "group", "N", "pos", "auc", "auc_lo", "auc_hi",
                        "sens", "sens_lo", "sens_hi", "spec", "ppv", "fpr",
                        f"sens_at_spec{sf.FIXED_SPEC}", f"spec_at_sens{sf.FIXED_SENS}"])
            for (label, g, N, npos, auc, lo, hi, sens, sl, sh, spec, ppv, fpr, sfs, sps) in combined:
                w.writerow([label, g, N, npos, f"{auc:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                            f"{sens:.4f}", f"{sl:.4f}", f"{sh:.4f}", f"{spec:.4f}",
                            f"{ppv:.4f}", f"{fpr:.4f}", f"{sfs:.4f}", f"{sps:.4f}"])
        print(f"\n[out] combined table -> {os.path.abspath(allout)}")

    # equity headline: sens@spec by group x method
    methods_seen = [m[0] for m in METHODS if any(m[0] in v for v in pivot.values())]
    key_groups = ["Overall", "NHPI", "Asian (all)", "White", "Other"]
    print(f"\n=== sens@spec={sf.FIXED_SPEC} by group x method (N in parens) ===")
    hdr = f"{'group':<14}" + "".join(f"{m:>14}" for m in methods_seen)
    print(hdr); print("-" * len(hdr))
    for g in key_groups + [g for g in pivot if g not in key_groups]:
        if g not in pivot:
            continue
        line = f"{g:<14}"
        for m in methods_seen:
            if m in pivot[g]:
                sfs, N, npos = pivot[g][m]
                line += f"{f'{sfs:.3f}({N})':>14}"
            else:
                line += f"{'-':>14}"
        print(line)
    print("\nPer-method tables in", OUTDIR + "/;", "read the [eth-map] audit in each run's stdout "
          "once to confirm the race mapping before trusting these.")


if __name__ == "__main__":
    main()
