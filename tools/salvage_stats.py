#!/usr/bin/env python3
"""Offline paper-statistics for the FedProx salvage, from gathered prediction CSVs.

This is the OFFLINE path (CSV in -> table out). The same numbers are also produced live during the
federated salvage run: per-site rows are logged at each node and the POOLED row is logged on the
server (see tools/salvage_runbook.md). Use this script when you can gather the prediction CSVs into
one directory; use the in-run logs when you cannot (e.g. no file access to a node).

Estimators live in custom/salvage_metrics.py (single source of truth, shared with
the client executor and the server aggregator); this script only does CSV IO, grouping, and output.

Input CSVs are as written by the executor's `_dump_predictions` (columns: site_id, round, method,
split, exam_id, view, path, prob_malignant, label). Rows are grouped by (method, round). For each
group the TEST endpoint is reported per site and pooled; the Youden threshold for each endpoint is
chosen on the SAME endpoint's VALIDATION predictions. POOLED is reported only when every site ran the
same model (method != deployed_local).

Usage:
    python tools/salvage_stats.py --pred-dir <gathered_csvs> --out-prefix results/fedprox_salvage
    python tools/salvage_stats.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# Single source of truth for the estimators: the shared repo-root custom/ (sibling of tools/).
_CUSTOM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "custom"))
if _CUSTOM not in sys.path:
    sys.path.insert(0, _CUSTOM)
from salvage_metrics import breast_aggregate, endpoint_metrics, format_endpoint, selftest_estimators  # noqa: E402

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    pd = None
    _PANDAS_ERR = e

_REQUIRED_COLS = ["site_id", "round", "method", "split", "exam_id", "view", "prob_malignant", "label"]
_POOLED_INVALID = {"deployed_local"}  # per-site models differ -> pooling them is meaningless


def analyze_endpoint(val_df, test_df, label):
    vp, vy = breast_aggregate(val_df["site_id"], val_df["exam_id"], val_df["view"],
                              val_df["prob_malignant"], val_df["label"])
    tp, ty = breast_aggregate(test_df["site_id"], test_df["exam_id"], test_df["view"],
                              test_df["prob_malignant"], test_df["label"])
    m = endpoint_metrics(vp, vy, tp, ty)
    m["endpoint"] = label
    return m


def analyze_group(df_group, allow_pooled=True):
    val = df_group[df_group["split"] == "val"]
    test = df_group[df_group["split"] == "test"]
    rows = []
    if len(val) and len(test):
        if allow_pooled:
            rows.append(analyze_endpoint(val, test, "POOLED"))
        for site in sorted(set(test["site_id"]) & set(val["site_id"])):
            rows.append(analyze_endpoint(val[val["site_id"] == site],
                                         test[test["site_id"] == site], site))
    return rows


def load_predictions(pred_dir):
    if pd is None:
        raise RuntimeError(f"pandas is required to load CSVs: {_PANDAS_ERR}")
    paths = sorted(glob.glob(os.path.join(pred_dir, "*predictions*.csv")))
    if not paths:
        raise FileNotFoundError(f"no *predictions*.csv found under {pred_dir}")
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        missing = [c for c in _REQUIRED_COLS if c not in d.columns]
        if missing:
            print(f"[warn] skipping {p}: missing columns {missing}", file=sys.stderr)
            continue
        frames.append(d[_REQUIRED_COLS])
    if not frames:
        raise RuntimeError("no usable prediction CSVs (all missing required columns)")
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] {len(paths)} files, {len(df)} view-level rows, "
          f"{df[['method','round']].drop_duplicates().shape[0]} (method,round) groups", file=sys.stderr)
    return df


def run(pred_dir, out_prefix=None):
    df = load_predictions(pred_dir)
    all_rows = []
    for (method, rnd), g in df.groupby(["method", "round"]):
        print(f"\n=== method={method} round={rnd} ===")
        for row in analyze_group(g, allow_pooled=method not in _POOLED_INVALID):
            row["method"] = method
            row["round"] = int(rnd)
            all_rows.append(row)
            print("  " + format_endpoint(row["endpoint"], row))
    if out_prefix and all_rows:
        out_csv = f"{out_prefix}_metrics.csv"
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        cols = ["method", "round", "endpoint", "n_breasts", "n_pos", "val_auc", "test_auc",
                "auc_lo", "auc_hi", "youden_thr", "sensitivity", "sens_lo", "sens_hi",
                "specificity", "spec_lo", "spec_hi"]
        pd.DataFrame(all_rows)[cols].to_csv(out_csv, index=False)
        _write_markdown(all_rows, f"{out_prefix}_table.md")
        print(f"\n[out] wrote {out_csv} and {out_prefix}_table.md", file=sys.stderr)
    return all_rows


def _write_markdown(rows, path):
    lines = ["| method | round | endpoint | n (pos) | val AUC | test AUC [95% CI] | Youden thr | "
             "Sens [95% CI] | Spec [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['round']} | {r['endpoint']} | {r['n_breasts']} ({r['n_pos']}) | "
            f"{r['val_auc']:.4f} | {r['test_auc']:.4f} [{r['auc_lo']:.4f}–{r['auc_hi']:.4f}] | "
            f"{r['youden_thr']:.3f} | {r['sensitivity']:.3f} [{r['sens_lo']:.3f}–{r['sens_hi']:.3f}] | "
            f"{r['specificity']:.3f} [{r['spec_lo']:.3f}–{r['spec_hi']:.3f}] |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", help="directory of gathered prediction CSVs")
    ap.add_argument("--out-prefix", default=None, help="write <prefix>_metrics.csv and <prefix>_table.md")
    ap.add_argument("--selftest", action="store_true", help="run estimator self-tests and exit")
    args = ap.parse_args()
    if args.selftest:
        ok = selftest_estimators()
        print("[selftest]", "ALL PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    if not args.pred_dir:
        ap.error("--pred-dir is required (or use --selftest)")
    run(args.pred_dir, args.out_prefix)


if __name__ == "__main__":
    main()
