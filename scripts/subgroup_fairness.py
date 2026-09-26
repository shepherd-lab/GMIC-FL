#!/usr/bin/env python
# Part of the federated-learning extension of GMIC.
# Copyright (C) 2026 Shepherd Research Lab, University of Hawaiʻi Cancer Center.
# Licensed under GNU AGPLv3 (see LICENSE and NOTICE).
"""Per-race subgroup fairness for one site -- post-hoc, no model rerun / FL job.

Joins a deployed model's Site A test predictions to registry race/ethnicity
(ETH_DESCR) and reports per-group AUC + operating-point metrics with CIs.
Addresses the reviewers' central point: the paper's equity measure was cross-SITE
AUC spread, not performance across demographic GROUPS. This adds the group view,
plus the broader clinical metrics (sens@spec, spec@sens, PPV, FPR) reviewers asked for.

JOIN (fixed 2026-08): the prediction CSV's `exam_id` column is a *positional* index
assigned during exam-list enumeration (data_loader `_unpack_exam_into_images`), NOT the
registry exam_id -- so it CANNOT be joined to the metadata. The real identity lives in
the `path` column, whose basename is the loader's image_id:
    {patient_id}_{exam_id}_{laterality}_{view}     (data_loader build_exam_list, line 358)
        e.g.  10033_10033_20060416_L_CC.png
We rebuild that exact key from the metadata's own columns and match on it -- identical
construction on both sides, so matching is exact. Breasts are aggregated by the metadata's
(patient_id, exam_id, laterality), and race is read from that same joined row. Site is not
filtered: race/ethnicity exists only for HIPIMR, and --meta is the HIPIMR-only registry CSV.

Prediction CSV columns: site_id,round,method,split,exam_id,view,path,prob_malignant,label
  (only `path`, `prob_malignant`, `label` are used; exam_id/view/site_id are ignored.)
Metadata CSV (gmic_df_HIPIMR_*.csv): patient_id, exam_id, laterality, view, file_path,
  image_filename, ..., ETH, ETH_DESCR, ...

Usage:
  python subgroup_fairness.py \
      --pred  HIPIMR_predictions_ditto_perround_round49_test.csv \
      --meta  gmic_df_HIPIMR_full_20260604_113112.csv \
      --val   HIPIMR_predictions_ditto_perround_round49_val.csv   # sets the Youden operating point

Run once per deployed model (Ditto r49, then pretrained r0) to show whether personalization
narrows any per-group sensitivity gap. Works in a notebook too:
  from subgroup_fairness import run; run("pred.csv","meta.csv",val_csv="val.csv")
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve
from scipy.stats import norm

# ---------------- config ----------------
CI = 0.95
N_BOOT = 2000
SEED = 0
FIXED_SPEC = 0.90      # sensitivity at this specificity
FIXED_SENS = 0.90      # specificity at this sensitivity
MIN_POS = 5            # groups with fewer positives flagged as unstable
STRICT_MATCH = True    # hard-fail if any prediction breast has no metadata match

# Default race/ethnicity code -> reporting-group map. The "code" is the text before the first
# ')' in the race column value (e.g. '5) NHOPI' -> '5'), lower-cased; keys are matched as
# PREFIXES, longest first, so '3a' wins over '3'. Reserved keys "_default" (code present but
# unmapped) and "_blank" (value missing) set the catch-all labels. Override the whole map at
# runtime with --eth-map <json> so the tool need not carry any site-specific scheme.
DEFAULT_ETH_MAP = {
    "5":  "NHPI",
    "2":  "White",
    "3a": "Asian: Japanese", "3b": "Asian: Chinese",
    "3c": "Asian: Filipino", "3d": "Asian: Korean",
    "3":  "Asian: Other",              # any other 3x code
    "_default": "Other",               # e.g. 1 Hispanic, 4 Black, 6 AIAN
    "_blank":   "Unknown",             # race column empty/missing
}

# ---------------- helpers ----------------
def _stem(p):
    """Basename without extension: '/a/b/10033_10033_20060416_L_CC.png' -> that id string."""
    return os.path.splitext(os.path.basename(str(p).replace("\\", "/")))[0]

def eth_to_group(desc, mapping=None):
    """Map a race/ethnicity description to a reporting group via `mapping` (default
    DEFAULT_ETH_MAP). Non-reserved keys are matched as prefixes of the parsed code, longest
    first. VERIFY against the [eth-map] audit the run prints before trusting the groups."""
    mapping = DEFAULT_ETH_MAP if mapping is None else mapping
    d = (desc or "").strip()
    if not d:
        return mapping.get("_blank", "Unknown")
    code = d.split(")")[0].strip().lower()
    for k in sorted((x for x in mapping if not str(x).startswith("_")), key=len, reverse=True):
        if code.startswith(str(k).lower()):
            return mapping[k]
    return mapping.get("_default", "Other")

def is_asian_group(desc, mapping=None):
    """True if desc resolves to any 'Asian...' group (drives the aggregated 'Asian (all)' row)."""
    return eth_to_group(desc, mapping).startswith("Asian")

def group_order(mapping):
    """Row order for whatever groups `mapping` yields: Overall, then each distinct group (with an
    'Asian (all)' aggregate inserted before the first 'Asian:' subgroup), then the blank label."""
    seen, ordered, asian_done = [], ["Overall"], False
    for k, g in mapping.items():
        if str(k).startswith("_") or g in seen:
            continue
        seen.append(g)
        if g.startswith("Asian:") and not asian_done:
            ordered.append("Asian (all)"); asian_done = True
        ordered.append(g)
    ordered.append(mapping.get("_blank", "Unknown"))
    return ordered

def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def load_meta(meta_csv, eth_col="ETH_DESCR", eth_code_col="ETH"):
    """Index the HIPIMR registry by image_id, reconstructed identically to the data loader:
    f'{patient_id}_{exam_id}_{laterality}_{view}'. Also indexes by image_filename stem and
    file_path stem as fallbacks, so we match whichever form the prediction `path` took."""
    by_img = {}
    dups = 0
    for r in load(meta_csv):
        info = {
            "patient_id": str(r["patient_id"]),
            "exam_id":    str(r["exam_id"]),
            "laterality": str(r["laterality"]).strip().upper()[:1] or str(r["laterality"]),
            "view":       str(r["view"]).strip().upper(),
            "eth":        (r.get(eth_code_col) or "").strip(),
            "eth_descr":  (r.get(eth_col) or "").strip(),
        }
        keys = {f"{r['patient_id']}_{r['exam_id']}_{r['laterality']}_{r['view']}"}
        if r.get("image_filename"): keys.add(_stem(r["image_filename"]))
        if r.get("file_path"):      keys.add(_stem(r["file_path"]))
        for k in keys:
            if k in by_img and by_img[k]["patient_id"] != info["patient_id"]:
                dups += 1
            by_img[k] = info
    if dups:
        print(f"[warn] {dups} metadata image-key collisions across different patients "
              f"(unexpected; check patient_id/exam_id uniqueness).")
    return by_img

def breast_aggregate(rows, meta):
    """Join each prediction row to registry metadata via path->image_id, then aggregate to the
    breast unit (patient_id, exam_id, laterality) -> mean prob, max label. Returns per-breast
    prob/label dicts, a per-breast metadata dict, and the list of unmatched prediction ids."""
    P, Y, INFO = defaultdict(list), defaultdict(list), {}
    unmatched = []
    for r in rows:
        img = _stem(r["path"])
        m = meta.get(img)
        if m is None:
            unmatched.append(img)
            continue
        key = (m["patient_id"], m["exam_id"], m["laterality"])
        P[key].append(float(r["prob_malignant"]))
        Y[key].append(int(float(r["label"])))
        INFO[key] = m
    probs = {k: float(np.mean(P[k])) for k in P}
    labs  = {k: int(max(Y[k]))       for k in P}
    return probs, labs, INFO, unmatched

def delong_ci(y, s, alpha=CI):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]; m, n = len(pos), len(neg)
    if m == 0 or n == 0: return float("nan"), float("nan"), float("nan")
    cmp = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = float(cmp.mean())
    if m < 2 or n < 2: return auc, float("nan"), float("nan")
    V10, V01 = cmp.mean(1), cmp.mean(0)
    se = float(np.sqrt(V10.var(ddof=1) / m + V01.var(ddof=1) / n))
    z = norm.ppf(1 - (1 - alpha) / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)

def op_metrics(y, s, thr):
    y = np.asarray(y).astype(int); pred = (np.asarray(s, float) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return dict(sens=tp / (tp + fn) if tp + fn else float("nan"),
                spec=tn / (tn + fp) if tn + fp else float("nan"),
                ppv=tp / (tp + fp) if tp + fp else float("nan"),
                fpr=fp / (fp + tn) if fp + tn else float("nan"))

def sens_at_spec(y, s, target):
    fpr, tpr, _ = roc_curve(y, s); ok = (1 - fpr) >= target
    return float(tpr[ok].max()) if ok.any() else float("nan")

def spec_at_sens(y, s, target):
    fpr, tpr, _ = roc_curve(y, s); ok = tpr >= target
    return float((1 - fpr)[ok].max()) if ok.any() else float("nan")

def boot_sens_ci(y, s, thr, n_boot=N_BOOT, seed=SEED, alpha=CI):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); y = np.asarray(y); s = np.asarray(s); v = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        v.append(op_metrics(y[b], s[b], thr)["sens"])
    if not v: return (float("nan"), float("nan"))
    return (float(np.nanpercentile(v, 100 * (1 - alpha) / 2)),
            float(np.nanpercentile(v, 100 * (1 + alpha) / 2)))

def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s); return float(thr[int(np.argmax(tpr - fpr))])

# ---------------- main ----------------
def run(pred_csv, meta_csv, threshold=None, val_csv=None, out_csv=None,
        site="HIPIMR", eth_col="ETH_DESCR", eth_code_col="ETH", eth_map=None):
    eth_map = DEFAULT_ETH_MAP if eth_map is None else eth_map
    blank_label = eth_map.get("_blank", "Unknown")
    preds = load(pred_csv)
    print(f"[cols] prediction file columns: {list(preds[0].keys())}")
    meta = load_meta(meta_csv, eth_col=eth_col, eth_code_col=eth_code_col)
    print(f"[meta] indexed {len(meta)} image keys from {os.path.basename(meta_csv)}")

    probs, labs, info, unmatched = breast_aggregate(preds, meta)
    keys = list(probs)
    if unmatched:
        ex = ", ".join(unmatched[:5])
        msg = f"[match] {len(unmatched)}/{len(preds)} prediction rows had NO metadata match. examples: {ex}"
        if STRICT_MATCH:
            sys.exit(msg + "\n  Refusing to proceed: an unmatched row would silently drop a breast "
                            "from the fairness table. Check that --meta is the HIPIMR registry whose "
                            "patient_id/exam_id/laterality/view built these image filenames.")
        print("[warn] " + msg)
    print(f"[breasts] {len(keys)} breasts aggregated (patient_id, exam_id, laterality); "
          f"{int(sum(labs[k] for k in keys))} malignant.")

    # --- eth-map audit: every distinct (ETH, ETH_DESCR) -> assigned group, with breast counts ---
    seen = defaultdict(int)
    for k in keys:
        seen[(info[k]["eth"], info[k]["eth_descr"])] += 1
    print("\n[eth-map] distinct race codes seen (verify the group assignment is correct):")
    print(f"  {'ETH':<6}{'ETH_DESCR':<40}{'-> group':<20}{'breasts':>8}")
    n_unknown = 0
    for (code, descr), c in sorted(seen.items(), key=lambda kv: -kv[1]):
        g = eth_to_group(descr, eth_map)
        if g == blank_label: n_unknown += c
        dshow = (descr or "<blank>")
        if len(dshow) > 38: dshow = dshow[:37] + "…"
        print(f"  {code:<6}{dshow:<40}{g:<20}{c:>8}")
    if n_unknown:
        print(f"[warn] {n_unknown} breasts mapped to '{blank_label}' -- if that's unexpected, "
              f"fix the map (built-in DEFAULT_ETH_MAP or --eth-map) to match the values above.")

    # --- operating threshold ---
    if threshold is None:
        if val_csv:
            vprobs, vlabs, _, vun = breast_aggregate(load(val_csv), meta)
            if vun and STRICT_MATCH:
                sys.exit(f"[match] {len(vun)} validation rows had no metadata match; aborting.")
            vk = list(vprobs)
            threshold = youden([vlabs[k] for k in vk], [vprobs[k] for k in vk])
            print(f"\n[threshold] Youden on Site A validation ({os.path.basename(val_csv)}) = {threshold:.4f}")
        else:
            threshold = youden([labs[k] for k in keys], [probs[k] for k in keys])
            print(f"\n[threshold] Youden on THIS test set = {threshold:.4f}  "
                  f"(pass --val for an unbiased operating point)")

    groups = defaultdict(list)
    for k in keys:
        groups[eth_to_group(info[k]["eth_descr"], eth_map)].append(k)
    asian = [k for k in keys if is_asian_group(info[k]["eth_descr"], eth_map)]
    if asian and any(g.startswith("Asian:") for g in groups):
        groups["Asian (all)"] = asian

    order = group_order(eth_map)
    rows = []
    for g in order:
        ks = keys if g == "Overall" else groups.get(g, [])
        if not ks: continue
        y = [labs[k] for k in ks]; s = [probs[k] for k in ks]; npos = int(sum(y))
        auc, lo, hi = delong_ci(y, s)
        om = op_metrics(y, s, threshold); sl, sh = boot_sens_ci(y, s, threshold)
        rows.append((g, len(ks), npos, auc, lo, hi, om["sens"], sl, sh,
                     om["spec"], om["ppv"], om["fpr"],
                     sens_at_spec(y, s, FIXED_SPEC), spec_at_sens(y, s, FIXED_SENS)))

    print(f"\n{site} subgroup metrics | thr={threshold:.4f} | "
          f"sens@spec={FIXED_SPEC} | spec@sens={FIXED_SENS}\n")
    h = (f"{'group':<17}{'N':>5}{'pos':>4}  {'AUC [95% DeLong]':<21}"
         f"{'sens':>7}{'[95% CI]':>16}{'spec':>7}{'PPV':>7}{'FPR':>7}{'s@sp':>7}{'sp@s':>7}")
    print(h); print("-" * len(h))
    for (g, N, npos, auc, lo, hi, sens, sl, sh, spec, ppv, fpr, sfs, sps) in rows:
        flag = " *" if npos < MIN_POS else ""
        print(f"{g:<17}{N:>5}{npos:>4}  {auc:.3f} [{lo:.3f},{hi:.3f}]  "
              f"{sens:>6.3f} [{sl:.3f},{sh:.3f}]{spec:>7.3f}{ppv:>7.3f}{fpr:>7.3f}"
              f"{sfs:>7.3f}{sps:>7.3f}{flag}")
    print(f"\n  * = fewer than {MIN_POS} positives: unstable, report N and treat as exploratory.")

    # --- machine-readable table, written next to wherever you run this (top-level /workspace) ---
    if out_csv is None:
        out_csv = _stem(pred_csv) + "_subgroups.csv"
    base = os.path.basename(pred_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred_file", "group", "N", "pos", "auc", "auc_lo", "auc_hi",
                    "sens", "sens_lo", "sens_hi", "spec", "ppv", "fpr",
                    f"sens_at_spec{FIXED_SPEC}", f"spec_at_sens{FIXED_SENS}",
                    "threshold", "unstable"])
        for (g, N, npos, auc, lo, hi, sens, sl, sh, spec, ppv, fpr, sfs, sps) in rows:
            w.writerow([base, g, N, npos, f"{auc:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                        f"{sens:.4f}", f"{sl:.4f}", f"{sh:.4f}", f"{spec:.4f}",
                        f"{ppv:.4f}", f"{fpr:.4f}", f"{sfs:.4f}", f"{sps:.4f}",
                        f"{threshold:.4f}", int(npos < MIN_POS)])
    print(f"[out] wrote {len(rows)} group rows -> {os.path.abspath(out_csv)}")
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Site A test prediction CSV (deployed model)")
    ap.add_argument("--meta", required=True, help="HIPIMR registry CSV with patient_id/exam_id/laterality/view + ETH_DESCR")
    ap.add_argument("--val", default=None, help="optional Site A validation prediction CSV (Youden operating point)")
    ap.add_argument("--threshold", type=float, default=None, help="fixed operating threshold (overrides --val)")
    ap.add_argument("--out", default=None, help="output CSV path (default: <pred_stem>_subgroups.csv in cwd)")
    ap.add_argument("--site", default="HIPIMR", help="site label for the output header")
    ap.add_argument("--eth-col", default="ETH_DESCR", help="registry column holding the race/ethnicity description")
    ap.add_argument("--eth-code-col", default="ETH", help="registry column holding the race/ethnicity code")
    ap.add_argument("--eth-map", default=None,
                    help="JSON {code: group} (+ optional _default/_blank) overriding the built-in race map")
    a = ap.parse_args()
    emap = None
    if a.eth_map:
        with open(a.eth_map, encoding="utf-8") as f:
            emap = json.load(f)
    run(a.pred, a.meta, threshold=a.threshold, val_csv=a.val, out_csv=a.out,
        site=a.site, eth_col=a.eth_col, eth_code_col=a.eth_code_col, eth_map=emap)
