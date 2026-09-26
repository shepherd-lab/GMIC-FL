"""Pure estimators for the FedProx salvage paper-stats, dependency-light (numpy only; scipy used
for the normal quantile if present, else the 95% z is hard-coded). Lives in app/custom so the FL
client executor (per-site, logged at each node) AND the server aggregator (pooled, logged on the
server) can both import it; tools/salvage_stats.py imports it too for the offline CSV path.

All metrics are BREAST-LEVEL (GMIC's reported unit): a breast = (site, exam, laterality); breast
prob = mean of its views' malignant probs; breast label = max over views. The reportable bundle per
endpoint: AUC + DeLong 95% CI on test, val AUC (for round selection), and the Youden-J operating
point chosen on VALIDATION and applied to TEST with Wilson 95% CIs on sens/spec.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

try:
    from scipy.stats import norm as _norm
    def _z(alpha=0.05):
        return float(_norm.ppf(1.0 - alpha / 2.0))
except Exception:  # scipy optional
    def _z(alpha=0.05):
        if abs(alpha - 0.05) < 1e-9:
            return 1.959963984540054
        raise RuntimeError("scipy required for alpha != 0.05")


# ---------------- breast aggregation (mirrors fl_utils.breast_aggregate, site-aware) ------------ #
def parse_laterality(view):
    s = str(view).strip().upper()
    if not s or s[0] not in ("L", "R"):
        raise ValueError(f"cannot parse laterality from view {view!r} (expected L-*/R-*)")
    return s[0]


def breast_aggregate(site_ids, exam_ids, views, probs, labels):
    """View-level rows -> breast-level (prob=mean, label=max), keyed by (site, exam, laterality)."""
    groups = OrderedDict()
    for sid, eid, v, p, y in zip(site_ids, exam_ids, views, probs, labels):
        key = (sid, eid, parse_laterality(v))
        g = groups.get(key)
        if g is None:
            g = {"p": [], "y": []}
            groups[key] = g
        g["p"].append(float(p))
        g["y"].append(int(y))
    bp = np.array([np.mean(g["p"]) for g in groups.values()], dtype=float)
    by = np.array([int(max(g["y"])) for g in groups.values()], dtype=int)
    return bp, by


# ---------------- DeLong AUC + CI (fast algorithm, Sun & Xu 2014) -------------------------------- #
def _midrank(x):
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = x.shape[0]
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_auc_ci(labels, scores, alpha=0.05):
    """Single-classifier AUC with a DeLong CI on the logit scale. (auc, lo, hi, n_pos, n_neg)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    m, n = pos.shape[0], neg.shape[0]
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), m, n
    tx, ty = _midrank(pos), _midrank(neg)
    txy = _midrank(np.concatenate([pos, neg]))
    txy_pos, txy_neg = txy[:m], txy[m:]
    auc = (txy_pos.sum() - m * (m + 1) / 2.0) / (m * n)
    v01 = (txy_pos - tx) / n
    v10 = 1.0 - (txy_neg - ty) / m
    s01 = np.var(v01, ddof=1) if m > 1 else 0.0
    s10 = np.var(v10, ddof=1) if n > 1 else 0.0
    var_auc = s01 / m + s10 / n
    se = math.sqrt(var_auc) if var_auc > 0 else 0.0
    if se == 0.0 or auc in (0.0, 1.0):
        return float(auc), float(auc), float(auc), m, n
    z = _z(alpha)
    eps = 1e-12
    a = min(max(auc, eps), 1 - eps)
    logit = math.log(a / (1 - a))
    se_logit = se / (a * (1 - a))
    lo = 1.0 / (1.0 + math.exp(-(logit - z * se_logit)))
    hi = 1.0 / (1.0 + math.exp(-(logit + z * se_logit)))
    return float(auc), float(lo), float(hi), m, n


# ---------------- Youden threshold (on val) + sens/spec with Wilson CIs (on test) --------------- #
def youden_threshold(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    if P == 0 or N == 0:
        return 0.5
    cands = np.unique(scores)
    cands = np.concatenate([[cands[0] - 1e-9], cands])
    best_j, best_t = -np.inf, 0.5
    for t in cands:
        pred = scores >= t
        sens = int(np.sum(pred & (labels == 1))) / P
        spec = int(np.sum(~pred & (labels == 0))) / N
        j = sens + spec - 1.0
        if j > best_j or (abs(j - best_j) < 1e-12 and t > best_t):
            best_j, best_t = j, float(t)
    return best_t


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    z = _z(alpha)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def endpoint_metrics(val_probs, val_labels, test_probs, test_labels):
    """One reportable endpoint from breast-level arrays. Youden on val -> applied to test."""
    thr = youden_threshold(val_labels, val_probs)
    auc, lo, hi, n_pos, n_neg = delong_auc_ci(test_labels, test_probs)
    val_auc, _, _, _, _ = delong_auc_ci(val_labels, val_probs)
    test_labels = np.asarray(test_labels).astype(int)
    test_probs = np.asarray(test_probs).astype(float)
    pred = test_probs >= thr
    P = int((test_labels == 1).sum())
    Nn = int((test_labels == 0).sum())
    tp = int(np.sum(pred & (test_labels == 1)))
    tn = int(np.sum(~pred & (test_labels == 0)))
    sens = tp / P if P else float("nan")
    spec = tn / Nn if Nn else float("nan")
    sens_lo, sens_hi = wilson_ci(tp, P)
    spec_lo, spec_hi = wilson_ci(tn, Nn)
    return {
        "n_breasts": int(n_pos + n_neg), "n_pos": int(n_pos),
        "val_auc": val_auc,
        "test_auc": auc, "auc_lo": lo, "auc_hi": hi,
        "youden_thr": float(thr),
        "sensitivity": sens, "sens_lo": sens_lo, "sens_hi": sens_hi,
        "specificity": spec, "spec_lo": spec_lo, "spec_hi": spec_hi,
    }


def format_endpoint(label, m):
    return (f"{label:<10} n={m['n_breasts']:>4} (pos={m['n_pos']:>3})  valAUC={m['val_auc']:.4f}  "
            f"testAUC={m['test_auc']:.4f} [{m['auc_lo']:.4f}-{m['auc_hi']:.4f}]  thr={m['youden_thr']:.3f}  "
            f"Sens={m['sensitivity']:.3f} [{m['sens_lo']:.3f}-{m['sens_hi']:.3f}]  "
            f"Spec={m['specificity']:.3f} [{m['spec_lo']:.3f}-{m['spec_hi']:.3f}]")


def selftest_estimators():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=500)
    s = np.round(rng.rand(500) + 0.3 * y, 2)
    auc, lo, hi, npos, nneg = delong_auc_ci(y, s)
    ranks = _midrank(s)
    auc_ref = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    assert abs(auc - auc_ref) < 1e-9, (auc, auc_ref)
    assert lo < auc < hi
    lo_w, hi_w = wilson_ci(8, 10)
    assert abs(lo_w - 0.4901) < 1e-3 and abs(hi_w - 0.9433) < 1e-3, (lo_w, hi_w)
    yy = np.array([0, 0, 0, 1, 1, 1]); ss = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = endpoint_metrics(ss, yy, ss, yy)   # endpoint_metrics(probs, labels, probs, labels)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0, m
    bp, by = breast_aggregate(["A"] * 4, ["e1"] * 4, ["L-CC", "L-MLO", "R-CC", "R-MLO"],
                              [0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
    assert len(bp) == 2 and abs(bp[0] - 0.3) < 1e-9 and abs(bp[1] - 0.7) < 1e-9, (bp, by)
    assert list(by) == [0, 1], by
    return True


if __name__ == "__main__":
    print("salvage_metrics self-test:", "PASS" if selftest_estimators() else "FAIL")
