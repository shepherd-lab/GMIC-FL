# ============================================================================
# fl_utils.py - Federated-method primitives for the GMIC executor
# ----------------------------------------------------------------------------
# Reusable, framework-agnostic helpers (pure torch; no NVFLARE import) used by
# bc_executor.GMICFederatedExecutor to implement:
#   local | fedavg | fedprox | fedbn | ditto | ditto_modulewise
#
# Parameter -> {global, local, fusion} grouping was VERIFIED against the actual
# GMIC model (model/gmic.py) by importing it and walking named_parameters():
#   global : ds_net.*            (ResNetV2 downsampling backbone)
#            left_postprocess_net.*  (1x1 saliency conv; part of the global net)
#   local  : dn_resnet.*         (ResNetV1 patch backbone)
#   fusion : mil_attn_V/U/w.*    (gated MIL attention)
#            classifier_linear.* (patch-level classifier head)
#            fusion_dnn.*        (final global+local fusion head)
# Every one of the model's 259 state_dict keys maps to exactly one group
# (0 unassigned). BatchNorm lives entirely in the two backbones (41 BN modules:
# 21 in ds_net, 20 in dn_resnet -> 205 BN state_dict keys).
# ============================================================================

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "GLOBAL_PREFIXES",
    "LOCAL_PREFIXES",
    "FUSION_PREFIXES",
    "module_group",
    "assert_grouping_complete",
    "bn_state_keys",
    "load_global_into",
    "proximal_penalty",
    "clone_detached_state",
    "tolerant_load_pretrained",
    "gmic_outputs",
    "malignant_score",
    "gmic_malignant_loss",
    "classification_metrics",
    "site_selection_metrics",
    "VALID_SELECTION_METRICS",
    "replay_personalized",
    "breast_aggregate",
]

# ---- Confirmed prefix -> group mapping (see module docstring) ----------------
GLOBAL_PREFIXES: Tuple[str, ...] = ("ds_net", "left_postprocess_net")
LOCAL_PREFIXES: Tuple[str, ...] = ("dn_resnet",)
FUSION_PREFIXES: Tuple[str, ...] = (
    "mil_attn_V",
    "mil_attn_U",
    "mil_attn_w",
    "classifier_linear",
    "fusion_dnn",
)

# Non-learnable buffer that carries no information; ignore everywhere.
_IGNORE_PREFIXES: Tuple[str, ...] = ("_device_ref",)


def module_group(param_name: str) -> str:
    """Map a parameter/buffer name to exactly one of {global, local, fusion}.

    Raises KeyError for any name that is not in the ignore set and does not
    match a known prefix -- a wrong/incomplete grouping silently breaks
    module-wise Ditto, so we fail loudly instead.
    """
    top = param_name.split(".", 1)[0]
    if top in GLOBAL_PREFIXES:
        return "global"
    if top in LOCAL_PREFIXES:
        return "local"
    if top in FUSION_PREFIXES:
        return "fusion"
    if top in _IGNORE_PREFIXES:
        return "ignore"
    raise KeyError(
        f"Parameter '{param_name}' (top-level '{top}') is not assigned to any "
        f"module group. Update GLOBAL/LOCAL/FUSION_PREFIXES in fl_utils.py."
    )


def assert_grouping_complete(model: nn.Module) -> Dict[str, int]:
    """Assert every learnable parameter maps to exactly one group.

    Returns a dict of per-group parameter counts (ignored buffers excluded).
    """
    sizes = {"global": 0, "local": 0, "fusion": 0}
    for name, _ in model.named_parameters():
        g = module_group(name)
        if g == "ignore":
            continue
        sizes[g] += 1
    return sizes


def bn_state_keys(model: nn.Module) -> set:
    """Return the full set of BatchNorm state_dict keys for FedBN.

    Covers weight/bias/running_mean/running_var/num_batches_tracked for every
    BatchNorm1d/2d/3d module. These keys are skipped on the *load* side so each
    site keeps its own BN affine params and running statistics.
    """
    sd = model.state_dict()
    keys = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for suffix in (
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ):
                k = f"{name}.{suffix}" if name else suffix
                if k in sd:
                    keys.add(k)
    return keys


def load_global_into(
    model: nn.Module,
    global_state_dict: Dict[str, torch.Tensor],
    skip_keys: Iterable[str] = (),
) -> Dict[str, int]:
    """Copy global tensors into `model`, skipping any key in `skip_keys`.

    Shape-mismatched or unknown keys are skipped (never crashes on a partial
    payload). Used for the per-round global load:
      - fedbn -> skip_keys = bn_state_keys(model)  (local BN preserved)
      - others -> skip_keys = ()                    (full load)

    Returns counts: {loaded, skipped, mismatched, missing_in_payload}.
    """
    skip = set(skip_keys)
    own = model.state_dict()
    to_load = {}
    skipped = mismatched = 0
    for k, v in global_state_dict.items():
        if k in skip:
            skipped += 1
            continue
        if k not in own:
            continue  # unexpected key in payload; ignore
        if tuple(own[k].shape) != tuple(v.shape):
            mismatched += 1
            continue
        to_load[k] = v
    missing_in_payload = sum(
        1 for k in own if k not in to_load and k not in skip
    )
    # strict=False: keys we deliberately omitted stay at their current values.
    model.load_state_dict(to_load, strict=False)
    return {
        "loaded": len(to_load),
        "skipped": skipped,
        "mismatched": mismatched,
        "missing_in_payload": missing_in_payload,
    }


def clone_detached_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a detached CPU-independent clone of the model's state_dict.

    Used to freeze the round's received global weights as the proximal/Ditto
    reference `w_ref`.
    """
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def proximal_penalty(
    model: nn.Module,
    ref_state_dict: Dict[str, torch.Tensor],
    lam: Union[float, Dict[str, float]],
) -> torch.Tensor:
    """Scalar proximal penalty  Sum_p (lam/2) * ||p - ref_p||^2  over params.

    `lam` is either a single float (FedProx, plain Ditto) or a dict keyed by
    module group {"global","local","fusion"} (module-wise Ditto), in which case
    the per-parameter weight is chosen via module_group(name).

    Only learnable parameters are penalized (buffers like BN running stats are
    excluded). `ref_state_dict` should be a detached/frozen copy of the round's
    received global weights.
    """
    is_dict = isinstance(lam, dict)
    penalty = None
    for name, p in model.named_parameters():
        if name not in ref_state_dict:
            continue
        l = lam[module_group(name)] if is_dict else lam
        if l == 0:
            continue
        ref = ref_state_dict[name].to(device=p.device, dtype=p.dtype)
        term = (l / 2.0) * ((p - ref) ** 2).sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        # No params penalized (e.g. all lambdas zero): return a 0 on the model's
        # device so it can be added to the task loss without device errors.
        dev = next(model.parameters()).device
        return torch.zeros((), device=dev)
    return penalty


def tolerant_load_pretrained(
    model: nn.Module,
    ckpt_path: str,
    device: Union[str, torch.device] = "cpu",
    log_fn=None,
    backbone_match_threshold: float = 0.9,
    raise_on_partial: bool = False,
) -> Dict[str, object]:
    """Load a released GMIC checkpoint with full visibility into the match.

    Tries strict=True first; on failure falls back to strict=False and reports
    exactly which keys matched, were missing, and were unexpected. The released
    GMIC checkpoints key 1:1 to this model EXCEPT for two cosmetic items:
      - missing  '_device_ref'  (empty device-tracking buffer; carries no learned info)
      - unexpected 'shared_rep_filter.weight'  (an auxiliary 256x256x4x4 conv head outside
        the classification forward path; the source GMIC variant bolted it on after
        fusion_dnn -- this model's forward never uses it, so it is safely discarded;
        backbones + all four output heads still load 100%)
    so a clean load reports matched=258/259.

    Guards against a silent partial load: if the fraction of *backbone*
    (ds_net/dn_resnet) keys that matched falls below `backbone_match_threshold`,
    this logs an error (and raises if `raise_on_partial=True`).

    Returns a report dict with counts and the missing/unexpected key lists.
    """

    def _log(msg: str, level: int = logging.INFO):
        if log_fn is not None:
            try:
                log_fn(msg)
                return
            except Exception:
                pass
        logger.log(level, msg)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]  # some releases wrap as {"model": state_dict}

    own_keys = set(model.state_dict().keys())

    try:
        model.load_state_dict(sd, strict=True)
        report = {
            "strict": True,
            "matched": len(own_keys),
            "missing": 0,
            "unexpected": 0,
            "missing_keys": [],
            "unexpected_keys": [],
        }
        _log(f"[pretrained] strict load OK: {ckpt_path} ({len(own_keys)} keys)")
        return report

    except RuntimeError:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = list(missing)
        unexpected = list(unexpected)
        matched = len(own_keys) - len(missing)

        # Backbone-completeness guard (ignores the benign _device_ref buffer).
        backbone_keys = [
            k for k in own_keys if k.split(".", 1)[0] in ("ds_net", "dn_resnet")
        ]
        bb_missing = [k for k in missing if k.split(".", 1)[0] in ("ds_net", "dn_resnet")]
        bb_match_frac = (
            (len(backbone_keys) - len(bb_missing)) / max(len(backbone_keys), 1)
        )

        report = {
            "strict": False,
            "matched": matched,
            "missing": len(missing),
            "unexpected": len(unexpected),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "backbone_match_fraction": bb_match_frac,
        }
        _log(
            f"[pretrained] strict load failed; strict=False matched={matched}/"
            f"{len(own_keys)} missing={len(missing)} unexpected={len(unexpected)} "
            f"backbone_match={bb_match_frac:.3f}",
            logging.WARNING,
        )
        if missing:
            _log(f"[pretrained] missing keys: {missing[:20]}", logging.WARNING)
        if unexpected:
            _log(f"[pretrained] unexpected keys: {unexpected[:20]}", logging.WARNING)

        if bb_match_frac < backbone_match_threshold:
            emsg = (
                f"[pretrained] BACKBONE LOAD INCOMPLETE: only {bb_match_frac:.1%} of "
                f"backbone keys matched (threshold {backbone_match_threshold:.0%}). "
                f"The pretrained baseline would be meaningless. Check ckpt/remap."
            )
            _log(emsg, logging.ERROR)
            if raise_on_partial:
                raise RuntimeError(emsg)
        return report


# ============================================================================
# Native GMIC loss (malignant head only)
# ----------------------------------------------------------------------------
# Verified against upstream nyukat/GMIC (git show upstream/master:src/modeling/gmic.py):
# GMIC has THREE deep-supervised heads, all INDEPENDENT SIGMOIDS (never softmax):
#   y_global  = top-t%% pool of the sigmoid saliency map   -> probability in [0,1]
#   y_local   = sigmoid(classifier_linear(z))              -> probability in [0,1]
#   y_fusion  = sigmoid(fusion_dnn([GMP(h_g), z]))         -> probability in [0,1]
# (the only softmax in GMIC is the patch ATTENTION, not a class head.)
# Output order is [benign, malignant]; malignant = index 1 (run_model.py:
#   benign_pred, malignant_pred = pred[0,0], pred[0,1]).
#
# This project's data carries a single binary label (benign == not-malignant), so we
# supervise/evaluate the MALIGNANT head only and leave the benign head (index 0)
# present-but-unsupervised (so the released checkpoint still loads 1:1).
#
# The model's forward() here returns the fusion head as a RAW LOGIT (pre-sigmoid) so
# the fusion term can use the numerically-stable BCEWithLogits; y_global/y_local are
# already probabilities, so they use plain BCE. Class imbalance is handled by ONE
# per-sample weight applied IDENTICALLY to all three heads.
# ============================================================================

def gmic_outputs(outputs):
    """Unpack GMIC forward() output -> (fusion_logits, y_global, y_local, saliency).

    forward() returns a 4-tuple. Tolerant if a bare tensor is passed (treated as the
    fusion logits, the other heads None) so legacy/inference callers don't break.
    """
    if isinstance(outputs, (tuple, list)):
        fusion_logits = outputs[0]
        y_global = outputs[1] if len(outputs) > 1 else None
        y_local = outputs[2] if len(outputs) > 2 else None
        saliency = outputs[3] if len(outputs) > 3 else None
        return fusion_logits, y_global, y_local, saliency
    return outputs, None, None, None


def malignant_score(outputs, malignant_index: int = 1):
    """Per-image malignant probability = sigmoid(fusion_logit)[:, malignant_index].

    Cast to fp32 first: under autocast the fusion output is fp16/bf16, and bf16 cannot be
    converted to numpy (`.numpy()` raises). fp32 is also what every downstream consumer
    (AUC, DeLong dumps, saliency join) expects.
    """
    fusion_logits, _, _, _ = gmic_outputs(outputs)
    return torch.sigmoid(fusion_logits.float())[:, malignant_index]


def gmic_malignant_loss(outputs, targets, lambda_l1: float = 1e-5,
                        pos_weight: float = 3.0, malignant_index: int = 1):
    """Native GMIC deep-supervised loss on the malignant head only.

        L = BCEWithLogits(fusion[:,m]) + BCE(y_global[:,m]) + BCE(y_local[:,m])
            + lambda_l1 * mean_b( sum_ij |saliency[:,m]| )

    `targets` are breast-level binary labels (1 = cancer). A single per-sample weight
    w (= pos_weight for positives else 1.0) is applied IDENTICALLY to all three head
    losses so every head sees the same effective class balance.
    """
    fusion_logits, y_global, y_local, saliency = gmic_outputs(outputs)
    m = malignant_index
    # Cast to fp32: F.binary_cross_entropy is unsafe in fp16/bf16. The executor already
    # computes this loss OUTSIDE autocast, but cast defensively so any caller is safe.
    fusion_logits = fusion_logits.float()
    t = targets.to(dtype=fusion_logits.dtype).view(-1)
    w = torch.ones_like(t)
    w[t > 0.5] = float(pos_weight)
    eps = 1e-6

    loss = F.binary_cross_entropy_with_logits(fusion_logits[:, m], t, weight=w)
    if y_global is not None:
        loss = loss + F.binary_cross_entropy(y_global[:, m].float().clamp(eps, 1 - eps), t, weight=w)
    if y_local is not None:
        loss = loss + F.binary_cross_entropy(y_local[:, m].float().clamp(eps, 1 - eps), t, weight=w)
    if saliency is not None and lambda_l1:
        sal_m = saliency[:, m].float()
        l1 = sal_m.abs().sum(dim=tuple(range(1, sal_m.dim()))).mean()
        loss = loss + float(lambda_l1) * l1
    return loss


def classification_metrics(targets, probs, threshold: float = 0.5):
    """AUC (threshold-free) + operating-point metrics (acc/sens/spec/precision) at `threshold`.

    targets/probs are 1-D numpy arrays. AUC is NaN on a degenerate (single-class) split.
    """
    import numpy as _np
    from sklearn.metrics import roc_auc_score as _auc, accuracy_score as _acc
    targets = _np.asarray(targets).reshape(-1)
    probs = _np.asarray(probs).reshape(-1)
    n = len(targets)
    n_pos = int((targets == 1).sum())
    n_neg = int((targets == 0).sum())
    pred = (probs > float(threshold)).astype(int)
    tp = int(((pred == 1) & (targets == 1)).sum())
    tn = int(((pred == 0) & (targets == 0)).sum())
    fp = int(((pred == 1) & (targets == 0)).sum())
    fn = int(((pred == 0) & (targets == 1)).sum())
    auc = float(_auc(targets, probs)) if (n_pos > 0 and n_neg > 0) else float("nan")
    return {
        "auc": auc,
        "accuracy": 100.0 * _acc(targets, pred) if n else 0.0,
        "sensitivity": (tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "specificity": (tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "precision": (tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
        "threshold": float(threshold),
        "n_pos": n_pos, "total_samples": n,
    }


def _parse_laterality(view):
    """Extract breast laterality (L/R) from a view label. Robust to L-CC / R-MLO / 'LCC' /
    lowercase. Raises on anything that doesn't begin with L or R (fail loud, never drop)."""
    s = str(view).strip().upper()
    if not s or s[0] not in ("L", "R"):
        raise ValueError(f"cannot parse laterality from view {view!r} (expected L-*/R-*)")
    return s[0]


def breast_aggregate(exam_ids, views, probs, labels):
    """Aggregate per-IMAGE predictions to BREAST level (GMIC's reported unit).

    A breast = (exam_id, laterality), laterality parsed from the view label. The breast's
    prediction = MEAN of its views' malignant probabilities; the breast's label = MAX over its
    views (both views of a breast share a label, so max == any; max is robust to a stray 0).
    Don't assume exactly 2 views -- average whatever views exist for the breast.

    Returns (breast_probs, breast_labels) as numpy arrays. Every image must map to a breast;
    an unparseable view raises (no silent drop).
    """
    import numpy as _np
    from collections import OrderedDict
    groups = OrderedDict()  # (exam_id, lat) -> {"p": [..], "y": [..]}
    for eid, v, p, y in zip(exam_ids, views, probs, labels):
        key = (eid, _parse_laterality(v))
        g = groups.setdefault(key, {"p": [], "y": []})
        g["p"].append(float(p)); g["y"].append(int(y))
    bp = _np.array([_np.mean(g["p"]) for g in groups.values()], dtype=float)
    by = _np.array([int(max(g["y"])) for g in groups.values()], dtype=int)
    return bp, by


def replay_personalized(v_model, v_optimizer, anchor_trajectory, loss_fn, proximal_fn,
                        lam, device, n_rounds, e_personalized, batch_iter_factory,
                        anchor_is_start_of_round=True):
    """Personalized-only Ditto replay against a CACHED global trajectory.

    Reproduces interleaved Ditto's personal-model trajectory WITHOUT recomputing FedAvg:
    for each round t in 0..n_rounds-1, anchor to the cached global w^t (the START-of-round
    broadcast = previous round's aggregated output, per the verified anchor convention), run
    e_personalized local epochs of  loss_fn(v(x), y) + proximal_fn(v, anchor, lam), and CARRY
    v FORWARD (never reset). This is exactly the interleaved personal update with the global
    pass replaced by a cache read.

    Args:
      v_model/v_optimizer : the personal model + its optimizer (carried across rounds).
      anchor_trajectory   : list of state_dicts, anchor_trajectory[t] = the global w to
                            regularize toward in round t. With anchor_is_start_of_round=True,
                            this must be [w_init, w^0, w^1, ...] so round t anchors to w^{t-1}
                            (round 0 -> pretrained init). The caller builds this mapping.
      loss_fn(outputs, targets) -> scalar ; proximal_fn(model, ref_state, lam) -> scalar.
      batch_iter_factory()  -> fresh iterator of (inputs, targets, meta) for the train split.

    Returns the per-round list of (val-evaluable) v states is the caller's job; this mutates
    v_model in place and returns it. Deterministic given seeded data + fixed init.
    """
    assert len(anchor_trajectory) >= n_rounds, "anchor trajectory shorter than n_rounds"
    v_model.train()
    for t in range(n_rounds):
        ref = anchor_trajectory[t]   # caller already applied the start-of-round (w^{t-1}) mapping
        for _ in range(e_personalized):
            for inputs, targets, _meta in batch_iter_factory():
                inputs = inputs.to(device)
                targets = targets.to(device)
                v_optimizer.zero_grad(set_to_none=True)
                outputs = v_model(inputs)
                loss = loss_fn(outputs, targets) + proximal_fn(v_model, ref, lam)
                # Stability guard (no effect on finite losses, so the bit-identical equivalence test
                # is unchanged): skip a non-finite step so an extreme lambda can't poison the
                # carried-forward v for the rest of the replay. Mirrors the interleaved personal pass.
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                v_optimizer.step()
    return v_model


VALID_SELECTION_METRICS = ("worst_site", "mean_site", "pooled")


def site_selection_metrics(per_site):
    """Aggregate per-site validation AUCs into the three candidate selection metrics.

    per_site: dict {site_id: {"auc": float, "n_pos": int, "n": int}} or {site_id: auc}.
    Returns {"pooled": ..., "mean_site": ..., "worst_site": ..., "per_site": {...}}.

      - worst_site : min over per-site AUCs  -> the equity target (PacificFed thesis); the
                     underserved site must improve. Noisiest (min of small-n CIs).
      - mean_site  : unweighted mean of per-site AUCs (conventional; can hide disparity).
      - pooled     : n-weighted mean of per-site AUCs as a proxy for a single pooled-set AUC
                     (ignores site structure). NOTE: a true pooled AUC needs the concatenated
                     scores+labels; when only per-site AUCs are available this is the standard
                     sample-size-weighted approximation -- pass concatenated arrays to
                     classification_metrics for the exact pooled AUC.

    ALL THREE are always returned/logged; selection uses ONE pre-committed metric (default
    worst_site) chosen by the caller. Sites with NaN AUC (degenerate val split) are excluded
    from the aggregates (and noted in per_site).
    """
    import numpy as _np
    aucs, weights = [], []
    per = {}
    for sid, v in per_site.items():
        if isinstance(v, dict):
            auc = float(v.get("auc", float("nan")))
            w = float(v.get("n", v.get("total_samples", 1)) or 1)
        else:
            auc, w = float(v), 1.0
        per[sid] = auc
        if not _np.isnan(auc):
            aucs.append(auc); weights.append(w)
    if not aucs:
        return {"pooled": float("nan"), "mean_site": float("nan"),
                "worst_site": float("nan"), "per_site": per}
    aucs = _np.asarray(aucs); weights = _np.asarray(weights)
    return {
        "pooled": float((aucs * weights).sum() / weights.sum()),
        "mean_site": float(aucs.mean()),
        "worst_site": float(aucs.min()),
        "per_site": per,
    }
