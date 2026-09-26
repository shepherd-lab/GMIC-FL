"""Shared GMIC training utilities reused by local and federated pipelines.

This module centralizes model construction, optimizer configuration, freezing,
evaluation, and random-search helpers so both `local_train_gmic.py` and
federated executors can import without duplicating logic.

Safe to import: no argparse side effects or heavy I/O at import time.
"""
from __future__ import annotations

import os
import random
import logging
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model.gmic import GMIC

logger = logging.getLogger(__name__)

from constants.constants import PERCENT_T_DICT

__all__ = [
    "build_gmic_from_args",
    "load_state_dict_forgiving",
    "load_pretrained_if_requested",
    "configure_optimizers",
    "apply_freezing_plan",
    "evaluate_model",
    "set_seed",
    "set_requires_grad",
    "EarlyStopper",
    "summarize_parameter_devices",
    "first_batch_input_device_str",
]


def build_gmic_from_args(args):
    """
    Construct DataParallel-compatible GMIC strictly from CLI args.
    """
    params = {}

    # IO / model-related knobs from CLI
    params["image_path"] = args.image_path
    params["num_classes"] = args.num_classes
    params["use_v1_global"] = bool(args.use_v1_global)

    # Optional structured params only if provided
    if hasattr(args, "cam_h") and hasattr(args, "cam_w"):
        params["cam_size"] = (args.cam_h, args.cam_w)

    if hasattr(args, "K"):
        params["K"] = args.K

    if hasattr(args, "post_dim"):
        params["post_processing_dim"] = args.post_dim

    if hasattr(args, "percent_t"):
        if args.percent_t not in PERCENT_T_DICT:
            raise ValueError(
                f"percent_t '{args.percent_t}' not in PERCENT_T_DICT keys {list(PERCENT_T_DICT.keys())}"
            )
        params["percent_t"] = PERCENT_T_DICT[args.percent_t]

    if hasattr(args, "lambda_l1"):
        params["lambda_l1"] = args.lambda_l1

    if hasattr(args, "crop_h") and hasattr(args, "crop_w"):
        params["crop_shape"] = (args.crop_h, args.crop_w)

    # Construct with DataParallel-compatible version
    return GMIC(params)


def load_state_dict_forgiving(model: nn.Module, ckpt_path: str, device="cpu", log_fn=None):
    """
    Load a state_dict with strict=True first, then retry strict=False.
    Avoids print(); uses log_fn if provided, else uses module logger.
    """
    sd = torch.load(ckpt_path, map_location=device)

    def _log(msg: str, level: int = logging.INFO):
        if log_fn is not None:
            try:
                log_fn(msg)
                return
            except Exception:
                pass
        try:
            logger.log(level, msg)
        except Exception:
            pass

    try:
        model.load_state_dict(sd, strict=True)
        _log("Loaded checkpoint with strict=True")
    except RuntimeError:
        _log("Strict load failed; retrying with strict=False", logging.WARNING)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            _log("[load] Missing keys: " + ", ".join(missing), logging.WARNING)
        if unexpected:
            _log("[load] Unexpected keys: " + ", ".join(unexpected), logging.WARNING)


def load_pretrained_if_requested(model: nn.Module, args, log_fn=None):
    """
    Load checkpoint or pretrained model if requested.
    Avoids print(); uses log_fn if provided, else uses module logger.
    """
    def _log(msg: str, level: int = logging.INFO):
        if log_fn is not None:
            try:
                log_fn(msg)
                return
            except Exception:
                pass
        try:
            logger.log(level, msg)
        except Exception:
            pass

    if getattr(args, "load_checkpoint", None) and os.path.isfile(args.load_checkpoint):
        _log(f"Loading checkpoint: {args.load_checkpoint}")
        load_state_dict_forgiving(
            model,
            args.load_checkpoint,
            device=getattr(args, "device", "cpu"),
            log_fn=log_fn,
        )
        return

    if hasattr(model, "load_pretrained_by_index") and getattr(args, "pretrained_model_index", None):
        _log(f"Loading pretrained model by index: {args.pretrained_model_index}")
        model.load_pretrained_by_index(args.pretrained_model_index)


def _collect_params(obj) -> Iterable[nn.Parameter]:
    params = []
    if obj is None:
        return params
    if isinstance(obj, nn.Module):
        params.extend([p for p in obj.parameters() if p.requires_grad])
    elif isinstance(obj, dict):
        for v in obj.values():
            params.extend(_collect_params(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            params.extend(_collect_params(v))
    return params


def configure_optimizers(model: nn.Module, args):
    """Configure optimizer param groups and scheduler.

    Expects args to have lr_heads, lr_backbone, weight_decay, patience.
    Returns (optimizer, scheduler, early_stopper)
    """
    global_backbone = getattr(model, "global_backbone", None)
    local_backbone = getattr(model, "local_backbone", None)
    head_names = ["global_1x1_and_post", "mil_attention_block", "classifier_heads"]

    head_params = []
    for name in head_names:
        m = getattr(model, name, None)
        ps = _collect_params(m)
        if ps:
            head_params.append({"params": ps, "lr": args.lr_heads})

    backbone_params = []
    backbone_params += _collect_params(global_backbone)
    backbone_params += _collect_params(local_backbone)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})
    param_groups += head_params
    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_heads}]

    # Optimizer selection (args.optimizer_name): "adam" (default, unchanged) or "adamw".
    # Both keep the per-group LR structure (backbone lr_backbone / heads lr_heads) and read
    # weight_decay; betas stay at torch defaults (0.9, 0.999). AdamW decouples weight decay so
    # it actually regularizes (Adam's L2-WD interacts poorly with adaptive per-param scaling).
    opt_name = (getattr(args, "optimizer_name", "adam") or "adam").lower()
    wd = getattr(args, "weight_decay", 1e-5)
    if opt_name == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = optim.Adam(param_groups, weight_decay=wd)
    else:
        logger.warning("[optim] unknown optimizer '%s'; using Adam", opt_name)
        optimizer = optim.Adam(param_groups, weight_decay=wd)

    # Scheduler selection (args.lr_scheduler):
    #   None / "" / "plateau" -> ReduceLROnPlateau(mode=max on val_auc). None keeps the
    #       historical default so current behavior is unchanged when the option is unset.
    #   "cosine" -> CosineAnnealingLR over args.cosine_t_max (default = epochs).
    #   "none" (explicit string) -> no scheduler (returns None).
    # scheduler.step() in the train loop is wrapped so a cosine sched (steps w/o metric) and
    # a plateau sched (steps w/ val_auc) and None all work.
    sched_name = getattr(args, "lr_scheduler", None)
    sched_name = (str(sched_name).lower() if sched_name is not None else None)
    sp = int(getattr(args, "scheduler_patience", 2))
    min_lr = float(getattr(args, "min_lr", 0.0))
    if sched_name in (None, "", "plateau"):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=sp, min_lr=min_lr
        )
    elif sched_name == "cosine":
        t_max = int(getattr(args, "cosine_t_max", 0)) or int(getattr(args, "epochs", 40))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    elif sched_name == "none":
        scheduler = None
    else:
        logger.warning("[optim] unknown lr_scheduler '%s'; using ReduceLROnPlateau", sched_name)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=sp, min_lr=min_lr
        )
    early_stopper = EarlyStopper(patience=getattr(args, "patience", 4))
    return optimizer, scheduler, early_stopper


def apply_freezing_plan(model: nn.Module, args):
    if getattr(args, "freeze_all_backbones", False):
        backbones = getattr(model, "backbones", None) or getattr(model, "gb", None)
        if backbones is not None:
            set_requires_grad(backbones, False)
        else:
            for name, p in model.named_parameters():
                if "backbone" in name or "encoder" in name:
                    p.requires_grad = False
    if getattr(args, "freeze_transformer", False):
        if hasattr(model, "transformer"):
            set_requires_grad(model.transformer, False)
        else:
            for name, p in model.named_parameters():
                if "transformer" in name:
                    p.requires_grad = False


def evaluate_model(model: nn.Module, data_loader, criterion, device, split="val",
                   decision_threshold: float = 0.5, eval_threshold_sweep=None):
    """Evaluate model and return BREAST-LEVEL metrics dict (GMIC's reported unit).

    Classification metrics (AUC, sensitivity, specificity, precision, accuracy) are computed at
    the BREAST level: per-image malignant probabilities are aggregated to a breast = (exam_id,
    laterality), prediction = mean of that breast's views, label = max over its views. This
    matches the paper ("breast-level predictions = average of the two image-level predictions")
    and means the logged/selected AUC is the reportable number, not an image-level proxy.

    Loss stays PER-IMAGE (a training quantity computed per forward sample; reported as mean
    per-image loss, not breast-averaged). Only the classification metrics move to breast level.

    AUC is threshold-independent. acc/sens/spec/precision at `decision_threshold`. The "sweep"
    list (default thresholds [.3..,.7]) is also breast-level. n_pos/total_samples in the result
    are BREAST counts (n_breasts).
    """
    if eval_threshold_sweep is None:
        eval_threshold_sweep = [0.3, 0.4, 0.5, 0.6, 0.7]
    model.eval()
    img_probs, img_targets, img_exam, img_view = [], [], [], []
    total_loss, num_batches = 0.0, 0
    with torch.no_grad():
        for inputs, targets, metadata in data_loader.get_batch_iterator(split):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out = model(inputs)
            # GMIC forward returns (fusion_logits, y_global, y_local, saliency);
            # tolerate a bare tensor (legacy). Evaluate the MALIGNANT head (index 1).
            fusion_logits = out[0] if isinstance(out, (tuple, list)) else out
            t = targets.to(dtype=fusion_logits.dtype).view(-1)
            loss = F.binary_cross_entropy_with_logits(fusion_logits[:, 1], t)  # per-image loss
            total_loss += loss.item()
            num_batches += 1
            probs = torch.sigmoid(fusion_logits[:, 1]).cpu().numpy()
            img_probs.extend(probs)
            img_targets.extend(targets.cpu().numpy())
            # metadata carries per-image (exam_id, view) for breast aggregation
            img_exam.extend(md.get("exam_id") for md in metadata)
            img_view.extend(md.get("view") for md in metadata)
    avg_loss = total_loss / max(num_batches, 1)  # per-image mean loss (unchanged unit)
    thr = float(decision_threshold)

    from fl_utils import classification_metrics, breast_aggregate
    if img_probs:
        # Aggregate per-image -> breast level BEFORE any classification metric.
        breast_probs, breast_labels = breast_aggregate(img_exam, img_view, img_probs, img_targets)
    else:
        breast_probs, breast_labels = np.array([]), np.array([])

    m = classification_metrics(breast_labels, breast_probs, threshold=thr)  # breast-level
    m["loss"] = avg_loss
    m["n_images"] = len(img_probs)  # keep image count for visibility (loss denominator)
    # total_samples / n_pos in m are now BREAST counts (from classification_metrics).
    n_neg = int((breast_labels == 0).sum()) if len(breast_labels) else 0
    m["n_neg"] = n_neg
    sweep = []
    for s_thr in eval_threshold_sweep:
        sm = classification_metrics(breast_labels, breast_probs, threshold=s_thr)
        sm["n_flagged"] = int((breast_probs > float(s_thr)).sum()) if len(breast_probs) else 0
        sweep.append({k: sm[k] for k in
                      ("threshold", "sensitivity", "specificity", "precision", "n_flagged")})
    m["sweep"] = sweep
    return m


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_requires_grad(obj, requires_grad: bool):
    if isinstance(obj, dict):
        for v in obj.values():
            set_requires_grad(v, requires_grad)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            set_requires_grad(v, requires_grad)
        return
    if hasattr(obj, "parameters") and callable(getattr(obj, "parameters")):
        for p in obj.parameters():
            p.requires_grad = requires_grad
        return
    try:
        for p in obj:
            p.requires_grad = requires_grad
        return
    except TypeError:
        raise TypeError(f"Unsupported type for set_requires_grad: {type(obj)}")


class EarlyStopper:
    def __init__(self, patience=4):
        self.best = -float("inf")
        self.count = 0
        self.patience = patience
        self.best_state = None

    def reset(self):
        """Clear best/count/best_state (call at the start of each federated round so
        best-val selection is within-round and never crosses rounds)."""
        self.best = -float("inf")
        self.count = 0
        self.best_state = None

    def step(self, metric, model):
        if metric > self.best:
            self.best = metric
            self.count = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        self.count += 1
        return False

    def should_stop(self):
        return self.count >= self.patience


#############################
# Multi-GPU Diagnostics
#############################

def summarize_parameter_devices(model: nn.Module, max_examples: int = 12) -> dict:
    """Return a summary of where model parameters live (device distribution).

    Args:
        model: torch model (possibly DataParallel)
        max_examples: how many example param entries to include
    Returns:
        dict with keys: device_counts (dict), total_params (int), unique_devices (list), examples (list)
    """
    if isinstance(model, torch.nn.DataParallel):  # unwrap for inspection
        model = model.module
    device_counts: dict[str, int] = {}
    total_params = 0
    examples = []
    for name, p in model.named_parameters():
        dev = str(p.device)
        n = p.numel()
        device_counts[dev] = device_counts.get(dev, 0) + n
        total_params += n
        if len(examples) < max_examples:
            examples.append(f"{name} -> {dev} ({n})")
    summary = {
        "device_counts": device_counts,
        "total_params": total_params,
        "unique_devices": sorted(device_counts.keys()),
        "examples": examples,
    }
    return summary


def first_batch_input_device_str(batch_inputs) -> str:
    """Produce a compact string describing input tensor devices (handles tuple/list)."""
    devices = set()

    def _collect(x):
        if torch.is_tensor(x):
            devices.add(str(x.device))
        elif isinstance(x, (list, tuple)):
            for y in x:
                _collect(y)

    _collect(batch_inputs)
    return ",".join(sorted(devices)) if devices else "<no-tensor>"