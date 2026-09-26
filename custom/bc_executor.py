# ============================================================================
# bc_executor.py - NVFLARE Executor with flexible CSV/PKL input
#
# Part of a federated-learning extension of GMIC (https://github.com/nyukat/GMIC).
# Original GMIC: Copyright (C) 2020 the GMIC authors (NYU), licensed under GNU AGPLv3.
# Federated-learning additions: Copyright (C) 2026 Shepherd Research Lab,
# University of Hawaiʻi Cancer Center. Added 2026; likewise licensed under GNU
# AGPLv3. See LICENSE and NOTICE.
# ============================================================================

import os

# NOTE: we intentionally do NOT force PYTORCH_CUDA_ALLOC_CONF=expandable_segments here.
# It was tried as a fragmentation fix, but "device not ready" driver faults on RSNA-AU's A4000
# (driver 595.97) correlated exactly with expandable_segments being ON -- the only config that
# ever ran clean on that card (the original mw run, to round 20) had it OFF. expandable_segments
# uses the driver's virtual-memory API and is flaky on some GPU/driver combos; leave it off.
# With per-site precision, no site needs it: RSNA-AU runs fp32 (fits its 16 GiB natively) and
# RSNA/HIPIMR run fp16 (ample headroom). A container env var can still opt in if ever needed.
import csv
import copy
import json
import math
import time
import logging
import threading
import contextlib

import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from nvflare.app_common.utils.fl_model_utils import FLModel, FLModelUtils, ParamsType
from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from train.training_core import first_batch_input_device_str, summarize_parameter_devices
from data_loader.data_loader import GMICDataLoader  # This should be the final data loader class
from constants.constants import PERCENT_T_DICT
from train.training_core import (
    build_gmic_from_args,
    configure_optimizers,
    apply_freezing_plan,
    evaluate_model,
    set_seed,
)
from fl_utils import (
    bn_state_keys,
    load_global_into,
    proximal_penalty,
    clone_detached_state,
    tolerant_load_pretrained,
    gmic_malignant_loss,
    malignant_score,
    gmic_outputs,
)

def _git_hash():
    """Short git commit of the running code (best-effort; 'unknown' if unavailable).

    Echoed into [run-config] so the log records exactly what code ran. Resolution order,
    designed so DEPLOYED NVFLARE jobs are attributable even though NVFLARE copies the
    custom/ dir WITHOUT its .git (which is why `git` alone logged 'unknown'):
      1. env var GMIC_GIT_COMMIT (a launcher / run script may set it), then
      2. a GIT_COMMIT file next to this module -- stamp it before submitting with
         `git rev-parse --short HEAD > app/custom/GIT_COMMIT`; it then travels with the job
         to every client, then
      3. `git rev-parse` in the source tree (dev / simulator-from-clone), then
      4. 'unknown'.
    """
    env = os.environ.get("GMIC_GIT_COMMIT")
    if env and env.strip():
        return env.strip()
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        stamp = os.path.join(here, "GIT_COMMIT")
        if os.path.isfile(stamp):
            with open(stamp) as f:
                val = f.read().strip()
            if val:
                return val
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# Allowed federated methods (selected via the client job config "method" arg).
VALID_METHODS = ("local", "fedavg", "fedprox", "fedbn", "ditto", "ditto_modulewise")

# The task loss is the native GMIC deep-supervised 3-head malignant BCE + L1
# (BCEWithLogits(fusion) + BCE(global) + BCE(local) + lambda_l1*L1), computed in
# _compute_task_loss. "gmic" is the only supported value.
VALID_LOSSES = ("gmic",)


class GMICFederatedExecutor(Executor):
    """NVFLARE Executor mirroring local_train_gmic features (preprocessing, freezing, grouped LR, logging)."""

    def __init__(
            self,
            # Defaults reflect the recipe used in the paper's runs (1 epoch/round, backbone-
            # trained). A job config overrides any of these; see the per-method configs.
            epochs: int = 1,                      # FL: 1 local epoch per round
            patience: int = 6,
            lr_heads: float = 3e-5,
            lr_backbone: float = 3e-7,
            weight_decay: float = 1e-6,
            batch_size: int = 32,
            data_path: str | None = None,         # single-site/centralized CSV; else use data_path_map
            data_path_map: dict | None = None,    # {client_identity: csv_path}; per-client CSV from one config
            image_path: str = "/workspace/data",
            model_path: str = "/workspace/models/sample_model_5.p",
            device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
            input_format: str = "csv",
            enable_preprocessing: bool = True,
            force_preprocessing: bool = False,    # re-crop only when explicitly requested
            cache_validation: bool = True,
            use_predefined_splits: bool = True,
            val_split: float = 0.15,
            test_split: float = 0.15,
            random_seed: int = 42,
            output_dir: str = "/workspace/data/processed",
            preprocess_cache_dir: str | None = None,  # stable crop/cache dir, independent of run output_dir
            preprocess_cache_dir_map: dict | None = None,  # {identity: cache_dir}; per-client cache (simulator)
            results_dir: str = "/workspace/gmic_results_centralized",
            out_dir: str | None = None,
            num_processes: int = 4,
            log_file: str = "/workspace/outputs_centralized/executor.log",
            file_integrity_check: bool = True,
            fail_on_integrity_error: bool = False,
            # Freezing options
            freeze_all_backbones: bool = False,
            unfreeze_global_last: bool = False,
            unfreeze_local_last: bool = False,
            # Percent_t key
            percent_t_key: str = "1",
            percent_t: str | None = None,
            # ROI / architecture params
            K: int = 6,
            cam_h: int = 46,
            cam_w: int = 30,
            crop_h: int = 256,
            crop_w: int = 256,
            post_dim: int = 256,
            num_classes: int = 2,
            use_v1_global: bool = False,
            lambda_l1: float = 1e-5,
            # Optional pretrained / checkpoint
            pretrained_model_index: str = "ensemble",
            load_checkpoint: str = "",
            # Logging
            train_log_batch_interval: int = 5,
            # ---------------------------------------------------------------------------------
            # CONSTRAINED-GPU / DIAGNOSTIC TOGGLES (all optional; every default below reproduces
            # stock behavior). These were added to train a site on a 16 GiB RTX A4000 (client
            # "RSNA-AU") without moving it to an A100. Leave them at their defaults on capable GPUs;
            # a site sets them via its config only when it needs them. See README "Training on a
            # memory-constrained GPU" for the recipe.
            # ---------------------------------------------------------------------------------
            # Background heartbeat (seconds) for long phases: a watchdog THREAD logs where the
            # round is stuck even while the main thread is blocked inside a CUDA kernel or a
            # stalled read -- which is exactly when per-batch logging goes silent. Logs only;
            # never aborts, so a slow-but-healthy site is never killed. 0/None disables (default).
            heartbeat_interval_s: int = 0,
            # Diagnostic: torch.cuda.synchronize() after each stage of the Ditto personal step.
            # CUDA launches are async, so without this the heartbeat's `stage` names the first
            # SYNC POINT the CPU reaches, not the op that actually wedged -- e.g. a stuck backward
            # surfaces as stage=step because scaler.step is the first thing that waits. Costs the
            # CPU/GPU overlap (negligible here: the pass is compute-bound) and is off by default.
            stage_sync: bool = False,
            log_lr_each_epoch: bool = False,
            per_file_logging: bool = False,
            tolerant_missing_metadata: bool = False,
            gpus: str = "",
            debug_devices: bool = False,
            # Multi-GPU & memory optimization
            use_amp: bool = False,
            # Ditto personal-pass precision, independent of use_amp. None => follow use_amp.
            # False => force fp32 personal pass (the recipe that ran 20+ rounds on RSNA-AU's A4000)
            # while the main pass keeps AMP.
            personal_amp: bool | None = None,
            # PER-SITE override map, e.g. {"RSNA-AU": false}. A site listed here uses that precision for
            # its personal pass; sites not listed fall back to personal_amp/use_amp. Resolved from
            # the FL identity at runtime, so ONE config deployed to all sites gives RSNA-AU fp32 (its
            # A4000 faults on the fp16 backward) while RSNA/HIPIMR keep AMP -- no per-site app copies,
            # and no need to move a site to an A100. The sites genuinely conflict: RSNA-AU's fp32
            # footprint fits its 16 GiB card but fp16 faults there; RSNA's fp32 footprint OOMs a
            # small card so it needs AMP -- one global precision cannot satisfy both.
            personal_amp_by_site: dict | None = None,
            # PER-SITE personal-pass micro-batch size, e.g. {"RSNA-AU": 8}. The personal pass processes
            # each loader batch in chunks of this size, accumulating gradients to the full effective
            # batch, so a card too small for the full fp32 batch (RSNA-AU's 16 GiB A4000, which
            # oversubscribes to host RAM and DEADLOCKS on the full batch) stays under its memory
            # ceiling. Only BatchNorm stats change (computed on the smaller chunk); the optimizer
            # still sees the full effective batch. Unlisted sites use batch_size (no chunking).
            personal_batch_size_by_site: dict | None = None,
            # PER-SITE MAIN-pass micro-batch, e.g. {"RSNA-AU": 8}. Same idea as the personal map, for the
            # shared-w training pass. Needed once the backbone UNFREEZES (round freeze_backbone_epochs):
            # autograd then retains all backbone activations for backward, so the full fp16 batch
            # overflows RSNA-AU's 16 GiB A4000 and deadlocks. Chunks accumulate to the full effective
            # batch (only BN sees the chunk); unlisted sites use batch_size (no chunking).
            train_batch_size_by_site: dict | None = None,
            grad_accumulation: int = 1,
            memory_efficient: bool = False,
            pre_train_task_name: str = AppConstants.TASK_GET_WEIGHTS,
            train_task_name: str = AppConstants.TASK_TRAIN,
            submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
            validate_task_name: str = AppConstants.TASK_VALIDATION,
            test_task_name: str = "test",
            exclude_vars=None,
            gmic_parameters: dict | None = None,
            loss: str = "gmic",
            optimizer: dict | None = None,
            task_names: dict | None = None,
            # ---- Federated method selection (single switch across 6 regimes) ----
            method: str = "fedavg",
            fedprox_mu: float = 0.01,
            ditto_lambda: float = 0.1,
            lambda_global: float = 1.0,
            lambda_local: float = 0.1,
            lambda_fusion: float | None = None,   # ties to lambda_local if None
            use_fedbn: bool = False,
            local_epochs: int | None = None,      # alias/override for `epochs`
            pretrained_weights_path: str = "",    # falls back to model_path if empty
            pos_weight: float = 3.0,              # malignant-class weight for GMIC BCE
            # ---- Imbalance / overfitting levers (all default to current/off behavior) ----
            use_balanced_sampler: bool = False,   # view-level class-balanced batch sampling
            lr_scheduler: str | None = None,      # None/"plateau"(default)/"cosine"/"none"
            scheduler_patience: int = 2,          # ReduceLROnPlateau patience
            min_lr: float = 0.0,                  # scheduler floor
            cosine_t_max: int = 0,                # cosine period (0 -> epochs)
            decision_threshold: float = 0.5,      # operating point for acc/sens/spec/precision
            eval_threshold_sweep=None,            # sweep thresholds (default [.3,.4,.5,.6,.7]); on
            freeze_backbone_epochs: int = 0,      # keep backbones frozen for first N epochs
            optimizer_name: str = "adam",         # "adam" (default, unchanged) or "adamw"
            use_augmentation: bool = False,       # train-only conservative augmentation
            augmentation: dict | None = None,     # {horizontal_flip,max_rotation_deg,intensity_jitter}
            cache_global_trajectory: bool = False,  # persist per-round SENT global w^t for Ditto replay
            cache_incoming_global: bool = False,  # per round: save RECEIVED aggregate (all rounds; the Ditto-replay anchor trajectory)
            eval_incoming_global: bool = True,    # also eval val/test + dump preds on each round's aggregate (the per-round shared-global stats; ~6min/round)
            incoming_global_saliency: bool = False,  # also dump per-round saliency .npz in that eval (GBs over many rounds; only for a short run where you want per-round heatmaps)
            resume_from_local_round: int = -1,    # resumed-job round 0: submit cached round-N weights untrained
            resume_ckpt_dir: str | None = None,   # dir holding {client}_gmic_model_round_{N}.pth (default: results_dir)
            salvage_eval_rounds: list | None = None,  # test-only salvage: reconstruct+eval these rounds' globals (no training)
            salvage_ckpt_dir: str | None = None,  # dir holding global_trajectory/ + round ckpts for salvage (default: results_dir)
        ):
            """GMIC Federated Executor (feature parity with local trainer)."""
            super().__init__()

            # Local logger (safe even if stdout is broken)
            self._logger = logging.getLogger(self.__class__.__name__)

            # Allow legacy alias 'out_dir' via kwargs fallback (NVFLARE may pass unexpected)
            # Results directory (allow alias out_dir)
            chosen_results = out_dir if out_dir else results_dir
            self.results_dir = chosen_results or "/workspace/gmic_results_centralized"
            self.epochs = epochs
            self.lr_heads = lr_heads
            self.lr_backbone = lr_backbone
            self.weight_decay = weight_decay
            self.batch_size = batch_size
            self.data_path = data_path
            self.data_path_map = data_path_map or {}
            self._identity = None  # set in handle_event(START_RUN); used to resolve data_path_map
            self.image_path = image_path
            self.model_path = model_path
            self.device = device
            self.input_format = input_format
            self.enable_preprocessing = enable_preprocessing
            self.cache_validation = cache_validation
            self.force_preprocessing = force_preprocessing
            self.use_predefined_splits = use_predefined_splits
            self.val_split = val_split
            self.test_split = test_split
            self.random_seed = random_seed
            self.output_dir = output_dir
            # Preprocessing crops/cache live here, decoupled from the run output_dir so a new
            # run (new output_dir) reuses existing crops without copying. Default: output_dir.
            self.preprocess_cache_dir = preprocess_cache_dir or output_dir
            self.preprocess_cache_dir_map = preprocess_cache_dir_map or {}
            self.log_file = log_file
            self.file_integrity_check = file_integrity_check
            self.fail_on_integrity_error = fail_on_integrity_error
            self.num_processes = num_processes
            self.freeze_all_backbones = freeze_all_backbones
            self.unfreeze_global_last = unfreeze_global_last
            self.unfreeze_local_last = unfreeze_local_last
            # Allow legacy percent_t alias if provided
            self.percent_t_key = percent_t if percent_t is not None else percent_t_key
            self.K = K
            self.cam_h = cam_h
            self.cam_w = cam_w
            self.crop_h = crop_h
            self.crop_w = crop_w
            self.post_dim = post_dim
            self.num_classes = num_classes
            self.use_v1_global = use_v1_global
            self.lambda_l1 = lambda_l1
            self.pretrained_model_index = pretrained_model_index
            self.load_checkpoint = load_checkpoint
            self.train_log_batch_interval = train_log_batch_interval
            self.heartbeat_interval_s = int(heartbeat_interval_s or 0)
            self.stage_sync = bool(stage_sync)
            self.log_lr_each_epoch = log_lr_each_epoch
            self.per_file_logging = per_file_logging
            self.tolerant_missing_metadata = tolerant_missing_metadata
            self.gpus = gpus
            self.debug_devices = debug_devices
            self.use_amp = use_amp
            # Personal-pass precision, DECOUPLED from the main pass. The GLOBAL default (None =>
            # follow use_amp) plus a per-site override map. The effective value for THIS site is
            # resolved lazily once the FL identity is known (_resolve_personal_amp), so a single
            # config can run RSNA-AU fp32 while RSNA/HIPIMR run AMP. _personal_amp holds the provisional
            # default until then.
            self._personal_amp_default = self.use_amp if personal_amp is None else bool(personal_amp)
            self._personal_amp_by_site = dict(personal_amp_by_site or {})
            self._personal_amp = self._personal_amp_default
            self._personal_amp_resolved = False
            self._personal_batch_size_by_site = {
                str(k): int(v) for k, v in (personal_batch_size_by_site or {}).items()}
            self._train_batch_size_by_site = {
                str(k): int(v) for k, v in (train_batch_size_by_site or {}).items()}
            self.grad_accumulation = max(1, grad_accumulation)
            self.memory_efficient = memory_efficient
            self._scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if torch.cuda.is_available() else None
            # Separate scaler for the Ditto personal pass: v has its OWN optimizer, and a scaler's
            # scale factor / inf-tracking is per-optimizer-step, so sharing one across both passes
            # would let a skipped step in one pass perturb the other's scale schedule. Built on the
            # provisional PERSONAL amp setting and rebuilt if the per-site resolution differs.
            self._v_scaler = torch.cuda.amp.GradScaler(enabled=self._personal_amp) if torch.cuda.is_available() else None
            self.pre_train_task_name = pre_train_task_name
            self.train_task_name = train_task_name
            self.submit_model_task_name = submit_model_task_name
            self.validate_task_name = validate_task_name
            self.test_task_name = test_task_name
            self.exclude_vars = exclude_vars
            self.model = None
            self.optimizer = None
            self.criterion = None
            self.data_loader = None
            self.persistence_manager = None
            self.best_metrics = {"val_auc": 0.0, "test_auc": 0.0}
            self.training_history = []
            self.validation_history = []
            self.test_history = []
            # Per-round metrics for the RECEIVED global model (server aggregate), evaluated
            # on val+test BEFORE local training. Lets the best global be picked offline by
            # val AUC with test already on hand. Populated when cache_incoming_global=True.
            self.incoming_global_history = []
            # Per-round metrics for the DEPLOYED PERSONAL model (Ditto family), val+test, mirroring
            # incoming_global_history; saved into final_results.json for the same per-round table.
            self.personal_perround_history = []
            self.gmic_parameters_cfg = gmic_parameters or {}
            self.loss_name = (loss or "gmic").lower()
            if self.loss_name not in VALID_LOSSES:
                raise ValueError(
                    f"Unknown loss '{loss}'. Only {VALID_LOSSES} is supported "
                    f"(the native GMIC deep-supervised malignant BCE + L1 loss).")
            self.optimizer_cfg = optimizer or {"name": "adam", "weight_decay": self.weight_decay}
            self.task_names = task_names or {}
            self.pre_train_task_name = self.task_names.get("pre_train", self.pre_train_task_name)
            self.train_task_name = self.task_names.get("train", self.train_task_name)
            self.validate_task_name = self.task_names.get("validate", self.validate_task_name)
            self.test_task_name = self.task_names.get("test", self.test_task_name)
            self.submit_model_task_name = self.task_names.get("submit_model", self.submit_model_task_name)
            self.patience = patience

            # ---- Federated method configuration ----
            self.method = (method or "fedavg").lower()
            if self.method not in VALID_METHODS:
                self._logger.warning(
                    "[EXEC][WARN] Unknown method '%s'; falling back to 'fedavg'. Valid: %s",
                    self.method, VALID_METHODS,
                )
                self.method = "fedavg"
            self.fedprox_mu = float(fedprox_mu)
            self.ditto_lambda = float(ditto_lambda)
            self.lambda_global = float(lambda_global)
            self.lambda_local = float(lambda_local)
            self.lambda_fusion = float(lambda_fusion) if lambda_fusion is not None else float(lambda_local)
            self.lam_dict = {
                "global": self.lambda_global,
                "local": self.lambda_local,
                "fusion": self.lambda_fusion,
            }
            self.use_fedbn = bool(use_fedbn)
            self.pos_weight = float(pos_weight)
            # ---- Imbalance / overfitting levers ----
            self.use_balanced_sampler = bool(use_balanced_sampler)
            self.lr_scheduler_name = lr_scheduler
            self.scheduler_patience = int(scheduler_patience)
            self.min_lr = float(min_lr)
            self.cosine_t_max = int(cosine_t_max)
            self.decision_threshold = float(decision_threshold)
            self.eval_threshold_sweep = (list(eval_threshold_sweep)
                                         if eval_threshold_sweep else [0.3, 0.4, 0.5, 0.6, 0.7])
            self.freeze_backbone_epochs = int(freeze_backbone_epochs)
            self.optimizer_name = (optimizer_name or "adam").lower()
            self.use_augmentation = bool(use_augmentation)
            self.augmentation_cfg = augmentation or {}
            self.cache_global_trajectory = bool(cache_global_trajectory)
            self.cache_incoming_global = bool(cache_incoming_global)
            self.eval_incoming_global = bool(eval_incoming_global)
            self.incoming_global_saliency = bool(incoming_global_saliency)
            self.resume_from_local_round = int(resume_from_local_round)
            self.resume_ckpt_dir = resume_ckpt_dir
            # LOGICAL-round offset for resumed runs: NVFlare restarts current_round at 0 on a resume,
            # which would overwrite/misnumber the per-round caches (global_trajectory, incoming_global,
            # local ckpts) that the Ditto replay relies on. Resuming from round N (reseed at round 0,
            # first new training at round 1 == logical N+1) -> stamp artifacts at current_round + N so
            # the trajectory stays contiguous across however many crash-resume segments it takes.
            # 0 for a fresh run (resume_from_local_round=-1), so normal runs are byte-for-byte unchanged.
            self._round_offset = self.resume_from_local_round if self.resume_from_local_round >= 0 else 0
            self.salvage_eval_rounds = [int(r) for r in salvage_eval_rounds] if salvage_eval_rounds else []
            self.salvage_ckpt_dir = salvage_ckpt_dir
            # Guardrail: balanced sampler + a large pos_weight double-counts the imbalance
            # correction (sampler evens the batch mix AND the loss up-weights positives).
            if self.use_balanced_sampler and self.pos_weight > 2.0:
                self._logger.warning(
                    "[EXEC][WARN] use_balanced_sampler=True AND pos_weight=%.2f (>2.0): these "
                    "STACK the imbalance correction and may over-correct toward positives. "
                    "Consider pos_weight~1.0 when the balanced sampler is on.", self.pos_weight
                )
            self.pretrained_weights_path = pretrained_weights_path or ""
            if local_epochs is not None:
                self.epochs = int(local_epochs)
            # Ditto personal model `v` and its optimizer persist across rounds.
            self._v_model = None
            self._v_optimizer = None
            self._global_ref = None
            self._prox_ref = None
            self._prox_lambda = None
            self._opt_args = None
            self._pretrained_report = None

            # Use output_dir as the canonical results_dir for model/metrics artifacts
            self.results_dir = self.output_dir


    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle NVFLARE events"""
        if event_type == EventType.START_RUN:
            # Capture this client's FL identity (HIPIMR/RSNA-AU/RSNA-US) BEFORE initialize(),
            # which has no fl_ctx, so initialize() can resolve a per-client data_path_map.
            try:
                self._identity = fl_ctx.get_identity_name()
            except Exception:
                self._identity = None
            self.initialize()
        elif event_type == EventType.END_RUN:
            self._save_final_results(fl_ctx)


    def initialize(self):
        """Initialize data loader, model, optimizer, and bookkeeping.
        Backward-compatible: will use JSON-driven overrides if present, otherwise defaults.
        """
        # If this executor instance was recreated for a later round/end_run, recover persisted best metrics
        try:
            self._recover_persisted_best_metrics()
        except Exception as e:
            self._logger.warning("[EXEC][WARN] Failed to recover persisted best metrics: %s", e, exc_info=True)

        # 0a. Per-client path resolution for SIMULATOR runs (all clients share one filesystem):
        #   - {site} in any path config -> this client's FL identity, so output/cache/tb/log
        #     stay separate per client (e.g. output_dir="/ws/sim/ditto/l0.1/{site}").
        #   - preprocess_cache_dir_map {identity: dir} -> per-client crop cache (the unprefixed
        #     crops would otherwise collide). Resolved before the {site} pass so a mapped dir may
        #     itself contain {site}. On real (one-host-per-client) deployments neither is set and
        #     this is a no-op.
        ident = self._identity or ""
        def _sub_site(p):
            return p.replace("{site}", ident) if (isinstance(p, str) and "{site}" in p and ident) else p
        if self.preprocess_cache_dir_map:
            mapped_cache = self.preprocess_cache_dir_map.get(self._identity)
            if mapped_cache:
                self.preprocess_cache_dir = mapped_cache
            else:
                self._logger.warning(
                    "[EXEC] preprocess_cache_dir_map set but identity=%r not a key (keys=%s); "
                    "using preprocess_cache_dir=%s", self._identity,
                    list(self.preprocess_cache_dir_map.keys()), self.preprocess_cache_dir,
                )
        self.output_dir = _sub_site(self.output_dir)
        self.results_dir = _sub_site(self.results_dir)
        self.preprocess_cache_dir = _sub_site(self.preprocess_cache_dir)
        self.log_file = _sub_site(self.log_file)
        if ident and ("{site}" in str(getattr(self, "output_dir", "")) or self.preprocess_cache_dir_map):
            self._logger.info(
                "[EXEC] site paths resolved for %s: output_dir=%s preprocess_cache_dir=%s",
                ident, self.output_dir, self.preprocess_cache_dir,
            )

        # 0. Per-client data path: if a data_path_map is given, each client picks its OWN
        # CSV by FL identity (HIPIMR/RSNA-AU/RSNA-US), so all sites keep distinct filenames from
        # ONE distributed config. No per-client fallback: a map present but missing this
        # identity is a config error (raise) rather than silently loading the wrong CSV.
        # When no map is given at all, self.data_path is used (single-site / centralized jobs).
        if self.data_path_map:
            mapped = self.data_path_map.get(self._identity)
            if not mapped:
                raise ValueError(
                    f"[EXEC] data_path_map is set but client identity {self._identity!r} is not "
                    f"a key (map keys={list(self.data_path_map.keys())}). Add this client to "
                    f"data_path_map -- there is no data_path fallback."
                )
            self._logger.info("[EXEC] data_path_map: identity=%s -> %s", self._identity, mapped)
            self.data_path = mapped

        # 1. Effective data paths
        # Preprocessing crops/cache live in preprocess_cache_dir (decoupled from output_dir).
        set_seed(self.random_seed)
        processed_pkl = os.path.join(self.preprocess_cache_dir, "processed_exam_list.pkl")
        cropped_dir = os.path.join(self.preprocess_cache_dir, "cropped_images")
        
        # **NEW: Force preprocessing if flag is set**
        # Must clear the FULL preprocessing state so the loader rebuilds from scratch.
        # In particular cropped_exam_list.pkl (the Stage-1 exam list) must go too: if it
        # survives while cropped_images/ is deleted, the loader sees "Stage 1 done" and
        # resumes at "Stage2 only" -> center extraction then fails on the missing PNGs and
        # silently trains WITHOUT best-centers (corrupting the local-module ROI patches).
        if self.force_preprocessing:
            self._logger.warning("[EXEC] force_preprocessing=True: Clearing cached preprocessed data")
            import shutil
            stale_files = [
                processed_pkl,
                os.path.join(self.preprocess_cache_dir, "cropped_exam_list.pkl"),         # Stage-1 list
                os.path.join(self.preprocess_cache_dir, "preprocessing_cache_info.json"),  # cache validator
                os.path.join(self.preprocess_cache_dir, "preprocessing_progress.json"),    # resume manifest
            ]
            for f in stale_files:
                if os.path.isfile(f):
                    os.remove(f)
                    self._logger.info("[EXEC] Deleted: %s", f)
            if os.path.isdir(cropped_dir):
                shutil.rmtree(cropped_dir, ignore_errors=True)
                self._logger.info("[EXEC] Deleted: %s", cropped_dir)
        
        # Now check for cache (will be empty if force_preprocessing=True)
        if os.path.isfile(processed_pkl) and os.path.isdir(cropped_dir):
            effective_data_path = processed_pkl
            effective_image_path = cropped_dir
            effective_input_format = "pkl"
            effective_enable_preprocessing = False
            self._logger.info("[EXEC] Using cached processed data: %s", processed_pkl)
        else:
            effective_data_path = self.data_path
            effective_image_path = self.image_path
            effective_input_format = self.input_format
            effective_enable_preprocessing = self.enable_preprocessing
            self._logger.info(
                "[EXEC] Using source data: %s (enable_preprocessing=%s)",
                self.data_path, self.enable_preprocessing
            )

        # 2. Data loader
        self.data_loader = GMICDataLoader(
            data_path=effective_data_path,
            image_path=effective_image_path,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            use_predefined_splits=self.use_predefined_splits,
            val_split=self.val_split,
            test_split=self.test_split,
            input_format=effective_input_format,
            enable_preprocessing=effective_enable_preprocessing,
            output_dir=self.preprocess_cache_dir,  # loader uses this only for crops/cache
            num_processes=self.num_processes,
            cache_validation=self.cache_validation,
            log_file=self.log_file,
            per_file_logging=self.per_file_logging,
            tolerant_missing_metadata=self.tolerant_missing_metadata,
            file_integrity_check=self.file_integrity_check,
            fail_on_integrity_error=self.fail_on_integrity_error,
            use_balanced_sampler=self.use_balanced_sampler,
            use_augmentation=self.use_augmentation,
            augmentation=self.augmentation_cfg,
        )
        self.data_loader.print_summary()

        # 3. Iterations per split
        split_info = self.data_loader.get_split_info()
        self._n_train_iterations = max(1, math.ceil(split_info["train_size"] / max(self.batch_size, 1)))
        self._n_val_iterations = max(1, math.ceil(split_info["val_size"] / max(self.batch_size, 1)))
        self._n_test_iterations = max(1, math.ceil(split_info["test_size"] / max(self.batch_size, 1)))

        # 4. GMIC parameters
        def _to_tuple(x):
            return tuple(x) if isinstance(x, (list,)) else x

        gmic_defaults = {
            "device_type": "gpu" if "cuda" in str(self.device) else "cpu",
            "gpu_number": 0,
            "max_crop_noise": (100, 100),
            "max_crop_size_noise": 100,
            "image_path": effective_image_path,
            "cam_size": (self.cam_h, self.cam_w),
            "K": self.K,
            "crop_shape": (self.crop_h, self.crop_w),
            "post_processing_dim": self.post_dim,
            "num_classes": self.num_classes,
            "use_v1_global": self.use_v1_global,
            "percent_t": PERCENT_T_DICT.get(self.percent_t_key, PERCENT_T_DICT["1"]),
            "lambda_l1": self.lambda_l1,
        }
        overrides = dict(getattr(self, "gmic_parameters_cfg", {}) or {})
        for k in ["max_crop_noise", "cam_size", "crop_shape"]:
            if k in overrides:
                overrides[k] = _to_tuple(overrides[k])
        pt = overrides.get("percent_t")
        if isinstance(pt, str) and pt.startswith("auto:"):
            key = pt.split(":", 1)[1]
            overrides["percent_t"] = PERCENT_T_DICT.get(key, PERCENT_T_DICT["1"])
        self.gmic_parameters = {**gmic_defaults, **overrides}
        # C1: single source of truth for percent_t. The model is built from
        # percent_t_key (-> PERCENT_T_DICT). Fail loud if a gmic_parameters.percent_t
        # override disagrees, so a config sweep can't silently use a different threshold.
        _pt_key_val = PERCENT_T_DICT.get(self.percent_t_key, PERCENT_T_DICT["1"])
        _pt_override = overrides.get("percent_t")
        if isinstance(_pt_override, (int, float)) and abs(float(_pt_override) - float(_pt_key_val)) > 1e-9:
            raise ValueError(
                f"percent_t mismatch: gmic_parameters.percent_t={_pt_override} but "
                f"percent_t_key='{self.percent_t_key}' -> {_pt_key_val}. Use ONE source "
                f"(prefer percent_t_key; remove percent_t from gmic_parameters)."
            )

        # 5. Model build
        class _Args: pass
        a = _Args()
        a.device_type = gmic_defaults["device_type"]
        a.gpu_number = 0
        a.image_path = self.gmic_parameters.get("image_path")
        a.num_classes = self.gmic_parameters.get("num_classes", 2)
        a.cam_h, a.cam_w = self.gmic_parameters.get("cam_size", (self.cam_h, self.cam_w))
        a.K = self.gmic_parameters.get("K", self.K)
        a.post_dim = self.gmic_parameters.get("post_processing_dim", self.post_dim)
        a.percent_t = self.percent_t_key
        a.lambda_l1 = self.gmic_parameters.get("lambda_l1", self.lambda_l1)
        a.crop_h, a.crop_w = self.gmic_parameters.get("crop_shape", (self.crop_h, self.crop_w))
        a.use_v1_global = self.gmic_parameters.get("use_v1_global", self.use_v1_global)

        self.model = build_gmic_from_args(a)
        # Helper: underlying model reference (for DP awareness)
        self._underlying = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        if "cuda" in str(self.device):
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                self._logger.warning(
                    "[EXEC] CUDA '%s' requested but no driver found — falling back to CPU", self.device
                )
                self.device = torch.device("cpu")
        self.model.to(self.device)

        """
        # Optional pretrained load (if no explicit checkpoint provided)
        if (not self.load_checkpoint) and getattr(self, 'pretrained_model_index', None):
            try:
                load_pretrained_if_requested(self.model, type('obj', (), {
                    'pretrained_model_index': self.pretrained_model_index,
                    'model_path': self.model_path,
                    'device': self.device,
                }))
                self._logger.info("[EXEC] Loaded pretrained model index %s", self.pretrained_model_index)
            except Exception as e:
                self._logger.warning("[EXEC][WARN] pretrained load failed: %s", e, exc_info=True)

        if self.load_checkpoint and os.path.isfile(self.load_checkpoint):
            try:
                state = torch.load(self.load_checkpoint, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                self._logger.info("[EXEC] Loaded checkpoint: %s", self.load_checkpoint)
            except Exception as e:
                self._logger.warning("[EXEC][WARN] Failed to load checkpoint %s: %s", self.load_checkpoint, e, exc_info=True)
        """

        # 5b. Always initialize from the pretrained GMIC checkpoint (decision: fine-tune,
        # not train-from-scratch). The released checkpoint keys match this model 1:1
        # except two benign items: missing '_device_ref' (an empty device-tracking buffer,
        # carries no learned info) and unexpected 'shared_rep_filter.weight' (an auxiliary
        # 256x256x4x4 conv head OUTSIDE the classification forward path; the GMIC variant
        # that produced these checkpoints bolted it on after fusion_dnn -- this model's
        # forward never uses it, so it is safely discarded). Backbones + all four output
        # heads load 100%, so a clean load reports matched=258/259.
        # This makes "init from pretrained" robust and self-contained — independent of
        # whether the server persistor seeds round-0 with the same checkpoint — and is
        # what makes the round-0 baseline meaningful and `method=local` start pretrained.
        ckpt = self.pretrained_weights_path or self.model_path
        self._pretrained_report = None
        if ckpt and os.path.isfile(ckpt):
            # R4: raise_on_partial=True and NO broad except -> a partial backbone load
            # (or a corrupt checkpoint) aborts initialize() loudly. A half-loaded model
            # would make the cold-start baseline silently meaningless.
            self._pretrained_report = tolerant_load_pretrained(
                self._underlying, ckpt, device=self.device,
                log_fn=lambda m: self._logger.info(m), raise_on_partial=True,
            )
            self._logger.info("[EXEC] Pretrained init from %s: %s", ckpt, self._pretrained_report)
        else:
            # FAIL LOUD: this executor always fine-tunes from the released GMIC checkpoint.
            # A missing checkpoint previously fell back to RANDOM init and ran silently --
            # which contaminated a federated run when one client's sample_model_5.p was absent
            # (its random-derived update poisoned the aggregate). Refuse to start instead.
            raise FileNotFoundError(
                f"[EXEC] pretrained checkpoint required but not found at {ckpt!r}. This run is "
                f"supposed to init from pretrained weights; refusing to start from random init "
                f"(it silently contaminates the aggregate and the round-0 baseline). Place the "
                f"checkpoint at this path on THIS client, or set pretrained_weights_path/model_path."
            )

        # 6. Freezing
        class _FreezeArgs: pass
        fa = _FreezeArgs()
        fa.freeze_all_backbones = self.freeze_all_backbones
        fa.unfreeze_global_last = self.unfreeze_global_last
        fa.unfreeze_local_last = self.unfreeze_local_last
        apply_freezing_plan(self.model, fa)

        # 7. Optimizer. The GMIC task loss is computed in _compute_task_loss, so there is no
        # nn criterion object; self.criterion stays None (evaluate_model accepts None).

        class _OptArgs: pass
        oa = _OptArgs()
        oa.lr_heads = self.lr_heads
        oa.lr_backbone = self.lr_backbone
        oa.weight_decay = self.weight_decay
        oa.patience = self.patience
        oa.optimizer_name = self.optimizer_name
        # Scheduler selection (None -> historical ReduceLROnPlateau; preserves current behavior)
        oa.lr_scheduler = self.lr_scheduler_name
        oa.scheduler_patience = self.scheduler_patience
        oa.min_lr = self.min_lr
        oa.cosine_t_max = self.cosine_t_max
        oa.epochs = self.epochs
        self.optimizer, self.scheduler, self.early_stopper = configure_optimizers(self.model, oa)
        # Stash optimizer args so the Ditto personal model `v` can build its own optimizer.
        self._opt_args = oa
        # Run-start LR sanity: record the effective per-group LRs the optimizer actually got
        # (heads + backbone are separate groups; a plumbing bug could collapse them to one LR).
        _bb, _hd = self._effective_group_lrs()
        self._logger.info("[EXEC] effective LRs at init: heads=%.3e backbone=%.3e (config lr_heads=%s lr_backbone=%s)",
                          _hd, _bb, self.lr_heads, self.lr_backbone)
        try:
            self._log_gpu_diagnostics()
        except Exception as e:
            self._logger.warning("[gpu-diag] diagnostics failed (non-fatal): %s", e)

        # 8. Multi-GPU
        gpu_ids = []
        if self.gpus:
            try:
                gpu_ids = [int(x) for x in self.gpus.split(',') if x.strip()]
            except ValueError:
                self._logger.warning("[EXEC][WARN] Invalid gpus string: %s", self.gpus)

        if torch.cuda.is_available() and len(gpu_ids) > 0:
            # Ensure primary device matches first GPU id
            self.device = torch.device(f"cuda:{gpu_ids[0]}")
            self.model.to(self.device)
            if len(gpu_ids) > 1:
                self._logger.info("[EXEC] Enabling DataParallel on GPUs: %s", gpu_ids)
                self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)

        # 10. Optional PKL snapshot
        if self.data_loader.input_format == "csv":
            pkl_output_path = os.path.join(os.path.dirname(self.model_path), "converted_data.pkl")
            try:
                self.data_loader.save_pkl_format(pkl_output_path)
            except Exception as e:
                self._logger.warning("[WARN] Failed to save PKL snapshot: %r", e, exc_info=True)

        self._logger.info(
            "Initialized GMIC executor with %s data: %s",
            self.data_loader.input_format.upper(), split_info
        )


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        try:
            # --- round headers ---
            # current_round is the RAW NVFlare counter (restarts at 0 on resume) -> use it for control
            # flow (reseed/baseline at ==0, test at ==total_rounds-1). logical_round adds the resume
            # offset and is what stamps every persisted artifact + metric, so a multi-segment (crash-
            # resumed) run yields one contiguous round numbering. Equal to current_round on a fresh run.
            current_round = int(shareable.get_header(AppConstants.CURRENT_ROUND, 0))
            total_rounds = int(shareable.get_header(AppConstants.NUM_ROUNDS, 1))
            logical_round = current_round + self._round_offset
            self.log_info(fl_ctx, f"Starting round {current_round}/{total_rounds}"
                                  + (f" (logical {logical_round})" if self._round_offset else "")
                                  + " - Train+Validate+Test")

            # Per-run config echo so sweep results can be matched to the intended config from
            # the log. Show only the lambda(s) that are ACTIVE for the selected method.
            if self.method == "ditto_modulewise":
                lam_desc = (f"lambda_global={self.lambda_global} lambda_local={self.lambda_local} "
                            f"lambda_fusion={self.lambda_fusion}")
            elif self.method == "ditto":
                lam_desc = f"ditto_lambda={self.ditto_lambda}"
            elif self.method == "fedprox":
                lam_desc = f"fedprox_mu={self.fedprox_mu}"
            else:
                lam_desc = "(no lambda for this method)"
            self.log_info(
                fl_ctx,
                f"[run-config] method={self.method} use_fedbn={self.use_fedbn} {lam_desc} "
                f"loss={self.loss_name} pos_weight={self.pos_weight} epochs={self.epochs} "
                f"lr_heads={self.lr_heads} lr_backbone={self.lr_backbone} "
                f"balanced_sampler={self.use_balanced_sampler} "
                f"lr_scheduler={self.lr_scheduler_name} decision_threshold={self.decision_threshold} "
                f"eval_threshold_sweep={self.eval_threshold_sweep} "
                f"optimizer={self.optimizer_name} weight_decay={self.weight_decay} "
                f"freeze_backbone_epochs={self.freeze_backbone_epochs} "
                f"use_augmentation={self.use_augmentation} "
                f"seed={self.random_seed} git={_git_hash()}"
                + (f" aug={self.data_loader.aug_cfg}" if (self.use_augmentation and getattr(self, 'data_loader', None) is not None) else "")
            )

            # --- receive global model (support both FLModel and legacy raw weights) ---
            incoming = None
            fl_model_in = FLModelUtils.from_shareable(shareable)
            if fl_model_in and fl_model_in.params:
                incoming = self._ensure_torch_state(fl_model_in.params, fl_ctx)
            elif AppConstants.MODEL_WEIGHTS in shareable:
                incoming = self._ensure_torch_state(shareable[AppConstants.MODEL_WEIGHTS], fl_ctx)

            # --- SALVAGE eval (test-only, no training): reconstruct each configured round's global
            # via the server's weighted aggregation of resubmitted SENT weights, evaluate it +
            # this site's own sent model on val+test, and dump preds. Short-circuits the round. ---
            if self.salvage_eval_rounds:
                try:
                    return self._salvage_round(fl_ctx, shareable, incoming)
                except Exception as e:
                    self.log_exception(fl_ctx, f"[salvage] round failed: {e}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            # --- RESUME re-seed (round 0 only): submit THIS client's cached round-N weights
            # UNTRAINED so the server's weighted aggregation reconstructs the round-N global,
            # then training resumes normally from round 1. Each client uses its OWN local file
            # -- no gathering/copying, and the per-site weight (train_size) is computed here
            # from the freshly preprocessed data, matching the original aggregation weight.
            # The cold global just broadcast at round 0 is intentionally ignored. ---
            if self.resume_from_local_round >= 0 and current_round == 0:
                try:
                    return self._reseed_round(fl_ctx, shareable)
                except Exception as e:
                    self.log_exception(fl_ctx, f"[resume] re-seed round failed: {e}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            # --- round-0 pretrained baseline (BEFORE consuming the global) ---
            # initialize() pretrained-loads the model, so right now `self.model` holds the
            # cold-start weights. Dump its val/test predictions once so every method is
            # comparable to the "no local adaptation at round 0" reference row.
            if current_round == 0:
                try:
                    self._dump_baseline(fl_ctx)
                except Exception as e:
                    self.log_warning(fl_ctx, f"[baseline] dump failed: {e}")

            # --- method-aware consumption of the global model ---
            # local: skip load | fedbn/use_fedbn: skip BN keys | else: full load.
            # Also captures the frozen global ref (FedProx/Ditto) and lazily inits Ditto v.
            self._consume_global(incoming, fl_ctx)

            # --- crash-recovery: persist the RECEIVED server aggregate for this round ---
            # `incoming` is the server's aggregated global from the PRIOR round's training
            # (at round 0 it's the source_ckpt seed). This is byte-for-byte the artifact the
            # server deletes with its transient run dir, so saving the raw incoming dict here
            # gives a clean warm-start file from ANY surviving client. Distinct from
            # cache_global_trajectory (which saves this client's post-training SENT weights).
            # Saved BEFORE training so a mid-train crash still leaves the round's global on disk.
            if getattr(self, "cache_incoming_global", False) and incoming is not None:
                try:
                    self._save_incoming_global(fl_ctx, logical_round, incoming)
                except Exception as e:
                    self.log_warning(fl_ctx, f"[recv-global] save failed for round {logical_round}: {e}")
                # Per-round eval + prediction dump on the received global (self.model right
                # after _consume_global, i.e. the aggregate as this site consumes it). Done
                # here, BEFORE training mutates the weights, so best-global selection can be
                # done offline by val AUC with test metrics + preds already on disk. Heavy over
                # many rounds (a val+test eval + saliency dump each round); skip via
                # eval_incoming_global=False when you only need the saved trajectory (e.g. a long
                # FedAvg run whose purpose is the Ditto-replay anchor).
                if getattr(self, "eval_incoming_global", True):
                    try:
                        self._track_incoming_global(fl_ctx, logical_round)
                    except Exception as e:
                        self.log_warning(fl_ctx, f"[incoming-global] eval failed for round {logical_round}: {e}")

            # --- TRAIN (with early stopping) ---
            self.log_info(fl_ctx, "Phase 1: Training (early stopping enabled)...")
            train_metrics = self._local_train(fl_ctx, abort_signal, shareable)

            # C2: the aggregation contribution is THIS round's FINAL post-epoch local weights
            # (textbook FedAvg/FedProx/FedBN; for Ditto, the global-tracked w). Capture them
            # BEFORE loading any best-val checkpoint, so the sent w is never a stale best-state.
            final_state = {k: v.detach().cpu().clone() for k, v in self._underlying.state_dict().items()}

            # Per-round GLOBAL trajectory cache (Ditto consolidation): persist the round's SENT
            # global w^t (== final_state, the aggregation contribution) under an unambiguous,
            # distinct-from-deployed name so a single FedAvg run can be REPLAYED to train Ditto/
            # module-wise personalized models for any lambda without recomputing FedAvg. This is
            # the start-of-round anchor for round t+1 (anchor convention: round-(t+1) personal
            # update regularizes toward w^t). Gated on cache_global_trajectory (default off).
            if getattr(self, "cache_global_trajectory", False):
                try:
                    self._save_global_round(fl_ctx, logical_round, final_state)
                except Exception as e:
                    self.log_warning(fl_ctx, f"[global-cache] save failed for round {logical_round}: {e}")

            # Deployed/evaluated model = best-val OF THIS ROUND. Load it into the global-tracked
            # model for eval/dumps only (non-Ditto; Ditto deploys v). The SENT w stays = final_state.
            if (self.method not in ("ditto", "ditto_modulewise")
                    and getattr(self, 'early_stopper', None)
                    and getattr(self.early_stopper, 'best_state', None)):
                try:
                    self._underlying.load_state_dict(self.early_stopper.best_state, strict=False)
                    self.log_info(fl_ctx, f"Deployed/eval = best-val-of-round (val_auc={self.early_stopper.best:.4f}); sent w = final-epoch")
                    self._save_best_model(fl_ctx, {"auc": float(self.early_stopper.best)}, shareable, model_type="best_val_round", round_idx=logical_round)
                except Exception as e:
                    self.log_warning(fl_ctx, f"Failed to load best-val weights for eval: {e}")

            # --- Ditto personal pass: train v with task_loss(v) + proximal(v, w_ref) ---
            # (v persists across rounds; only the global-tracked model w is sent/aggregated.)
            if self.method in ("ditto", "ditto_modulewise"):
                self._train_personal_model(fl_ctx, abort_signal)

            # Deployed/evaluated model: personal v for Ditto-family, else the global-tracked model.
            deployed = self._v_model if self.method in ("ditto", "ditto_modulewise") else self.model

            num_examples = int(train_metrics.get("train_samples", self.batch_size))
            if num_examples <= 0:
                # ensure a positive weight so FedAvg doesn’t drop this update
                num_examples = max(1, self.batch_size)

            if not abort_signal.triggered:
                self._save_local_model(fl_ctx, shareable, model=deployed, round_idx=logical_round)

            # --- VAL (on the deployed model: v for Ditto-family, else w) ---
            self.log_info(fl_ctx, "Phase 2: Validation...")
            val_core = self._evaluate_model(fl_ctx, split="val", model=deployed)
            # Prefix validation metrics so server selector finds 'val_auc'
            val_metrics = {f"val_{k}": v for k, v in val_core.items() if k in ("auc", "accuracy", "loss", "samples")}

            # Per-site validation metric for the deployed model (offline ranking / fairness).
            self.log_info(
                fl_ctx,
                f"[per-site-metric] site={fl_ctx.get_identity_name()} round={logical_round} "
                f"method={self.method} deployed_val_auc={val_core.get('auc', float('nan')):.4f}"
            )

            # Update best tracking
            if val_core.get('auc', -1) > self.best_metrics.get('val_auc', -1):
                self.best_metrics['val_auc'] = val_core['auc']
                try:
                    self._save_best_model(fl_ctx, {"auc": val_core['auc']}, shareable, model_type="best_val_overall", model=deployed, round_idx=logical_round)
                except Exception:
                    pass

            # --- TEST (final round only) ---
            if current_round == (total_rounds - 1):
                self.log_info(fl_ctx, "Phase 3: Testing...")
                test_core = self._evaluate_model(fl_ctx, split="test", model=deployed)
                test_metrics = {f"test_{k}": v for k, v in test_core.items() if k in ("auc", "accuracy", "loss", "samples")}
                if test_core.get('auc', -1) > self.best_metrics.get('test_auc', -1):
                    self.best_metrics['test_auc'] = test_core['auc']
                    try:
                        self._save_best_model(fl_ctx, test_core, shareable, model_type="best_test_overall", model=deployed, round_idx=logical_round)
                    except Exception:
                        pass
                # End-of-run prediction dump for offline AUC / DeLong CIs / cross-site sigma(AUC).
                for _split in ("val", "test"):
                    try:
                        self._dump_predictions(fl_ctx, split=_split, model=deployed,
                                               round_idx=logical_round, method_tag=self.method)
                    except Exception as e:
                        self.log_warning(fl_ctx, f"[predictions] dump ({_split}) failed: {e}")
            else:
                self.log_info(fl_ctx, "Phase 3: Skipping test (not final round)")
                test_metrics = {}

            # --- Per-round PERSONALIZED-model trajectory (Ditto family) ---
            # Dump the deployed personal model v's val+test preds EVERY round (tag "<method>_perround",
            # logged [personal-perround]) so v has the same per-round trajectory the shared methods get
            # from incoming_global -- enabling best-val selection + pooled/DeLong/Youden for the PERSONAL
            # model from preds alone, with NO post-hoc per-node dump (each client writes its own live).
            if self.method in ("ditto", "ditto_modulewise") and getattr(self, "eval_incoming_global", True):
                try:
                    self._track_personal_perround(fl_ctx, logical_round, deployed)
                except Exception as e:
                    self.log_warning(fl_ctx, f"[personal-perround] tracking failed: {e}")

            # --- package update for FedAvg (DXO/FLModel) ---
            # C2: send the FINAL-epoch local weights (captured pre-reload), NOT the best-val
            # checkpoint -- standard FedAvg/FedProx/FedBN expect the current round's local update.
            updated_weights = {k: v.detach().cpu().numpy() for k, v in final_state.items()}

            combined_metrics = {**train_metrics, **val_metrics, **test_metrics, "round": int(logical_round)}
            if 'val_auc' not in combined_metrics and 'val_auc' in val_metrics:
                combined_metrics['val_auc'] = val_metrics['val_auc']

            # Safety filter: only include numeric values
            numeric_metrics = {}
            for key, value in combined_metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_metrics[key] = float(value) if not isinstance(value, int) else int(value)
                else:
                    self.log_warning(fl_ctx, f"Skipping non-numeric metric: {key} = {value}")

            # Attach validation metric in meta so IntimeModelSelector can read it
            val_auc_for_meta = numeric_metrics.get('val_auc') or numeric_metrics.get('best_val_auc')
            model_meta = {
                MetaKey.NUM_STEPS_CURRENT_ROUND: num_examples,
                "num_examples": num_examples,
            }
            if val_auc_for_meta is not None and not math.isnan(float(val_auc_for_meta)):
                model_meta['val_auc'] = float(val_auc_for_meta)
            else:
                self.log_warning(fl_ctx, "[metrics-debug] val_auc missing/NaN (degenerate split?); selector may warn")

            fl_model_out = FLModel(
                params=updated_weights,
                params_type=ParamsType.FULL,
                metrics=numeric_metrics,
                meta=model_meta,
            )

            self.log_info(fl_ctx, (
                f"[metrics-debug] meta keys: {list(model_meta.keys())} val_auc_meta={model_meta.get('val_auc')} "
                f"numeric_key_count={len(numeric_metrics)} has_val_auc={'val_auc' in numeric_metrics}"
            ))

            # Debug print of outgoing metrics formatting
            try:
                preview_items = []
                for i, k in enumerate(sorted(numeric_metrics.keys())):
                    if i >= 25:
                        preview_items.append(f"...+{len(numeric_metrics) - 25} more")
                        break
                    v = numeric_metrics[k]
                    if isinstance(v, float):
                        preview_items.append(f"{k}={v:.6g}")
                    else:
                        preview_items.append(f"{k}={v}")
                self.log_info(fl_ctx, f"[metrics-debug] outgoing metrics: {' | '.join(preview_items)}")
            except Exception as e:
                self.log_warning(fl_ctx, f"[metrics-debug] failed to format metrics preview: {e}")

            reply = FLModelUtils.to_shareable(fl_model_out)
            reply.set_return_code(ReturnCode.OK)

            self.log_info(
                fl_ctx,
                f"Round {current_round} done - "
                f"train_loss={train_metrics.get('train_loss', 'N/A')} "
                f"val_acc={val_metrics.get('val_accuracy', 'N/A')} "
                + (f"test_acc={test_metrics.get('test_accuracy', 'N/A')}" if test_metrics else "")
            )
            return reply

        except Exception as e:
            self.log_exception(fl_ctx, f"Error in all-in-one task: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)


    def _salvage_ckpt_path(self, fl_ctx: FLContext, round_n) -> str | None:
        """Path to this client's SENT weights for a round: prefer the authoritative
        global_trajectory file, fall back to the deployed local checkpoint (equal at epochs=1)."""
        client = fl_ctx.get_identity_name()
        base = self.salvage_ckpt_dir or self.results_dir
        for c in (os.path.join(base, "global_trajectory", f"{client}_global_round_{int(round_n)}.pth"),
                  os.path.join(base, f"{client}_gmic_model_round_{int(round_n)}.pth")):
            if os.path.exists(c):
                return c
        return None

    def _write_salvage_json(self, fl_ctx: FLContext, site, method_tag, round_n, metrics):
        """Durable per-row stats copy in this node's results_dir (survives on the client)."""
        try:
            persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
            os.makedirs(persistent_dir, exist_ok=True)
            rec = {"site": site, "method": method_tag, "src_round": int(round_n), **metrics}
            out = os.path.join(persistent_dir,
                               f"{site}_salvage_{method_tag}_round{int(round_n)}_metrics.json")
            with open(out, "w") as f:
                json.dump(rec, f, indent=2)
        except Exception as e:
            self.log_warning(fl_ctx, f"[salvage] could not write stats json: {e}")

    def _salvage_round(self, fl_ctx: FLContext, shareable: Shareable, incoming) -> Shareable:
        """Test-only salvage of an interrupted run: reconstruct + evaluate per-round globals.

        Drives a plain FedAvg job with NO training over salvage_eval_rounds = [N0, N1, ...]. At FL
        round r the client:
          - r>=1: the broadcast `incoming` is the server's train-size-weighted aggregate of round
            (r-1)'s resubmitted SENT weights == the reconstructed global built from contributions of
            source round salvage_eval_rounds[r-1]. Evaluate it on val+test and dump preds
            (method=incoming_global, round=that source round) -> the shared-global endpoint
            (per-site + pooled computed offline by tools/salvage_stats.py).
          - r<K: load this client's OWN sent weights for N=salvage_eval_rounds[r], evaluate on
            val+test and dump preds (method=deployed_local, round=N) -> the as-deployed per-site
            endpoint; submit those weights weighted by train_size so the server reconstructs round
            N's global for evaluation next round.
          - r==K (trailing round): unit-weight dummy submission; only the eval of the last
            reconstruction matters.

        Weighting (requirement 1) is the freshly-preprocessed train_size, matching the original
        FedAvg aggregation weight at epochs=1. deployed_local at round N == the sent weights ==
        (epochs=1) the best-val deployed model, so its val AUC reproduces the original logged
        deployed_val_auc[N] -- a built-in end-to-end pipeline check. Server config: a normal FedAvg
        controller with num_rounds = len(salvage_eval_rounds) + 1.
        """
        from salvage_metrics import breast_aggregate, endpoint_metrics, format_endpoint

        current_round = int(shareable.get_header(AppConstants.CURRENT_ROUND, 0))
        client = fl_ctx.get_identity_name()
        rounds = self.salvage_eval_rounds
        K = len(rounds)

        def _eval_and_dump(model, src_round, method_tag):
            """Dump val+test preds, compute + LOG this site's breast-level stats, and return the
            breast-level {split: (probs, labels)} so incoming_global can be pooled on the server."""
            breasts = {}
            for split in ("val", "test"):
                if not self.data_loader.get_data_for_split(split):
                    continue
                try:
                    rows = self._dump_predictions(fl_ctx, split=split, model=model,
                                                  round_idx=int(src_round), method_tag=method_tag,
                                                  save_saliency=False)
                except Exception as e:
                    self.log_warning(fl_ctx, f"[salvage] {method_tag} preds ({split}) failed: {e}")
                    continue
                if rows:
                    bp, by = breast_aggregate([r["site_id"] for r in rows], [r["exam_id"] for r in rows],
                                              [r["view"] for r in rows], [r["prob_malignant"] for r in rows],
                                              [r["label"] for r in rows])
                    breasts[split] = (bp, by)
            if "val" in breasts and "test" in breasts:
                m = endpoint_metrics(breasts["val"][0], breasts["val"][1],
                                     breasts["test"][0], breasts["test"][1])
                # Per-site paper row: logged AND written to a durable JSON in this node's results_dir
                # (alongside the prediction CSVs), so it survives on the client regardless of logs.
                self.log_info(fl_ctx, f"[salvage-stats] {method_tag} src_round={int(src_round)} "
                                      + format_endpoint(client, m))
                self._write_salvage_json(fl_ctx, client, method_tag, int(src_round), m)
            return breasts

        # Ship each site's de-identified breast-level (prob,label) to the server so it can log BOTH
        # the per-site rows AND the pooled row to the PERSISTENT server log (/workspace/server_logs),
        # which survives the job-workspace deletion -- and reaches sites whose own logs/files are not
        # accessible (e.g. RSNA-AU). Scores+labels only: no images, no IDs.
        def _record(method_tag, src_round, breasts):
            return {"site": client, "method": method_tag, "src_round": int(src_round),
                    "splits": {s: {"p": [float(x) for x in breasts[s][0]],
                                   "y": [int(x) for x in breasts[s][1]]} for s in breasts}}
        records = []

        # (0) The server attaches the POOLED rows it computed last round to this round's broadcast
        # (via SalvageShareableGenerator). Log them here and write a durable client-side copy, so the
        # pooled endpoint lives on the clients too (two-way: server log/jsonl + every client).
        try:
            raw_pooled = shareable.get_header("salvage_pooled_results")
            if raw_pooled:
                for row in (json.loads(raw_pooled) if isinstance(raw_pooled, str) else raw_pooled):
                    self.log_info(fl_ctx, f"[salvage-pooled] (from server) src_round={row.get('src_round')} "
                                          + format_endpoint("POOLED", row))
                    self._write_salvage_json(fl_ctx, "POOLED", "incoming_global",
                                             int(row.get("src_round", -1)), row)
        except Exception as e:
            self.log_warning(fl_ctx, f"[salvage] reading pooled-from-server failed: {e}")

        # (1) Evaluate the reconstructed global broadcast this round (aggregate of last submission).
        if current_round >= 1 and current_round - 1 < K and incoming is not None:
            src = int(rounds[current_round - 1])
            self._underlying.load_state_dict(self._ensure_torch_state(incoming, fl_ctx), strict=False)
            self.log_info(fl_ctx, f"[salvage] evaluating reconstructed global (from round-{src} contributions)")
            gb = _eval_and_dump(self.model, src, "incoming_global")
            if gb:
                records.append(_record("incoming_global", src, gb))

        # Per-site weight = train-set size from the freshly preprocessed (cached) data.
        try:
            num_examples = int(self.data_loader.get_split_info().get("train_size", 0)) or int(self.batch_size)
        except Exception as e:
            self.log_warning(fl_ctx, f"[salvage] could not read train_size ({e}); using batch_size")
            num_examples = int(self.batch_size)

        # (2) If rounds remain, load + eval + submit this client's own SENT weights for that round.
        if current_round < K:
            n = int(rounds[current_round])
            path = self._salvage_ckpt_path(fl_ctx, n)
            if path is None:
                raise FileNotFoundError(f"[salvage] no sent/local checkpoint for {client} round {n}")
            sd = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(sd, dict) and not any(torch.is_tensor(v) for v in sd.values()):
                for wrap in ("model", "state_dict", "model_dict"):
                    if isinstance(sd.get(wrap), dict):
                        sd = sd[wrap]
                        break
            self._underlying.load_state_dict(sd, strict=False)
            self.log_info(fl_ctx, f"[salvage] loaded own round-{n} sent weights <- {path}")
            db = _eval_and_dump(self._underlying, n, "deployed_local")
            if db:
                records.append(_record("deployed_local", n, db))
            weight = num_examples
            self.log_info(fl_ctx, f"[salvage] submitting round-{n} weights weight(train_size)={weight}")
        else:
            weight = 1
            self.log_info(fl_ctx, "[salvage] trailing round: unit-weight dummy submission (aggregate unused)")

        final_state = {k: v.detach().cpu().clone() for k, v in self._underlying.state_dict().items()}
        updated_weights = {k: v.numpy() for k, v in final_state.items()}
        model_meta = {MetaKey.NUM_STEPS_CURRENT_ROUND: weight, "num_examples": weight}
        metrics = {"round": int(current_round), "salvage": 1, "train_samples": int(weight)}
        fl_model_out = FLModel(
            params=updated_weights, params_type=ParamsType.FULL, metrics=metrics, meta=model_meta
        )
        reply = FLModelUtils.to_shareable(fl_model_out)
        # Header channel for the server-side stats logger (read by SalvagePooledAggregator). JSON
        # string so it survives serialization regardless of FOBS config; absent on the trailing
        # round / sites with no data -> server simply skips that contribution.
        if records:
            try:
                reply.set_header("salvage_stats", json.dumps(records))
            except Exception as e:
                self.log_warning(fl_ctx, f"[salvage] could not attach stats payload: {e}")
        reply.set_return_code(ReturnCode.OK)
        return reply

    @staticmethod
    def _unwrap_state_dict(sd):
        """Tolerate a {'model'/'state_dict': sd} wrapper; our savers write a flat dict."""
        if isinstance(sd, dict) and not any(torch.is_tensor(v) for v in sd.values()):
            for wrap in ("model", "state_dict", "model_dict"):
                if isinstance(sd.get(wrap), dict):
                    return sd[wrap]
        return sd

    def _restore_personal_model(self, fl_ctx: FLContext):
        """Crash-resume: reload the Ditto personal model v from its round-N checkpoint.

        `_save_local_model` persists the DEPLOYED model, which for Ditto-family IS v, so
        `{client}_gmic_model_round_{N}.pth` is exactly the state to restore (the same file the
        shared-model methods use for their global reseed -- hence the method-dependent split in
        _reseed_round). Raises rather than falling back to the pretrained init: a silent cold v
        is the precise failure this exists to prevent, and it would not be visible in the logs.

        The v optimizer's Adam moment estimates were never persisted and cannot be recovered; the
        caller rebuilds them fresh, which re-warms within a few steps on an already-warm model.
        """
        client_name = fl_ctx.get_identity_name()
        round_n = int(self.resume_from_local_round)
        ckpt_dir = self.resume_ckpt_dir or self.results_dir
        ckpt = os.path.join(ckpt_dir, f"{client_name}_gmic_model_round_{round_n}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"[resume] personal model checkpoint not found for {client_name}: {ckpt} "
                f"(refusing to restart Ditto's personal model v from the pretrained init on a "
                f"resumed run -- that would desynchronize v from the round-{round_n} global)")
        # map_location="cpu": load_state_dict copies to the module's device anyway, so staging the
        # checkpoint on the GPU only adds a transient full-model allocation on an already-tight card.
        sd = self._unwrap_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        load_res = self._v_model.load_state_dict(sd, strict=False)
        self.log_info(
            fl_ctx,
            f"[resume] restored personal model v <- {ckpt} "
            f"(missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}); "
            f"v optimizer state rebuilt fresh (never persisted)"
        )

    def _reseed_round(self, fl_ctx: FLContext, shareable: Shareable) -> Shareable:
        """Resumed-job round 0: submit this client's cached round-N weights UNTRAINED.

        Each client loads its OWN `{client}_gmic_model_round_{N}.pth` and returns those weights
        as its round-0 contribution, weighted by its train-set size. The server's weighted
        aggregator then computes Sum(w_i * n_i)/Sum(n_i) -- exactly the round-N global -- and
        broadcasts it, so round 1 onward trains as normal. No file gathering or server-side
        file access is needed (clients ship weights over the existing FL channel), and the
        per-site weight is the freshly-preprocessed train_size (the canonical FedAvg
        data-volume weight, ~equal to the original round-N aggregation weight at epochs=1).
        """
        current_round = int(shareable.get_header(AppConstants.CURRENT_ROUND, 0))
        client_name = fl_ctx.get_identity_name()
        round_n = int(self.resume_from_local_round)
        ckpt_dir = self.resume_ckpt_dir or self.results_dir
        # WHICH file holds the shared global w^N is method-dependent. For the shared-model methods
        # the deployed model IS w, so the plain round checkpoint is the aggregation contribution.
        # For Ditto-family the deployed (and therefore saved) model is the PERSONAL model v --
        # re-aggregating three sites' personal models would yield a meaningless "global" -- so read
        # the SENT-global trajectory cache instead, which is exactly w^N (requires
        # cache_global_trajectory, which the Ditto runs set).
        if self.method in ("ditto", "ditto_modulewise"):
            ckpt = os.path.join(ckpt_dir, "global_trajectory",
                                f"{client_name}_global_round_{round_n}.pth")
            what = f"sent global w^{round_n} (Ditto-family: deployed ckpt is the personal v, not w)"
        else:
            ckpt = os.path.join(ckpt_dir, f"{client_name}_gmic_model_round_{round_n}.pth")
            what = f"round-{round_n} deployed global"
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"[resume] cached checkpoint not found for {client_name}: {ckpt} (expected the {what})")

        sd = self._unwrap_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        load_res = self._underlying.load_state_dict(sd, strict=False)
        self.log_info(
            fl_ctx,
            f"[resume] loaded {client_name} round-{round_n} weights <- {ckpt} "
            f"(missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}); "
            f"submitting UNTRAINED for server re-aggregation"
        )

        # The aggregation contribution IS the loaded round-N weights (no training this round).
        final_state = {k: v.detach().cpu().clone() for k, v in self._underlying.state_dict().items()}
        updated_weights = {k: v.numpy() for k, v in final_state.items()}

        # Per-site weight = train-set size from the freshly preprocessed data.
        try:
            split_info = self.data_loader.get_split_info()
            num_examples = int(split_info.get("train_size", 0)) or int(self.batch_size)
        except Exception as e:
            self.log_warning(fl_ctx, f"[resume] could not read train_size ({e}); falling back to batch_size")
            num_examples = int(self.batch_size)

        # Quick val pass on the loaded weights so IntimeModelSelector has a metric this round.
        val_auc = None
        try:
            val_core = self._evaluate_model(fl_ctx, split="val", model=self._underlying)
            auc = val_core.get("auc")
            if auc is not None and not math.isnan(float(auc)):
                val_auc = float(auc)
        except Exception as e:
            self.log_warning(fl_ctx, f"[resume] val pass failed (non-fatal): {e}")

        model_meta = {MetaKey.NUM_STEPS_CURRENT_ROUND: num_examples, "num_examples": num_examples}
        metrics = {"round": int(current_round), "resume_seed": 1, "train_samples": int(num_examples)}
        if val_auc is not None:
            model_meta["val_auc"] = val_auc
            metrics["val_auc"] = val_auc

        fl_model_out = FLModel(
            params=updated_weights, params_type=ParamsType.FULL, metrics=metrics, meta=model_meta
        )
        reply = FLModelUtils.to_shareable(fl_model_out)
        reply.set_return_code(ReturnCode.OK)
        self.log_info(
            fl_ctx,
            f"[resume] round {current_round} re-seed complete: site={client_name} "
            f"weight(num_examples)={num_examples} val_auc={val_auc}; "
            f"server will aggregate all clients -> round-{round_n} global"
        )
        return reply


    def _ensure_torch_state(self, params: dict, fl_ctx=None) -> dict:
        """Convert incoming FLModel.params (possibly numpy) into a proper torch state_dict."""
        ref_model = self._underlying if hasattr(self, '_underlying') else self.model
        ref = ref_model.state_dict()
        device = next(ref_model.parameters()).device
        out = {}

        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                t = v.to(device)
            elif isinstance(v, np.ndarray):
                dtype = ref[k].dtype if k in ref else torch.float32
                t = torch.from_numpy(v).to(device=device, dtype=dtype)
            elif np.isscalar(v):
                dtype = ref[k].dtype if k in ref else torch.float32
                t = torch.tensor(v, device=device, dtype=dtype)
            else:
                if fl_ctx:
                    self.log_warning(fl_ctx, f"Skipping param {k} of unsupported type {type(v)}")
                continue

            if "num_batches_tracked" in k:
                t = t.to(torch.long)
            out[k] = t
        return out


    def _get_model_weights(self) -> Shareable:
        """Get current GMIC model weights"""
        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=weights,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_train_iterations}
        )
        return outgoing_dxo.to_shareable()


    def _compute_task_loss(self, outputs, targets):
        """Task loss on the malignant head: native GMIC deep-supervised BCE + L1
        (BCEWithLogits(fusion) + BCE(global) + BCE(local) + lambda_l1*L1, malignant
        channel, shared pos_weight)."""
        return gmic_malignant_loss(
            outputs, targets, lambda_l1=self.lambda_l1, pos_weight=self.pos_weight
        )


    def _apply_backbone_freeze(self, fl_ctx, frozen: bool):
        """Toggle requires_grad on the global+local backbones (freeze_backbone_epochs lever).

        Idempotent; logs only on a state change. Heads stay trainable throughout.
        """
        if getattr(self, "_backbone_frozen", None) == frozen:
            return
        mdl = self._underlying
        for attr in ("global_backbone", "local_backbone"):
            bb = getattr(mdl, attr, None)
            if bb is not None and hasattr(bb, "parameters"):
                for p in bb.parameters():
                    p.requires_grad = not frozen
        self._backbone_frozen = frozen
        self.log_info(fl_ctx, f"[freeze] backbones {'FROZEN' if frozen else 'UNFROZEN'} (heads always trainable)")

    def _effective_group_lrs(self):
        """Return (backbone_lr, heads_lr) from the live optimizer param groups.

        configure_optimizers appends the backbone group FIRST (lr=lr_backbone) then the head
        groups (lr=lr_heads), so group 0 = backbone, groups 1+ = heads. Robust to the no-backbone
        fallback (single group) by reporting that group's lr for both.
        """
        groups = self.optimizer.param_groups
        if not groups:
            return float("nan"), float("nan")
        bb_lr = float(groups[0]["lr"])
        hd_lr = float(groups[1]["lr"]) if len(groups) > 1 else bb_lr
        return bb_lr, hd_lr

    def _scheduler_step(self, val_auc, epoch=None):
        """Step the LR scheduler, tolerating plateau (needs metric) vs cosine/step (no metric).

        Cosine-after-unfreeze: when the backbone is frozen for the first N epochs
        (freeze_backbone_epochs>0), a CosineAnnealingLR is HELD during those frozen epochs
        (not stepped), so its decay clock starts at unfreeze and cosine_t_max counts only
        post-unfreeze epochs. Plateau/None are unaffected (plateau tracks val_auc throughout).
        """
        sched = getattr(self, "scheduler", None)
        if sched is None:
            return
        # Hold a cosine schedule during the frozen warmup (decay starts at unfreeze).
        if (epoch is not None
                and self.freeze_backbone_epochs > 0
                and epoch < self.freeze_backbone_epochs
                and isinstance(sched, optim.lr_scheduler.CosineAnnealingLR)):
            self._logger.info(
                "[scheduler] cosine HELD during frozen epoch %d (decay starts at unfreeze)", epoch + 1
            )
            return
        try:
            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_auc)
            else:
                sched.step()
        except Exception as e:
            self._logger.warning("[scheduler] step failed: %s", e)


    def _local_train(self, fl_ctx, abort_signal, shareable: Shareable):
        """Local training with per-epoch validation, scheduler stepping, and early stopping.

        Returns final aggregate training metrics plus best validation AUC observed.
        """
        has_val = len(self.data_loader.get_data_for_split('val')) > 0
        self.model.train()
        # C2: reset the early stopper each round so best-val selection is within-round
        # (best-state never crosses rounds, for either the sent or the deployed model).
        if getattr(self, 'early_stopper', None):
            self.early_stopper.reset()
        total_samples = 0
        epochs_run = 0
        round_best_val = -float('inf')
        last_epoch_loss = 0.0
        last_epoch_auc = 0.0
        last_epoch_acc = 0.0

        train_bs = max(1, int(self._train_batch_size_by_site.get(
            fl_ctx.get_identity_name(), self.batch_size)))
        if train_bs < self.batch_size:
            self.log_info(
                fl_ctx,
                f"[main-train] micro-batch={train_bs} (loader batch {self.batch_size} chunked; grads "
                f"accumulated to the full batch, only BN sees the chunk) -- keeps the unfrozen "
                f"backbone pass under this card's memory ceiling")

        for epoch in range(self.epochs):
            if abort_signal.triggered:
                break
            epochs_run = epoch + 1
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_preds = []
            epoch_targets = []
            self.log_info(fl_ctx, f"Training epoch {epoch + 1}/{self.epochs}")

            # Cumulative epoch index across the WHOLE run (not the per-call epoch, which resets
            # to 0 every FL round). Centralized (1 round, many epochs): == epoch, unchanged. FL
            # (epochs=1 per round): == logical_round, so any lever that counts "epochs" correctly
            # spans rounds. Without this, the freeze gate (epoch<N) was ALWAYS true in FL (epoch
            # is always 0 when epochs=1) and silently froze the backbone every round forever --
            # whereas centrally (epochs=40) it correctly froze epochs 0..N-1 then trained.
            logical_round = int(shareable.get_header(AppConstants.CURRENT_ROUND, 0)) + self._round_offset
            global_epoch = logical_round * self.epochs + epoch

            # Backbone freeze lever: keep backbones frozen for the first N epochs, then unfreeze
            # once (anti-overfitting warmup). No-op when freeze_backbone_epochs==0 (default).
            if self.freeze_backbone_epochs > 0:
                self._apply_backbone_freeze(fl_ctx, frozen=(global_epoch < self.freeze_backbone_epochs))

            for batch_idx, (inputs, targets, metadata) in enumerate(self.data_loader.get_batch_iterator('train')):
                if abort_signal.triggered:
                    break
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                accumulate = self.grad_accumulation
                if batch_idx % accumulate == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                amp_device = "cuda" if "cuda" in str(self.device) else "cpu"
                total = inputs.size(0)
                n_chunks = max(1, math.ceil(total / train_bs))
                t_mb = time.time()
                # Micro-chunk the loader batch so each forward/backward holds only `train_bs` worth of
                # activations. Critical once the backbone UNFREEZES: autograd then retains all backbone
                # activations for backward, and the full fp16 batch overflows a 16 GiB card -> host-RAM
                # oversubscription -> deadlock. Loss OUTSIDE autocast in fp32 (BCE on sigmoided probs is
                # unsafe under autocast). Each chunk's loss is weighted by its share and divided by the
                # grad-accum factor, so the accumulated gradient == a single full loader-batch step's;
                # only BN sees the smaller chunk. n_chunks==1 (train_bs >= batch) is the un-chunked path.
                effective_loss = 0.0
                for ci in range(n_chunks):
                    cx = inputs[ci * train_bs:(ci + 1) * train_bs]
                    cy = targets[ci * train_bs:(ci + 1) * train_bs]
                    if cx.size(0) == 0:
                        continue
                    with torch.autocast(device_type=amp_device, enabled=self.use_amp):
                        out = self.model(cx)
                    loss_c = self._compute_task_loss(out, cy) * (cx.size(0) / total) / accumulate
                    if self.use_amp and self._scaler is not None:
                        self._scaler.scale(loss_c).backward()
                    else:
                        loss_c.backward()
                    effective_loss += float(loss_c.item()) * accumulate
                    # Malignant probability for batch AUC = sigmoid(fusion_logit)[:, 1]
                    epoch_preds.extend(malignant_score(out).detach().cpu().numpy().reshape(-1))
                    epoch_targets.extend(cy.detach().cpu().numpy().reshape(-1))
                # FedProx: (mu/2)||w - w_global||^2, once per loader batch (None for
                # fedavg/fedbn/local/ditto -> skipped).
                if getattr(self, "_prox_ref", None) is not None:
                    prox = proximal_penalty(
                        self._underlying, self._prox_ref, self._prox_lambda) / accumulate
                    if self.use_amp and self._scaler is not None:
                        self._scaler.scale(prox).backward()
                    else:
                        prox.backward()
                    effective_loss += float(prox.item()) * accumulate

                if (batch_idx + 1) % accumulate == 0:
                    if self.use_amp and self._scaler is not None:
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        self.optimizer.step()

                total_samples += total
                epoch_loss += effective_loss
                epoch_batches += 1

                if self.debug_devices and batch_idx == 0:
                    try:
                        inp_dev = first_batch_input_device_str(inputs)
                        summary = summarize_parameter_devices(self.model)
                        self.log_info(fl_ctx, f"[devices] first_batch_input_devices={inp_dev}")
                        self.log_info(fl_ctx, f"[devices] param_devices={summary['unique_devices']} counts={summary['device_counts']}")
                        for ex in summary['examples']:
                            self.log_info(fl_ctx, f"[devices] example {ex}")
                    except Exception as e:
                        self.log_warning(fl_ctx, f"[devices] diagnostics failed: {e}")

                if self.train_log_batch_interval and (batch_idx % self.train_log_batch_interval == 0):
                    self.log_info(
                        fl_ctx,
                        f"Epoch {epoch + 1} Batch {batch_idx} Loss {effective_loss:.4f} "
                        f"({time.time() - t_mb:.1f}s) chunks={n_chunks}x{train_bs} {self._cuda_mem_str()}")

                if self.memory_efficient and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if epoch_batches == 0:
                break

            ep_preds_np = np.array(epoch_preds)
            ep_targets_np = np.array(epoch_targets)

            try:
                ep_auc = roc_auc_score(ep_targets_np, ep_preds_np) if len(np.unique(ep_targets_np)) > 1 else 0.0
            except Exception as e:
                self.log_warning(fl_ctx, f"roc_auc_score failed at epoch {epoch + 1}: {e}")
                ep_auc = 0.0

            ep_pred_labels = (ep_preds_np > 0.5).astype(int)
            ep_acc = 100 * accuracy_score(ep_targets_np, ep_pred_labels) if len(ep_targets_np) else 0.0
            ep_loss_avg = epoch_loss / max(epoch_batches, 1)

            # Per-epoch effective per-group LRs (always on): group 0 = backbone, groups 1+ = heads
            # (per configure_optimizers). Logged explicitly labeled so a plumbing bug giving both
            # groups the same LR is visible, and so the backbone LR is recorded (not just heads).
            bb_lr, hd_lr = self._effective_group_lrs()
            self.log_info(fl_ctx, f"Epoch {epoch + 1} LRs: heads={hd_lr:.3e} backbone={bb_lr:.3e}")
            # Aggregated epoch summary (greppable trajectory; watch train loss/AUC asymptote).
            self.log_info(
                fl_ctx,
                f"Epoch {epoch + 1} summary train_loss={ep_loss_avg:.4f} train_auc={ep_auc:.4f} "
                f"train_acc={ep_acc:.2f}%"
            )

            last_epoch_loss = ep_loss_avg
            last_epoch_auc = ep_auc
            last_epoch_acc = ep_acc

            val_auc = None
            if has_val:
                val_metrics = self._evaluate_model(fl_ctx, split='val')
                val_auc = val_metrics['auc']
                # plateau scheduler steps on val_auc; cosine (and others) step without a metric;
                # None scheduler -> no-op. Handle all three. `epoch` is passed so a cosine
                # schedule can HOLD during the frozen warmup and only begin decaying at unfreeze
                # (so cosine_t_max counts post-unfreeze epochs, not frozen ones).
                self._scheduler_step(val_auc, epoch=global_epoch)
                if self.early_stopper.step(val_auc, self.model):
                    round_best_val = val_auc
                if self.early_stopper.should_stop():
                    self.log_info(fl_ctx, f"Early stopping triggered at epoch {epoch + 1} (best_val_auc={self.early_stopper.best:.4f})")
                    break
                self.model.train()

            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': float(ep_loss_avg),
                'train_auc': float(ep_auc),
                'train_accuracy': float(ep_acc),
                'val_auc': float(val_auc) if val_auc is not None else None,
                'lr_groups': [pg['lr'] for pg in self.optimizer.param_groups],
            })

        best_val = self.early_stopper.best if (has_val and getattr(self.early_stopper, 'best_state', None)) else (
            round_best_val if round_best_val > -float('inf') else 0.0
        )
        early_stopped = int(epochs_run < self.epochs)

        train_metrics = {
            'train_samples': int(total_samples),
            'epochs_completed': int(epochs_run),
            'best_val_auc': float(best_val),
            'early_stopped': early_stopped,
            'patience': int(self.early_stopper.patience) if hasattr(self, 'early_stopper') else None,
            'train_loss': float(last_epoch_loss),
            'train_auc': float(last_epoch_auc),
            'train_accuracy': float(last_epoch_acc),
        }

        self.log_info(fl_ctx, f"Training finished (epochs_run={epochs_run}, best_val_auc={best_val:.4f})")
        return train_metrics


    def _evaluate_model(self, fl_ctx: FLContext, split='val', model=None):
        """Run evaluation and return base metric dict (auc, accuracy, loss, samples).

        `model` defaults to self.model (the global-tracked w); pass the deployed
        model (e.g. the Ditto personal model v) to evaluate it instead.
        """
        eval_model = model if model is not None else self.model
        # evaluate_model is one opaque call with no internal logging, so a stall inside it is
        # invisible. The heartbeat keeps reporting from a blocked state (log-only, never aborts).
        t0 = time.time()
        with self._heartbeat("evaluate", {"site": fl_ctx.get_identity_name(), "split": split,
                                          "t_last_progress": t0}):
            core = evaluate_model(eval_model, self.data_loader, self.criterion, self.device,
                                  split=split, decision_threshold=self.decision_threshold,
                                  eval_threshold_sweep=self.eval_threshold_sweep)
        out = {
            "auc": float(core["auc"]),
            "accuracy": float(core["accuracy"]),
            "loss": float(core["loss"]),
            "samples": int(core.get("total_samples", 0)),
            # operating-point metrics at decision_threshold (AUC stays threshold-free)
            "sensitivity": float(core.get("sensitivity", float("nan"))),
            "specificity": float(core.get("specificity", float("nan"))),
            "precision": float(core.get("precision", float("nan"))),
            "n_pos": int(core.get("n_pos", 0)),
            "n_images": int(core.get("n_images", 0)),
        }
        # Metrics are BREAST-LEVEL (GMIC's reported unit): n/n_pos are BREAST counts, ~1/2 the
        # image count. AUC/sens/spec selected/reported on breasts. Loss is per-image (its
        # denominator is n_images, shown for visibility). n_pos kept for DeLong CIs.
        self.log_info(
            fl_ctx,
            f"{split.capitalize()} evaluation (breast-level) - AUC {out['auc']:.4f} "
            f"Acc {out['accuracy']:.2f}% Loss {out['loss']:.4f} (per-image, n_images={out['n_images']}) "
            f"| n_breasts={out['samples']} n_pos={out['n_pos']} @thr={self.decision_threshold:.2f} "
            f"Sens {out['sensitivity']:.4f} Spec {out['specificity']:.4f} Prec {out['precision']:.4f}"
        )
        # Additive operating-point sweep (greppable; denominators logged so small-n noise is
        # judgeable). Does NOT affect selection/early-stop/saved-best — logging only.
        n_pos = int(core.get("n_pos", 0))
        n_neg = int(core.get("n_neg", 0))
        for sw in core.get("sweep", []):
            self.log_info(
                fl_ctx,
                f"[threshold-sweep] split={split} (breast-level) thr={sw['threshold']:.2f} "
                f"sens={sw['sensitivity']:.4f} spec={sw['specificity']:.4f} prec={sw['precision']:.4f} "
                f"n_flagged={sw['n_flagged']} n_pos={n_pos} n_neg={n_neg}"
            )
        return out


    def _consume_global(self, incoming, fl_ctx: FLContext):
        """Method-aware load of the received global weights into the tracked model w.

          local             -> skip the load (each site trains independently from its
                               pretrained init; it still returns w so NVFLARE plumbing
                               is happy, but never consumes the aggregate).
          fedbn / use_fedbn -> load global but SKIP BatchNorm keys (local BN preserved).
          else              -> full load.

        Also captures the frozen global reference w_ref (FedProx/Ditto) and lazily
        initializes the Ditto personal model v exactly once (from the pretrained w).
        """
        method = self.method
        needs_ref = method in ("fedprox", "ditto", "ditto_modulewise")
        self._prox_ref = None
        self._prox_lambda = None

        if incoming is None:
            self.log_info(fl_ctx, "No global weights in shareable; keeping current model")
        elif method == "local":
            self.log_info(fl_ctx, "[method=local] skipping global load (independent local training)")
        else:
            skip = set()
            if method == "fedbn" or self.use_fedbn:
                skip = bn_state_keys(self._underlying)
            stats = load_global_into(self._underlying, incoming, skip_keys=skip)
            self.log_info(
                fl_ctx,
                f"[method={method}] consumed global: {stats} (bn_keys_skipped={len(skip)})"
            )

        # Frozen reference = the round's received global weights (current w state).
        if needs_ref and incoming is not None:
            self._global_ref = clone_detached_state(self._underlying)
            if method == "fedprox":
                self._prox_ref = self._global_ref
                self._prox_lambda = float(self.fedprox_mu)
        else:
            self._global_ref = None

        # Lazily create the Ditto personal model v ONCE, from the (pretrained) w.
        if method in ("ditto", "ditto_modulewise") and self._v_model is None:
            self._v_model = copy.deepcopy(self._underlying).to(self.device)
            # Crash-resume: v PERSISTS across rounds and is never re-derived from the global, so a
            # resumed segment MUST reload it -- otherwise the run carries a warm round-N global but
            # a personal model restarted from the pretrained init, silently desynchronizing the two
            # trajectories by N rounds. Restored here (not in _reseed_round) because the reseed
            # short-circuits before _consume_global, so v is first created on the next round.
            if self.resume_from_local_round >= 0:
                self._restore_personal_model(fl_ctx)
            try:
                opt, _, _ = configure_optimizers(self._v_model, self._opt_args)
            except Exception as e:
                self.log_warning(fl_ctx, f"[ditto] failed to build v optimizer via configure_optimizers: {e}")
                opt = torch.optim.Adam(self._v_model.parameters(), lr=self.lr_heads,
                                       weight_decay=self.weight_decay)
            self._v_optimizer = opt
            self.log_info(fl_ctx, "[ditto] initialized personal model v from pretrained init (persists across rounds)")


    def _sync(self):
        """Wait for queued GPU work, so the CURRENT stage label is the one that wedged.

        Call AFTER an op while `stage` still names it: if the sync blocks, the heartbeat reports
        that op rather than the next sync point. No-op unless stage_sync is set (and off CUDA).
        """
        if getattr(self, "stage_sync", False) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _log_gpu_diagnostics(self):
        """Log GPU / CUDA / cuDNN / driver identity + effective allocator config, ONCE at init.

        Every fact here is obtainable in-process, so a remote site we cannot shell into still
        reports exactly which card + driver + cuDNN it is running -- the info needed to decide
        whether an fp16 backward kernel that deadlocks is a known GPU/driver issue. total VRAM in
        particular settles the card model (an fp16 personal-pass hang was observed with ~18.9 GiB
        RESERVED, which is impossible on a 16 GiB card, so RSNA-AU is NOT the assumed 16 GiB A4000).
        """
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")
        self._logger.info("[gpu-diag] PYTORCH_CUDA_ALLOC_CONF=%s", alloc_conf)
        if not torch.cuda.is_available():
            self._logger.info("[gpu-diag] CUDA not available (CPU run)")
            return
        try:
            props = torch.cuda.get_device_properties(0)
            self._logger.info(
                "[gpu-diag] device=%s total_mem=%.1f GiB torch=%s cuda=%s cudnn=%s",
                props.name, props.total_memory / 2**30, torch.__version__,
                torch.version.cuda, torch.backends.cudnn.version())
        except Exception as e:
            self._logger.warning("[gpu-diag] device props unavailable: %s", e)
        # Driver version is not exposed by torch; nvidia-smi runs inside the container (no host
        # access needed). Best-effort, short timeout, never fatal.
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            if out.stdout.strip():
                self._logger.info("[gpu-diag] nvidia-smi: %s", out.stdout.strip().replace("\n", " | "))
        except Exception as e:
            self._logger.info("[gpu-diag] nvidia-smi unavailable (non-fatal): %s", e)

    @staticmethod
    def _cuda_mem_str():
        """allocated/reserved/peak, for logs. Empty string off-CUDA so callers stay unconditional."""
        if not torch.cuda.is_available():
            return ""
        return (f"mem {torch.cuda.memory_allocated() / 2**30:.2f}/"
                f"{torch.cuda.memory_reserved() / 2**30:.2f}/"
                f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB alloc/resv/peak")

    @contextlib.contextmanager
    def _heartbeat(self, label, state):
        """Background thread that logs `label` + `state` every heartbeat_interval_s.

        Exists because per-batch logging goes silent in exactly the situation you most need it:
        when the main thread is blocked inside a CUDA kernel, a stalled mount read, or a wedged
        driver. A separate thread keeps emitting, so a remote site that cannot be shelled into
        still shows WHERE it stopped and for how long -- the difference between "slow" and "hung"
        without touching the box.

        Logs only; it never aborts or raises, so a slow-but-healthy site is never killed. `state`
        is a mutable dict the caller updates (e.g. {"batch": i}); it is read, not written, here.

        Uses self._logger directly rather than log_info(fl_ctx): FLContext is not documented as
        thread-safe, and a diagnostic must never be able to destabilize the run it is observing.
        """
        interval = int(getattr(self, "heartbeat_interval_s", 0) or 0)
        if interval <= 0:
            yield state
            return
        who = state.get("site", "?")
        stop = threading.Event()
        started = time.time()

        def _beat():
            while not stop.wait(interval):
                now = time.time()
                since = now - float(state.get("t_last_progress", started))
                self._logger.warning(
                    "[heartbeat] site=%s %s: %s | %.0fs since last progress, %.0fs in phase | %s "
                    "(no output between beats means the phase is blocked, not idle)",
                    who, label,
                    " ".join(f"{k}={v}" for k, v in state.items()
                             if k not in ("site", "t_last_progress")),
                    since, now - started, self._cuda_mem_str(),
                )

        t = threading.Thread(target=_beat, name=f"hb-{label}", daemon=True)
        t.start()
        try:
            yield state
        finally:
            stop.set()
            t.join(timeout=5)

    def _resolve_personal_amp(self, fl_ctx: FLContext):
        """Fix the personal-pass precision for THIS site, once, from the per-site override map.

        Single config deployed to every site: the map (e.g. {"RSNA-AU": false}) lets RSNA-AU run its
        personal pass in fp32 -- its A4000 faults on the fp16 backward -- while RSNA/HIPIMR keep AMP,
        with no per-site app duplication and no forced A100 move. Rebuilds the v scaler to match the
        resolved precision. Idempotent: resolves + logs exactly once.
        """
        if getattr(self, "_personal_amp_resolved", False):
            return
        site = fl_ctx.get_identity_name()
        overridden = site in self._personal_amp_by_site
        eff = bool(self._personal_amp_by_site.get(site, self._personal_amp_default))
        if eff != self._personal_amp and torch.cuda.is_available():
            self._v_scaler = torch.cuda.amp.GradScaler(enabled=eff)
        self._personal_amp = eff
        self._personal_amp_resolved = True
        self.log_info(
            fl_ctx,
            f"[ditto] personal-pass precision for site={site}: amp={eff} "
            f"({'per-site override' if overridden else 'default'}; main use_amp={bool(self.use_amp)})")

    def _train_personal_model(self, fl_ctx: FLContext, abort_signal: Signal):
        """Ditto personal pass: train v for self.epochs with task_loss(v) + proximal(v, w_ref).

        v and its optimizer are instance attributes that persist across rounds; v is
        never reset to the global. lambda is a scalar (ditto) or a per-group dict
        (ditto_modulewise: {global, local, fusion}).

        Precision follows `use_amp`, exactly like the main w pass. This pass used to run plain
        fp32 while the main pass ran under autocast, which made it need ~2x the activation memory
        of the pass right before it -- the deployed model was the single most expensive thing in
        the round. That is a hard hardware floor, not a tuning knob: at batch 32 / 2944x1920 it
        OOMs on 16GB- and 24GB-class cards (T4, L4) and only fits on A100-class memory. Following
        use_amp also makes v numerically consistent with every shared global in the study, all of
        which have always trained under autocast.
        """
        v = self._v_model
        opt = self._v_optimizer
        if v is None or opt is None or self._global_ref is None:
            self.log_warning(fl_ctx, "[ditto] personal model/ref not ready; skipping personal pass")
            return
        lam = self.ditto_lambda if self.method == "ditto" else self.lam_dict
        self._resolve_personal_amp(fl_ctx)  # per-site precision, once (fp32 for RSNA-AU, AMP elsewhere)
        personal_amp = self._personal_amp
        scaler = getattr(self, "_v_scaler", None) if personal_amp else None
        self.log_info(fl_ctx, f"[ditto] personal pass: epochs={self.epochs} lambda={lam} "
                              f"amp={bool(personal_amp)} (main use_amp={bool(self.use_amp)}) "
                              f"stage_sync={bool(self.stage_sync)}")
        # The main loop has just filled the caching allocator; hand the reserve back before this
        # pass allocates its own activations. Purely an allocator hint -- it frees nothing live
        # and does not change the math.
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.log_info(
                fl_ctx,
                f"[ditto] pre-personal-pass cache release: "
                f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated / "
                f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved"
            )
        v.train()
        client_name = fl_ctx.get_identity_name()
        personal_bs = max(1, int(self._personal_batch_size_by_site.get(client_name, self.batch_size)))
        if personal_bs < self.batch_size:
            self.log_info(
                fl_ctx,
                f"[ditto] personal micro-batch={personal_bs} (loader batch {self.batch_size} split "
                f"into chunks; gradients accumulated to the full effective batch, only BN sees the "
                f"smaller chunk) -- keeps the personal pass under this card's memory ceiling")
        for epoch in range(self.epochs):
            if abort_signal.triggered:
                break
            n_batches = 0
            n_skipped = 0
            t_epoch = time.time()
            # batch=-1 + stage=load means we never got the FIRST batch out of the data loader --
            # that alone separates an I/O/loader stall from a compute stall, without a shell on
            # the box. `stage` then narrows a compute stall to the exact operation.
            state = {"site": client_name, "epoch": f"{epoch + 1}/{self.epochs}", "batch": -1,
                     "stage": "load", "t_last_progress": t_epoch}
            with self._heartbeat("ditto-personal", state):
                for batch_idx, (inputs, targets, metadata) in enumerate(
                        self.data_loader.get_batch_iterator('train')):
                    if abort_signal.triggered:
                        break
                    t_batch = time.time()
                    state["batch"] = batch_idx
                    state["stage"] = "h2d"
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    total = inputs.size(0)
                    n_chunks = max(1, math.ceil(total / personal_bs))
                    amp_device = "cuda" if "cuda" in str(self.device) else "cpu"
                    opt.zero_grad(set_to_none=True)
                    # Micro-chunk the loader batch so each forward/backward only holds `personal_bs`
                    # worth of activations. Gradients accumulate across chunks, so the optimizer step
                    # equals a single full-batch step; only BatchNorm sees the smaller chunk. When
                    # n_chunks==1 (personal_bs >= batch_size) this is the un-chunked path.
                    nonfinite = False
                    task_loss_sum = 0.0
                    state["stage"] = "forward/backward"
                    for ci in range(n_chunks):
                        cx = inputs[ci * personal_bs:(ci + 1) * personal_bs]
                        cy = targets[ci * personal_bs:(ci + 1) * personal_bs]
                        if cx.size(0) == 0:
                            continue
                        with torch.autocast(device_type=amp_device, enabled=personal_amp):
                            out = v(cx)
                        # Loss OUTSIDE autocast in fp32 (GMIC heads emit sigmoided probs; BCE is
                        # unsafe under autocast). Weighted by the chunk's share of the batch so the
                        # accumulated gradient equals a single full-batch step's (mean-reduced loss
                        # + the per-call L1 term both scale correctly, since the weights sum to 1).
                        loss_c = self._compute_task_loss(out, cy) * (cx.size(0) / total)
                        # Stability guard: v PERSISTS across rounds, so one non-finite step poisons
                        # it for the whole run. Drop the ENTIRE batch's step if any chunk is bad.
                        if not torch.isfinite(loss_c):
                            nonfinite = True
                            break
                        (scaler.scale(loss_c) if (scaler is not None and personal_amp)
                         else loss_c).backward()
                        task_loss_sum += float(loss_c.item())
                    if nonfinite:
                        n_skipped += 1
                        opt.zero_grad(set_to_none=True)
                        self.log_warning(
                            fl_ctx,
                            f"[ditto] non-finite personal loss (epoch {epoch + 1} "
                            f"batch {batch_idx}); skipping step")
                        state["t_last_progress"] = time.time()
                        continue
                    # Proximal penalty ONCE per optimizer step -- it regularizes the weights, not
                    # per-sample, so adding it per chunk would multiply its strength.
                    state["stage"] = "prox"
                    prox = proximal_penalty(v, self._global_ref, lam)
                    if not torch.isfinite(prox):
                        n_skipped += 1
                        opt.zero_grad(set_to_none=True)
                        self.log_warning(fl_ctx, f"[ditto] non-finite proximal (epoch {epoch + 1} "
                                                 f"batch {batch_idx}); skipping step")
                        state["t_last_progress"] = time.time()
                        continue
                    (scaler.scale(prox) if (scaler is not None and personal_amp) else prox).backward()
                    state["stage"] = "step"
                    if scaler is not None and personal_amp:
                        scaler.unscale_(opt)  # before clip, so max_norm applies to true gradients
                        torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=5.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=5.0)
                        opt.step()
                    self._sync()
                    n_batches += 1
                    state["stage"] = "load"  # back to waiting on the iterator for the next batch
                    state["t_last_progress"] = time.time()
                    # Per-batch progress (mirrors the main loop's train_log_batch_interval). The
                    # displayed loss is the full-batch task loss + proximal; elapsed makes "slow"
                    # quantifiable, and chunks=NxM shows the micro-batching in effect.
                    if self.train_log_batch_interval and (batch_idx % self.train_log_batch_interval == 0):
                        self.log_info(
                            fl_ctx,
                            f"[ditto] personal epoch {epoch + 1} batch {batch_idx} "
                            f"loss {task_loss_sum + float(prox.item()):.4f} "
                            f"({time.time() - t_batch:.1f}s) chunks={n_chunks}x{personal_bs} "
                            f"{self._cuda_mem_str()}")
            dt = time.time() - t_epoch
            self.log_info(
                fl_ctx,
                f"[ditto] personal epoch {epoch + 1}/{self.epochs} done ({n_batches} batches, "
                f"{n_skipped} skipped non-finite, {dt:.1f}s total, "
                f"{dt / max(n_batches, 1):.1f}s/batch)")


    def _dump_baseline(self, fl_ctx: FLContext):
        """Round-0 cold-start reference: evaluate the pretrained weights on val+test and
        dump per-sample predictions tagged method=pretrained_baseline, round=0. This is
        the row every trained method is measured against and verifies the load+pipeline.
        """
        report = getattr(self, "_pretrained_report", None)
        if report is not None:
            self.log_info(fl_ctx, f"[baseline] pretrained load report: {report}")
        for split in ("val", "test"):
            if not self.data_loader.get_data_for_split(split):
                continue
            core = self._evaluate_model(fl_ctx, split=split, model=self.model)
            self.log_info(
                fl_ctx,
                f"[baseline] site={fl_ctx.get_identity_name()} method=pretrained_baseline "
                f"round=0 {split}_auc={core['auc']:.4f} {split}_acc={core['accuracy']:.2f} n={core['samples']}"
            )
            self._dump_predictions(fl_ctx, split=split, model=self.model,
                                   round_idx=0, method_tag="pretrained_baseline")


    def _track_incoming_global(self, fl_ctx: FLContext, round_idx):
        """Per-round evaluation + prediction dump for the RECEIVED global model.

        Runs BEFORE local training, on self.model right after _consume_global, so it measures
        the server aggregate exactly as this site consumes it (method-aware: full load for
        fedavg/fedprox/ditto; local BN preserved under fedbn). `local` has no aggregate, so it
        is skipped. Evaluating val AND test every round lets the best global be selected
        offline by val AUC with test metrics + preds already on disk. Predictions AND saliency
        maps are dumped per round; a small per-round metrics JSON is also written so the
        selection table survives a crash before END_RUN.
        """
        if self.method == "local":
            self.log_info(fl_ctx, "[incoming-global] method=local has no server aggregate; skipping eval")
            return
        client_name = fl_ctx.get_identity_name()
        metrics = {"client": client_name, "round": int(round_idx), "method": self.method}
        for split in ("val", "test"):
            if not self.data_loader.get_data_for_split(split):
                continue
            core = self._evaluate_model(fl_ctx, split=split, model=self.model)
            metrics[f"{split}_auc"] = float(core["auc"])
            metrics[f"{split}_accuracy"] = float(core["accuracy"])
            metrics[f"{split}_loss"] = float(core["loss"])
            metrics[f"{split}_samples"] = int(core["samples"])
            self.log_info(
                fl_ctx,
                f"[incoming-global] site={client_name} round={int(round_idx)} method={self.method} "
                f"{split}_auc={core['auc']:.4f} n={core['samples']}"
            )
            try:
                self._dump_predictions(fl_ctx, split=split, model=self.model,
                                       round_idx=round_idx, method_tag="incoming_global",
                                       save_saliency=getattr(self, "incoming_global_saliency", False))
            except Exception as e:
                self.log_warning(fl_ctx, f"[incoming-global] preds dump ({split}) failed: {e}")
        self.incoming_global_history.append(metrics)
        # Durable per-round table so best-global selection survives a crash before END_RUN.
        try:
            persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
            os.makedirs(persistent_dir, exist_ok=True)
            out = os.path.join(
                persistent_dir,
                f"{client_name}_incoming_global_round{int(round_idx)}_metrics.json")
            with open(out, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            self.log_warning(fl_ctx, f"[incoming-global] metrics json failed: {e}")


    def _track_personal_perround(self, fl_ctx: FLContext, round_idx, model):
        """Per-round eval + pred dump of the DEPLOYED PERSONAL model (Ditto family) on val+test.

        Mirrors _track_incoming_global, but for the personal model v (not the shared global), so the
        personal model gets the SAME per-round trajectory the shared methods get from incoming_global.
        That lets best-val selection + pooled/DeLong/Youden stats be computed for the PERSONAL model
        from preds alone, with NO post-hoc per-node dump (we lack local access to every client; each
        client's own executor writes these live during the run). Tagged '<method>_perround' and logged
        under [personal-perround] to stay DISTINCT from the shared-global [incoming-global] stats.
        """
        client_name = fl_ctx.get_identity_name()
        tag = f"{self.method}_perround"
        metrics = {"client": client_name, "round": int(round_idx), "method": tag}
        for split in ("val", "test"):
            if not self.data_loader.get_data_for_split(split):
                continue
            core = self._evaluate_model(fl_ctx, split=split, model=model)
            metrics[f"{split}_auc"] = float(core["auc"])
            metrics[f"{split}_accuracy"] = float(core["accuracy"])
            metrics[f"{split}_loss"] = float(core["loss"])
            metrics[f"{split}_samples"] = int(core["samples"])
            self.log_info(
                fl_ctx,
                f"[personal-perround] site={client_name} round={int(round_idx)} method={self.method} "
                f"{split}_auc={core['auc']:.4f} n={core['samples']}"
            )
            try:
                self._dump_predictions(fl_ctx, split=split, model=model,
                                       round_idx=round_idx, method_tag=tag,
                                       save_saliency=False)
            except Exception as e:
                self.log_warning(fl_ctx, f"[personal-perround] preds dump ({split}) failed: {e}")
        if not hasattr(self, "personal_perround_history"):
            self.personal_perround_history = []
        self.personal_perround_history.append(metrics)
        # Durable per-round table (mirrors incoming-global), distinct filename.
        try:
            persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
            os.makedirs(persistent_dir, exist_ok=True)
            out = os.path.join(
                persistent_dir,
                f"{client_name}_{tag}_round{int(round_idx)}_metrics.json")
            with open(out, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            self.log_warning(fl_ctx, f"[personal-perround] metrics json failed: {e}")


    def _dump_predictions(self, fl_ctx: FLContext, split, model, round_idx, method_tag,
                          save_saliency=True):
        """Save per-sample (view-level) malignancy scores + labels + site_id to CSV so
        breast-level AUC, DeLong CIs and cross-site sigma(AUC) can be computed offline.

        save_saliency=False skips the (potentially large) per-image saliency .npz; the
        prediction CSV (the actual preds) is always written. Default True writes both.
        """
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        os.makedirs(persistent_dir, exist_ok=True)
        client = fl_ctx.get_identity_name()
        eval_model = model if model is not None else self.model
        eval_model.eval()
        rows = []
        sal_maps = []    # malignant saliency channel per image, aligned with `rows`
        sal_paths = []   # same 'path' key as the CSV -> joins map<->prediction offline
        sal_labels = []
        with torch.no_grad():
            for inputs, targets, metas in self.data_loader.get_batch_iterator(split):
                inputs = inputs.to(self.device)
                outputs = eval_model(inputs)
                probs = malignant_score(outputs).detach().cpu().numpy().reshape(-1)
                tnp = targets.detach().cpu().numpy().reshape(-1)
                # Malignant saliency channel (already a sigmoid probability), (B, h, w)
                sal_np = None
                if save_saliency:
                    _, _, _, saliency = gmic_outputs(outputs)
                    sal_np = saliency[:, 1].detach().cpu().numpy() if saliency is not None else None
                for i, meta in enumerate(metas):
                    rows.append({
                        "site_id": client,
                        "round": round_idx,
                        "method": method_tag,
                        "split": split,
                        "exam_id": meta.get("exam_id"),
                        "view": meta.get("view"),
                        "path": meta.get("path"),
                        "prob_malignant": float(probs[i]),
                        "label": int(tnp[i]),
                    })
                    if save_saliency and sal_np is not None:
                        sal_maps.append(sal_np[i].astype(np.float32))
                        sal_paths.append(meta.get("path"))
                        sal_labels.append(int(tnp[i]))
        out = os.path.join(
            persistent_dir, f"{client}_predictions_{method_tag}_round{round_idx}_{split}.csv"
        )
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "site_id", "round", "method", "split", "exam_id", "view",
                "path", "prob_malignant", "label"])
            writer.writeheader()
            writer.writerows(rows)
        self.log_info(fl_ctx, f"[predictions] wrote {len(rows)} rows -> {out}")

        # Malignant saliency maps for the cross-site heatmap comparison. Keyed by the
        # SAME 'path' identifier as the prediction CSV, so map<->prediction<->label<->
        # site<->method all join offline with no model needed. Values are post-sigmoid
        # probabilities in [0,1] (outputs[3][:,1]), never logits.
        if sal_maps:
            sal_out = os.path.join(
                persistent_dir,
                f"{client}_saliency_{method_tag}_round{round_idx}_{split}.npz",
            )
            np.savez_compressed(
                sal_out,
                saliency=np.stack(sal_maps, axis=0),   # (N, h, w) float32 in [0,1]
                path=np.array(sal_paths),              # join key (matches CSV 'path')
                label=np.array(sal_labels, dtype=np.int64),
                site_id=str(client), method=str(method_tag),
                split=str(split), round=int(round_idx),
            )
            self.log_info(fl_ctx, f"[saliency] wrote {len(sal_maps)} maps -> {sal_out}")
        return rows


    def _save_global_round(self, fl_ctx: FLContext, round_idx, state_dict):
        """Persist the round's SENT global w^t for the Ditto-consolidation replay cache.

        Distinct from the deployed/best-val artifacts: filename is
        `{client}_global_round_{t}.pth`, a plain state_dict of the round's final-epoch global
        (the aggregation contribution). Replaying this trajectory reproduces interleaved Ditto
        exactly (anchor convention: round-(t+1) personal update anchors to w^t).
        """
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        global_dir = os.path.join(persistent_dir, "global_trajectory")
        os.makedirs(global_dir, exist_ok=True)
        client_name = fl_ctx.get_identity_name()
        out = os.path.join(global_dir, f"{client_name}_global_round_{int(round_idx)}.pth")
        torch.save(state_dict, out)
        self.log_info(fl_ctx, f"[global-cache] saved sent global w^{int(round_idx)} -> {out}")
        return out

    def _save_incoming_global(self, fl_ctx: FLContext, round_idx, state_dict):
        """Persist the RECEIVED server aggregate for crash recovery.

        `state_dict` is the raw `incoming` global (the server's aggregated model from the
        prior round; at round 0, the source_ckpt seed) — byte-for-byte the artifact the
        server deletes with its transient run dir. Filename is
        `{client}_incoming_global_round_{t}.pth`, kept distinct from the SENT-global cache so
        the two are never confused.

        EVERY round is kept (no pruning): the best global by val AUC is often an earlier
        round, so the full trajectory must survive for offline model selection — not just
        the latest. This also still serves crash recovery: to resume, point the server
        persistor's `source_ckpt_file_full_name` at the highest-round file from any
        surviving client (all earlier rounds remain available too).
        """
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        recv_dir = os.path.join(persistent_dir, "incoming_global")
        os.makedirs(recv_dir, exist_ok=True)
        client_name = fl_ctx.get_identity_name()
        out = os.path.join(recv_dir, f"{client_name}_incoming_global_round_{int(round_idx)}.pth")
        torch.save(state_dict, out)
        self.log_info(fl_ctx, f"[recv-global] saved received aggregate (round {int(round_idx)}) -> {out}")
        return out

    def _save_local_model(self, fl_ctx: FLContext, shareable: Shareable, model=None, round_idx=None):
        """Save current (deployed) model state"""
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        os.makedirs(persistent_dir, exist_ok=True)

        client_name = fl_ctx.get_identity_name()
        current_round = round_idx if round_idx is not None else shareable.get_header(AppConstants.CURRENT_ROUND)

        save_model = model if model is not None else self.model
        model_path = f"{persistent_dir}/{client_name}_gmic_model_round_{current_round}.pth"
        torch.save(save_model.state_dict(), model_path)
        self.log_info(fl_ctx, f"Model saved: {model_path}")


    def _save_best_model(self, fl_ctx, metrics, shareable: Shareable, model_type="best", model=None, round_idx=None):
        """Save best performing (deployed) model (expects metrics with keys including auc)."""
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        os.makedirs(persistent_dir, exist_ok=True)

        client_name = fl_ctx.get_identity_name()
        current_round = round_idx if round_idx is not None else shareable.get_header(AppConstants.CURRENT_ROUND)

        save_model = model if model is not None else self.model
        model_path = f"{persistent_dir}/{client_name}_{model_type}_gmic_model.pth"
        torch.save(save_model.state_dict(), model_path)

        metrics_path = f"{persistent_dir}/{client_name}_{model_type}_gmic_metrics.json"
        enriched = {
            "client": client_name,
            "round": current_round,
            "model_type": model_type,
            **metrics
        }
        auc_val = metrics.get('auc')
        if isinstance(auc_val, (int, float)):
            if 'val' in model_type and 'val_auc' not in enriched:
                enriched['val_auc'] = auc_val
            if 'test' in model_type and 'test_auc' not in enriched:
                enriched['test_auc'] = auc_val

        with open(metrics_path, "w") as f:
            json.dump(enriched, f, indent=2)

        auc_val = metrics.get('auc') or metrics.get(f'{model_type}_auc') or 'N/A'
        self.log_info(fl_ctx, f"✅ {model_type} model saved (AUC={auc_val})")


    def _save_final_results(self, fl_ctx: FLContext):
        """Save final training results and metrics summary"""
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        os.makedirs(persistent_dir, exist_ok=True)

        client_name = fl_ctx.get_identity_name()

        # If initialize() failed (e.g. data loader threw), self.data_loader is None. Don't raise
        # a confusing secondary AttributeError in END_RUN that masks the real initialize() error;
        # log and return so the original traceback stays the visible one.
        if self.data_loader is None:
            self.log_warning(fl_ctx, "[END_RUN] data_loader is None (initialize failed) — skipping final results; see the earlier initialize() error")
            return

        # Final safeguard: if best_metrics are still default zeros, attempt late recovery
        if (self.best_metrics.get('val_auc', 0.0) == 0.0 or self.best_metrics.get('test_auc', 0.0) == 0.0):
            try:
                self._recover_persisted_best_metrics()
            except Exception as e:
                self.log_warning(fl_ctx, f"Late recovery of best metrics failed: {e}")

        results = {
            "client_name": client_name,
            "training_history": self.training_history,
            "validation_history": self.validation_history,
            "test_history": self.test_history,
            "incoming_global_history": self.incoming_global_history,
            "personal_perround_history": getattr(self, "personal_perround_history", []),
            "best_metrics": self.best_metrics,
            "data_splits": self.data_loader.get_split_info(),
            "class_distributions": {
                "train": self.data_loader.get_class_distribution("train"),
                "val": self.data_loader.get_class_distribution("val"),
                "test": self.data_loader.get_class_distribution("test")
            }
        }

        results_path = f"{persistent_dir}/{client_name}_final_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.log_info(fl_ctx, f"Final results saved to: {results_path}")
        self.log_info(fl_ctx, "Training Complete!")
        self.log_info(fl_ctx, f"Best Validation AUC: {self.best_metrics['val_auc']:.4f}")
        self.log_info(fl_ctx, f"Best Test AUC: {self.best_metrics['test_auc']:.4f}")


    def _recover_persisted_best_metrics(self):
        """Load previously saved best_val_overall / best_test_overall metrics JSON files to restore best_metrics.

        This is needed because the simulator may create a fresh ClientRunner (and executor instance)
        for the END_RUN phase, resetting in-memory best_metrics to 0.0. We persist metrics per round
        already (e.g., site-1_best_val_overall_gmic_metrics.json) – load them if present.
        """
        persistent_dir = getattr(self, 'results_dir', self.output_dir or "/workspace/gmic_results")
        if not os.path.isdir(persistent_dir):
            return

        try:
            best_val_file = None
            best_test_file = None
            for fname in os.listdir(persistent_dir):
                if fname.endswith('_best_val_overall_gmic_metrics.json'):
                    best_val_file = os.path.join(persistent_dir, fname)
                elif fname.endswith('_best_test_overall_gmic_metrics.json'):
                    best_test_file = os.path.join(persistent_dir, fname)

            updated = False
            if best_val_file and os.path.isfile(best_val_file):
                with open(best_val_file, 'r') as f:
                    data = json.load(f)
                auc_val = data.get('auc') or data.get('val_auc')
                if isinstance(auc_val, (int, float)) and auc_val > self.best_metrics.get('val_auc', 0.0):
                    self.best_metrics['val_auc'] = float(auc_val)
                    updated = True

            if best_test_file and os.path.isfile(best_test_file):
                with open(best_test_file, 'r') as f:
                    data = json.load(f)
                auc_val = data.get('auc') or data.get('test_auc')
                if isinstance(auc_val, (int, float)) and auc_val > self.best_metrics.get('test_auc', 0.0):
                    self.best_metrics['test_auc'] = float(auc_val)
                    updated = True

            if updated:
                self._logger.info("[EXEC] Recovered persisted best metrics: %s", self.best_metrics)

        except Exception as e:
            self._logger.warning("[EXEC][WARN] Could not recover best metrics: %s", e, exc_info=True)
