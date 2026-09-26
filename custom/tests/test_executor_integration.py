# ============================================================================
# test_executor_integration.py - end-to-end CPU smoke of the executor per method
# ----------------------------------------------------------------------------
# Drives GMICFederatedExecutor.execute() with a tiny SYNTHETIC data loader and a
# bare FLContext, on CPU, for every `method`. Requires nvflare + torch.
# Run from custom/tests/:   python test_executor_integration.py
#
# Covers brief tests:
#   4. CPU smoke per method (one round completes, returns a valid Shareable)
#   5. Ditto bookkeeping (v + its optimizer persist across two rounds; not reset)
#   6. No-regression for fedavg (full global load, no prox, sends full w state)
#
# Portability note: NVFLARE imports the Unix-only `resource` module, which is
# absent on Windows. We inject a tiny stub below so this runs in the Windows
# clone; on the restricted Linux machine the stub is simply ignored.
# Uses full-size 2944x1920 synthetic inputs (the GMIC global net asserts
# cam_size==(46,30)); a few CPU forward/backward passes per method make this
# take ~1-3 min.
# ============================================================================
import os
import sys
import copy
import types
import logging
import tempfile

# ---- make NVFLARE importable on Windows (no-op on Linux) -------------------
if "resource" not in sys.modules:
    try:
        import resource  # noqa: F401
    except Exception:
        _r = types.ModuleType("resource")
        _r.RLIMIT_NOFILE = 7
        _r.RLIMIT_CORE = 4
        _r.getrlimit = lambda *a: (1024, 4096)
        _r.setrlimit = lambda *a: None
        _r.getrusage = lambda *a: types.SimpleNamespace(ru_maxrss=0)
        _r.RUSAGE_SELF = 0
        sys.modules["resource"] = _r

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)                     # this tests/ dir (sibling test imports)
sys.path.insert(0, os.path.dirname(THIS_DIR))    # custom/ (bc_executor, fl_utils, model.*, ...)

logging.basicConfig(level=logging.ERROR)  # keep executor's info logs quiet

from nvflare.apis.fl_context import FLContext            # noqa: E402
from nvflare.apis.signal import Signal                    # noqa: E402
from nvflare.apis.shareable import Shareable              # noqa: E402
from nvflare.app_common.app_constant import AppConstants  # noqa: E402
from nvflare.app_common.utils.fl_model_utils import (     # noqa: E402
    FLModel, FLModelUtils, ParamsType,
)

import bc_executor as BE                                   # noqa: E402
from model.gmic import GMIC                                # noqa: E402
from train.training_core import configure_optimizers       # noqa: E402

GMIC_PARAMS = {
    "device_type": "cpu", "gpu_number": 0,
    "max_crop_noise": (100, 100), "max_crop_size_noise": 100,
    "image_path": "x", "cam_size": (46, 30), "K": 6,
    "crop_shape": (256, 256), "post_processing_dim": 256,
    "num_classes": 2, "use_v1_global": False,
    "percent_t": 0.02, "lambda_l1": 1e-5,
}


class FakeLoader:
    """Minimal synthetic data loader matching GMICDataLoader's iterator contract."""
    input_format = "pkl"

    def __init__(self):
        self._x = torch.randn(1, 1, 2944, 1920)
        self._splits = {"train": 2, "val": 2, "test": 2}

    def get_data_for_split(self, split):
        return [{"i": i} for i in range(self._splits.get(split, 0))]

    def get_batch_iterator(self, split):
        n = self._splits.get(split, 0)
        for i in range(n):
            target = torch.tensor([i % 2], dtype=torch.long)
            meta = [{"exam_id": i, "view": "L-CC", "path": f"/fake/{split}_{i}.png"}]
            yield (self._x.clone(), target, meta)

    def get_split_info(self):
        return {"train_size": 2, "val_size": 2, "test_size": 2}

    def get_class_distribution(self, split="train"):
        return {"exam_level": {0: 1, 1: 1}, "view_level": {0: 1, 1: 1}, "total": 2}

    def print_summary(self):
        pass


class _OA:  # optimizer args for configure_optimizers
    lr_heads = 1e-4
    lr_backbone = 1e-5
    weight_decay = 1e-5
    patience = 4


def make_executor(method, results_dir, **over):
    ex = BE.GMICFederatedExecutor(
        method=method, epochs=1, batch_size=1, device="cpu", loss="gmic", pos_weight=3.0,
        patience=4, results_dir=results_dir, output_dir=results_dir, **over,
    )
    ex._logger = logging.getLogger("test")
    model = GMIC(copy.deepcopy(GMIC_PARAMS))
    ex.model = model
    ex._underlying = model
    ex.device = torch.device("cpu")
    ex.criterion = nn.CrossEntropyLoss()
    oa = _OA()
    ex.optimizer, ex.scheduler, ex.early_stopper = configure_optimizers(model, oa)
    ex._opt_args = oa
    ex.data_loader = FakeLoader()
    ex._pretrained_report = {"matched": 258, "missing": 1, "unexpected": 1}
    return ex


def make_shareable(model, round_idx, total_rounds):
    params = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    sh = FLModelUtils.to_shareable(FLModel(params=params, params_type=ParamsType.FULL))
    sh.set_header(AppConstants.CURRENT_ROUND, round_idx)
    sh.set_header(AppConstants.NUM_ROUNDS, total_rounds)
    return sh


def run_round(ex, global_model, round_idx, total_rounds):
    sh = make_shareable(global_model, round_idx, total_rounds)
    fl_ctx = FLContext()
    return ex.execute("train", sh, fl_ctx, Signal())


def assert_valid_reply(reply, method):
    assert isinstance(reply, Shareable), f"{method}: reply not a Shareable"
    rc = reply.get_return_code()
    assert rc == "OK", f"{method}: return code = {rc}"
    fl_out = FLModelUtils.from_shareable(reply)
    assert fl_out is not None and fl_out.params, f"{method}: no params in reply"
    return fl_out


def test_smoke_all_methods():
    methods = ["local", "fedavg", "fedprox", "fedbn", "ditto", "ditto_modulewise"]
    srv = GMIC(copy.deepcopy(GMIC_PARAMS))  # stand-in for the server global model
    for m in methods:
        with tempfile.TemporaryDirectory() as td:
            ex = make_executor(m, td)
            reply = run_round(ex, srv, round_idx=0, total_rounds=1)  # round0 == final
            fl_out = assert_valid_reply(reply, m)
            assert len(fl_out.params) == len(ex._underlying.state_dict()), \
                f"{m}: sent {len(fl_out.params)} keys, expected full state"
            files = os.listdir(td)
            assert any("pretrained_baseline" in f for f in files), f"{m}: no baseline dump"
            assert any(f"_{m}_round0" in f and f.endswith(".csv") for f in files), \
                f"{m}: no final prediction dump"
            # saliency maps exported alongside predictions (keyed by the same path)
            assert any("saliency" in f and f.endswith(".npz") for f in files), \
                f"{m}: no saliency npz dump"
            print(f"[smoke] method={m:17s} OK ({len(fl_out.params)} keys sent, "
                  f"{sum(f.endswith('.csv') for f in files)} csv + "
                  f"{sum(f.endswith('.npz') for f in files)} npz dumps)")
    print("[smoke] all methods OK")


def test_fedavg_no_regression():
    """fedavg: full global load, no proximal term, sends full w state == underlying."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("fedavg", td)
        srv = GMIC(copy.deepcopy(GMIC_PARAMS))
        reply = run_round(ex, srv, round_idx=0, total_rounds=1)
        fl_out = assert_valid_reply(reply, "fedavg")
        assert ex._prox_ref is None, "fedavg must not set a proximal reference"
        assert ex._v_model is None, "fedavg must not create a personal model"
        sd = ex._underlying.state_dict()
        assert set(fl_out.params.keys()) == set(sd.keys())
    print("[no-regression] fedavg full load, no prox, sends full w state. OK")


def test_ditto_persistence_across_rounds():
    """v and its optimizer persist across two rounds and v is not reset to global."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto", td)
        srv0 = GMIC(copy.deepcopy(GMIC_PARAMS))
        run_round(ex, srv0, round_idx=0, total_rounds=2)
        v_id_r0 = id(ex._v_model)
        opt_id_r0 = id(ex._v_optimizer)
        v_param_r0 = dict(ex._v_model.named_parameters())["fusion_dnn.weight"].detach().clone()
        assert ex._v_model is not None and ex._v_optimizer is not None

        srv1 = GMIC(copy.deepcopy(GMIC_PARAMS))  # a DIFFERENT global arrives
        run_round(ex, srv1, round_idx=1, total_rounds=2)
        assert id(ex._v_model) == v_id_r0, "v was re-created (should persist)"
        assert id(ex._v_optimizer) == opt_id_r0, "v optimizer was re-created (should persist)"
        v_param_r1 = dict(ex._v_model.named_parameters())["fusion_dnn.weight"].detach().clone()
        assert not torch.allclose(v_param_r0, v_param_r1), "v did not continue training"
        w_param = dict(ex._underlying.named_parameters())["fusion_dnn.weight"].detach()
        assert not torch.allclose(v_param_r1, w_param), "v collapsed onto global w"
    print("[ditto] personal model + optimizer persist across rounds; v != w. OK")


def test_amp_path():
    """B1 end-to-end guard: a use_amp=True round must complete. The GMIC forward runs
    fine under autocast; the bug was the loss (BCE) and malignant_score().numpy() on
    autocast (bf16) outputs. Every other test runs use_amp=False, so this is the guard
    that the AMP regression can't return.
    """
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("fedavg", td, use_amp=True)
        srv = GMIC(copy.deepcopy(GMIC_PARAMS))
        reply = run_round(ex, srv, round_idx=0, total_rounds=1)
        assert_valid_reply(reply, "fedavg(use_amp=True)")
    print("[amp] use_amp=True round completed (forward autocast / loss+score fp32). OK")


def test_loss_validation():
    """A1 guard: resolve_criterion accepts the valid loss names and REJECTS unknown ones
    (so a typo'd/invalid `loss` fails loudly instead of silently routing to a default).
    Notably 'bce' is NOT valid and must raise -- it used to silently route to the GMIC loss.
    """
    assert BE.resolve_criterion("gmic") is None
    assert BE.resolve_criterion("gmic_bce") is None
    assert isinstance(BE.resolve_criterion("cross_entropy"), nn.CrossEntropyLoss)
    assert isinstance(BE.resolve_criterion("CE"), nn.CrossEntropyLoss)   # case-insensitive
    # Empty / None = "unspecified" -> safe default to gmic (mirrors method defaulting), not an error.
    assert BE.resolve_criterion("") is None
    assert BE.resolve_criterion(None) is None
    # Genuinely unknown loss names must raise (no silent fallback; 'bce' notably is NOT valid).
    for bad in ("bce", "bce_with_logits", "mse", "gmicc"):
        raised = False
        try:
            BE.resolve_criterion(bad)
        except ValueError:
            raised = True
        assert raised, f"resolve_criterion({bad!r}) should raise ValueError but did not"
    print("[loss-validation] valid names resolve; unknown (incl. 'bce') raise ValueError. OK")


ALL = [
    test_smoke_all_methods,
    test_fedavg_no_regression,
    test_ditto_persistence_across_rounds,
    test_amp_path,
    test_loss_validation,
]

if __name__ == "__main__":
    failures = 0
    for t in ALL:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
    print("\n==== %d/%d integration tests passed ====" % (len(ALL) - failures, len(ALL)))
    sys.exit(1 if failures else 0)
