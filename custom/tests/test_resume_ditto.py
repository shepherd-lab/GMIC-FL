# ============================================================================
# test_resume_ditto.py - crash-resume correctness for the Ditto family
# ----------------------------------------------------------------------------
# The resume path was built for the shared-model methods (fedavg/fedprox/fedbn),
# where the DEPLOYED model is the global w, so `{client}_gmic_model_round_N.pth`
# is the right thing to re-aggregate. For Ditto-family the deployed (and saved)
# model is the PERSONAL model v, which makes that same file wrong for the reseed
# and simultaneously the ONLY on-disk copy of v. These tests pin both halves:
#
#   1. Ditto reseed re-aggregates the SENT GLOBAL (global_trajectory/), never the
#      personal checkpoint -- otherwise the "global" is an average of three sites'
#      personal models.
#   2. Non-Ditto reseed still reads the plain deployed checkpoint (no regression).
#   3. On resume, v is RESTORED from its round-N checkpoint rather than lazily
#      cold-started from the pretrained init (the silent-desync bug).
#   4. A missing personal checkpoint RAISES instead of cold-starting v.
#
# Reuses the synthetic-loader harness from test_executor_integration.
# Run from custom/tests/:   python -m pytest test_resume_ditto.py -q
# ============================================================================
import os
import copy
import tempfile

import pytest
import torch

from test_executor_integration import (  # noqa: E402  (installs the `resource` stub on import)
    GMIC_PARAMS, make_executor, make_shareable, assert_valid_reply,
)
from nvflare.apis.fl_context import FLContext  # noqa: E402
from model.gmic import GMIC                    # noqa: E402

CLIENT = "site-1"
ROUND_N = 20
PROBE = "fusion_dnn.weight"  # any real parameter; used to tell the two saved models apart


class _Ctx(FLContext):
    """FLContext whose identity is a fixed client name (drives the ckpt filenames)."""

    def get_identity_name(self):
        return CLIENT


def _distinct_model(seed):
    torch.manual_seed(seed)
    m = GMIC(copy.deepcopy(GMIC_PARAMS))
    # Force a clearly distinguishable probe parameter so "which file was loaded" is unambiguous.
    with torch.no_grad():
        dict(m.named_parameters())[PROBE].fill_(float(seed))
    return m


def _seed_resume_dir(td, personal, sent_global):
    """Write the two artifacts a crashed Ditto run leaves behind, as the executor names them."""
    torch.save(personal.state_dict(), os.path.join(td, f"{CLIENT}_gmic_model_round_{ROUND_N}.pth"))
    gdir = os.path.join(td, "global_trajectory")
    os.makedirs(gdir, exist_ok=True)
    torch.save(sent_global.state_dict(), os.path.join(gdir, f"{CLIENT}_global_round_{ROUND_N}.pth"))


def _probe(sd_or_model):
    if hasattr(sd_or_model, "state_dict"):
        sd_or_model = sd_or_model.state_dict()
    v = sd_or_model[PROBE]
    return float(torch.as_tensor(v).flatten()[0])


def test_ditto_reseed_reaggregates_sent_global_not_personal():
    """The Ditto reseed must ship w^N (global_trajectory/), NOT the personal v checkpoint."""
    for method in ("ditto", "ditto_modulewise"):
        with tempfile.TemporaryDirectory() as td:
            personal, sent_global = _distinct_model(3), _distinct_model(7)
            _seed_resume_dir(td, personal, sent_global)
            ex = make_executor(method, td, resume_from_local_round=ROUND_N, resume_ckpt_dir=td)
            sh = make_shareable(_distinct_model(9), round_idx=0, total_rounds=5)
            reply = ex._reseed_round(_Ctx(), sh)
            fl_out = assert_valid_reply(reply, method)
            assert _probe(fl_out.params) == pytest.approx(7.0), (
                f"{method}: reseed shipped the personal model (or the broadcast) instead of the "
                f"sent global w^{ROUND_N} -- aggregating this yields a meaningless global"
            )
            assert _probe(ex._underlying) == pytest.approx(7.0)
    print("[resume] ditto reseed re-aggregates the sent global, not v. OK")


def test_nonditto_reseed_reads_deployed_ckpt():
    """No regression: for shared-model methods deployed == w, so the plain ckpt stays correct."""
    with tempfile.TemporaryDirectory() as td:
        deployed, decoy = _distinct_model(3), _distinct_model(7)
        _seed_resume_dir(td, deployed, decoy)  # decoy global_trajectory must be IGNORED here
        ex = make_executor("fedavg", td, resume_from_local_round=ROUND_N, resume_ckpt_dir=td)
        sh = make_shareable(_distinct_model(9), round_idx=0, total_rounds=5)
        fl_out = assert_valid_reply(ex._reseed_round(_Ctx(), sh), "fedavg")
        assert _probe(fl_out.params) == pytest.approx(3.0), \
            "fedavg reseed changed behavior; it must still read the deployed round ckpt"
    print("[resume] fedavg reseed unchanged. OK")


def test_resume_restores_personal_model_v():
    """v is reloaded from its round-N ckpt, not cold-started from the incoming/pretrained w."""
    for method in ("ditto", "ditto_modulewise"):
        with tempfile.TemporaryDirectory() as td:
            personal, sent_global = _distinct_model(3), _distinct_model(7)
            _seed_resume_dir(td, personal, sent_global)
            ex = make_executor(method, td, resume_from_local_round=ROUND_N, resume_ckpt_dir=td)
            incoming = {k: v.clone() for k, v in _distinct_model(9).state_dict().items()}
            ex._consume_global(incoming, _Ctx())  # first non-reseed round: v is created here
            assert ex._v_model is not None
            assert _probe(ex._v_model) == pytest.approx(3.0), (
                f"{method}: v was cold-started from the pretrained/incoming w instead of restored "
                f"from its round-{ROUND_N} checkpoint -- v would lag the global by {ROUND_N} rounds"
            )
            # the shared global still tracks the broadcast, and v stayed distinct from it
            assert _probe(ex._underlying) == pytest.approx(9.0)
            assert ex._v_optimizer is not None, "v optimizer must be rebuilt after restore"
    print("[resume] personal model v restored from its round-N checkpoint. OK")


def test_missing_personal_ckpt_raises_rather_than_cold_starting():
    """A silent cold v is the exact bug this guards; absence must fail loudly."""
    with tempfile.TemporaryDirectory() as td:
        _seed_resume_dir(td, _distinct_model(3), _distinct_model(7))
        os.remove(os.path.join(td, f"{CLIENT}_gmic_model_round_{ROUND_N}.pth"))
        ex = make_executor("ditto_modulewise", td, resume_from_local_round=ROUND_N,
                           resume_ckpt_dir=td)
        incoming = {k: v.clone() for k, v in _distinct_model(9).state_dict().items()}
        with pytest.raises(FileNotFoundError, match="personal model checkpoint not found"):
            ex._consume_global(incoming, _Ctx())
    print("[resume] missing personal ckpt raises. OK")


def test_fresh_run_still_cold_starts_v():
    """resume_from_local_round=-1 (fresh run) must be untouched: v starts from w."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td)  # default resume_from_local_round=-1
        incoming = {k: v.clone() for k, v in _distinct_model(9).state_dict().items()}
        ex._consume_global(incoming, _Ctx())
        assert _probe(ex._v_model) == pytest.approx(9.0), \
            "fresh run must still initialize v from the consumed global w"
    print("[resume] fresh run cold-starts v from w. OK")


if __name__ == "__main__":
    test_ditto_reseed_reaggregates_sent_global_not_personal()
    test_nonditto_reseed_reads_deployed_ckpt()
    test_resume_restores_personal_model_v()
    test_missing_personal_ckpt_raises_rather_than_cold_starting()
    test_fresh_run_still_cold_starts_v()
    print("all resume tests passed")
