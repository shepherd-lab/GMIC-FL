# ============================================================================
# test_ditto_amp_personal.py - the Ditto personal pass follows use_amp
# ----------------------------------------------------------------------------
# The personal pass used to run plain fp32 while the main w pass ran under autocast, so it needed
# ~2x the activation memory of the pass immediately before it -- a hard hardware floor that OOMs
# on T4/L4-class cards. It now follows `use_amp` like every other forward in the executor.
#
# These run on CPU, where autocast is a no-op for memory but still exercises the control flow
# (scaler branch, unscale-before-clip ordering, the non-finite guard, and v still learning).
# The memory claim itself is not testable here; it is pinned by the config + docstring.
#
# Run from custom/tests/:   python -m pytest test_ditto_amp_personal.py -q
# ============================================================================
import contextlib
import copy
import tempfile
from unittest import mock

import torch

from test_executor_integration import (  # noqa: E402  (installs the `resource` stub on import)
    GMIC_PARAMS, make_executor, run_round,
)
from model.gmic import GMIC  # noqa: E402

PROBE = "fusion_dnn.weight"


def _v_probe(ex):
    return dict(ex._v_model.named_parameters())[PROBE].detach().clone()


def test_personal_pass_has_its_own_scaler():
    """v needs a DEDICATED scaler: a shared one would couple the two optimizers' scale schedules.

    Both scalers are None on a CPU-only box (they are built only when cuda is available), which
    would make an identity assertion vacuously pass. Pretend cuda exists for construction so the
    two objects are actually created and can be compared.
    """
    with tempfile.TemporaryDirectory() as td, mock.patch.object(torch.cuda, "is_available",
                                                                return_value=True):
        ex = make_executor("ditto_modulewise", td, use_amp=True)
        assert ex._scaler is not None and ex._v_scaler is not None, "scalers were not constructed"
        assert ex._v_scaler is not ex._scaler, "v must not share the main pass's GradScaler"
    print("[amp] personal pass has a dedicated GradScaler. OK")


def test_personal_pass_trains_under_amp():
    """A use_amp=True Ditto round completes and v actually updates (the B1-style end-to-end guard)."""
    for method in ("ditto", "ditto_modulewise"):
        with tempfile.TemporaryDirectory() as td:
            ex = make_executor(method, td, use_amp=True)
            srv = GMIC(copy.deepcopy(GMIC_PARAMS))
            run_round(ex, srv, round_idx=0, total_rounds=2)
            before = _v_probe(ex)
            run_round(ex, srv, round_idx=1, total_rounds=2)
            after = _v_probe(ex)
            assert torch.isfinite(after).all(), f"{method}: v went non-finite under AMP"
            assert not torch.allclose(before, after), f"{method}: v stopped learning under AMP"
    print("[amp] personal pass trains v under use_amp=True. OK")


def _personal_pass_output_dtypes(ex, srv, use_amp_expected):
    """Run one Ditto round, recording the dtypes v's forward produced in the personal pass.

    Hooks _compute_task_loss, which the personal pass calls with the raw forward outputs. The main
    w pass calls it too, so the recorder is armed only once the personal pass has started.
    """
    seen = []
    original = ex._compute_task_loss

    def recording_loss(outputs, targets):
        if ex._recording:
            stack = [outputs]
            while stack:
                o = stack.pop()
                if torch.is_tensor(o):
                    seen.append(o.dtype)
                elif isinstance(o, dict):
                    stack.extend(o.values())
                elif isinstance(o, (list, tuple)):
                    stack.extend(o)
        return original(outputs, targets)

    ex._recording = False
    ex._compute_task_loss = recording_loss
    original_personal = ex._train_personal_model

    def arming_personal(fl_ctx, abort_signal):
        ex._recording = True
        try:
            return original_personal(fl_ctx, abort_signal)
        finally:
            ex._recording = False

    ex._train_personal_model = arming_personal
    run_round(ex, srv, round_idx=0, total_rounds=2)
    assert seen, "personal pass never reached the loss (nothing recorded)"
    return seen


def test_personal_forward_actually_runs_under_autocast():
    """The behavioral claim: with use_amp=True v's FORWARD runs in reduced precision.

    This is what buys the ~2x activation-memory reduction. Under CPU autocast the reduced dtype is
    bfloat16, so seeing any non-fp32 output dtype proves the forward was inside the autocast
    context -- the old code ran this forward in plain fp32 and would fail here.
    """
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=True)
        dtypes = _personal_pass_output_dtypes(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), True)
        assert any(d != torch.float32 for d in dtypes), (
            f"personal forward produced only {set(dtypes)} -- it did NOT run under autocast, so it "
            f"still costs full fp32 activation memory")
    print(f"[amp] personal forward runs under autocast (dtypes={set(dtypes)}). OK")


def test_personal_forward_is_fp32_when_amp_off():
    """Mirror image: use_amp=False must still produce a genuinely fp32 forward."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=False)
        dtypes = _personal_pass_output_dtypes(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), False)
        assert all(d == torch.float32 for d in dtypes), \
            f"use_amp=False personal forward was not fp32 (got {set(dtypes)})"
    print("[amp] personal forward is fp32 when use_amp=False. OK")


def test_stage_sync_does_not_change_training():
    """stage_sync only inserts torch.cuda.synchronize(); v must still train identically.

    Guards the diagnostic against becoming a behavior change: syncing is a scheduling barrier,
    never a numerical one, so the same seed must give the same v.
    """
    probes = {}
    for flag in (False, True):
        with tempfile.TemporaryDirectory() as td:
            torch.manual_seed(1234)
            ex = make_executor("ditto_modulewise", td, use_amp=True, stage_sync=flag)
            torch.manual_seed(1234)
            run_round(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), round_idx=0, total_rounds=1)
            probes[flag] = _v_probe(ex)
    assert torch.allclose(probes[False], probes[True], atol=1e-6), \
        "stage_sync altered v -- a diagnostic must not change the math"
    print("[amp] stage_sync is behavior-neutral. OK")


def test_personal_pass_sets_stage_labels():
    """The heartbeat's `stage` must track where the personal pass is, so a wedge is localizable.

    The fine-grained per-AMP-op stages (unscale/clip/scaler-update) were scaffolding for the fp16
    hang diagnosis and were collapsed when the pass was micro-chunked; the live stages are now
    load / h2d / forward/backward / prox / step. This pins that the compute stages are set: the
    task-loss call happens under 'forward/backward', and _sync (after the step) under 'step'.
    """
    import bc_executor as BE
    seen, holder = [], {}
    real_hb = BE.GMICFederatedExecutor._heartbeat

    @contextlib.contextmanager
    def capturing_hb(self, label, state):
        holder["state"] = state  # the dict the personal loop mutates
        with real_hb(self, label, state) as s:
            yield s

    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            ex = make_executor("ditto_modulewise", td, use_amp=True, heartbeat_interval_s=0)
        # capture the live stage at the task-loss call (forward/backward) and at _sync (step).
        # eval phases enter a heartbeat too but their state has no 'stage' key -> None, harmless.
        orig_loss = ex._compute_task_loss
        ex._compute_task_loss = lambda o, t: (
            seen.append(holder.get("state", {}).get("stage")), orig_loss(o, t))[1]
        real_sync = ex._sync
        ex._sync = lambda: (seen.append(holder.get("state", {}).get("stage")), real_sync())[1]
        BE.GMICFederatedExecutor._heartbeat = capturing_hb
        try:
            run_round(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), round_idx=0, total_rounds=1)
        finally:
            BE.GMICFederatedExecutor._heartbeat = real_hb
    for expected in ("forward/backward", "step"):
        assert expected in seen, f"stage '{expected}' never set; got {set(x for x in seen if x)}"
    print(f"[amp] personal pass sets stage labels: {sorted(set(x for x in seen if x))}. OK")


def test_personal_amp_false_forces_fp32_while_main_stays_amp():
    """The A100 revert path: personal_amp=False must run v's forward in fp32 even with use_amp=True.

    This is the one-line fallback if AMP can't run the personal pass on a given card -- the main
    pass keeps AMP, the personal pass returns to the validated fp32 recipe. Asserts on the forward
    dtypes so it tests actual precision, not just that a flag was stored.
    """
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=True, personal_amp=False)
        assert ex._personal_amp is False
        dtypes = _personal_pass_output_dtypes(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), False)
        assert all(d == torch.float32 for d in dtypes), \
            f"personal_amp=False still ran a non-fp32 forward (got {set(dtypes)})"
    print("[amp] personal_amp=False forces fp32 while main keeps AMP. OK")


def test_personal_amp_none_follows_use_amp():
    """Default (None) must track use_amp so existing configs are unchanged."""
    with tempfile.TemporaryDirectory() as td:
        ex_amp = make_executor("ditto_modulewise", td, use_amp=True)  # personal_amp defaults None
        assert ex_amp._personal_amp is True
    with tempfile.TemporaryDirectory() as td:
        ex_fp32 = make_executor("ditto_modulewise", td, use_amp=False)
        assert ex_fp32._personal_amp is False
    print("[amp] personal_amp=None follows use_amp. OK")


def test_personal_amp_by_site_resolves_per_identity():
    """One config, per-site precision: the override map picks fp32 for RSNA-AU, AMP for others.

    This is the whole point of the map -- RSNA-AU's A4000 faults on the fp16 backward, so it must run
    fp32, while RSNA/HIPIMR keep AMP, WITHOUT duplicating the app per site. Resolution keys on the FL
    identity and rebuilds the v scaler to match.
    """
    from nvflare.apis.fl_context import FLContext

    class Ctx(FLContext):
        def __init__(self, name):
            super().__init__()
            self._name = name

        def get_identity_name(self):
            return self._name

    # RSNA-AU is in the map -> fp32; a site NOT in the map falls back to the default (use_amp=True).
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=True,
                           personal_amp_by_site={"RSNA-AU": False})
        ex._resolve_personal_amp(Ctx("RSNA-AU"))
        assert ex._personal_amp is False, "RSNA-AU should resolve to fp32 via the override"
        assert ex._personal_amp_resolved is True
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=True,
                           personal_amp_by_site={"RSNA-AU": False})
        ex._resolve_personal_amp(Ctx("RSNA-US"))
        assert ex._personal_amp is True, "an unlisted site should keep the AMP default"
    print("[amp] per-site override: RSNA-AU->fp32, others->AMP, from one config. OK")


def test_personal_amp_override_drives_a_genuinely_fp32_forward():
    """End-to-end: with the RSNA-AU override resolved, the personal forward is actually fp32.

    Resolves as RSNA-AU first (memoized), then runs the pass -- the resolved fp32 must reach the
    kernels, not just the flag. Asserts on the forward dtypes.
    """
    from nvflare.apis.fl_context import FLContext

    class Ctx(FLContext):
        def get_identity_name(self):
            return "RSNA-AU"

    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=True,
                           personal_amp_by_site={"RSNA-AU": False})
        ex._resolve_personal_amp(Ctx())  # memoized -> the later run_round reuses it
        assert ex._personal_amp is False and ex._personal_amp_resolved is True
        dtypes = _personal_pass_output_dtypes(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), False)
        assert all(d == torch.float32 for d in dtypes), \
            f"RSNA-AU personal forward was not fp32 despite the override (got {set(dtypes)})"
    print("[amp] per-site fp32 override drives a genuinely fp32 forward. OK")


def test_fp32_personal_pass_still_supported():
    """use_amp=False keeps the original fp32 path (sim jobs / A100 reruns are unaffected)."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=False)
        srv = GMIC(copy.deepcopy(GMIC_PARAMS))
        run_round(ex, srv, round_idx=0, total_rounds=2)
        before = _v_probe(ex)
        run_round(ex, srv, round_idx=1, total_rounds=2)
        after = _v_probe(ex)
        assert torch.isfinite(after).all()
        assert not torch.allclose(before, after), "v stopped learning in the fp32 path"
    print("[amp] fp32 personal pass unchanged when use_amp=False. OK")


def test_nonfinite_loss_still_skips_step():
    """The persist-across-rounds poison guard must survive the AMP refactor."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=True)
        srv = GMIC(copy.deepcopy(GMIC_PARAMS))
        run_round(ex, srv, round_idx=0, total_rounds=2)  # creates v + ref
        before = _v_probe(ex)
        # Force every task loss non-finite; v must be left exactly as-is, not poisoned.
        ex._compute_task_loss = lambda outputs, targets: torch.tensor(float("nan"))
        ex._train_personal_model(__import__("nvflare.apis.fl_context", fromlist=["FLContext"]).FLContext(),
                                 __import__("nvflare.apis.signal", fromlist=["Signal"]).Signal())
        after = _v_probe(ex)
        assert torch.equal(before, after), "non-finite loss poisoned v (guard broken)"
    print("[amp] non-finite loss skips the step; v untouched. OK")


if __name__ == "__main__":
    test_personal_pass_has_its_own_scaler()
    test_personal_forward_actually_runs_under_autocast()
    test_personal_forward_is_fp32_when_amp_off()
    test_personal_amp_false_forces_fp32_while_main_stays_amp()
    test_personal_amp_none_follows_use_amp()
    test_personal_amp_by_site_resolves_per_identity()
    test_personal_amp_override_drives_a_genuinely_fp32_forward()
    test_personal_pass_trains_under_amp()
    test_fp32_personal_pass_still_supported()
    test_nonfinite_loss_still_skips_step()
    print("all AMP personal-pass tests passed")
