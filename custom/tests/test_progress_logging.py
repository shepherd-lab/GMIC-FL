# ============================================================================
# test_progress_logging.py - the round must never go silent on a remote site
# ----------------------------------------------------------------------------
# A module-wise Ditto run stalled inside the Ditto personal pass for ~18 hours and the log showed
# NOTHING after "[ditto] personal pass: ...". The personal pass logged only at epoch end, so on a
# site we cannot shell into there was no way to tell a hang from normal work. Two fixes, pinned
# here:
#
#   1. per-batch progress in the personal pass (mirrors the main loop's train_log_batch_interval)
#   2. a background heartbeat THREAD that keeps logging even while the main thread is blocked --
#      the only mechanism that reports from inside a wedged CUDA kernel or a stalled read
#
# The heartbeat is log-only by design: a slow-but-healthy site must never be killed by a watchdog.
#
# Run from custom/tests/:   python -m pytest test_progress_logging.py -q
# ============================================================================
import copy
import logging
import tempfile
import time


from test_executor_integration import (  # noqa: E402  (installs the `resource` stub on import)
    GMIC_PARAMS, make_executor, run_round,
)
from model.gmic import GMIC  # noqa: E402


class _Capture(logging.Handler):
    """Collects formatted records so tests can assert on what actually reached the log."""

    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        # getMessage() already interpolates record.args -- re-applying % here would raise and
        # silently degrade every assertion to the raw format string (no interpolated values).
        self.lines.append(record.getMessage())


def _attach(ex):
    cap = _Capture()
    ex._logger = logging.getLogger(f"capture-{id(cap)}")
    ex._logger.handlers = [cap]
    ex._logger.setLevel(logging.DEBUG)
    ex._logger.propagate = False
    return cap


def test_heartbeat_fires_while_the_main_thread_is_blocked():
    """The core claim: output continues even when the worker thread is stuck.

    Per-batch logging cannot do this -- it only emits between batches. This simulates the exact
    failure (a call that never returns) and asserts the log still shows where we are.
    """
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, heartbeat_interval_s=1)
        cap = _attach(ex)
        state = {"site": "RSNA-AU", "batch": 37, "stage": "forward", "t_last_progress": time.time()}
        with ex._heartbeat("ditto-personal", state):
            time.sleep(2.5)  # stand-in for a blocked CUDA kernel / stalled mount read
        beats = [l for l in cap.lines if "[heartbeat]" in l]
        assert beats, "heartbeat produced NO output while the main thread was blocked"
        assert any("batch=37" in l for l in beats), f"heartbeat lost the position: {beats}"
        assert any("RSNA-AU" in l for l in beats), f"heartbeat lost the site: {beats}"
        # the stage is what separates an I/O stall from a compute stall remotely
        assert any("stage=forward" in l for l in beats), f"heartbeat lost the stage: {beats}"
    print(f"[progress] heartbeat fired {len(beats)}x from a blocked thread. OK")


def test_heartbeat_stops_after_the_phase_and_never_raises():
    """It must not leak a thread, and must never abort the phase it observes."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, heartbeat_interval_s=1)
        cap = _attach(ex)
        with ex._heartbeat("evaluate", {"site": "RSNA-AU", "t_last_progress": time.time()}):
            time.sleep(1.5)
        n_after_exit = len([l for l in cap.lines if "[heartbeat]" in l])
        time.sleep(2.5)  # would emit more beats if the thread had leaked
        assert len([l for l in cap.lines if "[heartbeat]" in l]) == n_after_exit, \
            "heartbeat thread kept running after the phase ended"
    print("[progress] heartbeat stops cleanly at phase exit. OK")


def test_heartbeat_disabled_by_zero_interval():
    """0 disables it entirely (and the context manager still works)."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, heartbeat_interval_s=0)
        cap = _attach(ex)
        with ex._heartbeat("ditto-personal", {"site": "RSNA-AU", "t_last_progress": time.time()}):
            time.sleep(1.5)
        assert not [l for l in cap.lines if "[heartbeat]" in l], "heartbeat ran while disabled"
    print("[progress] heartbeat_interval_s=0 disables it. OK")


def test_personal_pass_logs_every_batch():
    """The regression that started this: the personal pass must report per-batch progress.

    Asserts on the batch lines specifically -- the old code emitted only the epoch-end summary,
    so this fails against it.
    """
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, train_log_batch_interval=1)
        seen = []
        original = ex.log_info
        ex.log_info = lambda ctx, msg: (seen.append(msg), original(ctx, msg))[1]
        run_round(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), round_idx=0, total_rounds=1)
        batch_lines = [m for m in seen if "personal epoch" in m and "batch" in m]
        assert batch_lines, f"personal pass logged no per-batch progress; got {[m for m in seen if 'ditto' in m]}"
        assert any("s)" in m for m in batch_lines), \
            f"per-batch lines carry no elapsed time, so 'slow' is unquantifiable: {batch_lines}"
        summary = [m for m in seen if "personal epoch" in m and "done" in m]
        assert summary and "s/batch" in summary[0], f"epoch summary lacks timing: {summary}"
    print(f"[progress] personal pass logged {len(batch_lines)} per-batch lines + timed summary. OK")


def test_evaluate_is_covered_by_a_heartbeat():
    """evaluate_model is one opaque call; a stall inside it must still surface."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, heartbeat_interval_s=1)
        cap = _attach(ex)
        import train.training_core as tc
        original = tc.evaluate_model

        def slow_eval(*a, **k):
            time.sleep(2.5)  # stall inside the opaque call
            return original(*a, **k)

        import bc_executor as BE
        BE.evaluate_model = slow_eval
        try:
            from nvflare.apis.fl_context import FLContext
            ex._evaluate_model(FLContext(), split="val")
        finally:
            BE.evaluate_model = original
        beats = [l for l in cap.lines if "[heartbeat]" in l and "evaluate" in l]
        assert beats, "a stalled evaluate_model produced no heartbeat output"
    print("[progress] evaluate stalls are visible via heartbeat. OK")


def test_heartbeat_distinguishes_loader_stall_from_compute_stall():
    """batch=-1/stage=load means the loader never yielded; stage=forward means compute wedged.

    This is the discrimination that a remote site cannot otherwise give us: with no shell access,
    the stage field is the only way to tell 'stuck reading images' from 'stuck in a CUDA kernel'.
    """
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, heartbeat_interval_s=1)
        cap = _attach(ex)
        with ex._heartbeat("ditto-personal", {"site": "RSNA-AU", "batch": -1, "stage": "load",
                                              "t_last_progress": time.time()}):
            time.sleep(1.5)
        beats = [l for l in cap.lines if "[heartbeat]" in l]
        assert any("batch=-1" in l and "stage=load" in l for l in beats), \
            f"a first-batch loader stall is not identifiable from the beat: {beats}"
    print("[progress] heartbeat separates loader stalls from compute stalls. OK")


if __name__ == "__main__":
    test_heartbeat_fires_while_the_main_thread_is_blocked()
    test_heartbeat_distinguishes_loader_stall_from_compute_stall()
    test_heartbeat_stops_after_the_phase_and_never_raises()
    test_heartbeat_disabled_by_zero_interval()
    test_personal_pass_logs_every_batch()
    test_evaluate_is_covered_by_a_heartbeat()
    print("all progress-logging tests passed")
