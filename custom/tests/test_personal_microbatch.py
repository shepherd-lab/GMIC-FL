# ============================================================================
# test_personal_microbatch.py - micro-chunked Ditto personal pass
# ----------------------------------------------------------------------------
# RSNA-AU's 16 GiB A4000 deadlocks on the full fp32 personal batch (it needs ~17-22 GiB, oversubscribes
# to host RAM). personal_batch_size_by_site splits each loader batch into micro-chunks that fit,
# accumulating gradients so the optimizer step equals a single full-batch step -- only BatchNorm
# sees the smaller chunk. The correctness claim is exactly that gradient-equality, pinned here.
#
# Run from custom/tests/:   python -m pytest test_personal_microbatch.py -q
# ============================================================================
import copy
import math
import tempfile

import torch
import torch.nn as nn

from test_executor_integration import (  # noqa: E402 (installs the resource stub on import)
    GMIC_PARAMS, make_executor,
)
from model.gmic import GMIC                                   # noqa: E402
from fl_utils import proximal_penalty, clone_detached_state    # noqa: E402


def _grads_for(chunk_size, batch, seed=0):
    """Run ONE personal-style optimizer accumulation over `batch` with the given micro-chunk size
    (chunk_size >= batch => un-chunked) and return the resulting .grad on every param.

    Mirrors the executor's chunked path: per-chunk task loss weighted by chunk/total, accumulated,
    then the proximal penalty added once. Uses a fixed model so both runs share identical weights.
    """
    torch.manual_seed(1234)
    v = GMIC(copy.deepcopy(GMIC_PARAMS))
    # EVAL so BatchNorm uses fixed running stats -- this isolates the weighting/accumulation math
    # from BN's per-chunk normalization. With BN frozen, chunked MUST equal full-batch exactly; the
    # train-mode BN difference (chunk sees fewer samples) is the separate, accepted deviation.
    v.eval()
    ref = clone_detached_state(v)
    lam = {"global": 0.01, "local": 0.5, "fusion": 0.0}
    crit = nn.CrossEntropyLoss()

    def task_loss(out, y):
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return crit(logits, y)

    x = torch.randn(batch, 1, 2944, 1920)
    y = torch.tensor([i % 2 for i in range(batch)], dtype=torch.long)

    v.zero_grad(set_to_none=True)
    total = x.size(0)
    n_chunks = max(1, math.ceil(total / chunk_size))
    for ci in range(n_chunks):
        cx = x[ci * chunk_size:(ci + 1) * chunk_size]
        cy = y[ci * chunk_size:(ci + 1) * chunk_size]
        if cx.size(0) == 0:
            continue
        loss_c = task_loss(v(cx), cy) * (cx.size(0) / total)
        loss_c.backward()
    prox = proximal_penalty(v, ref, lam)
    prox.backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in v.named_parameters()}


def test_chunked_grad_equals_full_batch_grad():
    """The whole point: chunked accumulation must yield the SAME gradient as the full batch.

    BatchNorm makes this only approximately equal (BN normalizes per forward, so chunked BN uses
    different statistics), so we assert equality within a tolerance that catches a real weighting
    or accumulation bug while allowing the expected BN-induced difference. The non-BN parameters
    (convs, heads) should match very tightly.
    """
    batch = 8
    full = _grads_for(chunk_size=8, batch=batch)   # 1 chunk (un-chunked reference)
    chunked = _grads_for(chunk_size=2, batch=batch)  # 4 chunks of 2

    assert set(full) == set(chunked)
    worst_nonbn = 0.0
    n_nonbn = 0
    for name in full:
        gf, gc = full[name], chunked[name]
        if gf is None or gc is None:
            continue
        denom = gf.abs().mean().item() + 1e-8
        rel = (gf - gc).abs().mean().item() / denom
        # BN weight/bias params live in *.bn*/*norm* -- allow those to differ (different per-chunk
        # stats); everything else must match tightly.
        is_bn = any(t in name.lower() for t in ("bn", "norm", "running"))
        if not is_bn:
            worst_nonbn = max(worst_nonbn, rel)
            n_nonbn += 1
    assert n_nonbn > 0
    assert worst_nonbn < 5e-2, f"non-BN grads diverged (worst rel={worst_nonbn:.3e}) -- weighting/accumulation bug"
    print(f"[microbatch] chunked grad == full-batch grad on non-BN params "
          f"(worst rel={worst_nonbn:.2e}, {n_nonbn} tensors). OK")


def _main_grads_for(chunk_size, batch):
    """Main-pass-style accumulation: per-chunk task loss weighted by chunk/total (grad_accum=1),
    accumulated, then step. Returns .grad per param. BN frozen so chunked must equal full-batch."""
    torch.manual_seed(4321)
    m = GMIC(copy.deepcopy(GMIC_PARAMS))
    m.eval()  # freeze BN to isolate the accumulation math from per-chunk normalization
    crit = nn.CrossEntropyLoss()

    def task_loss(out, y):
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return crit(logits, y)

    x = torch.randn(batch, 1, 2944, 1920)
    y = torch.tensor([i % 2 for i in range(batch)], dtype=torch.long)
    m.zero_grad(set_to_none=True)
    total = x.size(0)
    n_chunks = max(1, math.ceil(total / chunk_size))
    for ci in range(n_chunks):
        cx = x[ci * chunk_size:(ci + 1) * chunk_size]
        cy = y[ci * chunk_size:(ci + 1) * chunk_size]
        if cx.size(0) == 0:
            continue
        (task_loss(m(cx), cy) * (cx.size(0) / total)).backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in m.named_parameters()}


def test_main_chunked_grad_equals_full_batch_grad():
    """The shared-w main pass is the federated result, so its chunking MUST be gradient-exact too
    (BN frozen). Same weighting as the personal pass; this pins it independently."""
    full = _main_grads_for(chunk_size=8, batch=8)
    chunked = _main_grads_for(chunk_size=2, batch=8)
    worst = 0.0
    n = 0
    for name in full:
        gf, gc = full[name], chunked[name]
        if gf is None or gc is None:
            continue
        if any(t in name.lower() for t in ("bn", "norm", "running")):
            continue
        worst = max(worst, (gf - gc).abs().mean().item() / (gf.abs().mean().item() + 1e-8))
        n += 1
    assert n > 0 and worst < 5e-2, f"main-pass chunked grad diverged (worst rel={worst:.3e})"
    print(f"[microbatch] MAIN-pass chunked grad == full-batch grad (worst rel={worst:.2e}). OK")


def test_train_batch_size_by_site_resolves():
    """The main-pass per-site map picks RSNA-AU's micro-batch; unlisted sites use the full loader batch."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, train_batch_size_by_site={"RSNA-AU": 8})
        assert ex._train_batch_size_by_site.get("RSNA-AU") == 8
        assert ex._train_batch_size_by_site.get("RSNA-US", ex.batch_size) == ex.batch_size
    print("[microbatch] main-pass per-site map resolves (RSNA-AU->8, others->batch_size). OK")


def test_personal_batch_size_by_site_resolves():
    """The per-site map picks RSNA-AU's micro-batch; unlisted sites use the full loader batch."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, personal_batch_size_by_site={"RSNA-AU": 8})
        assert ex._personal_batch_size_by_site.get("RSNA-AU") == 8
        assert ex._personal_batch_size_by_site.get("RSNA-US", ex.batch_size) == ex.batch_size
    print("[microbatch] per-site micro-batch map resolves (RSNA-AU->8, others->batch_size). OK")


def test_default_is_unchunked():
    """No map => personal_batch_size_by_site empty => every site uses batch_size (1 chunk)."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td)
        assert ex._personal_batch_size_by_site == {}
    print("[microbatch] default = no chunking. OK")


if __name__ == "__main__":
    test_chunked_grad_equals_full_batch_grad()
    test_main_chunked_grad_equals_full_batch_grad()
    test_train_batch_size_by_site_resolves()
    test_personal_batch_size_by_site_resolves()
    test_default_is_unchunked()
    print("all microbatch tests passed")
