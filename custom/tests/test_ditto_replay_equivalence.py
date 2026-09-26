# ============================================================================
# test_ditto_replay_equivalence.py - Ditto consolidation: interleaved == replay
# Run from custom/tests/:  python test_ditto_replay_equivalence.py
#
# The KEY correctness proof for the FedAvg+Ditto consolidation (Phase 3.1):
#   replaying a CACHED per-round global trajectory to train the personalized model v must
#   be BIT-IDENTICAL to the interleaved formulation that trains the global inline.
#
# This is a real weight-equivalence check on synthetic CPU rounds (R=3, 2 clients, a
# MID-RANGE lambda where the proximal term actually binds -- not lambda->0/inf which would
# mask an off-by-one anchor bug). It exercises:
#   - anchor convention: round-t personal update anchors to w^{t-1} (start-of-round global)
#   - v carried forward across rounds (never reset)
#   - same E_personalized per round
# A bit-mismatch here means replay != interleaved -> STOP (do not sweep).
#
# Uses a tiny linear model (not full GMIC) so the test is fast and deterministic; the logic
# under test (replay_personalized + the anchor mapping) is model-agnostic.
# ============================================================================
import os
import sys
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)                     # this tests/ dir (sibling test imports)
sys.path.insert(0, os.path.dirname(THIS_DIR))    # custom/ (bc_executor, fl_utils, model.*, ...)

import fl_utils as F  # noqa: E402

torch.manual_seed(0)


class Tiny(nn.Module):
    """Tiny stand-in: linear head. forward returns a 2-col logit so malignant-index=1 works
    with the same proximal/loss helpers; here we use a plain BCE-style loss for speed."""
    def __init__(self, d=8):
        super().__init__()
        self.fc = nn.Linear(d, 2, bias=True)

    def forward(self, x):
        return self.fc(x)


def make_batches(rng, n_batches=3, bs=4, d=8):
    """Deterministic fixed dataset (same every call) so train passes are reproducible."""
    data = []
    for b in range(n_batches):
        x = torch.tensor(rng.RandomState(100 + b).randn(bs, d), dtype=torch.float32)
        y = torch.tensor((rng.RandomState(200 + b).rand(bs) > 0.5).astype(int), dtype=torch.long)
        data.append((x, y, None))
    return data


# fixed dataset (identical for both runs)
_BATCHES = make_batches(np.random, n_batches=3, bs=4, d=8)


def loss_fn(outputs, targets):
    return nn.functional.cross_entropy(outputs, targets)


def prox(model, ref_state, lam):
    """Simple L2-to-ref proximal (scalar lam), mirroring proximal_penalty's shape."""
    p = 0.0
    for name, param in model.named_parameters():
        p = p + (lam / 2.0) * ((param - ref_state[name].to(param.device)) ** 2).sum()
    return p


def build_global_trajectory(R, d=8):
    """Synthetic 'FedAvg' global trajectory: w_init (pretrained) then w^0..w^{R-1}.
    Values are arbitrary-but-fixed; what matters is the anchor MAPPING, not how they were made."""
    traj = []
    for t in range(R + 1):  # index 0 = init, 1..R = w^0..w^{R-1}
        m = Tiny(d)
        torch.manual_seed(500 + t)
        for param in m.parameters():
            param.data = torch.randn_like(param)
        traj.append({k: v.detach().clone() for k, v in m.state_dict().items()})
    return traj


def run_interleaved(R, lam, e_pers, traj, d=8):
    """The INTERLEAVED reference: at round t, anchor = traj[t] (start-of-round = w^{t-1}),
    train v for e_pers epochs, carry v forward. (Global pass omitted -- it's identical to the
    cached traj by construction; we only test the personal-model reproduction.)"""
    v = Tiny(d)
    torch.manual_seed(42)
    for param in v.parameters():
        param.data = torch.randn_like(param)
    opt = torch.optim.SGD(v.parameters(), lr=0.05)
    v.train()
    for t in range(R):
        ref = traj[t]                          # start-of-round anchor (w^{t-1}); round0 -> init
        for _ in range(e_pers):
            for x, y, _m in _BATCHES:
                opt.zero_grad(set_to_none=True)
                out = v(x)
                l = loss_fn(out, y) + prox(v, ref, lam)
                l.backward(); opt.step()
    return v


def run_replay(R, lam, e_pers, traj, d=8):
    """The REPLAY path under test: identical init/opt, anchors read from the cached traj via
    replay_personalized. Must match interleaved bit-for-bit."""
    v = Tiny(d)
    torch.manual_seed(42)
    for param in v.parameters():
        param.data = torch.randn_like(param)
    opt = torch.optim.SGD(v.parameters(), lr=0.05)
    # anchor_trajectory[t] must be the start-of-round global for round t = traj[t]
    F.replay_personalized(
        v_model=v, v_optimizer=opt, anchor_trajectory=traj,
        loss_fn=loss_fn, proximal_fn=prox, lam=lam, device=torch.device("cpu"),
        n_rounds=R, e_personalized=e_pers, batch_iter_factory=lambda: iter(_BATCHES),
    )
    return v


def test_interleaved_equals_replay():
    R, lam, e_pers = 3, 0.1, 2           # mid-range lambda where the prox term binds
    traj = build_global_trajectory(R)
    v_il = run_interleaved(R, lam, e_pers, traj)
    v_rp = run_replay(R, lam, e_pers, traj)
    max_delta = 0.0
    for (n1, p1), (n2, p2) in zip(v_il.named_parameters(), v_rp.named_parameters()):
        assert n1 == n2
        max_delta = max(max_delta, float((p1 - p2).abs().max()))
    assert max_delta == 0.0, f"interleaved vs replay DIVERGED: max|delta|={max_delta} (anchor off-by-one or v-reset?)"
    print(f"[equivalence] interleaved == replay, BIT-IDENTICAL (R={R} lambda={lam} E={e_pers}) max|delta|={max_delta}")


def test_anchor_offbyone_is_detectable():
    """Sanity: the test is SENSITIVE -- a deliberate one-round anchor shift must break equality
    (proves a passing equivalence test is meaningful, not vacuous)."""
    R, lam, e_pers = 3, 0.1, 2
    traj = build_global_trajectory(R)
    v_il = run_interleaved(R, lam, e_pers, traj)
    # replay with a WRONG (shifted-by-one) anchor mapping -> must NOT match
    v = Tiny(8)
    torch.manual_seed(42)
    for param in v.parameters():
        param.data = torch.randn_like(param)
    opt = torch.optim.SGD(v.parameters(), lr=0.05)
    F.replay_personalized(
        v_model=v, v_optimizer=opt, anchor_trajectory=traj[1:],  # WRONG: anchors to w^t not w^{t-1}
        loss_fn=loss_fn, proximal_fn=prox, lam=lam, device=torch.device("cpu"),
        n_rounds=R, e_personalized=e_pers, batch_iter_factory=lambda: iter(_BATCHES),
    )
    diverged = any(float((p1 - p2).abs().max()) > 0
                   for (_, p1), (_, p2) in zip(v_il.named_parameters(), v.named_parameters()))
    assert diverged, "off-by-one anchor shift was NOT detected -> equivalence test is vacuous!"
    print("[equivalence] off-by-one anchor shift IS detected -> test is sensitive (non-vacuous)")


def test_selection_metrics():
    """site_selection_metrics returns all three; worst_site=min, mean=avg, pooled=n-weighted."""
    per = {"siteA": {"auc": 0.90, "n": 100}, "siteB": {"auc": 0.70, "n": 300}}
    m = F.site_selection_metrics(per)
    assert abs(m["worst_site"] - 0.70) < 1e-9
    assert abs(m["mean_site"] - 0.80) < 1e-9
    assert abs(m["pooled"] - (0.90 * 100 + 0.70 * 300) / 400) < 1e-9   # 0.75
    # NaN site excluded
    m2 = F.site_selection_metrics({"a": float("nan"), "b": 0.8})
    assert abs(m2["worst_site"] - 0.8) < 1e-9
    print(f"[selection] all three metrics OK (worst={m['worst_site']:.3f} mean={m['mean_site']:.3f} pooled={m['pooled']:.3f})")


ALL = [test_interleaved_equals_replay, test_anchor_offbyone_is_detectable, test_selection_metrics]

if __name__ == "__main__":
    failures = 0
    for t in ALL:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
    print("\n==== %d/%d replay-equivalence tests passed ====" % (len(ALL) - failures, len(ALL)))
    sys.exit(1 if failures else 0)
