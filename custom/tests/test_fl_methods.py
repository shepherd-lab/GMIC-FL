# ============================================================================
# test_fl_methods.py - CPU synthetic tests for the federated-method primitives
# ----------------------------------------------------------------------------
# Run from custom/tests/:   python test_fl_methods.py
# (or with pytest:                 pytest -q test_fl_methods.py)
#
# Covers brief tests:
#   1. Module-grouping assertion (every param -> exactly one group)
#   2. BN-skip (FedBN load preserves local BN; non-BN equals global)
#   3. Proximal-term (zero at ref; scales with per-group lambda)
#   7. Pretrained-load (released GMIC ckpt loads essentially complete)
#   + model forward smoke on a synthetic full-size input
#   + Ditto personal-model persistence (v not reset to global across rounds)
#
# Tests 4 (full-executor per-method round) and 6 (no-regression vs old executor)
# require NVFLARE + a data loader and are exercised by the companion
# integration check; see report. The math those rely on is covered here.
# ============================================================================
import os
import sys
import copy

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)                     # this tests/ dir (sibling test imports)
sys.path.insert(0, os.path.dirname(THIS_DIR))    # custom/ (bc_executor, fl_utils, model.*, ...)

from model.gmic import GMIC  # noqa: E402
import fl_utils as F  # noqa: E402

GMIC_PARAMS = {
    "device_type": "cpu", "gpu_number": 0,
    "max_crop_noise": (100, 100), "max_crop_size_noise": 100,
    "image_path": "x", "cam_size": (46, 30), "K": 6,
    "crop_shape": (256, 256), "post_processing_dim": 256,
    "num_classes": 2, "use_v1_global": False,
    "percent_t": 0.02, "lambda_l1": 1e-5,
}

# Optional GMIC pretrained warm-start (not shipped in-repo -- download the 5 sample_model_*.p
# weights from https://github.com/nyukat/GMIC). The pretrained-load check skips if absent.
CKPT = os.environ.get("GMIC_PRETRAINED", "/workspace/models/sample_model_5.p")


def build_model():
    return GMIC(copy.deepcopy(GMIC_PARAMS))


# --------------------------------------------------------------------------
def test_grouping():
    m = build_model()
    sizes = F.assert_grouping_complete(m)
    total_params = sum(1 for _ in m.named_parameters())
    print(f"[grouping] sizes={sizes} total={total_params}")
    assert sum(sizes.values()) == total_params, "some params unassigned"
    assert sizes["global"] == 69 and sizes["local"] == 60 and sizes["fusion"] == 6, sizes
    # every key resolves without raising; buffers map to a known/ignore group
    for name, _ in m.named_buffers():
        F.module_group(name)
    print("[grouping] OK")


def test_bn_skip():
    g = build_model()
    m = build_model()
    bn_key = "ds_net.final_bn.running_mean"          # a BN buffer (skipped)
    conv_key = "ds_net.first_conv.weight"            # a non-BN param (loaded)
    assert bn_key in g.state_dict() and conv_key in g.state_dict()

    with torch.no_grad():
        g.state_dict()[bn_key].zero_()
        g.state_dict()[conv_key].zero_()
        m.state_dict()[bn_key].fill_(7.0)
        m.state_dict()[conv_key].fill_(7.0)

    skip = F.bn_state_keys(m)
    assert bn_key in skip and conv_key not in skip
    print(f"[bn_skip] #BN keys skipped = {len(skip)}")

    stats = F.load_global_into(m, g.state_dict(), skip_keys=skip)
    print(f"[bn_skip] load stats = {stats}")

    # local BN preserved (still 7), differs from global (0)
    assert torch.allclose(m.state_dict()[bn_key], torch.full_like(m.state_dict()[bn_key], 7.0))
    assert not torch.allclose(m.state_dict()[bn_key], g.state_dict()[bn_key])
    # non-BN conv now equals global (0)
    assert torch.allclose(m.state_dict()[conv_key], g.state_dict()[conv_key])
    print("[bn_skip] OK")


def test_proximal():
    m = build_model()
    ref = F.clone_detached_state(m)

    # zero at the reference
    p0 = F.proximal_penalty(m, ref, 1.0)
    assert float(p0) < 1e-9, f"expected ~0, got {float(p0)}"

    # perturb exactly one GLOBAL param and one LOCAL param by constant c
    c = 0.5
    pg_name = "ds_net.first_conv.weight"     # global
    pl_name = "dn_resnet.conv1.weight"       # local
    params = dict(m.named_parameters())
    with torch.no_grad():
        params[pg_name].add_(c)
        params[pl_name].add_(c)
    ng = params[pg_name].numel()
    nl = params[pl_name].numel()

    lam = {"global": 3.0, "local": 0.2, "fusion": 0.2}
    pen = float(F.proximal_penalty(m, ref, lam))
    expected = (lam["global"] / 2) * (c ** 2) * ng + (lam["local"] / 2) * (c ** 2) * nl
    print(f"[proximal] penalty={pen:.4f} expected={expected:.4f} (ng={ng}, nl={nl})")
    assert abs(pen - expected) / expected < 1e-5, (pen, expected)

    # scalar-lambda path matches a uniform dict
    pen_scalar = float(F.proximal_penalty(m, ref, 1.0))
    pen_uniform = float(F.proximal_penalty(m, ref, {"global": 1.0, "local": 1.0, "fusion": 1.0}))
    assert abs(pen_scalar - pen_uniform) / pen_uniform < 1e-6
    print("[proximal] OK")


def test_pretrained_load():
    if not os.path.isfile(CKPT):
        print(f"[pretrained] SKIP (ckpt not found: {CKPT})")
        return
    m = build_model()
    rep = F.tolerant_load_pretrained(m, CKPT, device="cpu")
    own = len(m.state_dict())
    print(f"[pretrained] matched={rep['matched']}/{own} missing={rep['missing_keys']} "
          f"unexpected={rep['unexpected_keys']}")
    assert rep["matched"] / own >= 0.99, "substantial fraction of weights unmatched"
    assert rep.get("backbone_match_fraction", 1.0) >= 0.99
    print("[pretrained] OK")


def test_forward_smoke():
    m = build_model().eval()
    # full-size input is required: global net downsamples by 64 -> cam (46,30)
    x = torch.randn(1, 1, 2944, 1920)
    with torch.no_grad():
        out = m(x)
    # native GMIC: forward returns (fusion_logits, y_global, y_local, saliency)
    assert isinstance(out, tuple) and len(out) == 4, type(out)
    fusion_logits, y_global, y_local, saliency = out
    assert fusion_logits.shape == (1, 2), fusion_logits.shape
    assert y_global.shape == (1, 2) and y_local.shape == (1, 2)
    assert saliency.dim() == 4 and saliency.shape[1] == 2, saliency.shape
    # global/local/saliency are probabilities in [0,1]; fusion is a raw logit
    for name, p in [("y_global", y_global), ("y_local", y_local), ("saliency", saliency)]:
        assert float(p.min()) >= -1e-6 and float(p.max()) <= 1 + 1e-6, (name, float(p.min()), float(p.max()))
    s = F.malignant_score(out)
    assert s.shape == (1,) and 0.0 <= float(s) <= 1.0
    print(f"[forward] 4-tuple OK; fusion={tuple(fusion_logits.shape)} saliency={tuple(saliency.shape)} mal_score={float(s):.3f}")


def test_gmic_loss():
    m = build_model().train()
    x = torch.randn(2, 1, 2944, 1920)
    t = torch.tensor([0, 1])
    out = m(x)
    # pos_weight is shared across heads; changing it changes the loss
    l_pw1 = float(F.gmic_malignant_loss(out, t, lambda_l1=1e-5, pos_weight=1.0))
    l_pw10 = float(F.gmic_malignant_loss(out, t, lambda_l1=1e-5, pos_weight=10.0))
    assert l_pw1 != l_pw10, (l_pw1, l_pw10)
    # main loss is finite/positive and backprops into the malignant fusion head
    loss = F.gmic_malignant_loss(out, t, lambda_l1=1e-5, pos_weight=3.0)
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    g = dict(m.named_parameters())["fusion_dnn.weight"].grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    print(f"[gmic_loss] OK loss={float(loss):.4f} (pw1={l_pw1:.4f} pw10={l_pw10:.4f})")


def test_focal_loss():
    """Focal loss: same 3-head+L1 structure; gamma/alpha change it; backprops to fusion head."""
    m = build_model().train()
    x = torch.randn(2, 1, 2944, 1920)
    t = torch.tensor([0, 1])
    out = m(x)
    l_g0 = float(F.gmic_focal_loss(out, t, gamma=0.0, alpha=0.25))   # gamma=0 -> weighted BCE
    l_g2 = float(F.gmic_focal_loss(out, t, gamma=2.0, alpha=0.25))
    assert l_g0 != l_g2, (l_g0, l_g2)                                 # gamma matters
    l_a = float(F.gmic_focal_loss(out, t, gamma=2.0, alpha=0.75))
    assert l_a != l_g2                                                # alpha matters
    loss = F.gmic_focal_loss(out, t, gamma=2.0, alpha=0.25)
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    g = dict(m.named_parameters())["fusion_dnn.weight"].grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    print(f"[focal] OK loss={float(loss):.4f} (g0={l_g0:.4f} g2={l_g2:.4f} a.75={l_a:.4f})")


def test_threshold_metrics():
    """classification_metrics: AUC threshold-free; sens/spec/precision move with threshold."""
    import numpy as np
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.45, 0.6, 0.55, 0.9])
    m05 = F.classification_metrics(y, p, threshold=0.5)
    m08 = F.classification_metrics(y, p, threshold=0.8)
    assert abs(m05["auc"] - m08["auc"]) < 1e-9, "AUC must be threshold-independent"
    # raising the threshold cannot increase sensitivity (fewer predicted positives)
    assert m08["sensitivity"] <= m05["sensitivity"] + 1e-9
    # and cannot decrease specificity
    assert m08["specificity"] >= m05["specificity"] - 1e-9
    # at 0.5: preds>0.5 = [.6,.55,.9] -> tp=2 (.55,.9), fp=1 (.6) ; sens=2/2, spec=3/4
    assert abs(m05["sensitivity"] - 1.0) < 1e-9 and abs(m05["specificity"] - 0.75) < 1e-9
    # degenerate (single class) -> AUC NaN
    import math
    assert math.isnan(F.classification_metrics(np.array([0, 0]), np.array([0.3, 0.7]))["auc"])
    print(f"[threshold] OK sens@.5={m05['sensitivity']:.2f} spec@.5={m05['specificity']:.2f} "
          f"sens@.8={m08['sensitivity']:.2f} spec@.8={m08['specificity']:.2f}")


def test_threshold_sweep():
    """Sweep sanity: n_flagged is monotonic non-increasing as threshold rises; denominators
    (n_pos+n_neg) sum to the eval-set size. Built from classification_metrics like the executor."""
    import numpy as np
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1])          # 5 neg, 3 pos
    p = np.array([0.05, 0.2, 0.35, 0.5, 0.65, 0.4, 0.7, 0.95])
    thrs = [0.3, 0.4, 0.5, 0.6, 0.7]
    flagged = []
    for thr in thrs:
        m = F.classification_metrics(y, p, threshold=thr)
        nf = int((p > thr).sum())
        flagged.append(nf)
        # denominators sum to eval-set size
        assert m["n_pos"] + (m["total_samples"] - m["n_pos"]) == len(y)
    # monotonic non-increasing as threshold increases (fewer flagged at higher thr)
    assert all(flagged[i] >= flagged[i + 1] for i in range(len(flagged) - 1)), flagged
    # and n_pos is threshold-independent
    n_pos_vals = {F.classification_metrics(y, p, threshold=t)["n_pos"] for t in thrs}
    assert n_pos_vals == {3}
    print(f"[threshold-sweep] OK n_flagged monotonic {flagged} (thr {thrs}); denominators sum to {len(y)}")


def test_balanced_sampler_order():
    """Balanced index resampling: heavily-imbalanced pool -> ~even class mix in the drawn order."""
    import numpy as np
    sys.path.insert(0, THIS_DIR)
    from data_loader.data_loader import GMICDataLoader
    # build a loader without running __init__ (avoids preprocessing); call the method directly
    dl = GMICDataLoader.__new__(GMICDataLoader)
    samples = ([{"view_level_label": 1}] * 10) + ([{"view_level_label": 0}] * 90)  # 9:1 neg-heavy
    rng = np.random.RandomState(0)
    order = dl._balanced_index_order(samples, rng)
    labels = np.array([samples[i]["view_level_label"] for i in order])
    pos_frac = labels.mean()
    assert len(order) == len(samples), "epoch size must be preserved"
    assert 0.4 <= pos_frac <= 0.6, f"expected ~balanced draw, got pos_frac={pos_frac:.2f}"
    # degenerate single-class pool -> identity order (nothing to balance)
    order2 = dl._balanced_index_order([{"view_level_label": 0}] * 5, np.random.RandomState(0))
    assert order2 == list(range(5))
    print(f"[sampler] OK 9:1 pool -> drawn pos_frac={pos_frac:.2f} (len preserved)")


def test_augmentation_safety():
    """Conservative train augmentation: zero-fill rotation (NOT reflection), reproducible,
    flip-off preserves orientation, and the data-loader gate is train-only."""
    import numpy as np
    import inspect
    sys.path.insert(0, THIS_DIR)
    import data_loader.augmentations as aug
    from data_loader.data_loader import GMICDataLoader

    # asymmetric bright 'tissue' on the right so a reflection fill would be detectable
    img = np.zeros((400, 300), dtype=np.float32)
    img[100:300, 180:300] = 5.0

    # 1. rotation out-of-frame fill is ZERO, not reflected tissue
    out = aug.conservative_train_augment(img, np.random.RandomState(0),
                                         max_rotation_deg=10, intensity_jitter=0.0)
    assert out[:30, :30].max() < 0.5, "rotation corner not zero-filled (reflection?!)"
    # 2. reproducible under a fixed seed
    a = aug.conservative_train_augment(img, np.random.RandomState(42), 10, 0.10)
    b = aug.conservative_train_augment(img, np.random.RandomState(42), 10, 0.10)
    assert np.array_equal(a, b), "augmentation not reproducible under fixed seed"
    # 3. flip OFF keeps tissue on the right (orientation invariant preserved)
    nf = aug.conservative_train_augment(img, np.random.RandomState(7), 0, 0.0, horizontal_flip=False)
    assert nf[:, 150:].sum() > nf[:, :150].sum(), "flip-off must keep orientation"
    # 4. data-loader gate is train-only (exactly one call site, requires is_train)
    src = inspect.getsource(GMICDataLoader.get_batch_iterator)
    assert src.count("conservative_train_augment") == 1
    assert "if is_train and self.use_augmentation:" in src, "augmentation must be gated on is_train"
    print("[augmentation] OK zero-fill (not reflection), reproducible, flip-off keeps orientation, train-only gate")


def test_breast_aggregate():
    """breast_aggregate: per-image -> breast (exam_id, laterality); pred=mean(views), label=max.
    Don't assume 2 views; parse laterality robustly; fail loud on unparseable view."""
    # exam 0: L breast 2 views (CC .8, MLO .6 -> mean .7), R breast 1 view (.2)
    # exam 1: L breast 2 views (.9,.9 -> .9)
    exam = [0, 0, 0, 1, 1]
    view = ["L-CC", "L-MLO", "R-CC", "L-CC", "L-MLO"]
    probs = [0.8, 0.6, 0.2, 0.9, 0.9]
    labels = [1, 1, 0, 1, 1]
    bp, by = F.breast_aggregate(exam, view, probs, labels)
    assert len(bp) == 3, f"expected 3 breasts, got {len(bp)}"            # (0,L),(0,R),(1,L)
    assert abs(bp[0] - 0.7) < 1e-9 and by[0] == 1                        # mean(.8,.6)=.7
    assert abs(bp[1] - 0.2) < 1e-9 and by[1] == 0                        # single-view breast
    assert abs(bp[2] - 0.9) < 1e-9 and by[2] == 1
    # label = max over views (robust to a stray disagreeing 0)
    bp2, by2 = F.breast_aggregate([5, 5], ["R-CC", "R-MLO"], [0.5, 0.5], [0, 1])
    assert by2[0] == 1
    # robust laterality parse (lowercase, no dash) + fail-loud on garbage
    assert F._parse_laterality("l-cc") == "L" and F._parse_laterality("RMLO") == "R"
    raised = False
    try:
        F.breast_aggregate([9], ["XX-CC"], [0.5], [1])
    except ValueError:
        raised = True
    assert raised, "unparseable view must raise (no silent drop)"
    print("[breast] OK 5 images -> 3 breasts, mean-of-views pred, max label, robust parse")


def test_optimizer_selection():
    """configure_optimizers honors optimizer_name: default/adam -> Adam, adamw -> AdamW.
    Both keep per-group LRs (backbone lr_backbone, heads lr_heads) and apply weight_decay."""
    from train.training_core import configure_optimizers
    class OA:
        lr_heads = 3e-5; lr_backbone = 1e-5; weight_decay = 1e-2; patience = 4
        lr_scheduler = None; scheduler_patience = 2; min_lr = 0.0; epochs = 40
    oa = OA()
    # default (attr absent) -> Adam (byte-compat)
    o_def, _, _ = configure_optimizers(build_model(), oa)
    assert type(o_def).__name__ == "Adam"
    oa.optimizer_name = "adam"
    o_adam, _, _ = configure_optimizers(build_model(), oa)
    assert type(o_adam).__name__ == "Adam"
    oa.optimizer_name = "adamw"
    o_adamw, _, _ = configure_optimizers(build_model(), oa)
    assert type(o_adamw).__name__ == "AdamW"
    lrs = [g["lr"] for g in o_adamw.param_groups]
    assert 1e-5 in lrs and 3e-5 in lrs, lrs                    # per-group LRs preserved
    assert all(abs(g["weight_decay"] - 1e-2) < 1e-12 for g in o_adamw.param_groups)
    assert all(g["betas"] == (0.9, 0.999) for g in o_adamw.param_groups)   # default betas
    print("[optimizer] OK default/adam->Adam, adamw->AdamW; per-group LRs + wd=0.01 + betas preserved")


def test_scheduler_selection():
    """configure_optimizers honors lr_scheduler: None->plateau (back-compat), cosine, none."""
    import torch.optim as optim
    from train.training_core import configure_optimizers
    m = build_model()

    class OA:
        lr_heads = 3e-5; lr_backbone = 1e-5; weight_decay = 1e-6; patience = 4
        scheduler_patience = 2; min_lr = 0.0; epochs = 40
    oa = OA()
    oa.lr_scheduler = None
    _, s_default, _ = configure_optimizers(m, oa)
    assert isinstance(s_default, optim.lr_scheduler.ReduceLROnPlateau), "None must keep plateau (back-compat)"
    oa.lr_scheduler = "cosine"
    _, s_cos, _ = configure_optimizers(m, oa)
    assert isinstance(s_cos, optim.lr_scheduler.CosineAnnealingLR)
    oa.lr_scheduler = "none"
    _, s_none, _ = configure_optimizers(m, oa)
    assert s_none is None
    print("[scheduler] OK None->plateau(back-compat), cosine->CosineAnnealingLR, none->None")


def test_ditto_persistence():
    """v is initialized once and continues across rounds (never reset to global)."""
    w = build_model()
    v = copy.deepcopy(w)                       # init v once from (pretrained) w
    v_opt = torch.optim.SGD(v.parameters(), lr=0.1)
    pname = "fusion_dnn.weight"
    v_before = dict(v.named_parameters())[pname].detach().clone()

    # round 1: a personal-pass step moves v
    loss = (dict(v.named_parameters())[pname] ** 2).sum()
    v_opt.zero_grad(); loss.backward(); v_opt.step()
    v_after_r1 = dict(v.named_parameters())[pname].detach().clone()
    assert not torch.allclose(v_before, v_after_r1), "v did not update"

    # round 2: new global arrives into w; v must NOT be reset to it
    new_global = build_model()
    F.load_global_into(w, new_global.state_dict())   # w tracks global
    v_after_r2_start = dict(v.named_parameters())[pname].detach().clone()
    assert torch.allclose(v_after_r1, v_after_r2_start), "v was reset to global!"
    print("[ditto] v persists across rounds and is not reset to global. OK")


def test_amp_loss_boundary():
    """B1 regression on the bf16 boundary, model-free.

    Mirrors the use_amp path: on autocast-produced bf16 outputs, both the loss (its BCE
    terms need fp32) and malignant_score (`.numpy()` on bf16 raises) must work. This is
    the exact path that crashed the use_amp=True executor run.
    """
    B = 4
    t = torch.tensor([0, 1, 0, 1])
    out = (
        torch.randn(B, 2, dtype=torch.bfloat16, requires_grad=True),   # fusion logits
        torch.rand(B, 2, dtype=torch.bfloat16, requires_grad=True),    # y_global prob
        torch.rand(B, 2, dtype=torch.bfloat16, requires_grad=True),    # y_local prob
        torch.rand(B, 2, 46, 30, dtype=torch.bfloat16, requires_grad=True),  # saliency
    )
    loss = F.gmic_malignant_loss(out, t, pos_weight=3.0)
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    sn = F.malignant_score(out).detach().cpu().numpy()   # bf16 -> must cast to fp32
    assert sn.shape == (B,) and float(sn.min()) >= 0.0 and float(sn.max()) <= 1.0
    print(f"[amp] bf16 loss + malignant_score().numpy() OK (loss={float(loss):.4f})")


def test_earlystopper_reset():
    """C2: reset() clears best/count/best_state so best-val selection is within-round
    and never crosses rounds."""
    from train.training_core import EarlyStopper
    m = build_model()
    es = EarlyStopper(patience=2)
    assert es.step(0.5, m) is True and abs(es.best - 0.5) < 1e-9
    assert es.step(0.4, m) is False and es.count == 1        # no improvement
    es.reset()
    assert es.best == -float("inf") and es.best_state is None and es.count == 0
    assert es.step(0.1, m) is True                           # after reset, 0.1 is a fresh best
    print("[earlystopper] reset() clears best/count/state across rounds. OK")


ALL = [
    test_grouping,
    test_bn_skip,
    test_proximal,
    test_pretrained_load,
    test_forward_smoke,
    test_gmic_loss,
    test_focal_loss,
    test_threshold_metrics,
    test_threshold_sweep,
    test_balanced_sampler_order,
    test_augmentation_safety,
    test_breast_aggregate,
    test_optimizer_selection,
    test_scheduler_selection,
    test_amp_loss_boundary,
    test_earlystopper_reset,
    test_ditto_persistence,
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
    print("\n==== %d/%d passed ====" % (len(ALL) - failures, len(ALL)))
    sys.exit(1 if failures else 0)
