#!/usr/bin/env python3
"""Ditto sweep by REPLAY against a cached FedAvg global trajectory -- one FedAvg run, N cheap Ditto
fits (no FedAvg recompute, no federation).

Background. Ditto's global model w is trained by ordinary FedAvg; the personal model v is trained with
task_loss(v) + (lambda/2)||v - w^t||^2, anchored each round to the start-of-round global w^t. The
personal pass never influences the global, so ONE FedAvg run's per-round global trajectory is the
correct anchor for ANY lambda. `replay_personalized` reproduces the interleaved personal trajectory
from that cached trajectory; `test_ditto_replay_equivalence.py` proves the anchor mapping is exact.

This driver does, per site, per lambda-config: rebuild the personal model v from the pretrained init,
then for each round t train v for `epochs` local epochs anchored to trajectory[t], evaluating val AUC
each round and keeping the best (the deployed = best-val-of-round convention). It then ranks configs
by worst-site (or mean-site) val AUC, exactly like ditto_sweep/launcher.py -- but without re-running
FedAvg for each lambda.

Anchor trajectory. trajectory[t] = the global broadcast at the START of round t = w^{t-1}, which is
exactly what the executor caches as `incoming_global/{prefix}_incoming_global_round_{t}.pth` (round 0
= the pretrained seed). The broadcast global is identical at every site, so any one site's
incoming_global cache IS the global trajectory -- pass its dir + prefix via --traj-dir/--traj-prefix.
(Run the FedAvg job with cache_incoming_global=true to produce it.)

It reuses the executor's EXACT data loader, pretrained model, loss, optimizer, and eval (constructs a
GMICFederatedExecutor per site and calls initialize()), so the only new logic is the replay loop.
Determinism note: replay reproduces interleaved Ditto's anchor DYNAMICS exactly and is reproducible,
but it does not bit-match a real interleaved run's balanced-sampler draws (the interleaved global pass
would have advanced the RNG first) -- that is sampling noise, not bias, and is fine for lambda
selection on val AUC.

Runs on the DGX (needs GPU + the per-site crop caches the FedAvg run used). Cannot be exercised on the
CPU clone. Sanity check after launch: each site's round-0 val AUC should ~= its pretrained baseline
(v is the pretrained init, anchored to itself, so round 0 is one epoch of light local adaptation).

Example
-------
  # scalar lambda sweep, replaying HIPIMR's cached global trajectory
  python replay_sweep.py --base-job ../<job> \
    --traj-dir /workspace/sim/fedavg/HIPIMR/incoming_global --traj-prefix HIPIMR \
    --lambdas 0.05,0.1,0.5 --clients HIPIMR,RSNA-AU,RSNA-US --gpu 0 \
    --out /workspace/sim/ditto_replay/sweep_summary.json

  # module-wise, anchored at 0.1, global tethered upward / heads freer
  python replay_sweep.py --base-job ../<job> --traj-dir ... --traj-prefix HIPIMR --modulewise \
    --anchor 0.1 --global-values 0.1,0.5,1.0 --local-values 0.01,0.05,0.1 --fusion-values 0.01,0.05,0.1
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
import tempfile

import torch

# Portable scratch dir for the executor's (unused-by-replay) output/log paths -- /tmp does not exist
# on Windows (the RSNA-AU node), so derive from the OS temp dir.
_SCRATCH = os.path.join(tempfile.gettempdir(), "ditto_replay_scratch")

# Estimators/components live in the shared repo-root custom/ (a sibling of base_job); path it.
def _add_custom_to_path(base_job):
    custom = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(base_job)), "custom"))
    if custom not in sys.path:
        sys.path.insert(0, custom)
    return custom


def _lam_str(x):
    return "%g" % float(x)


def load_client_args(base_job):
    cfg = os.path.join(base_job, "app", "config", "config_fed_client.json")
    with open(cfg) as f:
        return json.load(f)["executors"][0]["executor"]["args"]


def load_trajectory(traj_dir, prefix, device):
    """Ordered [state_dict] anchor trajectory from {prefix}_incoming_global_round_{t}.pth (t=0..)."""
    pat = re.compile(rf"{re.escape(prefix)}_incoming_global_round_(\d+)\.pth$")
    found = {}
    for p in glob.glob(os.path.join(traj_dir, f"{prefix}_incoming_global_round_*.pth")):
        m = pat.search(os.path.basename(p))
        if m:
            found[int(m.group(1))] = p
    if not found:
        raise FileNotFoundError(f"no {prefix}_incoming_global_round_*.pth under {traj_dir}")
    rounds = sorted(found)
    # require contiguous 0..max so anchor[t] maps to round t (a gap would silently shift anchors)
    if rounds != list(range(rounds[0], rounds[-1] + 1)) or rounds[0] != 0:
        raise ValueError(f"trajectory rounds not contiguous from 0: {rounds}")
    traj = []
    for t in rounds:
        sd = torch.load(found[t], map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and not any(torch.is_tensor(v) for v in sd.values()):
            for wrap in ("model", "state_dict", "model_dict"):
                if isinstance(sd.get(wrap), dict):
                    sd = sd[wrap]
                    break
        traj.append(sd)
    return traj


def build_runs(a):
    """List of {label, display, lam} where lam is a float (scalar) or {global,local,fusion} dict."""
    if not a.modulewise:
        out = []
        for lam in [float(x) for x in a.lambdas.split(",") if x.strip()]:
            out.append({"label": f"l{_lam_str(lam)}", "display": _lam_str(lam), "lam": lam})
        return out
    anchor = float(a.anchor)
    gv = [float(x) for x in (a.global_values or a.group_values).split(",") if x.strip()]
    lv = [float(x) for x in (a.local_values or a.group_values).split(",") if x.strip()]
    fv = [float(x) for x in (a.fusion_values or a.group_values).split(",") if x.strip()]

    def mk(g, l, f):
        return {"label": f"mw_g{_lam_str(g)}_l{_lam_str(l)}_f{_lam_str(f)}",
                "display": f"g={_lam_str(g)} l={_lam_str(l)} f={_lam_str(f)}",
                "lam": {"global": g, "local": l, "fusion": f}}

    seen, runs = set(), []
    for r in [mk(anchor, anchor, anchor)] + \
             [mk(v, anchor, anchor) for v in gv] + \
             [mk(anchor, v, anchor) for v in lv] + \
             [mk(anchor, anchor, v) for v in fv]:
        if r["label"] not in seen:
            seen.add(r["label"]); runs.append(r)
    return runs


def build_site(base_job, args, site, device):
    """Construct + initialize the executor for one site; return the wired object."""
    from bc_executor import GMICFederatedExecutor
    a = dict(args)
    a["device"] = device
    a["gpus"] = device.split(":")[-1] if ":" in device else "0"
    # scratch output paths so we never touch the real run dirs (replay writes nothing here anyway)
    a["output_dir"] = os.path.join(_SCRATCH, "{site}")
    a["log_file"] = os.path.join(_SCRATCH, "{site}", "replay.log")
    ex = GMICFederatedExecutor(**a)
    ex._identity = site
    ex.initialize()
    return ex


def replay_site(ex, trajectory, lam, device, n_rounds):
    """Replay Ditto for one site at one lambda; return best val AUC + the round it occurred."""
    from train.training_core import configure_optimizers, evaluate_model
    from fl_utils import proximal_penalty, replay_personalized

    v = copy.deepcopy(ex._underlying).to(device)
    opt = configure_optimizers(v, ex._opt_args)[0]
    loss_fn = ex._compute_task_loss            # (outputs, targets) -> task loss (no fl_ctx)
    best_auc, best_round = float("-inf"), -1
    for t in range(n_rounds):
        replay_personalized(
            v_model=v, v_optimizer=opt, anchor_trajectory=[trajectory[t]],
            loss_fn=loss_fn, proximal_fn=proximal_penalty, lam=lam, device=device,
            n_rounds=1, e_personalized=int(ex.epochs),
            batch_iter_factory=lambda: ex.data_loader.get_batch_iterator("train"),
        )
        core = evaluate_model(v, ex.data_loader, ex.criterion, device, split="val",
                              decision_threshold=ex.decision_threshold)
        auc = float(core.get("auc", float("nan")))
        if auc == auc and auc > best_auc:   # auc==auc filters NaN
            best_auc, best_round = auc, t
    return best_auc, best_round


def aggregate(per_site):
    vals = [v for v in per_site.values() if isinstance(v, (int, float)) and v == v]
    if not vals:
        return {"worst_site": None, "mean_site": None}
    return {"worst_site": min(vals), "mean_site": sum(vals) / len(vals)}


def print_report(results, clients, metric):
    ranked = sorted(results, key=lambda x: (x["agg"].get(metric) if x["agg"].get(metric) is not None else -1),
                    reverse=True)
    wcfg = max(16, max((len(r["display"]) for r in results), default=8))
    hdr = "  ".join(f"{c:>10}" for c in clients)
    print("\n" + "=" * (wcfg + 26 + 12 * len(clients)))
    print("DITTO REPLAY SWEEP RESULTS")
    print("=" * (wcfg + 26 + 12 * len(clients)))
    print(f"{'config':>{wcfg}}  {'worst_site':>10}  {'mean_site':>10}   {hdr}")
    for r in results:
        ws = "%.4f" % r["agg"]["worst_site"] if r["agg"]["worst_site"] is not None else "   NA"
        ms = "%.4f" % r["agg"]["mean_site"] if r["agg"]["mean_site"] is not None else "   NA"
        per = "  ".join(("%10.4f" % r["per_site"][c]) if r["per_site"].get(c) == r["per_site"].get(c)
                        else f"{'NA':>10}" for c in clients)
        star = "  <- BEST" if r is ranked[0] and r["agg"].get(metric) is not None else ""
        print(f"{r['display']:>{wcfg}}  {ws:>10}  {ms:>10}   {per}{star}")
    if ranked and ranked[0]["agg"].get(metric) is not None:
        print(f"\nBEST ({metric}): {ranked[0]['display']}  {metric}={ranked[0]['agg'][metric]:.4f}")
    print("=" * (wcfg + 26 + 12 * len(clients)) + "\n")


def combine_summaries(paths, metric):
    """Merge per-node replay summaries (REAL-WORLD path: each site ran on its own node) into one
    cross-site table. Per config label, union the per_site maps across files, then recompute
    worst/mean over all sites and select. Configs are matched by label, so run the SAME lambdas/
    grid on every node."""
    by_label = {}
    clients = []
    for p in paths:
        with open(p) as f:
            s = json.load(f)
        for c in s.get("clients", []):
            if c not in clients:
                clients.append(c)
        for r in s["results"]:
            slot = by_label.setdefault(r["label"], {"label": r["label"], "display": r["display"], "per_site": {}})
            slot["per_site"].update({k: v for k, v in r.get("per_site", {}).items()
                                     if isinstance(v, (int, float))})
    results = []
    for slot in by_label.values():
        results.append({**slot, "agg": aggregate(slot["per_site"])})
    print_report(results, clients, metric)
    return results, clients


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-job", help="job dir providing app/config + app/custom (required unless --combine)")
    ap.add_argument("--traj-dir", help="dir with {prefix}_incoming_global_round_*.pth (required unless --combine)")
    ap.add_argument("--traj-prefix", help="site prefix of the cached global trajectory (required unless --combine)")
    ap.add_argument("--clients", default="HIPIMR,RSNA-AU,RSNA-US")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--rounds", type=int, default=0, help="replay rounds (0 = all cached rounds)")
    ap.add_argument("--lambdas", default="0.05,0.1,0.5")
    ap.add_argument("--modulewise", action="store_true")
    ap.add_argument("--anchor", type=float, default=0.1)
    ap.add_argument("--group-values", default="0.05,0.1,0.5,1.0")
    ap.add_argument("--global-values", default=None)
    ap.add_argument("--local-values", default=None)
    ap.add_argument("--fusion-values", default=None)
    ap.add_argument("--metric", default="worst_site", choices=["worst_site", "mean_site"])
    ap.add_argument("--out", default=None, help="write summary JSON here")
    ap.add_argument("--combine", default=None,
                    help="REAL-WORLD: comma-separated per-node summary JSONs (or a glob) to merge into "
                         "one cross-site selection table. With this, no replay is run.")
    a = ap.parse_args()

    # Real-world combine: each node ran replay for its own site; merge their summaries + select.
    if a.combine:
        paths = []
        for tok in a.combine.split(","):
            paths.extend(sorted(glob.glob(tok.strip())) if any(ch in tok for ch in "*?[") else [tok.strip()])
        results, clients = combine_summaries([p for p in paths if p], a.metric)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
            with open(a.out, "w") as f:
                json.dump({"mode": "combined", "metric": a.metric, "clients": clients, "results": results}, f, indent=2)
            print(f"[replay] combined summary -> {a.out}")
        return

    if not (a.base_job and a.traj_dir and a.traj_prefix):
        ap.error("--base-job, --traj-dir, --traj-prefix are required (unless --combine)")
    _add_custom_to_path(a.base_job)
    args = load_client_args(a.base_job)
    clients = [c.strip() for c in a.clients.split(",") if c.strip()]
    device = a.gpu if a.gpu.startswith("cuda") else f"cuda:{a.gpu}"
    runs = build_runs(a)

    trajectory = load_trajectory(a.traj_dir, a.traj_prefix, device)
    n_rounds = a.rounds if a.rounds > 0 else len(trajectory)
    n_rounds = min(n_rounds, len(trajectory))
    print(f"[replay] {'module-wise' if a.modulewise else 'scalar'} sweep: {len(runs)} configs x "
          f"{len(clients)} sites, {n_rounds} replay rounds from {a.traj_prefix} trajectory "
          f"(len {len(trajectory)})", flush=True)

    # Build each site's executor ONCE (loads data + pretrained), reuse across all configs.
    sites = {}
    for c in clients:
        print(f"[replay] initializing site {c} ...", flush=True)
        sites[c] = build_site(a.base_job, args, c, device)

    results = []
    for r in runs:
        per_site = {}
        for c in clients:
            auc, rnd = replay_site(sites[c], trajectory, r["lam"], device, n_rounds)
            per_site[c] = auc
            print(f"[replay] {r['display']:<22} {c:<10} best_val_auc={auc:.4f} @round {rnd}", flush=True)
        agg = aggregate(per_site)
        results.append({**r, "per_site": per_site, "agg": agg})

    print_report(results, clients, a.metric)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"mode": "module-wise" if a.modulewise else "scalar", "metric": a.metric,
                       "n_rounds": n_rounds, "clients": clients, "traj_prefix": a.traj_prefix,
                       "results": results}, f, indent=2)
        print(f"[replay] summary -> {a.out}")


if __name__ == "__main__":
    main()
