#!/usr/bin/env python3
"""Ditto lambda sweep via the NVFLARE simulator on one multi-GPU host (e.g. the DGX).

For each ditto_lambda it renders a job from a base job (custom code) + the ditto config
templates, runs `nvflare simulator` with 3 simulated clients (HIPIMR/RSNA-AU/RSNA-US) each
pinned to its own GPU, collects each client's best PERSONAL-model val AUC, and reports the
lambda that maximizes the WORST-site AUC (the equity-thesis selection metric). All three
aggregates (worst_site / mean_site / pooled-as-mean) are logged.

Key design
----------
- Per-client isolation on the shared sim filesystem comes from the executor's
  data_path_map (CSVs) + preprocess_cache_dir_map (crops) + {site} path substitution.
- The crop cache is FIXED per site and SHARED across all lambda runs, so preprocessing
  happens once. A warmup run (num_rounds=1) builds the caches before any parallel runs to
  avoid a cold-cache write race; auto-skipped if the caches already exist.
- Up to len(gpu_pools) lambdas run concurrently (default 2), each pool driving one
  simulator process across 3 GPUs.

This script can't be exercised on the Windows clone (no GPUs / real data); run it on the
DGX. It only shells out to `nvflare simulator` and reads JSON the executor writes.

Modes
-----
- scalar (default): one ditto_lambda per run (method=ditto), ranked by worst-site val AUC.
- module-wise (--modulewise): method=ditto_modulewise with per-group lambdas {global,local,fusion}.
  A one-at-a-time sweep ANCHORED at --anchor (default 0.1, the scalar sweet spot): a baseline with
  all three groups at the anchor, then each group varied over --group-values while the others hold.
- fedprox (--fedprox): one fedprox_mu per run (method=fedprox), ranked by worst-site val AUC. Same
  harness/selection as the scalar Ditto sweep; the deployed/ranked model is the shared global w (no
  personal model). Use it to give FedProx the same mu search Ditto got for lambda.

Examples
--------
  # scalar lambda sweep
  python launcher.py --base-job ../<job> --lambdas 0.01,0.05,0.1,0.5,1.0,2.0 \
    --rounds 20 --gpu-pools "1,2,3;4,5,6" \
    --output-base /workspace/sim/ditto --work-root /workspace/sim/ditto_runs

  # fedprox mu sweep (fair counterpart to the ditto lambda sweep)
  python launcher.py --base-job ../<job> --fedprox --mus 0.001,0.01,0.05,0.1,0.5,1.0 \
    --rounds 20 --gpu-pools "1,2,3;4,5,6" \
    --output-base /workspace/sim/fedprox --work-root /workspace/sim/fedprox_runs

  # module-wise sweep anchored at the scalar best (0.1)
  python launcher.py --base-job ../<job> --modulewise --anchor 0.1 \
    --group-values 0.05,0.1,0.5,1.0 --rounds 20 --gpu-pools "1,2,3;4,5,6" \
    --output-base /workspace/sim/ditto_mw --work-root /workspace/sim/ditto_mw_runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time


# --------------------------------------------------------------------------- IO helpers
def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _lam_str(lam: float) -> str:
    """Stable, filename-safe label for a lambda value (0.1 -> '0.1', 1.0 -> '1')."""
    s = ("%g" % lam)
    return s


def _git_short(ref_dir):
    """Short HEAD hash (+ -dirty) of the repo containing ref_dir, or None."""
    try:
        h = subprocess.run(["git", "-C", ref_dir, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        if not h:
            return None
        dirty = subprocess.run(["git", "-C", ref_dir, "diff", "--quiet"]).returncode != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return None


# --------------------------------------------------------------------------- job rendering
def _base_cfg_paths(base_job):
    return (
        os.path.join(base_job, "app", "config", "config_fed_client.json"),
        os.path.join(base_job, "app", "config", "config_fed_server.json"),
        os.path.join(base_job, "meta.json"),
    )


def render_job(label, arg_overrides, rounds, base_job, work_root, output_base, clients):
    """Materialize a runnable simulator job dir for one run, identified by `label`.

    Uses base_job's OWN config as the single source (its app/config + meta.json) and copies
    its app/custom code verbatim. `arg_overrides` (e.g. {method, ditto_lambda} for scalar Ditto,
    or {method, lambda_global, lambda_local, lambda_fusion} for module-wise) plus the output/tb/log
    paths + num_rounds are overridden here; {site} is left intact for the executor to substitute per
    client. Outputs land under {output_base}/{label}/{site}. Returns the job dir path.
    """
    job_dir = os.path.join(work_root, f"job_{label}")
    app_dir = os.path.join(job_dir, "app")
    cfg_dir = os.path.join(app_dir, "config")
    os.makedirs(cfg_dir, exist_ok=True)

    # 1) custom code (executor, model, data_loader, ...) from the shared repo-root custom/
    #    (jobs no longer carry their own copy; custom/ is a sibling of base_job)
    src_custom = os.path.join(os.path.dirname(os.path.abspath(base_job)), "custom")
    dst_custom = os.path.join(app_dir, "custom")
    if os.path.isdir(dst_custom):
        shutil.rmtree(dst_custom)
    shutil.copytree(src_custom, dst_custom)
    # Stamp the repo commit so the simulator run is attributable (the copied custom/ has no
    # .git, so the executor's `git` probe would otherwise log git=unknown). _git_hash() reads
    # this file. base_job lives in the repo, so resolve HEAD from there.
    gh = _git_short(base_job)
    if gh:
        with open(os.path.join(dst_custom, "GIT_COMMIT"), "w") as f:
            f.write(gh + "\n")

    base_client, base_server, base_meta = _base_cfg_paths(base_job)

    # 2) client config: base_job's config + this run's overrides ({site} preserved)
    client_cfg = _load_json(base_client)
    args = client_cfg["executors"][0]["executor"]["args"]
    args.update(arg_overrides)
    run_root = f"{output_base}/{label}"
    args["output_dir"] = f"{run_root}/{{site}}"
    args["log_file"] = f"{run_root}/{{site}}/executor.log"
    _dump_json(os.path.join(cfg_dir, "config_fed_client.json"), client_cfg)

    # 3) server config: base_job's config + num_rounds / num_clients
    server_cfg = _load_json(base_server)
    wf_args = server_cfg["workflows"][0]["args"]
    wf_args["num_rounds"] = int(rounds)
    wf_args["num_clients"] = len(clients)
    _dump_json(os.path.join(cfg_dir, "config_fed_server.json"), server_cfg)

    # 4) meta.json
    meta = _load_json(base_meta)
    meta["min_clients"] = len(clients)
    meta["mandatory_clients"] = list(clients)
    _dump_json(os.path.join(job_dir, "meta.json"), meta)

    return job_dir


def _nvidia_smi(query, extra=()):
    out = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits", *extra],
                         capture_output=True, text=True, timeout=20)
    return out.stdout.strip().splitlines()


def gpu_free_gib():
    """{gpu_index(str): free_GiB} from nvidia-smi."""
    free = {}
    for line in _nvidia_smi("gpu=index,memory.free"):
        idx, mb = [x.strip() for x in line.split(",")]
        free[idx] = float(mb) / 1024.0
    return free


def gpu_compute_apps():
    """[(pid, name, used_MiB)] of all compute processes on the box (for spotting orphans)."""
    apps = []
    for line in _nvidia_smi("compute-apps=pid,process_name,used_memory"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            apps.append((parts[0], parts[1], parts[2]))
    return apps


def preflight_gpu_check(pools, min_free_gib):
    """Refuse to launch onto GPUs that aren't mostly free (orphaned workers / competing jobs are the
    usual cause of the 20-min OOM-then-NA failures). Returns True if OK to proceed."""
    target = sorted({g for pool in pools for g in pool}, key=lambda x: int(x) if x.isdigit() else x)
    try:
        free = gpu_free_gib()
    except Exception as e:
        print(f"[gpu-check] could not query nvidia-smi ({e}); skipping check")
        return True
    print("[gpu-check] target GPU free memory:")
    bad = []
    for g in target:
        f = free.get(str(g))
        print(f"  GPU {g}: {f:.1f} GiB free" if f is not None else f"  GPU {g}: (not reported by nvidia-smi)")
        if f is not None and f < min_free_gib:
            bad.append((g, f))
    if not bad:
        print(f"[gpu-check] OK (all target GPUs >= {min_free_gib} GiB free)")
        return True
    print(f"\n[gpu-check] BLOCKED: target GPU(s) below {min_free_gib} GiB free -> GMIC Ditto would OOM:")
    for g, f in bad:
        print(f"  GPU {g}: only {f:.1f} GiB free")
    # Point at GPUs that ARE usable so you can just retarget (--gpu-pools ...).
    usable = sorted((g for g, f in free.items() if f >= min_free_gib),
                    key=lambda x: int(x) if x.isdigit() else x)
    if usable:
        print(f"[gpu-check] GPUs with >= {min_free_gib} GiB free you can use instead: {','.join(usable)}")
    try:
        apps = gpu_compute_apps()
        if apps:
            # When names are '[Not Found]' the PIDs are in another namespace (host / another
            # container) and cannot be killed from in here -- don't suggest a futile kill.
            elsewhere = sum(1 for _, name, _ in apps if name in ("[Not Found]", "", "N/A"))
            print("[gpu-check] processes holding GPU memory (pid, name, MiB):")
            for pid, name, mem in apps:
                print(f"            {pid:>8}  {name:<28} {mem} MiB")
            if elsewhere == len(apps):
                print("[gpu-check] all names are '[Not Found]' -> these run on the HOST or another\n"
                      "            container, not here; you can't kill them from this shell. Either use\n"
                      "            the free GPUs above, or from the host: kill them / nvidia-smi --gpu-reset.")
            else:
                print("[gpu-check] likely orphaned simulator workers; free them with:  kill -9 "
                      + " ".join(p for p, name, _ in apps if name not in ("[Not Found]", "", "N/A")))
    except Exception:
        pass
    print("[gpu-check] (override with --skip-gpu-check once you're sure the GPUs are usable.)")
    return False


def build_runs(a):
    """Build the list of runs to sweep: each is {"label","display","overrides"}.

    Scalar (default): one run per --lambdas value, method=ditto.
    FedProx (--fedprox): one run per --mus value, method=fedprox (ranked by the shared global w's
    worst-site val AUC -- the fair mu counterpart to the scalar Ditto lambda sweep).
    Module-wise (--modulewise): a one-at-a-time sweep ANCHORED at --anchor (default 0.1, the global
    Ditto sweet spot). The baseline sets all three groups to the anchor (== scalar Ditto at the
    anchor); then each group {global, local, fusion} is varied over --group-values in turn while the
    other two stay at the anchor. method=ditto_modulewise.
    """
    if a.fedavg:
        # One FedAvg run that caches the per-round global trajectory (the anchor for Ditto replay).
        # Same simulator/data/seed as the ditto sweeps, so replay reproduces the interleaved baseline;
        # and being simulator-only it never touches RSNA-AU (no nightly crashes / resume needed).
        return [{"label": "fedavg", "display": "fedavg",
                 "overrides": {"method": "fedavg", "cache_incoming_global": True,
                               "cache_global_trajectory": True}}]
    # Sweeps select on the deployed/personal val AUC, so the per-round incoming-global eval +
    # trajectory caching (inherited from the base config) are pure wasted memory/time here -- and the
    # extra forward passes raise OOM risk. Disable them for sweep runs (only --fedavg needs them).
    no_cache = {"cache_incoming_global": False, "cache_global_trajectory": False}
    if a.fedprox:
        runs = []
        for mu in [float(x) for x in a.mus.split(",") if x.strip()]:
            ms = _lam_str(mu)
            runs.append({"label": f"mu{ms}", "display": f"mu={ms}",
                         "overrides": {"method": "fedprox", "fedprox_mu": mu, **no_cache}})
        return runs
    if a.fedbn:
        # FedBN has no hyperparameter (BN kept local on every load) -> a SINGLE run, not a sweep.
        # Ranked/reported like the others by the shared-backbone+local-BN model's per-site val AUC.
        return [{"label": "fedbn", "display": "fedbn",
                 "overrides": {"method": "fedbn", "use_fedbn": True, **no_cache}}]
    if not a.modulewise:
        runs = []
        for lam in [float(x) for x in a.lambdas.split(",") if x.strip()]:
            ls = _lam_str(lam)
            runs.append({"label": f"l{ls}", "display": ls,
                         "overrides": {"method": "ditto", "ditto_lambda": lam, **no_cache}})
        return runs

    anchor = float(a.anchor)
    values = [float(x) for x in a.group_values.split(",") if x.strip()]
    groups = ["global", "local", "fusion"]

    def mk(g, l, f):
        label = f"mw_g{_lam_str(g)}_l{_lam_str(l)}_f{_lam_str(f)}"
        return {"label": label, "display": f"g={_lam_str(g)} l={_lam_str(l)} f={_lam_str(f)}",
                "overrides": {"method": "ditto_modulewise",
                              "lambda_global": g, "lambda_local": l, "lambda_fusion": f, **no_cache}}

    seen, runs = set(), []
    base = mk(anchor, anchor, anchor)
    runs.append(base); seen.add(base["label"])
    for gi in groups:
        for v in values:
            trip = {"global": anchor, "local": anchor, "fusion": anchor}
            trip[gi] = v
            r = mk(trip["global"], trip["local"], trip["fusion"])
            if r["label"] not in seen:
                runs.append(r); seen.add(r["label"])
    return runs


def cache_paths(base_job, clients):
    """Per-site preprocess cache dirs from base_job's preprocess_cache_dir_map."""
    args = _load_json(_base_cfg_paths(base_job)[0])["executors"][0]["executor"]["args"]
    cmap = args.get("preprocess_cache_dir_map", {})
    return {c: cmap.get(c) for c in clients}


def caches_ready(base_job, clients):
    """True iff every site already has a built crop cache (processed_exam_list.pkl)."""
    for c, d in cache_paths(base_job, clients).items():
        if not d or not os.path.isfile(os.path.join(d, "processed_exam_list.pkl")):
            return False
    return True


# --------------------------------------------------------------------------- running
def simulator_cmd(job_dir, workspace, clients, gpus, threads):
    return [
        "nvflare", "simulator", job_dir,
        "-w", workspace,
        "-n", str(len(clients)),
        "-t", str(threads),
        "-c", ",".join(clients),
        "--gpu", ",".join(str(g) for g in gpus),
    ]


def launch(job_dir, workspace, clients, gpus, threads, log_path):
    """Start one simulator subprocess; returns (Popen, log_file_handle)."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    lf = open(log_path, "w")
    cmd = simulator_cmd(job_dir, workspace, clients, gpus, threads)
    lf.write("CMD: " + " ".join(cmd) + "\n")
    lf.flush()
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return proc, lf


def kill_run(proc, label):
    """Hard-kill a run AND its detached simulator workers. NVFlare's simulator_worker processes call
    setsid (session leaders), so they survive their parent and keep holding GPU memory -- killing only
    `proc` would leak them. Match every process whose cmdline references this run's workspace
    (ws_<label>/ or ws_<label> as an arg) and SIGKILL it. The label is unique per run, so this can't
    touch another run; the trailing [/ ] boundary keeps e.g. ws_l1 from matching ws_l10."""
    try:
        proc.kill()
    except Exception:
        pass
    try:
        out = subprocess.run(["pgrep", "-f", f"ws_{label}[/ ]"], capture_output=True, text=True, timeout=10).stdout
        for pid in out.split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass
    except Exception:
        pass


# --------------------------------------------------------------------------- metrics
def collect_metrics(output_base, label, clients):
    """Read each client's best personal-model val AUC for one run (subdir {output_base}/{label}).

    Looks for {site}_best_val_overall_gmic_metrics.json, falling back to {site}_final_results.json.
    A non-finite or <= 0 value is a SENTINEL (the personal model never produced a valid val AUC, so
    best_metrics stayed at its 0.0 default), NOT a real score -- it is reported as None ("NA") so it
    cannot masquerade as a catastrophic 0.0 and pollute worst_site/mean_site. Returns {site: auc_or_None}.
    """
    out = {}
    for c in clients:
        site_dir = os.path.join(output_base, label, c)
        auc = None
        primary = os.path.join(site_dir, f"{c}_best_val_overall_gmic_metrics.json")
        if os.path.isfile(primary):
            try:
                m = _load_json(primary)
                auc = m.get("val_auc", m.get("auc"))
            except Exception:
                auc = None
        if auc is None:
            fr = os.path.join(site_dir, f"{c}_final_results.json")
            if os.path.isfile(fr):
                try:
                    m = _load_json(fr)
                    auc = (m.get("best_val_auc")
                           or (m.get("best_metrics") or {}).get("val_auc"))
                except Exception:
                    auc = None
        ok = isinstance(auc, (int, float)) and math.isfinite(float(auc)) and float(auc) > 0.0
        out[c] = float(auc) if ok else None
    return out


def aggregate(per_site):
    """worst_site / mean_site over the sites that produced a metric."""
    vals = [v for v in per_site.values() if isinstance(v, (int, float))]
    if not vals:
        return {"worst_site": None, "mean_site": None, "n": 0}
    return {"worst_site": min(vals), "mean_site": sum(vals) / len(vals), "n": len(vals)}


# --------------------------------------------------------------------------- reporting
def print_report(results, clients, metric, title="DITTO LAMBDA SWEEP RESULTS"):
    sites_hdr = "  ".join(f"{c:>10}" for c in clients)
    wcfg = max(16, max((len(r["display"]) for r in results), default=8))
    width = wcfg + 26 + 12 * len(clients)
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"{'config':>{wcfg}}  {'worst_site':>10}  {'mean_site':>10}   {sites_hdr}")
    ranked = sorted(
        results,
        key=lambda r: (r["agg"].get(metric) if r["agg"].get(metric) is not None else -1),
        reverse=True,
    )
    for r in results:
        agg = r["agg"]
        ws = "%.4f" % agg["worst_site"] if agg["worst_site"] is not None else "   NA"
        ms = "%.4f" % agg["mean_site"] if agg["mean_site"] is not None else "   NA"
        per = "  ".join(
            ("%10.4f" % r["per_site"][c]) if isinstance(r["per_site"].get(c), (int, float))
            else f"{'NA':>10}" for c in clients
        )
        star = "  <- BEST" if (ranked and r is ranked[0] and agg.get(metric) is not None) else ""
        print(f"{r['display']:>{wcfg}}  {ws:>10}  {ms:>10}   {per}{star}")
    if ranked and ranked[0]["agg"].get(metric) is not None:
        best = ranked[0]
        print(f"\nBEST ({metric}): {best['display']}  {metric}={best['agg'][metric]:.4f}")
    else:
        print("\nNo config produced a usable metric (check run logs).")
    print("=" * width + "\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Ditto (scalar or module-wise) lambda sweep via the NVFLARE simulator.")
    ap.add_argument("--base-job", required=True,
                    help="ditto job dir providing app/config + app/custom (e.g. ../<job>). "
                         "Its config is the single source; the launcher overrides only the method + "
                         "lambda field(s), num_rounds, and the output/tb/log paths per run.")
    ap.add_argument("--lambdas", default="0.01,0.05,0.1,0.5,1.0,2.0",
                    help="scalar Ditto: comma-separated ditto_lambda values")
    ap.add_argument("--fedprox", action="store_true",
                    help="sweep method=fedprox proximal mu (--mus), ranked by worst-site val AUC")
    ap.add_argument("--mus", default="0.001,0.01,0.05,0.1,0.5,1.0",
                    help="fedprox: comma-separated fedprox_mu values")
    ap.add_argument("--fedbn", action="store_true",
                    help="single method=fedbn run (no hyperparameter; BN kept local), reported "
                         "per-site like the sweeps")
    ap.add_argument("--fedavg", action="store_true",
                    help="run ONE FedAvg job that caches the per-round global trajectory (the anchor "
                         "for Ditto replay via replay_sweep.py); no sweep, no RSNA-AU")
    ap.add_argument("--modulewise", action="store_true",
                    help="sweep method=ditto_modulewise per-group lambdas (one-at-a-time, anchored)")
    ap.add_argument("--anchor", type=float, default=0.1,
                    help="module-wise anchor: the value all three groups hold at (default 0.1, the "
                         "global Ditto sweet spot); each group is then varied over --group-values")
    ap.add_argument("--group-values", default="0.05,0.1,0.5,1.0",
                    help="module-wise: values to try for each group {global,local,fusion} in turn")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--gpu-pools", default="1,2,3;4,5,6",
                    help="';'-separated GPU triples; one concurrent run per pool")
    ap.add_argument("--clients", default="HIPIMR,RSNA-AU,RSNA-US")
    ap.add_argument("--threads", type=int, default=3, help="simulator -t (clients run concurrently)")
    ap.add_argument("--output-base", default="/workspace/sim/ditto",
                    help="output_dir base; executor appends /<run-label>/<site>")
    ap.add_argument("--work-root", default="/workspace/sim/ditto_runs",
                    help="where rendered job dirs + simulator workspaces + run logs go")
    ap.add_argument("--metric", default="worst_site", choices=["worst_site", "mean_site"])
    ap.add_argument("--prepare-cache", action="store_true",
                    help="ONLY build the per-site crop caches (1-round solo run), then exit. "
                         "This is the explicit data-prep step; the sweep itself is read-only.")
    ap.add_argument("--poll", type=int, default=20, help="seconds between scheduler polls")
    ap.add_argument("--dry-run", action="store_true", help="render jobs + print plan, don't run")
    ap.add_argument("--min-free-gib", type=float, default=10.0,
                    help="pre-flight: required free memory on each target GPU before launching "
                         "(GMIC Ditto peaks ~7 GiB/client); set 0 to disable the floor")
    ap.add_argument("--skip-gpu-check", action="store_true", help="bypass the pre-flight GPU check")
    ap.add_argument("--run-timeout-min", type=float, default=90.0,
                    help="kill a run (and its detached simulator workers) if it exceeds this many "
                         "minutes -- NVFlare sims can hang and camp on GPUs. Set above a healthy run's "
                         "wall time; 0 disables.")
    a = ap.parse_args()

    clients = [c.strip() for c in a.clients.split(",") if c.strip()]
    pools = [[g.strip() for g in pool.split(",") if g.strip()]
             for pool in a.gpu_pools.split(";") if pool.strip()]
    os.makedirs(a.work_root, exist_ok=True)

    runs = build_runs(a)
    mode = ("fedprox" if a.fedprox else "fedbn" if a.fedbn else
            "module-wise" if a.modulewise else "scalar")
    title = ("FEDPROX MU SWEEP RESULTS" if a.fedprox else
             "FEDBN RUN RESULTS" if a.fedbn else
             "DITTO MODULE-WISE SWEEP RESULTS" if a.modulewise else "DITTO LAMBDA SWEEP RESULTS")
    print(f"[sweep] mode={mode} runs={len(runs)} rounds={a.rounds} clients={clients}")
    if a.modulewise:
        print(f"[sweep] anchor={_lam_str(a.anchor)} group_values={a.group_values}")
    print(f"[sweep] gpu_pools={pools} (=> {len(pools)} concurrent runs)  metric={a.metric}")
    print(f"[sweep] output_base={a.output_base}  work_root={a.work_root}")

    # Pre-render all jobs (cheap; also validates base_job config early)
    for r in runs:
        r["job_dir"] = render_job(r["label"], r["overrides"], a.rounds, a.base_job,
                                  a.work_root, a.output_base, clients)
        print(f"[render] {r['display']} -> {r['job_dir']}")

    if a.dry_run:
        print("[dry-run] jobs rendered; not launching.")
        for r in runs:
            ws = os.path.join(a.work_root, f"ws_{r['label']}")
            print("  would run:", " ".join(simulator_cmd(r["job_dir"], ws, clients, pools[0], a.threads)))
        return

    # Pre-flight: fail in 2 seconds (not 20 minutes of OOM) if the target GPUs aren't free. Orphaned
    # NVFlare simulator workers from a prior sweep are the usual cause of an all-NA table.
    if a.min_free_gib > 0 and not a.skip_gpu_check:
        if not preflight_gpu_check(pools, a.min_free_gib):
            sys.exit(2)

    # Build the shared per-site crop caches ONCE (1-round solo run) before any parallel runs, if
    # they're missing -- preprocessing the (train/val/test SPLITS of the) data is fine; what the
    # sweep must never do is use TEST data for selection (it doesn't: ranked by VAL AUC, see
    # collect_metrics / the server IntimeModelSelector). Solo-first avoids two cold parallel runs
    # racing on the same cache. Skipped when caches already exist.
    def _build_cache():
        warm_job = render_job("cacheprep", runs[0]["overrides"], 1, a.base_job,
                              a.work_root, a.output_base + "_cacheprep", clients)
        ws = os.path.join(a.work_root, "ws_cacheprep")
        proc, lf = launch(warm_job, ws, clients, pools[0], a.threads,
                          os.path.join(a.work_root, "cacheprep.log"))
        rc = proc.wait()
        lf.close()
        return rc

    if a.prepare_cache:
        print(f"[prepare-cache] building per-site crop caches via a 1-round solo run (gpus={pools[0]})")
        rc = _build_cache()
        ok = caches_ready(a.base_job, clients)
        print(f"[prepare-cache] done rc={rc}; caches_ready={ok}")
        if not ok:
            print("[prepare-cache] WARNING: caches still not detected; check cacheprep.log.")
        return

    if not caches_ready(a.base_job, clients):
        print(f"[cache] per-site caches missing -> building once (1-round solo run, gpus={pools[0]})")
        rc = _build_cache()
        print(f"[cache] build done rc={rc}; caches_ready={caches_ready(a.base_job, clients)}")
    else:
        print("[cache] per-site caches present -> reusing (no preprocessing).")

    # Schedule the sweep across GPU pools (one running job per pool).
    pending = list(runs)
    running = {}  # pool_idx -> (run, proc, lf, start_ts)
    timeout_s = a.run_timeout_min * 60
    print(f"[run] starting sweep of {len(pending)} runs across {len(pools)} pools "
          f"(per-run timeout {a.run_timeout_min} min)")
    try:
        while pending or running:
            for pi in range(len(pools)):
                if pi not in running and pending:
                    r = pending.pop(0)
                    ws = os.path.join(a.work_root, f"ws_{r['label']}")
                    log_path = os.path.join(a.work_root, f"run_{r['label']}.log")
                    proc, lf = launch(r["job_dir"], ws, clients, pools[pi], a.threads, log_path)
                    running[pi] = (r, proc, lf, time.time())
                    print(f"[run] launched {r['display']} on pool{pi}={pools[pi]} -> {log_path}")
            for pi in list(running.keys()):
                r, proc, lf, t0 = running[pi]
                rc = proc.poll()
                if rc is None and timeout_s > 0 and (time.time() - t0) > timeout_s:
                    # Hung run (NVFlare simulator can spin forever): kill it + its detached workers so
                    # it stops camping on the GPUs. The config is left with no result file -> NA.
                    print(f"[run] {r['display']} on pool{pi} TIMED OUT after {a.run_timeout_min} min "
                          f"-> killing (hung); freeing pool")
                    kill_run(proc, r["label"])
                    lf.close()
                    del running[pi]
                    continue
                if rc is not None:
                    lf.close()
                    dt = time.time() - t0
                    print(f"[run] {r['display']} on pool{pi} finished rc={rc} in {dt/60:.1f} min")
                    del running[pi]
            if running:
                time.sleep(a.poll)
    finally:
        # Never leave detached workers behind (e.g. on Ctrl-C) -- that is what camps on the GPUs.
        for pi, (r, proc, lf, t0) in list(running.items()):
            if proc.poll() is None:
                print(f"[run] cleanup: killing still-running {r['display']}")
                kill_run(proc, r["label"])
            try:
                lf.close()
            except Exception:
                pass

    # FedAvg trajectory run: no sweep to collect -- just point at the cached global trajectory.
    if a.fedavg:
        print("\n[fedavg] done. Per-round global trajectory (anchor for Ditto replay) cached at:")
        for c in clients:
            print(f"  {c}: {a.output_base}/fedavg/{c}/incoming_global/{c}_incoming_global_round_*.pth")
        print("\nNext: python ditto_sweep/replay_sweep.py --base-job %s \\\n"
              "        --traj-dir %s/fedavg/%s/incoming_global --traj-prefix %s --lambdas 0.05,0.1,0.5 \\\n"
              "        --clients %s --gpu 0 --out %s/ditto_replay_summary.json"
              % (a.base_job, a.output_base, clients[0], clients[0], ",".join(clients), a.work_root))
        return

    # Collect + report
    for r in runs:
        r["per_site"] = collect_metrics(a.output_base, r["label"], clients)
        r["agg"] = aggregate(r["per_site"])
    print_report(runs, clients, a.metric, title=title)

    summary_path = os.path.join(a.work_root, "sweep_summary.json")
    _dump_json(summary_path, {
        "mode": mode, "metric": a.metric, "rounds": a.rounds, "clients": clients,
        "anchor": a.anchor if a.modulewise else None,
        "results": [{k: r[k] for k in ("label", "display", "overrides", "per_site", "agg")} for r in runs],
    })
    print(f"[sweep] summary written -> {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
