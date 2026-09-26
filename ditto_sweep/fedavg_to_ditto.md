# One FedAvg run → Ditto by replay (no FedAvg recompute per λ)

Ditto's global model is trained by plain FedAvg; the personal model `v` is trained with
`task_loss(v) + (λ/2)‖v − w^t‖²`, anchored each round to the start-of-round global `w^t`. The personal
pass never affects the global, so **one FedAvg run's per-round global trajectory is the correct anchor
for every λ**. `replay_personalized` reproduces the interleaved personal trajectory from that cache —
proven bit-identical in `test_ditto_replay_equivalence.py`. So: run FedAvg once, replay Ditto N times.

The global trajectory is the per-round broadcast global, cached by `cache_incoming_global` as
`incoming_global/<site>_incoming_global_round_{t}.pth` (`round_t` = `w^{t-1}`; round 0 = pretrained
seed). It is **identical at every site**, so each node already holds the whole trajectory locally.

---

## REAL-WORLD path (the 3 physical nodes)

### 1. FedAvg real-world — `<job>`

In `<job>/app/config/config_fed_client.json` set:
- `"method": "fedavg"` (Ditto's global is FedAvg, not FedProx)
- `"cache_incoming_global": true` (the anchor trajectory — keep all rounds)
- remove/empty `"salvage_eval_rounds"` (so it's a normal training run, not salvage)
- `"resume_from_local_round": -1` (fresh start)

In `config_fed_server.json` set `"num_rounds"` to your target (e.g. 60), aggregator/shareable_generator
stock (already reverted). Submit the job. Each node writes its `incoming_global/` trajectory to its
`results_dir`.

**Surviving nightly RSNA-AU crashes (numbering is now resume-safe).** Every per-round artifact is stamped
`current_round + resume_from_local_round`, so a resumed segment continues the logical numbering instead
of overwriting round 0. Per crash:
1. Find the last **completed** logical round `N` (highest `incoming_global_round_{N}.pth`).
2. Set `"resume_from_local_round": N` and `"resume_ckpt_dir"` → the run dir holding the cached weights.
3. Resubmit. Round 0 reseeds (server reconstructs the round-`N` global, train-size weighted); round 1
   = logical `N+1`; trajectory continues `N+1, N+2, …`.
4. Repeat with `N` = the new last-completed round each time.

Result: one contiguous `incoming_global_round_{0..M}.pth` on every node.

### 2. Ditto real-world — replay, ON EACH NODE

Data is siloed, so the replay runs **per node** on that node's local data + its own cached trajectory
(no federation, no FedAvg recompute). Copy `ditto_sweep/replay_sweep.py` into each node's job
environment (it imports the same `app/custom`) and run, with `--clients` = that one site:

```bash
# on the HIPIMR node
python replay_sweep.py --base-job <<job>> --clients HIPIMR \
  --traj-dir <run>/incoming_global --traj-prefix HIPIMR \
  --lambdas 0.05,0.1,0.5 --gpu 0 --out uhcc_replay.json
# on the RSNA-AU node (Windows OK)        -> --clients RSNA-AU --traj-prefix RSNA-AU --out hpu_replay.json
# on the RSNA node                    -> --clients RSNA-US --traj-prefix RSNA-US --out rsna_replay.json
```

Run the **same** `--lambdas` (or the same module-wise grid: `--modulewise --anchor 0.1
--global-values 0.1,0.5,1.0 --local-values 0.01,0.05,0.1 --fusion-values 0.01,0.05,0.1`) on every node
so the configs align. Each node writes a small per-site summary JSON (just AUCs — no patient data).

### 3. Combine + select (off the nodes)

Gather the three summary JSONs and merge into the cross-site worst/mean-site table:

```bash
python replay_sweep.py --combine "uhcc_replay.json,hpu_replay.json,rsna_replay.json" \
  --metric worst_site --out ditto_replay_combined.json
```

That prints the same sweep table as the interleaved launcher and picks the worst-site-best λ.

**Sanity check:** each node's round-0 replay val AUC ≈ its pretrained baseline; the module-wise
`g=l=f=0.1` baseline ≈ scalar λ=0.1.

### Fallback: interleaved Ditto real-world (no replay)

Fully wired (`"method": "ditto"`, `"ditto_lambda": L`) — but each λ is a **full federated run that
recomputes FedAvg** and is subject to RSNA-AU crashes (resume via the same offset procedure). Use only if
the per-node replay is impractical; it's N× the cost.

---

## Last-ditch: everything in the simulator (no RSNA-AU)

If real-world can't finish in time, run the whole thing in the simulator on the DGX (all sites' data
local) — same data/seed, so results are consistent, and RSNA-AU is out of the loop entirely:

```bash
# one FedAvg run that caches the trajectory
python ditto_sweep/launcher.py --base-job ../<job> --fedavg --rounds 60 \
  --gpu-pools "0,1,2" --output-base /workspace/sim/fedavg --work-root /workspace/sim/fedavg_runs
# central replay (all sites on one box -> no --combine needed)
python ditto_sweep/replay_sweep.py --base-job ../<job> \
  --traj-dir /workspace/sim/fedavg/fedavg/HIPIMR/incoming_global --traj-prefix HIPIMR \
  --lambdas 0.05,0.1,0.5 --clients HIPIMR,RSNA-AU,RSNA-US --gpu 0 --out /workspace/sim/ditto_replay.json
```

## Why replay == interleaved (and the one caveat)

`test_ditto_replay_equivalence.py` proves the anchor mapping and v-carried-forward reproduce
interleaved Ditto **bit-for-bit** on fixed data. With the real balanced sampler, replay won't
bit-match a live interleaved run (its global pass would have advanced the RNG first) — sampling noise,
not bias, and fine for λ selection on val AUC.
