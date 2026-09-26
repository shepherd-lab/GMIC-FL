# Resuming an interrupted federated run after a crash

A memory-constrained site (e.g. a 16 GiB RTX A4000 running the fp32 personal pass — it overflows
to ~22 GiB, oversubscribes host RAM, thrashes at ~2.3 hr/round) can make a long run progress slowly
and die every ~20 rounds. That is accepted: the plan is crash → resume → repeat until all rounds
finish. This runbook covers **two** recovery paths — pick by *what survived the crash*:

- **A. Client checkpoints survived (usual case).** Restart the same job pointing at the last
  completed round; clients re-submit their cached weights and the server re-aggregates. This is the
  built-in `resume_from_local_round` path in the executor.
- **B. The server lost its aggregated global** (its transient run dir was deleted, but the client
  round-N checkpoints exist). Rebuild the global offline and warm-start from it with
  `warmstart_persistor` (swapped in for the default persistor). See the last section.

Both are safe to repeat any number of times; logical round numbering stays contiguous.

## Why path A resumes cleanly

A client returns its trained global to the server only at the END of `execute()` — after the
personal pass and the `{site}_gmic_model_round_N.pth` save. So the server aggregates round N (and
advances to N+1) only once **every** site has finished round N and written both:

- `global_trajectory/{site}_global_round_N.pth` — the sent global `w^N`, read by the reseed (Ditto
  reads this, NOT the deployed ckpt, because the deployed model is the personal `v`).
- `{site}_gmic_model_round_N.pth` — the deployed personal model `v` at round N, reloaded so `v`
  continues instead of cold-starting.

Requires `cache_global_trajectory: true` (set in the Ditto configs) and an `output_dir` on
persistent storage.

## Path A — steps

Job folders are per-method (`ditto/`, `ditto_mw/`, `fedprox/`, …); edit the config in the folder
you are running. (Remember the code is copied in per the README's "Running a job": `cp -r custom
<job>/app/custom` before submitting.)

1. **Find N** — the last round completed by ALL sites. Run the helper against the run's results dir:

   ```
   python tools/resume_params.py /workspace/results/<method>/<...>     # the run's output_dir
   ```

   It prints `resume_from_local_round` and the server `num_rounds`. (Manual equivalent: the highest
   `{site}_gmic_model_round_N.pth` present for the crash-prone site — its highest is the safe minimum.)

2. **Edit `<job>/app/config/config_fed_client.json`:**
   - `resume_from_local_round: N`
   - leave `output_dir` and `resume_ckpt_dir` unchanged (the reseed reads the round-N ckpts there).
   - leave `personal_amp_by_site`, `use_amp`, etc. as-is (precision / per-site settings are preserved).

3. **Edit `<job>/app/config/config_fed_server.json`:**
   - `num_rounds: <total> - N`  (raw round 0 reseeds round N; raw 1..(total-1-N) train logical
     N+1..total-1, landing the final-round test at logical total-1 = the original plan).

4. **Resubmit** the job. At raw round 0 each client re-submits its cached round-N weights UNTRAINED;
   the server's weighted aggregation reconstructs the round-N global, `v` is restored from its
   round-N ckpt, and training continues at logical N+1. Confirm in the log:
   - `[resume] loaded <site> round-N weights ... submitting UNTRAINED for server re-aggregation`
   - `[resume] restored personal model v <- ...`  (the `v` optimizer's Adam moments are rebuilt
     fresh — never persisted — and re-warm within a few steps; this is the only lossy part.)

5. When it dies again, repeat from step 1. Each resume finds the new (higher) N from the files the
   previous segment saved, so numbering stays contiguous to the final round.

## Path B — server lost the global (`warmstart_persistor`)

If only the client round-N checkpoints survived, reconstruct the global and warm-start:

1. Gather the three clients' `{site}_gmic_model_round_N.pth` under a dir the **server** can read
   (e.g. under its mounted `/workspace`).
2. In `<job>/app/config/config_fed_server.json`, replace the `persistor` component's
   `nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor` with
   `warmstart_persistor.WarmStartPersistor` (kept in `custom/`, so it resolves like any component),
   and set `warmstart_round: N` and `warmstart_dir: <that dir>`. `warmstart_round: -1` = ordinary
   behavior.
3. It computes the sample-weighted mean of the clients' round-N files (identical to what the server
   would have broadcast at N+1 — FedProx/FedBN aggregate the same way, the proximal/BN handling is
   client-side), writes a flat state_dict, and points the persistor's source at it. NVFLARE
   broadcasts that reconstructed global at round 0 and the run continues for the remaining rounds.
   (The offline re-aggregation itself lives in `custom/resume_aggregate.py`.)

## Notes

- If a site shows `personal ckpts without a global_trajectory match`, that round can't be reseeded
  for it — `resume_params.py` already excludes it and picks a lower safe N.
- After the FINAL segment reaches the last logical round, run `pool_report_job` (repointed at this
  results dir, with the run's method tag in the method list) for the pooled personal-model endpoint.
