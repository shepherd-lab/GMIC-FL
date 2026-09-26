# FedProx salvage: reconstruct per-round globals, score test, compute paper stats

Salvage an interrupted run with no resume/retraining. A test-only federated job reconstructs the
shared global at chosen rounds (train-size-weighted aggregation of the per-site SENT weights),
scores val+test, and **computes + logs each site's per-site paper stats during the run, on the client**
(breast-level AUC + DeLong CI, Youden(val)→test sens/spec + Wilson CIs) — `[salvage-stats]`.

**POOLED is computed offline**, not in-run. (The server uses the `nvflare ... FedAvg` ModelController
workflow, which does its own built-in aggregation and ignores custom `aggregator`/`shareable_generator`
components — so there is no server-side hook to fuse the sites mid-run.) This is fine: pooling is an
exact function of the per-site predictions, which land on every client, so `tools/salvage_stats.py`
reproduces the pooled row (and re-derives every per-site row) from the gathered CSVs.

## Where the output lands (on each client, persists with the client job folder)

`results_dir` = `/workspace/data/processed/<run>/`:
- `{site}_predictions_{method}_round{N}_{val,test}.csv` — raw breast/view predictions (pooling source).
- `{site}_salvage_{method}_round{N}_metrics.json` — that site's computed per-site row (durable file).
- client log: `[salvage-stats] method=... src_round=...`.

The estimators live in `.../salvage_metrics.py` (shared by the client executor and the offline tool);
the offline path is proven to reproduce the in-run per-site numbers exactly.

## What gets evaluated, per configured round N

- `incoming_global` — the **reconstructed shared global** built from round-N contributions (same model
  at every site → per-site **and** pooled). Indexing: aggregating round-N SENT weights yields the
  global that *entered round N+1*; files/logs are labeled by source round N. Select the pooled-best by
  each global's own **val** AUC (logged as `valAUC=`).
- `deployed_local` — each site's **as-deployed** model at round N (a different model per site →
  per-site only, never pooled).

## Config (already set in `<job>`)

`config_fed_client.json` (executor args):
- `"salvage_eval_rounds": [6, 8, 10, 14, 23, 26]` — shortlist = pooled candidates {6,8,10} ∪ per-site
  bests {14 RSNA-AU, 23 HIPIMR, 26 last}. Empty/absent ⇒ normal training (salvage off).
- `"salvage_ckpt_dir": "/workspace/data/processed/fedprox_3client_fixedinit_20260609"` — the
  interrupted run's dir (holds `global_trajectory/{site}_global_round_{N}.pth`, the authoritative
  SENT weights; falls back to `{site}_gmic_model_round_{N}.pth`). Defaults to `results_dir` if unset.

`config_fed_server.json`:
- `"num_rounds": 7` — `len(salvage_eval_rounds) + 1` (the trailing round evaluates the last
  reconstruction). **Revert to your training value (e.g. 40+) for a normal run.**

No training happens; each round only loads weights and runs eval, so the RSNA-AU-dropout window is
seconds. Weighting is the freshly-preprocessed `train_size` (matches the original aggregation at
epochs=1) — computed automatically per site, nothing to supply.

## Run it

Submit the `<job>` job. Read the per-site results off each client, then pool offline:

```
# on each client (results_dir persists with the client job folder)
cat  <run>/{site}_salvage_*_metrics.json          # that node's per-site rows (durable)
grep "\[salvage-stats\]"  <client log>            # same rows in the log

# pooled + a clean combined table, offline from the gathered prediction CSVs
python tools/salvage_stats.py --pred-dir <gathered_csvs> --out-prefix results/fedprox_salvage
```

**Built-in pipeline check (free):** `deployed_local` val AUC at round N reproduces the original logged
`deployed_val_auc[N]` (epochs=1: sent == best-val deployed). E.g. HIPIMR `deployed_local` src_round=23
val AUC ≈ 0.8924. If it matches, splits/loading/eval are faithful end-to-end.

## Offline stats (pooled + combined table)

Gather every `*_predictions_*.csv` from the nodes into one directory, then:

```bash
python tools/salvage_stats.py --pred-dir <gathered_csvs> --out-prefix results/fedprox_salvage
```

Writes `<prefix>_metrics.csv` and `<prefix>_table.md` with per-site **and** pooled rows. Validate
estimators with `--selftest`. (The pooled row is computed only here — see the note at top on why.)

## Reporting

Pick the **pooled-best `incoming_global` round** by pooled val AUC; report its pooled test row (the
FedProx bracket baseline) + per-site rows (heterogeneity panel). For the per-site personalization
view, use each site's own best round under `deployed_local` and/or `incoming_global`.
