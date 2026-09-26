# Ditto λ sweep (NVFLARE simulator, single multi-GPU host)

Finds the Ditto `ditto_lambda` (personalization ↔ pull-to-global trade-off) that maximizes
the **worst-site** personal-model **val** AUC, by running one simulator job per λ on the DGX.
All other hyperparameters match the federated run (loss=gmic, epochs=1/round, lr 3e-5/3e-7,
freeze 2, balanced sampler, pos_weight 1.0).

## What it does
For each λ it renders a job from `--base-job` (config **and** custom code) with only
`ditto_lambda`, `num_rounds`, and the output/tb/log paths overridden, runs `nvflare
simulator` with 3 clients (HIPIMR/RSNA-AU/RSNA-US) each on its own GPU, then reads each client's
best **val** AUC and ranks λ by worst-site.

- **Single config source**: `<job>/app/config` + `<job>/meta.json` (no duplicate
  template — the config you review is the one that runs).
- **Per-client isolation** on the shared filesystem: executor `data_path_map` (CSVs),
  `preprocess_cache_dir_map` (crops), and `{site}` path substitution.
- **Selection uses validation only.** λ is ranked by val AUC (`*_best_val_overall`), and the
  server's `IntimeModelSelector` keys on `val_auc`. **Test is never used for selection** — it's
  only evaluated on the final round for reporting. Preprocessing the train/val/test splits is
  fine; the caches are built once and reused across all λ.
- **Crop cache**: built once (1-round solo run) before any parallel runs, then reused
  read-only across λ. Auto-built if missing; skipped if present. `--prepare-cache` does the
  build as an explicit standalone step.
- **Concurrency**: one running job per GPU pool (default 2 pools × 3 GPUs).
- **Attribution**: each rendered job is stamped with the repo commit (`app/custom/GIT_COMMIT`)
  so simulator runs log `git=<hash>` instead of `git=unknown`.

## Prerequisites on the DGX
1. **All three sites' data present locally**, at the paths in
   `<job>/app/config/config_fed_client.json` → `data_path_map` (CSVs) and the full image
   paths *inside* those CSVs (`file_path` column). The DGX must hold HIPIMR + RSNA-AU + RSNA images.
2. `sample_model_5.p` at `/workspace/models/sample_model_5.p` (fail-loud if missing).
3. `nvflare` on PATH; GPUs 1–6 free (GPU 0 is the live FL run).
4. Edit `<job>/app/config/config_fed_client.json` if your CSV paths or cache locations
   differ from the defaults (`/workspace/data/...`,
   `/workspace/data/processed/ditto_sim_cache/<site>`).

## Run
```bash
cd ditto_sweep

# optional explicit cache build (otherwise the sweep auto-builds once on first run):
python launcher.py --base-job ../<job> --prepare-cache \
  --gpu-pools "1,2,3" --output-base /workspace/sim/ditto --work-root /workspace/sim/ditto_runs

# the sweep:
python launcher.py \
  --base-job ../<job> \
  --lambdas 0.01,0.05,0.1,0.5,1.0,2.0 \
  --rounds 20 \
  --gpu-pools "1,2,3;4,5,6" \
  --output-base /workspace/sim/ditto \
  --work-root  /workspace/sim/ditto_runs
```
- `--dry-run` renders jobs and prints the simulator commands without launching.
- `--metric mean_site` to rank by mean instead of worst-site (worst-site is the default).

### FedProx μ sweep (fair counterpart to the λ sweep)
Same harness, same selection (worst-site **val** AUC), but `method=fedprox` with one `fedprox_mu`
per run instead of a Ditto λ. The deployed/ranked model is the **shared global w** (FedProx has no
personal model), so this finds the μ that best balances the sites under one shared model — the fair
analogue to the Ditto λ search. The base FedProx run used μ=0.01, which was too weak; this searches
around it.
```bash
python launcher.py \
  --base-job ../<job> --fedprox \
  --mus 0.001,0.01,0.05,0.1,0.5,1.0 \
  --rounds 20 \
  --gpu-pools "1,2,3;4,5,6" \
  --output-base /workspace/sim/fedprox \
  --work-root  /workspace/sim/fedprox_runs
```
Crop caches (per-site) are shared with the Ditto sweep — built once, reused read-only. Output and
ranking are identical in form to the λ sweep (`sweep_summary.json`, per-site
`*_best_val_overall_gmic_metrics.json`); the table header reads `FEDPROX MU SWEEP RESULTS`.

## Output
- Live per-run logs: `<work-root>/run_l<λ>.log`, `<work-root>/cacheprep.log`.
- Per-site results: `<output-base>/l<λ>/<site>/<site>_best_val_overall_gmic_metrics.json`.
- Final ranked table is printed and saved to `<work-root>/sweep_summary.json`:
  ```
   lambda  worst_site   mean_site         HIPIMR         RSNA-AU    RSNA-US
      0.1      0.8420      0.8710       0.8830      0.8420      0.8880  <- BEST
      ...
  BEST (worst_site): lambda=0.1  worst_site=0.8420
  ```

## Notes / knobs
- **GPUs**: `--gpu-pools "1,2,3;4,5,6"` = two concurrent λ's, three clients each. Use a
  single pool (`"1,2,3"`) to run strictly sequentially, or `--threads 1` to round-robin
  clients on fewer GPUs.
- **Variant**: this sweeps plain `method=ditto` (single λ). For per-module λ's switch to
  `ditto_modulewise` (`lambda_global/local/fusion`) — different search, not wired here.
- **Final test metrics**: the sweep selects λ by val AUC. Re-run the winning λ (or evaluate
  its saved `*_best_val_overall` personal models on the test split) for reported test numbers.
- Cannot be tested on the Windows clone (no GPU/data); run on the DGX.
