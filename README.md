# Personalized Federated Learning for Equitable Breast Cancer Detection

A federated-learning (FL) training and evaluation system for breast cancer
detection across multiple clinical sites, built on the
[GMIC](https://github.com/nyukat/GMIC) mammography model and
[NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) (NVFLARE).

This code accompanies:

> Sollis LJ, Young PM, Bunnell A, Quon B, Hernandez BY, Wolfgruber TK, Shepherd J.
> **"Personalized Federated Learning for Equitable Breast Cancer Detection in
> Underrepresented Pacific Islander Populations."** MICCAI 2026 Workshop on
> Distributed, Collaborative, and Federated Learning (DeCaF); to appear in
> Springer LNCS.

> **Built on GMIC (NYU).** The underlying image model — the Globally-Aware
> Multiple-instance Classifier — and its preprocessing are the work of Shen et
> al. (NYU); see [`GMIC_MODEL_README.md`](GMIC_MODEL_README.md), the
> [original repository](https://github.com/nyukat/GMIC), and
> [arXiv:2002.07613](https://arxiv.org/abs/2002.07613). This repository extends
> that model with a federated training/evaluation system. It is a derivative
> work and, like GMIC, is licensed under **GNU AGPLv3** (see
> [LICENSE](LICENSE) and [NOTICE](NOTICE)).

---

## What this adds to GMIC

GMIC is a single-model image classifier. This repository wraps it in an NVFLARE
executor so several clinical sites can train a shared model **without pooling
patient data**, and adds the pieces needed to study cross-site and
cross-demographic equity:

- **Federated methods** — FedAvg, FedProx, FedBN, and personalized **Ditto**
  (both a scalar proximal weight and a module-wise variant with separate
  weights for the model's global / local / fusion blocks).
- **Per-round evaluation** — each round dumps per-site validation/test
  predictions so AUC, DeLong CIs, and operating points can be computed offline
  and pooled across sites.
- **Crash-resume** — an interrupted federated run can resume from the last
  completed round without restarting.
- **Constrained-GPU training** — optional per-site precision and micro-batch
  controls that let a memory-limited GPU participate (see below).
- **Subgroup fairness analysis** — post-hoc per-race/ethnicity metrics
  (AUC + CIs, sensitivity at fixed specificity, etc.) for any deployed model.

## Repository layout

| Path | Purpose |
|------|---------|
| `custom/` | **The FL code — single source of truth** (`bc_executor.py`, `fl_utils.py`, `data_loader/`, `model/`, `train/`, …). Copied into a job before running it. |
| `fedavg/`, `fedprox/`, `fedbn/`, `ditto/`, `ditto_mw/` | Federated jobs, **one per method** — config only (`app/config/`, `meta.json`). |
| `centralized/`, `local/` | Single-node (1-client) baselines — config only. `centralized` = all sites' data pooled in one CSV; `local` = one site's CSV. |
| `pool_report_job/` | Pools each site's per-round predictions into combined AUC/DeLong/operating-point statistics (keeps its own small executor). |
| `ditto_sweep/` | Hyperparameter sweeps (Ditto λ; FedProx μ). |
| `scripts/` | Analysis/utility scripts (run from the repo root): `subgroup_fairness.py` (per-race/ethnicity fairness for one site), `run_all_subgroups.py` (that across every method → combined table), `dump_ditto_perround_preds.py` (per-round predictions for Ditto runs). |
| `custom/tests/` | Unit/integration tests for the FL code (`pytest custom/tests/`). |
| `tools/` | Operational helpers (salvage/resume runbooks). |
| `site_folders/` | Deployment templates: NVFLARE server/client Docker kits and a CSV→GMIC converter. |
| `gmic-localhost.yml`, `master_template.yml` | NVFLARE provisioning: project spec + workspace template. |

## Running a job

There is **one folder per training regime** — `fedavg`, `fedprox`, `fedbn`,
`ditto`, `ditto_mw` (federated, 3 sites), plus `centralized` and `local`
(single-node baselines). Each holds only its config (the `method` flag +
hyperparameters); the FL code lives once in `custom/`. Before running a job,
copy the code into it:

```bash
cp -r custom ditto/app/custom          # code into the job (regime) you want to run
```

Then run it in **either mode** — the job config is the same, only the launch
and the environment paths differ:

```bash
# (a) local simulator — N clients on one machine
nvflare simulator ditto -w /tmp/ws -n 3 -t 3

# (b) real multi-site — provision (see "Deploying"), then submit from the admin console
```

The per-job `app/custom/` copies are git-ignored, so `custom/` stays the single
source of truth — edit code there and re-copy. **Sim vs. real is not a different
job**: point `data_path_map`, `output_dir`, and the preprocess-cache paths at your
environment (`{site}` is substituted per client in both modes). Optional
constrained-GPU knobs (`personal_amp_by_site`, `*_batch_size_by_site`,
`heartbeat_interval_s`) let a memory-limited site participate — see "Training on a
memory-constrained GPU"; they default off.

**Centralized / local-only baselines.** `centralized/` and `local/` are 1-client
jobs (`method: "local"`, `num_clients: 1`): they train a single model on one node.
Point `data_path` at the pooled CSV (all sites) for `centralized`, or at one site's
CSV for `local`, and run the same way (`nvflare simulator centralized -n 1 -t 1`).
Total epochs = `num_rounds × epochs`.

## Requirements & setup

The model, data pipeline, and FL stack run in a Docker container. See
[`GMIC_MODEL_README.md`](GMIC_MODEL_README.md) for the model/preprocessing
prerequisites and `Dockerfile` / `docker-compose.yml` for the container. In
brief: PyTorch + NVFLARE, one GPU per site, mammography images preprocessed to
2944×1920 16-bit PNGs.

Each site provides a metadata CSV (see
[`site_folders/sample_gmic_data_format.csv`](site_folders/sample_gmic_data_format.csv)
for the schema: `patient_id, exam_id, laterality, view, file_path,
exam_level_label, view_level_label, split_group, ...`). No patient data is
included in this repository.

The GMIC pretrained weights (the FL warm-start, `sample_model_1..5.p`) are **not**
shipped here — download them from the [GMIC repository](https://github.com/nyukat/GMIC)
and place them where your config expects (default `/workspace/models/`).

## Deploying the federated system

Deployment has two layers. NVFLARE provisioning generates each participant's FL
startup kit; a small Docker template wraps it into a runnable container.

1. **Provision the startup kits.** `gmic-localhost.yml` is the NVFLARE project
   spec (server, clients, admin) and `master_template.yml` is the standard
   NVFLARE workspace template. Edit the participant list to your sites, then:

   ```bash
   nvflare provision -p gmic-localhost.yml
   ```

   This writes a startup kit per participant (certificates, `fed_server.json` /
   `fed_client.json`, `start.sh`, `sub_start.sh`, `docker.sh`).

2. **Wrap each kit in a container.** `site_folders/server/` and
   `site_folders/client/` are **templates** — a `Dockerfile`, a
   `docker-compose.yml`, and a run script — for the server and for a client.
   Copy the matching template alongside a participant's startup kit, set the
   placeholders (`ORG_NAME` for the server; `SITE_NAME` for a client, matching
   the provisioned participant name), and run `run_server.sh` / `run_client.sh`.
   `site_folders/csv_to_gmic_converter.py` helps convert a site's registry into
   the expected metadata schema.

> These are templates, not our production kits: a deployer provisions their own
> project (their hosts, their certificates) rather than reusing ours.

## Configuring a federated method

A job's method and hyperparameters are set in `app/config/config_fed_client.json`
(executor args). Key knobs:

| Config key | Meaning |
|------------|---------|
| `method` | `fedavg`, `fedprox`, `fedbn`, `ditto`, or `ditto_modulewise`. |
| `fedprox_mu` | FedProx proximal strength (μ). |
| `ditto_lambda` | Ditto proximal weight (scalar Ditto). |
| `lambda_global` / `lambda_local` / `lambda_fusion` | Per-block Ditto weights (module-wise). |
| `use_fedbn` | Keep BatchNorm layers local (FedBN). |
| `use_amp` | Mixed-precision training. |
| `resume_from_local_round` | Resume an interrupted run from this round (`-1` = fresh). |

Per-folder `FEDERATED_METHODS.md` files document each method in detail.

## Training on a memory-constrained GPU

The federated methods run best on ample-memory GPUs, but a site on a smaller
card (e.g. a 16 GiB RTX A4000) can still participate using the optional,
**default-off** toggles below. They are resolved from the FL identity at
runtime, so one config deployed to every site affects only the listed site — no
per-site app copies, and no need to move a site to a larger GPU. Leave them
unset on capable GPUs; every default reproduces standard behavior.

| Config key (executor arg) | Effect when set |
|---------------------------|-----------------|
| `personal_amp_by_site` | Per-site precision for the Ditto personal pass, e.g. `{"SITE_X": false}` forces fp32 there while others keep AMP. Unset → follows `use_amp`. |
| `personal_batch_size_by_site` | Per-site micro-batch for the personal pass, e.g. `{"SITE_X": 8}`; chunks accumulate to the full effective batch (only BatchNorm sees the chunk). Unset → no chunking. |
| `train_batch_size_by_site` | Same, for the main (shared-weight) training pass. |
| `heartbeat_interval_s` | Seconds between watchdog-thread progress logs during long phases; logs only, never aborts. `0` (default) disables. |
| `memory_efficient` | Release cached CUDA memory between passes. |
| `stage_sync`, `debug_devices` | Diagnostics for locating a stalled/faulting CUDA op. |

Micro-batch chunking is gradient-exact (the optimizer sees the full effective
batch); only BatchNorm statistics are computed on the smaller chunk.

## Analysis tools

- **Pooled statistics** — submit `pool_report_job` to combine per-site
  predictions into AUC, DeLong CIs, Youden thresholds, and operating-point
  metrics.
- **Subgroup fairness** — after a run, compute per-race/ethnicity metrics for a
  site's deployed model. The site label and the race/ethnicity columns and
  code→group mapping are all runtime parameters, so the tool carries no
  site-specific schema:

  ```bash
  python scripts/subgroup_fairness.py \
      --pred  <SITE>_predictions_<method>_round<N>_test.csv \
      --val   <SITE>_predictions_<method>_round<N>_val.csv \
      --meta  <site_registry>.csv \
      --site  <SITE> --eth-col <race_column> --eth-map <map.json>
  ```

  `scripts/run_all_subgroups.py` runs this across every method (auto-selecting
  each method's best-validation round) and writes a combined table plus a
  group × method summary.

## Citation

If you use this code, please cite both the federated-learning paper (above) and
the original GMIC work:

```bibtex
@inproceedings{sollis2026personalized,
  title     = {Personalized Federated Learning for Equitable Breast Cancer
               Detection in Underrepresented Pacific Islander Populations},
  author    = {Sollis, L. J. and Young, P. M. and Bunnell, A. and Quon, B. and
               Hernandez, B. Y. and Wolfgruber, T. K. and Shepherd, J.},
  booktitle = {MICCAI 2026 Workshop on Distributed, Collaborative, and
               Federated Learning (DeCaF)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}

@article{shen2021gmic,
  title   = {An interpretable classifier for high-resolution breast cancer
             screening images utilizing weakly supervised localization},
  author  = {Shen, Yiqiu and Wu, Nan and Phang, Jason and Park, Jungkyu and
             Liu, Kangning and Tyagi, Sudarshini and Heacock, Laura and
             Kim, S. Gene and Moy, Linda and Cho, Kyunghyun and Geras, Krzysztof J.},
  journal = {Medical Image Analysis},
  year    = {2021}
}
```

## License

GNU Affero General Public License v3.0 (AGPLv3). This is a derivative of GMIC
(© 2020 the GMIC authors, NYU) and remains under the same license; the
federated-learning additions are © 2026 Shepherd Research Lab, University of
Hawaiʻi Cancer Center. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
