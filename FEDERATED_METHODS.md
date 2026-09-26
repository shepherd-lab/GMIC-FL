# GMIC Federated Methods — implementation & restricted-machine handoff

One configurable executor + aggregation path supporting six training regimes, switched by
a single `method` flag in the client job config (the λ sweep launches by editing config,
not code). `method ∈ {local, fedavg, fedprox, fedbn, ditto, ditto_modulewise}`, default
`fedavg`. The executor lives once in the repo-root `custom/` and is copied into a job's
`app/custom/` before running it; the per-method job folders differ only in their config.

## Files
- `app/custom/fl_utils.py` — primitives (pure torch): `module_group`, `bn_state_keys`,
  `load_global_into`, `proximal_penalty`, `clone_detached_state`, `tolerant_load_pretrained`,
  `gmic_outputs`, `malignant_score`, `gmic_malignant_loss`.
- `app/custom/model/gmic.py` — `forward()` returns `(fusion_logits, y_global, y_local, saliency)`.
- `app/custom/bc_executor.py` — method switch, pretrained init, GMIC loss, Ditto, dumps.
- `app/custom/train/training_core.py` — tuple-aware `evaluate_model` (malignant head).
- `app/config/config_fed_client.json` — `method` block.
- `custom/tests/test_fl_methods.py`, `test_executor_integration.py` — CPU tests (`pytest custom/tests/`).

## Model / loss (verified against upstream nyukat/GMIC + the GMIC papers)
- Param→group (module-wise Ditto): `global = ds_net + left_postprocess_net`,
  `local = dn_resnet`, `fusion = mil_attn_V/U/w + classifier_linear + fusion_dnn`
  (135 params, 0 unassigned). BatchNorm = 41 modules / 205 keys, all in the two backbones.
- **Loss = native GMIC deep supervision, malignant head only** (data carries a single
  binary cancer-vs-else label; benign ≡ ¬malignant):
  `BCEWithLogits(fusion[:,1]) + BCE(y_global[:,1]) + BCE(y_local[:,1]) + λ·Σ|saliency[:,1]|`.
  Independent sigmoids, never softmax. Global/local are post-sigmoid probabilities → `BCE`;
  fusion is a raw logit → `BCEWithLogits`. **The loss is computed OUTSIDE `autocast`** (fp32);
  `malignant_score` casts to fp32 before `sigmoid`/`.numpy()` (bf16/fp16 safety under AMP).
  Class imbalance: a single per-sample `pos_weight` (default 3) applied identically to all
  three heads. Benign head (index 0) is present-but-unsupervised so `sample_model_5.p` loads 1:1.
- `malignant = index 1` everywhere (run_model.py: `benign,malignant = pred[0,0],pred[0,1]`).
- Send convention: FULL weights; server `nvflare...FedAvg` controller aggregates by
  NUM_STEPS — one aggregator serves all methods, no server change.

## Per-method behavior
| method | global load | extra train | sent | deployed/eval |
|---|---|---|---|---|
| `local` | skipped (pretrained init) | — | w (ignored) | w |
| `fedavg` | full | — | w | w |
| `fedprox` | full | `+(mu/2)‖w−w_g‖²` | w | w |
| `fedbn` | full, **skip BN keys** | — | w | w (shared + local BN) |
| `ditto` | full (w) | personal pass trains `v`: `loss(v)+prox(v,w_ref,λ)` | w | **v** |
| `ditto_modulewise` | full (w); BN-skip on w if `use_fedbn` | per-group λ personal pass | w | **v** |

## Config schema (client args)
`method, fedprox_mu(0.01), ditto_lambda(0.1), lambda_global(1.0), lambda_local(0.1),
lambda_fusion(null→local), use_fedbn(false), local_epochs(null→epochs), pos_weight(3.0),
loss("gmic"), pretrained_weights_path, lambda_l1(1e-5)`. percent_t is set **only** via
`percent_t_key` (→ PERCENT_T_DICT; "1"→0.02); a disagreeing `gmic_parameters.percent_t`
now raises (single source of truth). λ grid: `lambda_global∈{1,5,10} × lambda_local∈{0.01,0.1,0.5}`.

### Config gotchas — read before the sweep
- **λ flags are inert unless `method=ditto_modulewise`.** `lambda_global/local/fusion` only reach
  the proximal term for `ditto_modulewise`; `ditto` uses the single `ditto_lambda`; FedAvg/FedProx/
  FedBN/local use none. **The λ sweep MUST set `method=ditto_modulewise`** — sweeping λ with any
  other method changes nothing. Each run logs a `[run-config] method=… lambda_…=…` line at round
  start; check it against the intended config.
- **`val_split`/`test_split` are IGNORED when `use_predefined_splits=true`** (the default). Splits
  come from the CSV `split_group` column; the two ratios only apply in the random-split fallback.
- **`loss` has one valid value: `gmic`** (the native GMIC deep-supervised malignant BCE + L1;
  computed in `_compute_task_loss`). Any other value raises at init — no silent fallback.
- Removed dead config keys (no effect, were misleading): `optimizer` block (top-level
  `lr_heads`/`lr_backbone`/`weight_decay` are the live source), `pretrained_model_index`,
  `load_checkpoint`, `gmic_parameters.gpu_number`, `results_dir` (dumps go to `output_dir`).
- `force_preprocessing` is now an explicit client-config flag (was an implicit default).

## Outputs for offline analysis (per site, in `results_dir`)
- `{site}_predictions_{tag}_round{R}_{val,test}.csv` — columns
  `site_id, round, method, split, exam_id, view, path, prob_malignant, label`
  (`tag` = `pretrained_baseline` at round 0, else the method). `prob_malignant` = fp32 sigmoid.
- `{site}_saliency_{tag}_round{R}_{val,test}.npz` — `saliency` (N,h,w fp32, malignant channel,
  post-sigmoid), `path`, `label`, `site_id`, `method`, `split`, `round`.
- **Join key = `path`**, identical in the CSV and the NPZ → map↔prediction↔label↔site↔method
  joins offline with no model. AUC/DeLong CIs from the CSV; paired saliency stats from the NPZ.
- Per-round `[per-site-metric] site=… round=… method=… deployed_val_auc=…` log line.
  Degenerate (zero-positive) eval splits report AUC=NaN (+`n_pos`), excluded from ranking.

### Saliency analysis — read before writing the comparison script
- **Map orientation:** maps are in the model-input (standardized / chest-wall-flipped) space, NOT
  raw-DICOM native. The loader flips/orients each image before the forward pass, so the saved map
  aligns pixel-for-pixel with the image *as the model saw it*. Overlay on the oriented image, not
  the raw file. (Both saliency channels live at `outputs[3]`; we save the malignant channel `[:,1]`.)
- **Paired comparison spans two jobs.** Within one job the only same-image pair is Ditto's
  `{site}_saliency_ditto_round{R}_test.npz` (personalized `v`) vs.
  `{site}_saliency_pretrained_baseline_round0_test.npz` (round-0 global). A **FedAvg-converged-global
  vs. personalized** comparison pairs maps **across two separate runs**, joined by `path` (the test
  split is deterministic — seeded/`split_group` — so `path` keys match across jobs). When building
  the script, pull the FedAvg-global map from the *FedAvg job's* final-round dump
  (`{site}_saliency_fedavg_round{last}_test.npz`), NOT the `pretrained_baseline` dump:
  **round-0 baseline is the cold pretrained model, not the FedAvg-converged global.** Grabbing the
  baseline by mistake is the easy error — it answers a different question (cold-start, not federated).

### `horizontal_flip` column semantics — COUNTERINTUITIVE, do not "fix"
The flip logic (`data_loader._flip_image` / `loading.flip_image`) is **byte-identical to upstream
`nyukat/GMIC`**. The column answers "is the source image *already* mirrored?", NOT "should I flip?":
- **`"NO"` (default, and what all our CSVs use) = the normal GMIC mode that DOES flip** — it mirrors
  **right**-laterality views (`R-CC`/`R-MLO`) so every breast faces the same way (chest wall on a
  consistent side), exactly as the pretrained weights expect. With all-`"NO"`, right breasts ARE
  already standardized. The flip happens **once**, at load (`data_loader.py:_flip_image`); the crop
  stage does not also mirror (it only picks the chest-wall edge for padding). No double-flip.
- **`"YES"` = the source PNG is already mirrored** → then it flips the **left** views instead (to undo).
- **Do NOT set `"YES"` for true-source `R` rows.** That would *stop* flipping right breasts and start
  flipping left ones — the exact mis-orientation you'd be trying to prevent. Leave the column `"NO"`
  (or absent → defaults `"NO"`) unless a site's images are pre-mirrored at export. The cold-parity
  gate on each site validates orientation empirically; if one site's AUC is near-random, that site's
  PNGs may be pre-mirrored → set `"YES"` for that site only.

## Preprocessing: cropping + center (DIVERGES from upstream — read before trusting scale analysis)
This fork's Stage 1 **resizes+pads** every crop to a fixed 2944×1920 (`resize_and_pad_keep_aspect`).
Upstream GMIC does NOT — it saves the **raw variable-size crop** and the augmentation does the final
2944×1920 window crop centered on `best_center`. Two consequences, both verified:
- **`best_center` is INERT here** (Stage 2 optimal-center search is neutralized → image-center).
  Proven: (1) the train/eval augmentation output is byte-identical for ANY center once the image
  already == the model window (`max|diff|=0.0`); (2) the K=6 ROI patches come from the model's
  saliency map, never `best_center` (no reference in `model/`). Upstream's center is meaningful only
  because it feeds the variable-crop→window step this fork's resize removed. The optimal-center
  machinery (`get_optimal_centers`/`calc_optimal_centers`) is KEPT, unimported, recoverable.
- **Cropping never clips or distorts** the breast: isotropic fit + zero-pad, 100% tissue preserved,
  aspect ratio exact (verified across thousands of aspect ratios — never clips).
- **The one real artifact: per-site scale variability.** Padding fraction varies with source aspect
  ratio (~10–39%), so the breast sits at different scales in the frame, and that variation is
  **site-correlated** (RSNA vs HIPIMR differ). This aliases onto the spatial-scale λ-probe / saliency
  comparison — the paper's second contribution. Stage 1 now logs `[PAD] sid=… pad_frac=…` per image;
  aggregate the per-site distribution from a run (`grep '\[PAD\]'`).
- **OPEN DECISION (defer to the padding numbers):** if per-site padding is small/overlapping → keep
  resize (**C′**, current), delete the center machinery in a follow-up. If sharply site-separated →
  weigh **B′** (revert to upstream variable-crop + revive the center search — machinery preserved for
  this) or a scale-normalizing crop (future work). Do NOT delete the center machinery until decided.

## Clone environment (for running the tests here)
Not installed by default. Install on demand: CPU `torch`, `nvflare`, `opencv-python-headless`,
`imageio`, and **pin `numpy<2`** (an opencv install pulls numpy 2.x which breaks anaconda
scipy/sklearn). NVFLARE imports the Unix-only `resource` module — `test_executor_integration.py`
injects a `resource` stub so it runs on Windows.
Tested: `test_fl_methods.py` 8/8, `test_executor_integration.py` 4/4 (incl. a `use_amp=True` round).

## RUN ORDER on the restricted machine (air-gapped; needs real data/GPU)
1. **Cold-parity gate (first).** Load `sample_model_5.p`, run inference on a few real
   preprocessed images per site, confirm `sigmoid(fusion)[:,1]` ranks cancers > controls.
   If near-random, preprocessing diverges from GMIC's pipeline (2944×1920, breast crop,
   per-image zero-mean/unit-std) — fix before anything else. (The executor also runs a
   round-0 `pretrained_baseline` dump automatically.)
2. **Centralized baseline.** `method=local`, **one client on the pooled CSV, `num_rounds=1`,
   `epochs≈40` + early stopping** — must be a single continuous loop (NOT `num_rounds=40,
   epochs=1`, which would trip the cross-round early-stopper reload; see C2). This is the ceiling.
3. **Federated baselines:** `fedavg`, then `fedprox`, `fedbn`.
4. **λ sweep:** `ditto` / `ditto_modulewise` over the grid (short rounds to rank, finish the
   winner at full length).
Collect `predictions_*.csv` + `saliency_*.npz` per site/round/method for offline analysis.
Also: populate the site configs (paths/site IDs — placeholders here) and confirm the label
contract (the loader now hard-asserts strictly {0,1}, breast-level view agreement, on load).

## OPEN — needs your decision before multi-round FL (C2, not yet changed)
The early stopper persists across rounds, so `execute()` reloads the **all-time-best** `w`
(possibly from a prior round) before packaging the FedAvg update — wrong for the global model
(FedAvg expects the current round's local weights); `v` (Ditto) is unaffected. Pending choice:
(A) reset the stopper per round [recommended] or (B) drop the cross-round reload for `w`.
Running centralized as `num_rounds=1` (step 2) sidesteps it.

## Commit history (branch `main`)
`d9d27f5` federated methods · `4e2824f` GMIC loss · `29caeed` saliency dumps + softmax cleanup ·
(this) QC fixes: AMP loss/score fp32, label-contract assert + converter reconcile, image-load
skip, degenerate-AUC NaN, raise_on_partial, percent_t single-source.
