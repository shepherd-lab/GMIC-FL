"""PTFileModelPersistor that resumes a crashed run by re-aggregating client checkpoints.

Drop-in replacement for PTFileModelPersistor. When `warmstart_round >= 0`, it computes
the sample-weighted mean of every client's `{site}_gmic_model_round_{warmstart_round}.pth`
in `warmstart_dir`, writes the result as a flat state_dict, and points the persistor's
`source_ckpt_file_full_name` at it -- so NVFLARE's normal load path broadcasts that
reconstructed global to all clients at round 0. The FedAvg/FedProx run then continues for
the remaining rounds. Set `warmstart_round: -1` (the default) for ordinary behavior.

Requires the server to have filesystem access to `warmstart_dir` (e.g. gather the three
client round-x files under the server's mounted /workspace before launch).
"""

import os

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

import torch

from resume_aggregate import aggregate_round_checkpoints


class WarmStartAggregatePersistor(PTFileModelPersistor):
    def __init__(
        self,
        model,
        source_ckpt_file_full_name=None,
        warmstart_dir=None,
        warmstart_round: int = -1,
        warmstart_weights: dict | None = None,
        warmstart_pattern: str = "*_gmic_model_round_{round}.pth",
        **kwargs,
    ):
        super().__init__(
            model=model,
            source_ckpt_file_full_name=source_ckpt_file_full_name,
            **kwargs,
        )
        self.warmstart_dir = warmstart_dir
        self.warmstart_round = int(warmstart_round)
        self.warmstart_weights = warmstart_weights or {}
        self.warmstart_pattern = warmstart_pattern
        self._warmstart_done = False

    def load_model(self, fl_ctx):
        # One-shot: aggregate and redirect source_ckpt the first time the controller asks
        # for the initial global. On any failure, fall back to the configured source_ckpt so
        # a bad warmstart arg degrades to a normal (cold) start rather than crashing the run.
        if (
            self.warmstart_round >= 0
            and self.warmstart_dir
            and not self._warmstart_done
        ):
            self._warmstart_done = True
            try:
                agg, info = aggregate_round_checkpoints(
                    self.warmstart_dir,
                    self.warmstart_round,
                    self.warmstart_weights,
                    self.warmstart_pattern,
                    logger=lambda m: self.log_info(fl_ctx, m),
                )
                out = os.path.join(
                    self.warmstart_dir,
                    f"_warmstart_round{self.warmstart_round}_aggregated.pth",
                )
                torch.save(agg, out)
                self.source_ckpt_file_full_name = out
                self.log_info(
                    fl_ctx,
                    f"[warmstart] resume global written -> {out} "
                    f"(sites={info['sites']} norm_weights={[round(w, 4) for w in info['norm_weights']]}); "
                    f"broadcasting as round-0 global",
                )
            except Exception as e:
                self.log_error(
                    fl_ctx,
                    f"[warmstart] aggregation FAILED ({e}); falling back to configured "
                    f"source_ckpt_file_full_name={self.source_ckpt_file_full_name}",
                )
        return super().load_model(fl_ctx)
