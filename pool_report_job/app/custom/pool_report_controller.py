"""Server workflow for the pooling job: ONE broadcast round -> collect each site's breast-level
(prob, label) for the shared-global predictions -> concatenate across sites -> compute the POOLED
breast-level AUC (DeLong 95% CI) + Youden(val)->test sens/spec (Wilson 95% CI) via salvage_metrics ->
log one `[salvage-pooled]` line per (method, round) to the SERVER log (readable through the admin
console) and write a durable JSON.

This is the piece the stock FedAvg ModelController could not do via an `aggregator` component (it
ignores that component). A custom ModelController, by contrast, receives every client's FLModel result
directly, so it can pool and report. No persistor/model is needed (persistor_id="" disables it).

Resume / disconnect safety
--------------------------
A re-submit appends instead of starting over:
  * the existing `out_file` is loaded and merged by (method, round) -- re-running never overwrites
    completed rows, it adds the new ones;
  * results are flushed (atomic temp+rename) after EVERY (method, round) is pooled, not just at the
    end, so a crash mid-loop keeps everything computed so far;
  * an optional `rounds` whitelist is forwarded to the clients (via task meta) so a resume can ask
    for just the rounds still missing, shrinking the exchange;
  * a round is only written when ALL responding sites contributed both its val and test arrays AND
    the expected number of clients responded -- so a partial (e.g. 2-of-3-site) re-run can never
    overwrite a correct pooled row with an under-covered one; that round is simply retried next time.
"""
import json
import os

import numpy as np

from nvflare.app_common.utils.fl_model_utils import FLModel, ParamsType
from nvflare.app_common.workflows.model_controller import ModelController

from salvage_metrics import endpoint_metrics, format_endpoint


class PoolReportController(ModelController):
    def __init__(self, task_name="pool_report", poolable_methods=None, timeout=7200, out_file=None,
                 rounds=None, min_clients=3):
        super().__init__(persistor_id="")  # no persistor / model -- this job only gathers + pools
        self.task_name = task_name
        self.poolable = set(poolable_methods) if poolable_methods else {"incoming_global", "pretrained_baseline"}
        self.timeout = int(timeout)
        self.out_file = out_file
        # optional whitelist of rounds (as strings) to (re)compute -- forwarded to the clients so a
        # resume can fetch just the rounds still missing. None => all rounds the clients have.
        self.rounds = [str(r) for r in rounds] if rounds else None
        # expected number of responding sites; a round is only written when every responder covered
        # it, so under-covered re-runs never clobber a good row.
        self.min_clients = int(min_clients)

    def _load_existing(self):
        """Existing out_file -> {(method, round): row}. Resume merge target; missing/corrupt => {}."""
        rows = {}
        if self.out_file and os.path.exists(self.out_file):
            try:
                with open(self.out_file) as f:
                    for row in json.load(f):
                        rows[(row.get("method"), str(row.get("round")))] = row
            except Exception as e:
                self.warning(f"[pool-report] could not read existing {self.out_file} (starting fresh): {e}")
        return rows

    def _flush(self, rows):
        """Atomically write the full merged row set (temp + rename) -- called after every computation."""
        if not self.out_file:
            return
        try:
            os.makedirs(os.path.dirname(self.out_file) or ".", exist_ok=True)
            ordered = [rows[k] for k in sorted(rows.keys(), key=lambda k: (str(k[0]), str(k[1])))]
            tmp = self.out_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(ordered, f, indent=2)
            os.replace(tmp, self.out_file)
        except Exception as e:
            self.warning(f"[pool-report] could not write {self.out_file}: {e}")

    def run(self):
        existing = self._load_existing()  # merge target -- resume = append, never overwrite
        if existing:
            self.info(f"[pool-report] resuming: {len(existing)} (method,round) row(s) already in {self.out_file}")

        data_meta = {"rounds": self.rounds} if self.rounds else {}
        self.info("[pool-report] broadcasting pool_report task to all clients"
                  + (f" for rounds={self.rounds}" if self.rounds else ""))
        results = self.send_model_and_wait(
            task_name=self.task_name,
            data=FLModel(params={"_": [0.0]}, params_type=ParamsType.FULL, meta=data_meta),
            timeout=self.timeout,
        )
        n_sites = len(results)
        self.info(f"[pool-report] received {n_sites} client result(s)")
        if n_sites < self.min_clients:
            self.warning(f"[pool-report] only {n_sites}/{self.min_clients} sites responded -- not writing "
                         f"(pooled stats would be under-covered); re-submit to resume. Existing results kept.")
            return

        # "method|round" -> split -> {"p":[], "y":[]}, plus per-key coverage so we only write rounds
        # every responder contributed to.
        pooled = {}
        coverage = {}  # "method|round" -> split -> set(sites)
        for r in results:
            meta = r.meta or {}
            site = meta.get("site", "?")
            raw = meta.get("salvage_preds")
            if not raw:
                self.warning(f"[pool-report] no salvage_preds from {site}")
                continue
            try:
                preds = json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                self.warning(f"[pool-report] bad payload from {site}: {e}")
                continue
            for key, splits in preds.items():
                d = pooled.setdefault(key, {})
                cov = coverage.setdefault(key, {})
                for split, arr in splits.items():
                    dd = d.setdefault(split, {"p": [], "y": []})
                    dd["p"].extend(arr.get("p", []))
                    dd["y"].extend(arr.get("y", []))
                    cov.setdefault(split, set()).add(site)

        n_written = 0
        for key in sorted(pooled.keys()):
            method, _, rnd = key.partition("|")
            if method not in self.poolable:
                continue
            splits = pooled[key]
            if "val" not in splits or "test" not in splits:
                self.warning(f"[pool-report] {key}: missing val/test across sites; skipping")
                continue
            cov = coverage.get(key, {})
            if len(cov.get("val", set())) < n_sites or len(cov.get("test", set())) < n_sites:
                self.warning(f"[pool-report] {key}: only val={sorted(cov.get('val', []))} "
                             f"test={sorted(cov.get('test', []))} of {n_sites} sites -- under-covered, "
                             f"skipping (will retry on resubmit)")
                continue
            try:
                m = endpoint_metrics(np.asarray(splits["val"]["p"], float), np.asarray(splits["val"]["y"], int),
                                     np.asarray(splits["test"]["p"], float), np.asarray(splits["test"]["y"], int))
            except Exception as e:
                self.warning(f"[pool-report] pooled compute failed for {key}: {e}")
                continue
            self.info(f"[salvage-pooled] method={method} round={rnd} " + format_endpoint("POOLED", m))
            existing[(method, str(rnd))] = {"method": method, "round": rnd, **m}
            self._flush(existing)  # write after EVERY computation, not just at the end
            n_written += 1

        if n_written == 0:
            self.warning("[pool-report] no NEW poolable (method,round) groups produced -- check that all "
                         "sites returned the requested predictions")
        self.info(f"[pool-report] done: {n_written} new/updated row(s); {len(existing)} total in {self.out_file}")
