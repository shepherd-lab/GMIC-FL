# pool_report_job — pooled salvage stats via the FL channel (no node file access)

Computes the **POOLED** breast-level AUC (DeLong CI) + Youden(val)→test sens/spec (Wilson CI) across
all sites for the salvage `incoming_global` predictions, and logs it to the **server** log — so you can
read it through the admin console even for a node whose files you can't reach (e.g. RSNA-AU).

It's a one-round, no-GPU job: each client reads the prediction CSVs the salvage already wrote to its
`results_dir`, breast-aggregates, and ships the de-identified `(prob, label)` arrays; the server
concatenates across sites, pools, and logs `[salvage-pooled]`. (This works where the earlier
server-side attempt didn't, because a custom `ModelController` receives the client results directly —
the stock `FedAvg` workflow ignored the `aggregator` component.)

## Run it

Submit to the SAME federation that ran the salvage (server + the 3 site clients), via the admin
console — e.g. `submit_job pool_report_job`. One round, a few seconds per client, no GPU.

Set the path if your salvage `results_dir` differs (it's in `config_fed_client.json` →
`results_dir`, default `/workspace/data/processed/fedprox_3client_fixedinit_20260609`). The executor
globs `{site}_predictions_incoming_global_round*_{val,test}.csv` there.

## Resume after a disconnect (don't start over)

A re-submit **appends** to the existing `out_file` instead of recomputing from scratch:

- The server loads the existing `out_file`, keyed by `(method, round)`, and **merges** new rows in —
  completed rounds are never overwritten, only added/updated. Results are flushed (atomic
  temp+rename) **after every** `(method, round)` is pooled, not just at the end, so a crash mid-loop
  keeps everything computed so far.
- A round is only written when **all** responding sites covered both its val and test arrays **and**
  the expected number of clients (`min_clients`, default 3) responded — so an under-covered re-run
  (e.g. one site dropped) can never clobber a correct pooled row; that round is just retried next
  time. If fewer than `min_clients` respond, nothing is written and the prior `out_file` is kept.
- To fetch **only the rounds still missing**, set `rounds` on the server workflow (a list of round
  ids, e.g. `"rounds": ["27","28","29"]`). The server forwards it to the clients via the task, so
  they ship only those — no client redeploy needed. Omit `rounds` to do all rounds (merge still
  makes that idempotent — already-done rounds are just recomputed to the same value).
- The 7200s `timeout` (matching the 2h `heart_beat_timeout`/`retry_timeout`) means a single exchange
  waits up to 2h for RSNA-AU to reconnect and finish, so most blips never even need a re-submit.

> Note: NVFlare's broadcast is one atomic exchange — the server pools after all sites' payloads
> arrive. So within a single submit, the per-round write protects against a server-side crash during
> pooling; a RSNA-AU disconnect *during the exchange* is covered by the 2h reconnect window and, failing
> that, by re-submit + merge. The job is seconds of compute, so a fresh exchange is cheap.

## Read the result

On the server (admin console), grep `log.txt`:

```
salvage-pooled
```

One line per `(method, round)`:
`[salvage-pooled] method=incoming_global round=26 POOLED n=... (pos=...) valAUC=...  testAUC=... [CI]  thr=...  Sens=... [CI]  Spec=... [CI]`

Also written to `/workspace/server_logs/pooled_stats_fedavg.json` (the server config's `out_file`).
Use a per-method `out_file` name (e.g. `pooled_stats_fedprox.json`) so each run keeps its own pooled
stats instead of overwriting the previous method's.

## Pick the round to report

Choose the `incoming_global` round with the highest **pooled valAUC** → report its pooled **test** AUC
+ DeLong CI as the FedProx baseline. Sanity: each pooled `n` should be ~the sum of the three sites'
test breasts (≈ 878 + 396 + 396 = 1670 for test, ~826 for val).

## Notes

- `deployed_local` is deliberately excluded (a different model per site → not poolable).
- The pooled math is identical to `tools/salvage_stats.py`; verified equal on synthetic 3-site data.
- Needs all three sites online for the one round (seconds, no GPU — minimal RSNA-AU exposure).
- Resilient to RSNA-AU network blips: the controller's `timeout` is 7200s (2h), matching the
  `heart_beat_timeout`/`retry_timeout` raised in `master_template.yml`. A site that drops and
  reconnects within 2h rejoins the still-running round instead of aborting the job. (The 900s default
  used to cut that off at 15 min.) Requires the re-provisioned/hot-edited 2h resources to be deployed.
