#!/usr/bin/env python3
"""Compute the exact resume config after a crash, from the checkpoints on disk.

Repeated crash-resume (RSNA-AU's A4000 thrashes fp32 and dies every ~20 rounds) only needs two
values set correctly each time, and getting them wrong is the one real failure mode. This scans
the run's results dir and prints them, so each resume is copy-paste rather than guesswork.

WHY THE HIGHEST `{site}_gmic_model_round_N.pth` IS THE SAFE ROUND
----------------------------------------------------------------
A client returns its trained global to the server only at the END of execute() -- after the
personal pass and the `{site}_gmic_model_round_N.pth` save. So the server aggregates round N
(advancing to N+1) only once EVERY site has finished round N entirely and written both its
`global_trajectory/{site}_global_round_N.pth` (the Ditto reseed source) and its
`{site}_gmic_model_round_N.pth` (the personal model v restore source). Thus any round for which
the personal ckpt exists is fully resumable across all sites. We take the MIN across sites of
each site's highest such round: if RSNA-AU crashed mid-round N+1 while others finished it, RSNA-AU's max
is N (the safe common round), and the other sites' stale round-(N+1) files are simply redone.

USAGE
-----
    python resume_params.py <results_dir> [total_rounds=60]

e.g.  python resume_params.py /workspace/data/processed/ditto_mw_g0.01_l0.5_f0_amp_3client_20260721

Prints the client `resume_from_local_round` and the server `num_rounds` to set, and flags any
site that is missing the matching global_trajectory file (which would make the reseed fail).
"""
import os
import re
import sys
from collections import defaultdict

_PERSONAL = re.compile(r"^(?P<site>.+)_gmic_model_round_(?P<round>\d+)\.pth$")
_GLOBAL = re.compile(r"^(?P<site>.+)_global_round_(?P<round>\d+)\.pth$")


def _rounds_by_site(names, pattern):
    out = defaultdict(set)
    for n in names:
        m = pattern.match(n)
        if m:
            out[m.group("site")].add(int(m.group("round")))
    return out


def compute(results_dir, total_rounds=60):
    try:
        top = os.listdir(results_dir)
    except FileNotFoundError:
        print(f"ERROR: results_dir not found: {results_dir}")
        return 1
    gt_dir = os.path.join(results_dir, "global_trajectory")
    gt = os.listdir(gt_dir) if os.path.isdir(gt_dir) else []

    personal = _rounds_by_site(top, _PERSONAL)   # site -> {rounds with personal ckpt}
    glob = _rounds_by_site(gt, _GLOBAL)          # site -> {rounds with sent-global ckpt}

    if not personal:
        print(f"No '{{site}}_gmic_model_round_N.pth' files under {results_dir} -- nothing to "
              f"resume from (has any round completed?).")
        return 1

    # Each site is resumable at rounds where BOTH files exist; its highest such round is its max.
    site_max = {}
    for site, prounds in sorted(personal.items()):
        both = prounds & glob.get(site, set())
        missing_global = prounds - glob.get(site, set())
        hi = max(both) if both else None
        site_max[site] = hi
        note = ""
        if missing_global:
            note = f"  [!] personal ckpts without a global_trajectory match: {sorted(missing_global)}"
        print(f"  site={site:10s} highest fully-saved round={hi}"
              f" (personal max={max(prounds)}, global max={max(glob.get(site, {-1}))}){note}")

    usable = [r for r in site_max.values() if r is not None]
    if not usable:
        print("No round has BOTH the personal and global ckpt on any site -- cannot resume.")
        return 1
    n = min(usable)  # safe common round = min across sites
    print()
    print(f"==> last round completed by ALL sites: N = {n}")
    print(f"==> set client config  resume_from_local_round: {n}")
    print(f"==> set server config  num_rounds:            {total_rounds - n}   ( = {total_rounds} - {n} )")
    print(f"    (keep output_dir / resume_ckpt_dir unchanged so the round-{n} ckpts are found)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    rd = sys.argv[1]
    tr = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    sys.exit(compute(rd, tr))
