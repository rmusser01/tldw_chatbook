# Textual follow-up CI routing

2026-11-15; TASK-37360, TASK-40969, TASK-1378.3, existing TASK-39144 / ADR-103.

The census retains every inherited selector and adds 27 passing cases: five
Media, four Context & Memory, and 18 Speech cases. Speech includes nine new
regressions plus nine narrowly affected cancellation, navigation-payload and
model-library cleanup contracts. Original coverage was inventoried before
selecting wrappers; broad inherited compatibility wrappers were not added.
There are 2061 unique selectors, with a matching checker floor.

The UI matrix uses 38 shards with the existing maximum of 18 simultaneous
jobs. This creates a third scheduling wave. The workflow has 50 eligible test
jobs and unchanged peak concurrency of 30 test runners, 31 with perf-guard.
Job/test deadlines, lane composition and required aggregation are unchanged.

## Measurements and assumptions

`capacity-entry-weights.json` preserves raw measured costs for all 2034 inherited
entries. The source comprises 31 currently available historical run sets from
2026-11-10 19:39Z through 2026-11-12 15:31Z (1034 successful UI shard logs), plus
three recent successful runs (108 UI jobs). Current historical attempts differ
from the original 28-run set; failed UI jobs were excluded. Each run/shard was
used once, preferring its available latest successful job. Seven rerun archives
needed earlier-attempt recovery; one unavailable first attempt was covered by
its second and third attempts. Original downloaded logs remain task scratch,
not tracked QA.

Measured inherited work is 467.0781627 pytest minutes. An explicit proportional
reserve of 9.5618373 minutes preserves the old rounded `13.24 * 36 = 476.64`
envelope. The reserve distribution is an assumption, not observed timing. The
27 new cases cost 1385.997 raw Windows JUnit seconds (23.09995 minutes). Speech
uses the slowest successful observed case across its listed receipts. Only
selected passing cases were priced; some receipts also contain other failed
cases, which are excluded. Case timings and exact receipt hashes are in
`capacity-summary.json`.

Matching unchanged Media controls took 0.349–0.393 times their Windows time in
the three recent CI runs; Context controls took 0.143–0.207. Raw Windows charges
are conservative relative to these observations, without guaranteeing future
CI performance. `control-comparison.json` preserves the exact matches.

| Charge scenario | Shards | Ideal floor | Rebalanced maximum | Swaps |
| --- | ---: | ---: | ---: | ---: |
| Raw measured Windows | 37 | 13.50649 | Impossible at 13.3 target | — |
| Raw measured Windows | 38 | 13.15105 | 13.22248 | 34 |
| 2.61 sensitivity only | 38 | 14.12976 | Impossible at 13.3 target | — |
| 2.61 sensitivity only | 41 | 13.09587 | 13.29952 | 33 |

All values are modeled pytest minutes. Adding the retained 1.7-minute setup
allowance to the applied maximum gives 14.92248 minutes, leaving 5.07752 of the
20-minute cap. The 2.61 multiplier calibrated a different quiet Unix host in
earlier work; here it is a sensitivity scenario, not a Windows calibration.

Cleanup qualification uses a validated accepted-package boundary fixture with
the existing real `preack_overlap` UI orchestration scenario. It does not claim
native scanner, security-boundary or runtime qualification on Windows. The
next PR run must still execute the new shard order; TASK-39144's required
post-change observation remains open.

## Reproduce without historical logs

From the repository root, using standard Python and the existing replay tool:

```powershell
python Docs/superpowers/qa/2026-11-14-textual-pattern-followups/ci/replay_capacity.py
```

This verifies the final census distribution against portable measured weights
and the separate reserve/charge assumptions. It should report 2061 entries,
maximum 13.2224827 and `matches_recorded_order: true`.

To reproduce the original 34-swap search, export the base census, then let the
script append the recorded 27 selectors in their insertion order:

```powershell
python -c "import subprocess; from pathlib import Path; Path('base-census.txt').write_bytes(subprocess.check_output(['git', 'show', 'ecad404bd4567b57139716a6deaa6b37fe833749:scripts/ui_pr_gate_census.txt']))"
python Docs/superpowers/qa/2026-11-14-textual-pattern-followups/ci/replay_capacity.py --census base-census.txt --rebalance
```

Use `--charge-multiplier 2.61 --shards 41 --rebalance` on that base census to
reproduce the sensitivity scenario. The script does not rewrite the census.
The archived model requires its recorded selector set; later census additions
need fresh costs rather than an untimed default.

## Validation

Before the count change, the census checker/replay tests passed: 104 cases,
zero failures/errors/skips, 53.81 seconds; JUnit SHA256
`39830559b1a90612220742e27ceea8bcc6437a339ce8188416068fb4553c5b4e`.
Final census checker passes at 2061 entries/floor. Nine affected count,
38-shard unique-coverage, serial-command and required-aggregation guards pass
with zero failures/errors/skips in 9.78 seconds; JUnit SHA256
`c694e94b63b0526b177cb92b340340e0abe2a6bced7a5fee16206184a0f58534`.
The 104 checker/replay tests were not repeated because their algorithms did
not change. Both the final-order replay and the base-census 34-swap
reproduction match the recorded order and maximum.

Scoped diff whitespace checks pass. Ruff retains the same single pre-existing
`B905` in `scripts/ui_lane_shard_replay.py`, with no new diagnostics. Formatting
remains at its baseline in the replay tool and two workflow test files; the
checker and new reproduction script pass format checks. The reproduction
script also passes Ruff. No unrelated formatting rewrite or full suite was run.
