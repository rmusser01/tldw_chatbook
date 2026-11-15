# PR3569 rebase: CI census reconciliation

2026-11-15. Rebase target: `2ff0e9a5e5f29566c92ea3f49c82776b57eacfe9`.
Prior branch head: `6b947be8288bdb4a92e48339a23b98da81f3ba94`.
Existing policy: TASK-39144 / ADR-103; no new runtime or lane boundary.

All 2076 incoming dev selectors and the same 27 Textual follow-up cases remain:
2103 unique selectors and matching checker floor. Forty shards, maximum 18
simultaneous UI jobs, retain three scheduling waves. There are 52 eligible test
jobs, with unchanged peak concurrency of 30 test runners, 31 with perf-guard.
Incoming architecture-review/code-identity and variant-stream lane additions,
job/test deadlines, serial commands and required aggregation remain intact.

## Separate evidence and assumptions

- `capacity-entry-weights.json` retains the original 2034 measured entry costs,
  byte-for-byte unchanged. Its 31 historical run sets / 1034 green UI jobs plus
  three recent runs / 108 green UI jobs are not downloaded again.
- The measured 467.0781627 pytest minutes retain the original 476.64-minute
  envelope through the separately recorded proportional 9.5618373-minute
  reserve. That distribution remains an assumption, not observed timing.
- The original 27 new cases retain 1385.997 raw Windows JUnit seconds. Their
  source receipts and bounded unchanged-control comparison remain recorded.
- `upstream-entry-charges.json` explicitly prices all 42 incoming selectors from
  their existing task notes at the pinned dev revision. None uses an untimed
  fallback. These loaded-host serial measurements are not Windows or actual
  new CI observations. The latest Plugins PR's UI jobs were all queued when
  checked; the previously cached CI logs do not time these new selectors.
- Prior report, model, scripts and exact 2061-selector census are retained under
  `pre-rebase-6b947be828/`. The first six files are byte-for-byte prior artifacts;
  `census.txt` is the verified prior commit's file. Downloaded full logs remain
  scratch, outside tracked QA.

| Incoming group | Listed seconds | Explicit reserve | Charged seconds |
| --- | ---: | ---: | ---: |
| Plugins: TASK-40311 | 384.0 | 0 | 384.0 |
| Plugins: TASK-40712 | 51.0 | 0 | 51.0 |
| Plugins: TASK-40308 | 64.0 | 0 | 64.0 |
| Workspaces category: TASK-40319 | 682.5 | 21.1 | 703.6 |
| Workspace lifecycle: TASK-40318 | 602.1 | 21.7 | 623.8 |
| Two archive files: TASK-40320 | 39.8 | 0 | 39.8 |
| Total | 1823.4 | 42.8 | 1866.2 |

The Workspace notes report whole-run totals higher than their listed per-case
values. Their 42.8-second difference is retained conservatively and distributed
proportionally within each associated group. Its cause is unknown; it is not
claimed as setup or observed per-entry work. The reproduction script asserts
that the Plugins subtotal is 499 seconds and the input groups sum to 1823.4,
42.8 and 1866.2 respectively.

Total: original model 499.73995 minutes + incoming 1866.2 / 60 =
530.8432833 pytest minutes.

| Scenario | Shards | Ideal floor | Rebalanced maximum | Swaps |
| --- | ---: | ---: | ---: | ---: |
| Listed charges plus explicit reserve | 39 | 13.611366 | Impossible at 13.3 target | — |
| Listed charges plus explicit reserve | 40 | 13.271082 | 13.287801 | 61 |
| Exact incoming cases, without new reserve | 40 | 13.253249 | 13.291665 | 58 |
| Original Windows charges ×2.61 only | 41 | 13.854493 | Impossible at 13.3 target | — |
| Original Windows charges ×2.61 only | 43 | 13.210098 | 13.299370 | 31 |

All values are modeled pytest minutes. The applied maximum plus the historical
1.7-minute setup allowance gives 14.987801 minutes, leaving 5.012199 of the
20-minute cap. The 2.61 multiplier remains a sensitivity scenario, not a Windows
calibration; loaded-host incoming charges/reserves are not multiplied. The next
PR run must execute the new shard order; no actual 40-shard execution is claimed.
TASK-39144's post-change observation remains open.

## Reproduce without GitHub logs or Git history

From the repository root:

```powershell
python Docs/superpowers/qa/2026-11-14-textual-pattern-followups/ci/replay_capacity.py
```

Expect 2103 entries, 40 shards, maximum 13.2878012 and
`matches_recorded_order: true`.

Reproduce the exact 61-swap search from the archived prior census; the script
appends the recorded 42 incoming selectors in their original insertion order:

```powershell
python Docs/superpowers/qa/2026-11-14-textual-pattern-followups/ci/replay_capacity.py --census Docs/superpowers/qa/2026-11-14-textual-pattern-followups/ci/pre-rebase-6b947be828/census.txt --rebalance
```

For the separate Windows-only sensitivity use `--charge-multiplier 2.61 --shards
43 --rebalance` with that archived census. Inputs retain measured weights and
reserves separately. The script never rewrites the census. Reproduce the original
38-shard model with the archived script and its `--census .../census.txt`.

## Validation

Rebase census checker passes at 2103 entries/floor. Twelve targeted count,
40-shard unique/order-coverage, required-aggregation, architecture-review,
architecture-guard and recovered-stream lane contract cases pass with zero
failures/errors/skips in 20.72 seconds. Parent JUnit SHA256:
`205f9ed9931ec421b734bd1d620a1071c290bd5783505ac1932556ccceb87b16`.
Both the final-order replay and the archived-census 61-swap reproduction match
the recorded order and maximum.

Scoped diff checks pass. Ruff retains the same one baseline `B905` in the replay
tool, with zero new diagnostics. The same three pre-existing formatter-debt files
remain; checker and current reproducer pass formatting. Earlier 104
checker/replay and nine 38-shard guard results stay dated in the archived report;
they are not relabeled as rebase verification. No full suite was run.
