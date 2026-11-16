# Fresh capacity model for dev cc6006fc

2026-11-15; PR3569 resync onto `cc6006fcf59328353004206bba8724c399216749`.
This model supersedes the earlier PR layout for the current union. The older
QA reports retain their own dated scope and are not rewritten as fresh evidence.
Existing TASK-39144 / ADR-103 scaling policy applies; no new ADR is required.

All 2145 upstream selectors and the same 27 Textual follow-up cases remain:
2172 unique selectors, matching floor, 40 shards, maximum 19 concurrent UI jobs.
The upstream 13.0-minute replay target, serial commands, deadlines and all new
upstream lanes remain unchanged. Forty shards at 19 concurrent jobs create a
third scheduling wave. The workflow has 52 eligible test jobs and peak
concurrency of 31 test runners, 32 with perf-guard.

## Fresh inputs and limits

The upstream [capacity decision](https://github.com/rmusser01/tldw_chatbook/commit/46a9132c3cdd808c8da8bcac5d010c4ebaffc9f1)
uses 25 PR run sets from 2026-11-14 03:20Z through 19:32Z, with 867 successful
UI jobs: 17 hosted run sets / 599 jobs and eight self-hosted / 268. That exact
count and window are recovered here. Three complete cached runs were reused;
only 22 missing archives were downloaded. Failed jobs are excluded. Two rerun
archives required their second-attempt jobs alongside the latest third attempt;
their first-attempt archives return 404. Each run/shard is used once. Downloaded
logs remain disposable scratch, not tracked QA. Run identities and URLs are in
`capacity-summary.json`.

Every original 2034 upstream selector has a measured weight. All 111 upstream
tail selectors have positive task-note serial charges, totaling 2174.82 seconds
against the upstream rounded 2174.8. Inputs retain task paths, revisions and
hashes. These charges follow the upstream selection, including conservative
earlier timings where its notes chose them. They are separate from the measured
weights and from the 27 Windows JUnit charges.

The recovered upstream total is 492.9432526 pytest minutes, slightly above the
upstream rounded 492.8. Current available attempt logs are used, and this small
difference is disclosed rather than forced downward. No old PR reserve or
weight multiplier is carried forward: the upstream cost factor is 1.0. The 27
PR cases retain 1385.997 measured Windows seconds, giving 516.0432026 total
modeled pytest minutes. The earlier bounded unchanged-control calibration and
receipt hashes remain in the prior report; they are not a performance guarantee.

| Scenario | Shards | Ideal floor | Rebalanced maximum | Swaps |
| --- | ---: | ---: | ---: | ---: |
| Raw Windows charges | 39 | 13.231877 | Impossible at 13.0 | — |
| Raw Windows charges | 40 | 12.901080 | 12.997614 | 36 |
| Windows-only ×2.61 sensitivity | 42 | 13.172241 | Impossible at 13.0 | — |
| Windows-only ×2.61 sensitivity | 43 | 12.865910 | 12.995365 | 34 |

All values are modeled pytest minutes. The upstream median model offsets,
1.82 minutes hosted and 0.58 self-hosted, project about 14.817614 and 13.577614
minutes respectively for the applied maximum. Those are estimates, not observed
execution or guaranteed worst-case headroom. The 2.61 factor applies only to the
27 Windows charges as an uncalibrated sensitivity; it does not set this count.
Actual new-matrix execution and TASK-39144's post-change observations remain open.

## Reproduce without GitHub logs

Run from the repository root:

```powershell
python Docs/superpowers/qa/2026-11-14-textual-pattern-followups/ci/resync-cc6006fc/replay_capacity.py
```

Expect 2172 entries, 40 shards, maximum 12.9976141 and
`matches_recorded_order: true`. Reproduce the 36-swap search from the exact
upstream census; the script appends the recorded 27 cases in insertion order:

```powershell
python Docs/superpowers/qa/2026-11-14-textual-pattern-followups/ci/resync-cc6006fc/replay_capacity.py --census Docs/superpowers/qa/2026-11-14-textual-pattern-followups/ci/resync-cc6006fc/upstream-census.txt --rebalance
```

Add `--windows-multiplier 2.61 --shards 43` to that command for the sensitivity
scenario. The existing replay algorithm is reused, and the script never edits
the census. Counts, positive prices and charge sums are asserted. Future census
changes require new priced inputs rather than an untimed default.

## Validation

For the verified 2172-selector resync model, the census checker passes and 15 targeted
count, 40-shard unique/order-coverage, selector, aggregation and incoming-lane
contract cases pass with zero failures/errors/skips in 61.41 seconds. JUnit
SHA256: `2bf8bea306696190755533575ceb0d80e6af710e6be1e370d2afdf6349be7942`.
The final-order and upstream-census 36-swap replays both match the recorded
order and maximum. Ruff retains the same one baseline finding with no new
diagnostics; three inherited formatting-debt files remain, while the checker
and new reproducer pass formatting. Scoped diff checks pass.

The actual-method Library focus regression is routed separately to the existing
Architecture guard lane through exactly
`Tests/Architecture/test_library_media_wiring.py::test_emptied_page_focus_callback_retains_admitted_owner`,
mirrored in `ARCHITECTURE_GUARD_TARGETS`. This file previously had no PR route;
only this function is enrolled. Its final shape is one mounted
`bootstrap_profile` case looping twelve held-callback scenarios, with every call
bound to the real screen/controller. The strict unbound-call pin stays at 722.
After the test's loop captures were bound through explicit default arguments,
the final selector passed with zero failures/errors/skips in a 79.63-second
pytest run. Its exact JUnit case time is 68.533 seconds; JUnit SHA256:
`5d9d9b9a7567f810d54b445ab30295884285084f66887245a26d53d86b98df12`.
Final test-file SHA256:
`db39a65102abc45ec227b3bf4f1327373263e5ec1e791b4454d58a6179712af6`.
The earlier eight-target qualification passed in 195.29 seconds and retains
evidence for the unchanged pin; its 44.685-second selector timing is historical.
That run's JUnit SHA256 is
`40427d737e9c12e58f826ff7df573117701d8f3405048fe25c026d24a19d5a26`.
The earlier ten unmounted cases and their 17.03-second run are retained as
historical qualification in `architecture-route.json`, not as final evidence.
Eighteen successful Architecture jobs in the recovered metadata had a
10.367-minute maximum and 7.967-minute median against the unchanged 20-minute
cap; this bounded 68.533-second addition does not require a new job. The final
mounted selector runs last, after the existing terminal-privacy file, so no
sandboxed target follows the new bootstrap-profile case. The complete 42-target
inspection is in `architecture-lane-order.json`. It also records an unchanged
upstream mixed-profile file at position two; that inherited placement is not
rewritten or claimed to be a sandboxed-only sequence.

The Architecture route leaves the UI census at 2172, so the model above remains
unchanged. Prior 38/40-shard results stay dated in their original reports.
After the final bootstrap-tail placement, three targeted lane-order/mirror and
selector resolution/overlap checks pass with zero failures/errors/skips in
35.05 seconds; JUnit SHA256
`a76dc837cc43fd3287797521dadd27075b945b3bddd6ea129c93399bff223560`.
The earlier five routing/census checks remain historical evidence in the JSON.
The UI census and model are unchanged at 2172, 40 shards and concurrency 19.
Modified routing Python passes Ruff; checker/reproducer format and scoped diff
checks pass. The restored UTF-8 arrow/ellipsis checker comments remain intact.
No full test sweep or new 40-shard CI execution is claimed.
