---
id: TASK-42569
title: "Two citation-repair Stop tests are red on dev and their file runs on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-26 08:14
updated_date: '2026-11-26 16:43'
labels:
- follow-up
- console
- dev-red
priority: medium
references:
- Tests/Chat/test_console_local_citation_boundary.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 27 while running covering tests for TASK-42552 (2026-11-26). `Tests/Chat/test_console_local_citation_boundary.py::test_citation_repair_stop_while_checking_cancels_before_dispatch[direct]` and `[agent]` fail on dev 6e41919880, and no PR lane runs the file, so nothing reports them. The tests pin that a Send while citations are being checked is refused and that a Stop then cancels the repair before it reaches the provider. A real regression there, a repair sent after the user pressed Stop, would read as old noise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes say whether the product or the test is wrong, with the cause and the commit that introduced it
- [x] #2 A product defect is fixed with a test red before and green after; a drifted test is repaired with every assertion kept
- [x] #3 The whole file passes at `-n 4`, and any other red in it is listed and handled the same way
- [x] #4 The file runs on a PR lane that blocks merges to dev, timed with that lane's flags, and the lane contract is updated
<!-- AC:END -->

## Implementation Plan

1. Reproduce the two reds on dev; find where the second Send first suspends and which commit put that suspension before its active-run check.
2. Decide product or test; repair while keeping every assertion; show the repaired test catches a repair sent after Stop.
3. Run the whole file at `-n 4` and serially with recovered-suites' flags; handle any other red.
4. Lane the file in recovered-suites' bootstrap block, time it, update `RECOVERED_SUITES_TARGETS`.

## Implementation Notes

**The Stop test (2 nodes): the test was wrong; it drifted at aed1b13501 (2026-09-27).**
- **What the product does.** The citation check yields exactly once (`await asyncio.sleep(0)`) between "Checking citations…" and its Stop check, then dispatches the repair.
- **What the test assumed.** It waited for that state, then awaited a second `submit_draft` and expected it to be refused without yielding to the loop.
- **Why that broke.** aed1b13501 put `await self.hook_admission_reason()` (an `asyncio.to_thread` config read) in `_submit_draft_lifecycle` before `_submit_draft_inner`'s active-run check. Stepping the blocked send by hand shows its first suspension is today's equivalent, the hook admission's preparation read (`run_preparation_read`). So the check resumed, sent the repair and finished, and the second Send was then accepted as a new turn (`assert True is False` at the `accepted` assertion).
- **A second yield.** Replacing the hook admission with a non-yielding one made the Send refused, but the check still moved on before Stop (the refusal's own row persistence yields too).
- **Product or test.** Stop during checking does cancel before dispatch: Stop cancels the run task, and the held check commits "canceled" from its `CancelledError` handler as well as from its explicit checks. The product is right; the test's one-loop-turn premise is not.

**Fix (test only).** `_hold_citation_check` holds the controller's one check yield until the test releases it. It swaps `console_chat_controller.asyncio` for a copy whose `sleep(0)` waits on an event when called from `_select_post_generation_body`. The test sends, is refused, checks that Stop is allowed, stops, then releases. Every assertion is unchanged.

**Red/green for the Stop test.**
- Dev: 2 failed.
- Branch: 2 passed.
- Negative control: a check that ignores Stop (it swallows the `CancelledError` at the yield and drops both explicit cancel checks) turns both nodes red on the provider-call count (`2 == 1` direct, `1 == 0` agent); reverted.

**A second red, under the lane's flags only: a storage-lease flake.**
- **The failure.** `test_agent_replaced_placeholder_does_not_transfer_finalizer` passes at `-n 4`, but under `--storage-lease-guard` it errored in 2 of 3 whole-file serial runs, and alone in 2 of 3 runs (earlier, 3 of 3 alone passed). The error is "left 1 new live storage lease(s) open: SQLite …/boundary.sqlite on thread console-stream-persist_0".
- **The race.** The test's replacement reply is persisted by `_persist_deferred_message` on the store's single-slot stream-persistence worker (task-33081, 940a2e82b8). Whether that deferred write runs on the worker, and so opens a thread-local handle, before the turn's own terminal persistence inserts the row is timing-dependent. The worker's handle is never closed in the test. In the app, quit close and backup drain close registered handles.
- **The fix (test only).** The module's autouse teardown now drains every `_persisted_store`'s worker (a no-op submitted and awaited) and calls the database registry's `close_if_idle()`, which closes every registered handle including the worker's. Only then does it close the main-thread connection. Each drain is guarded like the persistence loop, so a failed drain still lets every persistence close (review round).
- **Evidence.** There is no CI log, since no lane ran the file before. Single runs of the test under the guard, on the final file:

  | Variant | Passed |
  | --- | --- |
  | With the fix | 10 of 10 |
  | Without `close_if_idle()` | 0 of 6 |
  | Without the drain | 6 of 6 |

  The drain covers a worker still mid-write at teardown, when `close_if_idle()` refuses and closes nothing. That case was not reproduced here.

**Whole file.**
- `-n 4`: 103 passed (2 red before).
- Serially with recovered-suites' flags (`--storage-lease-guard --timeout=180`): 103 passed in 72.6-76.5 s at load 14-20, before the lease fix. Under load it is 111.7-236.3 s (the loaded soak below), and the reviewer measured 225.7 s at load 36-54, so charge the lane about 4 minutes.
- Soak, after both fixes: 20 whole-file runs serially with the lane's flags, 20 of 20 green with 103 passed each. A first soak ran at light load, 4.6-13.9, at 52-94 s per run. A second soak ran at realistic load, 24.1-60.4 (other agents plus a concurrent `-n 4` soak of the touched files), at 111.7-236.3 s per run.

**Lane (AC#4).** The file is in recovered-suites' bootstrap block, after `test_console_route_census_capture.py`. It is bootstrap_profile by name in `Tests/conftest.py`. `RECOVERED_SUITES_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`) and the step's note are updated.

Files: `Tests/Chat/test_console_local_citation_boundary.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
