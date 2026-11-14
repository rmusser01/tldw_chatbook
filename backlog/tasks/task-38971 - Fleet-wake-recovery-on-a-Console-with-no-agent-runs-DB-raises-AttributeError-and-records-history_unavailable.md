---
id: TASK-38971
title: "Fleet-wake recovery on a Console with no agent runs DB raises AttributeError and records history_unavailable"
status: Done
assignee: ['@claude']
updated_date: '2026-11-06 04:57'
created_date: 2026-11-06 01:21
labels:
- follow-up
- console
- agents
priority: low
references:
- tldw_chatbook/Chat/console_fleet_wake.py
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/UI/Console_Modules/agent.py
- Tests/UI/test_console_send_acknowledgement_extended.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

Every test in `Tests/UI/test_console_send_acknowledgement_extended.py` logs "wake recovery failed (exception_type=AttributeError)", passing and failing tests alike. A probe that wrapped the warning traced it to `console_fleet_wake.py:1441`, `ledger = self._runs_db().automatic_work`, which fails with `AttributeError: 'NoneType' object has no attribute 'automatic_work'`.

**Why `_runs_db()` returns None**
- `_runs_db()` (`tldw_chatbook/Chat/console_fleet_wake.py:1643-1647`) returns None when the controller has no `_agent_bridge`.
- The bridge is None when there is no durable ChaChaNotes DB (docstring at `tldw_chatbook/UI/Console_Modules/agent.py:1140-1143`; the in-memory test harness is one such case). It is also None when `ensure_activity_receipt_service()` returns None (`console_runtime.py`, in `ensure_agent_bridge`).
- The other readers guard against None (`console_fleet_wake.py:342-347`, `:1038`). `recover()` does not.

**What happens next**
- The `except` block at `:1475-1490` logs the warning and fails closed: `_recovery_ready=False`, reason `history_unavailable`.
- `start_recovery` runs only once (`:1378-1388`), and `console_runtime.py:5297-5302` calls it when the controller is created, so recovery never retries.
- `wait_for_recovery()` therefore stays False for goal runs (`console_goal_runs.py:501`, `:942`) on a Console where no wake can exist.
- The visible effect is a misleading warning, plus that recovery state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With no agent runs DB, wake recovery ends without raising and logs no "wake recovery failed" warning; the notes record the resulting state (ready with nothing to recover, or a distinct reason) and why; a test is red on dev
- [x] #2 A runs DB whose read raises still fails closed with history_unavailable, and the existing fleet-wake tests pass
- [x] #3 A run of Tests/UI/test_console_send_acknowledgement_extended.py logs no wake-recovery warning
<!-- AC:END -->

## Implementation Plan

1. Add a test beside the failed-audit case in `Tests/Chat/test_automatic_wake_recovery.py`: a coordinator whose controller has no agent bridge recovers ready, with no failure reason and no "wake recovery failed" warning; red on dev.
2. In `ConsoleFleetWakeCoordinator.recover`, treat a missing runs DB as nothing to audit (ready, no failure reason) instead of dereferencing None; a runs DB whose read raises still fails closed.
3. Run the fleet-wake recovery tests and `Tests/UI/test_console_send_acknowledgement_extended.py`, and check the latter logs no wake-recovery warning.

## Implementation Notes

`ConsoleFleetWakeCoordinator.recover` now handles a missing runs DB before it reads the ledger. If `_runs_db()` returns None, recovery ends ready, with no failure reason, and returns. It no longer dereferences `None.automatic_work`, which raised AttributeError, logged "wake recovery failed" and fenced the runtime as `history_unavailable`.

**Resulting state, and why.** Recovery is ready and has nothing to recover: `wait_for_recovery()` is True, `recovery_result` is None, and `pause_reason()` is None.
- The bridge carries the agent-runs DB. It is None only when there is no durable ChaChaNotes DB, or no receipt service (`ensure_agent_bridge` / `ensure_activity_receipt_service`). With no runs DB, no wake was ever recorded, so the audit has nothing to read.
- The case does not change during a runtime's life: the bridge exists iff the DB is durable. So a distinct reason would only re-label the same empty audit.
- Goal recovery already reads a None `recovery_result` as "nothing to project" (`console_goal_runs.recover` returns `()`). A wake delivery with no ledger still pauses itself as `history_unavailable` (`console_fleet_wake.py` delivery path), so being ready admits no work that needs the ledger.

**Red then green.** I added `test_a_console_with_no_runs_db_has_nothing_to_recover` to `Tests/Chat/test_automatic_wake_recovery.py`, beside the AC#2 case. Before the fix (dev e8eeae7db9) it is red at `assert await wake.wait_for_recovery()` (`assert False`), and the run logs `wake recovery failed (exception_type=AttributeError)`. After the fix the file passes, 7 of 7.

**AC#2.** `test_failed_startup_audit_is_visible_and_never_admits_work` still fails closed with `history_unavailable` (it patches the ledger read to raise) and passes.

Covering run: 28 fleet-wake, goal and launch-recovery files under `-n 4`.
- Head: 61 failed, 179 passed.
- Base worktree e8eeae7db9, same command: 61 failed, 178 passed. The 61 failing node ids are the same on both, and the 179th pass at head is the new test.
- 39 of the 61 fail with the known local `RecoveryRequired: raw_source_selection_changed` trap (TASK-38111). None of the 61 is in a PR lane. The one laned launch-recovery node passes.
- `Tests/Chat/test_automatic_launch_recovery.py` covers recovery start, waiting and failure, so I ran it again under a temporary `pytestmark = pytest.mark.bootstrap_profile` and then removed the marker: 22 of 22 passed at head.
- The other 11 unmarked fleet-wake files ran under the same temporary marker, which I removed afterwards. Head: 9 failed, 53 passed. The fix reverted (`console_fleet_wake.py` from e8eeae7db9): 9 failed, 53 passed. The same 8 node ids fail both times. The ninth is a different parametrization of the same automatic-work-pause test (`model_call_budget` at head, `output_tokens_budget` with the fix reverted), so that test flakes under load ~45.

**AC#3.** `Tests/UI/test_console_send_acknowledgement_extended.py`, `-n 4 -rA` (captured stderr printed for all 8 tests): the head logs 0 "wake recovery failed" lines and the base worktree logs 8.

Files: `tldw_chatbook/Chat/console_fleet_wake.py`, `Tests/Chat/test_automatic_wake_recovery.py`. This changes no UI, so no User Guide change.
