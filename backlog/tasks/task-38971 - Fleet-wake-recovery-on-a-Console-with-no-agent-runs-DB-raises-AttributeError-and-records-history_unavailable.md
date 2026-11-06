---
id: TASK-38971
title: "Fleet-wake recovery on a Console with no agent runs DB raises AttributeError and records history_unavailable"
status: To Do
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
- [ ] #1 With no agent runs DB, wake recovery ends without raising and logs no "wake recovery failed" warning; the notes record the resulting state (ready with nothing to recover, or a distinct reason) and why; a test is red on dev
- [ ] #2 A runs DB whose read raises still fails closed with history_unavailable, and the existing fleet-wake tests pass
- [ ] #3 A run of Tests/UI/test_console_send_acknowledgement_extended.py logs no wake-recovery warning
<!-- AC:END -->
