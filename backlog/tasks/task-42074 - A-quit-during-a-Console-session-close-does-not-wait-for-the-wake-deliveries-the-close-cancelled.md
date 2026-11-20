---
id: TASK-42074
title: "A quit during a Console session close does not wait for the wake deliveries the close cancelled"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 02:35
updated_date: 2026-11-20 20:52
labels:
- follow-up
- console
priority: low
references:
- tldw_chatbook/Chat/console_fleet_wake.py
- tldw_chatbook/Chat/console_runtime.py
- Tests/Chat/test_console_fleet_wake_safety.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-42028 (round 17). TASK-42028 makes quit wait, within the shared dispose deadline, for wake deliveries that `ConsoleFleetWakeCoordinator.dispose()` cancelled, before it retires AgentRuns.

`fence_conversation` (session close) still removes the deliveries it cancels from `_delivery_tasks` straight away. `Tests/Chat/test_console_fleet_wake_safety.py` pins that removal. So when the app quits while a closed session's cancelled delivery is still settling its attempt, the quit does not wait for it. The settlement can then race the AgentRuns retirement: at quit this is logged as "SQLite retirement incomplete", and the attempt may be left for the next start's `recover()`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A quit that starts while a session close's cancelled wake delivery is still settling waits for that delivery, within the dispose deadline, before AgentRuns is retired
- [x] #2 A test drives a quit during that settlement window and fails without the fix
- [x] #3 A session close still releases its fence on a graceful close and still keeps no strong reference to a delivery that never settles; `test_session_fence_releases_uncooperative_delivery_task_reference` stays green
<!-- AC:END -->

## Implementation Plan

1. `fence_conversation` still drops the deliveries it cancels from `_delivery_tasks` (so a graceful close can release its fence), but also adds them to a new `weakref.WeakSet` on the coordinator, which holds no strong reference.
2. `ConsoleRuntime._dispose_owned` adds that set's live tasks to its bounded drain, beside `_delivery_tasks` (TASK-42028), so the quit joins them before AgentRuns retires.
3. Test (Tests/UI/test_console_launch_wake.py): a launch wake's turn is in flight, the session closes, the cancelled delivery's settlement is held until the quit starts, and the quit must not retire AgentRuns before it settles. Show it red without the fix.
4. Unit test (Tests/Chat/test_console_fleet_wake_safety.py): a fenced delivery is reachable for the drain and is collectable once nothing else holds it; the existing uncooperative-reference test stays green.
5. Raise the console_runtime.py size ratchet by the measured growth if it fails.

## Implementation Notes

`fence_conversation` still drops the deliveries it cancels from `_delivery_tasks`, so a graceful close releases its fence as before. It now also adds them to `ConsoleFleetWakeCoordinator._fenced_delivery_tasks`, a `weakref.WeakSet`. A delivery that never settles is therefore not kept alive. `ConsoleRuntime._dispose_owned` adds that set to its bounded drain, beside `_delivery_tasks` (TASK-42028) and under the same lock. A quit therefore joins a settling, session-cancelled delivery within the dispose deadline, before AgentRuns retires.

**Tests** (both in `Tests/UI/test_console_launch_wake.py`, a whole-file UI census entry, so the census is unchanged):
- `test_quit_settles_a_wake_delivery_a_session_close_cancelled_before_retiring_agent_runs`: a launch wake stalls in its stream and `close_session` fences it. Its `_finish_attempt` is held until the quit starts plus 0.5 s. The test asserts the delivery left `_delivery_tasks` and is still settling, then asserts AgentRuns was not retired before it settled.
  - Red with base sources (2026-11-20 16:09Z, `-p no:xdist`): "the quit retired AgentRuns while the wake delivery a session close cancelled was still settling its attempt: [[<Task cancelling ... _deliver() ...>]]". Green with the fix.
- `test_session_fence_keeps_only_a_weak_handle_on_a_cancelled_delivery`: the fenced delivery is reachable for the drain, and is collected once nothing else holds it.
  - Red under a temporary `set()` in place of the `WeakSet`: "the fence kept a strong reference to its delivery". Green with the `WeakSet`.

**Deviation from plan step 4.** The unit pin went into `test_console_launch_wake.py`, not `Tests/Chat/test_console_fleet_wake_safety.py`, because no PR lane runs that file. That file is unchanged, and `test_session_fence_releases_uncooperative_delivery_task_reference` in it stays green.

**Runs** (16:09-16:13Z, load 4.6-11.6, all through the wrapper):

| Run | Result |
|---|---|
| `test_console_launch_wake.py` `-n 4` | 21 passed |
| `test_console_runtime_ownership.py` `-n 4` | 94 passed, 1 xfailed |
| The other 12 covering files `-n 4` | 126 passed, 16 failed |
| `Tests/Architecture/test_module_size_ratchet.py` serial | 209 passed |
| `ruff check --select F` on the changed files | clean |
| `Tests/lint/ruff_gate.py` | RATCHET OK |
| `preflight.sh` | all checks passed |

The 16 failures are the same test ids with base sources (16 failed, 3 passed). They are in `test_console_headless_wake_invariants.py` (11), `test_console_fleet_wake_view_mark.py` (4) and `test_console_fleet_wake_staleness.py` (1), which no lane runs, and they fail on the local profile trap (`RecoveryRequired: raw_source_selection_changed`).

Earlier runs made during the 15:50-16:07Z disk-full window were discarded and re-run.

**Proposed follow-up.** `ConsoleChatController.maintenance_drain` waits only on `_fleet_wake._delivery_tasks`. A delivery that a session fence cancelled is therefore not waited for during maintenance, the same gap this task closed for quit. A fix cannot simply add `_fenced_delivery_tasks` to that loop's truthiness test. The `WeakSet` keeps settled tasks for as long as anything references them, so the drain must wait only on members where `not task.done()`.

**Files:**
- `tldw_chatbook/Chat/console_fleet_wake.py`
- `tldw_chatbook/Chat/console_runtime.py`
- `Tests/UI/test_console_launch_wake.py`
- `Tests/Architecture/test_module_size_ratchet.py` (console_runtime.py 7087 -> 7089)
- `backlog/docs/lessons-testing-evidence.md` (gc after `await task` needs a `sleep(0)` first)
