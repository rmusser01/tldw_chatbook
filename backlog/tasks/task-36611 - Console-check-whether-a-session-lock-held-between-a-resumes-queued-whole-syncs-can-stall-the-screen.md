---
id: TASK-36611
title: >-
  Console: check whether a session lock held between a resume's queued whole
  syncs can stall the screen
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-24 07:50'
updated_date: '2026-10-27 16:09'
labels:
  - console
  - reliability
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.4 changed `Tests/UI/test_console_tab_publication_maintenance.py::test_admitted_real_tab_publication_rechecks_replaced_controller` to wait for idle across three drained pauses. The test took the session lock while a resume's whole sync was still queued, and Pilot's wait for pending messages timed out (`WaitForScreenTimeout`). The test is flaky on dev: one review saw it fail 1 run in 4, and on 2026-10-24 it failed 4 of 4 on dev under load. With dev's version of the test, the branch failed every time. The branch's fix changed only the test.

Nobody has checked whether production can reach the same state, where a session lock held between a resume's queued whole syncs blocks the Console screen from processing its messages. If it can, a user would see the Console stop responding until the lock is released.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Why the test's wait timed out is diagnosed and recorded in the task notes
- [x] #2 A deterministic test either reproduces the stall through production code paths or shows that no production caller holds the session lock across a resume's queued whole syncs
- [x] #3 If the stall is reproducible, the Console keeps processing messages while the lock is held, and the test fails on the base build
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure: probe the test's timeline (which syncs run where, what holds which lock) on dev, old vs current wait.
2. Diagnose the WaitForScreenTimeout and record it (AC1).
3. Decide whether a production holder of the session locks can wait on the Console screen's message pump while a screen-pump whole sync waits on the lock.
4. Write a deterministic test through production paths (real holder sync, real screen-pump queued sync); show with a mutation that it fails when the holder needs the screen pump (AC2).
5. If a production stall exists, fix it test-first (AC3); otherwise make the flaky pin's precondition explicit and deterministic.
6. Gates: targeted files vs dev, preflight, rebase.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Outcome: production can reach the state, but it does not stall the screen beyond one tab publication; the 30 s timeout was a circular wait that only the test could create. No product change. The pins now wait for the screen's pump to be idle before taking the lock, and a new pin covers the production case.

**Why the wait timed out (AC1).** A `sys.monitoring` probe on the sync methods ran dev 3be493413a with dev's original single settle. The whole sync that completed the Console attach ran on a worker and, as it finished, scheduled `_reconcile_console_after_attach` with `call_after_refresh`. The settle (no sync in progress, no replay, projection not pending) was already true at that moment, so the test took the surface `_session_sync_lock` at t=2.0797 s. At 2.0804 the reconciliation ran on the screen's pump and queued `_start_console_view_after_reconciliation`'s whole sync the same way. Textual awaits every after-refresh callback, from any widget, on the screen's own pump. At 2.089 that whole sync took the screen's `_console_session_tabs_sync_lock` and parked on the surface lock the test held. The test's own publication then waited on the tab lock and never reached the surface method. The test waited for that with `_settle(pilot, ...)`, and `pilot.pause()` calls `_wait_for_screen`, which needs the parked screen pump. The result was `WaitForScreenTimeout`: the test's lock blocked the screen-pump sync, which blocked the test's publication, which left the test waiting on Pilot, which needed the screen pump. One of 7 unforced runs of the old settle reproduced it (at the lock: attach not reconciled, one message in the screen's queue). With a whole sync queued just before the settle, the old settle failed 4 of 4 under `-n 4` with this exact error.

**Production (AC2).** Only `ChatScreen._sync_console_native_session_tabs` takes the tab lock, and only `ConsoleSessionSurface.sync_sessions`, which nothing else calls, takes the surface lock. Whole syncs run both on the screen's pump (the attach and view-start `call_after_refresh`, plus handlers that await one inline) and off it (console-sync workers, the poll timer's task, the send acknowledgement). So the screen's pump can wait on the tab lock while a worker's publication holds it.

That wait is bounded. Inside the lock, the holder awaits only the strip's mount event, the removed buttons' own pumps and the new buttons' own pumps, never the screen's pump. Measured in a probe (3 runs each under 3-way load), the screen's pump stayed parked 38-63 ms after the holder's 4-tab remount and 100-213 ms after a 13-tab remount. Without the contention, the queued sync would do the same remount on the screen's pump itself, so the screen does not wait longer than it would anyway.

New pin `Tests/UI/test_console_session_lock_screen_pump.py`:
- The holder is a real maintenance-resume worker whole sync, parked on the surface lock, which the test holds only as a gate.
- A screen-pump whole sync, queued with `call_after_refresh` as the view start does, parks behind it. The screen task's await chain confirms this.
- The test then releases the lock and holds nothing. The screen's pump must be free within 10 s, Pilot must answer, and the strip must show all four sessions.
- It passed 9 of 9 on dev code (8 of them under 4 parallel lanes).
- Negative control: a temporary await, inside `sync_sessions`' lock and off the screen pump, on a future resolved by `self.screen.call_later` failed it with "the screen's pump stayed parked behind the holder for 10.0s; holder: [... ConsoleSessionSurface.sync_sessions]". The mutation was restored by Edit.

**AC3.** There is no stall to fix: the wait is bounded and the holder does not depend on the screen's pump.

**Pin change** (`Tests/UI/test_console_tab_publication_maintenance.py`, all three pins):
- The new `_publication_quiet` replaces the three-drained-pauses heuristic. It requires all of these:
  - the attach reconciled and not reconciling;
  - the screen's task suspended in `MessagePump._get_message`, with its message queue, `_next_callbacks` and `_callbacks` empty;
  - no sync requested, in progress or replaying, and no tab publication open;
  - the projection not pending, and no Console worker running.
- Nothing awaits between the predicate and the lock acquire.
- Waits made while the test holds the lock poll the event loop (`_loop_wait`), not Pilot. The replaced-controller pin ends with a `pilot.pause()` after release, so a stall still fails it.
- My first version checked `_callbacks` but not the queue or the pump. The forced probe failed it 1 of 1, because the callback was still an `InvokeLater` in the screen's message queue. The final version passed 4 of 4 under the same forced probe.
- Repeats: the maintenance file passed 8 of 8 runs (24 tests) under 4 parallel lanes.

**Other changes**
- Census: the new file is appended at the tail (about 9 s locally), and `MINIMUM_FILES` goes from 908 to 909 with a dated comment.
- Lesson added to `lessons-testing-evidence.md`.
- Not changed: `Tests/UI/test_console_tab_publication_read_barrier.py` takes the surface lock after a settle as weak as the old one and clicks while holding it. It passed 3 of 3 here, and the same hazard applies to it.

Files: `Tests/UI/test_console_tab_publication_maintenance.py`, `Tests/UI/test_console_session_lock_screen_pump.py` (new), `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
