---
id: TASK-2309
title: Check now shows progress and completion
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: pressing "Check now" produces ~5 seconds of dead air — no progress
indicator, no completion signal, nothing preventing a confused second click
from queuing duplicate work.

UAT finding F19.

## Acceptance Criteria (the what)

- [ ] Triggering a check gives immediate visible acknowledgment, a busy
      state while running, and a completion signal (including the failure
      case).
- [ ] A second activation while a check runs is debounced or explicitly
      queued, never silently duplicated.

## Implementation Plan (the how)

1. The screen records which source ids have a check in flight
   (`_checks_in_flight`). `handle_check_now_requested` refuses a second
   activation for a source already being checked, with a toast that says so
   -- debounce, stated, never silent.
2. Immediate acknowledgment: a `Checking <name>...` toast posted before the
   worker starts, and a busy state that outlives the toast -- both Check now
   buttons (Sources pane and Inspector) go disabled and read `Checking...`
   for the source being checked.
3. Completion: the existing success/failure toasts gain the source's name and
   the run's own counters, and the busy state is cleared in a `finally` so a
   raising check cannot strand a permanently-disabled button.
4. Drop `exclusive=True` from the check worker in favour of a named group: it
   made a second press CANCEL the first mid-write, which is exactly the
   unsound cancellation-supersede TASK-1541 documents (a cancelled
   `execute_run` leaves its run row at `running` forever).
5. Tests: a new UI file covering acknowledgment, busy state, the debounce
   refusal, and the failure path's completion signal.
