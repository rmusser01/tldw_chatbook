---
id: TASK-25829
title: Footer database size and token timers stall the event loop
status: Done
assignee: []
created_date: '2026-08-31 05:09'
updated_date: '2026-08-31 13:50'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The application log records event loop stalls above one second with the footer database size and token polling timers active. In a keyboard-driven terminal application a stall of that length buffers keystrokes and delivers them late, which is the same input-integrity failure class addressed in earlier Console work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Footer polling work runs off the event loop
- [ ] #2 No footer-attributed stall exceeds the diagnostic threshold under normal use
- [ ] #3 Typing remains responsive while footer counters refresh
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
INVALID AS FILED -- I read causation into a correlation.

I attributed a 1307ms event-loop stall to the footer DB-size and token timers
because the diagnostic line listed them: 'event=event_loop_stall
active_timers=footer-db-size-periodic,footer-token-periodic,ui-heartbeat ...
lag_ms=1307'. But active_timers is a SNAPSHOT OF WHAT WAS ARMED, emitted as
context beside the measured lag (Utils/ui_responsiveness.py) -- it attributes
nothing. Any long-lived timer appears in every stall record regardless of blame.

And the work those timers do is already off the loop: DBStatusManager.
update_db_sizes wraps its collection in asyncio.to_thread(self._collect_db_sizes)
with only the status assignment back on the loop. There is no on-loop I/O to move.

THE STALL IS STILL REAL and worth chasing -- 1307ms of buffered keystrokes is
exactly the input-integrity class earlier Console work fought. The same record
carried mounts=4 removes=1, and the two other stalls I captured carried mounts=8
removes=2 and mounts=0 removes=0, which points at widget mount/removal cost
rather than at either timer. That is a different investigation and deserves its
own task with a profile behind it, not this one's premise.

LESSON: a diagnostic field that lists concurrent state is not an attribution.
Before naming a culprit from a log line, check whether the field means 'caused
by' or merely 'present at the time'.
<!-- SECTION:NOTES:END -->
