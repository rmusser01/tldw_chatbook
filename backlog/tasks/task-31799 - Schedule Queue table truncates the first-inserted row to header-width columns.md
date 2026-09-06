---
id: TASK-31799
title: Schedule Queue table truncates the first-inserted row to header-width columns
status: Done
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - schedules
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Creating the first reminder renders the queue row with Title clipped to 5 chars and Details to 7 ('UAT f' / 'One-tim') despite ample free width; any filter keystroke re-renders with correct widths and it stays correct. Columns are auto-sized before the first content insert and not re-measured on row add.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The first row added to an empty Schedule Queue renders with correctly measured column widths.
<!-- AC:END -->

## Implementation Notes

**No longer reproduces on current dev (Done-already-fixed) — regression guard added.**

Attempted to reproduce on current dev (tip 5894f4755e) both ways:

- Harness, faithful end-to-end: empty queue → create the first reminder through the real pushed `ReminderForm` flow → the `#scheduling-task-table` `DataTable` renders the first row with the Title column's `content_width` measured to the full title length (29, not the 5-char header width) and the compositor paints the full "UAT first reminder title here" text. No truncation.
- Live (isolated scratch profile, tmux): navigated to Schedules with an empty queue, created the first scheduled task via `n` → the queue row shows the full "UAT first reminder title here" title and full "One-time at 2026-09-13 16:00 UTC · …" details — not the "UAT f"/"One-tim" header-width clip the UAT reported.

The bug was found on an earlier dev (8e9d1128d4). PR #2454 (bbca9eaa85) subsequently reworked the same `_render_table` clear/add-row path (task-31713's `scroll_x` preservation), and the first-insert measurement is now correct. Rather than ship a speculative no-op fix, added a regression test that drives the real empty→first-row create flow and asserts both the measured column width and the painted output, so a future change cannot silently reintroduce the truncation.

**Files:** `Tests/UI/test_schedules_workbench.py` (`test_first_row_added_to_empty_queue_is_not_truncated`).
