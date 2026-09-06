---
id: TASK-31800
title: Reminder form 'Run at' placeholder and examples hardcode a past datetime
status: Done
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - schedules
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). The 'Run at' placeholder and helper/error copy use the literal '2026-08-28 09:00' (already in the past): UI/Screens/scheduling/forms/reminder_form.py:513, definition_detail.py:1266 and :1606. A user copying the example gets an already-due run time. Generate the example relative to now, or use an obviously-future fixed date.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The run-at example shown to users is always a future datetime (or clearly synthetic).
- [x] #2 All three literal sites updated consistently.
<!-- AC:END -->

## Implementation Notes

**Already fixed on current dev (Done-already-fixed) — regression guard added.**

The hardcoded literal is gone. PR #2454 (bbca9eaa85, task-31711 AC#4) replaced all 12 hardcoded example-date sites with a single `example_run_at_text(days_ahead=7)` helper (`tldw_chatbook/Scheduling/schedule_input_parsing.py`) that computes a never-in-the-past `"YYYY-MM-DD 09:00"` fresh relative to `datetime.now()`. The three sites the UAT flagged now all render it:

- `forms/reminder_form.py` — the `Run at` `Input` placeholder + the "A local time like {…}" hint (and the parse-error/preview copy).
- `definition_detail.py:1286` (example) and `:1627` (error copy).

A `grep` for any `YYYY-MM-DD` literal across the reminder/automation forms and detail panes now returns nothing.

**Live verification** (tmux, current dev, 2026-09-06): the `Run at` placeholder rendered "2026-09-13 09:00" (7 days ahead — future), and the hint read "A local time like 2026-09-13 09:00, or full ISO-8601 with offset."

**Test.** The helper had no direct coverage; added `Tests/Scheduling/test_example_run_at_text.py` pinning that the example is always in the future, has the shared `days_ahead` shape, and is not a fixed literal.

**Files:** `Tests/Scheduling/test_example_run_at_text.py` (regression test).
