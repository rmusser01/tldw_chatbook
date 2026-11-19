---
id: TASK-41765
title: "Two Media tests can pass with their patch inert: the queued-grip landing test and t22209's counting wrapper"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
priority: low
references:
- Tests/UI/test_library_media_side_by_side.py
- Tests/UI/test_library_media_reader_match_nav_t22209.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two Media tests can pass even when their patch never takes effect.

**The queued-grip landing test.**
- `Tests/UI/test_library_media_side_by_side.py::test_compact_media_empty_page_landing_survives_a_queued_grip_focus` (`:657`) reproduces TASK-40293's grip parking. It replaces the controller's `_reset_library_media_reader_to_no_selection` with `reset_then_park_on_grip` (`:682-692`), but never asserts that the replacement ran.
- When TASK-36952.3.3 moved the reset into the controller, the patch's screen receiver went stale. The rebase agent's probe showed the test still green with 0 interceptions.
- The rb4 rebase fixed the receiver. Its review (2026-11-17, finding M3) left the missing assertion as a follow-up, because ruling R1 allowed only receiver edits in the slice.
- The test is on a PR lane (`scripts/ui_pr_gate_census.txt:3100`). CI therefore runs a test that can pass without reproducing the race it was written for.

**t22209's counting wrapper.**
- In `Tests/UI/test_library_media_reader_match_nav_t22209.py`, `_wrap(module, name, key)` (`:103-113`) returns silently when `module` has no `name`.
- After a move, a stale wrap counts nothing, and the counts the test asserts lose their subject without an error (TASK-36952.3.6 report).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The queued-grip test fails when its patched reset never runs
- [ ] #2 t22209's wrapper fails when the name it wraps is missing
- [ ] #3 Each change is shown red with its patch made inert and green as committed, and the queued-grip test stays on its lane
<!-- AC:END -->
