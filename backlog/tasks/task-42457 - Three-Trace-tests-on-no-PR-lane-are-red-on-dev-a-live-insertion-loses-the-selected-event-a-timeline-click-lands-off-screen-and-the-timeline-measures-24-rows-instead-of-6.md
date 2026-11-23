---
id: TASK-42457
title: "Three Trace tests on no PR lane are red on dev: a live insertion loses the selected event, a timeline click lands off screen, and the timeline measures 24 rows instead of 6"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- trace
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_trace_filters_navigation.py
- Tests/UI/test_trajectory_timeline_integration.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23): TASK-42414's covering runs reported these three nodes red at plain dev 2f7790873a, with dev's Trace screen file and with its own. Re-run on the round-22 stack before its rebase (142e821116): 3 failed in 11.75 s at `-n 3`. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches). Paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** Neither file is in `scripts/ui_pr_gate_census.txt` or a `derived-artifacts.yml` step, so nothing reports these reds. TASK-42414's scan of 84 failed Derived Artifacts runs (2026-11-20 03:35Z to 2026-11-22 18:09Z) found no Trace or trajectory node failing other than its own, which fits: CI never runs them on a pull request.

**Failing nodes and messages.**
- `Tests/UI/test_trace_filters_navigation.py::test_live_insertion_before_selection_preserves_event_id_filters_and_viewport` (line 548): `AttributeError: 'NoneType' object has no attribute 'event_id'` at line 572. The selected event is gone after a live insertion before it.
- `Tests/UI/test_trajectory_timeline_integration.py::test_click_outside_active_brush_atomically_clears_time_and_selects` (line 318): `textual.pilot.OutOfBounds: Target offset is outside of currently-visible screen region.` from the click at line 343.
- `Tests/UI/test_trajectory_timeline_integration.py::test_timeline_renders_bars_for_timed_snapshot` (line 155): `assert 24 == 6` at line 158, where 24 is `TrajectoryTimeline(id='trajectory-timeline', classes='h-6').size.height` on an 80x24 screen. TASK-42414's implementer suspected the timeline is measured without the bundled stylesheet, so its `h-6` class sets no height. That is not confirmed. A full-height timeline could also push the other test's click target off screen; that link is a guess too.

`test_trace_filters_navigation.py` last changed in a0bdb8ad61 (2026-08-23) and `test_trajectory_timeline_integration.py` in ad8a378ef3 (2026-10-02).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the three failures has a recorded cause: test drift after a named product or stylesheet change, or a product defect
- [ ] #2 Drift is repaired in the tests with their assertions kept; a product defect is fixed in product code with a test shown red first
- [ ] #3 The three nodes pass, alone and in a `-n 4` run of their two files
- [ ] #4 The two files run in a PR lane, with their serial time under the lane's flags recorded in the notes
<!-- AC:END -->
