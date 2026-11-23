---
id: TASK-42457
title: "Three Trace tests on no PR lane are red on dev: a live insertion loses the selected event, a timeline click lands off screen, and the timeline measures 24 rows instead of 6"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 21:31
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
- [x] #1 Each of the three failures has a recorded cause: test drift after a named product or stylesheet change, or a product defect
- [x] #2 Drift is repaired in the tests with their assertions kept; a product defect is fixed in product code with a test shown red first
- [x] #3 The three nodes pass, alone and in a `-n 4` run of their two files
- [x] #4 The two files run in a PR lane, with their serial time under the lane's flags recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Reproduce the three reds at base, then trace each to the change that broke it (git history of the tests, the harness and the screen).
2. Repair drift in the tests, keeping every assertion; fix a product defect in the screen with a test shown red first.
3. Run the three nodes alone and their two files at -n 4.
4. Lane both files as whole-file tail entries in the UI census, timing each serially with the lane's flags.

## Implementation Notes

**Cause: test drift in all three; no product defect.**
- `test_timeline_renders_bars_for_timed_snapshot` (`24 == 6`) and `test_click_outside_active_brush_atomically_clears_time_and_selects` (`OutOfBounds`) have one cause. The shared `_mounted` harness in `Tests/UI/test_trajectory_screen.py` was a bare `App`.
  - 82650cc1f7 (TASK-21115, 2026-08-23) moved the timeline's `height: 6` and the filter bar's geometry from `DEFAULT_CSS` to `BUNDLED_CSS`, which only the app bundle loads. It moved ten other harnesses to `ConsolidatedCSSApp`, but not this one.
  - A probe at base: on an 80x24 screen the filter bar was at y=5 and 24 rows tall, and the timeline at y=29 and 24 rows tall, so the click target was off screen. The description's guess was right.
- `test_live_insertion_before_selection_preserves_event_id_filters_and_viewport` (`None.event_id`): 0dd6fc3b14 (2026-09-01) put record rows under a `record:` key prefix. It updated two sibling test files but not this one. `_move_cursor_to_key("event:3")` matched no row and did nothing, so the cursor stayed on the turn header.

**Fix (tests only, every assertion kept).**
- The harness is now `ConsolidatedCSSApp` with `CSS_PATH = BUNDLED_STYLESHEET`. Only `test_trajectory_timeline_integration.py` imports `_mounted`, and `test_trajectory_screen.py`'s own tests pass with it.
- The live-insertion test moves to `screen._record_key(...)`. It now also asserts that the cursor is on `event:3` before the insertion, the check that would have caught this drift.

**Red/green.** At base, the three Trace files at `-n 4` gave 3 failed, 91 passed. After the fix, each node passes alone (9.9 s, 8.0 s, 11.1 s) and the two files pass at `-n 4` (57 passed).

**Lane.** Both files are whole-file entries at the tail of `scripts/ui_pr_gate_census.txt`, and `MINIMUM_FILES` is the measured count. Serial time with the lane's flags (`-p no:xdist --timeout=180 --durations=0 --durations-min=0`), measured on a heavily loaded host:
- `test_trace_filters_navigation.py`: 32 tests in 35.2 s, load 42.6.
- `test_trajectory_timeline_integration.py`: 25 tests in 36.1 s, load 43.0.

**Soak.** 32 of 32 runs green (157 tests each, load 49-89; one combined serial run of the four Trace census files with the lane's flags: 124 passed in 222.9 s at load ~74-85): five Trace files at `-n 4` (`test_trace_responsive`, `test_trace_filters_navigation`, `test_trajectory_timeline_integration`, `test_trajectory_live`, `test_trajectory_screen`).

**Files.** `Tests/UI/test_trajectory_screen.py`, `Tests/UI/test_trace_filters_navigation.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
