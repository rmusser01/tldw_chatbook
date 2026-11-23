---
id: TASK-42460
title: "The Trace long-inspector test reads the overflow cue after one pause and sometimes finds it empty, on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 21:31
labels:
- follow-up
- trace
- tests
- flaky
priority: low
references:
- Tests/UI/test_trace_responsive.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in TASK-42414's covering runs. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**The flake.** `Tests/UI/test_trace_responsive.py::test_long_inspector_is_focusable_scrolls_to_painted_bottom_and_full_pane` (line 198) failed once at load 47 with `assert '▼ more — scroll' in ''` at line 225, then passed 4 of 4 serially. In 32 repeats at `-n 4` it failed 1 and 0 times with dev's Trace screen file, and 2 and 0 times with TASK-42414's, so it flakes at dev too. The test presses Enter, waits one `pilot.pause()` (line 216), and reads `#trajectory-inspector-overflow`. The assert just before it, `inspector.max_scroll_y > 0` (line 224), passed, so the overflow was there and only the cue text was missing. The cue is written by `_schedule_inspector_cue` (`UI/Screens/trajectory_screen.py:1675`), which runs a second pass 10 ms later against the inspector's layout. Whether the read lands before that pass, or the cue pass runs before layout is final, is not known.

**CI.** The file is in no PR lane: it is not in `scripts/ui_pr_gate_census.txt` and no `derived-artifacts.yml` step names it, so CI has never run it on a pull request. TASK-42414's scan of 84 failed Derived Artifacts runs (2026-11-20 03:35Z to 2026-11-22 18:09Z) found no Trace node failing other than its own. Low priority for that reason; once the file is laned, this becomes a lane flake.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The race is named with evidence: a temporary delay (reverted) in the cue's pass or the inspector's layout makes the failure at line 225 reproducible
- [x] #2 A product race is fixed in the screen, so the cue is right whenever the inspector overflows; a test-timing race is fixed in the test, which waits for the cue with a load-tolerant deadline. The test keeps its assertions
- [x] #3 The test passes in repeated runs of its file under parallel load (at least 32 runs at `-n 4`, or a soak recorded in the notes)
- [x] #4 The file runs in a PR lane, with its serial time under the lane's flags recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Name the race with a temporary, reverted delay (the cue's pass or the inspector's layout) that makes :225 fail reproducibly.
2. Product race: reconcile the cue in the screen when the inspector's layout lands. Test-timing race: wait for the cue with a deadline in the test. Keep every assertion.
3. Soak the file 32 times at -n 4 under load.
4. Lane the file as a whole-file tail entry in the UI census, timed serially with the lane's flags.

## Implementation Notes

**Race (AC #1).**
- A `call_after_refresh` pass and a 10 ms timer set the cue. While a layout is pending, `Screen._on_idle` does not run queued callbacks; it resumes the update timer, and `_on_timer_update` lays out and then hands them to `call_next`.
- When the update timer is starved, `Pilot.pause()` ends with its own `screen._on_timer_update()`. The test then reads the overflow (`max_scroll_y > 0`) before the queued cue pass has run.
- Control (temporary, reverted): in `_show_inspector`, the screen's update timer was paused and swapped for a no-op for 0.5 s. At base this failed the test at :225 3 of 3, and `test_overflow_cue_reconciles_into_the_100x30_compositor_frame` at :268 3 of 3.
- Ruled out by controls:
  - Delaying only the timer pass to 0.3 s passed 6 of 6.
  - A 50 ms busy pump after scheduling passed 8 of 8, and 64 natural runs at `-n 4` (load ~17) passed.

**Fix.**
- **Product.** The screen subscribes to its own `screen_layout_refresh_signal` (`immediate=True`) and sets the cue in the layout that shows the overflow. A `Static.update` with unchanged text is skipped, because an update lays out again. The two passes and the four calls that scheduled them are deleted.
- **Tests.**
  - With the fix and the same hold, the 100x30 test passes. The 60x18 test passes :225, then fails at its `end` step (:230): Textual's `end` binding is a deferred, animated `scroll_end`, the same race. Its `end` step now waits for `scroll_y == max_scroll_y` with a 10 s deadline. With that, both pass 3 of 3 under the hold.
  - Sibling: at load ~37, `test_inspector_scroll_survives_same_event_refresh_and_resets_on_change` failed 4 in 32 at base and 4 in 32 with the screen fix: 3 at :991 (deferred `scroll_home`, `8 == 0`) and 1 at :975 (`before` read mid-animation after `end`). It now waits for the settled `end` and for the scroll home: 64 of 64 at load 37-39.
  - Every assertion is kept.

**Soak (AC #3).** 32 of 32 runs green (157 tests each, load 49-89; one combined serial run of the four Trace census files with the lane's flags: 124 passed in 222.9 s at load ~74-85) (see TASK-42457), which includes `test_trace_responsive.py`.

**Lane (AC #4).** `test_trace_responsive.py` is a whole-file entry at the tail of the census: 37 tests in 55.6 s serially with the lane's flags, at load 48.8 (heavily loaded).

**Files.** `tldw_chatbook/UI/Screens/trajectory_screen.py`, `Tests/UI/test_trace_responsive.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md`.
