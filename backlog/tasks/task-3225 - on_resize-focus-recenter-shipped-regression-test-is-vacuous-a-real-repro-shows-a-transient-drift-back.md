---
id: TASK-3225
title: >-
  on_resize focus-recenter: shipped regression test is vacuous; a real repro
  shows a transient drift-back
status: To Do
assignee: []
created_date: '2026-08-09 14:34'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-3200 round 3 takeover verification (mutation-testing the three new repro tests) found that Tests/UI/test_master_shell_navigation.py::test_resize_does_not_strand_the_focused_button does not actually discriminate fixed vs reverted on_resize behavior: growing 80->90 cols in that scenario never produces a genuine straddle in the first place (strip.region never grows past its pre-resize width within the test's own timing budget, in this bare-widget test harness), so the test passes identically whether MainNavigationBar.on_resize routes through _recenter_strip or the old, focus-indifferent _scroll_active_destination_into_view. Reverting on_resize's wiring and re-running the test does NOT go red. A hand-built alternative scenario (same Tab-to-nav-settings setup at 80 cols, then SHRINKING to 70 instead of growing) does reproduce a genuine straddle, and does discriminate correctly at short windows (~0.1s post-resize: reverted code straddles immediately, fixed code does not). But probing further showed the FIXED code's correction is only transient: by ~0.28-0.3s after the resize (still under the periodic interval's 0.5s tick, so the interval cannot be masking it), the corrected position drifts back to the SAME straddling geometry as the reverted code, with _deliberate_focus_id, app.focused, and active_destination_id all unchanged throughout. Root cause not found in the time available -- the leading hypothesis is that Textual's pilot.resize_terminal (or the real terminal-resize path it emulates) fires on_resize more than once as layout settles, and a later firing's call_after_refresh(_recenter_strip) call somehow re-lands on the old, active-anchored scroll position despite the focus-conflict branch's own logic looking correct on inspection. This needs dedicated, unhurried investigation: instrument on_resize's own call count/timing directly (not just outcome geometry), and only then decide whether the fix needs a second corrective pass (e.g. re-affirm on a settle-debounce) or the test needs a longer, settle-tolerant assertion window with an explicit rationale for why that's still meaningful.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause of the ~0.3s drift-back is characterized with direct evidence (e.g. call counts/timestamps on on_resize and _recenter_strip), not just outcome geometry
- [ ] #2 test_resize_does_not_strand_the_focused_button (or a replacement) is a genuine, mutation-tested regression guard: it goes red when on_resize's _recenter_strip wiring is reverted and stays green with it restored, verified by an actual revert-and-rerun, not by inspection
- [ ] #3 If the drift indicates the on_resize fix is incomplete, the production fix is corrected; if the drift is a test-harness artifact unrelated to real terminal resizes, that is demonstrated and documented
<!-- AC:END -->
