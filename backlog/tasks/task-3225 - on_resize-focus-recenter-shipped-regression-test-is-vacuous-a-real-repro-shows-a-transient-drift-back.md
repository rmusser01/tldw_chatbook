---
id: TASK-3225
title: >-
  on_resize focus-recenter: shipped regression test is vacuous; a real repro
  shows a transient drift-back
status: Done
assignee: []
created_date: '2026-08-09 14:34'
updated_date: '2026-08-09 15:17'
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
- [x] #1 Root cause of the ~0.3s drift-back is characterized with direct evidence (e.g. call counts/timestamps on on_resize and _recenter_strip), not just outcome geometry
- [x] #2 test_resize_does_not_strand_the_focused_button (or a replacement) is a genuine, mutation-tested regression guard: it goes red when on_resize's _recenter_strip wiring is reverted and stays green with it restored, verified by an actual revert-and-rerun, not by inspection
- [x] #3 If the drift indicates the on_resize fix is incomplete, the production fix is corrected; if the drift is a test-harness artifact unrelated to real terminal resizes, that is demonstrated and documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root-caused with direct instrumentation, fixed at the source, and re-tested with a mutation-verified guard (task-3200 review round 4).

ROOT CAUSE (AC #1): not a second on_resize firing (round 3's hypothesis -- on_resize fires exactly once; confirmed by enter/exit logging). The ghost rule in MainNavigationBar.DEFAULT_CSS declared 'border: solid $background !important'. Textual's Button.-style-default gives a nav button border-top/border-bottom 'tall' and ZERO horizontal border cells, so a four-edge solid border made a ghosted button 2 cells WIDER (measured: #nav-workflows 14 -> 16 by toggling the class in isolation). In the strip's horizontal layout that reflowed every later button 2 cells right, moving a previously fully-visible button into a straddle AFTER the corrective scroll had landed -- and nothing re-checks after a ghost pass, because ghosting is supposed to be geometry-neutral (that is the whole reason it was chosen over display:none). Timeline evidence: scroll lands ~40ms with nav-settings x=45; ghost pass ghosts nav-workflows (x=-2, straddling); one layout pass later nav-workflows is w16 and nav-settings is x=47, straddling.

FIX (AC #3): the ghost rule now declares no box-model property at all -- colors/text-style only -- so the ghosted box is identical to the un-ghosted box in whichever CSS tier wins. 'visibility: hidden' was tried first and rejected with evidence: Widget.region returns an EMPTY region for an invisible widget (outer_size stays 14, region.width -> 0) and _ghost_clipped_buttons skips region.width <= 0, so a ghosted button could never be un-ghosted.

SCOPE, honestly: the drift-back was a bare-widget-harness-only defect. Under the real bundled stylesheet the ghost's border declaration never applied -- components/_buttons.tcss's 'Button { border: none; }' is in the CSS_PATH tier, which outranks widget DEFAULT_CSS regardless of !important (measured: ghosted width 14 both before and after the fix with CSS_PATH loaded). No shipped user saw it. Fixed anyway because the design's core invariant is geometry-neutrality, the harness is the whole deterministic suite for this feature, and the rule survived only by accident of an unrelated app-wide rule.

TEST (AC #2): test_resize_does_not_strand_the_focused_button rewritten. The shipped version was vacuous twice over -- growing 80->90 with active=schedules never straddles, AND two mechanisms heal any strand before a wall-clock assertion (the focus-aware interval, and _ghost_clipped_buttons's best-effort scroll_to_widget(focused) nudge, which fires off a stale region; traced scroll_x 86 -> 75 -> 96 in 40ms with on_resize reverted). The rewrite uses active=home so a recenter-on-active drags the focused button FULLY off-screen (the case the nudge cannot rescue), suppresses the interval backstop for the resize via a test-local subclass, and asserts full visibility at 8 checkpoints over 0.8s. Mutation-tested: reverting on_resize to _scroll_active_destination_into_view fails 3/3 (Region(x=141) vs strip Region(x=3,width=77)); restored passes 3/3. New test_ghosting_a_button_never_reflows_the_strip pins the root cause with no timing at all; restoring the border declaration fails it 3/3 with the exact +2 reflow.

Tests: test_master_shell_navigation.py 32/32 x5; phase6 5/5 x3; 13-file nav sweep 330 passed / 5 pre-existing failed / 1 skipped. Live tmux 80->90: focused non-active button fully visible with focus ring at +0.3/+1.2/+2.5/+6.5s and after shrinking back; both straddlers pixel-exact invisible.

Files: tldw_chatbook/UI/Navigation/main_navigation.py, Tests/UI/test_master_shell_navigation.py.
<!-- SECTION:NOTES:END -->
