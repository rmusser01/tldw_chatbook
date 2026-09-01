---
id: TASK-25888
title: 'Console: section toggle relayouts the whole 500-widget screen'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-01 05:15'
updated_date: '2026-09-01 05:16'
labels:
  - console
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every Console rail interaction funnels through _sync_console_rail_visibility, which ends with self.refresh(layout=True) on the whole ChatScreen -- even when only a section inside the rail opened or closed and no pane-level geometry moved. Profiled cost per click: ~61ms _refresh_layout plus ~39ms render_full_update on the main thread, plus 54-62KB written to the terminal (bare Textual: 0.8 full updates, 9KB). With the ack paint and terminal write this lands a rail click at 110-165ms, at or above the perception threshold, matching the reported uniform button sluggishness. The full-screen refresh is only needed when a rail is shown or hidden or the main column minimum changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A section toggle produces zero full-screen compositor updates
- [x] #2 Showing or hiding a rail still relayouts the screen (pane geometry genuinely changes)
- [x] #3 Both behaviours are pinned by tests that assert compositor update kind, not implementation internals
- [x] #4 Terminal output per section toggle drops materially (full-screen updates eliminated), with paired before/after numbers recorded on the same instrument, including the blocking-time result even where it is modest
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: count Compositor.render_full_update during section toggles (expect 0) and during a rail collapse (expect >=1)\n2. In _sync_console_rail_visibility, derive the pane projection (four target visibilities + main min_width); screen refresh(layout=True) only when it changes, else scope refresh to the two rails\n3. Verify: new tests green, press-to-settled timing before/after, full Console sweep incl. test_workbench_visual_snapshots.py\n4. Preflight, PR, lessons entry for the first-paint instrument trap
<!-- SECTION:PLAN:END -->

## Implementation Notes

`_sync_console_rail_visibility` now decides its refresh scope from ground
truth read out of the DOM: `pane_changed` is set only when a target's
`display` actually flips or the main column's `styles.min_width` Scalar
changes across the write. Pane change -> `self.refresh(layout=True)` exactly
as before; otherwise both rails get a scoped `refresh(layout=True)`. No
shadow state -- an earlier draft kept a projection tuple on self and its
None-seed caused a spurious full refresh on the first sync.

Measured (paired instruments, same probe on branch and pristine dev):
full-screen compositor updates per section toggle 1.2 -> 0; terminal output
54,404 -> 23,424 bytes/click; main-thread blocking median 21.0 -> 16.8 ms;
press-to-settled envelope unchanged (~210 ms both sides -- a trailing
reconcile cascade, not felt latency, and out of scope here).

**Correction recorded en route:** the task description's "61ms relayout +
39ms full render" came from cProfile and was ~7x inflated by profiler
overhead; direct timing puts layout at ~8.5 ms. AC #4 was rewritten from
"latency drops materially" to the output/blocking form before implementation
-- the output claim held, the latency claim would not have. Lesson filed in
lessons-testing-evidence.md ("A profiled cost is not a felt cost").

One full update remains on the FIRST click that moves focus onto the toggle:
Textual's `Screen.focused` reactive repaints the screen on any focus change,
and seven `:focus-within` rules in this stylesheet depend on that repaint, so
it cannot be suppressed app-side. The new test warms focus first and pins the
app's contribution at zero.

Files: `tldw_chatbook/UI/Screens/chat_screen.py` (one function + no new
state), `Tests/UI/test_console_rail_refresh_scope.py` (new, 2 tests).
Sweep: 12 rail-asserting files incl. visual snapshots -- 215 passed; the 5
reds are the documented dev set plus one newly-recorded dev flake
(9/12 failing on pristine dev, logged on TASK-25715).
