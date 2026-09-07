---
id: TASK-3224
title: Round-1 ghost-disables-click regression in clean-first-run nav test
status: Done
assignee: []
created_date: '2026-08-09 14:20'
updated_date: '2026-08-09 15:17'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Round 1 of task-3200 (commit 071a6c403) made genuinely clip-ghosted nav-strip buttons disabled=True (Button.press() becomes a no-op). This collides with the pre-existing Tests/UI/test_product_maturity_phase1_first_run.py::test_clean_first_run_launches_home_and_exposes_setup_orientation, which presses nav-console/nav-library/nav-settings by ID at 140 columns without first paging/scrolling. At that width nav-settings can be genuinely clip-ghosted (straddling the strip's right edge) when reached this way, so the direct press silently no-ops and the awaited screen transition times out at 10s. A/B-bisected during task-3200 round 3 takeover: base 451d95340 passes reliably (~2.5s); round 1 (071a6c403) onward fails, reproducing at round 2 HEAD (755a8b5e5) and unchanged by round 3's uncommitted diff (confirmed byte-identical failure with round-3 content swapped for round-2 content). Not caused by round 3 and out of its scope to fix. Timing-dependent, not 100% deterministic (roughly 4/5 in a local sample) because it depends on how much strip-settle has completed before the direct press, which is itself a hint the fix belongs in either the test (page/scroll to a destination before pressing it, matching what a real mouse click requires) or in production code (auto-reveal the target before honoring a press that isn't gated by real screen-position mouse targeting).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_clean_first_run_launches_home_and_exposes_setup_orientation passes reliably (no timeout) regardless of nav-strip scroll/ghost state when exercising nav-console/nav-library/nav-settings
- [x] #2 Root cause documented: whether the fix is test-side (page/scroll before a direct press) or production-side (reveal-before-press for non-mouse-targeted press paths), with the trade-off recorded
- [x] #3 No regression to the round-1/round-2/round-3 ghosting contract (a genuinely-clipped, unrevealed nav button must still read disabled=True to a real mouse click)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed test-side (task-3200 review round 4).

REPRODUCED at HEAD before touching anything: 4 failures / 5 runs, each a 13s timeout (baseline pass is ~3.3s).

ROOT CAUSE / DIRECTION (AC #2): test-side, not production-side. The test pressed #nav-settings by id at 140 columns with the strip still at its default scroll position. Round 1 of task-3200 made a genuinely clip-ghosted nav button disabled=True precisely because a real mouse click can never land on a button painted as blank space, and Enter on an invisible button was the defect being closed -- so the direct press silently no-opped and the awaited screen transition never came. The contract this test protects is 'every one of these destinations is reachable from the nav bar and renders its copy', NOT 'a programmatic press works on an off-screen widget'; the press-by-id was incidental mechanism. Trade-off recorded: fixing it production-side (auto-reveal before honoring a non-mouse-targeted press) would have re-opened the exact hole round 1 closed, since a press path that reveals-then-navigates is indistinguishable from Enter on an invisible button.

FIX: new _click_nav_destination helper reveals the target with the product's own affordance (the 'More >' pager, shown exactly when destinations overflow) and only presses once the button is genuinely fully inside the strip's viewport and enabled. The retry loop is load-bearing, not padding: MainNavigationBar's 0.5s settle interval re-anchors the strip on the ACTIVE destination, so a paged-in destination can scroll back out before the press lands.

AC #3 (no regression to the ghosting contract): production code untouched for this item; a genuinely-clipped, unrevealed nav button still reads disabled=True. test_master_shell_navigation.py's ghost/disabled tests (including test_tab_cycling_never_focuses_a_ghosted_nav_button and test_press_on_a_ghosted_nav_button_is_a_no_op) all still pass, 32/32 x5.

AC #1: 5/5 passing at ~3.3s. Instrumented once to confirm the reveal branch actually fires (exactly one page per run) -- the helper is load-bearing, not a no-op wrapper. 13-file nav sweep is now 330 passed / 5 failed (the 5 long-known schedules/MCP failures) / 1 skipped -- the 6th failure this task tracked is gone.

Files: Tests/UI/test_product_maturity_phase1_first_run.py.
<!-- SECTION:NOTES:END -->
