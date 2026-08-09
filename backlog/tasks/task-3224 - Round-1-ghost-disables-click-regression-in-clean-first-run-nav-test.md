---
id: TASK-3224
title: Round-1 ghost-disables-click regression in clean-first-run nav test
status: To Do
assignee: []
created_date: '2026-08-09 14:20'
updated_date: '2026-08-09 14:20'
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
- [ ] #1 test_clean_first_run_launches_home_and_exposes_setup_orientation passes reliably (no timeout) regardless of nav-strip scroll/ghost state when exercising nav-console/nav-library/nav-settings
- [ ] #2 Root cause documented: whether the fix is test-side (page/scroll before a direct press) or production-side (reveal-before-press for non-mouse-targeted press paths), with the trade-off recorded
- [ ] #3 No regression to the round-1/round-2/round-3 ghosting contract (a genuinely-clipped, unrevealed nav button must still read disabled=True to a real mouse click)
<!-- AC:END -->
