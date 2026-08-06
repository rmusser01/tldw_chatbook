---
id: TASK-1370
title: Fix top nav tab bar clipping at 120 cols
status: Done
assignee: []
created_date: '2026-08-05 21:11'
updated_date: '2026-08-05 22:39'
labels:
  - navigation
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT evidence (assessB5-settings-landing-120x35.txt) shows the app-level top nav clips 'Schedules' to 'hedules' at 120 cols with a scroll arrow. App-level navigation issue, not Settings-specific.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All top nav destinations render without mid-word clipping at 120 cols
- [x] #2 Overflow behavior at narrower widths is explicit (scroll hint or dropdown) rather than silently cutting a label
- [x] #3 Regression test covers the nav bar at 120 and 100 cols
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: headless Pilot test rendering MainNavigationBar at 120x35 and 100x30, assert no displayed nav button is partially clipped (geometric check vs strip region)
2. Root cause: horizontal scroll strip clips at arbitrary offsets; scroll_to_widget only guarantees the active button, leaving neighbors cut mid-word at both edges
3. Fix: replace silent scroll-clipping with explicit overflow — buttons that do not fully fit get display=False (whole or hidden), active destination always kept visible, "More: Ctrl+P" hint shown whenever any destination is hidden
4. Update scroll-contract tests (test_chrome_ux_fixes.py, test_master_shell_navigation.py, test_destination_visual_parity_correction.py) to the visibility-based contract
5. Run new test + Tests/UI/test_screen_navigation.py + nav/settings subsets
ADR required: no
ADR path: N/A
Reason: layout fix within existing nav widget; no architectural boundary changes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Follow-up (spec review): fixed a stale-cache residual — nav buttons grow +2 cells when gaining is-active/:focus (the state border appears on otherwise borderless buttons), so a tightly packed bar (budget − used ∈ {0,1}) overflow-clipped the rightmost visible label on focus/activation ("^6 Watchlists" -> "^6 Watchlist" at 91 cols) and the 0.5s interval could not self-correct. Fix: cached widths are now state-normalized at measurement (mount-time active/focused border subtracted via _STATE_BORDER_CELLS=2) and packing reserves _STATE_GROWTH_SLACK=4 (worst case: active and focus on two different buttons) in both the all-fits check and the overflow budget. Regression test test_nav_state_growth_does_not_clip_at_tight_packing[focus/activate] computes the tight width from measured widths at runtime (91 cols today) and asserts no visible label clips after focusing or pressing the rightmost shown button. Verification: pytest Tests/UI/test_nav_overflow_clipping.py Tests/UI/test_chrome_ux_fixes.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_screen_navigation.py — 91 passed; ruff clean; 91-col visual probe with focused buttons shows no clipping.
<!-- SECTION:NOTES:END -->
