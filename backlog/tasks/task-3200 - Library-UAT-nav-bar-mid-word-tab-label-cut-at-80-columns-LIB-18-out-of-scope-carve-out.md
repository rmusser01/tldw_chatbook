---
id: TASK-3200
title: >-
  Library UAT nav-bar mid-word tab-label cut at 80 columns (LIB-18 out-of-scope
  carve-out)
status: To Do
assignee: []
created_date: '2026-08-07 16:44'
labels:
  - library
  - ux
  - navigation
  - uat-2026-08-06
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 4 of the Library UAT P2 batch (task-2858) reproduced LIB-18's third finding live at 80x24: the shared MainNavigationBar's horizontally-scrolling destination strip clips the last partially-visible tab label mid-word (e.g. "Watchlists" -> "⌃6 Watc") right before the "More ›" overflow affordance, instead of hiding that partial button. This is shared app-wide chrome (tldw_chatbook/UI/Navigation/main_navigation.py), not Library-specific, and the fix is not a small one: the strip fills 1fr and lays out full-width buttons with CSS overflow-x: auto (main_navigation.py:104-113), so a button that only PARTIALLY fits at the viewport's right edge is visually clipped by the scroll container rather than being excluded -- the existing 'More ›' hint (main_navigation.py:226-237) already signals overflow exists but does not stop the strip's own clipping. A real fix needs the strip to measure whole-button widths and stop rendering (or scroll-clip) at a button boundary instead of relying on CSS overflow alone, touching shared navigation code exercised by every screen -- out of scope for the Library UAT P2 batch. Recorded per task-2858's Task 4 directive to record out-of-scope shared-chrome findings rather than force a fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The MainNavigationBar destination strip never clips a tab's label mid-word at narrow widths (e.g. 80 columns) -- either the partially-visible button is fully hidden until it can render whole, or its label degrades gracefully (e.g. abbreviates) without an ellipsis or hard cut inside a word
- [ ] #2 Fix is verified live at 80x24 (and spot-checked at 100/120) with a fresh tmux session, confirming no destination label is cut mid-word
- [ ] #3 A rendered-geometry or string-assertion test pins the fix (region widths or captured label text), matching the existing F-001 overflow-hint test coverage in Tests/UI/test_main_navigation*.py or equivalent
<!-- AC:END -->
