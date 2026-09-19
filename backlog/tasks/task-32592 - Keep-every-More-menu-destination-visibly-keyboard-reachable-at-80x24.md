---
id: TASK-32592
title: Keep every More-menu destination visibly keyboard reachable at 80x24
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 00:18'
updated_date: '2026-09-15 00:53'
labels:
  - design-system
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-component-first-ui-audit.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 80x24 the More menu clips Research and Meetings. Keyboard focus can reach Meetings without painting its label or focus cue, so users cannot see which destination they will activate. The defect reproduces in both themes and exists on current dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every registered destination can be visibly focused and activated through More at 80x24 in both textual-dark and textual-light.
- [x] #2 Moving focus to an initially offscreen destination brings its complete label and focus indicator into view.
- [x] #3 Opening, traversing and dismissing More preserves correct route selection and returns focus predictably without activating a different destination.
- [x] #4 The same menu remains bounded and usable through a 120x40 to 80x24 to 120x40 resize sequence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md (existing); backlog/decisions/161-component-pattern-library.md (existing)
Reason: repair visible keyboard reachability within the existing overflow-menu pattern, preserving routes and visual values.

1. Add a production-styled dark/light regression that traverses and activates every destination at 80x24, verifies painted labels/focus and checks the same menu through 120-to-80-to-120 resizing.
2. Replace the clipped menu body with a standard vertical scroll container while retaining labels, routing and Escape/return-focus behavior.
3. Run the targeted menu/navigation tests, inspect bounded before/after captures, and record completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the clipped More body with a standard VerticalScroll that does not take focus itself. Existing dimensions, labels, routes and styling remain in place; focused destinations scroll into view, including after viewport resizing. The new regression reproduced invisible Research/Meetings focus before the fix, then found and covered the separate resize case.

Verification:52 targeted navigation tests pass, including every destination activated at80x24 in dark/light and a same-menu120-to80-to120 resize. Native terminal Shift+Tab now visibly reveals Research and Meetings at80x24; evidence is Docs/superpowers/qa/2026-09-14-component-fixes/more-last-80.ansi. Fatal Ruff and formatter checks pass on the changed code. No full suite run.

ADR check:no new ADR; direct repair under ADR150 focus and ADR161 composition contracts. Modified nav_overflow_menu.py; added test_nav_overflow_layout.py.
<!-- SECTION:NOTES:END -->
