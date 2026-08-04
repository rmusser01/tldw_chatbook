---
id: TASK-2093
title: >-
  Tab bar: stop truncating destinations and make digit hints honest (F-001,
  F-002)
status: In Progress
assignee: []
created_date: '2026-08-03 17:25'
updated_date: '2026-08-04 12:22'
labels:
  - ux-review
  - chrome
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At <=100 cols tabs collapse ('8 Workflows' -> '8') and later destinations become unreachable; labels '1 Home' imply bare-digit keys but the binding is Ctrl+digit. Evidence: library/roleplay/mcp-100x30.png, app.py:3493. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All destinations remain reachable at 100 cols (ellipsis/overflow/scroll),Digit affordance labels match the actual keybinding,Rendered-layout test at 100 cols
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (chrome labels + one affordance control; no behavior/route changes). Root cause found: the bar never squeezes labels -- the strip clips mid-button at its scroll window ('8 Workflows' shows as '8' at 100 cols), destinations past the fold have no click path, and the 'More: Ctrl+P' Static is unconditional chrome (its own compose comment says 'exactly when the destinations overflow'). No glyph convention exists elsewhere in the app. Steps: 1. RED tests: (a) nav_button_label renders the ctrl glyph ('⌃1 Home' ... '⌃0 ACP', unnumbered Lab/Logs/Settings) -- updates to test_master_shell_navigation.py's unit pin and both exact-label list pins (test_master_shell_navigation_order_and_labels, test_main_navigation_copy_and_order); (b) 100x30 rendered test: overflow hint displays, pressing it pages the strip right until nav-settings is visible inside the strip window, pressing again wraps to nav-home; (c) hint hides when everything fits (160 cols); (d) visual_audit's shared hint assertion updated to the conditional contract; 'More: Ctrl+P' text assertions -> 'More ›' where the label is collected regardless of display. 2. main_navigation.py: glyph in nav_button_label; hint becomes a compact Button ('More ›') with a page-right/wrap press handler; display synced to strip.max_scroll_x on mount + resize; DEFAULT_CSS for the quiet button look. 3. Run nav/shell/audit/replay suites + ruff.
<!-- SECTION:PLAN:END -->
