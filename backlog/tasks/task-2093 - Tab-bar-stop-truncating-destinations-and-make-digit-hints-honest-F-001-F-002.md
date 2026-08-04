---
id: TASK-2093
title: >-
  Tab bar: stop truncating destinations and make digit hints honest (F-001,
  F-002)
status: Done
assignee: []
created_date: '2026-08-03 17:25'
updated_date: '2026-08-04 12:48'
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
- [x] #1 All destinations remain reachable at 100 cols (ellipsis/overflow/scroll),Digit affordance labels match the actual keybinding,Rendered-layout test at 100 cols
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (chrome labels + one affordance control; no behavior/route changes). Root cause found: the bar never squeezes labels -- the strip clips mid-button at its scroll window ('8 Workflows' shows as '8' at 100 cols), destinations past the fold have no click path, and the 'More: Ctrl+P' Static is unconditional chrome (its own compose comment says 'exactly when the destinations overflow'). No glyph convention exists elsewhere in the app. Steps: 1. RED tests: (a) nav_button_label renders the ctrl glyph ('⌃1 Home' ... '⌃0 ACP', unnumbered Lab/Logs/Settings) -- updates to test_master_shell_navigation.py's unit pin and both exact-label list pins (test_master_shell_navigation_order_and_labels, test_main_navigation_copy_and_order); (b) 100x30 rendered test: overflow hint displays, pressing it pages the strip right until nav-settings is visible inside the strip window, pressing again wraps to nav-home; (c) hint hides when everything fits (160 cols); (d) visual_audit's shared hint assertion updated to the conditional contract; 'More: Ctrl+P' text assertions -> 'More ›' where the label is collected regardless of display. 2. main_navigation.py: glyph in nav_button_label; hint becomes a compact Button ('More ›') with a page-right/wrap press handler; display synced to strip.max_scroll_x on mount + resize; DEFAULT_CSS for the quiet button look. 3. Run nav/shell/audit/replay suites + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause found by probe: the bar never squeezes labels -- the strip clips mid-button at its scroll window ('8 Workflows' renders as '8' at 100 cols) and destinations past the fold had no click path; the 'More: Ctrl+P' Static was unconditional despite its compose comment documenting conditional intent. Fix: (a) the hint is now a compact 'More ›' Button that pages the strip right and wraps at the far end (click + keyboard-focusable path to every destination), displayed exactly when strip.max_scroll_x > 0 (synced on mount + resize, starts hidden so wide bars never flash it); (b) nav_button_label uses the control glyph ('⌃1 Home' ... '⌃0 ACP') so the affordance matches the actual ctrl+digit binding at zero extra width per tab. One subtlety handled: the pager must clamp to max_scroll_x and wrap only when already at the end (first version wrapped immediately whenever a page exceeded the overflow). Files: tldw_chatbook/UI/Navigation/main_navigation.py; tests updated in test_master_shell_navigation.py (unit pin, label lists, new 100x30 pager test), test_screen_navigation.py (copy_and_order pin + conditional contract), test_product_maturity_phase1_visual_audit.py (shared helper polls the conditional display; nav-button queries scoped to Button.nav-button since the hint is a Button now), test_product_maturity_phase1_empty_setup_states.py, unified_shell phase6 replays (text + query updates), test_product_maturity_phase6_first_time_release_replay.py (label pins). Verified: 4 initial REDs fixed to green; nav files 96 passed; full affected suite (nav + audit + 5 replay/empty files) 111 passed; visual audit across the 3-size matrix green; ruff clean on all touched files. Live 100x30 capture shows '⌃1 Home … ⌃7 Schedules More ›'. Deferral: Lab/Logs/Settings still carry no hotkey (labels honestly unnumbered; palette/pager cover them) -- adding bindings would be new scope. ADR: not required (chrome labels + one affordance control; routes/behavior unchanged). Commit f529c8be8.
<!-- SECTION:NOTES:END -->
