---
id: TASK-2835
title: Make tooltips visible to keyboard users on Settings screen
status: Done
assignee: []
created_date: '2026-08-05 23:38'
updated_date: '2026-08-05 23:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique: tooltips everywhere but hover-only, invisible to keyboard users. Dotted module paths moved into tooltips in round 2, which made this worse — keyboard-only users now cannot reach that information at all. Show tooltip-equivalent content on focus (Textual tooltip-on-focus or a focus-status line).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focusing a control with a tooltip surfaces its tooltip content without the mouse
- [x] #2 Behavior verified by pilot keyboard test
- [x] #3 No duplicate/cluttered display for mouse users
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Textual 8.2.7 has no tooltip-on-focus; add #settings-focus-help Static at the bottom of #settings-shell
2. Update it from the focused widget's tooltip in the existing @on(DescendantFocus) handler; clear when no tooltip
3. CSS rule in components/_agentic_terminal.tcss + regenerate bundle via build_css.py
4. Pilot keyboard test: Tab to a tooltip-bearing control shows its text; focusing a tooltip-less control clears
ADR required: no
ADR path: N/A
Reason: routine UX fix, no architectural decision
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- The project pins Textual >=8.0,<9 (8.2.7 installed); tooltips render on hover only, there is no tooltip-on-focus support, so the status-line pattern was used.
- Added a one-line `#settings-focus-help` Static at the bottom of `#settings-shell` (`compose_content`), updated from `event.widget.tooltip` in the existing `@on(DescendantFocus)` handler via a new `_update_focus_help()` helper. The line clears when the focused widget has no tooltip, so it never goes stale.
- Mouse users see no duplicate display: the line only changes on focus events, never on hover, and the native hover tooltip is untouched.
- CSS: `#settings-focus-help` rule added to `tldw_chatbook/css/components/_agentic_terminal.tcss` (next to the other settings rules); `tldw_cli_modular.tcss` regenerated with `build_css.py` (bundle is a generated file).
- Test added: `test_settings_focus_surfaces_tooltip_in_focus_help_line` — pure keyboard flow (Tab -> category button shows its description; j -> next button shows its tooltip; focusing the tooltip-less search Input clears the line).
- Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (generated), `Tests/UI/test_settings_configuration_hub.py`.
<!-- SECTION:NOTES:END -->
