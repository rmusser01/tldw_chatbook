---
id: TASK-33065
title: Theme list highlight is visible without relying on bold
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2. The focused list's highlighted row differs from the list background by 1.10:1 (light) and 1.12:1 (dark); only bold text marks it among bold headers and markers. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The highlighted row's fill contrasts at least 3:1 with the list background under Textual Dark and Textual Light
- [x] #2 The contrast test measures the fill, not just bold
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Give the focused list's highlighted row a $block-cursor-background fill (theme token) and verify >=3:1 against the list background under Textual Dark/Light numerically.
2. Extend test_settings_theme_card_contrast.py to measure the fill.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
#settings-theme-list > .option-list--option-highlighted now inverts like a focused card chip ($text-primary fill, $ds-surface-panel text, bold). $block-cursor-background was tried first and measured 2.84:1 on solarized_light. Measured fill vs list: textual-dark 6.26/5.61 (unfocused/focused), textual-light 9.90/8.97, gruvbox_dark 8.10/7.07, solarized_light 5.90/5.57; name text on the fill 5.12-10.30. test_theme_picker_use_and_try_chips_meet_contrast measures the fill (>=3:1) and the text on it (>=4.5:1), focused and not, via a new _highlighted_row helper.
<!-- SECTION:NOTES:END -->
