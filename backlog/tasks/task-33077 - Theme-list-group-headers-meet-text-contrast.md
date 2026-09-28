---
id: TASK-33077
title: Theme list group headers meet text contrast
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P3. Group headers render as disabled options at 2.54:1 (light) and 3.43:1 (dark). Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Group headers reach at least 4.5:1 under Textual Dark and Textual Light
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Style the list's disabled options (group headers and placeholders) with a readable muted token and bold instead of Textual's $text-disabled.
2. Rebuild the CSS bundle.
3. Test: measure header glyph contrast from compositor segments under Textual Dark and Light, >= 4.5:1.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
#settings-theme-list > .option-list--option-disabled is now $ds-text-muted, bold (was Textual's $text-disabled). This covers the group headers and the loading/none-yet placeholder rows, which are labels, not unavailable controls. Measured from compositor segments: Textual Dark 3.52 -> 6.77:1, Textual Light 2.59 -> 5.15:1. Test: test_theme_list_group_headers_meet_text_contrast (Tests/UI/test_settings_theme_card_contrast.py). Files: css/components/_settings_splash_theme.tcss, rebuilt tldw_cli_modular.tcss, settings.md.

P3 review fix (M2): the contrast test still looked for 'TEXTUAL' (renamed BUILT-IN by TASK-33073) and only `assert seen`, so BUILT-IN — below the fold — was never measured. It now takes the titles from the picker's _GROUP_TITLES, scrolls a screenful at a time, and asserts every header was measured (old title list fails: 'headers never seen: TEXTUAL').
<!-- SECTION:NOTES:END -->
