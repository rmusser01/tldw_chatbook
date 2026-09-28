---
id: TASK-33122
title: Theme list keeps launch and override markers discoverable at 80x24
status: Done
assignee: []
created_date: '2026-09-28 00:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from the P3 wave (TASK-33074 fix): to keep each row on one line at 80x24, the row drops the overrides marker and, when needed, the launch marker; the preview card title does not show them either, so at that size a user cannot see that a theme is the launch default or overrides another.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 80x24 the highlighted theme's launch and override status is visible somewhere on screen
- [x] #2 Rows still stay on one line
<!-- AC:END -->

## Implementation Plan

1. Probe: at 80x24 the card sits stacked under the list and is scrolled off (visible height 0), so a card-only status line would not be on screen; the list's own bottom border is.
2. When the highlighted row had to drop its launch/overrides markers, spell them out (full words) under the list; blank otherwise, so wide sizes (nothing dropped) show nothing extra.
3. Recompute on highlight and on refit (resize).
4. Tests: rendered-screen check at 80x24/211x44/235x52 with a saved theme that overrides a built-in and is the launch default; rows stay one line.
5. Docs/User_Guide/settings.md Theme section.

## Implementation Notes

- A reserved one-line `#settings-theme-row-status` Static under the theme list names whatever launch/overrides markers the highlighted row had to shorten or drop ("launch default · overrides built-in"); blank when the row shows everything. `_dropped_status(entry, width)` compares the fitted `_row` with the full marker tail, so rows keep their TASK-33074 one-line fitting untouched. Refreshed on highlight (`_show`) and on refit (resize).
- Deviation from the suggested card fix: at 80x24 the card is stacked under the list and scrolled fully out of view (visible height 0), so a card line would not be on screen. The list's border subtitle was tried first and rejected: the focus outline (`outline: solid`) paints over it, so it is invisible exactly while the user browses the list.
- The line is reserved (height 1) rather than toggled so the list never jumps as the highlight moves; at 80x24 the list keeps its 9 rows (the pane grows) and the line sits at y=20, inside the visible region.
- Verified in-process (Settings harness, saved "textual-dark" overriding the built-in and set as launch default): 80x24 row "Textual D…  ▮▮▮  active · launch", line "launch default · overrides built-in"; 211x44 row "… active · launch · overrides", line spelled out (the row shortened "overrides built-in" there too); 235x52 row shows all markers, line blank. Branch fix/theme-tail-lane-c off dev 89dd84943a, 2026-09-27.
- Files: tldw_chatbook/Widgets/settings_theme_picker.py, css/components/_settings_splash_theme.tcss (+ bundle), Tests/UI/test_settings_theme_picker.py, Tests/UI/test_settings_theme_picker_screen.py, Docs/User_Guide/settings.md.
