---
id: TASK-33069
title: Theme filter no-match state has a Clear chip and sane fallback
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
Critique #3 P3. With no matches the spec §9 Clear filter chip is missing, the preview keeps the last theme under a blank title, and clearing the filter highlights row 1 (possibly an unreadable file) instead of the active theme. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A no-match filter shows a Clear filter control and no stale preview
- [x] #2 Clearing the filter highlights the previously highlighted theme or the active theme
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a Clear filter button under the no-match message, shown only with the message.
2. Hide the preview when nothing is highlighted.
3. Remember the last highlighted theme; _render_list falls back to it, then to app.theme, before row 1.
4. Tests: no-match shows the button and no preview; clearing (button or backspace) re-highlights the previous theme, or the active one.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a Clear filter button (own row, shown only with the no-match message; the button's own display is toggled too so it isn't keyboard-reachable while hidden). The preview hides when nothing is highlighted (shared with 33067), so the card is empty rather than stale. The picker records the highlight from before a filter was typed (_prefilter_highlight); _render_list falls back to it, then to app.theme, before row 1 -- this also makes any re-render that loses the highlight land on the active theme. Clear filter re-renders, clears the filter and focuses the list. Tests: test_no_match_filter_offers_clear_and_restores_the_previous_highlight, test_clearing_a_no_match_filter_by_hand_falls_back_to_the_active_theme. Files: settings_theme_picker.py, settings.md.
<!-- SECTION:NOTES:END -->
