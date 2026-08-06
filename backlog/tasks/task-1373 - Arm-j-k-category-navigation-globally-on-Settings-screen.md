---
id: TASK-1373
title: Arm j/k category navigation globally on Settings screen
status: Done
assignee: []
created_date: '2026-08-05 23:38'
updated_date: '2026-08-05 23:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique power-user red flag: j/k category navigation only arms after a category button has focus. A power user landing on the screen or coming out of a field cannot immediately j/k between categories. Arm the bindings at screen level so j/k moves category focus from anywhere sensible (while never stealing keys from text inputs).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 j/k moves between category buttons without first focusing a category button
- [x] #2 j/k never fires while a text input/Select has focus (printable keys stay typed)
- [x] #3 Regression test drives real pilot.press from a non-category focus
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-scope j/k in SettingsScreen.on_key: keep arrows category-only, arm j/k whenever focus is not Input/TextArea/Select and no Select overlay is expanded (overlay type-searches on printable keys)
2. Add helper guard on SettingsScreen
3. Pilot tests: j/k from non-category focus moves category focus; j/k in search Input types text; j/k on Select does not move focus
ADR required: no
ADR path: N/A
Reason: routine UX fix, no architectural decision
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Re-scoped j/k in `SettingsScreen.on_key` (`tldw_chatbook/UI/Screens/settings_screen.py`): the existing branch only fired when a category button already had focus; it now also fires when no category button is focused, gated by a new `_jk_category_navigation_blocked()` guard. Arrow-key behavior is unchanged (rail-only).
- The guard returns True when the focused widget is an `Input`/`TextArea`/`Select`, or when ANY `Select` in the screen has an expanded overlay — the overlay mounts on the screen (not under its Select) and type-searches on printable keys, so `select.expanded` is checked directly. Input/TextArea consume printable keys before screen `on_key` fires, so the isinstance check is belt-and-braces.
- `_move_category_focus` already fell back to the active category when no rail button is focused, so no change was needed there.
- Tests added in `Tests/UI/test_settings_configuration_hub.py`: `test_settings_jk_category_navigation_arms_from_non_category_focus` (real pilot.press from a focused detail-pane button), `test_settings_jk_never_steals_keys_from_text_input` (j/k stay typed in the search Input), `test_settings_jk_never_steals_keys_from_select` (closed Select and open overlay).
- Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `Tests/UI/test_settings_configuration_hub.py`.
<!-- SECTION:NOTES:END -->
