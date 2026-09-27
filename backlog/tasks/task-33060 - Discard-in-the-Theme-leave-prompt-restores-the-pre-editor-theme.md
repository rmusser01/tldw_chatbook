---
id: TASK-33060
title: Discard in the Theme leave prompt restores the pre-editor theme
status: Done
assignee:
  - '@claude'
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P1 (2026-09-27, 28/40). After the editor's Try, choosing Discard on Back, a category switch or any leave prompt leaves an unlisted custom_<name> theme applied app-wide. The picker then marks no theme (or the wrong one) active and Revert cannot undo it. Cause: the editor's Try sets app.theme directly (settings_theme_editor.py on_apply_theme) and no discard path restores it. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Try then Discard (Back, category switch, Escape, quit/navigation prompt), the app runs the theme that was active before the editor opened
- [x] #2 The picker marks exactly one theme active after a discard, matching app.theme
- [x] #3 Try then Save keeps the saved theme applied (no regression)
- [x] #4 A Pilot test pins Try → Discard → original theme for each leave path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Editor Try records the theme that was active before the first Try of this editing session (and the theme it applied); a new editing session (the pane opening the editor) forgets it, and Save forgets it (the saved theme stays applied).
2. Add one editor method that restores the pre-Try theme, only while the app still runs the tried theme.
3. Call it from every Discard branch: Back/Escape (_confirm_theme_back), category switch (_confirm_theme_category_leave), screen navigation and quit (confirm_navigation, which confirm_quit delegates to).
4. The picker's app-level pending Revert is left untouched: an editor Try never entered it, so after a Discard it still points at the same target.
5. Pilot tests: Try -> Discard via Back, Escape, category switch, navigation and quit each end on the original theme with exactly one active picker row; Try -> Save keeps the saved theme.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Editor Try now records `(theme before this session's first Try, theme Try applied)`; `discard_try()` restores the first while the app still runs the second (a no-op otherwise, so a theme chosen elsewhere since is never clobbered). `set_editing_context` (called once per editor open) starts a fresh session; a successful Save forgets the undo so the saved theme stays applied. Every Discard calls it: `_confirm_theme_back` (Back and Esc), `_confirm_theme_category_leave`, and `confirm_navigation` (which `confirm_quit` delegates to).

Revert interaction: the editor's Try never entered the picker's app-level `theme_revert_change`, and Discard does not touch it either, so a pending Revert keeps its target and Discard is an exact undo of the editor Try alone.

Files: Widgets/settings_theme_editor.py, UI/Screens/settings_screen.py, Tests/UI/test_settings_theme_picker_screen.py (6 Pilot tests: Back, Esc, category switch, navigation, quit, Try→Save), Docs/User_Guide/settings.md.

Fix round after the whole-branch review:
- I-1 (prompt-free exits kept the tried custom_* palette; the "Reset then Back" gap above was one case). Rule: a Try never outlives its editor session unless saved -- every non-Save exit restores the pre-editor theme, with no extra prompt (Try is a preview, not an edit; asking Save/Discard about a preview the user never changed would be noise, and the exit already says "I'm done here"). Applied at two choke points every exit passes through: `ThemePane.show_picker` (Back/Esc, prompted or not) and `SettingsThemeEditor.on_unmount` (category switch, leaving Settings). The per-caller calls in the Back and category-leave Discard branches were dropped; `confirm_navigation`'s Discard keeps its call (the quit path does not unmount before exit). The picker's `_switch` now never records a `custom_*` theme as the Revert target: it maps to the listed theme behind it, else the launch default.
- I-2 (AC#3 for Save as / renamed Save): `_write_theme_file` passes the running Try's registration to `_reapply_if_active(tried=...)`, so the saved theme is applied under its saved name before the undo record is cleared.
- M-1: a second Try keeps the recorded "before" theme only while the app still runs the last Try; a theme chosen elsewhere in between (e.g. the palette) becomes what Discard restores.
- M-3: `discard_try`'s failure toast runs the theme name and error through `printable()` as well as `escape_markup()`.
Tests (each failed before the fix): test_try_then_clean_back_restores_the_pre_editor_theme[c,n], test_try_then_reset_then_back_restores_the_pre_editor_theme, test_try_then_clean_category_switch_restores_the_pre_editor_theme, test_try_then_clean_navigation_restores_when_the_editor_goes, test_revert_after_try_back_and_picker_try_targets_the_listed_theme, test_picker_revert_never_targets_an_unlisted_custom_theme, test_try_then_save_under_a_new_name_applies_the_saved_theme[save_as,rename_then_save], test_discard_keeps_a_theme_chosen_elsewhere_between_tries, test_discard_failure_toast_prints_the_theme_name_safely. Also Widgets/settings_theme_picker.py.
<!-- SECTION:NOTES:END -->
