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

Known gap (not in the ACs): Try, then the editor's Reset (edits cleared), then Back leaves no prompt and keeps the tried palette.

Files: Widgets/settings_theme_editor.py, UI/Screens/settings_screen.py, Tests/UI/test_settings_theme_picker_screen.py (6 Pilot tests: Back, Esc, category switch, navigation, quit, Try→Save), Docs/User_Guide/settings.md.
<!-- SECTION:NOTES:END -->
