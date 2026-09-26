---
id: TASK-32941
title: Leaving Theme with unsaved edits asks Save / Discard / Stay
status: Done
created_date: 2026-09-24 14:45
assignee:
- '@claude'
labels:
- settings
- theme
- data-loss
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Choosing another Settings category while the theme editor had unsaved edits threw the edits away without a word: the dirty marker was cleared and the editor was remounted from scratch. Speech & TTS already asks before a dirty draft is abandoned; Theme should protect the user's work the same way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Leaving Theme with unsaved edits asks Stay / Discard / Save before the category changes
- [x] #2 Stay (or Escape) keeps the Theme category and the edit
- [x] #3 Discard leaves and the unsaved markers are cleared on the next visit
- [x] #4 Save writes the theme file and then leaves; a Save that is refused or needs overwrite confirmation keeps the user on Theme
- [x] #5 Opening the editor still posts no modified-status message
<!-- AC:END -->

## Implementation Plan

1. Rewrite the hub test that pinned the silent drop into Stay / Discard / Save tests.
2. Add a `ThemeLeaveModal` mirroring `_GlobalSpeechTTSLeaveModal`.
3. In `SettingsScreen._select_category`, intercept leaving a dirty Theme and resolve the choice in a worker, with the same in-progress/bypass flags as the Speech guard.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`ThemeLeaveModal` (in `settings_theme_editor.py`, reusing the `settings-rag-profile-modal` frame so no CSS changed) returns save/discard/cancel. `SettingsScreen._select_category` routes a dirty Theme exit through `_confirm_theme_category_leave` (worker group `settings-theme-category-leave`, `_theme_leave_in_progress` / `_theme_leave_bypass`, same shape as the Speech guard). Save calls the editor's own `on_save_theme()` and leaves only if the editor is clean afterwards, so a refused name or a pending overwrite dialog keeps the user on Theme. The existing TASK-31252 dirty-flag clear still runs on the resolved exit. The `is_modified = reactive(False, init=False)` decision is untouched.

Scope: only category changes inside Settings are guarded; leaving the Settings screen itself is not.

Tests: `Tests/UI/test_settings_configuration_hub.py::test_theme_leave_with_unsaved_edits_{stay_keeps_category_and_edit,discard_clears_the_dirty_flag,save_writes_then_leaves}` (replacing `test_theme_dirty_flag_clears_when_leaving_the_category`), marked `@private_profile_test` so they run in a clean worktree.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/Widgets/settings_theme_editor.py`, `Tests/UI/test_settings_configuration_hub.py`, `Docs/User_Guide/settings.md`.
<!-- SECTION:NOTES:END -->
