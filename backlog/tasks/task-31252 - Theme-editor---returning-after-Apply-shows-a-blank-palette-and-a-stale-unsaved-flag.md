---
id: TASK-31252
title: Theme editor - returning after Apply shows a blank palette and a stale unsaved
  flag
status: Done
created_date: 2026-09-04 05:23
dependencies:
- TASK-31251
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: high
updated_date: 2026-09-04 06:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply registers the theme as custom_<name> and sets app.theme to it. Leaving the category and coming back remounts the editor, which calls load_theme(self.app.theme); 'custom_<name>' matches neither the built-in branch nor ALL_THEMES, so current_theme_data stays empty: ten inputs show the #RRGGBB placeholder, the Name box shows custom_new_theme, and the inspector still says 'Unsaved theme changes: Yes' because the screen's flag is never reset on remount. Reproduced live. Keep the mount-posts-no-status guard (is_modified init=False) that prevents the recompose storm. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Apply, leaving and re-entering Theme shows the applied theme's name and all ten colours
- [x] #2 The inspector's 'Unsaved theme changes' row and the rail marker reflect the freshly mounted editor's real state
- [x] #3 test_theme_category_settles_without_recompose_storm and test_settings_theme_editor_mount_posts_no_modified_status still pass
- [x] #4 A regression test applies a theme, remounts the editor, and asserts the palette and the unsaved flag
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
_initialize_editor resolves app.theme 'custom_<name>' through app.available_themes and shows the user-facing name (Apply keeps the custom_ prefix so an edited shipped theme never clobbers the shipped registration). SettingsScreen._select_category clears theme_editor_modified when leaving Theme. Mount still posts no ThemeModifiedStatus (storm guard kept). Tests: editor remount test + screen dirty-flag test.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
