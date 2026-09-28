---
id: TASK-32942
title: Theme editor survives backup-recovery pauses
status: Done
created_date: 2026-09-24 14:45
assignee:
- '@claude'
labels:
- settings
- theme
- backup
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since backup/recovery (TASK-32628) the theme editor reads and creates its themes folder through the recovery-aware file scope, which refuses while a backup or recovery holds the files. The editor's constructor and its theme-list loader did not handle that refusal, so opening Theme during a pause could crash Settings instead of saying the files are temporarily unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test reproduces the refusal and shows whether the editor crashes
- [x] #2 During a pause the editor opens without crashing and creates no directory
- [x] #3 The theme list says, in text, that theme files are unavailable while backup/recovery is in progress
<!-- AC:END -->

## Implementation Plan

1. Test: patch the raw scope to raise `RecoveryRequired`, mount the editor.
2. If it crashes, catch `RecoveryRequired` in the constructor's mkdir and in `_load_user_themes`, showing a text row in Your themes.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced: the new test failed with `RecoveryRequired` raised straight out of `SettingsThemeEditor()`, and nothing in Settings' compose catches it. The constructor now logs and continues; `_load_user_themes` adds a data-less leaf "Theme files unavailable while backup/recovery is in progress" (selecting it loads nothing). Save/Delete/Export/load already caught exceptions and notify.

Deliberate contract change: `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py::test_theme_constructor_has_no_directory_effect_during_pause` pinned the constructor *raising*; it now pins the real invariant (no directory created during the pause) without the raise.

Files: `tldw_chatbook/Widgets/settings_theme_editor.py`, `Tests/UI/test_settings_theme_editor.py`, `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py`.
<!-- SECTION:NOTES:END -->
