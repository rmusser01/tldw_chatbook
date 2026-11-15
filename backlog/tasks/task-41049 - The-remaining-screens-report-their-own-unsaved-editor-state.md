---
id: TASK-41049
title: "The remaining screens report their own unsaved-editor state"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- architecture
- backup-recovery
- ui
priority: low
references:
- tldw_chatbook/Backup_Recovery/unsaved_editors.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36948.3 gave ChatScreen a public `has_unsaved_editor_state()`, so the backup unsaved-work probe (`Backup_Recovery/unsaved_editors.py`) no longer reads ChatScreen's private fields.

The probe still reads private fields or DOM ids of:
- LibraryScreen and LibraryFileNotesWorkspace;
- SettingsScreen and PersonasScreen;
- STTSWindow, SpeechPlaygroundPane and the stts_screen module;
- ScreenStateStore;
- the app-hosted Console attachment stash.

TASK-36948.3's notes list each line. If any of these owners renames a field, the probe guesses or refuses.

On Settings the probe already answers "unknown-editor-state" for every category except Overview and Storage, because `_clean_settings_form` raises TypeError("unqualified settings form"). TASK-1378.1's review found this, and it predates that task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed screen or widget reports its unsaved state through a public method
- [ ] #2 unsaved_editors reads no private field of those owners, pinned by a guard like TASK-36948.3's
- [ ] #3 Settings gives a real answer for every category instead of unknown-editor-state
- [ ] #4 The probe's existing tests pass
<!-- AC:END -->
