---
id: TASK-33240
title: Theme editor Edit and Reset read theme files off the UI thread
status: Done
assignee: []
created_date: '2026-09-28 08:00'
labels:
  - settings
  - theme
  - perf
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from the TASK-33078 tail wave (PR #2877). Opening a saved theme in the editor (Edit) and Reset still resolve the theme file through the backup-scoped themes-folder scan on the UI thread (~210 ms with 50 saved themes), unlike the file actions that TASK-33078 moved to a worker. File operations must stay inside SettingsThemeEditor's raw backup scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With 50 saved themes, Edit and Reset never block the UI thread for more than 100 ms
- [x] #2 The editor shows the correct theme data once the read lands, and a stale read never overwrites a newer editor session
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Split the saved-theme read (scan + file read + validation) into a sync helper that runs on a worker thread under the editor's own raw backup scope and the app-wide file-action lock.
2. `load_user_theme` starts a new editing session at call time and returns a coroutine; the read's result is applied only if that session is still current (Back, another Edit, Clone/New, teardown all end it).
3. The picker's Edit on a saved theme runs the load as an app-owned file action and switches to the editor only once the data landed; the picker stays visible meanwhile (no flash of the previous palette).
4. Reset's confirmed step runs as a file action with a single off-thread read (it scanned twice on the UI thread before).
5. Tests: 50-theme stall measurement for Edit and Reset, stale-read races (Back mid-read, Edit another mid-read), migrate sync callers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Edit (on a saved theme) and Reset now resolve and read the theme file on a worker thread, through the same file-action machinery as Save/Rename/Delete/Import/Export (app-owned worker in THEME_FILE_ACTION_GROUP, app-wide `_file_lock`, quit barrier). The read stays in the editor's own raw backup scope (`raw._scope(self, ...)`).

- `SettingsThemeEditor.load_user_theme` starts a new editing session at call time and returns a coroutine; `_read_saved_theme` (scan + read + validate) runs via `asyncio.to_thread` under the file lock, and `_show_saved` applies it only if the session is still current. Superseded loads return None silently.
- `ThemePane.open_editor` runs a saved theme's load as a file action and switches to the editor only once the data landed. Decision: no loading state -- the picker stays up meanwhile, so the editor never flashes its previous palette. Opening another theme, leaving Theme or Back ends the session, so a late read opens nothing. It returns the worker (None for catalog themes, which open synchronously as before).
- Reset's confirmed step is `_as_file_action(self._reset_theme)`; it does one off-thread read (it scanned twice on the UI thread before) and lands only in its own session.
- Measured (50 saved themes, 211x44, GC paused, 3 runs, serial): Edit's full UI-thread stall incl. view switch 1119-1561 ms -> 76-102 ms (65-71 ms up to the read landing); Reset 1855-2176 ms -> 22-62 ms.
- Tests: `Tests/UI/test_settings_theme_file_actions_off_thread.py` -- 50-theme stall test for Edit/Reset and three race tests (Edit superseded by another Edit, Edit read landing after Back, Reset read landing after Back); each race test fails with the staleness guard removed. Existing tests now `await load_user_theme` / wait for the Edit worker.
- Files: `settings_theme_editor.py`, `settings_theme_picker.py` (open_editor only), tests, `Docs/User_Guide/settings.md`.
- Qodo round (PR #2886): Reset now takes its target, session and the on-screen palette in the confirm callback (`_reset_theme` is sync and returns the coroutine, like `load_user_theme`), so a worker that starts after Back + another Edit resets nothing; and a read landing after the user kept editing is skipped with a notice instead of wiping the unconfirmed edits (the dialog closes before the read lands, and the read can queue behind another file action on the app-wide lock). Edit needs neither: its session starts at the click and the picker stays up (editor hidden) until the read lands, so nothing can be edited in between.
<!-- SECTION:NOTES:END -->
