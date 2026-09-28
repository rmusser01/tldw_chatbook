---
id: TASK-33078
title: Theme editor file actions scan off the UI thread
status: Done
assignee:
  - '@claude'
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32957 (PR #2850 Qodo 4116061870). Each editor file action (Save, Save as, Rename, Delete, Import, Export) still resolves names through a ~210 ms backup-scoped folder scan on the UI thread with 50 theme files, and waits up to one picker scan (~414 ms) when they overlap. File operations must stay in SettingsThemeEditor (the backup participant recognises only that instance). Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With 50 saved themes no editor file action blocks the UI thread for more than 100 ms
- [x] #2 File operations still run through SettingsThemeEditor's backup scope
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure each action's event-loop stall with 50 saved themes in a private profile (baseline).
2. Make the file actions coroutines: the folder scan, reads, writes and unlinks go through asyncio.to_thread on the editor's own methods (raw._scope(self, ...) keeps the SettingsThemeEditor participant); validation, confirmations and registration stay on the UI thread.
3. Serialise actions with one asyncio.Lock on the editor; run UI-started actions as app-owned workers so leaving Theme/Settings cannot cut an action between its write and its follow-up steps.
4. The Rename/Import dialogs' inline checks and the dialog label read the last scan instead of scanning on the UI thread; the action re-scans before writing.
5. Tests: I/O-thread placement + stall < 100 ms per action with 50 themes, serialisation, racing the picker scan, leaving mid-action; migrate existing tests to await the coroutines.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All six file actions (Save, Save as, Rename, Delete, Import, Export) are coroutines on SettingsThemeEditor. Their I/O -- `_scan_theme_files`, `_read_toml`, `_write_toml`, `_unlink`, `_write_export_file`, `_parse_import` -- runs via `asyncio.to_thread` on the editor's own methods, so every backup-scoped operation still names the editor as its source (`raw._scope(self, ...)`; the scope is thread-local, as the picker's TASK-32957 scan already relied on). Everything that touches the UI or needs a decision -- validation, ConfirmationDialog, `register_theme`, notices, `_reapply_if_active` -- stays on the event loop. Every safety check is unchanged: link refusal, name validation/printable names, R16 path-free reasons, overwrite confirmation with re-resolve after the dialog (Qodo 4109320416), Export's identity/exclusive-create replace, RecoveryRequired -> "unavailable" notice.

Concurrency: one `asyncio.Lock` (`_file_lock`) per editor; each action (and each confirmed-dialog continuation) holds it across its scan-then-write, so two actions never interleave. `run_file_action()` starts an action as an **app-owned** worker: leaving Theme or Settings mid-action does not cancel it between its write and its registration/launch-default steps (DOM touches go through `_show_name`, a no-op once detached). Picker scans are unaffected: their generation guard means the rescan the action's ThemesChanged starts lands last. The Rename/Import prompts' inline checks and the dialog label read `_last_scan` (set by every scan, including the picker's) instead of scanning on the UI thread; the action re-scans before it writes. Delete's launch-default fallback and Rename's launch-default move now use the off-thread writer from TASK-33121.

Callers: the Settings screen's rename/import/save-as results and the pane's delete/export requests go through `run_file_action`; the three leave prompts (`confirm_navigation`, Back, category leave) `await editor.save_theme()`, keeping the "stay if the save is refused or awaits its overwrite confirmation" rule. `_theme_file_data` copies the colours (the write runs while edits may continue) and a save leaves `is_modified` set if the palette changed during the write.

Measured (50 themes, 211x44, private profile, max event-loop stall per action, 3 runs): before -- Save as 374-506 ms, Rename 1000-1926, Delete 359-673, Import 940-2149, Export read 322-726, Export write 35-58. After -- Save as 14-15, Rename 14-16, Delete 28-32, Import 15-16, Export read 6-9, Export write 16-17. Not changed and out of scope: Save's return to the picker (the ContentSwitcher/`-picker` restyle + list render, the same as Back, ~170 ms at 211x44 regardless of file count) and a modal push/dismiss repainting Settings (~250 ms); Edit/Reset still load a saved theme with a scan on the UI thread (not one of the listed actions).

Tests: Tests/UI/test_settings_theme_file_actions_off_thread.py (I/O-thread placement + <100 ms stall per action, GC paused while metering because full collections of the Settings graph stall 50-600 ms at random; serialisation; racing the picker scan; leaving Settings mid-rename). Existing theme tests now await the coroutines, and the isolated harnesses restore the real `run_worker`.

Files: tldw_chatbook/Widgets/settings_theme_editor.py, tldw_chatbook/Widgets/settings_theme_picker.py (ThemePane delete/export requests), tldw_chatbook/UI/Screens/settings_screen.py (callers + leave prompts), Docs/User_Guide/settings.md, tests above.
<!-- SECTION:NOTES:END -->
