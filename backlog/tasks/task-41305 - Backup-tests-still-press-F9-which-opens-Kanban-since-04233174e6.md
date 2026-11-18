---
id: TASK-41305
title: "Backup tests still press F9, which opens Kanban since 04233174e6"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 00:18
updated_date: '2026-11-17 02:05'
labels:
- follow-up
- backup
- tests
priority: medium
references:
- Tests/ProductionApp/test_backup_restore_composition.py
- Tests/ProductionApp/test_backup_restore_end_to_end.py
- Tests/Backup_Recovery/test_recovery_restart.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from review r15bt-rev1 of round 15 (coordinator, 2026-11-16).

**Why.** Several backup journeys open Settings by pressing F9. F9 was unbound until 04233174e6 (2026-10-18) bound it to Kanban, so the press now leaves Settings and the journey fails to find `#settings-backup-restore`. TASK-41116 fixed the same press in `test_f9_replacement_workflow.py` (146f5156af).

**Evidence.** Sites still pressing F9: `test_backup_restore_composition.py:44`, `:92`, `:204`; `test_backup_restore_end_to_end.py:219`; `test_recovery_restart.py:149`, `:333`. Five of them share the composition file's `_ENTRY` prelude.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed site opens Settings through SHELL_DESTINATION_SHORTCUTS['settings'] instead of F9
- [x] #2 Each affected test is shown red before and green after, or each one load prevents completing is named with its blocker
<!-- AC:END -->

## Implementation Plan

1. Import `SHELL_DESTINATION_SHORTCUTS` in the composition `_ENTRY` prelude, which every listed script shares.
2. Replace each `pilot.press('f9')` with the Settings shortcut.
3. Run each affected test before and after, at -n 4.

## Implementation Notes

All six presses now open Settings through `SHELL_DESTINATION_SHORTCUTS['settings']`. One import line in `test_backup_restore_composition.py`'s `_ENTRY` prelude serves every site: the composition scripts, `test_backup_restore_end_to_end.py` and `test_recovery_restart.py` all build their child from `_ENTRY.split("async def main():", 1)[0]`.

The same commands were run before and after (`-n 4 --timeout=900`, ProductionApp and Backup_Recovery in separate invocations, load 60-86). Before, the failure was `NoMatches '#settings-backup-restore' on KanbanBoardScreen`, or `AssertionError: <class ...KanbanBoardScreen>`.
- `test_recovery_restart.py`: `test_actual_normal_app_guarded_recovery_handoff` x5, `test_actual_handoff_execs_fresh_recovery_ui` x2 and `test_actual_cli_restarts_only_after_successful_normal_run[success]` went from 8 failed to green; `[shutdown_failure]` passed both times. 9 passed after.
- `test_backup_restore_composition.py`: `test_canonical_f9_entry_reuses_app_service_and_shutdown_closes_it` went from red to green. `test_actual_mounted_backup_publishes_verified_archive_after_navigation` gets past Settings now but is not stably green: it passed in the after run here, and review r15bt-rev2's run hit the script's 75 s `asyncio.timeout` while waiting for the backup to finish. That wait is its load blocker, not F9. `test_mounted_isolated_restore_uses_reviewed_local_destination_slots` now gets through Settings, the restore and "Restoration validated", then fails a later, non-F9 assertion (`config_path == parent / 'config' / 'config.toml'`, from `ProfileCatalog.resolve`). That file's remaining reds belong to TASK-35577.
- `test_backup_restore_end_to_end.py`, all five parameters: now reach the restarted recovery UI ("Recovery copies", "Restored profiles") and fail later on fixed waits in the restart journey (`TimeoutError` and `wait(timeout=5)`) when run together at -n 4, load 60-70. `[plain]` alone at -n 2 passed. Load blocks the other four here; this is not the F9 press.

