---
id: TASK-41125
title: "Three Backup_Recovery test files are red on dev and run on no lane: test_activation_startup x16, test_backup_known_profiles x1, test_inert_extraction x1"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-15 20:07'
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- tests
- dev-red
priority: medium
references:
- Tests/Backup_Recovery/test_activation_startup.py
- Tests/UI/test_backup_known_profiles.py
- Tests/Backup_Recovery/test_inert_extraction.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** They pin restored-profile start-up gating, the known-profiles review and manual extraction. Red and unlaned, they guard nothing.

**Evidence.** Through run.sh at `-n 4` (load 50-58): 18 failed, 28 passed.
- `test_activation_startup.py`: 16 failed (`test_restored_startup_callbacks_remain_inactive` x4, `test_restored_scheduler_does_not_dispatch_or_sync` x5, `test_remote_scheduling_edits_keep_local_fallback` x6, `test_cancelled_speech_waiter_retains_actual_execution_hold`). The child raises `RecoveryRequired: replacement_identity_unverified` on a config import after binding (also recorded at dev 26b2e677dd in TASK-40711's notes, 17 then).
- `test_backup_known_profiles.py::test_default_profile_review_keeps_known_profiles_and_added_selection`: "Source review failed: Review the selected file and local folders, then try again. … (backup_operation_failed)" (`:78`).
- `test_inert_extraction.py::test_fresh_manual_extraction_survives_damaged_registry_without_runtime_imports`: the child fails `AttributeError: module '__main__' has no attribute '__file__'` (`:373`).

`Tests/ProductionApp/test_backup_restore_composition.py` (2 failed, 1 error at 26b2e677dd) is TASK-35577's.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each node passes, or is retired with the commit that changed what it pinned named; each red is shown first with the same command
- [x] #2 A product defect gets a test red before its fix
- [x] #3 The three files run on a PR lane or the push-to-dev owner job
<!-- AC:END -->

## Implementation Plan

1. Show each red with one command (-n 4).
2. test_activation_startup: compose config before the synthetic activation bind and allow exactly the private SQLite helper exec (the test_activation_notes pattern).
3. test_inert_extraction: import the admission module before the child replaces bootstrap._registry.
4. test_backup_known_profiles: find the cause; a product defect gets the red test first, then the fix.
5. Lane the three files (owner job or a PR lane).

## Implementation Notes

Commits 6673ea6661 (fixes), 7d9c3988b1 (lanes). Same command before and after: `pytest Tests/Backup_Recovery/test_activation_startup.py Tests/UI/test_backup_known_profiles.py Tests/Backup_Recovery/test_inert_extraction.py -n 4 --timeout=300`: 18 failed, 28 passed (load 50-58) -> 46 passed (load 82).

- **test_activation_startup x16 (test fix).** Two causes. (1) The scripts bind a synthetic activation (no restore journal) and only then import `Scheduling`/`app`, which import `config`, whose `CLI_APP_CLIENT_ID = installation_client_id()` calls `replacement_data_identity()`; that refuses an activation without a committed journal (`replacement_identity_unverified`). The scripts now import config before the bind and close the startup lease that import took (without the close, the first DB open refuses `close_owners_before_scope_change`), as `test_activation_notes.py` does. (2) Every SQLite open runs the private helper child (9c6dbd2bf0), which the scripts' no-process audit hook refused (`private_sqlite_helper_unavailable` in the `ordinary` cases); the hook now allows exactly the helper's argv with this PID, as `test_activation_notes.py` does. Not a retirement: the file was already red at its creation merge 24094f23d5 (the child run on that archived tree fails the same way), so no later commit changed what it pins.
- **test_inert_extraction x1 (test fix).** `admission.py` snapshots bootstrap's reader functions and their modules' `__file__` at import (`_PAUSE_METADATA_MODULES`, 3e618d7119, merged by 14e13bb82c). The child replaced `bootstrap._registry` with a function defined in its `python -c` `__main__`, which has no `__file__`, before admission was first imported. The child now imports admission (via `service_storage`) before the replacement; extraction imported it anyway, and no forbidden module is involved.
- **test_backup_known_profiles x1 (product defect, AC#2).** `dependency_unavailable` from `resolve_inventory_groups`: for a known profile with no ChaChaNotes database yet, the core is `unavailable` and depends on all its dependent owners, but the builtin visual identity owner's `_base_entries` raised (it reads the core for its references), so it declared only its `references_unavailable` item and the core's edge to its root dangled. The other persona owners keep their root (`unavailable`) in the same state. `Persona_Visual/recovery.py` now does the same for the builtin root. New regression test `Tests/Backup_Recovery/test_inventory.py::test_profile_without_data_resolves_every_core_dependency`: red first (`ValueError: dependency_unavailable`), then green; the UI node is red/green the same way. Red also at the creation merge 24094f23d5 (in-process, archived tree).
- Covering runs: test_file_inventory, test_data_groups, test_visual_identity_restore, test_builtin_safety_directory, test_inventory, test_inventory_unused_owners at -n 4: 179 passed, 3 failed (the private_sqlite-boundary node, an order-dependent node now fixed by test_inventory's fixture, and `test_visual_identity_restore`'s child timeout, which fails the same way with the product change reverted at load 112). `Tests/Architecture/test_backup_owner_inventory.py` serially: passed. Ruff gate: passed.
- **Lanes (AC#3).** `test_activation_startup.py` and `test_inert_extraction.py`: owner job. `Tests/UI/test_backup_known_profiles.py`: UI census tail, MINIMUM_FILES 2146; serial with the UI lane's flags 24.34 s (18.06 s call, 3.34 s setup) at load 104-111.
