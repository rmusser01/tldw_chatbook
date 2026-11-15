---
id: TASK-41125
title: "Three Backup_Recovery test files are red on dev and run on no lane: test_activation_startup x16, test_backup_known_profiles x1, test_inert_extraction x1"
status: To Do
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
- [ ] #1 Each node passes, or is retired with the commit that changed what it pinned named; each red is shown first with the same command
- [ ] #2 A product defect gets a test red before its fix
- [ ] #3 The three files run on a PR lane or the push-to-dev owner job
<!-- AC:END -->
