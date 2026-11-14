---
id: TASK-39110
title: "Personal Context's databases have no backup owner, so a profile that has used them previews as incomplete with an unknown owner, although the interview-drafts policy says that store is excluded from backup"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 22:47
labels:
- follow-up
- backup
- recovery
- personal-context
priority: medium
references:
- tldw_chatbook/DB/private_sqlite.py
- tldw_chatbook/Personal_Context/interview_launch.py
- tldw_chatbook/Backup_Recovery/owner_registry.py
- tldw_chatbook/Backup_Recovery/inventory.py
- tldw_chatbook/Notes/recovery.py
- tldw_chatbook/Workspaces/project_index/recovery.py
- backlog/docs/backup-recovery-owner-inventory.md
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-backupinv, which lands TASK-39053. Line numbers are dev's.

**The ruling.** Owner decision (2026-11-06), TASK-39053 row 13: "Keep unsupported + follow-up." The follow-up names `db.workspace_project_index` and `personal_context.interview_drafts`.

**`db.workspace_project_index` is already handled.** Its exclusion is installed and tested:
- `ProjectIndexCacheDeclaration` (`Workspaces/project_index/recovery.py:9`) marks the tree `intentionally_excluded`, installed through the Workspaces factory (`Backup_Recovery/owner_registry.py:72`);
- `Tests/Backup_Recovery/test_workspace_project_index_owner.py` tests it.
- Only the inventory doc and TASK-39053's follow-up miss it.

**Personal Context is not handled.**
- Its interview-drafts policy says "Encrypted interview drafts are local-only and excluded from centralized backup." (`DB/private_sqlite.py:485`).
- But no recovery owner declares the file (`interview_launch.py:105`), and `Personal_Context` is not in `owner_registry.py`'s factory list.
- `personal_context.repository`'s database is in the same position. TASK-24400 (Done) says that store "remains outside centralized backup".

**Probe.** Run on dev (scratch profile, real `discover()` after `install_adapters()`):
- `db.workspace_project_index` and `notes.sync_state` (`_DeviceStateExclusion`, the precedent) come out `intentionally_excluded`;
- `tldw_chatbook_personal_context.db` and `tldw_chatbook_personal_context_interviews.db` come out owner `unknown`, `unsupported`;
- the preview is incomplete, with `unsupported_owner`.

So any profile that has used Personal Context gets an incomplete backup preview with an unknown owner, for files the code says are excluded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A backup preview of a profile holding the interview-drafts database (and its `-wal`/`-shm`/`-journal`) lists those files under `personal_context.interview_drafts` as intentionally excluded, not `unknown`, and capture of them is refused; a test is red before the change, showing the `unknown` row
- [ ] #2 The owner decides whether `personal_context.repository`'s database is excluded the same way, kept as an explicit `unsupported` row, or captured; the decision is recorded and the preview follows it
- [x] #3 The inventory doc names the installed exclusions for `db.workspace_project_index`, `notes.sync_state` and the Personal Context owners
- [x] #4 The new tests run on a PR lane (never the fast PR contract), with three timings of that step recorded
<!-- AC:END -->

## Implementation Plan

1. Record the owner decision (2026-11-06) for `personal_context.repository`: capture it, in its own design task; nothing is captured this round.
2. Red first: a real `discover()` after `install_adapters()` of a profile holding the interview-drafts database and its `-wal`/`-shm`/`-journal`, and the repository database, shows them as owner `unknown` today.
3. Generalize the `notes.sync_state` exclusion (`Notes/recovery.py` `_DeviceStateExclusion`) into one Backup_Recovery-local declaration of an exact SQLite store plus sidecars under the user data dir, and install it for `notes.sync_state` (unchanged behaviour) and `personal_context.interview_drafts` (intentionally excluded, capture/validate/relocate refused). Declare `personal_context.repository` with the same class as an explicit `unsupported` row (present files block, absent files are unused). The declarations live in `Backup_Recovery/config_adapter.py`'s factory, so installing them imports nothing from the Personal Context package.
4. Name the installed exclusions (`db.workspace_project_index`, `notes.sync_state`, the Personal Context owners) in the inventory doc.
5. Put the tests in `Tests/Architecture/test_backup_owner_inventory.py` (already on two lanes) and time its lane step three times.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decision (2026-11-06), AC#2:** for `personal_context.repository`: "Capture it: back up the database and design key custody so a restore can read it — a larger feature that needs its own design task." Not implemented this round: capture follows in its own design task. Until it lands the repository is an explicit `unsupported` row (below). AC#2 stays unticked: capture follows in its own design task.

**Approach.**
- `_LocalSQLite` (`Backup_Recovery/config_adapter.py`) generalizes `notes.sync_state`'s `_DeviceStateExclusion` (`Notes/recovery.py`, removed): one exact SQLite file under the user data dir plus its `-wal`, `-shm` and `-journal`, each through `_excluded_root`'s checked exact-kind evidence; `validate`/`capture`/`relocate` refuse with the declaration's reason code; no schema policy. `notes.sync_state` now installs `_LocalSQLite("notes.sync_state", "tldw_chatbook_notes_sync_state.db", "notes_device_state_excluded")`, the same behaviour.
- `personal_context.interview_drafts` installs it for `tldw_chatbook_personal_context_interviews.db`: intentionally excluded, capture refused (`personal_context_interview_drafts_excluded`), matching the policy text "Encrypted interview drafts are local-only and excluded from centralized backup."
- `personal_context.repository` installs it with `excluded=False` for `tldw_chatbook_personal_context.db`: a present file (and sidecar) is an explicit `unsupported` row under that owner, so a preview stays incomplete but no longer with `unknown` owner / `unsupported_owner`; an absent file is `unused`, so a profile that never used Personal Context is unaffected; capture is refused (`personal_context_repository_capture_unsupported`).
- Both are registered from `config_adapter.recovery_adapters()`, so installing them imports nothing from `Personal_Context` (its `__init__` pulls in crypto and services). Neither owner has a selectable backup group; `Tests/Backup_Recovery/test_data_groups.py`'s non-selectable set names both.

**Red first (AC#1).** `test_personal_context_stores_preview_under_their_installed_owners` (`Tests/Architecture/test_backup_owner_inventory.py`) builds a profile holding both databases and all three sidecars, runs `install_adapters()` and the real `discover()`, and checks owner and status per file, then that both owners' `capture` refuse. Before: all eight files were `('unknown', 'unsupported')`. After: drafts `('personal_context.interview_drafts', 'intentionally_excluded')` x4, repository `('personal_context.repository', 'unsupported')` x4, both captures refused, nothing written.

**AC#3.** `backlog/docs/backup-recovery-owner-inventory.md` gains "Installed SQLite exclusions (TASK-39110)", naming `db.workspace_project_index` (`ProjectIndexCacheDeclaration`), `notes.sync_state` and `personal_context.interview_drafts` (`_LocalSQLite`) with their exact paths, statuses and refusal codes, and the repository's explicit unsupported declaration; the four SQLite registry rows' status column say the same. User Guide: `Docs/User_Guide/settings/personal-context-profile.md` (Privacy notes) says drafts are never backed up and the database cannot be backed up yet.

**AC#4.** The new test is in `Tests/Architecture/test_backup_owner_inventory.py`, already on two lanes: architecture-guards "Run Architecture guard contracts" (serial) and the backup-recovery qualification workflow. No pin changed. Lane timings: architecture-guards "Run Architecture guard contracts" (35 targets, serial; it runs `Tests/Architecture/test_backup_owner_inventory.py`, `Tests/Utils/test_log_sanitizer.py`, `Tests/test_persistent_diagnostic_boundary.py` and `Tests/Persona_Buddy/test_controller_failure_logging.py`), with this round's nodes added, through `jobstep.sh` three times: 430 passed each, in 821.6 s, 564.2 s and 522.6 s, at 1-minute load 158, 52 and 33 when each started. Its last CI run took 7.2 min (PR run 42310788537), so the added nodes stay small against the 20-min cap; the local spread is load.

**Proposed follow-up (design task for AC#2).** Capture `personal_context.repository` with key custody a restore can use. Questions found while reading `Personal_Context/key_protector.py`, `link_key_custody.py` and `interview_launch.py`:
- The profile key lives in a verified secure OS keyring (`KeyringProfileKeyProtector`) or behind a passphrase wrapper (`PassphraseProfileKeyProtector`); an archive of the database alone is unreadable. Does the archive carry a wrapped key (under the backup's own credential), or does restore require the same keyring entry or passphrase? Where does the passphrase wrapper's salt/ciphertext live, and is it in the archive?
- Restore onto another machine or OS account: the keyring entry does not travel; what re-keys or re-wraps the records, and what does the user see when the key is missing?
- Link/sync key custody (`link_key_custody.py`) and server-trusted sync state: restored as-is, re-enrolled, or excluded (a restored device must not impersonate the old one)?
- Rollback and replacement: a restored database with a key that no longer matches the keyring.
- Whether the interview-drafts exclusion should also join the detached-layout preview's fixed owner list (`Backup_Recovery/detached_layout.py`, beside `notes.sync_state`); today a detached source holding drafts refuses as unknown content.

**Covering run.** 54 files (Backup_Recovery inventory, owner and data-group files, `Tests/UI/test_backup_data_groups.py`, the Workspace / Library Collections / notifications / store-kernel DB tests, `Tests/Media_DB`, the media and quiescence DB tests) at `-n 4` on this branch: 1379 passed, 27 failed, 3 skipped in 13 min (load ~40). Rerun per file on base `fu-base-r12`: 25 of the 27 fail there too. The other two were `test_core_owner_uses_literal_seam_and_preserves_connection_contract[media-first-create|media-harden]`, which pin media's exact connect kwargs; they now include the `factory`, as the ChaChaNotes case already does (TASK-39118). `test_every_durable_installed_adapter_has_a_named_selectable_group` was red on dev on `db.workspace_project_index` (missing from its non-selectable set), so it never reached the Personal Context owners; that row is added with them and the test passes.

**Files.** `tldw_chatbook/Backup_Recovery/config_adapter.py`, `tldw_chatbook/Notes/recovery.py`, `Tests/Architecture/test_backup_owner_inventory.py`, `Tests/Backup_Recovery/test_data_groups.py`, `backlog/docs/backup-recovery-owner-inventory.md`, `Docs/User_Guide/settings/personal-context-profile.md`.
<!-- SECTION:NOTES:END -->
