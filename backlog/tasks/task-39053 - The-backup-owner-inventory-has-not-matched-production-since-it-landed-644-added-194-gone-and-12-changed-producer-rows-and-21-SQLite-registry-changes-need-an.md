---
id: TASK-39053
title: "The backup owner inventory has not matched production since it landed: 644 added, 194 gone and 12 changed producer rows and 21 SQLite registry changes need an owner and cohort review"
status: To Do
created_date: 2026-11-06 10:09
labels:
- follow-up
- backup
- recovery
- architecture
- tests
- needs-owner-decision
priority: high
references:
- backlog/docs/backup-recovery-owner-inventory.md
- backlog/docs/sqlite-private-owner-inventory.md
- Tests/Architecture/test_backup_owner_inventory.py
- tldw_chatbook/DB/private_sqlite.py
- Tests/DB/test_store_kernel.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38989
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on Architecture PR #3436 (head b7c8e809b3), which lands TASK-38989. Line numbers are dev's and are the same on that head. TASK-38989 carved these two nodes out of its scope and left them red for an owner review. #3436 adds an `architecture-guards` PR job and keeps this file out of it.

**The drift.** It was recomputed with the test's own `census()` / `production_census()`. The result is the same on dev, on every round-10 branch and on e8eeae7db9:
- The doc has 1059 producer rows and production has 1509: **644 added, 194 gone, 12 changed**.
- The SQLite registry (`_SQLITE_OWNER_POLICIES`, `tldw_chatbook/DB/private_sqlite.py`) has **13 new members and 8 gone**.
- `Tests/Architecture/test_backup_owner_inventory.py` (serial, 23.6 s on dev): 2 failed, 9 passed. The two are `test_every_persistence_source_symbol_has_reviewed_owner_row` (`:146`) and `test_sqlite_registry_members_have_explicit_census_rows` (`:184`). No PR lane runs the file.
- It has been red since it landed:
  - at b5251e9a6e (2026-09-16, the squash that created it, TASK-32628): 407 / 86 / 9, SQLite 8 / 5;
  - at the doc's last edit, 81184d04f5 (2026-10-11): 536 / 170 / 10, SQLite 9 / 8.
  - Done tasks that recorded "11 passed" ran it on branch snapshots.

**What needs a decision.**
1. **13 new SQLite members:** chat.launch_wake, chat.trace_maintenance, db.chunking_lab, db.workspace_project_index, library.legacy_recovery, personal_context.interview_drafts, personal_context.repository, plugins.registry, recovery.operations.dreams, recovery.operations.guardian, tests.store_kernel, utils.db_upgrade_notice, workflows.local. dreams and guardian (`private_sqlite.py:216-227`, f927f98add) are `centralized_backup_allowed` and `recovery_capture_allowed`.
2. **A test owner in the production registry.** `"tests.store_kernel"` (`private_sqlite.py:420-424`, owner `Tests/DB/test_store_kernel`) was added by cbb39dd767 (TASK-33132). The SQLite private-owner census already accepts it (`backlog/docs/sqlite-private-owner-inventory.md:377`, row C97).
3. **8 rows that exist only in the doc.**
   - `sqlite:recovery.operations.receipts` (doc `:1349`) was never in the registry (`git log -S` finds no registry commit) and was stale when the test landed.
   - Seven `sqlite:settings.*` rows (doc `:178-184`) lost their registry entries in bd1a4258ea (TASK-33244, which removed the Tools settings window). That commit updated the other inventory, not this one.
4. **644 added producer rows.**
   - 535 are in 194 modules the doc never lists. The largest packages: Plugins 63 (`Plugins/authority_store.py` alone 25), Tools 43, Agents 40, Audio 40, Chat 35, Actor_Packs 33, Backup_Recovery 31.
   - 109 are in modules the doc already lists.
5. **194 gone rows.** 58 are in 15 deleted files (for example `UI/Tools_Settings_Window.py`, 25), 44 look like renames within a module, and the rest are removed or moved code.
6. **12 changed counts**, for example `ChatbookCreator` x5, `_connect_registered_sqlite` connect 1 -> 2, `BaseDB._get_connection` 1 -> 2, `KeyringServerCredentialStore._save_index` set_password 1 -> 3.

Classifying rows into owners and backup cohorts is a product decision about what a backup captures. The criteria leave the classifications to the owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 13 new SQLite members has a reviewed row with a classification and a cohort; where a member's `SQLiteOwnerPolicy` flags do not settle its cohort, the owner decides it and the decision is recorded
- [ ] #2 The owner decides whether a test-owned policy (`tests.store_kernel`) may stay in the production registry; the decision is recorded and the registry and census follow it
- [ ] #3 The 8 rows that exist only in the doc are removed, each with its cause recorded
- [ ] #4 Every added and changed producer row has an owner-reviewed classification (the owner signs off the classifications, or the cohorts they fall in, in this task's notes), and every gone row is removed
- [ ] #5 Both nodes pass; they are red before (2 of 2 on dev b2c618185f: 644 / 194 / 12, SQLite 13 / 8)
- [ ] #6 Negative control: a synthetic unreviewed producer turns the test red, and the failure message lists every added, gone and changed row
- [ ] #7 The file runs on a PR lane (the Architecture guards job, or another the owner picks; never the fast PR contract), with three timings of that lane's command recorded
<!-- AC:END -->
