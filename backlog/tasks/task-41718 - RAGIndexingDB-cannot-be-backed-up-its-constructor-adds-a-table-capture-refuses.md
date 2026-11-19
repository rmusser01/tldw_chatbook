---
id: TASK-41718
title: "RAGIndexingDB cannot be backed up: its constructor adds a table capture refuses"
status: Done
assignee: ['@claude']
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 16:08
labels:
- follow-up
- backup
- rag
priority: high
references:
- tldw_chatbook/DB/RAG_Indexing_DB.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41521 (round 16). `RAGIndexingDB`'s constructor adds an `embedding_cache` table, so capture refuses today's `rag_indexing.db` with `unsupported_schema`. The TASK-41521 test had to build that database from the owner's installed schema instead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A backup of a profile whose RAG index was created by the app captures `db.rag_indexing` without `unsupported_schema`
- [x] #2 A test creates the database through `RAGIndexingDB` and backs it up, red before the fix
<!-- AC:END -->

## Implementation Plan

1. Confirm the refusal: capture's exact-catalog check (`Backup_Recovery/rag_indexing.py` `_SCHEMA`) lacks the `embedding_cache` table and index that `RAGIndexingDB` has created on every open since TASK-34420.
2. Append the current catalog as an exact alternative at schema 0 (the pre-cache catalog stays accepted), as `DB/recovery_operations.py` does for the subscriptions index.
3. Tests: the existing adapter tests that build the database through `RAGIndexingDB` (red at base), a profile backup of an app-made index (TASK-41521's `[tools]` case, which built the database from the installed schema to dodge this), the pre-cache catalog still accepted, and unknown or partial catalogs still refused.

## Implementation Notes

**Summary.** Capture checks an indexing database against an exact installed catalog, and `Backup_Recovery/rag_indexing.py` still listed the catalog from before TASK-34420, when `RAGIndexingDB` began creating `embedding_cache` and `idx_embedding_cache_created` on every open. So every index the app made was refused with `unsupported_schema`, and every backup that included Retrieval failed. The current catalog is now appended to `_SCHEMA` as an exact alternative at the same unversioned schema 0 (appended, never inserted, as `DB/recovery_operations.py` does for the subscriptions index); the older catalog stays accepted for an index made before TASK-34420 and not opened since. Restore validation uses the same policy, so no other change was needed.

**Tests.**
- AC#1/#2 (profile backup): `Tests/Backup_Recovery/test_created_owner_rollback.py::...[tools]` now builds the index with `RAGIndexingDB` (an `extra` row) instead of from the owner's installed schema, the workaround TASK-41521 used; its `schema_dbs` producer path is gone. Red at base: the source backup failed in `capturing` (`backup_operation_failed`). Green: 1 passed, 277 s (`-n 4`, load 30-50).
- AC#2 (adapter): `Tests/Backup_Recovery/test_rag_indexing_recovery.py::test_indexing_snapshot_preserves_schema_rows_and_source` already built the database through `RAGIndexingDB` and captured it, and was red at base (`ValueError: unsupported_schema`); so were `test_sqlite_validation.py::test_native_rag_indexing_schema_zero_survives_restricted_validation[False|True]`. All three are green now. Neither file ran on any lane, which is why dev never showed it.
- New: `test_indexing_catalog_from_before_the_embedding_cache_is_accepted` (the older catalog still validates; red with the older alternative removed from `_SCHEMA`), and `test_indexing_unknown_schema_is_rejected[DROP INDEX idx_embedding_cache_created]` (the cache table without its index still refuses; red with the index left out of the new alternative). The unknown-table case still refuses.

**Lane.** `test_rag_indexing_recovery.py` (8 tests, about 5 s at `-n 4`) joins the Backup_Recovery owner suites job in `.github/workflows/derived-artifacts.yml`, and `BACKUP_OWNER_SUITE_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`. That pin failed on the lane edit alone (`At index 23 diff: 'Tests/Backup_Recovery/test_rag_indexing_recovery.py' != ...`) and passes with the entry added.

**Covering runs** (after TASK-41719's part was withdrawn from this branch; `-n 4`, load 43-65): `test_rag_indexing_recovery.py`, `test_sqlite_validation.py`, `test_rag_discovery.py` and the whole of `test_created_owner_rollback.py` (all five cases, 86-321 s each): 105 passed (7 min). `Tests/CI`: 1811 passed, 2 skipped. `./scripts/preflight.sh`: passed. Earlier, on the same TASK-41718 code, `Tests/Architecture/test_backup_owner_inventory.py` and `test_module_size_ratchet.py` (serial): 284 passed. Ruff reports nothing new on the changed lines.

**Files.** `tldw_chatbook/Backup_Recovery/rag_indexing.py`, `Tests/Backup_Recovery/test_rag_indexing_recovery.py`, `Tests/Backup_Recovery/test_created_owner_rollback.py`, `Tests/CI/test_ci_queue_pressure_contract.py`, `.github/workflows/derived-artifacts.yml`.
