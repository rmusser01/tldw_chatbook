---
id: TASK-42544
title: "Decide whether to drop the superseded ChaChaNotes index idx_console_trace_response_revision"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 23:44
updated_date: 2026-11-26 08:30
labels:
- follow-up
- db
- performance
priority: low
references:
- tldw_chatbook/DB/migrations/chachanotes_v55_to_v56_console_semantic_trace.sql
- tldw_chatbook/DB/migrations/chachanotes_v71_to_v72_voice_trace_provenance.sql
- Tests/DB/test_chachanotes_v56_console_trace_query_plans.py
- scripts/index_plan_pin_census.tsv
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
v56 created the partial index `idx_console_trace_response_revision ON console_trace_response_links(semantic_revision_id) WHERE semantic_revision_id IS NOT NULL`. v72 (b7dd8e53f2) added the full `idx_console_trace_response_links_semantic_revision_exact` on the same column. On a current file with no `sqlite_stat1`, the full index wins every product query on that column. Probed in TASK-42517:
- `WHERE semantic_revision_id = ?`
- the policy join in `console_semantic_revision.py`
- the voice-promotion locator check

The partial index is chosen only for a `semantic_revision_id IS NOT NULL` range scan, which no product query issues.

TASK-42517 re-pinned the v56 plan test to the full index. Its census row stays `plan-pinned`, but the evidence is a genuine v71 file, where the partial index was still chosen. So it is a dead index with a green pin: every write to `console_trace_response_links` maintains it, and nothing reads through it.

Recommendation: drop it in the next ChaChaNotes schema step that is going ahead anyway, with the same requalification as v88:
- the recovery catalog and the restore sandbox admitting exactly that `DROP INDEX`;
- the census row changed or removed (`check_index_plan_pins.py` keeps a row for every name any migration still creates);
- `EXPECTED_CHACHANOTES_INDEXES`.

Keeping it costs one extra index write per response link.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A recorded decision: drop the index in a named schema step, or keep it with the reason
- [x] #2 If dropped: no product query plan changes (each probed query still plans through the v72 exact index without `sqlite_stat1`), and the census, index literal and recovery catalog agree with the new schema
- [x] #3 The census row no longer claims a live plan pin for an index no current query chooses
<!-- AC:END -->

## Decision

Drop it in ChaChaNotes schema v89, the next schema step (coordinator's engineering call, 2026-11-26; dev's `_CURRENT_SCHEMA_VERSION` was 88 at the start, and no open PR claimed v89). No product query plans through it on a current file, and every write to `console_trace_response_links` maintains it.

## Implementation Plan

1. v89 step `chachanotes_v88_to_v89_drop_superseded_response_revision_index.sql` (one `DROP INDEX`), its `_migrate_from_v88_to_v89` through the shared `run_sql_version_step`, and `_CURRENT_SCHEMA_VERSION = 89` (size ratchet by the exact growth, dated).
2. Recovery: `CHACHANOTES_V89_SCHEMAS` (each v88 lineage member minus the dropped declaration), the policy lineage (`recovery_core.py`), the Subscriptions stamps (`recovery_operations.py`), and the restore sandbox admitting exactly that `DROP INDEX` on `console_trace_response_links` during v88->v89 (`sqlite_validation.py`), with a gate test whose DENY cases (any other DROP INDEX, the listed name on another table, outside the step, temp, trigger-sourced) each go red under a matching temporary loosening; the real-v88-candidate restore migration.
3. Census: remove the name from `EXPECTED_CHACHANOTES_INDEXES`; give `check_index_plan_pins.py` a `dropped` status (a row whose index a DB source drops with `DROP INDEX`; no plan pin claimed) with checker tests, and change the row to it.
4. Plans: each probed product query (`WHERE semantic_revision_id = ?`, the `console_semantic_revision.py` policy join, the voice-promotion locator check) still plans through `idx_console_trace_response_links_semantic_revision_exact` with no `sqlite_stat1` on a fresh v89 file and on a v88 file upgraded to v89.
5. Literal current-version pins follow (catalog names V88 -> V89 where they mean "current"; historical steps stay literal); Docs/Architecture/database-layer.md; lanes for any new or unlaned file touched.

## Implementation Notes

AC#1: dropped in ChaChaNotes **v89** (see Decision). v89 follows the v88 pattern.

- **Step.** `chachanotes_v88_to_v89_drop_superseded_response_revision_index.sql` (one `DROP INDEX`) runs through `_migrate_from_v88_to_v89` and the shared `run_sql_version_step`. `_CURRENT_SCHEMA_VERSION = 89`. The ChaChaNotes_DB.py ratchet goes 22573 -> 22580 (+7, dated).
- **Recovery (AC#2).**
  - `CHACHANOTES_V89_SCHEMAS` is each v88 lineage member minus the dropped declaration; an import-time check asserts every v88 member holds it.
  - Also: the policy lineage (v88 kept whole, plus `V89[2:]` and the 88->89 step), the Subscriptions `_SUBSCRIPTIONS_RESPONSE_REVISION_DROP_V89_SCHEMAS` stamps, and the Canvas frozen-catalog list.
  - The restore sandbox admits, only during the v88->v89 step, `DROP INDEX idx_console_trace_response_revision` on `console_trace_response_links`. It also admits the `sqlite_master` delete and `rootpage` update that SQLite's DROP INDEX asks for. A probe found the same request set on Ubuntu 24.04's SQLite 3.45.1 and on the local 3.49.1, with and without auto_vacuum.
- **Gate tests** (`Tests/Backup_Recovery/test_locator_indexes_sandbox.py`, +15). The step may drop its index, and does so with a real `DROP INDEX` statement. Outside the step these are refused, and the v88 locator step may not drop the index. DENY cases: another index on the same table (the v72 exact index included), another table's index, the listed name on another table, temp, trigger-sourced, CREATE, DROP TABLE, and a `sqlite_master.sql` rewrite. Each went red under a matching temporary loosening (all reverted):
  - name-only check: other-table;
  - table-only check: same-table and exact-index;
  - any drop: those plus other-table-index and other-table;
  - no `source is None`: trigger;
  - any catalog column: catalog-sql;
  - gate on whenever migrating: the outside-step catalog-delete and rootpage cases;
  - CREATE admitted: create.

  As in TASK-42517, dropping `database == "main"` alone stays green, because `authorize` refuses non-main databases first. With the step flag disabled, `test_prompt_mixin_restore.py::test_historical_candidate_migrates_without_touching_source` went red for every version (`sqlite_validation_unavailable`). Its new `[88]` case migrates a real v88 candidate, and its version pin now reads `_CURRENT_SCHEMA_VERSION`.
- **Plans (AC#2).** `test_response_revision_queries_keep_the_exact_index_after_v89[fresh|upgraded_from_v88]` (v56 plans file) runs the product's own SQL through a cursor that returns `EXPLAIN QUERY PLAN`. The three statements are `RESPONSE_REVISION_LOCATOR`, `SemanticRevisionCoordinator._reachable_policy_ids` and `ChatPersistenceService._voice_promotion_locator_exists`. With no `sqlite_stat1`, each plans through `idx_console_trace_response_links_semantic_revision_exact`, and the dropped index is gone.
  - Red at base: the partial index is still present.
  - When v89 also drops the exact index: `SCAN console_trace_response_links`.
  - The v72 locator inventory also gains an `upgraded_from_v88` case.
- **Census (AC#2, AC#3).** The row is now `dropped`, a new status in `check_index_plan_pins.py`: a historical migration creates the index and a later `DROP INDEX` under `tldw_chatbook/DB` removes it, so no plan is claimed. The row fails if nothing drops the name. A checker test covers this; it is red at base and green after, and a Python comment does not count as a drop. `EXPECTED_CHACHANOTES_INDEXES` loses the name (the census test is red at base). The v71 plan pin stays as history.
- **Current-version pins.** Moved from V88 to V89 in `test_core_owners.py` (versions, 58 catalogs, `[-14:]`, the step), `test_chachanotes_native76_compatibility.py` and `test_chachanotes_v77_agent_chat_starts_migration.py`. Historical steps stay literal. Docs/Architecture/database-layer.md has a v89 section.
- **Lane.** `Tests/DB/test_check_index_plan_pins.py` ran on no PR lane. It now runs in recovered-suites after the goal attempts file (Tests/DB keeps the bootstrap profile; the recorder counted 0). Serially with the step's flags: 26 tests in 3.4 s at load 29-31. The gate file grew in architecture-guards: 31 tests in 1.3 s.
- **Runs** (load 15-75):
  - touched files, 20/20 green at `-n 4` (332 passed each);
  - broad covering set of 126 files: 3200 passed, 108 failed. All 108 also fail at base except two, which are load flakes outside the schema: the eval re-backup round trip (TTS shutdown `CancelledError`, passes serially on the branch) and a private-sqlite helper exit-vs-protocol race;
  - `Tests/CI`: 2009 passed, 2 skipped;
  - `Tests/Architecture` serially: 1507 passed, 5 skipped;
  - preflight, `ruff --select F` and `ruff_gate.py`: clean;
  - steps: recovered suites 2066 passed, 1 skipped (1284 s); architecture guards 965 passed, 1 skipped (506 s); Dreams riders 1598 passed, 6 skipped (1007 s).

Files: tldw_chatbook/DB/{ChaChaNotes_DB,recovery_core,recovery_core_schema,recovery_operations}.py, tldw_chatbook/DB/migrations/chachanotes_v88_to_v89_drop_superseded_response_revision_index.sql, tldw_chatbook/Backup_Recovery/sqlite_validation.py, scripts/{check_index_plan_pins.py,index_plan_pin_census.tsv}, Docs/Architecture/database-layer.md, Tests/Backup_Recovery/test_{locator_indexes_sandbox,core_owners,prompt_mixin_restore,chachanotes_native76_compatibility}.py, Tests/ChaChaNotesDB/test_index_census.py, Tests/DB/test_{chachanotes_v56_console_trace_query_plans,chachanotes_v72_voice_trace_provenance_migration,chachanotes_v77_agent_chat_starts_migration,check_index_plan_pins}.py, Tests/Architecture/test_module_size_ratchet.py, the workflow and Tests/CI pins.

