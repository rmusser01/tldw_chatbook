---
id: TASK-42517
title: "Nine ChaChaNotes migration tests in three unlaned files are red on dev for reasons other than schema pins"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 19:23
updated_date: 2026-11-24 23:08
labels:
- follow-up
- tests
- dev-red
- db
priority: medium
references:
- Tests/DB/test_chachanotes_v47_messages_fts_backfill.py
- Tests/DB/test_chachanotes_v56_console_trace_query_plans.py
- Tests/DB/test_chachanotes_v72_voice_trace_provenance_migration.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-42516's sweep (2026-11-24) ran every `Tests/DB/test_chachanotes_v*.py` file, and every file that imports from the native-76 or v77 test files, on dev 9fa2bf1b1d at `-n 4` through the scratch-profile wrapper. TASK-42516 fixed and laned the reds caused by stale current-schema pins and by the v80-era, v86 mix-in, task-34399 and ADR-207 drift. Nine reds in three files have other causes and were left unfixed. No PR lane runs any of the three files, so nothing reports them.

**`Tests/DB/test_chachanotes_v47_messages_fts_backfill.py`, 5 nodes**
- **A historical fixture runs today's write code.** `test_upgrade_from_v45_defers_the_messages_fts_reinsert` and `test_backfill_resumes_after_interruption_and_matches_the_one_shot_state` fail in `_seed_v45` with `sqlite3.OperationalError: no such table: console_trace_graph_epoch`. Today's message-write path expects a table that a v45 file does not have.
- **The test harness hook never fires.** `test_hot_writer_survives_a_backfill_commit_inside_its_transaction` (:677) and `test_nested_writer_composition_survives_a_backfill_commit` (:745) fail with `AssertionError: the interleave never fired -- test is vacuous` (`assert 'chunk' in {}`).
- **A source-text pin, or possibly a real regression.** `test_hot_message_writers_reserve_the_write_lock_up_front` (:765) asserts that `'self.transaction(immediate=True)'` appears in `add_message`'s source. `add_message` now delegates, so the text is gone. Whether the write lock is still reserved up front is not established.

**`Tests/DB/test_chachanotes_v56_console_trace_query_plans.py`, 1 node: query-plan drift**
- `test_console_trace_query_uses_index_without_statistics[idx_console_trace_response_revision]`: the planner now uses `idx_console_trace_response_links_semantic_revision_exact` (`SEARCH console_trace_response_links USING INDEX … (semantic_revision_id=?)`), not the pinned index.

**`Tests/DB/test_chachanotes_v72_voice_trace_provenance_migration.py`, 3 nodes**
- **An unchanged-catalog check that spans the whole upgrade.** `test_genuine_v70_upgrades_without_rewriting_semantic_rows` (:135) asserts that the canvas schema is unchanged after an upgrade to the current schema. v82 (ADR-239, 01b17108c7) replaced `idx_canvas_revisions_canvas_sequence` and added `uq_canvas_revisions_single_root` on purpose.
- **A historical fixture runs today's code.** `test_genuine_v71_preserves_archived_rows_columns_indexes_and_triggers` (:168) fails with `CharactersRAGDBError: Failed to change conversation archive state`, after `no such column: archived_at`. Today's archive code runs against a v71 file, but that column came with v85.
- **Query-plan census drift.** `test_every_completed_pair_reconciliation_locator_uses_indexed_search` (:377): 8 locators now plan as `SCAN`, the first being `console_hook_continuation_receipts.assistant_message_id`.
  - **It is also an index plan pin, so those pins are not enforced today.** The test is the v72 file's only plan check. Its 12 locator→index pairs are exactly the 12 rows in `scripts/index_plan_pin_census.tsv` that are plan-pinned to this file (rows 27–38, on dev 9fa2bf1b1d and on this branch). One example is `idx_console_dispatch_checkpoints_user_message`; another is `idx_console_trace_response_links_semantic_revision_exact`.
    - While the test is red, no passing plan check stands behind those 12 rows. `scripts/check_index_plan_pins.py` still accepts them, because it checks statically that the file names each index next to `EXPLAIN QUERY PLAN` and a `sqlite_stat1` absence check; it never runs the test.
    - Fixing the test restores those plan pins.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the nine nodes is classified as product drift or test drift, with the commit that turned it red
- [x] #2 Product drift is fixed in the product with a test that is red before and green after; test drift is fixed in the test, every other assertion kept
- [x] #3 `test_hot_message_writers_reserve_the_write_lock_up_front` is settled by behaviour, not source text: the hot writer is shown to take the write lock before its first read, or the gap is fixed
- [x] #4 The three files run in a PR lane step, timed with that step's flags, with their config reads checked, and none of their nodes runs in a second step
<!-- AC:END -->

## Implementation Plan

1. Classify each of the nine nodes with the commit that turned it red.
2. v47 fixtures: tombstone with the v45-era UPDATE instead of today's `soft_delete_message` (which advances the v56 graph epoch); move the two interleave hooks from `CharactersRAGDB.execute_query` (no longer on `add_message`'s path) to the cursor `execute` the writer actually uses; replace the `add_message` source-text pin with a behavioural write-lock probe for every hot writer.
3. v56: re-pin the response-link revision locator to the schema-72 exact index that now wins (evidence: no-stats plan), and record the superseded index's status honestly.
4. v72: carve out exactly the later intentional schema changes (v82 canvas, v85 archive chronology) the whole-upgrade checks see; seed the v71 archive with the v71-era UPDATE; fix the 8 SCAN locators where they live (exact leading-column indexes in a new schema step) with the plan pins and census rows.
5. Lane the three files in one PR lane step (timed with its flags, config reads checked), none twice.

## Implementation Notes

Eight nodes were test drift and are fixed in the tests. One (the v72 locator plans) was product drift and is fixed with schema v88.

**v47 (5 nodes, all turned red by c78f641adf, 2026-08-30: the v56 semantic trace ledger)**
- `test_upgrade_from_v45...` and `test_backfill_resumes...` (test drift). `_seed_v45` tombstoned through today's `soft_delete_message`, which also advances the v56 `console_trace_graph_epoch`, a table a v45 file does not have. The seed now runs the v45-era UPDATE the method wraps (version-checked, rowcount 1). The FTS triggers do the rest, as before.
- Both interleave tests (test drift). `add_message` stopped calling `CharactersRAGDB.execute_query` and now runs `conn.execute` on the transaction cursor, so the patched hook never fired; the "test is vacuous" guard did its job. The hook now sits on the tracked cursor's `execute` (`base_db._QuiescentSQLiteCursor`), keyed on the writer's own connection. The assertions are unchanged.
- `test_hot_message_writers_reserve_the_write_lock_up_front` (AC#3; there was no gap). The source-text pin (`add_message` now delegates) is replaced by a behavioural probe for each of the ten `HOT_MESSAGE_WRITERS`. At the writer connection's first statement inside its transaction, a second connection with `timeout=0` tries `BEGIN IMMEDIATE`, and the test asserts it got "database is locked". So every hot writer holds the write lock before its first read. Negative control: `add_message` temporarily switched to `self.transaction()`. The probe then reported `('SELECT 1 FROM conversations ...', 'write lock was free')`, and the interleave test died with the real `database is locked`. Reverted.

**v56 (1 node): a newer index legitimately wins (test drift, b7dd8e53f2, schema 72).** v72 added the full `idx_console_trace_response_links_semantic_revision_exact` on the same column as v56's partial `idx_console_trace_response_revision`. On a current file with no `sqlite_stat1` the full index wins `WHERE semantic_revision_id = ?`, and also the product's join form (probed). The parametrized pin now names the winning index. A new node pins the v56 index on a genuine v71 file, where it still chose the locator. Its census row says it is superseded from schema 72. No product query plans through it now; whether to drop it is listed as a proposal.

**v72 (3 nodes)**
- `test_genuine_v70_upgrades...` (test drift, 01b17108c7: ADR-239's v82 step). The whole-upgrade canvas check now expects exactly the rewritten `idx_canvas_revisions_canvas_sequence` and the new `uq_canvas_revisions_single_root`. Every other canvas object is still pinned.
- `test_genuine_v71_preserves...` (test drift). The archive seed now runs the v71-era UPDATE. Today's method writes `archived_at`, a column v85 added (dd8d60c9e8). After that seed fix, the whole-upgrade checks needed the later intentional conversations changes carved out, as the test already did for v79/v80. Those are d24c4c04b2's v81 cleanup trigger (the first to turn it red, 2026-10-12), ADR-166's v85 column, index and `conversations_sync_update` rewrite, and 7970fbcada's v86 `conversations_ad` rewrite. The trigger and index text is read from the shipped migration files, and the archived row gains only `archived_at = NULL`.
- `test_every_completed_pair_reconciliation_locator_uses_indexed_search` (product drift). v72 gave every voice-promotion census locator an exact leading-column index. 38467b8fc6 (2026-10-04, hook receipts) and d24c4c04b2 (2026-10-12, response rules and machine follow-ups) added 8 locators without one. `_voice_promotion_locator_exists` runs exactly the test's `SELECT 1 FROM t WHERE c IN (?, ?)`, so each reconciliation scanned 8 tables. Those columns are also `messages` foreign keys, so message deletes scanned them as well. Fix: schema **v88** (`chachanotes_v87_to_v88_message_locator_indexes.sql`), with 8 indexes added through the shared `run_sql_version_step`. Done per the migration README: `EXPECTED_CHACHANOTES_INDEXES`, 8 plan-pinned census rows, the v72 test's `expected_indexes`, the recovery catalog (`CHACHANOTES_V88_*`, the policy lineage, the subscription stamps, and the restore sandbox admitting exactly those 8 `CREATE INDEX` on their tables), and Docs/Architecture/database-layer.md. The locator test now also runs on a file upgraded from v87. The restore test migrates a real v87 candidate through the sandbox. Red/green: base red with 8 `SCAN` locators. With one index removed from the .sql, both the fresh and upgraded cases went red (`'SCAN console_response_rule_drafts'`). With the full .sql both pass.

Pre-existing reds left alone (same at base): `Tests/ChaChaNotesDB/test_index_census.py` (4, earlier migrations' unpinned indexes), and `test_installed_restore_validation.py::test_existing_owned_directory_metadata_uses_verified_node_rollback`, the two portrait migration nodes and `test_voice_promotion_locator_inventory_covers_owned_message_revision_tables`.

Lane (AC#4): the three files run in dreams-tracking's riders step after `Tests/Tool_Packs`, with the other Tests/DB files (bootstrap profile by directory). A `RecoveryRequired.__init__` recorder counted 0 constructions. No other step lists them. Timings, serially with the step's flags at load 14-19: v47 13.7 s (23 tests), v56 1.7 s (24), v72 3.2 s (6). The step totals and the Linux run are in TASK-42513's notes; on Linux all 53 pass. Soak: 20/20 green at `-n 4` (with TASK-42513's files).

Files: Tests/DB/test_chachanotes_v{47_messages_fts_backfill,56_console_trace_query_plans,72_voice_trace_provenance_migration,87_console_trace_child_segments,77_agent_chat_starts_migration}.py, Tests/ChaChaNotesDB/test_index_census.py, Tests/Backup_Recovery/test_{core_owners,prompt_mixin_restore,chachanotes_native76_compatibility}.py, Tests/Architecture/test_module_size_ratchet.py (+7, dated), scripts/index_plan_pin_census.tsv, tldw_chatbook/DB/{ChaChaNotes_DB,recovery_core,recovery_core_schema,recovery_operations}.py, tldw_chatbook/DB/migrations/chachanotes_v87_to_v88_message_locator_indexes.sql, tldw_chatbook/Backup_Recovery/sqlite_validation.py, Docs/Architecture/database-layer.md, the workflow and Tests/CI pins.
