---
id: TASK-36510
title: Index the eight Console message-locator columns the voice promotion scans
status: Done
assignee:
- '@claude'
created_date: 2026-10-23 22:44
updated_date: '2026-11-03 23:11'
labels:
- follow-up
- performance
- database
priority: medium
references:
- Tests/Performance/test_speculative_voice_latency.py
- Packaging/speculative_voice_history_gate.py
- scripts/index_plan_pin_census.tsv
dependencies:
- TASK-36390
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36390 and its review (2026-10-23). The speculative-voice history gate promotes a completed pair by probing every message-locator column with `SELECT 1 ... WHERE <column> IN (?, ?)`. On a database with no `sqlite_stat1` (no DB module here runs `ANALYZE`, so this is every user's database), eight of those columns have no index that leads with them, and each probe scans its table:
- `console_hook_continuation_receipts`: `assistant_message_id` and `parent_assistant_message_id`
- `console_machine_followup_receipts`: `assistant_message_id` and `parent_assistant_message_id`
- `console_response_rule_drafts.message_id`, `console_response_rule_fixtures.message_id` and `console_response_rule_validations.message_id`
- `console_response_rule_assessments.message_id`. Its only index, `idx_response_rule_assessments_source`, is `(conversation_id, message_id)`.

The measured fresh-commit median paired delta is 5.1 ms, against the gate's 2.0 ms limit. A probe that indexed seven of the columns brought the gate's scan count to 0 and the delta to 0.06 ms.

The gate under-reports. It runs `ANALYZE` after seeding and leaves `conversation_id` all-NULL, so the planner skip-scans the assessments index, and the gate counts 14 scans (7 x 2) instead of 16.

TASK-36390 marks the history node `xfail(strict=True)` on this task.

The columns are `REFERENCES messages(id) ON DELETE CASCADE` children, so the indexes probably also help message deletes. Nobody has measured that.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On a fresh schema with no sqlite_stat1, each of the eight columns is searched through an index rather than scanned, by a schema migration
- [x] #2 Each new index has a query-plan pin captured with sqlite_stat1 absent
- [x] #3 The voice history gate reads its query plans before any ANALYZE, so a locator column without a leading index fails it
- [x] #4 The history node's strict xfail is removed and the node passes
<!-- AC:END -->

## Implementation Plan

1. Owner decision 2026-11-03: fold the indexes into #3345's v86->v87 step (`chachanotes_v86_to_v87_mobile_consolidation.sql`), no v88. A fresh database walks the same chain, and the recovery catalogs (`recovery_core_schema.CHACHANOTES_V87_SCHEMAS`, the shared-file Subscriptions lineage, `CHACHANOTES_MOBILE_V86_TO_V87_SQL`) derive from that file through `mobile_import_authority.CHAT_MIGRATION`, so one SQL edit reaches every builder.
2. Choose each index from a stats-free probe of the gate's exact `SELECT 1 ... WHERE <col> IN (?, ?) LIMIT 1`; reshape any shape the planner ignores. Re-plan every other query the new indexes could be chosen for (rule repository, dispatch repository, mobile row matches, message/conversation FK cleanup).
3. Let the restore sandbox's migration authorizer (`Backup_Recovery/sqlite_validation.py`) create the new indexes during a v86->v87 restore migration.
4. Pins: `EXPECTED_CHACHANOTES_INDEXES`, `scripts/index_plan_pin_census.tsv`, and stats-free plan assertions in `Tests/DB/test_response_rules_migration.py`.
5. Gate: drop the gate's `ANALYZE` so its plans are read on the stats-free schema users have; remove the strict xfail on the history node.
6. Run Tests/DB, Tests/ChaChaNotesDB, the restore/core-owner tests, the voice latency file, the plan-pin checker and preflight through run.sh.

## Implementation Notes

Folded into #3345's v86->v87 step, per the owner's 2026-11-03 decision; no v88.

**Indexes.** Eight plain single-column indexes, appended to `chachanotes_v86_to_v87_mobile_consolidation.sql`: `idx_hook_continuation_receipts_{assistant,parent}_message`, `idx_machine_followup_receipts_{assistant,parent}_message`, and `idx_response_rule_{assessments,drafts,fixtures,validations}_message`. A fresh database walks the same chain, and every recovery catalog (`recovery_core_schema.CHACHANOTES_V87_SCHEMAS`, `CHACHANOTES_MOBILE_V86_TO_V87_SQL`, the shared-file Subscriptions v87 lineage in `recovery_operations`) is derived from that file through `mobile_import_authority.CHAT_MIGRATION`. So the SQL file is the only schema edit, and `ChaChaNotes_DB.py` and `recovery_core_schema.py` needed none.

**Plan evidence (SQLite 3.49.1, fresh v87 file, `sqlite_stat1` absent, the gate's `SELECT 1 FROM t WHERE c IN (?, ?) LIMIT 1`).** Before the change, all eight showed `SCAN <table>`, and for assessments `SCAN ... USING COVERING INDEX idx_response_rule_assessments_source`. With the textbook shape, each one shows `SEARCH <table> USING COVERING INDEX <new index> (<column>=?)`. No reshape was needed. Other queries the indexes could be chosen for, re-planned before and after:
- Unchanged: the rule-repository reads and deletes, the dispatch chain read, the six `mobile_native_rows._ROW_MATCHES` probes (still on their primary keys), and the three existing pins.
- `followups WHERE conversation_id=? AND assistant_message_id=?` moved from `idx_machine_followup_receipts_conversation` to the new assistant index. It is still a SEARCH, and on a more selective column.
- `DELETE FROM messages WHERE id=?`: its six FK child searches went from SCAN to SEARCH.

**Restore path.** The restore sandbox migrates a v86 backup with this SQL. Its authorizer (`Backup_Recovery/sqlite_validation.py`) now lets the chachanotes owner create, and REINDEX, exactly these eight indexes. Red first: `test_mobile_import_schema.py` `[chat.db-...-81-87]` and `test_native_mixin_v86_migrates_to_mobile_v87_without_losing_attempts[recovery]` failed with `sqlite_validation_unavailable`. A temporary deny log showed `SQLITE_REINDEX` (27) on the new index. With the change, 17/17 passed.

**Gate.** `Packaging/speculative_voice_history_gate.py` no longer runs `ANALYZE`, and `_query_plans` raises if `sqlite_stat1` exists. Temporarily re-adding the `ANALYZE` raised `RuntimeError: history gate plans must be read without sqlite_stat1`. The strict xfail on `test_completed_pair_history_gate_meets_large_history_thresholds` is removed.

**Red/green evidence.**
- v87 SQL reverted, branch otherwise intact:
  - the 8 new `test_rule_lookup_indexes_without_statistics` params and the history node all failed (9 failed, 5 passed);
  - the gate reported `scan_count 16`, which is the 16 the description predicted once `ANALYZE` is gone, and a fresh-commit median paired delta of 7.44 ms;
  - `check_index_plan_pins.py` without the census rows: `FAIL: 8 index(es) are created but absent`.
- AC #3 negative control, with seven indexes and the assessments one omitted:
  - the HEAD gate (with `ANALYZE`) reported `passed True scan_count 0`, planning `SEARCH ... idx_response_rule_assessments_source (ANY(conversation_id) AND message_id=?)`;
  - the new gate reported `passed False scan_count 2`.
- Green: the gate reported `scan_count 0`, and the fresh-commit median paired delta fell from 7.44 ms to -0.08 ms (p95 delta 1.77 ms, large p95 7.1 ms).
- `Tests/DB/test_chachanotes_v72_voice_trace_provenance_migration.py::test_every_completed_pair_reconciliation_locator_uses_indexed_search` was red at the base head (`8 more items, first extra item: console_hook_continuation_receipts.assistant_message_id: SCAN`) and is green with the change.

**Timing caveat (machine load 22-42 on 18 CPUs, up to six agents).** The history node's fresh-commit thresholds are wall-clock. Its `scan_count` was 0 in every run.
- `Tests/Performance/test_speculative_voice_latency.py` passed in 10 of 11 runs at load 22-37.
- Direct gate runs at load 27-42 failed 7 of 11 times, always on the fresh-commit p95 paired delta (11.8-24 ms against 10 ms).
- The `-n 4` DB batch at load ~42 failed the node once, the same way.

The tails were symmetric: baseline maxima reached 918 ms, and the low deltas went down to -785 ms. So this is load noise, not locator cost. Medians stayed at 0.06-0.7 ms.

**Covering runs (all through run.sh, `-n 4`).**
- `Tests/DB Tests/ChaChaNotesDB Tests/Performance/test_speculative_voice_latency.py`: branch 303 failed / 3455 passed, base head c8ffa11e5e 303 failed / 3446 passed / 1 xfailed.
  - Branch-only: the history node (the load-noise failure above).
  - Base-only: the v72 locator test this change fixes.
  - The other 301 are identical and pre-existing on #3345's head: a stale `EXPECTED_TABLE_COLUMNS`, `_db_diagnostic_ref`, `_pin_sqlite_source` signature, and unpinned indexes/triggers from earlier steps.
- The 38 `Tests/Backup_Recovery` files that touch validation, catalogs or restore migration: branch 92 failed + 15 errors / 1331 passed, base 93 failed + 16 errors / 1329 passed.
  - Branch-only: `test_mobile_source_runs24.py::test_actual_source_version_capture_identity_and_integrated_plan[24|25]`, a 30 s restriction deadline (`sqlite_resource_limit`) on a 103-146 s test at load ~40. Rerun alone on the branch: 2 passed.
- `scripts/check_index_plan_pins.py`: OK, 338 declared and 121 plan-pinned.
- `./scripts/preflight.sh`: the only failure is the production diagnostic inventory (the `custom_tokenizers.py` Utils->Chat move from #3345's 321e06b19e). Its report is identical at the base head.

**Files.**
- `tldw_chatbook/DB/migrations/chachanotes_v86_to_v87_mobile_consolidation.sql`
- `tldw_chatbook/Backup_Recovery/sqlite_validation.py`
- `Packaging/speculative_voice_history_gate.py`
- `Tests/Performance/test_speculative_voice_latency.py`
- `Tests/DB/test_response_rules_migration.py`
- `Tests/ChaChaNotesDB/test_index_census.py`
- `scripts/index_plan_pin_census.tsv`
- `backlog/docs/lessons-testing-evidence.md`
