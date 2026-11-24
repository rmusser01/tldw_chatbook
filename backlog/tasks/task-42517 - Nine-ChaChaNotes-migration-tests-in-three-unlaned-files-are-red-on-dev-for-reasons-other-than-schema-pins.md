---
id: TASK-42517
title: "Nine ChaChaNotes migration tests in three unlaned files are red on dev for reasons other than schema pins"
status: To Do
assignee: []
created_date: 2026-11-24 19:23
updated_date: 2026-11-24 20:07
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
- [ ] #1 Each of the nine nodes is classified as product drift or test drift, with the commit that turned it red
- [ ] #2 Product drift is fixed in the product with a test that is red before and green after; test drift is fixed in the test, every other assertion kept
- [ ] #3 `test_hot_message_writers_reserve_the_write_lock_up_front` is settled by behaviour, not source text: the hot writer is shown to take the write lock before its first read, or the gap is fixed
- [ ] #4 The three files run in a PR lane step, timed with that step's flags, with their config reads checked, and none of their nodes runs in a second step
<!-- AC:END -->
