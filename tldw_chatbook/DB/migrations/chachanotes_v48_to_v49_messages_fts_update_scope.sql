-- ChaChaNotes v48 -> v49: scope `messages_au` to the columns the FTS index
-- actually depends on, so auxiliary `messages` writes stop rewriting the whole
-- reply into `messages_fts` (task-21128).
--
-- THE DEFECT
-- ----------
-- `messages_au` is declared `AFTER UPDATE ON messages` with no column list, so
-- it fires on EVERY update of the row. `messages_fts` is an external-content
-- fts5 table over ONE column (`content`), but a chat turn now issues three to
-- four auxiliary UPDATEs against the assistant row that touch no indexed
-- column at all -- `update_message_usage_local` (usage_json),
-- `update_message_metadata_local` (metadata_json), the attachment/variant
-- bookkeeping writes, and any `ranking`-only edit. Each one made the trigger
-- delete the reply's doclist and re-tokenize the entire reply back into the
-- index.
--
-- Measured on this schema over one simulated streamed turn with a 400-token
-- reply (scratch probe, task-21128): FOUR index rewrites, `messages_fts_data`
-- growing 55 -> 12,636 bytes. With this step: ONE rewrite, 3,201 bytes. WAL +
-- synchronous=NORMAL, so this is write amplification and page churn rather
-- than an fsync storm -- which is exactly why nothing user-visible ever
-- pointed at it.
--
-- THE COLUMN LIST, AND WHY `OF content` ALONE WOULD BE A DATA-EXPOSURE BUG
-- -----------------------------------------------------------------------
-- The trigger's correct dependency set is "every column the index stores"
-- plus "the column that decides whether the row belongs in the index at all":
--
--   * `content` -- the one column `messages_fts` indexes (asserted against the
--     live schema by test_the_update_of_list_covers_every_fts_relevant_column,
--     so adding a column to the fts5 table without widening this list fails).
--   * `deleted` -- soft delete is `UPDATE messages SET deleted = 1 ...`, which
--     never names `content`. Under `AFTER UPDATE OF content` the trigger would
--     not fire and the tombstoned row would STAY IN THE INDEX: measured, not
--     reasoned -- the scratch matrix for the `OF content` shape returns the
--     tombstoned rowid from a direct `messages_fts MATCH`. Scope it precisely:
--     all SIX production `messages_fts` consumers re-filter on `m.deleted = 0`
--     (ChaChaNotes_DB.py:9131, 10318, 12496, 13935; RAG_Search/simplified/
--     rag_service.py:2371, 2402), so this is an INDEX-LAYER leak -- the deleted
--     message's tokens retained in `messages_fts_data` and reachable by a
--     direct index query -- not a user-visible search leak. It is still a real
--     regression of the task-19567 guarantee (that guarantee is stated at the
--     index, precisely because consumer-side filtering is the thing that made
--     the original trigger bug invisible for so long), and its behavioural
--     witnesses live in Tests/DB/test_fts_soft_delete_index_witness.py, which
--     query the index directly for exactly that reason. Undelete
--     (`SET deleted = 0`) needs `deleted` in the list for the same reason in
--     reverse, as does the chatbook importer's `SET ... deleted = ?` fixup.
--
-- No SQL statement can change `content` without naming it in its SET clause,
-- so a narrowed `UPDATE OF` list cannot leave the index stale; SQLite also
-- fires an `UPDATE OF` trigger whenever a listed column APPEARS in the SET
-- clause, whether or not the value actually changes, which keeps the direction
-- of the remaining error conservative (re-index too often, never too rarely).
--
-- WHAT IS DELIBERATELY UNCHANGED
-- ------------------------------
-- Both v47 guards are preserved byte-for-byte: the delete half keeps
-- `old.deleted = 0` (the task-19567 corruption guard) AND the
-- `messages_fts_docsize` membership test (the task-21100 backfill-window
-- guard -- issuing an fts5 'delete' for a rowid absent from an
-- external-content index poisons its doclists, silently or with `database
-- disk image is malformed` depending on index state). The insert half keeps
-- `new.deleted = 0`, the leak guard. `messages_ai` and `messages_ad` are not
-- touched: an INSERT/DELETE has no column list to narrow.
--
-- WHY A SEPARATE STEP, NOT AN EDITED v47
-- --------------------------------------
-- Same reasoning v47 recorded against v46: a database already stamped 47 (or
-- 48) never replays the v47 step, so an in-place edit could not reach it and
-- two different `messages_au` shapes would ship under one stamp. This step
-- runs for every population and converges them.
--
-- RENUMBERED FROM v47->v48
-- ------------------------
-- Authored as v47->v48; the Console Library policy step
-- (`chachanotes_v47_to_v48_console_library_policy.sql`) merged first and took
-- 48, and schema versions must be CONTIGUOUS, so this one moved to v48->v49
-- rather than leaving a hole no database could cross. That step was re-read
-- rather than assumed before renumbering: it adds two tables, an index, the
-- `messages.assistant_generation_state` column and a rewrite of the four
-- `messages_sync_*` triggers, and it does NOT touch `messages_au`/`_ai`/
-- `_ad` -- so the pre-fix baseline this step replaces is byte-identical to
-- what v47 left behind. Its three new
-- `UPDATE messages SET assistant_generation_state = ...` dispatch writers
-- (Chat/console_dispatch_repository.py) name no indexed column, so they are
-- three MORE per-turn updates that the unscoped trigger would have turned
-- into full index rewrites, and that this step makes free.
--
-- DDL only; O(1); touches no index content, so an in-flight task-21100
-- backfill is unaffected (its cursor is `messages_fts_docsize` membership,
-- which this step does not write). The schema-version bump is a separate
-- rowcount-guarded UPDATE in the runner
-- (`CharactersRAGDB._migrate_from_v48_to_v49`). Bare `CREATE TRIGGER` after an
-- explicit `DROP` keeps the step re-enterable after an interrupted run
-- (task-19553).

DROP TRIGGER IF EXISTS messages_au;

CREATE TRIGGER messages_au
AFTER UPDATE OF content, deleted ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,content)
  SELECT 'delete',old.rowid,old.content
  WHERE old.deleted = 0
    AND EXISTS (SELECT 1 FROM messages_fts_docsize WHERE rowid = old.rowid);

  INSERT INTO messages_fts(rowid,content)
  SELECT new.rowid,new.content
  WHERE new.deleted = 0;
END;
