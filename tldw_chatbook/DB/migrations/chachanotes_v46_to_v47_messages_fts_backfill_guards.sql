-- ChaChaNotes v46 -> v47: guard the `messages` FTS 'delete' halves on index
-- MEMBERSHIP, so the deferred `messages_fts` backfill window is write-safe
-- (task-21100).
--
-- WHY THIS STEP EXISTS
-- --------------------
-- task-21100 moved the v45->v46 `messages_fts` reinsert out of the boot
-- path's version-bump transaction: the v46 step now only issues
-- `'delete-all'` (cheap shadow-table truncation) and the reinsert runs as a
-- chunked, resumable background backfill (`CharactersRAGDB
-- .backfill_messages_fts`). That opens a window in which a LIVE message row
-- is legitimately absent from the index -- and for an external-content FTS5
-- table, issuing the 'delete' command for a rowid that is not in the index
-- corrupts it. This is not theoretical or merely latent: with the v46-shaped
-- trigger (`WHERE old.deleted = 0` alone), a plain content UPDATE of a
-- not-yet-backfilled row raises `sqlite3.DatabaseError: database disk image
-- is malformed` on the UPDATE itself (reproduced in
-- Tests/DB/test_chachanotes_v47_messages_fts_backfill.py). It is the same
-- corruption class task-19567 fixed for tombstoned rows, reopened by the
-- deferral for un-backfilled rows.
--
-- THE GUARD
-- ---------
-- `messages_fts_docsize` is the FTS5 shadow table populated only by real
-- writes into the index (SQLite documents it for exactly this purpose; the
-- Subscriptions backfill reads it in production already). Membership in it
-- answers "is there something to delete" precisely, where `old.deleted = 0`
-- only answers it by proxy through the invariant "indexed == live":
--
--   * `messages_au`'s delete half keeps `old.deleted = 0` (the task-19567
--     corruption guard, and the shape the trigger census in
--     Tests/DB/test_fts_soft_delete_index_witness.py pins) and ADDS the
--     membership test. During the window: updating an un-backfilled live row
--     skips the 'delete' and the insert half indexes the new content -- the
--     row is simply indexed early, and the backfill (keyed on the same
--     docsize membership) then skips it. After the backfill completes the
--     extra condition is identically true for every live row, so steady-state
--     behaviour is byte-for-byte what v46 specified.
--   * `messages_ad` (hard delete) had NO guard at all -- the v4 base shape
--     survived task-19567 because only `*_au` triggers were repaired. It
--     gets the membership test alone: hard-deleting an un-backfilled row must
--     do nothing, and hard-deleting a TOMBSTONED row (never in the index)
--     corrupted the index on the shipped code -- a latent pre-existing bug
--     this step fixes as a side effect. Membership-only (rather than adding
--     `old.deleted = 0`) also means a row that somehow violated the
--     invariant would be cleaned out of the index rather than stranded in it.
--   * `messages_ai` is untouched: a freshly inserted rowid is never in the
--     index, and its `new.deleted = 0` leak guard already has behavioural
--     witnesses.
--
-- WHY A SEPARATE STEP, NOT PART OF THE EDITED v46
-- -----------------------------------------------
-- Databases stamped 46 by the ORIGINAL v46 (full inline rebuild, PR #1974)
-- never replay v46 again, so an edit there could not reach them and two
-- different `messages_au` shapes would ship under the same stamp. This step
-- runs for BOTH populations and converges them on one trigger shape; the
-- version stamp itself keeps the expensive work correctly targeted, because
-- only databases entering through the EDITED v46 ever see the delete-all +
-- deferred reinsert, while already-at-46 databases keep their complete index
-- (this step touches no index content -- asserted by
-- test_v47_leaves_a_complete_index_alone).
--
-- DDL only; O(1). The schema-version bump is a separate rowcount-guarded
-- UPDATE in the runner (`CharactersRAGDB._migrate_from_v46_to_v47`),
-- matching the v45->v46 precedent. Bare `CREATE TRIGGER` after an explicit
-- `DROP` keeps the step re-enterable after an interrupted run (task-19553).

DROP TRIGGER IF EXISTS messages_au;

CREATE TRIGGER messages_au
AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,content)
  SELECT 'delete',old.rowid,old.content
  WHERE old.deleted = 0
    AND EXISTS (SELECT 1 FROM messages_fts_docsize WHERE rowid = old.rowid);

  INSERT INTO messages_fts(rowid,content)
  SELECT new.rowid,new.content
  WHERE new.deleted = 0;
END;

DROP TRIGGER IF EXISTS messages_ad;

CREATE TRIGGER messages_ad
AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,content)
  SELECT 'delete',old.rowid,old.content
  WHERE EXISTS (SELECT 1 FROM messages_fts_docsize WHERE rowid = old.rowid);
END;
