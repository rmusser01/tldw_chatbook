-- ChaChaNotes v45 -> v46: bound `sync_log` to its reachable frontier
-- (task-19564).
--
-- WHAT WAS WRONG
-- --------------
-- 35 triggers write the COMPLETE row as JSON into `sync_log`, and nothing
-- ever removed a row. Every edit of a message left its previous full text
-- behind forever, and soft-deleting a message, note, character or keyword
-- left that entity's plaintext in the log indefinitely -- so "delete" did not
-- delete. The lane probe for task-19564 soft-deleted a message and read the
-- body straight back out of `sync_log`.
--
-- WHY THE CONTENT COLUMNS ARE NOT RETIRED
-- ---------------------------------------
-- The filing recommended retiring the content columns on the premise that
-- both readers have zero external callers. That premise is stale. It holds
-- for the two LEGACY readers (`get_sync_log_entries`,
-- `get_latest_sync_log_change_id`, test-only in this database), but four more
-- readers have since been added and three of them have live, non-test
-- callers:
--
--   * `read_committed_chat_sync_intent`  <- `ConsoleChatStore
--       .ensure_provider_continuation_durable`, which RAISES when the read
--       returns None, on every provider-continuation checkpoint.
--   * `read_committed_chat_delete_intent`
--   * `list_current_committed_chat_sync_intents` <- `ConsoleChatStore
--       ._reconcile_restored_chat_sync_intents`, on every conversation
--       restore with Sync v2 configured.
--
-- Each of them compares the sync_log payload to the live `messages` row
-- FIELD BY FIELD (`intent_payload != expected_intent`); the payload is the
-- proof that the exact row was committed. Dropping `content` from the
-- payload would make every comparison fail and silently disable Sync v2 while
-- turning continuation checkpoints into hard errors. So the log stays; what
-- goes is everything in it that no reader can reach.
--
-- THE RETENTION RULE (1 of 2): VERSIONED
-- --------------------------------------
-- Six of the nine writers are versioned and a `sync_log` row of theirs is
-- reachable only through a JOIN to its live entity row on
-- `entity_id` AND `version`:
--
--   messages, live     -> versions {v, v-1}.  v is the frontier the intent
--                         readers join to; v-1 is what
--                         `_previous_committed_chat_payload_hash` reads for
--                         the base hash.
--   messages, deleted  -> version {v} only. `read_committed_chat_sync_intent`
--                         refuses deleted rows and the delete path never asks
--                         for a base hash, so every content-bearing row below
--                         the tombstone is unreachable.
--   every other entity -> version {v} only. Nothing reads them at all.
--   orphaned (entity row gone) -> nothing is reachable.
--
-- Everything outside that frontier is deleted here, once, for databases that
-- already have it, and the triggers below keep it that way from now on.
-- Pruning only ever removes rows at versions STRICTLY BELOW the frontier, so
-- it cannot perturb `list_current_committed_chat_sync_intents`'s
-- `1 = (SELECT COUNT(*) ...)` single-intent check, which counts rows at the
-- current version.
--
-- THE RETENTION RULE (2 of 2): LATEST-ONLY
-- ----------------------------------------
-- The other three writers -- `chat_dictionaries`, `world_books` and
-- `world_book_entries` -- cannot use the version rule above, and a first cut
-- of this migration left them uncovered. Qodo's review of PR #1974
-- independently reached the same finding, so they are covered here:
--
--   * `chat_dictionaries` has a `last_modified` timestamp trigger, whose
--     nested UPDATE fires the `sync_update` emitter again. When the emitted
--     `last_modified` differs from the one the outer statement wrote, that
--     produces a FULL-PAYLOAD `update` row at the tombstone's OWN version, so
--     `version < NEW.version` leaves the deleted dictionary's plaintext
--     behind. Reproduced directly against a v45 database.
--   * `world_book_entries` has no `version` column and no `deleted` column --
--     every one of its sync rows is written at the literal version 1 -- and
--     its only delete path is a hard `DELETE`
--     (`world_book_manager.delete_world_book_entry`, wired to Personas > entry
--     delete), which orphans every content row it wrote. A version rule is
--     entirely inert for it.
--   * `world_books` has the same shape as `chat_dictionaries` and takes the
--     same rule, for uniformity rather than because its own timestamp trigger
--     exists (it does not).
--
-- So their rule is anchored to the log row itself rather than to a version:
-- at most ONE content-bearing row survives per entity -- the most recently
-- emitted -- and only while the entity is live. Content-free `delete`
-- tombstones are kept for the unversioned entity as its only record that the
-- delete happened.
--
-- WHY THAT IS ORDER-INDEPENDENT
-- -----------------------------
-- SQLite does not define the firing order of same-kind triggers, and a single
-- soft delete really does fire two emitters (the `sync_delete` tombstone and,
-- via the timestamp trigger, `sync_update`) whose relative order was observed
-- to FLIP when the emitters are recreated in a different order. A rule that
-- depends on that order would be unsound, so this one does not:
--
--   * The rule fires AFTER INSERT ON sync_log -- i.e. once per emission,
--     whoever emitted it -- and its predicate reads only (a) the base table's
--     state, which every AFTER trigger sees already final for the statement,
--     and (b) `change_id`, which is the table maximum at insert time.
--   * Each firing re-establishes the same post-condition for that entity:
--     content-bearing rows = {the row just inserted} if the entity is live,
--     {} otherwise. The last emission of the statement therefore fixes the
--     final state, and *which* trigger is last does not change it: if the
--     entity ends deleted the answer is {} either way, and if it ends live the
--     two candidate payloads are the same row rendered twice.
--   * The hard-delete companion below deliberately excludes `operation =
--     'delete'` for `world_book_entries`. Deleting everything there WOULD be
--     order-dependent: fired before the sibling tombstone emitter it removes
--     nothing, fired after it removes the tombstone.
--
-- Verified by running every scenario under six permutations of the emitters'
-- creation order, with a control run proving the permutation really does flip
-- the emission order (see task-19564's notes).
--
-- WHAT THIS DOES NOT DO
-- ---------------------
-- Soft-deleting a CONVERSATION does not soft-delete its messages (they stay
-- `deleted = 0` and come back on restore), so their frontier row is retained
-- -- exactly as `messages.content` itself is retained. After this migration
-- `sync_log` never holds entity text that the entity table does not; it is a
-- bounded frontier, not an unbounded shadow copy. Removing the plaintext for
-- live rows as well needs the payload to carry a content HASH instead, which
-- is a format change to a live sync proof; it is recommended as a follow-up in
-- task-19564's notes, not attempted here.
--
-- One pre-existing defect is deliberately NOT fixed here: hard-deleting a
-- `world_books` row cascades to `world_book_entries`, whose `sync_delete`
-- emitter reads `(SELECT client_id FROM world_books WHERE id = OLD.world_book_id)`
-- -- already gone during the cascade -- and raises `NOT NULL constraint
-- failed: sync_log.client_id`. No shipped path hard-deletes a world book
-- (`delete_world_book` is a soft delete), and retention neither causes nor
-- worsens it; it is recorded in task-19564's notes rather than fixed inside a
-- retention change.
--
-- ALSO IN THIS STEP (task-19567)
-- -----------------------------
-- Section 3 repairs the three FTS `*_au` triggers whose DELETE half was never
-- guarded. That is a different bug, but it is the same schema step and the
-- same guard family, and it was found by the direct-index witnesses written
-- for task-19567 -- one of them could not be written at all until it was
-- fixed. See that section's header for the reproduction.
--
-- The retention triggers are named `sync_log_prune_<entity>`, deliberately NOT
-- `<entity>_sync_log_prune`: `_` is a single-character wildcard in SQL LIKE,
-- so the latter matches `LIKE '<entity>_sync_%'` -- a namespace three tests
-- assert the exact membership of, because it belongs to the four triggers that
-- WRITE the log.
--
-- DDL and DML only. The schema-version bump is a separate rowcount-guarded
-- UPDATE in the runner (`CharactersRAGDB._migrate_from_v45_to_v46`), matching
-- the v43->v44 precedent. Bare `CREATE TRIGGER` statements are deliberate:
-- `_drop_superseded_trigger` drops a same-named trigger first, which makes
-- this step re-enterable after an interrupted run (task-19553).

/*------------------------------------------------------------------
  1. Retention triggers -- soft-delete and edit paths
------------------------------------------------------------------*/

/* messages keep a two-version frontier: the current version (joined by the
   intent readers) and the one below it (the base-hash lookup). A tombstoned
   message keeps only its tombstone, which carries no content. */
CREATE TRIGGER sync_log_prune_messages
AFTER UPDATE ON messages
WHEN NEW.version > 1
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'messages'
     AND entity_id = NEW.id
     AND version < (CASE WHEN NEW.deleted = 1
                         THEN NEW.version
                         ELSE NEW.version - 1 END);
END;

/* conversations/notes/character_cards/keywords/keyword_collections have no
   reader at all, so only the current version is retained. */
CREATE TRIGGER sync_log_prune_conversations
AFTER UPDATE ON conversations
WHEN NEW.version > 1
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'conversations'
     AND entity_id = NEW.id
     AND version < NEW.version;
END;

CREATE TRIGGER sync_log_prune_notes
AFTER UPDATE ON notes
WHEN NEW.version > 1
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'notes'
     AND entity_id = NEW.id
     AND version < NEW.version;
END;

CREATE TRIGGER sync_log_prune_character_cards
AFTER UPDATE ON character_cards
WHEN NEW.version > 1
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'character_cards'
     AND entity_id = CAST(NEW.id AS TEXT)
     AND version < NEW.version;
END;

CREATE TRIGGER sync_log_prune_keywords
AFTER UPDATE ON keywords
WHEN NEW.version > 1
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'keywords'
     AND entity_id = CAST(NEW.id AS TEXT)
     AND version < NEW.version;
END;

CREATE TRIGGER sync_log_prune_keyword_collections
AFTER UPDATE ON keyword_collections
WHEN NEW.version > 1
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'keyword_collections'
     AND entity_id = CAST(NEW.id AS TEXT)
     AND version < NEW.version;
END;

/*------------------------------------------------------------------
  2. Retention triggers -- hard delete
------------------------------------------------------------------*/
/* A hard DELETE emits no sync_log row of its own (there has never been a
   hard-delete sync trigger), and every reader joins to the entity row, so
   once the row is gone nothing in the log for it is reachable. */

CREATE TRIGGER sync_log_prune_messages_hard
AFTER DELETE ON messages
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'messages' AND entity_id = OLD.id;
END;

CREATE TRIGGER sync_log_prune_conversations_hard
AFTER DELETE ON conversations
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'conversations' AND entity_id = OLD.id;
END;

CREATE TRIGGER sync_log_prune_notes_hard
AFTER DELETE ON notes
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'notes' AND entity_id = OLD.id;
END;

CREATE TRIGGER sync_log_prune_character_cards_hard
AFTER DELETE ON character_cards
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'character_cards' AND entity_id = CAST(OLD.id AS TEXT);
END;

CREATE TRIGGER sync_log_prune_keywords_hard
AFTER DELETE ON keywords
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'keywords' AND entity_id = CAST(OLD.id AS TEXT);
END;

CREATE TRIGGER sync_log_prune_keyword_collections_hard
AFTER DELETE ON keyword_collections
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'keyword_collections' AND entity_id = CAST(OLD.id AS TEXT);
END;

/*------------------------------------------------------------------
  2b. Retention triggers -- the three latest-only writers
------------------------------------------------------------------*/
/* See "THE RETENTION RULE (2 of 2)" and "WHY THAT IS ORDER-INDEPENDENT" in
   this file's header. These fire on the LOG row, not on the entity row, so a
   statement whose emitters fire in an undefined order still converges: the
   last emission fixes the state, and the state it fixes does not depend on
   which emitter that was. */

CREATE TRIGGER sync_log_prune_chat_dictionaries
AFTER INSERT ON sync_log
WHEN NEW.entity = 'chat_dictionaries'
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'chat_dictionaries'
     AND entity_id = NEW.entity_id
     AND (version < NEW.version
          OR (operation <> 'delete'
              AND (change_id < NEW.change_id
                   OR NOT EXISTS (SELECT 1 FROM chat_dictionaries AS src
                                   WHERE CAST(src.id AS TEXT) = NEW.entity_id
                                     AND src.deleted = 0))));
END;

CREATE TRIGGER sync_log_prune_world_books
AFTER INSERT ON sync_log
WHEN NEW.entity = 'world_books'
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'world_books'
     AND entity_id = NEW.entity_id
     AND (version < NEW.version
          OR (operation <> 'delete'
              AND (change_id < NEW.change_id
                   OR NOT EXISTS (SELECT 1 FROM world_books AS src
                                   WHERE CAST(src.id AS TEXT) = NEW.entity_id
                                     AND src.deleted = 0))));
END;

/* No version clause: every world_book_entries sync row is written at the
   literal version 1, and the table has no `deleted` column, so "live" is
   "the row still exists". Tombstones are kept -- for a hard-delete-only
   entity the tombstone is the only record that the delete happened, and it
   carries ids only. */
CREATE TRIGGER sync_log_prune_world_book_entries
AFTER INSERT ON sync_log
WHEN NEW.entity = 'world_book_entries'
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'world_book_entries'
     AND entity_id = NEW.entity_id
     AND operation <> 'delete'
     AND (change_id < NEW.change_id
          OR NOT EXISTS (SELECT 1 FROM world_book_entries AS src
                          WHERE CAST(src.id AS TEXT) = NEW.entity_id));
END;

/* Hard-delete companions. chat_dictionaries/world_books emit nothing on a
   hard DELETE, so nothing would fire the log-side rule and the rows would be
   orphaned -- these clear them exactly as the six above do. */
CREATE TRIGGER sync_log_prune_chat_dictionaries_hard
AFTER DELETE ON chat_dictionaries
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'chat_dictionaries' AND entity_id = CAST(OLD.id AS TEXT);
END;

CREATE TRIGGER sync_log_prune_world_books_hard
AFTER DELETE ON world_books
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'world_books' AND entity_id = CAST(OLD.id AS TEXT);
END;

/* Unlike the two above, this one is defence in depth rather than the
   load-bearing path: world_book_entries DOES emit a tombstone on hard delete,
   and that emission is itself what fires the log-side rule, so the content is
   already gone without this trigger (the behavioural test still passes with
   it removed). It exists so the guarantee does not silently depend on that
   emitter staying unconditional.
   `operation <> 'delete'` here IS load-bearing: the tombstone comes from a
   sibling AFTER DELETE trigger whose order relative to this one is undefined,
   so deleting everything would remove the tombstone when this fires second
   and keep it when this fires first -- an order-dependent result. Excluding
   tombstones makes both orders converge on {tombstone}. */
CREATE TRIGGER sync_log_prune_world_book_entries_hard
AFTER DELETE ON world_book_entries
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'world_book_entries'
     AND entity_id = CAST(OLD.id AS TEXT)
     AND operation <> 'delete';
END;

/*------------------------------------------------------------------
  3. FTS update triggers whose DELETE half was never guarded (task-19567)
------------------------------------------------------------------*/
-- Found by the direct-index witnesses added for task-19567. Five of the eight
-- soft-deletable FTS `*_au` triggers guard their delete half with
-- `WHERE old.deleted = 0` (`notes_au` was repaired earlier for exactly this
-- reason -- see `_ensure_notes_fts_update_trigger_handles_undelete`); three --
-- `messages_au`, `keyword_collections_au`, `world_books_au` -- issued the FTS
-- `'delete'` unconditionally. Issuing it for a row that is NOT in an
-- external-content index corrupts that index.
--
-- `keyword_collections_au` was live-reachable: `add_keyword_collection` on a
-- name whose row is soft-deleted goes through `_add_generic_item`'s undelete
-- UPDATE and raised `sqlite3.DatabaseError: database disk image is malformed`
-- on the shipped code. `messages_au` and `world_books_au` have no undelete
-- path today, so they were latent -- fixed anyway, because "latent" here means
-- "one restore API away", and a census test now refuses to let the shape back
-- in.
--
-- The insert halves are unchanged; the `WHERE new.deleted = 0` guard on them
-- is the one Lane 5 mutated away with 475 tests still green, and it is now
-- pinned behaviourally by `Tests/DB/test_fts_soft_delete_index_witness.py`.

DROP TRIGGER IF EXISTS messages_au;

CREATE TRIGGER messages_au
AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,content)
  SELECT 'delete',old.rowid,old.content
  WHERE old.deleted = 0;

  INSERT INTO messages_fts(rowid,content)
  SELECT new.rowid,new.content
  WHERE new.deleted = 0;
END;

DROP TRIGGER IF EXISTS keyword_collections_au;

CREATE TRIGGER keyword_collections_au
AFTER UPDATE ON keyword_collections BEGIN
  INSERT INTO keyword_collections_fts(keyword_collections_fts,rowid,name)
  SELECT 'delete',old.id,old.name
  WHERE old.deleted = 0;

  INSERT INTO keyword_collections_fts(rowid,name)
  SELECT new.id,new.name
  WHERE new.deleted = 0;
END;

DROP TRIGGER IF EXISTS world_books_au;

CREATE TRIGGER world_books_au
AFTER UPDATE ON world_books BEGIN
  INSERT INTO world_books_fts(world_books_fts, rowid, name, description)
  SELECT 'delete', OLD.id, OLD.name, OLD.description
  WHERE OLD.deleted = 0;

  INSERT INTO world_books_fts(rowid, name, description)
  SELECT NEW.id, NEW.name, NEW.description
  WHERE NEW.deleted = 0;
END;

/* Repair whatever the unguarded halves already did, and bring both indexes
   into line with the guarantee for rows that predate it. NOT the FTS5
   `'rebuild'` command: that re-derives from the base table with no `deleted`
   filter and would index every tombstoned row, reintroducing exactly the leak
   these triggers exist to prevent. */
INSERT INTO messages_fts(messages_fts) VALUES('delete-all');

INSERT INTO messages_fts(rowid,content)
SELECT rowid, content FROM messages WHERE deleted = 0;

INSERT INTO keyword_collections_fts(keyword_collections_fts) VALUES('delete-all');

INSERT INTO keyword_collections_fts(rowid,name)
SELECT id, name FROM keyword_collections WHERE deleted = 0;

INSERT INTO world_books_fts(world_books_fts) VALUES('delete-all');

INSERT INTO world_books_fts(rowid, name, description)
SELECT id, name, description FROM world_books WHERE deleted = 0;

/*------------------------------------------------------------------
  4. One-time purge of the backlog every existing database already has
------------------------------------------------------------------*/
/* Without this the fix would only help new installs, leaving the plaintext
   in place for everyone who already has it. */

DELETE FROM sync_log
 WHERE entity = 'messages'
   AND change_id IN (
        SELECT s.change_id
          FROM sync_log AS s
          LEFT JOIN messages AS m ON m.id = s.entity_id
         WHERE s.entity = 'messages'
           AND (
                m.id IS NULL
                OR s.version < (CASE WHEN m.deleted = 1
                                     THEN m.version
                                     ELSE m.version - 1 END)
           )
   );

DELETE FROM sync_log
 WHERE entity = 'conversations'
   AND change_id IN (
        SELECT s.change_id
          FROM sync_log AS s
          LEFT JOIN conversations AS c ON c.id = s.entity_id
         WHERE s.entity = 'conversations'
           AND (c.id IS NULL OR s.version < c.version)
   );

DELETE FROM sync_log
 WHERE entity = 'notes'
   AND change_id IN (
        SELECT s.change_id
          FROM sync_log AS s
          LEFT JOIN notes AS n ON n.id = s.entity_id
         WHERE s.entity = 'notes'
           AND (n.id IS NULL OR s.version < n.version)
   );

DELETE FROM sync_log
 WHERE entity = 'character_cards'
   AND change_id IN (
        SELECT s.change_id
          FROM sync_log AS s
          LEFT JOIN character_cards AS cc
                 ON CAST(cc.id AS TEXT) = s.entity_id
         WHERE s.entity = 'character_cards'
           AND (cc.id IS NULL OR s.version < cc.version)
   );

DELETE FROM sync_log
 WHERE entity = 'keywords'
   AND change_id IN (
        SELECT s.change_id
          FROM sync_log AS s
          LEFT JOIN keywords AS k ON CAST(k.id AS TEXT) = s.entity_id
         WHERE s.entity = 'keywords'
           AND (k.id IS NULL OR s.version < k.version)
   );

DELETE FROM sync_log
 WHERE entity = 'keyword_collections'
   AND change_id IN (
        SELECT s.change_id
          FROM sync_log AS s
          LEFT JOIN keyword_collections AS kc
                 ON CAST(kc.id AS TEXT) = s.entity_id
         WHERE s.entity = 'keyword_collections'
           AND (kc.id IS NULL OR s.version < kc.version)
   );

/* The latest-only three. Same converged state the triggers above maintain:
   orphans keep nothing (except an unversioned entity's tombstones), a live
   entity keeps its newest content row, a soft-deleted one keeps only its
   tombstone. */

DELETE FROM sync_log
 WHERE entity = 'chat_dictionaries'
   AND change_id IN (
        SELECT s.change_id
          FROM sync_log AS s
          LEFT JOIN chat_dictionaries AS cd
                 ON CAST(cd.id AS TEXT) = s.entity_id
         WHERE s.entity = 'chat_dictionaries'
           AND (
                cd.id IS NULL
                OR s.version < cd.version
                OR (s.operation <> 'delete'
                    AND (cd.deleted = 1
                         OR s.change_id < (
                                SELECT MAX(s2.change_id)
                                  FROM sync_log AS s2
                                 WHERE s2.entity = 'chat_dictionaries'
                                   AND s2.entity_id = s.entity_id
                                   AND s2.operation <> 'delete')))
           )
   );

DELETE FROM sync_log
 WHERE entity = 'world_books'
   AND change_id IN (
        SELECT s.change_id
          FROM sync_log AS s
          LEFT JOIN world_books AS wb
                 ON CAST(wb.id AS TEXT) = s.entity_id
         WHERE s.entity = 'world_books'
           AND (
                wb.id IS NULL
                OR s.version < wb.version
                OR (s.operation <> 'delete'
                    AND (wb.deleted = 1
                         OR s.change_id < (
                                SELECT MAX(s2.change_id)
                                  FROM sync_log AS s2
                                 WHERE s2.entity = 'world_books'
                                   AND s2.entity_id = s.entity_id
                                   AND s2.operation <> 'delete')))
           )
   );

DELETE FROM sync_log
 WHERE entity = 'world_book_entries'
   AND change_id IN (
        SELECT s.change_id
          FROM sync_log AS s
          LEFT JOIN world_book_entries AS wbe
                 ON CAST(wbe.id AS TEXT) = s.entity_id
         WHERE s.entity = 'world_book_entries'
           AND s.operation <> 'delete'
           AND (
                wbe.id IS NULL
                OR s.change_id < (
                       SELECT MAX(s2.change_id)
                         FROM sync_log AS s2
                        WHERE s2.entity = 'world_book_entries'
                          AND s2.entity_id = s.entity_id
                          AND s2.operation <> 'delete')
           )
   );
