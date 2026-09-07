-- Migration: ChaChaNotes V34 to V35 conversation<->dictionary attachment index
-- (task-15469).
--
-- WHY. "Which conversations use this dictionary?" was answered by
-- `SELECT id, title, metadata FROM conversations
--    WHERE deleted = 0 AND metadata LIKE '%active_dictionaries%'`
-- -- a leading-wildcard LIKE, i.e. a full scan of `conversations` plus a JSON
-- parse of every match, on the event loop, on every dictionary row click in
-- Personas (and once per dictionary in `list_dictionaries(include_usage=True)`).
-- These two tables are a derived INDEX over `conversations.metadata` that turns
-- that scan into an indexed lookup.
--
-- WHY TRIGGERS, not maintenance at attach/detach time. The
-- `metadata.active_dictionaries` KEY is only written by
-- `LocalChatDictionaryService._write_active_dictionaries`, but the metadata
-- BLOB is read-modify-written wholesale by several unrelated paths
-- (`Chat/chat_persistence_service.py`, `Chat/rag_scope.py`, any
-- `CharactersRAGDB.update_conversation` caller, sync apply, tests). An index
-- maintained only at attach/detach time would silently rot the first time one
-- of those wrote a metadata blob carrying a different `active_dictionaries`
-- value. Triggers see every writer, including raw SQL, so the index cannot
-- disagree with the column it indexes.
--
-- WHY TWO TABLES. SQLite cannot reproduce Python's `int()` coercion for every
-- JSON element shape: `"1_0"` -> 10 (PEP 515), `"٣"` -> 3, `1e300` -> an
-- exact 300-digit int, a `NaN` literal parses in Python but is invalid JSON to
-- SQLite, and a duplicated JSON key resolves last-wins in Python but first-wins
-- in `json_each`. So `conversation_dictionary_attachments` indexes ONLY
-- unambiguous JSON integers (`json_each.type = 'integer'` AND the value's SQLite
-- storage class is also `integer`, which excludes an int too large for int64).
-- Every other element shape that could still coerce to an int in Python
-- (`real`/`text`/`true`/`false`), a metadata blob holding the marker substring
-- more than once (possible duplicate key), and metadata SQLite cannot parse at
-- all, mark the conversation in `conversation_dictionary_unresolved`; the
-- service re-checks exactly those rows in Python with its unchanged
-- `_active_dictionaries()` predicate, and the unresolved verdict always wins
-- over the index. Element types that can NEVER coerce in Python (`null`,
-- `object`, `array`) are skipped, matching Python. In a database written only
-- by this application the unresolved table stays empty.
--
-- The `metadata LIKE '%active_dictionaries%'` prefilter is carried over from
-- the scan VERBATIM, on purpose: the index must reproduce the old query's
-- results exactly, including its blind spots (a `"active_dictionaries"`
-- escaped key is invisible to that LIKE, so it must stay invisible here too).
--
-- EVERY `json_*` call below reads
-- `CASE WHEN json_valid(x) THEN x ELSE '{}' END`, never the raw column:
-- `json_each`/`json_type` RAISE "malformed JSON" on invalid input, and a
-- raising trigger would fail the conversation write itself. SQLite does not
-- guarantee that a `json_valid(...)` term in the same WHERE clause is
-- evaluated first, so the guard is structural rather than positional.
--
-- WHY NOT the pre-existing `conversation_dictionaries` junction table (V4
-- schema, `ChaChaNotes_DB._FULL_SCHEMA_SQL_V4`): it is dead -- no Python code
-- reads or writes it -- and unusable here. Its `conversation_id` is INTEGER
-- while `conversations.id` is a TEXT UUID, and its `dictionary_id` carries a
-- real FK to `chat_dictionaries(id)`, so a metadata blob naming a since-deleted
-- (or never-existent) dictionary id would fail the FK and take the whole
-- conversation write down with it. A derived index must accept whatever the
-- column it indexes actually contains.
--
-- Deliberately NO sync columns (`client_id`/`version`/`deleted`) and no
-- sync_log triggers: this is derived local state, rebuildable at any time from
-- `conversations.metadata`, which IS synced. Syncing the derivation as well
-- would be redundant and could contradict the column it derives from. Soft
-- deletion is not tracked here either -- readers join `conversations` and
-- filter `deleted = 0`, exactly as the scan did.

CREATE TABLE IF NOT EXISTS conversation_dictionary_attachments(
  conversation_id TEXT    NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE ON UPDATE CASCADE,
  dictionary_id   INTEGER NOT NULL,
  PRIMARY KEY (conversation_id, dictionary_id)
);
CREATE INDEX IF NOT EXISTS idx_conversation_dictionary_attachments_dictionary
  ON conversation_dictionary_attachments(dictionary_id);

CREATE TABLE IF NOT EXISTS conversation_dictionary_unresolved(
  conversation_id TEXT PRIMARY KEY REFERENCES conversations(id)
                        ON DELETE CASCADE ON UPDATE CASCADE
);

DROP TRIGGER IF EXISTS conversation_dictionary_index_ai;
DROP TRIGGER IF EXISTS conversation_dictionary_index_au;
DROP TRIGGER IF EXISTS conversation_dictionary_index_ad;

CREATE TRIGGER conversation_dictionary_index_ai
AFTER INSERT ON conversations
WHEN NEW.metadata LIKE '%active_dictionaries%'
BEGIN
  INSERT OR IGNORE INTO conversation_dictionary_attachments(conversation_id, dictionary_id)
  SELECT NEW.id, CAST(element.value AS INTEGER)
    FROM json_each(
           CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END,
           '$.active_dictionaries'
         ) AS element
   WHERE json_type(CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END) = 'object'
     AND json_type(CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END,
                   '$.active_dictionaries') = 'array'
     AND element.type = 'integer'
     AND typeof(element.value) = 'integer';

  INSERT OR IGNORE INTO conversation_dictionary_unresolved(conversation_id)
  SELECT NEW.id
   WHERE json_valid(NEW.metadata) = 0
      OR (length(NEW.metadata) - length(replace(NEW.metadata, 'active_dictionaries', ''))) / 19 <> 1
      OR EXISTS (
           SELECT 1
             FROM json_each(
                    CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END,
                    '$.active_dictionaries'
                  ) AS element
            WHERE json_type(CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END,
                            '$.active_dictionaries') = 'array'
              AND element.type NOT IN ('null', 'object', 'array')
              AND NOT (element.type = 'integer' AND typeof(element.value) = 'integer')
         );
END;

/* The DELETEs clear BOTH ids, not just OLD.id, because the FK's ON UPDATE
   CASCADE has already run by the time an AFTER UPDATE trigger fires: a
   conversation id change renames these rows to NEW.id first, so a
   `WHERE conversation_id = OLD.id` delete matches nothing and the row's OLD
   dictionary ids survive under the NEW id (reproduced: an UPDATE changing id
   AND metadata from [1,2] to [3] left 1, 2 AND 3 indexed). Clearing both ids
   is also what keeps this correct on a connection running WITHOUT
   `PRAGMA foreign_keys = ON`, where no cascade renames anything. */
CREATE TRIGGER conversation_dictionary_index_au
AFTER UPDATE ON conversations
WHEN NEW.metadata IS NOT OLD.metadata OR NEW.id <> OLD.id
BEGIN
  DELETE FROM conversation_dictionary_attachments
   WHERE conversation_id IN (OLD.id, NEW.id);
  DELETE FROM conversation_dictionary_unresolved
   WHERE conversation_id IN (OLD.id, NEW.id);

  INSERT OR IGNORE INTO conversation_dictionary_attachments(conversation_id, dictionary_id)
  SELECT NEW.id, CAST(element.value AS INTEGER)
    FROM json_each(
           CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END,
           '$.active_dictionaries'
         ) AS element
   WHERE NEW.metadata LIKE '%active_dictionaries%'
     AND json_type(CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END) = 'object'
     AND json_type(CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END,
                   '$.active_dictionaries') = 'array'
     AND element.type = 'integer'
     AND typeof(element.value) = 'integer';

  INSERT OR IGNORE INTO conversation_dictionary_unresolved(conversation_id)
  SELECT NEW.id
   WHERE NEW.metadata LIKE '%active_dictionaries%'
     AND (json_valid(NEW.metadata) = 0
      OR (length(NEW.metadata) - length(replace(NEW.metadata, 'active_dictionaries', ''))) / 19 <> 1
      OR EXISTS (
           SELECT 1
             FROM json_each(
                    CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END,
                    '$.active_dictionaries'
                  ) AS element
            WHERE json_type(CASE WHEN json_valid(NEW.metadata) THEN NEW.metadata ELSE '{}' END,
                            '$.active_dictionaries') = 'array'
              AND element.type NOT IN ('null', 'object', 'array')
              AND NOT (element.type = 'integer' AND typeof(element.value) = 'integer')
         ));
END;

/* The FKs above declare ON DELETE CASCADE and this pool always runs with
   `PRAGMA foreign_keys = ON` (see `_get_thread_connection`), so this trigger is
   belt-and-braces for any connection that ever runs without it -- deleting
   already-cascaded rows is a no-op. */
CREATE TRIGGER conversation_dictionary_index_ad
AFTER DELETE ON conversations BEGIN
  DELETE FROM conversation_dictionary_attachments WHERE conversation_id = OLD.id;
  DELETE FROM conversation_dictionary_unresolved  WHERE conversation_id = OLD.id;
END;

/* Backfill: the same predicates, applied to every pre-existing row. */
INSERT OR IGNORE INTO conversation_dictionary_attachments(conversation_id, dictionary_id)
SELECT conversation.id, CAST(element.value AS INTEGER)
  FROM conversations AS conversation
  JOIN json_each(
         CASE WHEN json_valid(conversation.metadata) THEN conversation.metadata ELSE '{}' END,
         '$.active_dictionaries'
       ) AS element
 WHERE conversation.metadata LIKE '%active_dictionaries%'
   AND json_type(CASE WHEN json_valid(conversation.metadata) THEN conversation.metadata ELSE '{}' END) = 'object'
   AND json_type(CASE WHEN json_valid(conversation.metadata) THEN conversation.metadata ELSE '{}' END,
                 '$.active_dictionaries') = 'array'
   AND element.type = 'integer'
   AND typeof(element.value) = 'integer';

INSERT OR IGNORE INTO conversation_dictionary_unresolved(conversation_id)
SELECT conversation.id
  FROM conversations AS conversation
 WHERE conversation.metadata LIKE '%active_dictionaries%'
   AND (json_valid(conversation.metadata) = 0
    OR (length(conversation.metadata) - length(replace(conversation.metadata, 'active_dictionaries', ''))) / 19 <> 1
    OR EXISTS (
         SELECT 1
           FROM json_each(
                  CASE WHEN json_valid(conversation.metadata) THEN conversation.metadata ELSE '{}' END,
                  '$.active_dictionaries'
                ) AS element
          WHERE json_type(CASE WHEN json_valid(conversation.metadata) THEN conversation.metadata ELSE '{}' END,
                          '$.active_dictionaries') = 'array'
            AND element.type NOT IN ('null', 'object', 'array')
            AND NOT (element.type = 'integer' AND typeof(element.value) = 'integer')
       ));

UPDATE db_schema_version
   SET version = 35
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 34;
