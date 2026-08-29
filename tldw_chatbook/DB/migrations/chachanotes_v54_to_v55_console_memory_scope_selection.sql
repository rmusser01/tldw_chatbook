-- V54 -> V55: local Console memory scope metadata and branch selections.
--
-- These tables are derived, private conversation state. They intentionally
-- carry no sync columns and do not participate in any sync trigger.

-- A composite child FK requires a parent UNIQUE key even though ``id`` is the
-- primary key. It prevents a scope/selection from pairing a valid globally
-- unique memory with a different conversation.
CREATE UNIQUE INDEX IF NOT EXISTS idx_console_memories_id_conversation
  ON console_conversation_memories(id, conversation_id);

CREATE TABLE IF NOT EXISTS console_conversation_memory_scopes(
  memory_id                   TEXT NOT NULL,
  conversation_id             TEXT NOT NULL
                                  REFERENCES conversations(id)
                                  ON DELETE CASCADE ON UPDATE CASCADE,
  coverage_kind               TEXT NOT NULL
                                  CHECK(coverage_kind IN ('prefix', 'range')),
  origin_kind                 TEXT NOT NULL
                                  CHECK(origin_kind IN ('automatic', 'manual_rewind')),
  selection_anchor_message_id TEXT,
  PRIMARY KEY (memory_id),
  FOREIGN KEY (memory_id, conversation_id)
    REFERENCES console_conversation_memories(id, conversation_id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY (conversation_id, selection_anchor_message_id)
    REFERENCES messages(conversation_id, id)
    ON DELETE RESTRICT ON UPDATE CASCADE,
  CHECK(
    (origin_kind = 'automatic'
      AND coverage_kind = 'prefix'
      AND selection_anchor_message_id IS NULL)
    OR
    (origin_kind = 'manual_rewind'
      AND selection_anchor_message_id IS NOT NULL)
  )
);
CREATE INDEX IF NOT EXISTS idx_console_memory_scopes_conversation_origin
  ON console_conversation_memory_scopes(conversation_id, origin_kind, coverage_kind);

CREATE TABLE IF NOT EXISTS console_conversation_memory_selections(
  sequence              INTEGER PRIMARY KEY AUTOINCREMENT,
  selection_id          TEXT NOT NULL UNIQUE,
  conversation_id       TEXT NOT NULL
                           REFERENCES conversations(id)
                           ON DELETE CASCADE ON UPDATE CASCADE,
  activation_message_id TEXT NOT NULL,
  selected_memory_id    TEXT,
  event_kind            TEXT NOT NULL CHECK(event_kind IN ('select', 'reset')),
  suppresses_legacy     INTEGER NOT NULL DEFAULT 0 CHECK(suppresses_legacy IN (0, 1)),
  created_at            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  revision              INTEGER NOT NULL DEFAULT 1 CHECK(revision > 0),
  active                INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
  FOREIGN KEY (conversation_id, activation_message_id)
    REFERENCES messages(conversation_id, id)
    ON DELETE RESTRICT ON UPDATE CASCADE,
  FOREIGN KEY (selected_memory_id, conversation_id)
    REFERENCES console_conversation_memories(id, conversation_id)
    ON DELETE RESTRICT ON UPDATE CASCADE,
  CHECK(
    (event_kind = 'select' AND selected_memory_id IS NOT NULL)
    OR
    (event_kind = 'reset' AND selected_memory_id IS NULL)
  )
);
CREATE INDEX IF NOT EXISTS idx_console_memory_selections_conversation_active_sequence
  ON console_conversation_memory_selections(conversation_id, active, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_console_memory_selections_activation
  ON console_conversation_memory_selections(conversation_id, activation_message_id);

-- Every historical generated memory has deterministic automatic prefix scope.
-- ``INSERT OR IGNORE`` makes a v54-stamped partial application safe to re-enter.
INSERT OR IGNORE INTO console_conversation_memory_scopes(
  memory_id, conversation_id, coverage_kind, origin_kind,
  selection_anchor_message_id
)
SELECT memory.id, memory.conversation_id, 'prefix', 'automatic', NULL
  FROM console_conversation_memories AS memory
  JOIN conversations AS conversation
    ON conversation.id = memory.conversation_id
 WHERE memory.source_kind = 'generated'
 ORDER BY memory.rowid;

-- Active generated records become non-suppressing select events only when the
-- captured leaf is a live, same-conversation durable message. Rebuild these
-- migration-owned rows on re-entry so a partial application retains the source
-- memory rowid as insertion-order authority. Invalid records deliberately
-- remain inert instead of guessing an activation anchor.
DELETE FROM console_conversation_memory_selections
 WHERE selection_id GLOB 'migration:auto-select:*';

-- A v54-stamped partial application contains only migration-owned rows. Reset
-- the AUTOINCREMENT state after removing all of them so reconstruction assigns
-- the same deterministic sequences as a first migration.
DELETE FROM sqlite_sequence
 WHERE name = 'console_conversation_memory_selections'
   AND NOT EXISTS (
     SELECT 1 FROM console_conversation_memory_selections
   );

INSERT INTO console_conversation_memory_selections(
  selection_id, conversation_id, activation_message_id, selected_memory_id,
  event_kind, suppresses_legacy, created_at, revision, active
)
SELECT 'migration:auto-select:' || memory.id,
       memory.conversation_id,
       memory.captured_leaf_message_id,
       memory.id,
       'select',
       0,
       CURRENT_TIMESTAMP,
       1,
       1
  FROM console_conversation_memories AS memory
  JOIN messages AS leaf
    ON leaf.id = memory.captured_leaf_message_id
   AND leaf.conversation_id = memory.conversation_id
   AND leaf.deleted = 0
 WHERE memory.source_kind = 'generated'
   AND memory.active = 1
   AND memory.captured_leaf_message_id IS NOT NULL
 ORDER BY memory.rowid;
