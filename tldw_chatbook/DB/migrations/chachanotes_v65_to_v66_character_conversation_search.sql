CREATE TABLE character_conversation_search_generations(
  generation_id TEXT PRIMARY KEY NOT NULL,
  data_authority_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('building', 'ready', 'failed')),
  policy_version INTEGER NOT NULL CHECK(policy_version > 0),
  source_revision INTEGER NOT NULL CHECK(source_revision >= 0),
  processed_conversations INTEGER NOT NULL DEFAULT 0 CHECK(processed_conversations >= 0),
  error_code TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TEXT,
  lease_expires_at TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX character_conversation_search_generations_authority_status
  ON character_conversation_search_generations(data_authority_id, status);
CREATE UNIQUE INDEX character_conversation_search_one_ready_generation
  ON character_conversation_search_generations(data_authority_id)
  WHERE status = 'ready';

CREATE TABLE character_conversation_search_revision(
  singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
  data_revision INTEGER NOT NULL CHECK(data_revision >= 0),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO character_conversation_search_revision(singleton_id, data_revision)
VALUES(1, 0);
CREATE TRIGGER character_conversation_search_revision_no_delete
BEFORE DELETE ON character_conversation_search_revision
BEGIN
  SELECT RAISE(ABORT, 'character conversation search revision is required');
END;

CREATE TABLE character_conversation_search_state(
  singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
  data_authority_id TEXT,
  active_policy_version INTEGER,
  activated INTEGER NOT NULL DEFAULT 0 CHECK(activated IN (0, 1)),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO character_conversation_search_state(singleton_id) VALUES(1);

CREATE TABLE character_conversation_search_dirty(
  conversation_id TEXT PRIMARY KEY NOT NULL,
  data_authority_id TEXT NOT NULL,
  source_revision INTEGER NOT NULL CHECK(source_revision >= 0),
  enqueued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX character_conversation_search_dirty_authority_revision
  ON character_conversation_search_dirty(data_authority_id, source_revision);

CREATE TABLE character_conversation_search_documents(
  document_id INTEGER PRIMARY KEY AUTOINCREMENT,
  data_authority_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  character_id INTEGER NOT NULL REFERENCES character_cards(id) ON DELETE CASCADE,
  character_label TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  eligibility_digest TEXT NOT NULL,
  validated_eligibility_digest TEXT NOT NULL,
  source_revision INTEGER NOT NULL CHECK(source_revision >= 0),
  generation_id TEXT NOT NULL
    REFERENCES character_conversation_search_generations(generation_id) ON DELETE CASCADE,
  UNIQUE(data_authority_id, generation_id, conversation_id)
);
CREATE INDEX character_conversation_search_documents_character
  ON character_conversation_search_documents(
    data_authority_id, character_id, generation_id, conversation_id
  );
CREATE INDEX character_conversation_search_documents_revision
  ON character_conversation_search_documents(data_authority_id, source_revision);

CREATE VIRTUAL TABLE character_conversation_fts USING fts5(
  character_label,
  title,
  body,
  content='character_conversation_search_documents',
  content_rowid='document_id'
);
CREATE TRIGGER character_conversation_search_documents_ai
AFTER INSERT ON character_conversation_search_documents BEGIN
  INSERT INTO character_conversation_fts(rowid, character_label, title, body)
  VALUES(new.document_id, new.character_label, new.title, new.body);
END;
CREATE TRIGGER character_conversation_search_documents_au
AFTER UPDATE ON character_conversation_search_documents
WHEN old.character_label IS NOT new.character_label
  OR old.title IS NOT new.title OR old.body IS NOT new.body
BEGIN
  INSERT INTO character_conversation_fts(
    character_conversation_fts, rowid, character_label, title, body
  ) VALUES('delete', old.document_id, old.character_label, old.title, old.body);
  INSERT INTO character_conversation_fts(rowid, character_label, title, body)
  VALUES(new.document_id, new.character_label, new.title, new.body);
END;
CREATE TRIGGER character_conversation_search_documents_ad
AFTER DELETE ON character_conversation_search_documents BEGIN
  INSERT INTO character_conversation_fts(
    character_conversation_fts, rowid, character_label, title, body
  ) VALUES('delete', old.document_id, old.character_label, old.title, old.body);
END;

CREATE TRIGGER character_conversation_search_messages_ai
AFTER INSERT ON messages BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT new.conversation_id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_messages_au
AFTER UPDATE ON messages BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT new.conversation_id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_messages_ad
AFTER DELETE ON messages BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT old.conversation_id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_conversations_ai
AFTER INSERT ON conversations BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT new.id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_conversations_au
AFTER UPDATE ON conversations BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT new.id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_conversations_ad
AFTER DELETE ON conversations BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT old.id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_characters_ai
AFTER INSERT ON character_cards BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT c.id, state.data_authority_id, revision.data_revision
    FROM conversations AS c,
         character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE c.character_id = new.id
     AND state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_characters_au
AFTER UPDATE ON character_cards BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT c.id, state.data_authority_id, revision.data_revision
    FROM conversations AS c,
         character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE c.character_id = new.id
     AND state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_characters_ad
AFTER DELETE ON character_cards BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT c.id, state.data_authority_id, revision.data_revision
    FROM conversations AS c,
         character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE c.character_id = old.id
     AND state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
