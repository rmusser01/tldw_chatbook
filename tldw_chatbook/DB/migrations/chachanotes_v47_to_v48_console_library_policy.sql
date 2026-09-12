-- ChaChaNotes v47 -> v48: device-local Console Library policy and dispatch
-- recovery ownership. Policy/checkpoint rows deliberately have no sync fields,
-- sync triggers, FTS projection, or export mirror. The assistant generation
-- state is whole-message data and therefore replaces all final Sync-v1 message
-- triggers in the same transaction that adds the column.

CREATE TABLE console_conversation_library_policy (
    conversation_id TEXT PRIMARY KEY
        REFERENCES conversations(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    schema_version INTEGER NOT NULL DEFAULT 1
        CHECK(schema_version > 0),
    auto_retrieve_on_send INTEGER NOT NULL DEFAULT 0
        CHECK(auto_retrieve_on_send IN (0, 1)),
    assistant_library_access INTEGER NOT NULL DEFAULT 0
        CHECK(assistant_library_access IN (0, 1)),
    policy_revision INTEGER NOT NULL DEFAULT 1
        CHECK(policy_revision > 0),
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE messages ADD COLUMN assistant_generation_state TEXT
    CHECK(
        assistant_generation_state IS NULL OR
        assistant_generation_state IN (
            'accepted', 'dispatch_started', 'continuation_active',
            'complete', 'stopped', 'failed', 'discarded'
        )
    );

CREATE TABLE console_dispatch_checkpoints (
    assistant_message_id TEXT PRIMARY KEY
        REFERENCES messages(id) ON DELETE CASCADE,
    user_message_id TEXT NOT NULL
        REFERENCES messages(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE CASCADE,
    schema_version INTEGER NOT NULL DEFAULT 1
        CHECK(schema_version > 0),
    preparation_id TEXT NOT NULL UNIQUE,
    attempt_id TEXT NOT NULL,
    state TEXT NOT NULL
        CHECK(state IN ('accepted', 'dispatch_started')),
    checkpoint_revision INTEGER NOT NULL DEFAULT 1
        CHECK(checkpoint_revision > 0),
    user_message_version INTEGER NOT NULL
        CHECK(user_message_version > 0),
    assistant_message_version INTEGER NOT NULL
        CHECK(assistant_message_version > 0),
    origin TEXT NOT NULL CHECK(origin IN ('manual', 'queued')),
    queue_entry_id TEXT,
    frozen_authority_json TEXT NOT NULL,
    resolved_destination_json TEXT NOT NULL,
    reconstructability_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_console_dispatch_checkpoint_conversation
    ON console_dispatch_checkpoints(conversation_id);

DROP TRIGGER IF EXISTS messages_sync_create;

DROP TRIGGER IF EXISTS messages_sync_update;

DROP TRIGGER IF EXISTS messages_sync_delete;

DROP TRIGGER IF EXISTS messages_sync_undelete;

CREATE TRIGGER messages_sync_create
AFTER INSERT ON messages BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,
                     'image_mime_type',NEW.image_mime_type,
                     'provider_continuation_json',NEW.provider_continuation_json,
                     'assistant_generation_state',NEW.assistant_generation_state,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER messages_sync_update
AFTER UPDATE ON messages
WHEN OLD.deleted = NEW.deleted AND (
     OLD.content IS NOT NEW.content OR
     OLD.image_data IS NOT NEW.image_data OR
     OLD.image_mime_type IS NOT NEW.image_mime_type OR
     OLD.provider_continuation_json IS NOT NEW.provider_continuation_json OR
     OLD.assistant_generation_state IS NOT NEW.assistant_generation_state OR
     OLD.ranking IS NOT NEW.ranking OR
     OLD.parent_message_id IS NOT NEW.parent_message_id OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,
                     'image_mime_type',NEW.image_mime_type,
                     'provider_continuation_json',NEW.provider_continuation_json,
                     'assistant_generation_state',NEW.assistant_generation_state,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER messages_sync_delete
AFTER UPDATE ON messages
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'assistant_generation_state',NEW.assistant_generation_state,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER messages_sync_undelete
AFTER UPDATE ON messages
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,
                     'image_mime_type',NEW.image_mime_type,
                     'provider_continuation_json',NEW.provider_continuation_json,
                     'assistant_generation_state',NEW.assistant_generation_state,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;
