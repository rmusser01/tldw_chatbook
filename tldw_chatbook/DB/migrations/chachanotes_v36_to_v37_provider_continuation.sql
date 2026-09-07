-- Migration: ChaChaNotes V36 to V37 provider continuation ownership.
--
-- The private checkpoint belongs to the exact assistant message/variant row.
-- It participates in ordinary whole-message versioning and sync intent, but
-- remains outside FTS and the local-only metadata/usage paths.

ALTER TABLE messages ADD COLUMN provider_continuation_json TEXT DEFAULT NULL;

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
                     'version',NEW.version,'client_id',NEW.client_id));
END;

/* Undelete restores the ordinary whole-message sync payload. */
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
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

UPDATE db_schema_version
   SET version = 37
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 36;
