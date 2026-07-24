-- Migration: ChaChaNotes V19 to V20 conversation metadata
-- Adds a nullable `metadata` column to `conversations` for per-conversation
-- runtime metadata (e.g. active_dictionaries), and redefines the
-- conversations_sync_* triggers so edits to the new column are reflected in
-- sync_log. The runner guards the ADD COLUMN with a PRAGMA check (SQLite has
-- no ADD COLUMN IF NOT EXISTS) so replayed/partial migrations are idempotent.

ALTER TABLE conversations ADD COLUMN metadata TEXT;

DROP TRIGGER IF EXISTS conversations_sync_create;
DROP TRIGGER IF EXISTS conversations_sync_update;
DROP TRIGGER IF EXISTS conversations_sync_delete;
DROP TRIGGER IF EXISTS conversations_sync_undelete;

CREATE TRIGGER conversations_sync_create
AFTER INSERT ON conversations BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('conversations',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'root_id',NEW.root_id,'forked_from_message_id',NEW.forked_from_message_id,
                     'parent_conversation_id',NEW.parent_conversation_id,'character_id',NEW.character_id,
                     'assistant_kind',NEW.assistant_kind,'assistant_id',NEW.assistant_id,
                     'persona_memory_mode',NEW.persona_memory_mode,'scope_type',NEW.scope_type,
                     'workspace_id',NEW.workspace_id,'state',NEW.state,'topic_label',NEW.topic_label,
                     'topic_label_source',NEW.topic_label_source,'topic_last_tagged_at',NEW.topic_last_tagged_at,
                     'topic_last_tagged_message_id',NEW.topic_last_tagged_message_id,'cluster_id',NEW.cluster_id,
                     'source',NEW.source,'external_ref',NEW.external_ref,
                     'runtime_backend',NEW.runtime_backend,'discovery_owner',NEW.discovery_owner,
                     'discovery_entity_id',NEW.discovery_entity_id,'system_prompt',NEW.system_prompt,
                     'metadata',NEW.metadata,
                     'title',NEW.title,'rating',NEW.rating,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER conversations_sync_update
AFTER UPDATE ON conversations
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.rating IS NOT NEW.rating OR
     OLD.forked_from_message_id IS NOT NEW.forked_from_message_id OR
     OLD.parent_conversation_id IS NOT NEW.parent_conversation_id OR
     OLD.character_id IS NOT NEW.character_id OR
     OLD.assistant_kind IS NOT NEW.assistant_kind OR
     OLD.assistant_id IS NOT NEW.assistant_id OR
     OLD.persona_memory_mode IS NOT NEW.persona_memory_mode OR
     OLD.scope_type IS NOT NEW.scope_type OR
     OLD.workspace_id IS NOT NEW.workspace_id OR
     OLD.state IS NOT NEW.state OR
     OLD.topic_label IS NOT NEW.topic_label OR
     OLD.topic_label_source IS NOT NEW.topic_label_source OR
     OLD.topic_last_tagged_at IS NOT NEW.topic_last_tagged_at OR
     OLD.topic_last_tagged_message_id IS NOT NEW.topic_last_tagged_message_id OR
     OLD.cluster_id IS NOT NEW.cluster_id OR
     OLD.source IS NOT NEW.source OR
     OLD.external_ref IS NOT NEW.external_ref OR
     OLD.runtime_backend IS NOT NEW.runtime_backend OR
     OLD.discovery_owner IS NOT NEW.discovery_owner OR
     OLD.discovery_entity_id IS NOT NEW.discovery_entity_id OR
     OLD.system_prompt IS NOT NEW.system_prompt OR
     OLD.metadata IS NOT NEW.metadata OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('conversations',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'root_id',NEW.root_id,'forked_from_message_id',NEW.forked_from_message_id,
                     'parent_conversation_id',NEW.parent_conversation_id,'character_id',NEW.character_id,
                     'assistant_kind',NEW.assistant_kind,'assistant_id',NEW.assistant_id,
                     'persona_memory_mode',NEW.persona_memory_mode,'scope_type',NEW.scope_type,
                     'workspace_id',NEW.workspace_id,'state',NEW.state,'topic_label',NEW.topic_label,
                     'topic_label_source',NEW.topic_label_source,'topic_last_tagged_at',NEW.topic_last_tagged_at,
                     'topic_last_tagged_message_id',NEW.topic_last_tagged_message_id,'cluster_id',NEW.cluster_id,
                     'source',NEW.source,'external_ref',NEW.external_ref,
                     'runtime_backend',NEW.runtime_backend,'discovery_owner',NEW.discovery_owner,
                     'discovery_entity_id',NEW.discovery_entity_id,'system_prompt',NEW.system_prompt,
                     'metadata',NEW.metadata,
                     'title',NEW.title,'rating',NEW.rating,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER conversations_sync_delete
AFTER UPDATE ON conversations
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('conversations',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER conversations_sync_undelete
AFTER UPDATE ON conversations
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('conversations',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'root_id',NEW.root_id,'forked_from_message_id',NEW.forked_from_message_id,
                     'parent_conversation_id',NEW.parent_conversation_id,'character_id',NEW.character_id,
                     'assistant_kind',NEW.assistant_kind,'assistant_id',NEW.assistant_id,
                     'persona_memory_mode',NEW.persona_memory_mode,'scope_type',NEW.scope_type,
                     'workspace_id',NEW.workspace_id,'state',NEW.state,'topic_label',NEW.topic_label,
                     'topic_label_source',NEW.topic_label_source,'topic_last_tagged_at',NEW.topic_last_tagged_at,
                     'topic_last_tagged_message_id',NEW.topic_last_tagged_message_id,'cluster_id',NEW.cluster_id,
                     'source',NEW.source,'external_ref',NEW.external_ref,
                     'runtime_backend',NEW.runtime_backend,'discovery_owner',NEW.discovery_owner,
                     'discovery_entity_id',NEW.discovery_entity_id,'system_prompt',NEW.system_prompt,
                     'metadata',NEW.metadata,
                     'title',NEW.title,'rating',NEW.rating,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

UPDATE db_schema_version
   SET version = 20
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 19;
