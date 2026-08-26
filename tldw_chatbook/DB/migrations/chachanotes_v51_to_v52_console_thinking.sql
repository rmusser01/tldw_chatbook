-- ChaChaNotes v51 -> v52: selected Console thinking evidence and replay policy.
-- Both nullable columns are whole-record sync state. Thinking remains separate
-- from visible content and is deliberately absent from every FTS trigger.

ALTER TABLE messages ADD COLUMN thinking_blocks_json TEXT DEFAULT NULL;
ALTER TABLE conversations ADD COLUMN thinking_history_policy TEXT DEFAULT NULL
    CHECK(
        thinking_history_policy IS NULL OR
        thinking_history_policy IN ('auto', 'include', 'exclude')
    );

-- Pre-v50 tombstones have no content-free base hash. Mark only those already
-- committed before this migration so the reader can reconstruct their prior
-- whole-record hash without treating new hashless tombstones as legacy.
UPDATE sync_log
   SET payload = json_set(
       payload,
       '$.legacy_pre_v50_base_reconstruction',
       json('true')
   )
 WHERE entity = 'messages'
   AND operation = 'delete'
   AND json_valid(payload)
   AND json_extract(payload, '$.deleted') = 1;

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
                     'thinking_blocks_json',NEW.thinking_blocks_json,
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
     OLD.thinking_blocks_json IS NOT NEW.thinking_blocks_json OR
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
                     'thinking_blocks_json',NEW.thinking_blocks_json,
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
                     'thinking_blocks_json',NEW.thinking_blocks_json,
                     'assistant_generation_state',NEW.assistant_generation_state,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

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
                     'metadata',NEW.metadata,'thinking_history_policy',NEW.thinking_history_policy,
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
     OLD.thinking_history_policy IS NOT NEW.thinking_history_policy OR
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
                     'metadata',NEW.metadata,'thinking_history_policy',NEW.thinking_history_policy,
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
                     'metadata',NEW.metadata,'thinking_history_policy',NEW.thinking_history_policy,
                     'title',NEW.title,'rating',NEW.rating,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;
