-- ADR-147: local archive state independent of workflow/deletion. Applied by the
-- cursor-based migration under its existing transaction (no executescript).
ALTER TABLE conversations ADD COLUMN archived INTEGER NOT NULL DEFAULT 0
    CHECK (archived IN (0, 1));
CREATE INDEX idx_conversations_archive
    ON conversations(archived, deleted, last_modified DESC, id DESC);
-- Archive-only writes retain local optimistic versions without sync events.
-- Any sync payload change still emits an update, including mixed writes.
DROP TRIGGER conversations_sync_update;
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
     OLD.root_id IS NOT NEW.root_id OR
     OLD.created_at IS NOT NEW.created_at OR
     OLD.client_id IS NOT NEW.client_id OR
     (OLD.archived IS NEW.archived AND (
         OLD.last_modified IS NOT NEW.last_modified OR
         OLD.version IS NOT NEW.version)))
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

-- The local archive version may be newer than the latest shared payload.
-- Retain that payload until a replacement sync event is actually emitted.
DROP TRIGGER sync_log_prune_conversations;
CREATE TRIGGER sync_log_prune_conversations
AFTER INSERT ON sync_log
WHEN NEW.entity = 'conversations'
BEGIN
  DELETE FROM sync_log
   WHERE entity = 'conversations'
     AND entity_id = NEW.entity_id
     AND version < (
         SELECT MAX(frontier.version) FROM sync_log AS frontier
          WHERE frontier.entity = 'conversations'
            AND frontier.entity_id = NEW.entity_id);
END;

UPDATE db_schema_version SET version = 71
    WHERE schema_name = 'rag_char_chat_schema' AND version = 70;
