-- ADR-224 (TASK-34427): make the conversation keyset, flashcard due, and
-- character-list query families sargable by normalizing their ordering
-- columns to the ADR-173 canonical UTC shape (YYYY-MM-DDTHH:MM:SS.mmmZ,
-- fixed width, so raw TEXT ordering equals chronological ordering).
--
-- Legacy shapes on disk, all parsed by julianday():
--   'YYYY-MM-DD HH:MM:SS'     SQLite CURRENT_TIMESTAMP (space separator)
--   'YYYY-MM-DDTHH:MM:SS[.f]+00:00'  pre-TASK-32803.2 .isoformat() writes
--   'YYYY-MM-DDTHH:MM:SS.mmmZ'       already canonical (GLOB guard skips)
--
-- Idempotent by construction: canonical rows fail the GLOB guard and
-- unparseable rows (julianday IS NULL) are left untouched rather than
-- silently written to NULL on a NOT NULL column.

-- 1. conversations.last_modified: normalize WITHOUT firing the two AFTER
--    UPDATE triggers whose side effects would be wrong for a format-only
--    change. conversations_sync_update copies NEW.last_modified into
--    sync_log (a storage-format change is not a content change for sync
--    consumers); character_conversation_search_conversations_au bumps the
--    search projection revision and stamps CURRENT_TIMESTAMP dirty rows
--    (the searchable content did not change, and the wall-clock stamps make
--    the migration non-deterministic). Both are dropped and recreated
--    verbatim (the v70->v71 precedent). The FTS mirror triggers fire but
--    re-insert identical titles, which is a no-op.
DROP TRIGGER conversations_sync_update;
DROP TRIGGER character_conversation_search_conversations_au;
UPDATE conversations
   SET last_modified = strftime('%Y-%m-%dT%H:%M:%fZ', julianday(last_modified))
 WHERE last_modified NOT GLOB '????-??-??T??:??:??.???Z'
   AND julianday(last_modified) IS NOT NULL;
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

-- 2. flashcards.next_review: NULL means "never reviewed, due now" and
--    stays NULL.
UPDATE flashcards
   SET next_review = strftime('%Y-%m-%dT%H:%M:%fZ', julianday(next_review))
 WHERE next_review IS NOT NULL
   AND next_review NOT GLOB '????-??-??T??:??:??.???Z'
   AND julianday(next_review) IS NOT NULL;

-- 3. Indexes for the raw (sargable) rewrites.
--    Keyset: character_id equality prefix, then the (last_modified DESC,
--    id DESC) order the keyset pages by. idx_conv_char (character_id alone)
--    cannot serve the order and idx_conversations_archive is archived-first.
CREATE INDEX IF NOT EXISTS idx_conv_char_lm
    ON conversations(character_id, last_modified DESC, id DESC);
--    Character browse: name in NOCASE order over exactly the rows the
--    browse queries admit (non-deleted, user-visible). SQLite >= 3.9 allows
--    the deterministic json_extract in the partial-index WHERE; the shipped
--    3.49.1 accepts it (feature-detected in Tests, ADR-224 fallback of a
--    stored is_user_visible column was not needed).
CREATE INDEX IF NOT EXISTS idx_character_cards_visible_name
    ON character_cards(name COLLATE NOCASE)
    WHERE deleted = 0
      AND json_extract(CASE WHEN json_valid(extensions) THEN extensions ELSE '{}' END,
                       '$.actor_pack_persona_portrait_owner') IS NULL;
