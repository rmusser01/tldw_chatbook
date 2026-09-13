-- ChaChaNotes v56 -> v57: fail-closed semantic message mutation boundary.
--
-- The guard function is connection-local Python state registered only by
-- CharactersRAGDB connection setup. A raw or generic BaseDB connection does
-- not have the function, so referenced-source mutations fail closed.

CREATE TRIGGER console_trace_semantic_revisions_retirement_guard
BEFORE UPDATE OF live_message_id, live_locator_retired_at
ON console_trace_semantic_revisions
WHEN OLD.live_message_id IS NOT NULL AND (
  NEW.live_message_id IS NOT NULL OR
  NEW.live_locator_retired_at IS NULL OR
  console_semantic_mutation_authorized(OLD.live_message_id, 'locator_retire') <> 1
)
BEGIN
  SELECT RAISE(ABORT, 'semantic mutation authorization required for locator retirement');
END;

CREATE TRIGGER messages_semantic_update_guard
BEFORE UPDATE OF
  id, conversation_id, parent_message_id, sender, role, content, image_data,
  image_mime_type, provider_continuation_json, thinking_blocks_json,
  assistant_generation_state
ON messages
WHEN EXISTS (
  SELECT 1
    FROM console_trace_semantic_revisions AS revision
   WHERE revision.live_message_id = OLD.id
) AND (
  OLD.id IS NOT NEW.id OR
  OLD.conversation_id IS NOT NEW.conversation_id OR
  OLD.parent_message_id IS NOT NEW.parent_message_id OR
  OLD.sender IS NOT NEW.sender OR
  OLD.role IS NOT NEW.role OR
  OLD.content IS NOT NEW.content OR
  OLD.image_data IS NOT NEW.image_data OR
  OLD.image_mime_type IS NOT NEW.image_mime_type OR
  OLD.provider_continuation_json IS NOT NEW.provider_continuation_json OR
  OLD.thinking_blocks_json IS NOT NEW.thinking_blocks_json OR
  OLD.assistant_generation_state IS NOT NEW.assistant_generation_state
) AND console_semantic_mutation_authorized(OLD.id, 'message_update') <> 1
BEGIN
  SELECT RAISE(ABORT, 'semantic mutation authorization required for message update');
END;

CREATE TRIGGER messages_semantic_delete_guard
BEFORE DELETE ON messages
WHEN EXISTS (
  SELECT 1
    FROM console_trace_semantic_revisions AS revision
   WHERE revision.live_message_id = OLD.id
) AND console_semantic_mutation_authorized(OLD.id, 'message_delete') <> 1
BEGIN
  SELECT RAISE(ABORT, 'semantic mutation authorization required for message delete');
END;

CREATE TRIGGER message_attachments_semantic_insert_guard
BEFORE INSERT ON message_attachments
WHEN EXISTS (
  SELECT 1
    FROM console_trace_semantic_revisions AS revision
   WHERE revision.live_message_id = NEW.message_id
) AND console_semantic_mutation_authorized(NEW.message_id, 'attachment_insert') <> 1
BEGIN
  SELECT RAISE(ABORT, 'semantic mutation authorization required for attachment insert');
END;

CREATE TRIGGER message_attachments_semantic_update_guard
BEFORE UPDATE ON message_attachments
WHEN (
  EXISTS (
    SELECT 1
      FROM console_trace_semantic_revisions AS revision
     WHERE revision.live_message_id = OLD.message_id
  ) OR EXISTS (
    SELECT 1
      FROM console_trace_semantic_revisions AS revision
     WHERE revision.live_message_id = NEW.message_id
  )
) AND (
  console_semantic_mutation_authorized(OLD.message_id, 'attachment_update') <> 1 OR
  OLD.message_id IS NOT NEW.message_id
)
BEGIN
  SELECT RAISE(ABORT, 'semantic mutation authorization required for attachment update');
END;

CREATE TRIGGER message_attachments_semantic_delete_guard
BEFORE DELETE ON message_attachments
WHEN EXISTS (
  SELECT 1
    FROM console_trace_semantic_revisions AS revision
   WHERE revision.live_message_id = OLD.message_id
) AND console_semantic_mutation_authorized(OLD.message_id, 'attachment_delete') <> 1
BEGIN
  SELECT RAISE(ABORT, 'semantic mutation authorization required for attachment delete');
END;

-- Reachable-policy discovery starts at one referenced revision and walks
-- forward through surface successors. These covering indexes keep that work
-- proportional to the reachable surface graph rather than calls × depth.
CREATE INDEX idx_console_trace_surface_nodes_revision
  ON console_trace_surface_nodes(semantic_revision_id, node_id)
  WHERE semantic_revision_id IS NOT NULL;
CREATE INDEX idx_console_trace_calls_surface_policy
  ON console_trace_calls(surface_node_id, policy_id)
  WHERE surface_node_id IS NOT NULL;
