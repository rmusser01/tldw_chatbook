-- Preserve the exact saved input identity without copying its value.
DROP TRIGGER console_trace_events_shape_guard;

CREATE TRIGGER console_trace_events_shape_guard
BEFORE INSERT ON console_trace_events
WHEN NOT (
  (NEW.event_type = 'call_boundary' AND
   NEW.turn_id IS NULL AND NEW.call_id IS NOT NULL AND
   NEW.surface_node_id IS NULL AND NEW.surface_replacement_id IS NULL AND
   NEW.request_header_id IS NULL AND NEW.artifact_id IS NULL AND
   NEW.omission_reason_code IS NULL) OR
  (NEW.event_type = 'turn_boundary' AND
   NEW.turn_id IS NOT NULL AND
   NEW.call_id IS NULL AND NEW.surface_node_id IS NULL AND
   NEW.surface_replacement_id IS NULL AND NEW.request_header_id IS NULL AND
   NEW.semantic_revision_id IS NULL AND NEW.artifact_id IS NULL AND
   NEW.omission_reason_code IS NULL) OR
  (NEW.event_type IN ('call_outcome', 'usage') AND
   NEW.turn_id IS NULL AND NEW.call_id IS NOT NULL AND
   NEW.surface_node_id IS NULL AND NEW.surface_replacement_id IS NULL AND
   NEW.request_header_id IS NULL AND NEW.semantic_revision_id IS NULL AND
   NEW.artifact_id IS NULL AND NEW.omission_reason_code IS NULL) OR
  (NEW.event_type = 'surface_append' AND
   NEW.turn_id IS NULL AND NEW.call_id IS NULL AND
   NEW.surface_node_id IS NOT NULL AND NEW.surface_replacement_id IS NULL AND
   NEW.request_header_id IS NULL AND NEW.semantic_revision_id IS NULL AND
   NEW.artifact_id IS NULL AND NEW.omission_reason_code IS NULL) OR
  (NEW.event_type = 'surface_replace' AND
   NEW.turn_id IS NULL AND NEW.call_id IS NULL AND
   NEW.surface_node_id IS NULL AND NEW.surface_replacement_id IS NOT NULL AND
   NEW.request_header_id IS NULL AND NEW.semantic_revision_id IS NULL AND
   NEW.artifact_id IS NULL AND NEW.omission_reason_code IS NULL) OR
  (NEW.event_type IN ('tool_call', 'tool_result') AND
   NEW.turn_id IS NULL AND NEW.call_id IS NOT NULL AND
   NEW.surface_node_id IS NULL AND NEW.surface_replacement_id IS NULL AND
   NEW.request_header_id IS NULL AND
   ((NEW.semantic_revision_id IS NOT NULL) +
    (NEW.artifact_id IS NOT NULL) +
    (NEW.omission_reason_code IS NOT NULL)) = 1) OR
  (NEW.event_type IN (
     'request_header_selection', 'provider_route_selection'
   ) AND
   NEW.turn_id IS NULL AND NEW.call_id IS NOT NULL AND
   NEW.surface_node_id IS NULL AND NEW.surface_replacement_id IS NULL AND
   NEW.request_header_id IS NOT NULL AND NEW.semantic_revision_id IS NULL AND
   NEW.artifact_id IS NULL AND NEW.omission_reason_code IS NULL) OR
  (NEW.event_type = 'response_selection' AND
   NEW.turn_id IS NULL AND NEW.call_id IS NOT NULL AND
   NEW.surface_node_id IS NULL AND NEW.surface_replacement_id IS NULL AND
   NEW.request_header_id IS NULL AND NEW.omission_reason_code IS NULL AND
   ((NEW.semantic_revision_id IS NOT NULL) +
    (NEW.artifact_id IS NOT NULL)) = 1) OR
  (NEW.event_type = 'gap' AND
   NEW.turn_id IS NULL AND NEW.call_id IS NULL AND
   NEW.surface_node_id IS NULL AND NEW.surface_replacement_id IS NULL AND
   NEW.request_header_id IS NULL AND NEW.semantic_revision_id IS NULL AND
   NEW.artifact_id IS NULL AND NEW.omission_reason_code IS NOT NULL)
)
BEGIN
  SELECT RAISE(ABORT, 'invalid trace event reference shape');
END;

CREATE TRIGGER console_trace_call_boundary_source_guard
BEFORE INSERT ON console_trace_events
WHEN NEW.event_type = 'call_boundary' AND NEW.semantic_revision_id IS NOT NULL
 AND NOT EXISTS (
   SELECT 1 FROM console_trace_calls AS call
   JOIN console_trace_owners AS owner ON owner.owner_id = call.owner_id
   JOIN console_trace_semantic_revisions AS revision
     ON revision.revision_id = NEW.semantic_revision_id
   WHERE call.call_id = NEW.call_id AND call.segment_id = NEW.segment_id
     AND revision.source_message_id = call.turn_id
     AND revision.source_conversation_id = owner.conversation_id
 )
BEGIN
  SELECT RAISE(ABORT, 'trace call boundary source does not own its turn');
END;
