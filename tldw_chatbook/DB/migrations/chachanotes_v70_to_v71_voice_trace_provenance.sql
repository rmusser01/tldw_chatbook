-- ChaChaNotes v70 -> v71: explicit post-dispatch voice trace provenance.
--
-- Ordinary ADR-097 reservations retain their existing default and chronology.
-- A terminal post-dispatch row is admitted only while the repository holds the
-- connection-local authorization for that exact call identity.

ALTER TABLE console_trace_calls
  ADD COLUMN reservation_provenance TEXT NOT NULL
  DEFAULT 'crash_durable_reserved'
  CHECK(reservation_provenance IN (
    'crash_durable_reserved', 'post_dispatch_promoted'
  ));

ALTER TABLE console_trace_calls
  ADD COLUMN import_reason_code TEXT DEFAULT NULL
  CHECK(import_reason_code IS NULL OR
        import_reason_code = 'provisional_voice_promoted');

DROP TRIGGER console_trace_calls_insert_reserved;

CREATE TRIGGER console_trace_calls_insert_reserved
BEFORE INSERT ON console_trace_calls
WHEN NOT (
  (
    NEW.reservation_provenance = 'crash_durable_reserved' AND
    NEW.import_reason_code IS NULL AND
    NEW.state = 'reserved' AND
    NEW.surface_node_id IS NULL AND NEW.request_header_id IS NULL AND
    NEW.provider_name IS NULL AND NEW.model_name IS NULL AND
    NEW.route_identity IS NULL AND NEW.dispatch_started_at IS NULL AND
    NEW.response_started_at IS NULL AND NEW.settled_at IS NULL AND
    NEW.provider_inactive_at IS NULL AND NEW.outcome IS NULL AND
    NEW.usage_json IS NULL AND NEW.integrity_state = 'pending' AND
    NEW.omission_reason_code IS NULL
  ) OR (
    NEW.reservation_provenance = 'post_dispatch_promoted' AND
    NEW.import_reason_code = 'provisional_voice_promoted' AND
    NEW.state IN ('complete', 'stopped', 'error', 'interrupted') AND
    NEW.surface_node_id IS NOT NULL AND NEW.request_header_id IS NOT NULL AND
    NEW.provider_name IS NOT NULL AND NEW.model_name IS NOT NULL AND
    NEW.route_identity IS NOT NULL AND NEW.dispatch_started_at IS NOT NULL AND
    NEW.settled_at IS NOT NULL AND NEW.provider_inactive_at IS NULL AND
    NEW.outcome = NEW.state AND
    NEW.integrity_state = 'complete' AND NEW.omission_reason_code IS NULL AND
    console_voice_trace_import_authorized(NEW.call_id) = 1
  )
)
BEGIN
  SELECT RAISE(
    ABORT,
    'provider call reservation provenance or terminal import authorization is invalid'
  );
END;

CREATE TRIGGER console_trace_calls_provenance_update_guard
BEFORE UPDATE ON console_trace_calls
WHEN
  OLD.reservation_provenance IS NOT NEW.reservation_provenance OR
  OLD.import_reason_code IS NOT NEW.import_reason_code
BEGIN
  SELECT RAISE(ABORT, 'provider call provenance is immutable');
END;

-- Direct completed-pair reconciliation locators. Primary/unique-key searches
-- remain documented by the migration test; every other census locator gets an
-- exact leading-column index so a larger conversation cannot turn one retry
-- into a history scan.
CREATE INDEX IF NOT EXISTS idx_console_dispatch_checkpoints_user_message
  ON console_dispatch_checkpoints(user_message_id);
CREATE INDEX IF NOT EXISTS idx_message_exchanges_message
  ON message_exchanges(message_id);
CREATE INDEX IF NOT EXISTS idx_message_generation_metadata_message
  ON message_generation_metadata(message_id);
CREATE INDEX IF NOT EXISTS idx_message_trajectory_metadata_message
  ON message_trajectory_metadata(message_id);
CREATE INDEX IF NOT EXISTS idx_rag_citation_traces_legacy_message
  ON rag_citation_traces(legacy_message_id);
CREATE INDEX IF NOT EXISTS idx_rag_message_trace_owners_message
  ON rag_message_trace_owners(message_id);
CREATE INDEX IF NOT EXISTS idx_transcript_annotations_message
  ON transcript_annotations(message_id);

CREATE INDEX IF NOT EXISTS idx_console_trace_semantic_revisions_source_message_exact
  ON console_trace_semantic_revisions(source_message_id);
CREATE INDEX IF NOT EXISTS idx_console_trace_semantic_revisions_predecessor_exact
  ON console_trace_semantic_revisions(predecessor_revision_id);
CREATE INDEX IF NOT EXISTS idx_console_trace_events_semantic_revision_exact
  ON console_trace_events(semantic_revision_id);

-- Existing partial indexes cover these locators, but exact full indexes keep
-- the query-plan proof independent of partial-index implication changes.
CREATE INDEX IF NOT EXISTS idx_console_trace_redaction_spans_semantic_revision_exact
  ON console_trace_redaction_spans(semantic_revision_id);
CREATE INDEX IF NOT EXISTS idx_console_trace_response_links_semantic_revision_exact
  ON console_trace_response_links(semantic_revision_id);
CREATE INDEX IF NOT EXISTS idx_console_trace_surface_nodes_semantic_revision_exact
  ON console_trace_surface_nodes(semantic_revision_id);
