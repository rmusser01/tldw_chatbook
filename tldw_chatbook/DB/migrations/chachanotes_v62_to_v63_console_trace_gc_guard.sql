-- ChaChaNotes v62 -> v63: epoch-safe semantic trace graph collection.
--
-- The mark ledger contains opaque identities and counters only. Trace payload
-- deletion remains fail-closed unless the connection-local collector grant is
-- active for the exact maintenance lease and marked graph epoch.

CREATE TABLE console_trace_gc_runs(
  request_id TEXT PRIMARY KEY NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK(status IN ('pending', 'marked', 'completed', 'stale_epoch')),
  operation_kind TEXT NOT NULL DEFAULT 'collect'
    CHECK(operation_kind IN ('collect', 'purge_conversation')),
  target_conversation_id TEXT DEFAULT NULL,
  target_owner_id TEXT DEFAULT NULL,
  target_root_segment_id TEXT DEFAULT NULL,
  marked_epoch INTEGER DEFAULT NULL CHECK(marked_epoch IS NULL OR marked_epoch >= 0),
  swept_epoch INTEGER DEFAULT NULL CHECK(swept_epoch IS NULL OR swept_epoch >= 0),
  result_json TEXT DEFAULT NULL
    CHECK(result_json IS NULL OR (json_valid(result_json) AND json_type(result_json) = 'object')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (operation_kind = 'collect'
      AND target_conversation_id IS NULL
      AND target_owner_id IS NULL
      AND target_root_segment_id IS NULL)
    OR
    (operation_kind = 'purge_conversation'
      AND length(target_conversation_id) > 0
      AND length(target_owner_id) > 0
      AND length(target_root_segment_id) > 0)
  )
);

CREATE TABLE console_trace_gc_marks(
  request_id TEXT NOT NULL REFERENCES console_trace_gc_runs(request_id),
  entity_kind TEXT NOT NULL CHECK(length(entity_kind) > 0),
  entity_id TEXT NOT NULL CHECK(length(entity_id) > 0),
  marked_epoch INTEGER NOT NULL CHECK(marked_epoch >= 0),
  PRIMARY KEY(request_id, entity_kind, entity_id)
);

CREATE TABLE console_trace_gc_segment_scopes(
  request_id TEXT NOT NULL REFERENCES console_trace_gc_runs(request_id),
  segment_id TEXT NOT NULL REFERENCES console_trace_segments(segment_id),
  through_sequence INTEGER DEFAULT NULL
    CHECK(through_sequence IS NULL OR through_sequence >= 0),
  PRIMARY KEY(request_id, segment_id)
);

CREATE TABLE console_trace_retention_roots(
  retention_id TEXT PRIMARY KEY NOT NULL,
  entity_kind TEXT NOT NULL
    CHECK(entity_kind IN ('owner', 'call', 'revision', 'artifact')),
  entity_id TEXT NOT NULL CHECK(length(entity_id) > 0),
  retain_until TEXT NOT NULL CHECK(length(retain_until) > 0),
  reason_code TEXT NOT NULL CHECK(length(reason_code) > 0),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(entity_kind, entity_id, reason_code)
);

CREATE INDEX idx_console_trace_retention_expiry
  ON console_trace_retention_roots(
    julianday(retain_until), entity_kind, entity_id
  );

CREATE TRIGGER console_trace_calls_open_root_epoch
AFTER UPDATE OF state ON console_trace_calls
WHEN (
  OLD.state IN ('reserved', 'dispatch_started', 'response_started')
) <> (
  NEW.state IN ('reserved', 'dispatch_started', 'response_started')
)
BEGIN
  UPDATE console_trace_graph_epoch
     SET epoch = epoch + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
END;

CREATE TRIGGER console_trace_retention_roots_insert_epoch
AFTER INSERT ON console_trace_retention_roots
BEGIN
  UPDATE console_trace_graph_epoch
     SET epoch = epoch + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
END;

CREATE TRIGGER console_trace_retention_roots_delete_epoch
AFTER DELETE ON console_trace_retention_roots
WHEN julianday(OLD.retain_until) > julianday('now')
BEGIN
  UPDATE console_trace_graph_epoch
     SET epoch = epoch + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
END;

CREATE TRIGGER console_trace_migration_root_epoch
AFTER UPDATE OF status ON console_trace_migration_state
WHEN (OLD.status <> 'logical_complete') <> (NEW.status <> 'logical_complete')
BEGIN
  UPDATE console_trace_graph_epoch
     SET epoch = epoch + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
END;

DROP TRIGGER console_trace_artifacts_no_delete;
DROP TRIGGER console_trace_calls_no_delete;
DROP TRIGGER console_trace_events_no_delete;
DROP TRIGGER console_trace_header_components_no_delete;
DROP TRIGGER console_trace_owners_no_delete;
DROP TRIGGER console_trace_policies_no_delete;
DROP TRIGGER console_trace_redaction_spans_no_delete;
DROP TRIGGER console_trace_request_headers_no_delete;
DROP TRIGGER console_trace_response_links_no_delete;
DROP TRIGGER console_trace_revision_bindings_no_delete;
DROP TRIGGER console_trace_segments_no_delete;
DROP TRIGGER console_trace_semantic_revisions_no_delete;
DROP TRIGGER console_trace_surface_nodes_no_delete;
DROP TRIGGER console_trace_surface_replacements_no_delete;

CREATE TRIGGER console_trace_artifacts_no_delete
BEFORE DELETE ON console_trace_artifacts
WHEN console_trace_gc_delete_authorized('console_trace_artifacts') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_calls_no_delete
BEFORE DELETE ON console_trace_calls
WHEN console_trace_gc_delete_authorized('console_trace_calls') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_events_no_delete
BEFORE DELETE ON console_trace_events
WHEN console_trace_gc_delete_authorized('console_trace_events') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_header_components_no_delete
BEFORE DELETE ON console_trace_header_components
WHEN console_trace_gc_delete_authorized('console_trace_header_components') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_owners_no_delete
BEFORE DELETE ON console_trace_owners
WHEN console_trace_gc_delete_authorized('console_trace_owners') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_policies_no_delete
BEFORE DELETE ON console_trace_policies
WHEN console_trace_gc_delete_authorized('console_trace_policies') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_redaction_spans_no_delete
BEFORE DELETE ON console_trace_redaction_spans
WHEN console_trace_gc_delete_authorized('console_trace_redaction_spans') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_request_headers_no_delete
BEFORE DELETE ON console_trace_request_headers
WHEN console_trace_gc_delete_authorized('console_trace_request_headers') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_response_links_no_delete
BEFORE DELETE ON console_trace_response_links
WHEN console_trace_gc_delete_authorized('console_trace_response_links') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_revision_bindings_no_delete
BEFORE DELETE ON console_trace_revision_bindings
WHEN console_trace_gc_delete_authorized('console_trace_revision_bindings') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_segments_no_delete
BEFORE DELETE ON console_trace_segments
WHEN console_trace_gc_delete_authorized('console_trace_segments') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_semantic_revisions_no_delete
BEFORE DELETE ON console_trace_semantic_revisions
WHEN console_trace_gc_delete_authorized('console_trace_semantic_revisions') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_surface_nodes_no_delete
BEFORE DELETE ON console_trace_surface_nodes
WHEN console_trace_gc_delete_authorized('console_trace_surface_nodes') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
CREATE TRIGGER console_trace_surface_replacements_no_delete
BEFORE DELETE ON console_trace_surface_replacements
WHEN console_trace_gc_delete_authorized('console_trace_surface_replacements') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;

CREATE TRIGGER console_trace_retention_roots_no_update
BEFORE UPDATE ON console_trace_retention_roots
BEGIN SELECT RAISE(ABORT, 'console_trace_retention_roots is immutable'); END;
CREATE TRIGGER console_trace_retention_roots_no_delete
BEFORE DELETE ON console_trace_retention_roots
WHEN console_trace_gc_delete_authorized('console_trace_retention_roots') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;
