-- ChaChaNotes v55 -> v56: reference-backed Console semantic trace storage.
--
-- Fast DDL only. This step deliberately does not inspect, decode, normalize,
-- rewrite, or delete message_exchanges.capture_blob or ordinary message data.

CREATE TABLE console_trace_segments(
  segment_id TEXT PRIMARY KEY NOT NULL,
  parent_segment_id TEXT DEFAULT NULL,
  inherited_through_sequence INTEGER DEFAULT NULL,
  inherited_surface_head_id TEXT DEFAULT NULL
    REFERENCES console_trace_surface_nodes(node_id),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (parent_segment_id IS NULL AND inherited_through_sequence IS NULL AND
     inherited_surface_head_id IS NULL) OR
    (parent_segment_id IS NOT NULL AND inherited_through_sequence >= 0 AND
     inherited_surface_head_id IS NOT NULL)
  ),
  CHECK(parent_segment_id IS NULL OR parent_segment_id <> segment_id),
  FOREIGN KEY(parent_segment_id, inherited_through_sequence)
    REFERENCES console_trace_events(segment_id, sequence)
);

CREATE TABLE console_trace_policies(
  policy_id TEXT PRIMARY KEY NOT NULL,
  credential_filter_version TEXT NOT NULL CHECK(length(credential_filter_version) > 0),
  pii_redaction_enabled INTEGER NOT NULL DEFAULT 0
    CHECK(pii_redaction_enabled IN (0, 1)),
  pii_ruleset_revision_id TEXT DEFAULT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (pii_redaction_enabled = 0) OR
    (pii_redaction_enabled = 1 AND pii_ruleset_revision_id IS NOT NULL)
  )
);

CREATE TABLE console_trace_semantic_revisions(
  revision_id TEXT PRIMARY KEY NOT NULL,
  source_conversation_id TEXT NOT NULL,
  source_message_id TEXT NOT NULL,
  revision_sequence INTEGER NOT NULL CHECK(revision_sequence >= 0),
  normalized_role TEXT NOT NULL CHECK(length(normalized_role) > 0),
  content_kind TEXT NOT NULL CHECK(length(content_kind) > 0),
  creation_reason TEXT NOT NULL CHECK(length(creation_reason) > 0),
  predecessor_revision_id TEXT DEFAULT NULL
    REFERENCES console_trace_semantic_revisions(revision_id),
  live_message_id TEXT DEFAULT NULL REFERENCES messages(id) ON DELETE RESTRICT,
  live_locator_retired_at TEXT DEFAULT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(predecessor_revision_id IS NULL OR predecessor_revision_id <> revision_id),
  CHECK(
    (predecessor_revision_id IS NULL AND revision_sequence = 0) OR
    (predecessor_revision_id IS NOT NULL AND revision_sequence > 0)
  ),
  CHECK(live_message_id IS NULL OR live_message_id = source_message_id),
  CHECK(
    (live_message_id IS NOT NULL AND live_locator_retired_at IS NULL) OR
    (live_message_id IS NULL AND live_locator_retired_at IS NOT NULL)
  ),
  UNIQUE(source_message_id, revision_sequence)
);

CREATE TABLE console_trace_artifacts(
  artifact_id TEXT PRIMARY KEY NOT NULL,
  identity_digest TEXT NOT NULL
    CHECK(length(identity_digest) = 64)
    CHECK(identity_digest NOT GLOB '*[^0-9a-f]*'),
  media_type TEXT NOT NULL CHECK(length(media_type) > 0),
  normalization_version TEXT NOT NULL CHECK(length(normalization_version) > 0),
  sanitized_bytes BLOB NOT NULL,
  byte_length INTEGER NOT NULL CHECK(byte_length >= 0),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(typeof(sanitized_bytes) = 'blob'),
  CHECK(length(sanitized_bytes) = byte_length)
);

CREATE TABLE console_trace_revision_bindings(
  revision_id TEXT NOT NULL
    REFERENCES console_trace_semantic_revisions(revision_id),
  policy_id TEXT NOT NULL REFERENCES console_trace_policies(policy_id),
  binding_outcome TEXT NOT NULL CHECK(binding_outcome IN ('artifact', 'omission')),
  artifact_id TEXT DEFAULT NULL REFERENCES console_trace_artifacts(artifact_id),
  omission_reason_code TEXT DEFAULT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(revision_id, policy_id),
  CHECK(
    (binding_outcome = 'artifact' AND artifact_id IS NOT NULL AND
     omission_reason_code IS NULL) OR
    (binding_outcome = 'omission' AND artifact_id IS NULL AND
     omission_reason_code IS NOT NULL AND length(omission_reason_code) > 0)
  )
);

CREATE TABLE console_trace_surface_nodes(
  node_id TEXT PRIMARY KEY NOT NULL,
  segment_id TEXT NOT NULL REFERENCES console_trace_segments(segment_id),
  sequence INTEGER NOT NULL CHECK(sequence >= 0),
  predecessor_node_id TEXT DEFAULT NULL
    REFERENCES console_trace_surface_nodes(node_id),
  component_kind TEXT NOT NULL CHECK(length(component_kind) > 0),
  reference_kind TEXT NOT NULL
    CHECK(reference_kind IN ('revision', 'artifact', 'omission')),
  semantic_revision_id TEXT DEFAULT NULL
    REFERENCES console_trace_semantic_revisions(revision_id),
  artifact_id TEXT DEFAULT NULL REFERENCES console_trace_artifacts(artifact_id),
  omission_reason_code TEXT DEFAULT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(segment_id, sequence),
  UNIQUE(node_id, sequence),
  CHECK(
    (reference_kind = 'revision' AND semantic_revision_id IS NOT NULL AND
     artifact_id IS NULL AND omission_reason_code IS NULL) OR
    (reference_kind = 'artifact' AND semantic_revision_id IS NULL AND
     artifact_id IS NOT NULL AND omission_reason_code IS NULL) OR
    (reference_kind = 'omission' AND semantic_revision_id IS NULL AND
     artifact_id IS NULL AND omission_reason_code IS NOT NULL AND
     length(omission_reason_code) > 0)
  )
);

CREATE TABLE console_trace_surface_replacements(
  replacement_id TEXT PRIMARY KEY NOT NULL,
  segment_id TEXT NOT NULL REFERENCES console_trace_segments(segment_id),
  predecessor_head_id TEXT NOT NULL
    REFERENCES console_trace_surface_nodes(node_id),
  start_node_id TEXT NOT NULL,
  start_sequence INTEGER NOT NULL CHECK(start_sequence >= 0),
  end_node_id TEXT NOT NULL,
  end_sequence INTEGER NOT NULL CHECK(end_sequence >= 0),
  replacement_node_id TEXT NOT NULL
    REFERENCES console_trace_surface_nodes(node_id),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(start_node_id, start_sequence)
    REFERENCES console_trace_surface_nodes(node_id, sequence),
  FOREIGN KEY(end_node_id, end_sequence)
    REFERENCES console_trace_surface_nodes(node_id, sequence),
  CHECK(end_sequence >= start_sequence)
);

CREATE TABLE console_trace_request_headers(
  header_id TEXT PRIMARY KEY NOT NULL,
  provider_name TEXT NOT NULL CHECK(length(provider_name) > 0),
  model_name TEXT NOT NULL CHECK(length(model_name) > 0),
  route_identity TEXT NOT NULL CHECK(length(route_identity) > 0),
  endpoint_identity TEXT NOT NULL CHECK(length(endpoint_identity) > 0),
  generation_parameters_json TEXT NOT NULL,
  adapter_defaults_json TEXT NOT NULL,
  response_format_json TEXT NOT NULL,
  reasoning_controls_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(json_valid(generation_parameters_json) AND
        json_type(generation_parameters_json) = 'object'),
  CHECK(json_valid(adapter_defaults_json) AND
        json_type(adapter_defaults_json) = 'object'),
  CHECK(json_valid(response_format_json) AND
        json_type(response_format_json) = 'object'),
  CHECK(json_valid(reasoning_controls_json) AND
        json_type(reasoning_controls_json) = 'object')
);

CREATE TABLE console_trace_header_components(
  header_id TEXT NOT NULL REFERENCES console_trace_request_headers(header_id),
  component_kind TEXT NOT NULL CHECK(length(component_kind) > 0),
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  artifact_id TEXT NOT NULL REFERENCES console_trace_artifacts(artifact_id),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(header_id, component_kind, ordinal)
);

CREATE TABLE console_trace_owners(
  owner_id TEXT PRIMARY KEY NOT NULL,
  conversation_id TEXT UNIQUE DEFAULT NULL
    REFERENCES conversations(id) ON DELETE SET NULL,
  root_segment_id TEXT NOT NULL REFERENCES console_trace_segments(segment_id),
  attached INTEGER NOT NULL DEFAULT 1 CHECK(attached IN (0, 1)),
  detached_at TEXT DEFAULT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (attached = 1 AND conversation_id IS NOT NULL AND detached_at IS NULL) OR
    (attached = 0 AND conversation_id IS NULL AND detached_at IS NOT NULL)
  )
);

CREATE TABLE console_trace_calls(
  call_id TEXT PRIMARY KEY NOT NULL,
  owner_id TEXT NOT NULL REFERENCES console_trace_owners(owner_id),
  segment_id TEXT NOT NULL REFERENCES console_trace_segments(segment_id),
  turn_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  call_sequence INTEGER NOT NULL CHECK(call_sequence >= 0),
  idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) > 0),
  policy_id TEXT NOT NULL REFERENCES console_trace_policies(policy_id),
  state TEXT NOT NULL DEFAULT 'reserved' CHECK(state IN (
    'reserved', 'not_dispatched', 'dispatch_started', 'dispatch_unknown',
    'response_started', 'complete', 'stopped', 'error', 'interrupted',
    'abandoned'
  )),
  surface_node_id TEXT DEFAULT NULL REFERENCES console_trace_surface_nodes(node_id),
  request_header_id TEXT DEFAULT NULL
    REFERENCES console_trace_request_headers(header_id),
  provider_name TEXT DEFAULT NULL,
  model_name TEXT DEFAULT NULL,
  route_identity TEXT DEFAULT NULL,
  dispatch_started_at TEXT DEFAULT NULL,
  response_started_at TEXT DEFAULT NULL,
  settled_at TEXT DEFAULT NULL,
  provider_inactive_at TEXT DEFAULT NULL,
  outcome TEXT DEFAULT NULL CHECK(outcome IS NULL OR outcome IN (
    'complete', 'stopped', 'error', 'interrupted', 'abandoned'
  )),
  usage_json TEXT DEFAULT NULL
    CHECK(usage_json IS NULL OR
          (json_valid(usage_json) AND json_type(usage_json) = 'object')),
  integrity_state TEXT NOT NULL DEFAULT 'pending'
    CHECK(integrity_state IN ('pending', 'complete', 'incomplete')),
  omission_reason_code TEXT DEFAULT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (surface_node_id IS NULL AND request_header_id IS NULL) OR
    (surface_node_id IS NOT NULL AND request_header_id IS NOT NULL)
  ),
  CHECK(
    (provider_name IS NULL AND model_name IS NULL AND route_identity IS NULL) OR
    (provider_name IS NOT NULL AND model_name IS NOT NULL AND
     route_identity IS NOT NULL)
  ),
  CHECK(
    state IN ('reserved', 'not_dispatched') OR
    (surface_node_id IS NOT NULL AND request_header_id IS NOT NULL AND
     provider_name IS NOT NULL AND dispatch_started_at IS NOT NULL)
  ),
  CHECK(
    (state IN ('complete', 'stopped', 'error', 'interrupted', 'abandoned') AND
     outcome = state) OR
    (state NOT IN ('complete', 'stopped', 'error', 'interrupted', 'abandoned') AND
     outcome IS NULL)
  ),
  CHECK(
    (state = 'reserved' AND dispatch_started_at IS NULL AND
     response_started_at IS NULL AND settled_at IS NULL AND
     provider_inactive_at IS NULL) OR
    (state = 'not_dispatched' AND dispatch_started_at IS NULL AND
     response_started_at IS NULL AND settled_at IS NOT NULL AND
     provider_inactive_at IS NULL) OR
    (state = 'dispatch_started' AND dispatch_started_at IS NOT NULL AND
     response_started_at IS NULL AND settled_at IS NULL AND
     provider_inactive_at IS NULL) OR
    (state = 'dispatch_unknown' AND dispatch_started_at IS NOT NULL AND
     response_started_at IS NULL AND settled_at IS NOT NULL AND
     provider_inactive_at IS NULL) OR
    (state = 'response_started' AND dispatch_started_at IS NOT NULL AND
     response_started_at IS NOT NULL AND settled_at IS NULL AND
     provider_inactive_at IS NULL) OR
    (state IN ('complete', 'stopped', 'interrupted') AND
     dispatch_started_at IS NOT NULL AND response_started_at IS NOT NULL AND
     settled_at IS NOT NULL AND provider_inactive_at IS NULL) OR
    (state = 'error' AND dispatch_started_at IS NOT NULL AND
     settled_at IS NOT NULL AND provider_inactive_at IS NULL) OR
    (state = 'abandoned' AND dispatch_started_at IS NOT NULL AND
     response_started_at IS NULL AND settled_at IS NOT NULL AND
     provider_inactive_at IS NOT NULL)
  ),
  CHECK(
    usage_json IS NULL OR
    state IN ('complete', 'stopped', 'error', 'interrupted', 'abandoned')
  )
);

CREATE TABLE console_trace_events(
  event_id TEXT PRIMARY KEY NOT NULL,
  segment_id TEXT NOT NULL REFERENCES console_trace_segments(segment_id),
  sequence INTEGER NOT NULL CHECK(sequence >= 0),
  event_type TEXT NOT NULL CHECK(event_type IN (
    'turn_boundary', 'call_boundary', 'surface_append', 'surface_replace',
    'tool_call', 'tool_result', 'request_header_selection',
    'provider_route_selection', 'response_selection', 'call_outcome',
    'usage', 'gap'
  )),
  turn_id TEXT DEFAULT NULL,
  call_id TEXT DEFAULT NULL REFERENCES console_trace_calls(call_id),
  surface_node_id TEXT DEFAULT NULL REFERENCES console_trace_surface_nodes(node_id),
  surface_replacement_id TEXT DEFAULT NULL
    REFERENCES console_trace_surface_replacements(replacement_id),
  request_header_id TEXT DEFAULT NULL
    REFERENCES console_trace_request_headers(header_id),
  semantic_revision_id TEXT DEFAULT NULL
    REFERENCES console_trace_semantic_revisions(revision_id),
  artifact_id TEXT DEFAULT NULL REFERENCES console_trace_artifacts(artifact_id),
  omission_reason_code TEXT DEFAULT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(segment_id, sequence),
  CHECK(semantic_revision_id IS NULL OR artifact_id IS NULL),
  CHECK(event_type <> 'surface_append' OR surface_node_id IS NOT NULL),
  CHECK(event_type <> 'surface_replace' OR surface_replacement_id IS NOT NULL),
  CHECK(event_type <> 'request_header_selection' OR request_header_id IS NOT NULL),
  CHECK(event_type NOT IN (
    'call_boundary', 'response_selection', 'call_outcome', 'usage'
  ) OR call_id IS NOT NULL),
  CHECK(event_type <> 'gap' OR omission_reason_code IS NOT NULL)
);

CREATE TABLE console_trace_response_links(
  response_link_id TEXT PRIMARY KEY NOT NULL,
  call_id TEXT NOT NULL UNIQUE REFERENCES console_trace_calls(call_id),
  link_kind TEXT NOT NULL CHECK(link_kind IN ('revision', 'artifact')),
  semantic_revision_id TEXT DEFAULT NULL
    REFERENCES console_trace_semantic_revisions(revision_id),
  artifact_id TEXT DEFAULT NULL REFERENCES console_trace_artifacts(artifact_id),
  verification_outcome TEXT NOT NULL
    CHECK(verification_outcome IN ('verified_equal', 'sanitized_artifact')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (link_kind = 'revision' AND semantic_revision_id IS NOT NULL AND
     artifact_id IS NULL AND verification_outcome = 'verified_equal') OR
    (link_kind = 'artifact' AND semantic_revision_id IS NULL AND
     artifact_id IS NOT NULL AND verification_outcome = 'sanitized_artifact')
  )
);

CREATE TABLE console_trace_redaction_spans(
  span_id TEXT PRIMARY KEY NOT NULL,
  policy_id TEXT NOT NULL REFERENCES console_trace_policies(policy_id),
  source_kind TEXT NOT NULL CHECK(source_kind IN ('revision', 'artifact')),
  semantic_revision_id TEXT DEFAULT NULL
    REFERENCES console_trace_semantic_revisions(revision_id),
  artifact_id TEXT DEFAULT NULL REFERENCES console_trace_artifacts(artifact_id),
  field_path TEXT NOT NULL CHECK(length(field_path) > 0),
  start_codepoint INTEGER NOT NULL CHECK(start_codepoint >= 0),
  end_codepoint INTEGER NOT NULL CHECK(end_codepoint > start_codepoint),
  category TEXT NOT NULL CHECK(length(category) > 0),
  rule_id TEXT NOT NULL CHECK(length(rule_id) > 0),
  detector_version TEXT NOT NULL CHECK(length(detector_version) > 0),
  outcome TEXT NOT NULL CHECK(outcome IN ('applied', 'omitted', 'unavailable')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (source_kind = 'revision' AND semantic_revision_id IS NOT NULL AND
     artifact_id IS NULL) OR
    (source_kind = 'artifact' AND semantic_revision_id IS NULL AND
     artifact_id IS NOT NULL)
  )
);

CREATE TABLE console_trace_migration_state(
  migration_name TEXT PRIMARY KEY NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK(status IN ('pending', 'running', 'logical_complete', 'failed')),
  last_exchange_id INTEGER DEFAULT NULL CHECK(last_exchange_id IS NULL OR last_exchange_id >= 0),
  processed_rows INTEGER NOT NULL DEFAULT 0 CHECK(processed_rows >= 0),
  processed_bytes INTEGER NOT NULL DEFAULT 0 CHECK(processed_bytes >= 0),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE console_trace_maintenance_state(
  singleton_id INTEGER PRIMARY KEY NOT NULL CHECK(singleton_id = 1),
  state TEXT NOT NULL DEFAULT 'idle'
    CHECK(state IN ('idle', 'marking', 'sweeping', 'compacting')),
  lease_id TEXT DEFAULT NULL,
  lease_owner TEXT DEFAULT NULL,
  lease_expires_at TEXT DEFAULT NULL,
  marked_epoch INTEGER DEFAULT NULL CHECK(marked_epoch IS NULL OR marked_epoch >= 0),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (state = 'idle' AND lease_id IS NULL AND lease_owner IS NULL AND
     lease_expires_at IS NULL AND marked_epoch IS NULL) OR
    (state <> 'idle' AND lease_id IS NOT NULL AND lease_owner IS NOT NULL AND
     lease_expires_at IS NOT NULL)
  )
);

CREATE TABLE console_trace_graph_epoch(
  singleton_id INTEGER PRIMARY KEY NOT NULL CHECK(singleton_id = 1),
  epoch INTEGER NOT NULL DEFAULT 0 CHECK(epoch >= 0),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_console_trace_segments_parent_boundary
  ON console_trace_segments(
    parent_segment_id, inherited_through_sequence, inherited_surface_head_id
  );
CREATE INDEX idx_console_trace_semantic_revisions_source
  ON console_trace_semantic_revisions(source_conversation_id, source_message_id,
                                      revision_sequence);
CREATE UNIQUE INDEX uq_console_trace_semantic_revisions_live_message
  ON console_trace_semantic_revisions(live_message_id)
  WHERE live_message_id IS NOT NULL;
CREATE INDEX idx_console_trace_revision_bindings_artifact
  ON console_trace_revision_bindings(artifact_id)
  WHERE artifact_id IS NOT NULL;
CREATE INDEX idx_console_trace_artifacts_identity
  ON console_trace_artifacts(identity_digest, media_type, normalization_version);
-- Deliberately non-unique: callers compare the stored bytes after digest lookup,
-- and a digest collision must receive a separate opaque artifact identity.
CREATE INDEX idx_console_trace_surface_nodes_segment_order
  ON console_trace_surface_nodes(segment_id, sequence);
CREATE INDEX idx_console_trace_surface_nodes_predecessor
  ON console_trace_surface_nodes(predecessor_node_id, segment_id);
CREATE INDEX idx_console_trace_surface_replacements_predecessor
  ON console_trace_surface_replacements(segment_id, predecessor_head_id);
CREATE INDEX idx_console_trace_header_components_artifact
  ON console_trace_header_components(artifact_id, header_id);
CREATE UNIQUE INDEX idx_console_trace_owners_root_segment
  ON console_trace_owners(root_segment_id);
CREATE UNIQUE INDEX uq_console_trace_calls_idempotency
  ON console_trace_calls(idempotency_key);
CREATE UNIQUE INDEX uq_console_trace_calls_owner_sequence
  ON console_trace_calls(owner_id, segment_id, turn_id, run_id, call_sequence);
CREATE INDEX idx_console_trace_calls_owner_order
  ON console_trace_calls(owner_id, turn_id, run_id, call_sequence);
CREATE INDEX idx_console_trace_calls_segment_order
  ON console_trace_calls(segment_id, turn_id, run_id, call_sequence);
CREATE INDEX idx_console_trace_events_segment_order
  ON console_trace_events(segment_id, sequence);
CREATE INDEX idx_console_trace_events_call_order
  ON console_trace_events(call_id, sequence)
  WHERE call_id IS NOT NULL;
CREATE INDEX idx_console_trace_response_revision
  ON console_trace_response_links(semantic_revision_id)
  WHERE semantic_revision_id IS NOT NULL;
CREATE INDEX idx_console_trace_response_artifact
  ON console_trace_response_links(artifact_id)
  WHERE artifact_id IS NOT NULL;
CREATE INDEX idx_console_trace_redaction_revision
  ON console_trace_redaction_spans(semantic_revision_id, policy_id, field_path,
                                   start_codepoint)
  WHERE semantic_revision_id IS NOT NULL;
CREATE INDEX idx_console_trace_redaction_artifact
  ON console_trace_redaction_spans(artifact_id, policy_id, field_path,
                                   start_codepoint)
  WHERE artifact_id IS NOT NULL;
CREATE INDEX idx_console_trace_migration_status
  ON console_trace_migration_state(status, migration_name);

INSERT INTO console_trace_migration_state(
  migration_name, status, last_exchange_id, processed_rows, processed_bytes
) VALUES ('legacy_exchange_normalization', 'pending', NULL, 0, 0);

INSERT INTO console_trace_maintenance_state(singleton_id, state)
VALUES (1, 'idle');

INSERT INTO console_trace_graph_epoch(singleton_id, epoch)
VALUES (1, 0);

CREATE TRIGGER console_trace_events_append_order
BEFORE INSERT ON console_trace_events
WHEN EXISTS (
  SELECT 1
    FROM console_trace_events AS existing
   WHERE existing.segment_id = NEW.segment_id
     AND existing.sequence >= NEW.sequence
)
BEGIN
  SELECT RAISE(
    ABORT,
    'trace event sequence must preserve per-segment append order'
  );
END;

CREATE TRIGGER console_trace_events_shape_guard
BEFORE INSERT ON console_trace_events
WHEN NOT (
  (NEW.event_type = 'turn_boundary' AND
   NEW.turn_id IS NOT NULL AND
   NEW.call_id IS NULL AND NEW.surface_node_id IS NULL AND
   NEW.surface_replacement_id IS NULL AND NEW.request_header_id IS NULL AND
   NEW.semantic_revision_id IS NULL AND NEW.artifact_id IS NULL AND
   NEW.omission_reason_code IS NULL) OR
  (NEW.event_type IN ('call_boundary', 'call_outcome', 'usage') AND
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

CREATE TRIGGER console_trace_events_lineage_guard
BEFORE INSERT ON console_trace_events
WHEN
  (NEW.surface_node_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
      FROM console_trace_surface_nodes AS surface
     WHERE surface.node_id = NEW.surface_node_id
       AND surface.segment_id = NEW.segment_id
  )) OR
  (NEW.surface_replacement_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
      FROM console_trace_surface_replacements AS replacement
     WHERE replacement.replacement_id = NEW.surface_replacement_id
       AND replacement.segment_id = NEW.segment_id
  )) OR
  (NEW.call_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
      FROM console_trace_calls AS call
     WHERE call.call_id = NEW.call_id
       AND call.segment_id = NEW.segment_id
  )) OR
  (NEW.request_header_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
      FROM console_trace_calls AS call
     WHERE call.call_id = NEW.call_id
       AND call.segment_id = NEW.segment_id
       AND call.request_header_id = NEW.request_header_id
  ))
BEGIN
  SELECT RAISE(ABORT, 'trace event ownership lineage is inconsistent');
END;

CREATE TRIGGER console_trace_events_owner_guard
BEFORE INSERT ON console_trace_events
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT NEW.segment_id, 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.conversation_id, owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1
    FROM effective_owner AS owner
   WHERE owner.attached = 1
     AND (
       NEW.semantic_revision_id IS NULL OR EXISTS (
         SELECT 1
           FROM console_trace_semantic_revisions AS revision
          WHERE revision.revision_id = NEW.semantic_revision_id
            AND revision.source_conversation_id = owner.conversation_id
       )
     )
)
BEGIN
  SELECT RAISE(
    ABORT,
    'trace event requires one active effective owner and matching revision domain'
  );
END;

CREATE TRIGGER console_trace_surface_nodes_contiguous
BEFORE INSERT ON console_trace_surface_nodes
WHEN
  (NEW.predecessor_node_id IS NULL AND (
    NEW.sequence <> 0 OR EXISTS (
      SELECT 1 FROM console_trace_segments AS segment
       WHERE segment.segment_id = NEW.segment_id
         AND segment.parent_segment_id IS NOT NULL
    )
  )) OR
  (NEW.predecessor_node_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
      FROM console_trace_surface_nodes AS predecessor
      JOIN console_trace_segments AS segment
        ON segment.segment_id = NEW.segment_id
     WHERE predecessor.node_id = NEW.predecessor_node_id
       AND predecessor.sequence = NEW.sequence - 1
       AND (
         (segment.parent_segment_id IS NULL AND
          predecessor.segment_id = NEW.segment_id) OR
         (segment.parent_segment_id IS NOT NULL AND (
           predecessor.segment_id = NEW.segment_id OR
           predecessor.node_id = segment.inherited_surface_head_id
         ))
       )
  ))
BEGIN
  SELECT RAISE(
    ABORT,
    'surface node predecessor must be contiguous and within segment lineage'
  );
END;

CREATE TRIGGER console_trace_surface_nodes_owner_guard
BEFORE INSERT ON console_trace_surface_nodes
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT NEW.segment_id, 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.conversation_id, owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1
    FROM effective_owner AS owner
   WHERE owner.attached = 1
     AND (
       NEW.semantic_revision_id IS NULL OR EXISTS (
         SELECT 1
           FROM console_trace_semantic_revisions AS revision
          WHERE revision.revision_id = NEW.semantic_revision_id
            AND revision.source_conversation_id = owner.conversation_id
       )
     )
)
BEGIN
  SELECT RAISE(
    ABORT,
    'surface node requires one active effective owner and matching revision domain'
  );
END;

CREATE TRIGGER console_trace_surface_replacements_lineage
BEFORE INSERT ON console_trace_surface_replacements
WHEN NOT EXISTS (
  WITH RECURSIVE
  inherited_chain(node_id, predecessor_node_id) AS (
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_segments AS segment
      JOIN console_trace_surface_nodes AS node
        ON node.node_id = segment.inherited_surface_head_id
     WHERE segment.segment_id = NEW.segment_id
    UNION ALL
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_surface_nodes AS node
      JOIN inherited_chain AS successor
        ON successor.predecessor_node_id = node.node_id
  ),
  predecessor_chain(node_id, predecessor_node_id) AS (
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_surface_nodes AS node
     WHERE node.node_id = NEW.predecessor_head_id
    UNION ALL
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_surface_nodes AS node
      JOIN predecessor_chain AS successor
        ON successor.predecessor_node_id = node.node_id
  ),
  replacement_chain(node_id, predecessor_node_id) AS (
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_surface_nodes AS node
     WHERE node.node_id = NEW.replacement_node_id
    UNION ALL
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_surface_nodes AS node
      JOIN replacement_chain AS successor
        ON successor.predecessor_node_id = node.node_id
  )
  SELECT 1
    FROM console_trace_segments AS segment
    JOIN console_trace_surface_nodes AS predecessor_head
      ON predecessor_head.node_id = NEW.predecessor_head_id
    JOIN console_trace_surface_nodes AS replacement
      ON replacement.node_id = NEW.replacement_node_id
   WHERE segment.segment_id = NEW.segment_id
     AND (
       (segment.parent_segment_id IS NULL AND
        predecessor_head.segment_id = NEW.segment_id) OR
       (segment.parent_segment_id IS NOT NULL AND (
         (predecessor_head.segment_id = NEW.segment_id AND EXISTS (
           SELECT 1 FROM predecessor_chain
            WHERE node_id = segment.inherited_surface_head_id
         )) OR
         EXISTS (
           SELECT 1 FROM inherited_chain
            WHERE node_id = predecessor_head.node_id
         )
       ))
     )
     AND replacement.segment_id = NEW.segment_id
     AND replacement.node_id <> predecessor_head.node_id
     AND EXISTS (
       SELECT 1 FROM predecessor_chain
        WHERE node_id = NEW.start_node_id
     )
     AND EXISTS (
       SELECT 1 FROM predecessor_chain
        WHERE node_id = NEW.end_node_id
     )
     AND EXISTS (
       SELECT 1 FROM replacement_chain
        WHERE node_id = NEW.predecessor_head_id
     )
)
BEGIN
  SELECT RAISE(ABORT, 'surface replacement references must remain within lineage');
END;

CREATE TRIGGER console_trace_surface_replacements_owner_guard
BEFORE INSERT ON console_trace_surface_replacements
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT NEW.segment_id, 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1 FROM effective_owner WHERE attached = 1
)
BEGIN
  SELECT RAISE(
    ABORT,
    'surface replacement requires one active effective owner'
  );
END;

CREATE TRIGGER console_trace_segments_inherited_surface
BEFORE INSERT ON console_trace_segments
WHEN NEW.parent_segment_id IS NOT NULL AND NOT EXISTS (
  WITH RECURSIVE
  parent_inherited_chain(node_id, predecessor_node_id) AS (
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_segments AS parent
      JOIN console_trace_surface_nodes AS node
        ON node.node_id = parent.inherited_surface_head_id
     WHERE parent.segment_id = NEW.parent_segment_id
    UNION ALL
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_surface_nodes AS node
      JOIN parent_inherited_chain AS successor
        ON successor.predecessor_node_id = node.node_id
  ),
  last_surface_event(head_id) AS (
    SELECT CASE event.event_type
             WHEN 'surface_append' THEN event.surface_node_id
             ELSE replacement.replacement_node_id
           END
      FROM console_trace_events AS event
      LEFT JOIN console_trace_surface_replacements AS replacement
        ON replacement.replacement_id = event.surface_replacement_id
     WHERE event.segment_id = NEW.parent_segment_id
       AND event.sequence <= NEW.inherited_through_sequence
       AND event.event_type IN ('surface_append', 'surface_replace')
     ORDER BY event.sequence DESC
     LIMIT 1
  ),
  expected_surface(head_id) AS (
    SELECT head_id FROM last_surface_event
    UNION ALL
    SELECT parent.inherited_surface_head_id
      FROM console_trace_segments AS parent
     WHERE parent.segment_id = NEW.parent_segment_id
       AND NOT EXISTS (SELECT 1 FROM last_surface_event)
  )
  SELECT 1
    FROM console_trace_segments AS parent
    JOIN expected_surface AS expected
      ON expected.head_id = NEW.inherited_surface_head_id
    JOIN console_trace_surface_nodes AS head
      ON head.node_id = expected.head_id
   WHERE parent.segment_id = NEW.parent_segment_id
     AND (
       head.segment_id = parent.segment_id OR
       EXISTS (
         SELECT 1 FROM parent_inherited_chain
          WHERE node_id = head.node_id
       )
     )
)
BEGIN
  SELECT RAISE(
    ABORT,
    'inherited surface head must match the parent surface at the event boundary'
  );
END;

CREATE TRIGGER console_trace_segments_parent_owner_guard
BEFORE INSERT ON console_trace_segments
WHEN NEW.parent_segment_id IS NOT NULL AND EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT NEW.parent_segment_id, 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1 FROM effective_owner WHERE attached = 0
)
BEGIN
  SELECT RAISE(ABORT, 'child segment cannot extend a detached owner prefix');
END;

CREATE TRIGGER console_trace_semantic_revisions_lineage
BEFORE INSERT ON console_trace_semantic_revisions
WHEN
  (NEW.live_message_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
      FROM messages AS source
     WHERE source.id = NEW.live_message_id
       AND source.id = NEW.source_message_id
       AND source.conversation_id = NEW.source_conversation_id
  )) OR
  (NEW.predecessor_revision_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
      FROM console_trace_semantic_revisions AS predecessor
     WHERE predecessor.revision_id = NEW.predecessor_revision_id
       AND predecessor.source_conversation_id = NEW.source_conversation_id
       AND predecessor.source_message_id = NEW.source_message_id
       AND predecessor.revision_sequence = NEW.revision_sequence - 1
  ))
BEGIN
  SELECT RAISE(
    ABORT,
    'semantic revision locator or predecessor lineage is inconsistent'
  );
END;

CREATE TRIGGER console_trace_calls_insert_reserved
BEFORE INSERT ON console_trace_calls
WHEN
  NEW.state <> 'reserved' OR
  NEW.surface_node_id IS NOT NULL OR
  NEW.request_header_id IS NOT NULL OR
  NEW.provider_name IS NOT NULL OR
  NEW.model_name IS NOT NULL OR
  NEW.route_identity IS NOT NULL OR
  NEW.dispatch_started_at IS NOT NULL OR
  NEW.response_started_at IS NOT NULL OR
  NEW.settled_at IS NOT NULL OR
  NEW.provider_inactive_at IS NOT NULL OR
  NEW.outcome IS NOT NULL OR
  NEW.usage_json IS NOT NULL OR
  NEW.integrity_state <> 'pending' OR
  NEW.omission_reason_code IS NOT NULL
BEGIN
  SELECT RAISE(ABORT, 'provider calls must be inserted as content-free reservations');
END;

CREATE TRIGGER console_trace_calls_owner_lineage
BEFORE INSERT ON console_trace_calls
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT NEW.segment_id, 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.owner_id, owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1
    FROM effective_owner AS owner
   WHERE owner.owner_id = NEW.owner_id
     AND owner.attached = 1
)
BEGIN
  SELECT RAISE(ABORT, 'provider call must use the active effective owner');
END;

CREATE TRIGGER console_trace_owners_empty_root_guard
BEFORE INSERT ON console_trace_owners
WHEN
  EXISTS (
    SELECT 1 FROM console_trace_surface_nodes
     WHERE segment_id = NEW.root_segment_id
  ) OR
  EXISTS (
    SELECT 1 FROM console_trace_surface_replacements
     WHERE segment_id = NEW.root_segment_id
  ) OR
  EXISTS (
    SELECT 1 FROM console_trace_calls
     WHERE segment_id = NEW.root_segment_id
  ) OR
  EXISTS (
    SELECT 1 FROM console_trace_events
     WHERE segment_id = NEW.root_segment_id
  ) OR
  EXISTS (
    SELECT 1 FROM console_trace_segments
     WHERE parent_segment_id = NEW.root_segment_id
  )
BEGIN
  SELECT RAISE(ABORT, 'trace owner must attach to a distinct empty root segment');
END;

CREATE TRIGGER console_trace_owners_active_prefix_guard
BEFORE INSERT ON console_trace_owners
WHEN EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT root.parent_segment_id, 0
      FROM console_trace_segments AS root
     WHERE root.segment_id = NEW.root_segment_id
       AND root.parent_segment_id IS NOT NULL
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1 FROM effective_owner WHERE attached = 0
)
BEGIN
  SELECT RAISE(ABORT, 'trace owner cannot revive a detached owner prefix');
END;

CREATE TRIGGER console_trace_calls_immutable_guard
BEFORE UPDATE ON console_trace_calls
WHEN
  OLD.call_id IS NOT NEW.call_id OR
  OLD.owner_id IS NOT NEW.owner_id OR
  OLD.segment_id IS NOT NEW.segment_id OR
  OLD.turn_id IS NOT NEW.turn_id OR
  OLD.run_id IS NOT NEW.run_id OR
  OLD.call_sequence IS NOT NEW.call_sequence OR
  OLD.idempotency_key IS NOT NEW.idempotency_key OR
  OLD.policy_id IS NOT NEW.policy_id OR
  OLD.created_at IS NOT NEW.created_at
BEGIN
  SELECT RAISE(ABORT, 'provider call ownership and identity are immutable');
END;

CREATE TRIGGER console_trace_calls_lifecycle_guard
BEFORE UPDATE ON console_trace_calls
WHEN NEW.state IS NOT OLD.state AND NOT (
  (OLD.state = 'reserved' AND NEW.state IN ('not_dispatched', 'dispatch_started')) OR
  (OLD.state = 'dispatch_started' AND (
    NEW.state IN ('response_started', 'dispatch_unknown', 'abandoned') OR
    (NEW.state = 'error' AND NEW.response_started_at IS NULL)
  )) OR
  (OLD.state = 'response_started' AND NEW.state IN (
    'complete', 'stopped', 'error', 'interrupted'
  ))
)
BEGIN
  SELECT RAISE(ABORT, 'invalid provider call lifecycle transition');
END;

CREATE TRIGGER console_trace_calls_binding_guard
BEFORE UPDATE ON console_trace_calls
WHEN NEW.surface_node_id IS NOT NULL AND (
  OLD.surface_node_id IS NOT NEW.surface_node_id OR
  OLD.request_header_id IS NOT NEW.request_header_id OR
  OLD.provider_name IS NOT NEW.provider_name OR
  OLD.model_name IS NOT NEW.model_name OR
  OLD.route_identity IS NOT NEW.route_identity
) AND NOT EXISTS (
  WITH RECURSIVE
  inherited_chain(node_id, predecessor_node_id) AS (
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_segments AS segment
      JOIN console_trace_surface_nodes AS node
        ON node.node_id = segment.inherited_surface_head_id
     WHERE segment.segment_id = NEW.segment_id
    UNION ALL
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_surface_nodes AS node
      JOIN inherited_chain AS successor
        ON successor.predecessor_node_id = node.node_id
  ),
  call_surface_chain(node_id, predecessor_node_id) AS (
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_surface_nodes AS node
     WHERE node.node_id = NEW.surface_node_id
    UNION ALL
    SELECT node.node_id, node.predecessor_node_id
      FROM console_trace_surface_nodes AS node
      JOIN call_surface_chain AS successor
        ON successor.predecessor_node_id = node.node_id
  )
  SELECT 1
    FROM console_trace_segments AS segment
    JOIN console_trace_surface_nodes AS surface
      ON surface.node_id = NEW.surface_node_id
    JOIN console_trace_request_headers AS header
      ON header.header_id = NEW.request_header_id
   WHERE segment.segment_id = NEW.segment_id
     AND header.provider_name = NEW.provider_name
     AND header.model_name = NEW.model_name
     AND header.route_identity = NEW.route_identity
     AND (
       (segment.parent_segment_id IS NULL AND
        surface.segment_id = NEW.segment_id) OR
       (segment.parent_segment_id IS NOT NULL AND (
         (surface.segment_id = NEW.segment_id AND EXISTS (
           SELECT 1 FROM call_surface_chain
            WHERE node_id = segment.inherited_surface_head_id
         )) OR
         EXISTS (
           SELECT 1 FROM inherited_chain
            WHERE node_id = surface.node_id
         )
       ))
     )
)
BEGIN
  SELECT RAISE(
    ABORT,
    'call binding must match reachable surface and immutable request header'
  );
END;

CREATE TRIGGER console_trace_calls_set_once_guard
BEFORE UPDATE ON console_trace_calls
WHEN
  (OLD.surface_node_id IS NOT NULL AND OLD.surface_node_id IS NOT NEW.surface_node_id) OR
  (OLD.request_header_id IS NOT NULL AND OLD.request_header_id IS NOT NEW.request_header_id) OR
  (OLD.provider_name IS NOT NULL AND OLD.provider_name IS NOT NEW.provider_name) OR
  (OLD.model_name IS NOT NULL AND OLD.model_name IS NOT NEW.model_name) OR
  (OLD.route_identity IS NOT NULL AND OLD.route_identity IS NOT NEW.route_identity) OR
  (OLD.dispatch_started_at IS NOT NULL AND OLD.dispatch_started_at IS NOT NEW.dispatch_started_at) OR
  (OLD.response_started_at IS NOT NULL AND OLD.response_started_at IS NOT NEW.response_started_at) OR
  (OLD.settled_at IS NOT NULL AND OLD.settled_at IS NOT NEW.settled_at) OR
  (OLD.provider_inactive_at IS NOT NULL AND OLD.provider_inactive_at IS NOT NEW.provider_inactive_at) OR
  (OLD.outcome IS NOT NULL AND OLD.outcome IS NOT NEW.outcome) OR
  (OLD.usage_json IS NOT NULL AND OLD.usage_json IS NOT NEW.usage_json) OR
  (OLD.omission_reason_code IS NOT NULL AND
   OLD.omission_reason_code IS NOT NEW.omission_reason_code) OR
  (OLD.integrity_state IS NOT NEW.integrity_state AND NOT (
    OLD.integrity_state = 'pending' AND
    NEW.integrity_state IN ('complete', 'incomplete')
  ))
BEGIN
  SELECT RAISE(ABORT, 'provider call settlement fields are set-once');
END;

CREATE TRIGGER console_trace_calls_terminal_guard
BEFORE UPDATE ON console_trace_calls
WHEN OLD.state IN (
  'not_dispatched', 'dispatch_unknown', 'complete', 'stopped', 'error',
  'interrupted', 'abandoned'
) AND (
  OLD.state IS NOT NEW.state OR
  OLD.surface_node_id IS NOT NEW.surface_node_id OR
  OLD.request_header_id IS NOT NEW.request_header_id OR
  OLD.provider_name IS NOT NEW.provider_name OR
  OLD.model_name IS NOT NEW.model_name OR
  OLD.route_identity IS NOT NEW.route_identity OR
  OLD.dispatch_started_at IS NOT NEW.dispatch_started_at OR
  OLD.response_started_at IS NOT NEW.response_started_at OR
  OLD.settled_at IS NOT NEW.settled_at OR
  OLD.provider_inactive_at IS NOT NEW.provider_inactive_at OR
  OLD.outcome IS NOT NEW.outcome OR
  OLD.usage_json IS NOT NEW.usage_json OR
  OLD.integrity_state IS NOT NEW.integrity_state OR
  OLD.omission_reason_code IS NOT NEW.omission_reason_code
)
BEGIN
  SELECT RAISE(ABORT, 'terminal provider call settlement is immutable');
END;

CREATE TRIGGER console_trace_response_links_owner_guard
BEFORE INSERT ON console_trace_response_links
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT call.segment_id, 0
      FROM console_trace_calls AS call
     WHERE call.call_id = NEW.call_id
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.owner_id, owner.conversation_id, owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1
    FROM console_trace_calls AS call
    JOIN effective_owner AS owner
      ON owner.owner_id = call.owner_id
   WHERE call.call_id = NEW.call_id
     AND owner.attached = 1
     AND (
       NEW.semantic_revision_id IS NULL OR EXISTS (
         SELECT 1
           FROM console_trace_semantic_revisions AS revision
          WHERE revision.revision_id = NEW.semantic_revision_id
            AND revision.source_conversation_id = owner.conversation_id
       )
     )
)
BEGIN
  SELECT RAISE(
    ABORT,
    'response link requires one active effective owner and matching revision domain'
  );
END;

CREATE TRIGGER console_trace_semantic_revisions_locator_only
BEFORE UPDATE ON console_trace_semantic_revisions
WHEN
  OLD.revision_id IS NOT NEW.revision_id OR
  OLD.source_conversation_id IS NOT NEW.source_conversation_id OR
  OLD.source_message_id IS NOT NEW.source_message_id OR
  OLD.revision_sequence IS NOT NEW.revision_sequence OR
  OLD.normalized_role IS NOT NEW.normalized_role OR
  OLD.content_kind IS NOT NEW.content_kind OR
  OLD.creation_reason IS NOT NEW.creation_reason OR
  OLD.predecessor_revision_id IS NOT NEW.predecessor_revision_id OR
  OLD.created_at IS NOT NEW.created_at OR
  NOT (
    OLD.live_message_id IS NOT NULL AND NEW.live_message_id IS NULL AND
    OLD.live_locator_retired_at IS NULL AND
    NEW.live_locator_retired_at IS NOT NULL
  )
BEGIN
  SELECT RAISE(ABORT, 'semantic revision updates are limited to locator retirement');
END;

CREATE TRIGGER console_trace_owners_detach_only
BEFORE UPDATE ON console_trace_owners
WHEN
  OLD.owner_id IS NOT NEW.owner_id OR
  OLD.root_segment_id IS NOT NEW.root_segment_id OR
  OLD.created_at IS NOT NEW.created_at OR
  NOT (
    OLD.attached = 1 AND NEW.attached = 0 AND
    OLD.conversation_id IS NOT NULL AND NEW.conversation_id IS NULL AND
    OLD.detached_at IS NULL AND NEW.detached_at IS NOT NULL
  )
BEGIN
  SELECT RAISE(ABORT, 'trace owner updates are limited to one-way detach');
END;

CREATE TRIGGER console_trace_migration_state_immutable_key
BEFORE UPDATE ON console_trace_migration_state
WHEN OLD.migration_name IS NOT NEW.migration_name
BEGIN
  SELECT RAISE(ABORT, 'trace migration identity is immutable');
END;

CREATE TRIGGER console_trace_maintenance_state_immutable_key
BEFORE UPDATE ON console_trace_maintenance_state
WHEN OLD.singleton_id IS NOT NEW.singleton_id
BEGIN
  SELECT RAISE(ABORT, 'trace maintenance singleton identity is immutable');
END;

CREATE TRIGGER console_trace_graph_epoch_monotonic
BEFORE UPDATE ON console_trace_graph_epoch
WHEN OLD.singleton_id IS NOT NEW.singleton_id OR NEW.epoch <> OLD.epoch + 1
BEGIN
  SELECT RAISE(ABORT, 'trace graph epoch must advance by exactly one');
END;

CREATE TRIGGER console_trace_artifacts_no_update
BEFORE UPDATE ON console_trace_artifacts BEGIN
  SELECT RAISE(ABORT, 'console_trace_artifacts is append-only');
END;
CREATE TRIGGER console_trace_events_no_update
BEFORE UPDATE ON console_trace_events BEGIN
  SELECT RAISE(ABORT, 'console_trace_events is append-only');
END;
CREATE TRIGGER console_trace_header_components_no_update
BEFORE UPDATE ON console_trace_header_components BEGIN
  SELECT RAISE(ABORT, 'console_trace_header_components is append-only');
END;
CREATE TRIGGER console_trace_policies_no_update
BEFORE UPDATE ON console_trace_policies BEGIN
  SELECT RAISE(ABORT, 'console_trace_policies is append-only');
END;
CREATE TRIGGER console_trace_redaction_spans_no_update
BEFORE UPDATE ON console_trace_redaction_spans BEGIN
  SELECT RAISE(ABORT, 'console_trace_redaction_spans is append-only');
END;
CREATE TRIGGER console_trace_request_headers_no_update
BEFORE UPDATE ON console_trace_request_headers BEGIN
  SELECT RAISE(ABORT, 'console_trace_request_headers is append-only');
END;
CREATE TRIGGER console_trace_response_links_no_update
BEFORE UPDATE ON console_trace_response_links BEGIN
  SELECT RAISE(ABORT, 'console_trace_response_links is append-only');
END;
CREATE TRIGGER console_trace_revision_bindings_no_update
BEFORE UPDATE ON console_trace_revision_bindings BEGIN
  SELECT RAISE(ABORT, 'console_trace_revision_bindings is append-only');
END;
CREATE TRIGGER console_trace_segments_no_update
BEFORE UPDATE ON console_trace_segments BEGIN
  SELECT RAISE(ABORT, 'console_trace_segments is append-only');
END;
CREATE TRIGGER console_trace_surface_nodes_no_update
BEFORE UPDATE ON console_trace_surface_nodes BEGIN
  SELECT RAISE(ABORT, 'console_trace_surface_nodes is append-only');
END;
CREATE TRIGGER console_trace_surface_replacements_no_update
BEFORE UPDATE ON console_trace_surface_replacements BEGIN
  SELECT RAISE(ABORT, 'console_trace_surface_replacements is append-only');
END;

CREATE TRIGGER console_trace_artifacts_no_delete
BEFORE DELETE ON console_trace_artifacts BEGIN
  SELECT RAISE(ABORT, 'console_trace_artifacts deletion prohibited');
END;
CREATE TRIGGER console_trace_calls_no_delete
BEFORE DELETE ON console_trace_calls BEGIN
  SELECT RAISE(ABORT, 'console_trace_calls deletion prohibited');
END;
CREATE TRIGGER console_trace_events_no_delete
BEFORE DELETE ON console_trace_events BEGIN
  SELECT RAISE(ABORT, 'console_trace_events deletion prohibited');
END;
CREATE TRIGGER console_trace_graph_epoch_no_delete
BEFORE DELETE ON console_trace_graph_epoch BEGIN
  SELECT RAISE(ABORT, 'console_trace_graph_epoch deletion prohibited');
END;
CREATE TRIGGER console_trace_header_components_no_delete
BEFORE DELETE ON console_trace_header_components BEGIN
  SELECT RAISE(ABORT, 'console_trace_header_components deletion prohibited');
END;
CREATE TRIGGER console_trace_maintenance_state_no_delete
BEFORE DELETE ON console_trace_maintenance_state BEGIN
  SELECT RAISE(ABORT, 'console_trace_maintenance_state deletion prohibited');
END;
CREATE TRIGGER console_trace_migration_state_no_delete
BEFORE DELETE ON console_trace_migration_state BEGIN
  SELECT RAISE(ABORT, 'console_trace_migration_state deletion prohibited');
END;
CREATE TRIGGER console_trace_owners_no_delete
BEFORE DELETE ON console_trace_owners BEGIN
  SELECT RAISE(ABORT, 'console_trace_owners deletion prohibited');
END;
CREATE TRIGGER console_trace_policies_no_delete
BEFORE DELETE ON console_trace_policies BEGIN
  SELECT RAISE(ABORT, 'console_trace_policies deletion prohibited');
END;
CREATE TRIGGER console_trace_redaction_spans_no_delete
BEFORE DELETE ON console_trace_redaction_spans BEGIN
  SELECT RAISE(ABORT, 'console_trace_redaction_spans deletion prohibited');
END;
CREATE TRIGGER console_trace_request_headers_no_delete
BEFORE DELETE ON console_trace_request_headers BEGIN
  SELECT RAISE(ABORT, 'console_trace_request_headers deletion prohibited');
END;
CREATE TRIGGER console_trace_response_links_no_delete
BEFORE DELETE ON console_trace_response_links BEGIN
  SELECT RAISE(ABORT, 'console_trace_response_links deletion prohibited');
END;
CREATE TRIGGER console_trace_revision_bindings_no_delete
BEFORE DELETE ON console_trace_revision_bindings BEGIN
  SELECT RAISE(ABORT, 'console_trace_revision_bindings deletion prohibited');
END;
CREATE TRIGGER console_trace_segments_no_delete
BEFORE DELETE ON console_trace_segments BEGIN
  SELECT RAISE(ABORT, 'console_trace_segments deletion prohibited');
END;
CREATE TRIGGER console_trace_semantic_revisions_no_delete
BEFORE DELETE ON console_trace_semantic_revisions BEGIN
  SELECT RAISE(ABORT, 'console_trace_semantic_revisions deletion prohibited');
END;
CREATE TRIGGER console_trace_surface_nodes_no_delete
BEFORE DELETE ON console_trace_surface_nodes BEGIN
  SELECT RAISE(ABORT, 'console_trace_surface_nodes deletion prohibited');
END;
CREATE TRIGGER console_trace_surface_replacements_no_delete
BEFORE DELETE ON console_trace_surface_replacements BEGIN
  SELECT RAISE(ABORT, 'console_trace_surface_replacements deletion prohibited');
END;
