-- ChaChaNotes v26 -> v27: canonical RAG citation provenance.
-- DDL only. The migration runner owns the transaction, identity row, and
-- schema-version update.

CREATE TABLE rag_identity_context(
  context_name TEXT PRIMARY KEY NOT NULL CHECK(context_name = 'default'),
  profile_id TEXT UNIQUE NOT NULL
    CHECK(length(CAST(profile_id AS BLOB)) BETWEEN 1 AND 256),
  local_authority_id TEXT UNIQUE NOT NULL
    CHECK(length(CAST(local_authority_id AS BLOB)) BETWEEN 1 AND 256),
  fingerprint_key_id TEXT UNIQUE NOT NULL
    CHECK(length(CAST(fingerprint_key_id AS BLOB)) BETWEEN 1 AND 256),
  created_at TEXT NOT NULL
);

CREATE TABLE rag_citation_traces(
  profile_id TEXT NOT NULL
    CHECK(length(CAST(profile_id AS BLOB)) BETWEEN 1 AND 256),
  trace_id TEXT NOT NULL
    CHECK(length(CAST(trace_id AS BLOB)) BETWEEN 1 AND 256),
  schema_version INTEGER NOT NULL CHECK(schema_version = 1),
  request_id TEXT NOT NULL
    CHECK(length(CAST(request_id AS BLOB)) BETWEEN 1 AND 256),
  generation_id TEXT NOT NULL
    CHECK(length(CAST(generation_id AS BLOB)) BETWEEN 1 AND 256),
  origin_scope_id TEXT NOT NULL
    CHECK(length(CAST(origin_scope_id AS BLOB)) BETWEEN 1 AND 256),
  origin TEXT NOT NULL
    CHECK(origin IN ('local','server','imported','legacy_inferred')),
  lifecycle TEXT NOT NULL CHECK(lifecycle = 'sealed'),
  completeness_at_seal TEXT NOT NULL
    CHECK(completeness_at_seal IN ('complete','partial','redacted','unavailable')),
  selected_attempt_id TEXT NOT NULL
    CHECK(length(CAST(selected_attempt_id AS BLOB)) BETWEEN 1 AND 256),
  policy_version TEXT NOT NULL
    CHECK(length(CAST(policy_version AS BLOB)) BETWEEN 1 AND 256),
  aggregate_json TEXT NOT NULL
    CHECK(length(CAST(aggregate_json AS BLOB)) <= 262144),
  visibility_state TEXT NOT NULL
    CHECK(visibility_state IN ('migrating','active')),
  created_at TEXT NOT NULL,
  sealed_at TEXT NOT NULL,
  connection_authority_id TEXT
    CHECK(connection_authority_id IS NULL OR length(CAST(connection_authority_id AS BLOB)) BETWEEN 1 AND 256),
  tenant_id TEXT
    CHECK(tenant_id IS NULL OR length(CAST(tenant_id AS BLOB)) BETWEEN 1 AND 256),
  server_trace_id TEXT
    CHECK(server_trace_id IS NULL OR length(CAST(server_trace_id AS BLOB)) BETWEEN 1 AND 256),
  wire_schema_version TEXT
    CHECK(wire_schema_version IS NULL OR length(CAST(wire_schema_version AS BLOB)) BETWEEN 1 AND 256),
  import_package_fingerprint TEXT
    CHECK(import_package_fingerprint IS NULL OR length(CAST(import_package_fingerprint AS BLOB)) BETWEEN 1 AND 256),
  external_trace_id TEXT
    CHECK(external_trace_id IS NULL OR length(CAST(external_trace_id AS BLOB)) BETWEEN 1 AND 256),
  legacy_conversation_id TEXT
    CHECK(legacy_conversation_id IS NULL OR length(CAST(legacy_conversation_id AS BLOB)) BETWEEN 1 AND 256),
  legacy_message_id TEXT
    CHECK(legacy_message_id IS NULL OR length(CAST(legacy_message_id AS BLOB)) BETWEEN 1 AND 256),
  PRIMARY KEY(profile_id, trace_id),
  CHECK(
    (
      origin = 'local'
      AND origin_scope_id = profile_id
      AND connection_authority_id IS NULL
      AND tenant_id IS NULL
      AND server_trace_id IS NULL
      AND wire_schema_version IS NULL
      AND import_package_fingerprint IS NULL
      AND external_trace_id IS NULL
      AND legacy_conversation_id IS NULL
      AND legacy_message_id IS NULL
    )
    OR (
      origin = 'server'
      AND connection_authority_id IS NOT NULL
      AND server_trace_id IS NOT NULL
      AND wire_schema_version IS NOT NULL
      AND origin_scope_id = COALESCE(tenant_id, 'authority-root')
      AND import_package_fingerprint IS NULL
      AND external_trace_id IS NULL
      AND legacy_conversation_id IS NULL
      AND legacy_message_id IS NULL
    )
    OR (
      origin = 'imported'
      AND origin_scope_id = import_package_fingerprint
      AND import_package_fingerprint IS NOT NULL
      AND external_trace_id IS NOT NULL
      AND connection_authority_id IS NULL
      AND tenant_id IS NULL
      AND server_trace_id IS NULL
      AND wire_schema_version IS NULL
      AND legacy_conversation_id IS NULL
      AND legacy_message_id IS NULL
    )
    OR (
      origin = 'legacy_inferred'
      AND origin_scope_id = profile_id
      AND legacy_conversation_id IS NOT NULL
      AND legacy_message_id IS NOT NULL
      AND connection_authority_id IS NULL
      AND tenant_id IS NULL
      AND server_trace_id IS NULL
      AND wire_schema_version IS NULL
      AND import_package_fingerprint IS NULL
      AND external_trace_id IS NULL
    )
  )
);

CREATE UNIQUE INDEX rag_citation_traces_server_identity_uq
ON rag_citation_traces(
  connection_authority_id,
  origin_scope_id,
  server_trace_id,
  wire_schema_version
)
WHERE origin = 'server';

CREATE UNIQUE INDEX rag_citation_traces_import_identity_uq
ON rag_citation_traces(
  profile_id,
  import_package_fingerprint,
  external_trace_id
)
WHERE origin = 'imported';

CREATE TABLE rag_evidence_runs(
  profile_id TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  run_id TEXT NOT NULL
    CHECK(length(CAST(run_id AS BLOB)) BETWEEN 1 AND 256),
  run_ordinal INTEGER NOT NULL CHECK(run_ordinal >= 0),
  stage TEXT NOT NULL CHECK(length(stage) BETWEEN 1 AND 256),
  redaction_state TEXT NOT NULL
    CHECK(redaction_state IN ('available','redacted','purged')),
  run_payload_json TEXT
    CHECK(run_payload_json IS NULL OR length(CAST(run_payload_json AS BLOB)) <= 4194304),
  started_at TEXT NOT NULL,
  ended_at TEXT,
  purged_at TEXT,
  PRIMARY KEY(profile_id, trace_id, run_id),
  UNIQUE(profile_id, trace_id, run_ordinal),
  FOREIGN KEY(profile_id, trace_id)
    REFERENCES rag_citation_traces(profile_id, trace_id) ON DELETE CASCADE,
  CHECK(
    (redaction_state = 'available' AND run_payload_json IS NOT NULL AND purged_at IS NULL)
    OR (redaction_state = 'redacted' AND run_payload_json IS NULL)
    OR (redaction_state = 'purged' AND run_payload_json IS NULL AND purged_at IS NOT NULL)
  )
);

CREATE TABLE rag_evidence_snapshots(
  profile_id TEXT NOT NULL
    CHECK(length(CAST(profile_id AS BLOB)) BETWEEN 1 AND 256),
  payload_id TEXT NOT NULL
    CHECK(length(CAST(payload_id AS BLOB)) BETWEEN 1 AND 256),
  governance_scope_id TEXT NOT NULL
    CHECK(length(CAST(governance_scope_id AS BLOB)) BETWEEN 1 AND 256),
  authority_id TEXT NOT NULL
    CHECK(length(CAST(authority_id AS BLOB)) BETWEEN 1 AND 256),
  confidentiality_policy_id TEXT NOT NULL
    CHECK(length(CAST(confidentiality_policy_id AS BLOB)) BETWEEN 1 AND 256),
  revocation_scope_id TEXT NOT NULL
    CHECK(length(CAST(revocation_scope_id AS BLOB)) BETWEEN 1 AND 256),
  origin_namespace TEXT NOT NULL
    CHECK(length(CAST(origin_namespace AS BLOB)) BETWEEN 1 AND 256),
  origin_payload_id TEXT NOT NULL
    CHECK(length(CAST(origin_payload_id AS BLOB)) BETWEEN 1 AND 256),
  storage_mode TEXT NOT NULL
    CHECK(storage_mode IN ('embedded','server_reference','ephemeral','redacted')),
  redaction_state TEXT NOT NULL
    CHECK(redaction_state IN ('available','redacted','purged')),
  retention_class TEXT NOT NULL CHECK(length(retention_class) BETWEEN 1 AND 256),
  snapshot_text TEXT
    CHECK(snapshot_text IS NULL OR length(CAST(snapshot_text AS BLOB)) <= 65536),
  title TEXT,
  source_identity_json TEXT
    CHECK(source_identity_json IS NULL OR length(CAST(source_identity_json AS BLOB)) <= 16384),
  locator_json TEXT
    CHECK(locator_json IS NULL OR length(CAST(locator_json AS BLOB)) <= 16384),
  lineage_json TEXT
    CHECK(lineage_json IS NULL OR length(CAST(lineage_json AS BLOB)) <= 16384),
  transformations_json TEXT
    CHECK(transformations_json IS NULL OR length(CAST(transformations_json AS BLOB)) <= 16384),
  content_hash TEXT
    CHECK(content_hash IS NULL OR length(CAST(content_hash AS BLOB)) BETWEEN 1 AND 256),
  comparison_fingerprint TEXT
    CHECK(comparison_fingerprint IS NULL OR length(CAST(comparison_fingerprint AS BLOB)) BETWEEN 1 AND 256),
  created_at TEXT NOT NULL,
  retain_until TEXT,
  purged_at TEXT,
  PRIMARY KEY(profile_id, payload_id),
  CHECK(
    (redaction_state = 'available' AND purged_at IS NULL)
    OR (redaction_state = 'redacted')
    OR (
      redaction_state = 'purged'
      AND snapshot_text IS NULL
      AND title IS NULL
      AND source_identity_json IS NULL
      AND locator_json IS NULL
      AND lineage_json IS NULL
      AND transformations_json IS NULL
      AND content_hash IS NULL
      AND comparison_fingerprint IS NULL
      AND purged_at IS NOT NULL
    )
  )
);

CREATE UNIQUE INDEX rag_evidence_snapshots_content_dedupe_uq
ON rag_evidence_snapshots(
  governance_scope_id,
  authority_id,
  confidentiality_policy_id,
  revocation_scope_id,
  content_hash
)
WHERE content_hash IS NOT NULL;

CREATE TABLE rag_answer_attempt_payloads(
  profile_id TEXT NOT NULL,
  payload_id TEXT NOT NULL
    CHECK(length(CAST(payload_id AS BLOB)) BETWEEN 1 AND 256),
  trace_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL
    CHECK(length(CAST(attempt_id AS BLOB)) BETWEEN 1 AND 256),
  redaction_state TEXT NOT NULL
    CHECK(redaction_state IN ('available','purged')),
  retention_class TEXT NOT NULL CHECK(length(retention_class) BETWEEN 1 AND 256),
  answer_body TEXT
    CHECK(answer_body IS NULL OR length(CAST(answer_body AS BLOB)) <= 1048576),
  body_integrity_hmac TEXT
    CHECK(body_integrity_hmac IS NULL OR length(CAST(body_integrity_hmac AS BLOB)) BETWEEN 1 AND 256),
  created_at TEXT NOT NULL,
  retain_until TEXT,
  purged_at TEXT,
  PRIMARY KEY(profile_id, payload_id),
  UNIQUE(profile_id, trace_id, attempt_id),
  FOREIGN KEY(profile_id, trace_id)
    REFERENCES rag_citation_traces(profile_id, trace_id) ON DELETE CASCADE,
  CHECK(
    (
      redaction_state = 'available'
      AND answer_body IS NOT NULL
      AND body_integrity_hmac IS NOT NULL
      AND purged_at IS NULL
    )
    OR (
      redaction_state = 'purged'
      AND answer_body IS NULL
      AND body_integrity_hmac IS NULL
      AND purged_at IS NOT NULL
    )
  )
);

CREATE TABLE rag_trace_evidence_refs(
  profile_id TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  prompt_set_id TEXT NOT NULL
    CHECK(length(CAST(prompt_set_id AS BLOB)) BETWEEN 1 AND 256),
  evidence_ordinal INTEGER NOT NULL CHECK(evidence_ordinal >= 0),
  run_id TEXT NOT NULL,
  snapshot_payload_id TEXT NOT NULL,
  marker_ordinal INTEGER NOT NULL CHECK(marker_ordinal >= 0),
  storage_mode TEXT NOT NULL
    CHECK(storage_mode IN ('embedded','server_reference','ephemeral','redacted')),
  PRIMARY KEY(profile_id, trace_id, prompt_set_id, evidence_ordinal),
  UNIQUE(profile_id, trace_id, prompt_set_id, marker_ordinal),
  FOREIGN KEY(profile_id, trace_id)
    REFERENCES rag_citation_traces(profile_id, trace_id) ON DELETE CASCADE,
  FOREIGN KEY(profile_id, trace_id, run_id)
    REFERENCES rag_evidence_runs(profile_id, trace_id, run_id) ON DELETE CASCADE,
  FOREIGN KEY(profile_id, snapshot_payload_id)
    REFERENCES rag_evidence_snapshots(profile_id, payload_id) ON DELETE RESTRICT
);

CREATE TABLE rag_message_trace_owners(
  profile_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  message_revision INTEGER NOT NULL CHECK(message_revision >= 0),
  trace_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('active','body_mismatch','deleted')),
  body_fingerprint TEXT NOT NULL
    CHECK(length(CAST(body_fingerprint AS BLOB)) BETWEEN 1 AND 256),
  idempotency_key TEXT NOT NULL
    CHECK(length(CAST(idempotency_key AS BLOB)) BETWEEN 1 AND 256),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(profile_id, message_id, message_revision, trace_id),
  UNIQUE(profile_id, idempotency_key),
  FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
  FOREIGN KEY(profile_id, trace_id)
    REFERENCES rag_citation_traces(profile_id, trace_id) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX rag_message_trace_owners_active_message_uq
ON rag_message_trace_owners(profile_id, message_id, message_revision)
WHERE state = 'active';

CREATE TABLE rag_source_observations(
  profile_id TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  prompt_set_id TEXT NOT NULL
    CHECK(length(CAST(prompt_set_id AS BLOB)) BETWEEN 1 AND 256),
  evidence_ordinal INTEGER NOT NULL CHECK(evidence_ordinal >= 0),
  snapshot_payload_id TEXT NOT NULL,
  resolver_kind TEXT NOT NULL
    CHECK(length(CAST(resolver_kind AS BLOB)) BETWEEN 1 AND 256),
  resolver_version TEXT NOT NULL
    CHECK(length(CAST(resolver_version AS BLOB)) BETWEEN 1 AND 256),
  availability TEXT NOT NULL CHECK(length(availability) BETWEEN 1 AND 256),
  permission_state TEXT NOT NULL CHECK(length(permission_state) BETWEEN 1 AND 256),
  content_state TEXT NOT NULL CHECK(length(content_state) BETWEEN 1 AND 256),
  location_state TEXT NOT NULL CHECK(length(location_state) BETWEEN 1 AND 256),
  capabilities_json TEXT NOT NULL
    CHECK(length(CAST(capabilities_json AS BLOB)) <= 8192),
  request_nonce TEXT NOT NULL
    CHECK(length(CAST(request_nonce AS BLOB)) BETWEEN 1 AND 256),
  observed_at TEXT NOT NULL,
  error_code TEXT CHECK(error_code IS NULL OR length(error_code) <= 256),
  PRIMARY KEY(
    profile_id,
    trace_id,
    prompt_set_id,
    evidence_ordinal,
    snapshot_payload_id,
    resolver_kind,
    resolver_version
  ),
  FOREIGN KEY(profile_id, trace_id)
    REFERENCES rag_citation_traces(profile_id, trace_id) ON DELETE CASCADE,
  FOREIGN KEY(profile_id, snapshot_payload_id)
    REFERENCES rag_evidence_snapshots(profile_id, payload_id) ON DELETE CASCADE
);

CREATE TABLE rag_payload_tombstones(
  profile_id TEXT NOT NULL
    CHECK(length(CAST(profile_id AS BLOB)) BETWEEN 1 AND 256),
  origin_namespace TEXT NOT NULL
    CHECK(length(CAST(origin_namespace AS BLOB)) BETWEEN 1 AND 256),
  origin_payload_id TEXT NOT NULL
    CHECK(length(CAST(origin_payload_id AS BLOB)) BETWEEN 1 AND 256),
  revocation_scope_id TEXT NOT NULL
    CHECK(length(CAST(revocation_scope_id AS BLOB)) BETWEEN 1 AND 256),
  reason_code TEXT NOT NULL CHECK(length(reason_code) BETWEEN 1 AND 256),
  policy_version TEXT NOT NULL
    CHECK(length(CAST(policy_version AS BLOB)) BETWEEN 1 AND 256),
  revoked_at TEXT NOT NULL,
  retain_until TEXT NOT NULL,
  PRIMARY KEY(profile_id, origin_namespace, origin_payload_id)
);

CREATE TABLE rag_artifact_owner_leases(
  profile_id TEXT NOT NULL,
  artifact_store_id TEXT NOT NULL
    CHECK(length(CAST(artifact_store_id AS BLOB)) BETWEEN 1 AND 256),
  artifact_id TEXT NOT NULL
    CHECK(length(CAST(artifact_id AS BLOB)) BETWEEN 1 AND 256),
  artifact_revision INTEGER NOT NULL CHECK(artifact_revision >= 0),
  trace_id TEXT NOT NULL,
  lease_id TEXT UNIQUE NOT NULL
    CHECK(length(CAST(lease_id AS BLOB)) BETWEEN 1 AND 256),
  state TEXT NOT NULL
    CHECK(state IN ('link_pending','live','unlink_pending','released')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  retain_until TEXT,
  PRIMARY KEY(profile_id, artifact_store_id, artifact_id, artifact_revision, trace_id),
  FOREIGN KEY(profile_id, trace_id)
    REFERENCES rag_citation_traces(profile_id, trace_id) ON DELETE RESTRICT
);

CREATE TABLE rag_artifact_owner_operations(
  profile_id TEXT NOT NULL,
  operation_id TEXT NOT NULL
    CHECK(length(CAST(operation_id AS BLOB)) BETWEEN 1 AND 256),
  artifact_store_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL,
  artifact_revision INTEGER NOT NULL CHECK(artifact_revision >= 0),
  trace_id TEXT NOT NULL,
  operation_kind TEXT NOT NULL CHECK(operation_kind IN ('link','unlink')),
  state TEXT NOT NULL CHECK(state IN ('pending','applied','acknowledged')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(profile_id, operation_id),
  UNIQUE(
    profile_id,
    artifact_store_id,
    artifact_id,
    artifact_revision,
    trace_id,
    operation_kind
  ),
  FOREIGN KEY(profile_id, artifact_store_id, artifact_id, artifact_revision, trace_id)
    REFERENCES rag_artifact_owner_leases(
      profile_id,
      artifact_store_id,
      artifact_id,
      artifact_revision,
      trace_id
    ) ON DELETE RESTRICT
);

CREATE TABLE rag_legacy_migration_journal(
  profile_id TEXT NOT NULL
    CHECK(length(CAST(profile_id AS BLOB)) BETWEEN 1 AND 256),
  conversation_id TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL
    CHECK(length(CAST(source_fingerprint AS BLOB)) BETWEEN 1 AND 256),
  state TEXT NOT NULL
    CHECK(state IN ('pending','running','complete','failed','diverged')),
  attempt_count INTEGER NOT NULL CHECK(attempt_count >= 0),
  started_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  next_message_cursor TEXT
    CHECK(next_message_cursor IS NULL OR length(CAST(next_message_cursor AS BLOB)) BETWEEN 1 AND 256),
  error_code TEXT CHECK(error_code IS NULL OR length(error_code) <= 256),
  completed_at TEXT,
  PRIMARY KEY(profile_id, conversation_id),
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
