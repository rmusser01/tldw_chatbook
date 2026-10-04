-- Native rules remain device-local: no sync or FTS triggers, no Chat metadata.
CREATE TABLE console_response_rule_revisions (
    rule_id TEXT NOT NULL CHECK(length(rule_id) > 0),
    revision INTEGER NOT NULL CHECK(revision > 0),
    definition_json TEXT NOT NULL CHECK(length(CAST(definition_json AS BLOB)) <= 8192 AND json_valid(definition_json)),
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    origin_json TEXT NOT NULL CHECK(length(CAST(origin_json AS BLOB)) <= 8192 AND json_valid(origin_json)),
    created_at TEXT NOT NULL,
    PRIMARY KEY(rule_id, revision)
);
CREATE TABLE console_response_rule_bindings (
    scope_kind TEXT NOT NULL CHECK(scope_kind IN ('chat','workspace','global')),
    scope_id TEXT NOT NULL CHECK(length(scope_id) > 0),
    rule_id TEXT NOT NULL CHECK(length(rule_id) > 0),
    revision INTEGER,
    state TEXT NOT NULL CHECK(state IN ('enabled','disabled','excluded')),
    binding_revision INTEGER NOT NULL CHECK(binding_revision > 0),
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    PRIMARY KEY(scope_kind, scope_id, rule_id),
    FOREIGN KEY(rule_id, revision) REFERENCES console_response_rule_revisions(rule_id, revision),
    CHECK(state = 'excluded' OR revision > 0),
    CHECK(conversation_id IS NULL OR (scope_kind = 'chat' AND scope_id = conversation_id))
);
CREATE TABLE console_response_rule_validations (
    rule_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    validation_json TEXT NOT NULL CHECK(length(CAST(validation_json AS BLOB)) <= 32768 AND json_valid(validation_json)),
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    message_version INTEGER NOT NULL CHECK(message_version > 0),
    PRIMARY KEY(rule_id, revision),
    FOREIGN KEY(rule_id, revision) REFERENCES console_response_rule_revisions(rule_id, revision) ON DELETE CASCADE
);
CREATE TABLE console_response_rule_drafts (
    draft_id TEXT PRIMARY KEY NOT NULL,
    scope_kind TEXT NOT NULL CHECK(scope_kind IN ('chat','workspace','global')),
    scope_id TEXT NOT NULL,
    rule_id TEXT,
    revision INTEGER,
    source_json TEXT NOT NULL CHECK(length(CAST(source_json AS BLOB)) <= 8192 AND json_valid(source_json)),
    result_json TEXT NOT NULL CHECK(length(CAST(result_json AS BLOB)) <= 65536 AND json_valid(result_json)),
    complaint TEXT NOT NULL CHECK(length(CAST(complaint AS BLOB)) <= 8192),
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY(rule_id, revision) REFERENCES console_response_rule_revisions(rule_id, revision) ON DELETE CASCADE
);
CREATE INDEX idx_response_rule_drafts_scope ON console_response_rule_drafts(scope_kind, scope_id);
CREATE TABLE console_response_rule_fixtures (
    rule_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    case_id TEXT NOT NULL,
    case_type TEXT NOT NULL CHECK(case_type IN ('recorded_violation','synthetic_violation','synthetic_correction','synthetic_acceptable')),
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    message_version INTEGER NOT NULL CHECK(message_version > 0),
    input_json TEXT CHECK(input_json IS NULL OR (length(CAST(input_json AS BLOB)) <= 65536 AND json_valid(input_json))),
    PRIMARY KEY(rule_id, revision, case_id),
    FOREIGN KEY(rule_id, revision) REFERENCES console_response_rule_revisions(rule_id, revision) ON DELETE CASCADE,
    CHECK((case_type = 'recorded_violation' AND input_json IS NULL) OR (case_type != 'recorded_violation' AND input_json IS NOT NULL))
);
CREATE TABLE console_response_rule_assessments (
    assessment_id TEXT PRIMARY KEY NOT NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK(state IN ('pending','completed','cancelled','stale')),
    assessment_json TEXT NOT NULL CHECK(length(CAST(assessment_json AS BLOB)) <= 65536 AND json_valid(assessment_json))
);
CREATE INDEX idx_response_rule_assessments_source ON console_response_rule_assessments(conversation_id, message_id);
CREATE TABLE console_machine_followup_receipts (
    operation_id TEXT NOT NULL CHECK(length(operation_id) > 0),
    parent_turn_id TEXT NOT NULL CHECK(length(parent_turn_id) > 0),
    settlement_id TEXT NOT NULL CHECK(length(settlement_id) > 0),
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    assistant_message_id TEXT NOT NULL,
    chain_id TEXT NOT NULL CHECK(length(chain_id) > 0),
    admitted_turns INTEGER NOT NULL CHECK(admitted_turns BETWEEN 1 AND 3),
    native_turns INTEGER NOT NULL CHECK(native_turns BETWEEN 0 AND 2 AND native_turns <= admitted_turns),
    contributors_json TEXT NOT NULL CHECK(json_valid(contributors_json) AND length(contributors_json) <= 64),
    PRIMARY KEY(operation_id, parent_turn_id, settlement_id)
);
CREATE INDEX idx_machine_followup_receipts_conversation ON console_machine_followup_receipts(conversation_id);
-- Permanent native conversation deletion has no surviving local Chat binding.
-- Explicit promotions and independently owned inactive drafts remain roots.
CREATE TRIGGER console_response_rules_conversation_cleanup
AFTER DELETE ON conversations BEGIN
    DELETE FROM console_response_rule_revisions
    WHERE NOT EXISTS (
        SELECT 1 FROM console_response_rule_bindings b
        WHERE b.rule_id = console_response_rule_revisions.rule_id
          AND b.revision = console_response_rule_revisions.revision
    ) AND NOT EXISTS (
        SELECT 1 FROM console_response_rule_drafts d
        WHERE d.rule_id = console_response_rule_revisions.rule_id
          AND d.revision = console_response_rule_revisions.revision
    );
END;
