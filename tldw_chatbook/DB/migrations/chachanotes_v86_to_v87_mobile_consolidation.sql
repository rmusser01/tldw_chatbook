CREATE TABLE mobile_consolidation_batches (
    operation_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    batch_key TEXT NOT NULL,
    mapping_json TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    PRIMARY KEY (operation_id, batch_key)
);
CREATE INDEX idx_hook_continuation_receipts_assistant_message ON console_hook_continuation_receipts(assistant_message_id);
CREATE INDEX idx_hook_continuation_receipts_parent_message ON console_hook_continuation_receipts(parent_assistant_message_id);
CREATE INDEX idx_machine_followup_receipts_assistant_message ON console_machine_followup_receipts(assistant_message_id);
CREATE INDEX idx_machine_followup_receipts_parent_message ON console_machine_followup_receipts(parent_assistant_message_id);
CREATE INDEX idx_response_rule_assessments_message ON console_response_rule_assessments(message_id);
CREATE INDEX idx_response_rule_drafts_message ON console_response_rule_drafts(message_id);
CREATE INDEX idx_response_rule_fixtures_message ON console_response_rule_fixtures(message_id);
CREATE INDEX idx_response_rule_validations_message ON console_response_rule_validations(message_id);
