-- TASK-42517: exact leading-column indexes for the voice-promotion message
-- locators classified after schema 72. v72 gave every census locator one, so
-- a larger history cannot turn one reconciliation into a table scan; the hook
-- receipt (38467b8fc6) and response-rule / machine follow-up (d24c4c04b2)
-- locators joined the census without them and planned as SCAN. The
-- `parent_assistant_message_id` and `message_id` columns are also `messages`
-- foreign keys, so a message delete no longer scans these tables either.
CREATE INDEX idx_hook_continuation_receipts_assistant_message
  ON console_hook_continuation_receipts(assistant_message_id);
CREATE INDEX idx_hook_continuation_receipts_parent_assistant_message
  ON console_hook_continuation_receipts(parent_assistant_message_id);
CREATE INDEX idx_machine_followup_receipts_assistant_message
  ON console_machine_followup_receipts(assistant_message_id);
CREATE INDEX idx_machine_followup_receipts_parent_assistant_message
  ON console_machine_followup_receipts(parent_assistant_message_id);
CREATE INDEX idx_response_rule_assessments_message
  ON console_response_rule_assessments(message_id);
CREATE INDEX idx_response_rule_drafts_message
  ON console_response_rule_drafts(message_id);
CREATE INDEX idx_response_rule_fixtures_message
  ON console_response_rule_fixtures(message_id);
CREATE INDEX idx_response_rule_validations_message
  ON console_response_rule_validations(message_id);
