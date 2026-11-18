-- ADR-126 / TASK-40608: bound newest-primary lookup without changing its rows.
BEGIN IMMEDIATE;
CREATE INDEX idx_agent_runs_newest_primary
ON agent_runs(conversation_id, created_at DESC, id DESC)
WHERE agent_kind = 'primary' AND status != 'superseded';
INSERT INTO schema_version(version) VALUES (32);
COMMIT;
