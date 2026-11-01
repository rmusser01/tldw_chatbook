-- Required native workflow checkpoints (ADR-285).
PRAGMA synchronous=FULL;
BEGIN IMMEDIATE;
CREATE TABLE workflow_checkpoints (attempt_id TEXT PRIMARY KEY REFERENCES workflow_results(attempt_id), workflow_id TEXT NOT NULL REFERENCES workflow_runs(id), payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072));
CREATE TRIGGER workflow_checkpoint_immutable BEFORE UPDATE ON workflow_checkpoints BEGIN SELECT RAISE(ABORT, 'workflow checkpoint is immutable'); END;
CREATE TABLE workflow_resume_commands (id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL REFERENCES workflow_runs(id), control_revision INTEGER NOT NULL CHECK(control_revision>0), resulting_revision INTEGER NOT NULL CHECK(resulting_revision>control_revision));
INSERT INTO schema_version(version) VALUES (27);
COMMIT;
