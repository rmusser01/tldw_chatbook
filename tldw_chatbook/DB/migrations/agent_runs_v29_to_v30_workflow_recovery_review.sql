-- ADR-356: recovery review remains available when private payload quota is full.
BEGIN IMMEDIATE;
ALTER TABLE workflow_runs ADD COLUMN recovery_reviewed_revision INTEGER NOT NULL DEFAULT 0 CHECK(typeof(recovery_reviewed_revision)='integer' AND recovery_reviewed_revision>=0 AND recovery_reviewed_revision<=control_revision AND (recovery_reviewed_revision=0 OR status IN ('closed','removed')));
CREATE TRIGGER workflow_recovery_review_immutable
BEFORE UPDATE OF recovery_reviewed_revision ON workflow_runs
WHEN OLD.recovery_reviewed_revision>0 AND NEW.recovery_reviewed_revision!=OLD.recovery_reviewed_revision
BEGIN SELECT RAISE(ABORT, 'workflow recovery review is immutable'); END;
INSERT INTO schema_version(version) VALUES (30);
COMMIT;
