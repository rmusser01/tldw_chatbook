-- ADR-285: exact repair-command equality metadata, never live authority.
BEGIN IMMEDIATE;
ALTER TABLE workflow_phase_attempts ADD COLUMN repair_instruction_sha256 TEXT CHECK(repair_instruction_sha256 IS NULL OR (typeof(repair_instruction_sha256)='text' AND length(CAST(repair_instruction_sha256 AS BLOB))=64 AND repair_instruction_sha256 NOT GLOB '*[^0-9a-f]*' AND purpose='planning' AND plan_revision IS NOT NULL));
CREATE TRIGGER workflow_repair_input_immutable
BEFORE UPDATE OF id,workflow_id,plan_revision,phase_id,purpose,command_id,native_attempt_id,repair_instruction_sha256 ON workflow_phase_attempts
WHEN OLD.repair_instruction_sha256 IS NOT NULL OR NEW.repair_instruction_sha256 IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'workflow repair input is immutable'); END;
INSERT INTO schema_version(version) VALUES (29);
COMMIT;
