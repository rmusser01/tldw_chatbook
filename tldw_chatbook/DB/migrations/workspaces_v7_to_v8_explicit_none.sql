-- ADR-139: omitted defaults may be provisioned; an explicit None must remain absent.
BEGIN IMMEDIATE;
ALTER TABLE workspace_records ADD COLUMN assistant_defaults_explicit_none
    INTEGER NOT NULL DEFAULT 0 CHECK (assistant_defaults_explicit_none IN (0, 1));
INSERT OR IGNORE INTO schema_version (version) VALUES (8);
COMMIT;
