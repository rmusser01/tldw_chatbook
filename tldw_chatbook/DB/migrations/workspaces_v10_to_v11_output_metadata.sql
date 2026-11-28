ALTER TABLE workspace_memberships ADD COLUMN metadata_json TEXT
    CHECK (metadata_json IS NULL OR (
        role = 'output'
        AND CASE WHEN json_valid(metadata_json) THEN json_type(metadata_json) = 'object' ELSE 0 END
        AND length(CAST(metadata_json AS BLOB)) <= 32768
    ));
ALTER TABLE workspace_memberships ADD COLUMN updated_at TEXT
    CHECK (updated_at IS NULL OR role = 'output');
