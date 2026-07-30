-- ChaChaNotes v27 -> v28: local conversation character authority.
-- DDL only. The migration runner owns the transaction, local-only backfill,
-- and schema-version update. Existing Sync V2 triggers remain unchanged.

ALTER TABLE conversations
ADD COLUMN assistant_authority_id TEXT
  CHECK(
    assistant_authority_id IS NULL
    OR (
      typeof(assistant_authority_id) = 'text'
      AND length(CAST(assistant_authority_id AS BLOB)) BETWEEN 1 AND 256
    )
  );
