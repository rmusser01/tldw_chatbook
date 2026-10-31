ALTER TABLE dream_stories ADD COLUMN goal TEXT;

INSERT OR IGNORE INTO schema_version(version) VALUES (4);
