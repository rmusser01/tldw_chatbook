ALTER TABLE dream_tracked_items ADD COLUMN suspension TEXT;

INSERT OR IGNORE INTO schema_version(version) VALUES (5);
