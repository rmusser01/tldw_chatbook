ALTER TABLE dream_tracked_items ADD COLUMN egress_marked INTEGER NOT NULL DEFAULT 0;

ALTER TABLE dream_tracked_items ADD COLUMN alert_rule_id INTEGER;

ALTER TABLE dream_tracked_items ADD COLUMN reminder_event_date TEXT;

INSERT OR IGNORE INTO schema_version(version) VALUES (6);
