-- Provenance normalization preserves historical timestamps.
DROP TRIGGER IF EXISTS update_subscriptions_timestamp;
-- Preserve only explicit user-entered authority. Absent/malformed lineage is unknown.
UPDATE subscriptions SET processing_options = json_set(
    CASE WHEN json_valid(processing_options) THEN
        CASE WHEN json_type(processing_options) = 'object' THEN processing_options ELSE '{}' END
    ELSE '{}' END,
    '$.url_provenance',
    CASE WHEN json_valid(processing_options) THEN
        CASE WHEN json_extract(processing_options, '$.url_provenance') = 'user_entered'
            THEN 'user_entered' ELSE 'unknown' END
    ELSE 'unknown' END
);
DELETE FROM schema_version;
INSERT INTO schema_version (version) VALUES (3);

CREATE TRIGGER update_subscriptions_timestamp
            AFTER UPDATE ON subscriptions
            BEGIN
                UPDATE subscriptions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
