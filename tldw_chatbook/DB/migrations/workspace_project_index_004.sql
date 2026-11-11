CREATE TABLE semantic_control(singleton INTEGER PRIMARY KEY CHECK(singleton=1), namespace TEXT NOT NULL, epoch INTEGER NOT NULL, semantic_revision INTEGER NOT NULL, enabled INTEGER NOT NULL, settings_json TEXT, state TEXT NOT NULL, reason TEXT, ready_generation_id TEXT);
CREATE TABLE semantic_generations(generation_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, epoch INTEGER NOT NULL, processing_revision INTEGER NOT NULL, semantic_revision INTEGER NOT NULL, provider_id TEXT NOT NULL, model_id TEXT NOT NULL, destination_fingerprint TEXT NOT NULL, config_fingerprint TEXT NOT NULL, artifact_fingerprint TEXT NOT NULL, format_id TEXT NOT NULL, dimension INTEGER NOT NULL, manifest_revision INTEGER NOT NULL, claim_token TEXT, checkpoint_chunk_id INTEGER, state TEXT NOT NULL, chunk_count INTEGER NOT NULL);
CREATE TABLE semantic_members(generation_id TEXT NOT NULL REFERENCES semantic_generations(generation_id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, relative_path TEXT NOT NULL, identity TEXT NOT NULL, content_sha256 TEXT NOT NULL, line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(generation_id,ordinal));
CREATE INDEX semantic_members_path ON semantic_members(generation_id,relative_path,ordinal);
DROP TABLE schema_version;
CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=4));
INSERT INTO schema_version VALUES(4);
