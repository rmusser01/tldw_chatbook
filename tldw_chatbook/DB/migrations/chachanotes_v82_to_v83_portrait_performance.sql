CREATE TABLE character_portrait_performances (
    id INTEGER PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES character_cards(id) ON DELETE CASCADE,
    expression_key TEXT NOT NULL CHECK(length(expression_key) > 0),
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    source_kind TEXT NOT NULL CHECK(source_kind IN ('portrait','legacy_expression','visual_identity')),
    source_ref_json TEXT NOT NULL CHECK(json_valid(source_ref_json) AND length(source_ref_json) <= 128),
    schema_version INTEGER NOT NULL DEFAULT 1 CHECK(schema_version = 1),
    revision INTEGER NOT NULL CHECK(revision > 0),
    active INTEGER NOT NULL CHECK(active IN (0,1)),
    UNIQUE(character_id, expression_key)
);
CREATE TABLE character_portrait_performance_frames (
    performance_id INTEGER NOT NULL REFERENCES character_portrait_performances(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('rest','small','open','blink_rest','blink_small','blink_open')),
    png BLOB NOT NULL CHECK(typeof(png) = 'blob' AND length(png) BETWEEN 1 AND 4194304),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0 AND width * height <= 1048576),
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
    PRIMARY KEY(performance_id, role)
);
