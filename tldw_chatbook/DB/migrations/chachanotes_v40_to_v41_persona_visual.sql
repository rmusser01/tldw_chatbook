-- Migration: ChaChaNotes V40 to V41 local Persona Visual runtime schema.
--
-- Persona JSON remains authoritative. These separate tables store only the
-- local Persona id/revision snapshot and one immutable visual graph.

CREATE TABLE persona_visual_packs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL CHECK(title <> ''),
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'archived', 'deleted')),
    active_version_id INTEGER,
    source_kind TEXT NOT NULL DEFAULT 'manual' CHECK(source_kind <> ''),
    source_context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
    FOREIGN KEY(id, active_version_id)
        REFERENCES persona_visual_pack_versions(pack_id, id)
);

CREATE TABLE persona_visual_pack_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_id INTEGER NOT NULL REFERENCES persona_visual_packs(id),
    version_number INTEGER NOT NULL CHECK(version_number > 0),
    renderer_type TEXT NOT NULL CHECK(renderer_type <> ''),
    manifest_version INTEGER NOT NULL CHECK(manifest_version > 0),
    manifest_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL
        CHECK(length(manifest_sha256) = 64
              AND manifest_sha256 NOT GLOB '*[^0-9a-f]*'),
    storage_relpath TEXT NOT NULL
        CHECK(storage_relpath <> ''
              AND substr(storage_relpath, 1, 1) <> '/'
              AND instr(storage_relpath, char(92)) = 0
              AND storage_relpath NOT GLOB '[A-Za-z]:*'
              AND storage_relpath <> '..'
              AND storage_relpath NOT LIKE '../%'
              AND storage_relpath NOT LIKE '%/../%'
              AND storage_relpath NOT LIKE '%/..'),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pack_id, version_number),
    UNIQUE(pack_id, id)
);

CREATE TABLE persona_visual_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_id INTEGER NOT NULL REFERENCES persona_visual_packs(id),
    pack_version_id INTEGER NOT NULL,
    asset_key TEXT NOT NULL CHECK(asset_key <> ''),
    role TEXT NOT NULL CHECK(role <> ''),
    storage_relpath TEXT NOT NULL
        CHECK(storage_relpath <> ''
              AND substr(storage_relpath, 1, 1) <> '/'
              AND instr(storage_relpath, char(92)) = 0
              AND storage_relpath NOT GLOB '[A-Za-z]:*'
              AND storage_relpath <> '..'
              AND storage_relpath NOT LIKE '../%'
              AND storage_relpath NOT LIKE '%/../%'
              AND storage_relpath NOT LIKE '%/..'),
    mime_type TEXT NOT NULL CHECK(mime_type <> ''),
    bytes INTEGER NOT NULL CHECK(bytes > 0),
    sha256 TEXT NOT NULL
        CHECK(length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    frame_count INTEGER CHECK(frame_count IS NULL OR frame_count > 0),
    duration_ms INTEGER CHECK(duration_ms IS NULL OR duration_ms > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(pack_id, pack_version_id)
        REFERENCES persona_visual_pack_versions(pack_id, id)
);

CREATE TABLE persona_visual_bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id TEXT NOT NULL CHECK(persona_id <> ''),
    persona_revision INTEGER NOT NULL CHECK(persona_revision >= 0),
    pack_id INTEGER NOT NULL REFERENCES persona_visual_packs(id),
    active_version_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'archived', 'deleted')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
    FOREIGN KEY(pack_id, active_version_id)
        REFERENCES persona_visual_pack_versions(pack_id, id)
);

CREATE UNIQUE INDEX idx_persona_visual_assets_version_key
    ON persona_visual_assets(pack_version_id, asset_key);
CREATE UNIQUE INDEX idx_persona_visual_bindings_persona_active
    ON persona_visual_bindings(persona_id)
    WHERE status = 'active';

UPDATE db_schema_version
   SET version = 41
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 40;
