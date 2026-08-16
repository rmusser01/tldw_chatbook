-- Migration: ChaChaNotes V38 to V39 local Visual Identity schema.
--
-- This is the activated-pack subset of the server schema pinned by ADR-067.
-- Local assets always belong to an immutable pack version; draft, job, and
-- idempotency storage remain server-only.

CREATE TABLE visual_identity_packs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'archived', 'deleted')),
    active_version_id INTEGER REFERENCES visual_identity_pack_versions(id),
    default_expression_key TEXT NOT NULL DEFAULT 'neutral',
    source_kind TEXT NOT NULL DEFAULT 'manual',
    source_context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE visual_identity_pack_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_id INTEGER NOT NULL REFERENCES visual_identity_packs(id),
    owner_user_id INTEGER NOT NULL,
    version_number INTEGER NOT NULL,
    default_expression_key TEXT NOT NULL DEFAULT 'neutral',
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pack_id, version_number)
);

CREATE TABLE visual_identity_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    pack_id INTEGER REFERENCES visual_identity_packs(id),
    pack_version_id INTEGER NOT NULL REFERENCES visual_identity_pack_versions(id),
    expression_key TEXT NOT NULL,
    original_expression_key TEXT NOT NULL DEFAULT '',
    display_label TEXT NOT NULL DEFAULT '',
    source_filename TEXT NOT NULL,
    storage_relpath TEXT NOT NULL,
    content_type TEXT NOT NULL,
    bytes INTEGER NOT NULL CHECK(bytes > 0),
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    source_context_json TEXT NOT NULL DEFAULT '{}',
    is_animated INTEGER NOT NULL DEFAULT 0 CHECK(is_animated IN (0, 1)),
    frame_count INTEGER,
    duration_ms INTEGER,
    preview_relpath TEXT,
    deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE visual_identity_bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    actor_kind TEXT NOT NULL CHECK(actor_kind IN ('character', 'persona')),
    actor_id TEXT NOT NULL,
    pack_id INTEGER NOT NULL REFERENCES visual_identity_packs(id),
    active_version_id INTEGER NOT NULL REFERENCES visual_identity_pack_versions(id),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'deleted')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_visual_identity_packs_owner_status
    ON visual_identity_packs(owner_user_id, status);
CREATE INDEX idx_visual_identity_assets_pack_expression
    ON visual_identity_assets(pack_id, pack_version_id, expression_key, deleted);
CREATE UNIQUE INDEX idx_visual_identity_bindings_actor_active
    ON visual_identity_bindings(owner_user_id, actor_kind, actor_id)
    WHERE status = 'active';

UPDATE db_schema_version
   SET version = 39
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 38;
