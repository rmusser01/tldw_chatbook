-- Independent Buddy authority; no Persona row or JSON mutation.
CREATE TABLE buddy_profiles (
    id TEXT PRIMARY KEY CHECK(length(id) BETWEEN 1 AND 128),
    name TEXT NOT NULL CHECK(length(name) BETWEEN 1 AND 256),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','archived','deleted')),
    source_key TEXT UNIQUE,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE buddy_visual_bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    buddy_id TEXT NOT NULL REFERENCES buddy_profiles(id),
    buddy_revision INTEGER NOT NULL CHECK(buddy_revision > 0),
    pack_id INTEGER NOT NULL REFERENCES persona_visual_packs(id),
    active_version_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','archived','deleted')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
    FOREIGN KEY(pack_id,active_version_id) REFERENCES persona_visual_pack_versions(pack_id,id)
);
CREATE UNIQUE INDEX idx_buddy_visual_bindings_active ON buddy_visual_bindings(buddy_id) WHERE status='active';
CREATE VIEW visual_owner_bindings AS
    SELECT id,persona_id,persona_revision,NULL AS buddy_id,NULL AS buddy_revision,
           pack_id,active_version_id,status,created_at,updated_at,version
      FROM persona_visual_bindings
    UNION ALL
    SELECT binding.id,NULL,0,binding.buddy_id,binding.buddy_revision,
           binding.pack_id,binding.active_version_id,binding.status,
           binding.created_at,binding.updated_at,binding.version
      FROM buddy_visual_bindings AS binding JOIN buddy_profiles AS buddy ON buddy.id=binding.buddy_id
     WHERE buddy.status='active' AND buddy.version=binding.buddy_revision;
