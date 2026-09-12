-- ChaChaNotes v44 -> v45: portable Actor Pack identity and Persona intents.
--
-- Persona JSON remains authoritative. The intent table holds only one bounded
-- actor mutation long enough to coordinate the JSON replace with the registry
-- and optional visual rows written in one SQLite transaction.

CREATE TABLE IF NOT EXISTS actor_portable_identities (
    actor_kind TEXT NOT NULL
        CHECK(actor_kind IN ('character', 'persona')),
    local_actor_id TEXT NOT NULL
        CHECK(length(local_actor_id) BETWEEN 1 AND 200
              AND instr(local_actor_id, char(0)) = 0),
    portable_uuid TEXT NOT NULL UNIQUE
        CHECK(length(portable_uuid) = 36
              AND portable_uuid = lower(portable_uuid)
              AND portable_uuid NOT GLOB '*[^0-9a-f-]*'
              AND length(replace(portable_uuid, '-', '')) = 32
              AND substr(portable_uuid, 9, 1) = '-'
              AND substr(portable_uuid, 14, 1) = '-'
              AND substr(portable_uuid, 15, 1) = '4'
              AND substr(portable_uuid, 19, 1) = '-'
              AND substr(portable_uuid, 20, 1) IN ('8', '9', 'a', 'b')
              AND substr(portable_uuid, 24, 1) = '-'),
    source_portable_uuid TEXT
        CHECK(source_portable_uuid IS NULL OR (
              length(source_portable_uuid) = 36
              AND source_portable_uuid = lower(source_portable_uuid)
              AND source_portable_uuid NOT GLOB '*[^0-9a-f-]*'
              AND length(replace(source_portable_uuid, '-', '')) = 32
              AND substr(source_portable_uuid, 9, 1) = '-'
              AND substr(source_portable_uuid, 14, 1) = '-'
              AND substr(source_portable_uuid, 15, 1) = '4'
              AND substr(source_portable_uuid, 19, 1) = '-'
              AND substr(source_portable_uuid, 20, 1) IN ('8', '9', 'a', 'b')
              AND substr(source_portable_uuid, 24, 1) = '-'
              AND source_portable_uuid <> portable_uuid)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
    PRIMARY KEY(actor_kind, local_actor_id)
);

CREATE TABLE IF NOT EXISTS actor_pack_persona_intents (
    intent_id TEXT PRIMARY KEY
        CHECK(length(intent_id) = 32
              AND intent_id = lower(intent_id)
              AND intent_id NOT GLOB '*[^0-9a-f]*'),
    persona_id TEXT NOT NULL
        CHECK(length(persona_id) BETWEEN 1 AND 200
              AND instr(persona_id, char(0)) = 0),
    operation TEXT NOT NULL CHECK(operation IN ('create', 'copy', 'update')),
    state TEXT NOT NULL CHECK(state IN ('prepared', 'committed', 'quarantined')),
    old_profile_json TEXT CHECK(old_profile_json IS NULL OR length(old_profile_json) <= 2097152),
    new_profile_json TEXT NOT NULL CHECK(length(new_profile_json) <= 2097152),
    old_profile_sha256 TEXT
        CHECK(old_profile_sha256 IS NULL OR (
              length(old_profile_sha256) = 64
              AND old_profile_sha256 NOT GLOB '*[^0-9a-f]*')),
    new_profile_sha256 TEXT NOT NULL
        CHECK(length(new_profile_sha256) = 64
              AND new_profile_sha256 NOT GLOB '*[^0-9a-f]*'),
    old_store_sha256 TEXT NOT NULL
        CHECK(length(old_store_sha256) = 64
              AND old_store_sha256 NOT GLOB '*[^0-9a-f]*'),
    new_store_sha256 TEXT NOT NULL
        CHECK(length(new_store_sha256) = 64
              AND new_store_sha256 NOT GLOB '*[^0-9a-f]*'),
    old_registry_uuid TEXT
        CHECK(old_registry_uuid IS NULL OR (
              length(old_registry_uuid) = 36
              AND old_registry_uuid = lower(old_registry_uuid)
              AND old_registry_uuid NOT GLOB '*[^0-9a-f-]*'
              AND length(replace(old_registry_uuid, '-', '')) = 32
              AND substr(old_registry_uuid, 9, 1) = '-'
              AND substr(old_registry_uuid, 14, 1) = '-'
              AND substr(old_registry_uuid, 15, 1) = '4'
              AND substr(old_registry_uuid, 19, 1) = '-'
              AND substr(old_registry_uuid, 20, 1) IN ('8', '9', 'a', 'b')
              AND substr(old_registry_uuid, 24, 1) = '-')),
    new_registry_uuid TEXT NOT NULL
        CHECK(length(new_registry_uuid) = 36
              AND new_registry_uuid = lower(new_registry_uuid)
              AND new_registry_uuid NOT GLOB '*[^0-9a-f-]*'
              AND length(replace(new_registry_uuid, '-', '')) = 32
              AND substr(new_registry_uuid, 9, 1) = '-'
              AND substr(new_registry_uuid, 14, 1) = '-'
              AND substr(new_registry_uuid, 15, 1) = '4'
              AND substr(new_registry_uuid, 19, 1) = '-'
              AND substr(new_registry_uuid, 20, 1) IN ('8', '9', 'a', 'b')
              AND substr(new_registry_uuid, 24, 1) = '-'),
    quarantine_reason TEXT
        CHECK(quarantine_reason IS NULL OR (
              length(quarantine_reason) BETWEEN 1 AND 64
              AND quarantine_reason NOT GLOB '*[^a-z0-9_]*')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_actor_pack_persona_intents_state
    ON actor_pack_persona_intents(state, created_at, intent_id);
