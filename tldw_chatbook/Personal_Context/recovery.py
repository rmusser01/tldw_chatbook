"""Every SQL catalog a Personal Context repository at schema v9 can have.

The repository migrates in place with ``ALTER TABLE`` (``repository.py``
``_inspect_schema``), so a fresh v9 database and a migrated one differ
textually, and backup validation compares the exact catalog. These are the
reachable layouts, in ``sqlite_schema`` order (type, name): the fresh one and
one per historical fresh schema (v1 ca75629011, v2 90a82c2dcd, v4 7e10c08f66,
v5 70e6a25935, v6 ab416d4612, v7 9f49e47dc4, v8 fd8e1ec4b9) migrated by the
current repository. v3 never had a constructor; it is only a migration step.
v1/v2 and v5/v6 migrate to the same catalog. Captured from the real
constructors; ``Tests/Personal_Context/test_repository.py`` rebuilds each.
"""

REPOSITORY_SCHEMA_VERSION = 9

_QUARANTINE_INDEX = 'CREATE INDEX quarantine_object_idx ON quarantine(object_type, object_id, version_id)'
_ENCRYPTED_OBJECTS = 'CREATE TABLE encrypted_objects (\n        object_type TEXT NOT NULL,\n        object_id TEXT NOT NULL,\n        version_id TEXT NOT NULL,\n        scope_id TEXT,\n        is_tombstone INTEGER NOT NULL DEFAULT 0,\n        algorithm TEXT NOT NULL,\n        nonce BLOB NOT NULL,\n        ciphertext BLOB NOT NULL,\n        wrapped_dek BLOB NOT NULL,\n        key_version INTEGER NOT NULL,\n        integrity_tag TEXT NOT NULL,\n        created_at TEXT NOT NULL,\n        PRIMARY KEY (object_type, object_id, version_id)\n    )'
_OUTBOX = 'CREATE TABLE encrypted_outbox (\n        outbox_id TEXT PRIMARY KEY,\n        sequence INTEGER NOT NULL UNIQUE,\n        object_type TEXT NOT NULL,\n        object_id TEXT NOT NULL,\n        version_id TEXT NOT NULL,\n        envelope_version TEXT NOT NULL,\n        status TEXT NOT NULL,\n        created_at TEXT NOT NULL,\n        destination_envelope_id TEXT,\n        quarantine_reason TEXT,\n        updated_at TEXT NOT NULL\n    )'
_FREEZE = 'CREATE TABLE first_link_freeze (\n        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n        plan_id TEXT NOT NULL,\n        snapshot_token TEXT NOT NULL,\n        source_profile_id TEXT NOT NULL,\n        source_purge_generation INTEGER NOT NULL,\n        source_key_version INTEGER NOT NULL,\n        created_at TEXT NOT NULL\n    )'
_REBASELINE = 'CREATE TABLE first_link_rebaseline_commit (\n        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n        plan_id TEXT NOT NULL,\n        target_profile_id TEXT NOT NULL,\n        target_integrity_key_id TEXT NOT NULL,\n        target_key_record_id TEXT NOT NULL,\n        target_purge_generation INTEGER NOT NULL,\n        rebaseline_version INTEGER NOT NULL,\n        created_at TEXT NOT NULL\n    )'
_LOCAL_RECORD_LINKS = 'CREATE TABLE local_record_links (\n        record_id TEXT PRIMARY KEY,\n        encrypted_link_version TEXT NOT NULL\n    )'
_LOCAL_RUNTIME_POLICY = 'CREATE TABLE local_runtime_policy (\n        scope_id TEXT PRIMARY KEY,\n        encrypted_policy_version TEXT NOT NULL\n    )'
_LOCAL_SCOPE_BINDINGS = 'CREATE TABLE local_scope_bindings (\n        scope_id TEXT PRIMARY KEY,\n        encrypted_binding_version TEXT NOT NULL\n    )'
_LOCAL_UNDO = 'CREATE TABLE local_undo (\n        undo_id TEXT PRIMARY KEY,\n        record_id TEXT NOT NULL,\n        encrypted_undo_version TEXT NOT NULL,\n        expires_at TEXT NOT NULL,\n        created_at TEXT NOT NULL\n    )'
_LOCAL_UNLINKED_SCOPES = 'CREATE TABLE local_unlinked_scopes (\n        scope_id TEXT PRIMARY KEY,\n        reviewed_plan_id TEXT NOT NULL\n    )'
_OBJECT_HEADS = 'CREATE TABLE object_heads (\n        object_type TEXT NOT NULL,\n        object_id TEXT NOT NULL,\n        version_id TEXT NOT NULL,\n        PRIMARY KEY (object_type, object_id)\n    )'
_SCHEMA_MARKER = 'CREATE TABLE personal_context_schema (\n        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n        version INTEGER NOT NULL\n    )'
_PROFILE_META = "CREATE TABLE profile_meta (\n        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n        profile_id TEXT NOT NULL,\n        purge_generation INTEGER NOT NULL,\n        destroyed INTEGER NOT NULL DEFAULT 0,\n        current_manifest_version TEXT NOT NULL,\n        created_at TEXT NOT NULL,\n        key_account TEXT NOT NULL DEFAULT ''\n    )"
_QUARANTINE = 'CREATE TABLE quarantine (\n        quarantine_id TEXT PRIMARY KEY,\n        object_type TEXT NOT NULL,\n        object_id TEXT NOT NULL,\n        version_id TEXT,\n        reason_code TEXT NOT NULL,\n        created_at TEXT NOT NULL\n    )'
_OUTBOX_SEQUENCE_INDEX_FROM_V1_V2 = 'CREATE UNIQUE INDEX encrypted_outbox_sequence_idx ON encrypted_outbox(sequence)'
_OUTBOX_FROM_V1_V2 = 'CREATE TABLE encrypted_outbox (\n        outbox_id TEXT PRIMARY KEY,\n        object_type TEXT NOT NULL,\n        object_id TEXT NOT NULL,\n        version_id TEXT NOT NULL,\n        envelope_version TEXT NOT NULL,\n        status TEXT NOT NULL,\n        created_at TEXT NOT NULL\n    , destination_envelope_id TEXT, quarantine_reason TEXT, updated_at TEXT, sequence INTEGER)'
_FREEZE_FROM_V4 = 'CREATE TABLE first_link_freeze (\n                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n                    plan_id TEXT NOT NULL,\n                    snapshot_token TEXT NOT NULL,\n                    created_at TEXT NOT NULL\n                , source_profile_id TEXT, source_purge_generation INTEGER, source_key_version INTEGER)'
_PROFILE_META_MIGRATED = "CREATE TABLE profile_meta (\n        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n        profile_id TEXT NOT NULL,\n        purge_generation INTEGER NOT NULL,\n        destroyed INTEGER NOT NULL DEFAULT 0,\n        current_manifest_version TEXT NOT NULL,\n        created_at TEXT NOT NULL\n    , key_account TEXT NOT NULL DEFAULT '')"
_FREEZE_FROM_V5_V6 = 'CREATE TABLE first_link_freeze (\n        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n        plan_id TEXT NOT NULL,\n        snapshot_token TEXT NOT NULL,\n        created_at TEXT NOT NULL\n    , source_profile_id TEXT, source_purge_generation INTEGER, source_key_version INTEGER)'
_REBASELINE_FROM_V7 = 'CREATE TABLE first_link_rebaseline_commit (\n        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n        plan_id TEXT NOT NULL,\n        target_profile_id TEXT NOT NULL,\n        target_integrity_key_id TEXT NOT NULL,\n        target_purge_generation INTEGER NOT NULL,\n        rebaseline_version INTEGER NOT NULL,\n        created_at TEXT NOT NULL\n    , target_key_record_id TEXT)'

REPOSITORY_SCHEMAS = (
    # fresh v9
    (
        REPOSITORY_SCHEMA_VERSION,
        (_QUARANTINE_INDEX, _ENCRYPTED_OBJECTS, _OUTBOX, _FREEZE, _REBASELINE, _LOCAL_RECORD_LINKS, _LOCAL_RUNTIME_POLICY, _LOCAL_SCOPE_BINDINGS, _LOCAL_UNDO, _LOCAL_UNLINKED_SCOPES, _OBJECT_HEADS, _SCHEMA_MARKER, _PROFILE_META, _QUARANTINE),
    ),
    # migrated from v1 or v2
    (
        REPOSITORY_SCHEMA_VERSION,
        (_OUTBOX_SEQUENCE_INDEX_FROM_V1_V2, _QUARANTINE_INDEX, _ENCRYPTED_OBJECTS, _OUTBOX_FROM_V1_V2, _FREEZE_FROM_V4, _REBASELINE, _LOCAL_RECORD_LINKS, _LOCAL_RUNTIME_POLICY, _LOCAL_SCOPE_BINDINGS, _LOCAL_UNDO, _LOCAL_UNLINKED_SCOPES, _OBJECT_HEADS, _SCHEMA_MARKER, _PROFILE_META_MIGRATED, _QUARANTINE),
    ),
    # migrated from v4
    (
        REPOSITORY_SCHEMA_VERSION,
        (_QUARANTINE_INDEX, _ENCRYPTED_OBJECTS, _OUTBOX, _FREEZE_FROM_V4, _REBASELINE, _LOCAL_RECORD_LINKS, _LOCAL_RUNTIME_POLICY, _LOCAL_SCOPE_BINDINGS, _LOCAL_UNDO, _LOCAL_UNLINKED_SCOPES, _OBJECT_HEADS, _SCHEMA_MARKER, _PROFILE_META_MIGRATED, _QUARANTINE),
    ),
    # migrated from v5 or v6
    (
        REPOSITORY_SCHEMA_VERSION,
        (_QUARANTINE_INDEX, _ENCRYPTED_OBJECTS, _OUTBOX, _FREEZE_FROM_V5_V6, _REBASELINE, _LOCAL_RECORD_LINKS, _LOCAL_RUNTIME_POLICY, _LOCAL_SCOPE_BINDINGS, _LOCAL_UNDO, _LOCAL_UNLINKED_SCOPES, _OBJECT_HEADS, _SCHEMA_MARKER, _PROFILE_META_MIGRATED, _QUARANTINE),
    ),
    # migrated from v7
    (
        REPOSITORY_SCHEMA_VERSION,
        (_QUARANTINE_INDEX, _ENCRYPTED_OBJECTS, _OUTBOX, _FREEZE, _REBASELINE_FROM_V7, _LOCAL_RECORD_LINKS, _LOCAL_RUNTIME_POLICY, _LOCAL_SCOPE_BINDINGS, _LOCAL_UNDO, _LOCAL_UNLINKED_SCOPES, _OBJECT_HEADS, _SCHEMA_MARKER, _PROFILE_META_MIGRATED, _QUARANTINE),
    ),
    # migrated from v8
    (
        REPOSITORY_SCHEMA_VERSION,
        (_QUARANTINE_INDEX, _ENCRYPTED_OBJECTS, _OUTBOX, _FREEZE, _REBASELINE, _LOCAL_RECORD_LINKS, _LOCAL_RUNTIME_POLICY, _LOCAL_SCOPE_BINDINGS, _LOCAL_UNDO, _LOCAL_UNLINKED_SCOPES, _OBJECT_HEADS, _SCHEMA_MARKER, _PROFILE_META_MIGRATED, _QUARANTINE),
    ),
)
