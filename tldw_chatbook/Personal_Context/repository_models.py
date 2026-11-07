"""Repository-only result and metadata types."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuarantineEntry:
    quarantine_id: str
    object_type: str
    object_id: str
    version_id: str | None
    reason_code: str
    created_at: str


_LOCAL_UNDO_SCHEMA = """
    CREATE TABLE local_undo (
        undo_id TEXT PRIMARY KEY,
        record_id TEXT NOT NULL,
        encrypted_undo_version TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
_LOCAL_RECORD_LINK_SCHEMA = """
    CREATE TABLE local_record_links (
        record_id TEXT PRIMARY KEY,
        encrypted_link_version TEXT NOT NULL
    )
    """
_LOCAL_UNLINKED_SCOPE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS local_unlinked_scopes (
        scope_id TEXT PRIMARY KEY,
        reviewed_plan_id TEXT NOT NULL
    )
    """
_FIRST_LINK_REBASELINE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS first_link_rebaseline_commit (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        plan_id TEXT NOT NULL,
        target_profile_id TEXT NOT NULL,
        target_integrity_key_id TEXT NOT NULL,
        target_key_record_id TEXT NOT NULL,
        target_purge_generation INTEGER NOT NULL,
        rebaseline_version INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE personal_context_schema (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE profile_meta (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        profile_id TEXT NOT NULL,
        purge_generation INTEGER NOT NULL,
        destroyed INTEGER NOT NULL DEFAULT 0,
        current_manifest_version TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE encrypted_objects (
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        scope_id TEXT,
        is_tombstone INTEGER NOT NULL DEFAULT 0,
        algorithm TEXT NOT NULL,
        nonce BLOB NOT NULL,
        ciphertext BLOB NOT NULL,
        wrapped_dek BLOB NOT NULL,
        key_version INTEGER NOT NULL,
        integrity_tag TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (object_type, object_id, version_id)
    )
    """,
    """
    CREATE TABLE object_heads (
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        PRIMARY KEY (object_type, object_id)
    )
    """,
    """
    CREATE TABLE local_runtime_policy (
        scope_id TEXT PRIMARY KEY,
        encrypted_policy_version TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE local_scope_bindings (
        scope_id TEXT PRIMARY KEY,
        encrypted_binding_version TEXT NOT NULL
    )
    """,
    _LOCAL_UNLINKED_SCOPE_SCHEMA,
    _LOCAL_UNDO_SCHEMA,
    _LOCAL_RECORD_LINK_SCHEMA,
    """
    CREATE TABLE encrypted_outbox (
        outbox_id TEXT PRIMARY KEY,
        sequence INTEGER NOT NULL UNIQUE,
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        envelope_version TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        destination_envelope_id TEXT,
        quarantine_reason TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE quarantine (
        quarantine_id TEXT PRIMARY KEY,
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT,
        reason_code TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE first_link_freeze (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        plan_id TEXT NOT NULL,
        snapshot_token TEXT NOT NULL,
        source_profile_id TEXT NOT NULL,
        source_purge_generation INTEGER NOT NULL,
        source_key_version INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    _FIRST_LINK_REBASELINE_SCHEMA,
    "CREATE INDEX quarantine_object_idx ON quarantine(object_type, object_id, version_id)",
)
