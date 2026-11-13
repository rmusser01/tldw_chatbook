"""Backup owner for the Personal Context repository (TASK-40133, TASK-40134).

Importing opens no store. The repository is ordinary SQLite with ciphertext
columns, so it is captured as it is, without its key, through the SQLite backup
route, and validated against the exact v9 catalogs below. The profile key
travels only as an encrypted credential record (``Backup_Recovery.credentials``).
No restore publishes the repository yet (``restore_plan``): readable restores
are TASK-40135. While a first link is frozen or committing, capture is refused.

Every SQL catalog a Personal Context repository at schema v9 can have.

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

from contextlib import closing
from dataclasses import replace
from pathlib import Path
from threading import Event

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.DB.recovery_operations import _SQLiteDeclaration

OWNER = "personal_context.repository"
LEAF = "tldw_chatbook_personal_context.db"
FIRST_LINK_IN_PROGRESS = "personal_context_first_link_in_progress"
NOT_RESTORED = "personal_context_not_restored"
VERSION_QUERY = "SELECT version FROM personal_context_schema WHERE singleton = 1"
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


class _Repository(_SQLiteDeclaration):
    def _state(self, connection) -> tuple[str, ...]:
        from tldw_chatbook.DB.recovery_sqlite import _validate_sqlite

        issues = _validate_sqlite(
            connection, self.versions, self.schemas, version_query=VERSION_QUERY
        )
        if issues:
            return issues
        # Mid-link state is never captured (decision 7).
        if connection.execute(
            "SELECT 1 FROM first_link_freeze UNION ALL "
            "SELECT 1 FROM first_link_rebaseline_commit LIMIT 1"
        ).fetchone():
            return (FIRST_LINK_IN_PROGRESS,)
        return ()

    def validate(self, candidate: Path) -> tuple[str, ...]:
        import sqlite3

        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.personal_context.repository", candidate, read_only=True
                )
            ) as connection:
                return self._state(connection)
        except (OSError, ValueError, sqlite3.Error):
            return ("personal_context_validation_unavailable",)

    def discover(self, config) -> tuple[StorageItem, ...]:
        items = super().discover(config)
        item = items[0]
        if item.status != "included":
            return items
        issues = self.validate(item.path)
        if not issues:
            return items
        if issues != (FIRST_LINK_IN_PROGRESS,):
            return (replace(item, status="unsupported"),)
        return (
            replace(item, status="unavailable"),
            StorageItem(
                OWNER,
                storage_logical_id(discovery_context(config), OWNER, "first_link"),
                None,
                "unavailable",
                item.dependencies,
            ),
        )

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite
        from tldw_chatbook.DB.recovery_sqlite import _checked_capture

        with _checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            # Mid-link state is refused before a byte is copied, not after.
            if FIRST_LINK_IN_PROGRESS in self.validate(item.path):
                raise ValueError(FIRST_LINK_IN_PROGRESS)
            copy_private_sqlite(
                "recovery.personal_context.repository",
                item.path,
                destination,
                progress_guard=guard,
            )

    def relocate(self, candidate: Path, mapping) -> None:
        # The restore planner never publishes the repository until TASK-40135.
        raise ValueError(NOT_RESTORED)


# Personal Context sync needs both owners approved for every restored
# generation. This review approves only the repository, and only where the
# restore published neither (TASK-40134 AC#9-10); `runtime.sync_state` and
# `config` keep their own review (TASK-40137), so sync does not resume here.
SYNC_OWNERS = (OWNER, "runtime.sync_state")
REVIEW_AVAILABLE = "personal_context_restore_review_available"
RESTORED_LINK_STATE = "personal_context_restored_link_state"


SYNC_REVIEW_REQUIRED = "personal_context_sync_restore_review_required"


def sync_allowed(db_path: Path | None = None) -> bool:
    """Whether Personal Context sync may load; false for any unreviewed restore.

    Like Sync V2's ``_sync_call``, the check without a path covers the selected
    profile's own generations; a repository path is checked as well.
    """
    from tldw_chatbook.Backup_Recovery.activation import execution_allowed

    paths = dict.fromkeys((None, None if db_path is None else Path(db_path)))
    return all(execution_allowed(SYNC_OWNERS, path) for path in paths)


def require_sync_allowed(db_path: Path | None = None) -> None:
    """Refuse to load Personal Context sync until every restore is reviewed."""
    if not sync_allowed(db_path):
        raise ValueError(SYNC_REVIEW_REQUIRED)


def _published_owners(witness) -> frozenset[str] | None:
    """Owners whose payload the generation's restore plan published, if known."""
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.plan_records import load_plan

    try:
        journal = Journal(Path(witness["store_root"]).parent, witness["operation_id"])
        restored = load_plan(journal).restore
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None
    return frozenset(key.split(":")[2] for key, _ in restored if key.count(":") >= 2)


def _restore_reviews(db_path: Path, *, approve: bool) -> tuple[str, ...]:
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore
    from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    states = []
    with acquire_storage(db_path) as lease:
        for witness in _witnesses(db_path, lease):
            store = ActivationStore(Path(witness["store_root"]))
            if OWNER not in witness["owners"] or store.allowed(
                witness["generation"], OWNER
            ):
                continue
            published = _published_owners(witness)
            if published is None or published & set(SYNC_OWNERS):
                states.append(RESTORED_LINK_STATE)  # Unknown counts as published.
            elif approve:
                store.approve(witness["generation"], OWNER)
            else:
                states.append(REVIEW_AVAILABLE)
    return tuple(states)


def restore_review(db_path: Path) -> str | None:
    """The restore review Settings shows: None, available, or the named refusal.

    A failed read shows nothing; the sync gate above fails closed on its own.
    """
    try:
        states = _restore_reviews(Path(db_path), approve=False)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return None
    if RESTORED_LINK_STATE in states:
        return RESTORED_LINK_STATE
    return REVIEW_AVAILABLE if states else None


def approve_restore_review(db_path: Path) -> None:
    """Approve only ``personal_context.repository`` for reviewable generations."""
    if RESTORED_LINK_STATE in _restore_reviews(Path(db_path), approve=True):
        raise ValueError(RESTORED_LINK_STATE)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _Repository(
            OWNER, None, LEAF, (REPOSITORY_SCHEMA_VERSION,), REPOSITORY_SCHEMAS,
            optional_default=True,
        ),
    )
