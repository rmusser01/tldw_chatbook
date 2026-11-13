"""Backup owner for the Personal Context repository (TASK-40133, TASK-40134).

Importing opens no store. The repository is ordinary SQLite with ciphertext
columns, so it is captured as it is, without its key, through the SQLite backup
route, and validated against the exact v9 catalogs below. The profile key
travels only as an encrypted credential record (``Backup_Recovery.credentials``).
A replace restore publishes it and always rewrites its key reference, to a
fresh account holding a key that authenticated it or to a marker naming why it
stays locked (TASK-40135); isolated restores still hold it back (TASK-40136).
While a first link is frozen or committing, capture is refused.

Every SQL catalog a Personal Context repository at schema v9 can have.

The repository migrates in place with ``ALTER TABLE`` (``repository.py``
``_inspect_schema``), so a fresh v9 database and a migrated one differ
textually, and backup validation compares the exact catalog. These are the
reachable layouts, in ``sqlite_schema`` order (type, name): the fresh one and
one per historical fresh schema (v1 ca75629011, v2 90a82c2dcd, v4 7e10c08f66,
v5 70e6a25935, v6 ab416d4612, v7 9f49e47dc4, v8 fd8e1ec4b9) migrated by the
current repository. v3 never had a constructor; it is only a migration step.
v1/v2 and v5/v6 migrate to the same catalog. The v7 build's own 6 -> 7 step
created the rebaseline table without ``target_key_record_id`` (added later by
the v8 step), so a v1-v6 database that build migrated has three more layouts.
Captured from the real constructors; ``Tests/Personal_Context/test_repository.py``
rebuilds each.
"""

import hashlib
import json
import os
import sqlite3
import zipfile
from contextlib import closing, suppress
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
    # v1 or v2, migrated by the v7 build, then by the current repository
    (
        REPOSITORY_SCHEMA_VERSION,
        (_OUTBOX_SEQUENCE_INDEX_FROM_V1_V2, _QUARANTINE_INDEX, _ENCRYPTED_OBJECTS, _OUTBOX_FROM_V1_V2, _FREEZE_FROM_V4, _REBASELINE_FROM_V7, _LOCAL_RECORD_LINKS, _LOCAL_RUNTIME_POLICY, _LOCAL_SCOPE_BINDINGS, _LOCAL_UNDO, _LOCAL_UNLINKED_SCOPES, _OBJECT_HEADS, _SCHEMA_MARKER, _PROFILE_META_MIGRATED, _QUARANTINE),
    ),
    # v4, migrated by the v7 build, then by the current repository
    (
        REPOSITORY_SCHEMA_VERSION,
        (_QUARANTINE_INDEX, _ENCRYPTED_OBJECTS, _OUTBOX, _FREEZE_FROM_V4, _REBASELINE_FROM_V7, _LOCAL_RECORD_LINKS, _LOCAL_RUNTIME_POLICY, _LOCAL_SCOPE_BINDINGS, _LOCAL_UNDO, _LOCAL_UNLINKED_SCOPES, _OBJECT_HEADS, _SCHEMA_MARKER, _PROFILE_META_MIGRATED, _QUARANTINE),
    ),
    # v5 or v6, migrated by the v7 build, then by the current repository
    (
        REPOSITORY_SCHEMA_VERSION,
        (_QUARANTINE_INDEX, _ENCRYPTED_OBJECTS, _OUTBOX, _FREEZE_FROM_V5_V6, _REBASELINE_FROM_V7, _LOCAL_RECORD_LINKS, _LOCAL_RUNTIME_POLICY, _LOCAL_SCOPE_BINDINGS, _LOCAL_UNDO, _LOCAL_UNLINKED_SCOPES, _OBJECT_HEADS, _SCHEMA_MARKER, _PROFILE_META_MIGRATED, _QUARANTINE),
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



# Personal Context sync needs both owners approved for every restored
# generation. This review approves only the repository, and only where the
# restore published neither (TASK-40134 AC#9-10); `runtime.sync_state` and
# `config` keep their own review (TASK-40137), so sync does not resume here.
SYNC_OWNERS = (OWNER, "runtime.sync_state")
REVIEW_AVAILABLE = "personal_context_restore_review_available"
RESTORED_LINK_STATE = "personal_context_restored_link_state"
REVIEW_UNAVAILABLE = "personal_context_restore_review_unavailable"


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
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.plan_records import load_plan

    try:
        control = Path(witness["store_root"]).parent
        operation = witness["operation_id"]
        # A read never creates the operation directory (as activation.py does).
        if not (control / ("operation-" + bootstrap._key(operation))).is_dir():
            return None
        restored = load_plan(Journal(control, operation)).restore
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
    """The restore review Settings shows: None, available, or a named reason.

    A failed read names the reason only while the (fail-closed) gate keeps sync
    off; an ordinary profile then shows nothing.
    """
    try:
        states = _restore_reviews(Path(db_path), approve=False)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return None if sync_allowed(db_path) else REVIEW_UNAVAILABLE
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


# TASK-40135: a restore never adopts, overwrites or deletes an entry it did not
# create (ADR-126 decision 8). Key material is used only after it authenticates
# the restored copy's manifest head; it then goes into a fresh account. Anything
# else leaves the marker, whose reason survives a restart. Staging only names
# the account; the entry is written just before publication (and again,
# idempotently, by ``recover --finish``), so an abandoned review leaves none.
MARKER = "recovery:setup_required:"
KEY_NOT_IN_BACKUP = "personal_context_key_not_in_backup"
KEY_INSTALL_UNAVAILABLE = "personal_context_key_install_unavailable"
RESTORE_READABLE = "personal_context_restore_readable"
RESTORE_LOCKED = "personal_context_restore_locked"
KEY_UNVERIFIED = "personal_context_key_unverified"
_RECOVERY_ACCOUNT = "personal-context:recovery_"


def _device_protector():
    """This device's secure profile keyring, or None; never a null backend."""
    from tldw_chatbook.Backup_Recovery.credentials import _credential_store

    from .key_protector import KeyringProfileKeyProtector

    try:
        return KeyringProfileKeyProtector(_credential_store()._keyring)
    except Exception:  # noqa: BLE001 - unavailable or insecure keyring
        return None


def _stored_value(protector, account: str):
    from .key_protector import _KEYRING_SERVICE

    try:
        return protector._keyring.get_password(_KEYRING_SERVICE, account)
    except Exception:  # noqa: BLE001 - backend errors can contain secret values
        return None


def restore_key(connection, *, record_value, target: Path, protector):
    """Pick the key a restore may install: ``(value, None)`` or ``(None, reason)``.

    Candidates are the archive's record, then the account the archived copy
    names on this device (the target path's legacy account for an empty
    reference). A value counts only once it authenticates this copy. A copy
    with no live profile holds no key: ``(None, None)``.
    """
    from .repository_keys import KEY_ACCOUNT, profile_key_opens

    meta = connection.execute(
        "SELECT destroyed, key_account FROM profile_meta WHERE singleton = 1"
    ).fetchone()
    if meta is None or meta[0]:
        return None, None
    stored = meta[1]
    candidates = [] if record_value is None else [(stored, record_value)]
    if protector is not None:
        account = (
            stored
            if isinstance(stored, str) and KEY_ACCOUNT.fullmatch(stored)
            else "personal-context:"
            + hashlib.sha256(os.fsencode(str(Path(target).absolute()))).hexdigest()
            if stored == ""
            else None
        )
        if account is not None:
            candidates.append((account, _stored_value(protector, account)))
    for account, value in candidates:
        if value is not None and profile_key_opens(connection, account, value):
            if protector is None:
                return None, KEY_INSTALL_UNAVAILABLE
            return value, None
    if record_value is not None and protector is None:
        return None, KEY_INSTALL_UNAVAILABLE
    return None, KEY_NOT_IN_BACKUP


def _issue(value, reason, logical_id: str) -> str | None:
    """The plan line for one repository; its logical id keeps profiles apart."""
    if reason is not None:
        return RESTORE_LOCKED + ":" + reason + ":" + logical_id
    return None if value is None else RESTORE_READABLE + ":" + logical_id


def restore_account(candidate: Path, logical_id: str) -> str:
    """The fresh account one staged restore names; unique per staging."""
    seed = os.fsencode(str(candidate)) + b"\x00" + logical_id.encode()
    return _RECOVERY_ACCOUNT + hashlib.sha256(seed).hexdigest()[:32]


def _archived_key(archive, doc, logical_id: str, target: Path):
    """``restore_key`` against the sealed archive's own copy and record."""
    row = next(row for row in doc.files if row.logical_id == logical_id)
    with zipfile.ZipFile(archive.path) as container:
        data = container.read(row.payload)
        record = None
        material = next(
            (r for r in doc.files if r.owner_id == "recovery.credentials"), None
        )
        if doc.credential_policy != "exclude" and material is not None:
            records = json.loads(container.read(material.payload))["records"]
            record = next(
                (
                    r.get("value")
                    for r in records
                    if r.get("kind") == "personal_context"
                    and r.get("file") == row.payload
                    and r.get("status") == "captured"
                ),
                None,
            )
    if hashlib.sha256(data).hexdigest() != row.sha256:
        raise ValueError("sealed_changed")
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    with closing(
        connect_private_sqlite("recovery.personal_context.repository", ":memory:")
    ) as connection:
        # An in-memory image cannot be in WAL mode; the bytes on disk are untouched.
        connection.deserialize(data[:18] + b"\x01\x01" + data[20:])
        return restore_key(
            connection, record_value=record, target=target, protector=_device_protector()
        )


def restore_plan_issue(archive, doc, logical_id: str, target: Path) -> str | None:
    """What the plan says before a replace restore runs: readable or locked, and why."""
    try:
        return _issue(*_archived_key(archive, doc, logical_id, target), logical_id)
    except (OSError, KeyError, sqlite3.Error, zipfile.BadZipFile, json.JSONDecodeError):
        raise ValueError("personal_context_validation_unavailable") from None


def stamp_restored_key(
    candidate: Path,
    *,
    record_value,
    target: Path,
    logical_id: str,
    planned: str | None,
    account: str,
) -> None:
    """Rewrite a staged copy's key reference; nothing is written to the keyring.

    Readable: the copy names ``account``, which must be unused. Locked: the
    copy names the marker. A removed profile names nothing. Only
    ``key_account`` changes. A result other than the reviewed plan's refuses
    before any write.
    """
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    protector = _device_protector()
    with closing(
        connect_private_sqlite("recovery.personal_context.repository", candidate)
    ) as connection:
        connection.execute("PRAGMA trusted_schema=OFF")
        value, reason = restore_key(
            connection, record_value=record_value, target=target, protector=protector
        )
        if _issue(value, reason, logical_id) != planned:
            raise ValueError("personal_context_restore_changed")
        if value is None and reason is None:
            reference = ""
        elif reason is not None:
            reference = MARKER + reason
        elif _stored_value(protector, account) is not None:
            raise ValueError("personal_context_key_account_in_use")
        else:
            reference = account
        with connection:
            connection.execute(
                "UPDATE profile_meta SET key_account = ? WHERE singleton = 1",
                (reference,),
            )


def install_restored_keys(archive, doc, plan, candidate: Path) -> None:
    """Write each readable restore's key into its fresh account before publication.

    Set-if-absent and read back; a rerun (``recover --finish``) accepts the same
    value already there. The key is re-derived from the archive and must still
    authenticate; anything else refuses before publication.
    """
    from .key_protector import decode_keyring_material

    for key, target in plan.restore:
        if RESTORE_READABLE + ":" + key not in plan.issues:
            continue
        value, reason = _archived_key(archive, doc, key, target)
        if reason is not None or value is None:
            raise ValueError("personal_context_restore_changed")
        protector = _device_protector()
        account = restore_account(candidate, key)
        material = decode_keyring_material(value)
        existing = _stored_value(protector, account)
        if existing is None:
            protector.install(account, material)
        elif existing != value:
            raise ValueError("personal_context_key_account_in_use")
        if protector.load(account) != material:
            raise ValueError("personal_context_key_install_failed")


def verify_restored_key(path: Path) -> str | None:
    """After publication: the restored copy opens with the key its account holds.

    On failure only the ``recovery_`` account the restore created is deleted;
    the database stays in place and Personal Context is locked.
    """
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    from .repository_keys import profile_key_opens

    account = protector = None
    try:
        with closing(
            connect_private_sqlite(
                "recovery.personal_context.repository", path, read_only=True
            )
        ) as connection:
            row = connection.execute(
                "SELECT destroyed, key_account FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if row is None or row[0] or not str(row[1]).startswith(_RECOVERY_ACCOUNT):
                return None
            account = row[1]
            protector = _device_protector()
            value = None if protector is None else _stored_value(protector, account)
            if value is not None and profile_key_opens(connection, account, value):
                return None
    except (OSError, ValueError, sqlite3.Error):
        pass
    if account is not None and protector is not None:
        with suppress(Exception):
            protector.delete(account)
    return KEY_UNVERIFIED


def verify_restored_keys(plan) -> tuple[str, ...]:
    """Check every Personal Context repository a finished restore published."""
    return tuple(
        sorted(
            {
                issue
                for key, path in plan.restore
                if key.count(":") >= 2 and key.split(":")[2] == OWNER
                if (issue := verify_restored_key(path)) is not None
            }
        )
    )
