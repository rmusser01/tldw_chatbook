"""Device-private durable receipts for approved one-time Notes imports.

The ledger deliberately stores only opaque identifiers, private digests, and
bounded lifecycle metadata. Source paths and note payload data never cross this
persistence boundary.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Notes.note_import_execution_models import (
    ApprovedNoteImportPlan,
    ImportEffectState,
    ImportExecutionReceipt,
    ImportItemOutcome,
    ImportSessionState,
    _canonical_json_digest,
    _private_payload_fingerprint,
    _private_source_locator_digest,
    _validate_reason_code,
)
from tldw_chatbook.Notes.note_import_plan_models import MAX_IMPORT_ENTRIES, ImportAction

_SCHEMA_VERSION = 1
_MIN_BATCH_SIZE = 1
_MAX_BATCH_SIZE = 100
_MAX_TRANSITIONS = MAX_IMPORT_ENTRIES
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")

_PAYLOAD_TABLE = "import_payload_effects"
_FOLDER_TABLE = "import_folder_effects"
_MEMBERSHIP_TABLE = "import_membership_effects"
_EFFECT_TABLES = frozenset({_PAYLOAD_TABLE, _FOLDER_TABLE, _MEMBERSHIP_TABLE})


SESSION_STATE_TRANSITIONS: Mapping[
    ImportSessionState, frozenset[ImportSessionState]
] = MappingProxyType(
    {
        ImportSessionState.PENDING: frozenset(
            {ImportSessionState.RUNNING, ImportSessionState.CANCELLED}
        ),
        ImportSessionState.RUNNING: frozenset(
            {
                ImportSessionState.CANCELLED,
                ImportSessionState.COMPLETED,
                ImportSessionState.NEEDS_ATTENTION,
            }
        ),
        ImportSessionState.NEEDS_ATTENTION: frozenset(
            {ImportSessionState.RUNNING, ImportSessionState.CANCELLED}
        ),
        ImportSessionState.COMPLETED: frozenset(),
        ImportSessionState.CANCELLED: frozenset(),
    }
)

ITEM_OUTCOME_TRANSITIONS: Mapping[ImportItemOutcome, frozenset[ImportItemOutcome]] = (
    MappingProxyType(
        {
            ImportItemOutcome.PENDING: frozenset(
                {
                    ImportItemOutcome.IMPORTED,
                    ImportItemOutcome.UPDATED,
                    ImportItemOutcome.SKIPPED,
                    ImportItemOutcome.FAILED,
                }
            ),
            ImportItemOutcome.IMPORTED: frozenset(),
            ImportItemOutcome.UPDATED: frozenset(),
            ImportItemOutcome.SKIPPED: frozenset(),
            ImportItemOutcome.FAILED: frozenset(),
        }
    )
)

EFFECT_STATE_TRANSITIONS: Mapping[ImportEffectState, frozenset[ImportEffectState]] = (
    MappingProxyType(
        {
            ImportEffectState.PENDING: frozenset(
                {ImportEffectState.APPLIED, ImportEffectState.FAILED}
            ),
            ImportEffectState.APPLIED: frozenset(),
            ImportEffectState.FAILED: frozenset(),
        }
    )
)


class ImportReceiptError(RuntimeError):
    """Base class for bounded receipt-ledger failures."""


class ImportReceiptConflictError(ImportReceiptError):
    """An approval identifier is already bound to different authority."""


class ImportReceiptTransitionError(ImportReceiptError):
    """A requested lifecycle transition is not explicitly allowed."""


@dataclass(frozen=True, slots=True, repr=False)
class ImportItemRecord:
    """Frozen private item projection returned by the repository."""

    item_id: str
    source_locator_digest: str
    selected_action: ImportAction
    outcome_count: int
    outcome: ImportItemOutcome
    target_note_id: str | None
    expected_version: int | None
    observed_version: int | None
    reason_code: str | None
    retryable: bool

    def __repr__(self) -> str:
        return (
            f"ImportItemRecord(outcome={self.outcome!r}, retryable={self.retryable!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ImportEffectRecord:
    """Frozen private independently replayable effect projection."""

    table: str
    effect_id: str
    item_id: str | None
    payload_index: int | None
    effect_kind: str
    state: ImportEffectState
    target_note_id: str | None
    target_folder_id: str | None
    expected_version: int | None
    observed_version: int | None
    reason_code: str | None
    retryable: bool

    def __repr__(self) -> str:
        return (
            f"ImportEffectRecord(effect_kind={self.effect_kind!r}, "
            f"state={self.state!r}, retryable={self.retryable!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ImportSessionSnapshot:
    """Frozen durable session and effect snapshot."""

    approval_id: str
    session_id: str
    plan_digest: str
    state: ImportSessionState
    batch_size: int
    total: int
    reason_code: str | None
    items: tuple[ImportItemRecord, ...] = ()
    payload_effects: tuple[ImportEffectRecord, ...] = ()
    folder_effects: tuple[ImportEffectRecord, ...] = ()
    membership_effects: tuple[ImportEffectRecord, ...] = ()

    def __repr__(self) -> str:
        return (
            f"ImportSessionSnapshot(state={self.state!r}, "
            f"batch_size={self.batch_size!r}, items={len(self.items)!r})"
        )


@dataclass(frozen=True, slots=True)
class ReceiptSchemaSnapshot:
    """Test-only immutable schema census."""

    user_version: int
    tables: tuple[str, ...]
    columns: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ItemTransition:
    """One requested item transition in an atomic repository update."""

    item_id: str = field(repr=False)
    outcome: ImportItemOutcome
    reason_code: str | None = None
    retryable: bool = False
    target_note_id: str | None = field(default=None, repr=False)
    observed_version: int | None = None


@dataclass(frozen=True, slots=True)
class EffectTransition:
    """One requested effect transition in an atomic repository update."""

    table: str
    effect_id: str = field(repr=False)
    state: ImportEffectState
    reason_code: str | None = None
    retryable: bool = False
    target_note_id: str | None = field(default=None, repr=False)
    target_folder_id: str | None = field(default=None, repr=False)
    observed_version: int | None = None


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS import_sessions (
        session_id TEXT PRIMARY KEY,
        approval_id TEXT NOT NULL UNIQUE,
        plan_digest TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending'
            CHECK (state IN ('pending', 'running', 'cancelled', 'completed', 'needs_attention')),
        batch_size INTEGER NOT NULL CHECK (batch_size BETWEEN 1 AND 100),
        total_count INTEGER NOT NULL CHECK (total_count >= 0),
        reason_code TEXT CHECK (
            reason_code IS NULL OR (
                length(reason_code) BETWEEN 1 AND 64
                AND reason_code NOT GLOB '*[^a-z0-9_]*'
                AND substr(reason_code, 1, 1) GLOB '[a-z]'
            )
        ),
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        CHECK (length(session_id) BETWEEN 1 AND 256),
        CHECK (length(approval_id) = 36),
        CHECK (length(plan_digest) = 64 AND plan_digest NOT GLOB '*[^0-9a-f]*')
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_items (
        session_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        source_locator_digest TEXT NOT NULL,
        selected_action TEXT NOT NULL
            CHECK (selected_action IN ('skip', 'create_new', 'update_existing')),
        outcome_count INTEGER NOT NULL CHECK (outcome_count > 0),
        outcome TEXT NOT NULL DEFAULT 'pending'
            CHECK (outcome IN ('pending', 'imported', 'updated', 'skipped', 'failed')),
        target_note_id TEXT,
        expected_version INTEGER CHECK (expected_version IS NULL OR expected_version >= 0),
        observed_version INTEGER CHECK (observed_version IS NULL OR observed_version >= 0),
        reason_code TEXT CHECK (
            reason_code IS NULL OR (
                length(reason_code) BETWEEN 1 AND 64
                AND reason_code NOT GLOB '*[^a-z0-9_]*'
                AND substr(reason_code, 1, 1) GLOB '[a-z]'
            )
        ),
        retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        PRIMARY KEY (session_id, item_id),
        FOREIGN KEY (session_id) REFERENCES import_sessions(session_id) ON DELETE CASCADE,
        CHECK (length(item_id) BETWEEN 1 AND 256),
        CHECK (
            length(source_locator_digest) = 64
            AND source_locator_digest NOT GLOB '*[^0-9a-f]*'
        ),
        CHECK (target_note_id IS NULL OR length(target_note_id) BETWEEN 1 AND 256),
        CHECK (outcome = 'failed' OR retryable = 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_payload_effects (
        effect_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        payload_index INTEGER NOT NULL CHECK (payload_index >= 0),
        payload_digest TEXT NOT NULL,
        effect_kind TEXT NOT NULL CHECK (effect_kind IN ('create_note', 'replace_content')),
        state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'applied', 'failed')),
        target_note_id TEXT,
        expected_version INTEGER CHECK (expected_version IS NULL OR expected_version >= 0),
        observed_version INTEGER CHECK (observed_version IS NULL OR observed_version >= 0),
        reason_code TEXT CHECK (
            reason_code IS NULL OR (
                length(reason_code) BETWEEN 1 AND 64
                AND reason_code NOT GLOB '*[^a-z0-9_]*'
                AND substr(reason_code, 1, 1) GLOB '[a-z]'
            )
        ),
        retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        FOREIGN KEY (session_id, item_id)
            REFERENCES import_items(session_id, item_id) ON DELETE CASCADE,
        UNIQUE (session_id, item_id, payload_index, effect_kind),
        CHECK (length(effect_id) BETWEEN 1 AND 256),
        CHECK (length(payload_digest) = 64 AND payload_digest NOT GLOB '*[^0-9a-f]*'),
        CHECK (target_note_id IS NULL OR length(target_note_id) BETWEEN 1 AND 256),
        CHECK (state = 'failed' OR retryable = 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_folder_effects (
        effect_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        folder_ordinal INTEGER NOT NULL CHECK (folder_ordinal >= 0),
        path_digest TEXT NOT NULL,
        effect_kind TEXT NOT NULL DEFAULT 'ensure_folder' CHECK (effect_kind = 'ensure_folder'),
        state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'applied', 'failed')),
        target_folder_id TEXT,
        reason_code TEXT CHECK (
            reason_code IS NULL OR (
                length(reason_code) BETWEEN 1 AND 64
                AND reason_code NOT GLOB '*[^a-z0-9_]*'
                AND substr(reason_code, 1, 1) GLOB '[a-z]'
            )
        ),
        retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        FOREIGN KEY (session_id) REFERENCES import_sessions(session_id) ON DELETE CASCADE,
        UNIQUE (session_id, path_digest),
        UNIQUE (session_id, folder_ordinal),
        CHECK (length(effect_id) BETWEEN 1 AND 256),
        CHECK (length(path_digest) = 64 AND path_digest NOT GLOB '*[^0-9a-f]*'),
        CHECK (target_folder_id IS NULL OR length(target_folder_id) BETWEEN 1 AND 256),
        CHECK (state = 'failed' OR retryable = 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_membership_effects (
        effect_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        payload_index INTEGER NOT NULL CHECK (payload_index >= 0),
        membership_ordinal INTEGER NOT NULL CHECK (membership_ordinal >= 0),
        folder_path_digest TEXT NOT NULL,
        effect_kind TEXT NOT NULL DEFAULT 'attach_membership'
            CHECK (effect_kind = 'attach_membership'),
        state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'applied', 'failed')),
        target_note_id TEXT,
        target_folder_id TEXT,
        reason_code TEXT CHECK (
            reason_code IS NULL OR (
                length(reason_code) BETWEEN 1 AND 64
                AND reason_code NOT GLOB '*[^a-z0-9_]*'
                AND substr(reason_code, 1, 1) GLOB '[a-z]'
            )
        ),
        retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        FOREIGN KEY (session_id, item_id)
            REFERENCES import_items(session_id, item_id) ON DELETE CASCADE,
        UNIQUE (session_id, item_id, payload_index, membership_ordinal),
        CHECK (length(effect_id) BETWEEN 1 AND 256),
        CHECK (
            length(folder_path_digest) = 64
            AND folder_path_digest NOT GLOB '*[^0-9a-f]*'
        ),
        CHECK (target_note_id IS NULL OR length(target_note_id) BETWEEN 1 AND 256),
        CHECK (target_folder_id IS NULL OR length(target_folder_id) BETWEEN 1 AND 256),
        CHECK (state = 'failed' OR retryable = 0)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_import_items_outcome ON import_items(session_id, outcome)",
    "CREATE INDEX IF NOT EXISTS idx_import_payload_state ON import_payload_effects(session_id, state)",
    "CREATE INDEX IF NOT EXISTS idx_import_folder_state ON import_folder_effects(session_id, state)",
    "CREATE INDEX IF NOT EXISTS idx_import_membership_state ON import_membership_effects(session_id, state)",
)


def _now() -> int:
    return max(1, time.time_ns())


def _validate_batch_size(batch_size: object) -> int:
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an integer.")
    if not _MIN_BATCH_SIZE <= batch_size <= _MAX_BATCH_SIZE:
        raise ValueError("batch_size must be between 1 and 100.")
    return batch_size


def _validate_id(value: object, *, field_name: str) -> str:
    if type(value) is not str or _SAFE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a bounded opaque identifier.")
    return value


def _validate_optional_id(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _validate_id(value, field_name=field_name)


def _validate_optional_version(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("observed_version must be an integer when provided.")
    if value < 0:
        raise ValueError("observed_version must be non-negative.")
    return value


def _validate_transition_metadata(
    *,
    failed: bool,
    reason_code: object,
    retryable: object,
) -> tuple[str | None, bool]:
    validated_reason = _validate_reason_code(reason_code)
    if type(retryable) is not bool:
        raise TypeError("retryable must be a boolean.")
    if not failed and (validated_reason is not None or retryable):
        raise ValueError("Only failed rows may carry failure metadata.")
    return validated_reason, retryable


def _folder_path_digest(segments: tuple[str, ...]) -> str:
    return _canonical_json_digest(
        {
            "segments": list(segments),
            "type": "tldw_note_import_folder_path",
            "version": 1,
        }
    )


def _outcome_count(action: ImportAction, payload_count: int) -> int:
    if action is ImportAction.CREATE_NEW:
        return payload_count
    return 1


def _copy_bounded_transitions(
    values: object,
    *,
    field_name: str,
) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection.")
    failure: ValueError | None = None
    try:
        copied = tuple(islice(values, _MAX_TRANSITIONS + 1))  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - hostile iterators must fail before DB access
        copied = ()
        failure = ValueError(
            f"{field_name} could not be read within the transition ceiling."
        )
    if failure is not None:
        raise failure from None
    if len(copied) > _MAX_TRANSITIONS:
        raise ValueError(f"{field_name} exceeds the transition ceiling.")
    return copied


def _safe_private_digest(factory) -> str:
    failure: ImportReceiptError | None = None
    try:
        digest = factory()
    except Exception:  # noqa: BLE001 - private canonicalizers are untrusted here
        digest = None
        failure = ImportReceiptError(
            "Private import receipt material could not be derived safely."
        )
    if failure is not None:
        raise failure from None
    if type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None:
        raise ImportReceiptError(
            "Private import receipt material could not be derived safely."
        )
    return digest


def _assert_compatible_authority(
    stored: object,
    proposed: object,
) -> None:
    if proposed is not None and stored is not None and proposed != stored:
        raise ImportReceiptConflictError(
            "Receipt reconciliation authority cannot be replaced."
        )


class NoteImportReceiptRepository:
    """Own the profile-local schema-v1 import receipt ledger."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    def __repr__(self) -> str:
        return "NoteImportReceiptRepository(<private>)"

    def _connect(self) -> sqlite3.Connection:
        connection = connect_private_sqlite("notes.sync_state", self._database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _initialize_schema(connection: sqlite3.Connection) -> None:
        current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current_version not in {0, _SCHEMA_VERSION}:
            raise ImportReceiptError("Unsupported private receipt schema version.")
        if current_version == _SCHEMA_VERSION:
            return
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 1")

    def begin(
        self,
        approved: ApprovedNoteImportPlan,
        *,
        batch_size: int,
    ) -> ImportSessionSnapshot:
        """Create or durably reopen one exact approved import session."""

        validated_batch_size = _validate_batch_size(batch_size)
        if type(approved) is not ApprovedNoteImportPlan:
            raise TypeError("approved must be an ApprovedNoteImportPlan.")
        approval_id = approved.approval_id
        plan_digest = approved._private_plan_digest()
        if _DIGEST_PATTERN.fullmatch(plan_digest) is None:
            raise ValueError("The approved plan digest is invalid.")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._initialize_schema(connection)
            existing = connection.execute(
                "SELECT plan_digest, batch_size FROM import_sessions WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if existing is not None:
                if existing != (plan_digest, validated_batch_size):
                    raise ImportReceiptConflictError(
                        "The approval is already bound to different receipt authority."
                    )
            else:
                seed_failure: ImportReceiptError | None = None
                try:
                    self._seed_approved_plan(
                        connection,
                        approved,
                        batch_size=validated_batch_size,
                    )
                except (ImportReceiptError, TypeError, ValueError, sqlite3.Error):
                    raise
                except Exception:  # noqa: BLE001 - sanitize unexpected private failures
                    seed_failure = ImportReceiptError(
                        "The approved import receipt could not be seeded safely."
                    )
                if seed_failure is not None:
                    raise seed_failure from None
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_session(approval_id)

    @staticmethod
    def _seed_approved_plan(
        connection: sqlite3.Connection,
        approved: ApprovedNoteImportPlan,
        *,
        batch_size: int,
    ) -> None:
        session_id = str(uuid4())
        timestamp = _now()
        plan = approved.plan
        total_count = sum(
            _outcome_count(item.selected_action, len(item.payloads))
            for item in plan.items
        )
        connection.execute(
            """
            INSERT INTO import_sessions (
                session_id, approval_id, plan_digest, state, batch_size,
                total_count, reason_code, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                approved.approval_id,
                approved._private_plan_digest(),
                ImportSessionState.PENDING.value,
                batch_size,
                total_count,
                None,
                timestamp,
                timestamp,
            ),
        )
        for item in plan.items:
            outcome_count = _outcome_count(
                item.selected_action,
                len(item.payloads),
            )
            expected_version = (
                item.match.note_version if item.match is not None else None
            )
            target_note_id = _validate_optional_id(
                item.match.note_id if item.match is not None else None,
                field_name="target_note_id",
            )
            connection.execute(
                """
                INSERT INTO import_items (
                    session_id, item_id, source_locator_digest, selected_action,
                    outcome_count, outcome, target_note_id, expected_version,
                    observed_version, reason_code, retryable, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    item.item_id,
                    _safe_private_digest(
                        lambda item=item: _private_source_locator_digest(item)
                    ),
                    item.selected_action.value,
                    outcome_count,
                    ImportItemOutcome.PENDING.value,
                    target_note_id,
                    expected_version,
                    None,
                    None,
                    0,
                    timestamp,
                    timestamp,
                ),
            )
            effect_kind: str | None = None
            if item.selected_action is ImportAction.CREATE_NEW:
                effect_kind = "create_note"
            elif (
                item.selected_action is ImportAction.UPDATE_EXISTING
                and item.replace_content
            ):
                effect_kind = "replace_content"
            if effect_kind is not None:
                for payload_index, payload in enumerate(item.payloads):
                    connection.execute(
                        """
                        INSERT INTO import_payload_effects (
                            effect_id, session_id, item_id, payload_index,
                            payload_digest, effect_kind, state, target_note_id,
                            expected_version, observed_version, reason_code,
                            retryable, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid4()),
                            session_id,
                            item.item_id,
                            payload_index,
                            _safe_private_digest(
                                lambda payload=payload: _private_payload_fingerprint(
                                    (payload,)
                                )
                            ),
                            effect_kind,
                            ImportEffectState.PENDING.value,
                            target_note_id,
                            expected_version,
                            None,
                            None,
                            0,
                            timestamp,
                            timestamp,
                        ),
                    )
            if item.add_membership:
                for membership_ordinal, membership in enumerate(item.memberships):
                    connection.execute(
                        """
                        INSERT INTO import_membership_effects (
                            effect_id, session_id, item_id, payload_index,
                            membership_ordinal, folder_path_digest, effect_kind,
                            state, target_note_id, target_folder_id, reason_code,
                            retryable, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid4()),
                            session_id,
                            item.item_id,
                            membership.payload_index,
                            membership_ordinal,
                            _safe_private_digest(
                                lambda membership=membership: _folder_path_digest(
                                    membership.folder_segments
                                )
                            ),
                            "attach_membership",
                            ImportEffectState.PENDING.value,
                            target_note_id,
                            None,
                            None,
                            0,
                            timestamp,
                            timestamp,
                        ),
                    )
        required_folder_paths: set[tuple[str, ...]] = set()
        for item in plan.items:
            if item.selected_action is ImportAction.SKIP or not item.add_membership:
                continue
            for membership in item.memberships:
                required_folder_paths.update(
                    tuple(membership.folder_segments[:depth])
                    for depth in range(1, len(membership.folder_segments) + 1)
                )
        for folder_ordinal, folder_path in enumerate(plan.proposed_folder_paths):
            if folder_path not in required_folder_paths:
                continue
            connection.execute(
                """
                INSERT INTO import_folder_effects (
                    effect_id, session_id, folder_ordinal, path_digest,
                    effect_kind, state, target_folder_id, reason_code,
                    retryable, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    session_id,
                    folder_ordinal,
                    _safe_private_digest(
                        lambda folder_path=folder_path: _folder_path_digest(folder_path)
                    ),
                    "ensure_folder",
                    ImportEffectState.PENDING.value,
                    None,
                    None,
                    0,
                    timestamp,
                    timestamp,
                ),
            )

    def get_session(self, approval_id: str) -> ImportSessionSnapshot:
        """Return the durable frozen snapshot for one approval."""

        _validate_id(approval_id, field_name="approval_id")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._initialize_schema(connection)
            snapshot = self._load_snapshot(connection, approval_id)
            connection.commit()
            return snapshot
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def load_session_snapshot(self, approval_id: str) -> ImportSessionSnapshot:
        """Return one complete durable session snapshot."""

        return self.get_session(approval_id)

    @staticmethod
    def _load_snapshot(
        connection: sqlite3.Connection,
        approval_id: str,
    ) -> ImportSessionSnapshot:
        session_row = connection.execute(
            """
            SELECT session_id, approval_id, plan_digest, state, batch_size,
                   total_count, reason_code
            FROM import_sessions WHERE approval_id = ?
            """,
            (approval_id,),
        ).fetchone()
        if session_row is None:
            raise KeyError("Import receipt session was not found.")
        session_id = session_row[0]
        item_rows = connection.execute(
            """
            SELECT item_id, source_locator_digest, selected_action, outcome_count,
                   outcome, target_note_id, expected_version, observed_version,
                   reason_code, retryable
            FROM import_items WHERE session_id = ? ORDER BY rowid
            """,
            (session_id,),
        ).fetchall()
        items = tuple(
            ImportItemRecord(
                item_id=row[0],
                source_locator_digest=row[1],
                selected_action=ImportAction(row[2]),
                outcome_count=row[3],
                outcome=ImportItemOutcome(row[4]),
                target_note_id=row[5],
                expected_version=row[6],
                observed_version=row[7],
                reason_code=row[8],
                retryable=bool(row[9]),
            )
            for row in item_rows
        )
        return ImportSessionSnapshot(
            approval_id=session_row[1],
            session_id=session_row[0],
            plan_digest=session_row[2],
            state=ImportSessionState(session_row[3]),
            batch_size=session_row[4],
            total=session_row[5],
            reason_code=session_row[6],
            items=items,
            payload_effects=NoteImportReceiptRepository._load_effects(
                connection, _PAYLOAD_TABLE, session_id
            ),
            folder_effects=NoteImportReceiptRepository._load_effects(
                connection, _FOLDER_TABLE, session_id
            ),
            membership_effects=NoteImportReceiptRepository._load_effects(
                connection, _MEMBERSHIP_TABLE, session_id
            ),
        )

    @staticmethod
    def _load_effects(
        connection: sqlite3.Connection,
        table: str,
        session_id: str,
    ) -> tuple[ImportEffectRecord, ...]:
        if table == _PAYLOAD_TABLE:
            rows = connection.execute(
                """
                SELECT effect_id, item_id, payload_index, effect_kind, state,
                       target_note_id, NULL, expected_version, observed_version,
                       reason_code, retryable
                FROM import_payload_effects WHERE session_id = ? ORDER BY rowid
                """,
                (session_id,),
            ).fetchall()
        elif table == _FOLDER_TABLE:
            rows = connection.execute(
                """
                SELECT effect_id, NULL, NULL, effect_kind, state, NULL,
                       target_folder_id, NULL, NULL, reason_code, retryable
                FROM import_folder_effects WHERE session_id = ? ORDER BY folder_ordinal
                """,
                (session_id,),
            ).fetchall()
        elif table == _MEMBERSHIP_TABLE:
            rows = connection.execute(
                """
                SELECT effect_id, item_id, payload_index, effect_kind, state,
                       target_note_id, target_folder_id, NULL, NULL,
                       reason_code, retryable
                FROM import_membership_effects WHERE session_id = ? ORDER BY rowid
                """,
                (session_id,),
            ).fetchall()
        else:
            raise ValueError("Unknown receipt effect table.")
        return tuple(
            ImportEffectRecord(
                table=table,
                effect_id=row[0],
                item_id=row[1],
                payload_index=row[2],
                effect_kind=row[3],
                state=ImportEffectState(row[4]),
                target_note_id=row[5],
                target_folder_id=row[6],
                expected_version=row[7],
                observed_version=row[8],
                reason_code=row[9],
                retryable=bool(row[10]),
            )
            for row in rows
        )

    def transition_session(
        self,
        approval_id: str,
        state: ImportSessionState,
        *,
        reason_code: str | None = None,
    ) -> ImportSessionSnapshot:
        """Advance one session through the exact lifecycle allowlist."""

        _validate_id(approval_id, field_name="approval_id")
        if type(state) is not ImportSessionState:
            raise TypeError("state must be an ImportSessionState.")
        validated_reason = _validate_reason_code(reason_code)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._initialize_schema(connection)
            row = connection.execute(
                "SELECT session_id, state FROM import_sessions WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                raise KeyError("Import receipt session was not found.")
            session_id = row[0]
            current = ImportSessionState(row[1])
            if state not in SESSION_STATE_TRANSITIONS[current]:
                raise ImportReceiptTransitionError(
                    "The requested session transition is not allowed."
                )
            if state is ImportSessionState.COMPLETED:
                self._validate_completion(connection, session_id)
            connection.execute(
                """
                UPDATE import_sessions SET state = ?, reason_code = ?, updated_at = ?
                WHERE approval_id = ?
                """,
                (state.value, validated_reason, _now(), approval_id),
            )
            snapshot = self._load_snapshot(connection, approval_id)
            connection.commit()
            return snapshot
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_completion(
        connection: sqlite3.Connection,
        session_id: str,
    ) -> None:
        invalid_items = connection.execute(
            """
            SELECT COUNT(*) FROM import_items
            WHERE session_id = ? AND (
                outcome IN ('pending', 'failed')
                OR (selected_action = 'skip' AND outcome != 'skipped')
                OR (selected_action = 'create_new' AND outcome != 'imported')
                OR (selected_action = 'update_existing' AND outcome != 'updated')
            )
            """,
            (session_id,),
        ).fetchone()[0]
        unapplied_effects = 0
        for statement in (
            "SELECT COUNT(*) FROM import_payload_effects WHERE session_id = ? AND state != 'applied'",
            "SELECT COUNT(*) FROM import_folder_effects WHERE session_id = ? AND state != 'applied'",
            "SELECT COUNT(*) FROM import_membership_effects WHERE session_id = ? AND state != 'applied'",
        ):
            unapplied_effects += connection.execute(
                statement,
                (session_id,),
            ).fetchone()[0]
        if invalid_items or unapplied_effects:
            raise ImportReceiptTransitionError(
                "A session can complete only after every approved outcome and effect."
            )

    def transition_item(
        self,
        approval_id: str,
        item_id: str,
        outcome: ImportItemOutcome,
        *,
        reason_code: str | None = None,
        retryable: bool = False,
        target_note_id: str | None = None,
        observed_version: int | None = None,
    ) -> ImportItemRecord:
        """Atomically transition one item outcome."""

        snapshot = self.transition_batch(
            approval_id,
            item_transitions=(
                ItemTransition(
                    item_id=item_id,
                    outcome=outcome,
                    reason_code=reason_code,
                    retryable=retryable,
                    target_note_id=target_note_id,
                    observed_version=observed_version,
                ),
            ),
        )
        return next(item for item in snapshot.items if item.item_id == item_id)

    def transition_effects(
        self,
        approval_id: str,
        transitions: Sequence[EffectTransition],
    ) -> tuple[ImportEffectRecord, ...]:
        """Atomically transition selected independently replayable effects."""

        copied = _copy_bounded_transitions(
            transitions,
            field_name="effect_transitions",
        )
        snapshot = self.transition_batch(
            approval_id,
            effect_transitions=copied,
        )
        by_key = {
            (effect.table, effect.effect_id): effect
            for effect in (
                *snapshot.payload_effects,
                *snapshot.folder_effects,
                *snapshot.membership_effects,
            )
        }
        return tuple(
            by_key[(transition.table, transition.effect_id)] for transition in copied
        )

    def transition_batch(
        self,
        approval_id: str,
        *,
        item_transitions: Sequence[ItemTransition] = (),
        effect_transitions: Sequence[EffectTransition] = (),
    ) -> ImportSessionSnapshot:
        """Apply selected item and effect transitions in one transaction."""

        _validate_id(approval_id, field_name="approval_id")
        items = _copy_bounded_transitions(
            item_transitions,
            field_name="item_transitions",
        )
        effects = _copy_bounded_transitions(
            effect_transitions,
            field_name="effect_transitions",
        )
        if not items and not effects:
            raise ValueError("At least one transition is required.")
        if len(items) + len(effects) > _MAX_TRANSITIONS:
            raise ValueError("The combined transition collection exceeds the ceiling.")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._initialize_schema(connection)
            session_row = connection.execute(
                "SELECT session_id, state FROM import_sessions WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError("Import receipt session was not found.")
            session_id = session_row[0]
            if ImportSessionState(session_row[1]) is not ImportSessionState.RUNNING:
                raise ImportReceiptTransitionError(
                    "Item and effect transitions require a running session."
                )
            self._validate_item_transitions(connection, session_id, items)
            self._validate_effect_transitions(connection, session_id, effects)
            timestamp = _now()
            for transition in items:
                reason_code, retryable = _validate_transition_metadata(
                    failed=transition.outcome is ImportItemOutcome.FAILED,
                    reason_code=transition.reason_code,
                    retryable=transition.retryable,
                )
                connection.execute(
                    """
                    UPDATE import_items
                    SET outcome = ?, target_note_id = COALESCE(?, target_note_id),
                        observed_version = COALESCE(?, observed_version),
                        reason_code = ?, retryable = ?, updated_at = ?
                    WHERE session_id = ? AND item_id = ?
                    """,
                    (
                        transition.outcome.value,
                        _validate_optional_id(
                            transition.target_note_id, field_name="target_note_id"
                        ),
                        _validate_optional_version(transition.observed_version),
                        reason_code,
                        int(retryable),
                        timestamp,
                        session_id,
                        transition.item_id,
                    ),
                )
            for transition in effects:
                self._update_effect(connection, session_id, transition, timestamp)
            snapshot = self._load_snapshot(connection, approval_id)
            connection.commit()
            return snapshot
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_item_transitions(
        connection: sqlite3.Connection,
        session_id: str,
        transitions: tuple[ItemTransition, ...],
    ) -> None:
        seen: set[str] = set()
        for transition in transitions:
            if type(transition) is not ItemTransition:
                raise TypeError("item_transitions must contain ItemTransition values.")
            item_id = _validate_id(transition.item_id, field_name="item_id")
            if item_id in seen:
                raise ValueError(
                    "An item may be transitioned only once per transaction."
                )
            seen.add(item_id)
            if type(transition.outcome) is not ImportItemOutcome:
                raise TypeError("outcome must be an ImportItemOutcome.")
            _validate_transition_metadata(
                failed=transition.outcome is ImportItemOutcome.FAILED,
                reason_code=transition.reason_code,
                retryable=transition.retryable,
            )
            _validate_optional_id(
                transition.target_note_id, field_name="target_note_id"
            )
            _validate_optional_version(transition.observed_version)
            row = connection.execute(
                """
                SELECT outcome, selected_action, target_note_id, observed_version
                FROM import_items WHERE session_id = ? AND item_id = ?
                """,
                (session_id, item_id),
            ).fetchone()
            if row is None:
                raise KeyError("Import receipt item was not found.")
            current = ImportItemOutcome(row[0])
            if transition.outcome not in ITEM_OUTCOME_TRANSITIONS[current]:
                raise ImportReceiptTransitionError(
                    "The requested item transition is not allowed."
                )
            action = ImportAction(row[1])
            allowed_outcomes = {
                ImportAction.SKIP: {
                    ImportItemOutcome.SKIPPED,
                    ImportItemOutcome.FAILED,
                },
                ImportAction.CREATE_NEW: {
                    ImportItemOutcome.IMPORTED,
                    ImportItemOutcome.FAILED,
                },
                ImportAction.UPDATE_EXISTING: {
                    ImportItemOutcome.UPDATED,
                    ImportItemOutcome.FAILED,
                },
            }
            if transition.outcome not in allowed_outcomes[action]:
                raise ImportReceiptTransitionError(
                    "The requested item outcome does not match its approved action."
                )
            _assert_compatible_authority(row[2], transition.target_note_id)
            _assert_compatible_authority(row[3], transition.observed_version)

    @staticmethod
    def _validate_effect_transitions(
        connection: sqlite3.Connection,
        session_id: str,
        transitions: tuple[EffectTransition, ...],
    ) -> None:
        seen: set[tuple[str, str]] = set()
        for transition in transitions:
            if type(transition) is not EffectTransition:
                raise TypeError(
                    "effect_transitions must contain EffectTransition values."
                )
            if transition.table not in _EFFECT_TABLES:
                raise ValueError("Unknown receipt effect table.")
            effect_id = _validate_id(transition.effect_id, field_name="effect_id")
            key = (transition.table, effect_id)
            if key in seen:
                raise ValueError(
                    "An effect may be transitioned only once per transaction."
                )
            seen.add(key)
            if type(transition.state) is not ImportEffectState:
                raise TypeError("state must be an ImportEffectState.")
            _validate_transition_metadata(
                failed=transition.state is ImportEffectState.FAILED,
                reason_code=transition.reason_code,
                retryable=transition.retryable,
            )
            _validate_optional_id(
                transition.target_note_id, field_name="target_note_id"
            )
            _validate_optional_id(
                transition.target_folder_id, field_name="target_folder_id"
            )
            _validate_optional_version(transition.observed_version)
            row = NoteImportReceiptRepository._select_effect_authority(
                connection,
                table=transition.table,
                session_id=session_id,
                effect_id=effect_id,
            )
            if row is None:
                raise KeyError("Import receipt effect was not found.")
            current = ImportEffectState(row[0])
            if transition.state not in EFFECT_STATE_TRANSITIONS[current]:
                raise ImportReceiptTransitionError(
                    "The requested effect transition is not allowed."
                )
            stored_note_id, stored_folder_id, stored_version = row[1:]
            _assert_compatible_authority(stored_note_id, transition.target_note_id)
            _assert_compatible_authority(
                stored_folder_id,
                transition.target_folder_id,
            )
            _assert_compatible_authority(stored_version, transition.observed_version)
            if transition.table == _PAYLOAD_TABLE:
                if transition.target_folder_id is not None:
                    raise ValueError("Payload effects cannot bind a folder identifier.")
            elif transition.table == _FOLDER_TABLE:
                if (
                    transition.target_note_id is not None
                    or transition.observed_version is not None
                ):
                    raise ValueError("Folder effects accept only a folder identifier.")
            elif transition.observed_version is not None:
                raise ValueError("Membership effects do not accept a note version.")
            if transition.state is ImportEffectState.APPLIED:
                final_note_id = transition.target_note_id or stored_note_id
                final_folder_id = transition.target_folder_id or stored_folder_id
                final_version = (
                    transition.observed_version
                    if transition.observed_version is not None
                    else stored_version
                )
                missing_identity = (
                    transition.table == _PAYLOAD_TABLE
                    and (final_note_id is None or final_version is None)
                ) or (transition.table == _FOLDER_TABLE and final_folder_id is None)
                missing_identity = missing_identity or (
                    transition.table == _MEMBERSHIP_TABLE
                    and (final_note_id is None or final_folder_id is None)
                )
                if missing_identity:
                    raise ImportReceiptTransitionError(
                        "Applied effects require their reconciliation identities."
                    )

    @staticmethod
    def _select_effect_authority(
        connection: sqlite3.Connection,
        *,
        table: str,
        session_id: str,
        effect_id: str,
    ) -> tuple[str, str | None, str | None, int | None] | None:
        if table == _PAYLOAD_TABLE:
            return connection.execute(
                """
                SELECT state, target_note_id, NULL, observed_version
                FROM import_payload_effects
                WHERE session_id = ? AND effect_id = ?
                """,
                (session_id, effect_id),
            ).fetchone()
        if table == _FOLDER_TABLE:
            return connection.execute(
                """
                SELECT state, NULL, target_folder_id, NULL
                FROM import_folder_effects
                WHERE session_id = ? AND effect_id = ?
                """,
                (session_id, effect_id),
            ).fetchone()
        return connection.execute(
            """
            SELECT state, target_note_id, target_folder_id, NULL
            FROM import_membership_effects
            WHERE session_id = ? AND effect_id = ?
            """,
            (session_id, effect_id),
        ).fetchone()

    @staticmethod
    def _update_effect(
        connection: sqlite3.Connection,
        session_id: str,
        transition: EffectTransition,
        timestamp: int,
    ) -> None:
        reason_code, retryable = _validate_transition_metadata(
            failed=transition.state is ImportEffectState.FAILED,
            reason_code=transition.reason_code,
            retryable=transition.retryable,
        )
        if transition.table == _PAYLOAD_TABLE:
            connection.execute(
                """
                UPDATE import_payload_effects
                SET state = ?, reason_code = ?, retryable = ?, updated_at = ?,
                    target_note_id = COALESCE(?, target_note_id),
                    observed_version = COALESCE(?, observed_version)
                WHERE session_id = ? AND effect_id = ?
                """,
                (
                    transition.state.value,
                    reason_code,
                    int(retryable),
                    timestamp,
                    transition.target_note_id,
                    transition.observed_version,
                    session_id,
                    transition.effect_id,
                ),
            )
        elif transition.table == _FOLDER_TABLE:
            connection.execute(
                """
                UPDATE import_folder_effects
                SET state = ?, reason_code = ?, retryable = ?, updated_at = ?,
                    target_folder_id = COALESCE(?, target_folder_id)
                WHERE session_id = ? AND effect_id = ?
                """,
                (
                    transition.state.value,
                    reason_code,
                    int(retryable),
                    timestamp,
                    transition.target_folder_id,
                    session_id,
                    transition.effect_id,
                ),
            )
        else:
            connection.execute(
                """
                UPDATE import_membership_effects
                SET state = ?, reason_code = ?, retryable = ?, updated_at = ?,
                    target_note_id = COALESCE(?, target_note_id),
                    target_folder_id = COALESCE(?, target_folder_id)
                WHERE session_id = ? AND effect_id = ?
                """,
                (
                    transition.state.value,
                    reason_code,
                    int(retryable),
                    timestamp,
                    transition.target_note_id,
                    transition.target_folder_id,
                    session_id,
                    transition.effect_id,
                ),
            )

    def reset_retryable_item(
        self,
        approval_id: str,
        *,
        item_id: str,
    ) -> ImportItemRecord:
        """Explicitly reset one retryable failed item to pending."""

        _validate_id(item_id, field_name="item_id")
        snapshot = self._reset_retryable(
            approval_id,
            table="import_items",
            key_value=item_id,
        )
        return next(item for item in snapshot.items if item.item_id == item_id)

    def reset_retryable_effect(
        self,
        approval_id: str,
        *,
        table: str,
        effect_id: str,
    ) -> ImportEffectRecord:
        """Explicitly reset one retryable failed effect to pending."""

        if table not in _EFFECT_TABLES:
            raise ValueError("Unknown receipt effect table.")
        _validate_id(effect_id, field_name="effect_id")
        snapshot = self._reset_retryable(
            approval_id,
            table=table,
            key_value=effect_id,
        )
        effects = (
            *snapshot.payload_effects,
            *snapshot.folder_effects,
            *snapshot.membership_effects,
        )
        return next(
            effect
            for effect in effects
            if effect.table == table and effect.effect_id == effect_id
        )

    def _reset_retryable(
        self,
        approval_id: str,
        *,
        table: str,
        key_value: str,
    ) -> ImportSessionSnapshot:
        _validate_id(approval_id, field_name="approval_id")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._initialize_schema(connection)
            session_row = connection.execute(
                "SELECT session_id, state FROM import_sessions WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError("Import receipt session was not found.")
            session_id = session_row[0]
            if (
                ImportSessionState(session_row[1])
                is not ImportSessionState.NEEDS_ATTENTION
            ):
                raise ImportReceiptTransitionError(
                    "Retry resets require a session that needs attention."
                )
            row = self._select_retryable_row(
                connection,
                table=table,
                session_id=session_id,
                key_value=key_value,
            )
            if row is None:
                raise KeyError("Import receipt row was not found.")
            if row != ("failed", 1):
                raise ImportReceiptTransitionError(
                    "Only retryable failed rows may be reset."
                )
            self._update_retryable_row(
                connection,
                table=table,
                session_id=session_id,
                key_value=key_value,
            )
            snapshot = self._load_snapshot(connection, approval_id)
            connection.commit()
            return snapshot
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _select_retryable_row(
        connection: sqlite3.Connection,
        *,
        table: str,
        session_id: str,
        key_value: str,
    ) -> tuple[str, int] | None:
        if table == "import_items":
            return connection.execute(
                """
                SELECT outcome, retryable FROM import_items
                WHERE session_id = ? AND item_id = ?
                """,
                (session_id, key_value),
            ).fetchone()
        if table == _PAYLOAD_TABLE:
            return connection.execute(
                """
                SELECT state, retryable FROM import_payload_effects
                WHERE session_id = ? AND effect_id = ?
                """,
                (session_id, key_value),
            ).fetchone()
        if table == _FOLDER_TABLE:
            return connection.execute(
                """
                SELECT state, retryable FROM import_folder_effects
                WHERE session_id = ? AND effect_id = ?
                """,
                (session_id, key_value),
            ).fetchone()
        return connection.execute(
            """
            SELECT state, retryable FROM import_membership_effects
            WHERE session_id = ? AND effect_id = ?
            """,
            (session_id, key_value),
        ).fetchone()

    @staticmethod
    def _update_retryable_row(
        connection: sqlite3.Connection,
        *,
        table: str,
        session_id: str,
        key_value: str,
    ) -> None:
        parameters = ("pending", None, 0, _now(), session_id, key_value)
        if table == "import_items":
            connection.execute(
                """
                UPDATE import_items
                SET outcome = ?, reason_code = ?, retryable = ?, updated_at = ?
                WHERE session_id = ? AND item_id = ?
                """,
                parameters,
            )
        elif table == _PAYLOAD_TABLE:
            connection.execute(
                """
                UPDATE import_payload_effects
                SET state = ?, reason_code = ?, retryable = ?, updated_at = ?
                WHERE session_id = ? AND effect_id = ?
                """,
                parameters,
            )
        elif table == _FOLDER_TABLE:
            connection.execute(
                """
                UPDATE import_folder_effects
                SET state = ?, reason_code = ?, retryable = ?, updated_at = ?
                WHERE session_id = ? AND effect_id = ?
                """,
                parameters,
            )
        else:
            connection.execute(
                """
                UPDATE import_membership_effects
                SET state = ?, reason_code = ?, retryable = ?, updated_at = ?
                WHERE session_id = ? AND effect_id = ?
                """,
                parameters,
            )

    def aggregate_receipt(self, approval_id: str) -> ImportExecutionReceipt:
        """Aggregate one session into the existing immutable receipt model."""

        _validate_id(approval_id, field_name="approval_id")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._initialize_schema(connection)
            snapshot = self._load_snapshot(connection, approval_id)
            payload_digests = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT payload_digest FROM import_payload_effects
                    WHERE session_id = ? ORDER BY rowid
                    """,
                    (snapshot.session_id,),
                ).fetchall()
            )
            receipt = self._build_receipt(snapshot, payload_digests)
            connection.commit()
            return receipt
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _build_receipt(
        snapshot: ImportSessionSnapshot,
        payload_digests: tuple[str, ...],
    ) -> ImportExecutionReceipt:
        counts = {outcome: 0 for outcome in ImportItemOutcome}
        for item in snapshot.items:
            counts[item.outcome] += item.outcome_count
        completed = sum(
            item.outcome_count
            for item in snapshot.items
            if item.outcome is not ImportItemOutcome.PENDING
        )
        failed_items = [
            item for item in snapshot.items if item.outcome is ImportItemOutcome.FAILED
        ]
        effects = (
            *snapshot.payload_effects,
            *snapshot.folder_effects,
            *snapshot.membership_effects,
        )
        reason_code = snapshot.reason_code
        if reason_code is None:
            reason_code = next(
                (
                    item.reason_code
                    for item in failed_items
                    if item.reason_code is not None
                ),
                None,
            )
        return ImportExecutionReceipt(
            approval_id=snapshot.approval_id,
            state=snapshot.state,
            total=snapshot.total,
            completed=completed,
            imported=counts[ImportItemOutcome.IMPORTED],
            updated=counts[ImportItemOutcome.UPDATED],
            skipped=counts[ImportItemOutcome.SKIPPED],
            failed=counts[ImportItemOutcome.FAILED],
            retryable=sum(
                item.outcome_count for item in failed_items if item.retryable
            ),
            reason_code=reason_code,
            _note_ids=tuple(
                dict.fromkeys(
                    (
                        *(
                            effect.target_note_id
                            for effect in snapshot.payload_effects
                            if effect.target_note_id is not None
                        ),
                        *(
                            item.target_note_id
                            for item in snapshot.items
                            if item.target_note_id is not None
                        ),
                    )
                )
            ),
            _folder_ids=tuple(
                dict.fromkeys(
                    effect.target_folder_id
                    for effect in effects
                    if effect.target_folder_id is not None
                )
            ),
            _source_locator_digests=tuple(
                item.source_locator_digest for item in snapshot.items
            ),
            _payload_fingerprints=payload_digests,
        )

    def _test_schema_snapshot(self) -> ReceiptSchemaSnapshot:
        """Return an immutable schema census for privacy contract tests."""

        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._initialize_schema(connection)
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = ? AND name LIKE ?
                    ORDER BY name
                    """,
                    ("table", "import_%"),
                ).fetchall()
            )
            columns: dict[str, tuple[str, ...]] = {}
            for table in tables:
                if table not in {
                    "import_sessions",
                    "import_items",
                    *_EFFECT_TABLES,
                }:
                    raise ImportReceiptError("Unexpected private receipt table.")
                columns[table] = tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM pragma_table_info(?) ORDER BY cid",
                        (table,),
                    ).fetchall()
                )
            connection.commit()
            return ReceiptSchemaSnapshot(
                user_version=user_version,
                tables=tables,
                columns=MappingProxyType(columns),
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


__all__ = [
    "EFFECT_STATE_TRANSITIONS",
    "ITEM_OUTCOME_TRANSITIONS",
    "SESSION_STATE_TRANSITIONS",
    "EffectTransition",
    "ImportEffectRecord",
    "ImportItemRecord",
    "ImportReceiptConflictError",
    "ImportReceiptError",
    "ImportReceiptTransitionError",
    "ImportSessionSnapshot",
    "ItemTransition",
    "NoteImportReceiptRepository",
    "ReceiptSchemaSnapshot",
]
