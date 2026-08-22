"""Device-private durable receipts for approved one-time Notes imports.

The ledger deliberately stores only opaque identifiers, private digests, and
bounded lifecycle metadata. Source paths and note payload data never cross this
persistence boundary.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from tldw_chatbook.Notes.note_import_execution_models import (
    MAX_RECEIPT_LEDGER_ROWS,
    ApprovedNoteImportPlan,
    ImportEffectState,
    ImportExecutionReceipt,
    ImportItemOutcome,
    ImportSessionState,
    _canonical_json_digest,
    _private_payload_fingerprint,
    _private_source_locator_digest,
    _receipt_ledger_row_count,
    _validate_reason_code,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    MAX_IMPORT_ENTRIES,
    ImportAction,
    ImportMatchKind,
    ImportPreviewItem,
    NoteImportPlan,
)
from tldw_chatbook.Notes.note_import_planner import PriorImportObservation
from tldw_chatbook.Notes.notes_sync_state_schema import (
    NotesSyncStateSchemaError,
    notes_sync_state_transaction,
)

_MIN_BATCH_SIZE = 1
_MAX_BATCH_SIZE = 100
_MAX_TRANSITIONS = MAX_IMPORT_ENTRIES
_SQL_PARAMETER_CHUNK = 32
_PRIOR_OBSERVATION_CHUNK_CAP = 900
_SQLITE_VARIABLE_LIMIT_FALLBACK = 999
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


class ImportEffectCategory(str, Enum):
    """Semantic category for one independently replayable import effect."""

    PAYLOAD = "payload"
    FOLDER = "folder"
    MEMBERSHIP = "membership"


_CATEGORY_TO_TABLE: Mapping[ImportEffectCategory, str] = MappingProxyType(
    {
        ImportEffectCategory.PAYLOAD: _PAYLOAD_TABLE,
        ImportEffectCategory.FOLDER: _FOLDER_TABLE,
        ImportEffectCategory.MEMBERSHIP: _MEMBERSHIP_TABLE,
    }
)
_TABLE_TO_CATEGORY: Mapping[str, ImportEffectCategory] = MappingProxyType(
    {table: category for category, table in _CATEGORY_TO_TABLE.items()}
)


def _table_for_category(category: object) -> str:
    if type(category) is not ImportEffectCategory:
        raise TypeError("category must be an ImportEffectCategory.")
    return _CATEGORY_TO_TABLE[category]


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

    category: ImportEffectCategory
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
    folder_path_digest: str | None = field(default=None, repr=False)
    parent_effect_id: str | None = field(default=None, repr=False)

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


@dataclass(frozen=True, slots=True, repr=False)
class ImportBatchResult:
    """Frozen changed-row projection returned by one transition batch."""

    items: tuple[ImportItemRecord, ...] = ()
    effects: tuple[ImportEffectRecord, ...] = ()

    def __repr__(self) -> str:
        return (
            f"ImportBatchResult(items={len(self.items)!r}, "
            f"effects={len(self.effects)!r})"
        )


class _PrivatePriorImportObservation(PriorImportObservation):
    """Planner-compatible prior observation with an opaque representation."""

    def __repr__(self) -> str:
        return "PriorImportObservation(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class _SeedBlueprint:
    """Fully validated, content-free rows ready for one atomic seed."""

    session: tuple[object, ...]
    items: tuple[tuple[object, ...], ...]
    payloads: tuple[tuple[object, ...], ...]
    folders: tuple[tuple[object, ...], ...]
    memberships: tuple[tuple[object, ...], ...]


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

    category: ImportEffectCategory
    effect_id: str = field(repr=False)
    state: ImportEffectState
    reason_code: str | None = None
    retryable: bool = False
    target_note_id: str | None = field(default=None, repr=False)
    target_folder_id: str | None = field(default=None, repr=False)
    observed_version: int | None = None


def _now() -> int:
    return max(1, time.time_ns())


def _validate_batch_size(batch_size: object) -> int:
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an integer.")
    if not _MIN_BATCH_SIZE <= batch_size <= _MAX_BATCH_SIZE:
        raise ValueError("batch_size must be between 1 and 100.")
    return batch_size


def _prior_observation_chunk_size(connection: sqlite3.Connection) -> int:
    """Return a bounded digest chunk that respects this connection's SQL limit."""

    try:
        variable_limit = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    except (AttributeError, sqlite3.Error):
        variable_limit = _SQLITE_VARIABLE_LIMIT_FALLBACK
    if type(variable_limit) is not int or variable_limit <= 0:
        variable_limit = _SQLITE_VARIABLE_LIMIT_FALLBACK
    available = variable_limit - 1  # Reserve one binding for session.state.
    if available < 1:
        raise ImportReceiptError(
            "The SQLite variable limit is too low for prior import recovery."
        )
    return min(_PRIOR_OBSERVATION_CHUNK_CAP, available)


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


def _sql_chunks(values: set[str]) -> tuple[tuple[str, ...], ...]:
    ordered = sorted(values)
    return tuple(
        tuple(ordered[offset : offset + _SQL_PARAMETER_CHUNK])
        for offset in range(0, len(ordered), _SQL_PARAMETER_CHUNK)
    )


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
    """Expose the receipt API over the shared private sync-state schema."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    def __repr__(self) -> str:
        return "NoteImportReceiptRepository(<private>)"

    @contextmanager
    def transaction(
        self,
        *,
        immediate: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        """Run receipt-ledger work in one standardized transaction.

        Args:
            immediate: Reserve SQLite's writer slot before yielding when true.

        Yields:
            The active private receipt-ledger connection.

        Raises:
            ImportReceiptError: If the shared private schema is unavailable.
            Exception: Re-raises operation errors after rolling back.
        """

        try:
            with notes_sync_state_transaction(
                self._database_path,
                immediate=immediate,
            ) as connection:
                yield connection
        except NotesSyncStateSchemaError:
            raise ImportReceiptError(
                "The private receipt schema is incompatible with canonical v1."
            ) from None

    def begin(
        self,
        approved: ApprovedNoteImportPlan,
        *,
        batch_size: int,
    ) -> ImportSessionSnapshot:
        """Create or durably reopen one exact approved import session.

        Args:
            approved: Opaque authority for the exact plan to seed or reopen.
            batch_size: Bounded number of effects processed per executor batch.

        Returns:
            The durable snapshot bound to the approval and batch size.

        Raises:
            ImportReceiptConflictError: If the approval is already bound to
                different receipt authority.
            ImportReceiptError: If the approved plan cannot be seeded safely.
            TypeError: If an argument has the wrong type.
            ValueError: If an argument violates a bounded value contract.
        """

        validated_batch_size = _validate_batch_size(batch_size)
        if type(approved) is not ApprovedNoteImportPlan:
            raise TypeError("approved must be an ApprovedNoteImportPlan.")
        approval_id = approved.approval_id
        plan_digest = approved._private_plan_digest()
        if _DIGEST_PATTERN.fullmatch(plan_digest) is None:
            raise ValueError("The approved plan digest is invalid.")
        blueprint = self._build_seed_blueprint(
            approved,
            batch_size=validated_batch_size,
            plan_digest=plan_digest,
        )

        with self.transaction(immediate=True) as connection:
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
                try:
                    self._seed_blueprint(connection, blueprint)
                except (ImportReceiptError, TypeError, ValueError, sqlite3.Error):
                    raise
                except Exception:  # noqa: BLE001 - sanitize unexpected private failures
                    raise ImportReceiptError(
                        "The approved import receipt could not be seeded safely."
                    ) from None
        return self.get_session(approval_id)

    @staticmethod
    def _build_seed_blueprint(
        approved: ApprovedNoteImportPlan,
        *,
        batch_size: int,
        plan_digest: str,
    ) -> _SeedBlueprint:
        """Validate and canonicalize every planned row before opening SQLite."""

        session_id = str(uuid4())
        timestamp = _now()
        plan = approved.plan
        bounds = plan.bounds
        if len(plan.items) > bounds.max_files or len(plan.items) > bounds.max_entries:
            raise ImportReceiptError("The approved import plan exceeds its bounds.")
        if len(plan.proposed_folder_paths) > bounds.max_entries:
            raise ImportReceiptError("The approved import plan exceeds its bounds.")
        if len(set(plan.proposed_folder_paths)) != len(plan.proposed_folder_paths):
            raise ImportReceiptError("The approved import folder plan is ambiguous.")

        required_folder_paths: set[tuple[str, ...]] = set()
        for item in plan.items:
            if len(item.payloads) > bounds.max_notes_per_file:
                raise ImportReceiptError("The approved import plan exceeds its bounds.")
            if any(
                len(payload.keywords) > bounds.max_keywords_per_note
                for payload in item.payloads
            ):
                raise ImportReceiptError("The approved import plan exceeds its bounds.")
            if item.selected_action is ImportAction.SKIP or not item.add_membership:
                continue
            if len(item.memberships) > bounds.max_entries:
                raise ImportReceiptError("The approved import plan exceeds its bounds.")
            for membership in item.memberships:
                path = tuple(membership.folder_segments)
                required_folder_paths.update(
                    path[:depth] for depth in range(1, len(path) + 1)
                )
        if len(required_folder_paths) > bounds.max_entries:
            raise ImportReceiptError("The approved import plan exceeds its bounds.")
        proposed_ordinals = {
            path: ordinal for ordinal, path in enumerate(plan.proposed_folder_paths)
        }
        if required_folder_paths.difference(proposed_ordinals):
            raise ImportReceiptError(
                "The approved import folder plan is missing a required path or prefix."
            )
        ordered_folders = sorted(
            required_folder_paths,
            key=lambda path: (len(path), proposed_ordinals[path]),
        )
        ledger_rows = _receipt_ledger_row_count(plan)
        if ledger_rows > MAX_RECEIPT_LEDGER_ROWS:
            raise ImportReceiptError(
                "The approved import receipt exceeds its ledger ceiling."
            )

        item_rows: list[tuple[object, ...]] = []
        payload_rows: list[tuple[object, ...]] = []
        membership_rows: list[tuple[object, ...]] = []
        total_count = sum(
            _outcome_count(item.selected_action, len(item.payloads))
            for item in plan.items
        )
        for item in plan.items:
            outcome_count = _outcome_count(
                item.selected_action,
                len(item.payloads),
            )
            update_match = (
                item.match
                if item.selected_action is ImportAction.UPDATE_EXISTING
                else None
            )
            expected_version = (
                update_match.note_version if update_match is not None else None
            )
            target_note_id = _validate_optional_id(
                update_match.note_id if update_match is not None else None,
                field_name="target_note_id",
            )
            item_rows.append(
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
                )
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
                    payload_rows.append(
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
            if item.selected_action is not ImportAction.SKIP and item.add_membership:
                for membership_ordinal, membership in enumerate(item.memberships):
                    path = tuple(membership.folder_segments)
                    membership_rows.append(
                        (
                            str(uuid4()),
                            session_id,
                            item.item_id,
                            membership.payload_index,
                            membership_ordinal,
                            _safe_private_digest(
                                lambda path=path: _folder_path_digest(path)
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

        folder_effect_ids = {path: str(uuid4()) for path in ordered_folders}
        folder_rows = tuple(
            (
                folder_effect_ids[path],
                session_id,
                folder_ordinal,
                _safe_private_digest(lambda path=path: _folder_path_digest(path)),
                folder_effect_ids.get(path[:-1]),
                "ensure_folder",
                ImportEffectState.PENDING.value,
                None,
                None,
                0,
                timestamp,
                timestamp,
            )
            for folder_ordinal, path in enumerate(ordered_folders)
        )
        if ledger_rows != (
            1
            + len(item_rows)
            + len(payload_rows)
            + len(folder_rows)
            + len(membership_rows)
        ):
            raise ImportReceiptError(
                "The approved import receipt plan is inconsistent."
            )
        return _SeedBlueprint(
            session=(
                session_id,
                approved.approval_id,
                plan_digest,
                ImportSessionState.PENDING.value,
                batch_size,
                total_count,
                None,
                timestamp,
                timestamp,
            ),
            items=tuple(item_rows),
            payloads=tuple(payload_rows),
            folders=folder_rows,
            memberships=tuple(membership_rows),
        )

    @staticmethod
    def _seed_blueprint(
        connection: sqlite3.Connection,
        blueprint: _SeedBlueprint,
    ) -> None:
        connection.execute(
            """
            INSERT INTO import_sessions (
                session_id, approval_id, plan_digest, state, batch_size,
                total_count, reason_code, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            blueprint.session,
        )
        connection.executemany(
            """
            INSERT INTO import_items (
                session_id, item_id, source_locator_digest, selected_action,
                outcome_count, outcome, target_note_id, expected_version,
                observed_version, reason_code, retryable, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            blueprint.items,
        )
        connection.executemany(
            """
            INSERT INTO import_payload_effects (
                effect_id, session_id, item_id, payload_index, payload_digest,
                effect_kind, state, target_note_id, expected_version,
                observed_version, reason_code, retryable, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            blueprint.payloads,
        )
        connection.executemany(
            """
            INSERT INTO import_folder_effects (
                effect_id, session_id, folder_ordinal, path_digest,
                parent_effect_id, effect_kind, state, target_folder_id,
                reason_code, retryable, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            blueprint.folders,
        )
        connection.executemany(
            """
            INSERT INTO import_membership_effects (
                effect_id, session_id, item_id, payload_index,
                membership_ordinal, folder_path_digest, effect_kind, state,
                target_note_id, target_folder_id, reason_code, retryable,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            blueprint.memberships,
        )

    def get_session(self, approval_id: str) -> ImportSessionSnapshot:
        """Return the durable frozen snapshot for one approval."""

        _validate_id(approval_id, field_name="approval_id")
        with self.transaction() as connection:
            return self._load_snapshot(connection, approval_id)

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
                       reason_code, retryable, NULL, NULL
                FROM import_payload_effects WHERE session_id = ? ORDER BY rowid
                """,
                (session_id,),
            ).fetchall()
        elif table == _FOLDER_TABLE:
            rows = connection.execute(
                """
                SELECT effect_id, NULL, NULL, effect_kind, state, NULL,
                       target_folder_id, NULL, NULL, reason_code, retryable,
                       path_digest, parent_effect_id
                FROM import_folder_effects WHERE session_id = ? ORDER BY folder_ordinal
                """,
                (session_id,),
            ).fetchall()
        elif table == _MEMBERSHIP_TABLE:
            rows = connection.execute(
                """
                SELECT effect_id, item_id, payload_index, effect_kind, state,
                       target_note_id, target_folder_id, NULL, NULL,
                       reason_code, retryable, folder_path_digest, NULL
                FROM import_membership_effects WHERE session_id = ? ORDER BY rowid
                """,
                (session_id,),
            ).fetchall()
        else:
            raise ValueError("Unknown receipt effect table.")
        return NoteImportReceiptRepository._effect_records(table, rows)

    @staticmethod
    def _effect_records(
        table: str,
        rows: Sequence[tuple[object, ...]],
    ) -> tuple[ImportEffectRecord, ...]:
        return tuple(
            ImportEffectRecord(
                category=_TABLE_TO_CATEGORY[table],
                effect_id=str(row[0]),
                item_id=None if row[1] is None else str(row[1]),
                payload_index=None if row[2] is None else int(row[2]),
                effect_kind=str(row[3]),
                state=ImportEffectState(str(row[4])),
                target_note_id=None if row[5] is None else str(row[5]),
                target_folder_id=None if row[6] is None else str(row[6]),
                expected_version=None if row[7] is None else int(row[7]),
                observed_version=None if row[8] is None else int(row[8]),
                reason_code=None if row[9] is None else str(row[9]),
                retryable=bool(row[10]),
                folder_path_digest=None if row[11] is None else str(row[11]),
                parent_effect_id=None if row[12] is None else str(row[12]),
            )
            for row in rows
        )

    @staticmethod
    def _item_records(
        rows: Sequence[tuple[object, ...]],
    ) -> tuple[ImportItemRecord, ...]:
        return tuple(
            ImportItemRecord(
                item_id=str(row[0]),
                source_locator_digest=str(row[1]),
                selected_action=ImportAction(str(row[2])),
                outcome_count=int(row[3]),
                outcome=ImportItemOutcome(str(row[4])),
                target_note_id=None if row[5] is None else str(row[5]),
                expected_version=None if row[6] is None else int(row[6]),
                observed_version=None if row[7] is None else int(row[7]),
                reason_code=None if row[8] is None else str(row[8]),
                retryable=bool(row[9]),
            )
            for row in rows
        )

    @staticmethod
    def _affected_item_ids(
        connection: sqlite3.Connection,
        session_id: str,
        items: tuple[ItemTransition, ...],
        effects: tuple[EffectTransition, ...],
    ) -> set[str]:
        affected = {transition.item_id for transition in items}
        for transition in effects:
            if transition.category is ImportEffectCategory.PAYLOAD:
                row = connection.execute(
                    "SELECT item_id FROM import_payload_effects WHERE session_id = ? AND effect_id = ?",
                    (session_id, transition.effect_id),
                ).fetchone()
                if row is not None:
                    affected.add(row[0])
            elif transition.category is ImportEffectCategory.MEMBERSHIP:
                row = connection.execute(
                    "SELECT item_id FROM import_membership_effects WHERE session_id = ? AND effect_id = ?",
                    (session_id, transition.effect_id),
                ).fetchone()
                if row is not None:
                    affected.add(row[0])
            else:
                rows = connection.execute(
                    """
                    WITH RECURSIVE descendants(effect_id, path_digest) AS (
                        SELECT effect_id, path_digest FROM import_folder_effects
                        WHERE session_id = ? AND effect_id = ?
                        UNION ALL
                        SELECT child.effect_id, child.path_digest
                        FROM import_folder_effects AS child
                        JOIN descendants AS parent
                          ON child.parent_effect_id = parent.effect_id
                        WHERE child.session_id = ?
                    )
                    SELECT DISTINCT membership.item_id
                    FROM descendants AS folder
                    JOIN import_membership_effects AS membership
                      ON membership.session_id = ?
                     AND membership.folder_path_digest = folder.path_digest
                    """,
                    (session_id, transition.effect_id, session_id, session_id),
                ).fetchall()
                affected.update(row[0] for row in rows)
        return affected

    @staticmethod
    def _validate_transition_identity_points(
        connection: sqlite3.Connection,
        session_id: str,
        transitions: tuple[EffectTransition, ...],
    ) -> None:
        for transition in transitions:
            if (
                transition.category is ImportEffectCategory.PAYLOAD
                and transition.target_note_id is not None
            ):
                action_row = connection.execute(
                    """SELECT item.selected_action
                    FROM import_payload_effects AS payload
                    JOIN import_items AS item
                      ON item.session_id = payload.session_id
                     AND item.item_id = payload.item_id
                    WHERE payload.session_id = ? AND payload.effect_id = ?""",
                    (session_id, transition.effect_id),
                ).fetchone()
                if action_row is None:
                    raise KeyError("Import receipt effect was not found.")
                if ImportAction(action_row[0]) is ImportAction.CREATE_NEW:
                    duplicate_create = connection.execute(
                        """SELECT 1 FROM import_payload_effects AS payload
                        JOIN import_items AS item
                          ON item.session_id = payload.session_id
                         AND item.item_id = payload.item_id
                        WHERE payload.session_id = ?
                          AND payload.target_note_id = ?
                          AND payload.effect_id != ?
                          AND item.selected_action = 'create_new' LIMIT 1""",
                        (
                            session_id,
                            transition.target_note_id,
                            transition.effect_id,
                        ),
                    ).fetchone()
                    update_alias = connection.execute(
                        """SELECT 1 FROM import_items
                        WHERE session_id = ? AND target_note_id = ?
                          AND selected_action = 'update_existing' LIMIT 1""",
                        (session_id, transition.target_note_id),
                    ).fetchone()
                    if duplicate_create is not None or update_alias is not None:
                        raise ImportReceiptConflictError(
                            "Create payload note identities must be unique per note unit."
                        )
            elif (
                transition.category is ImportEffectCategory.FOLDER
                and transition.target_folder_id is not None
            ):
                duplicate_folder = connection.execute(
                    """SELECT 1 FROM import_folder_effects
                    WHERE session_id = ? AND target_folder_id = ?
                      AND effect_id != ? LIMIT 1""",
                    (
                        session_id,
                        transition.target_folder_id,
                        transition.effect_id,
                    ),
                ).fetchone()
                if duplicate_folder is not None:
                    raise ImportReceiptConflictError(
                        "Distinct approved folder paths require distinct folder identities."
                    )

    @staticmethod
    def _load_dependency_snapshot(
        connection: sqlite3.Connection,
        approval_id: str,
        session_id: str,
        state: ImportSessionState,
        item_ids: set[str],
        folder_effect_ids: set[str],
    ) -> ImportSessionSnapshot:
        item_rows: list[tuple[object, ...]] = []
        payload_rows: list[tuple[object, ...]] = []
        membership_rows: list[tuple[object, ...]] = []
        for chunk in _sql_chunks(item_ids):
            placeholders = ",".join("?" for _ in chunk)
            parameters = (session_id, *chunk)
            item_rows.extend(
                connection.execute(
                    f"""SELECT item_id, source_locator_digest, selected_action,
                        outcome_count, outcome, target_note_id, expected_version,
                        observed_version, reason_code, retryable FROM import_items
                        WHERE session_id = ? AND item_id IN ({placeholders})
                        ORDER BY rowid""",
                    parameters,
                ).fetchall()
            )
            payload_rows.extend(
                connection.execute(
                    f"""SELECT effect_id, item_id, payload_index, effect_kind,
                        state, target_note_id, NULL, expected_version,
                        observed_version, reason_code, retryable, NULL, NULL
                        FROM import_payload_effects WHERE session_id = ?
                        AND item_id IN ({placeholders}) ORDER BY rowid""",
                    parameters,
                ).fetchall()
            )
            membership_rows.extend(
                connection.execute(
                    f"""SELECT effect_id, item_id, payload_index, effect_kind,
                        state, target_note_id, target_folder_id, NULL, NULL,
                        reason_code, retryable, folder_path_digest, NULL
                        FROM import_membership_effects WHERE session_id = ?
                        AND item_id IN ({placeholders}) ORDER BY rowid""",
                    parameters,
                ).fetchall()
            )
        folder_digests = {str(row[11]) for row in membership_rows}
        folder_rows_by_id: dict[str, tuple[object, ...]] = {}
        for chunk in _sql_chunks(folder_digests):
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""WITH RECURSIVE ancestors(
                        effect_id, effect_kind, state, target_folder_id,
                        reason_code, retryable, path_digest, parent_effect_id
                    ) AS (
                        SELECT effect_id, effect_kind, state, target_folder_id,
                               reason_code, retryable, path_digest, parent_effect_id
                        FROM import_folder_effects WHERE session_id = ?
                          AND path_digest IN ({placeholders})
                        UNION ALL
                        SELECT parent.effect_id, parent.effect_kind, parent.state,
                               parent.target_folder_id, parent.reason_code,
                               parent.retryable, parent.path_digest,
                               parent.parent_effect_id
                        FROM import_folder_effects AS parent
                        JOIN ancestors AS child
                          ON parent.effect_id = child.parent_effect_id
                        WHERE parent.session_id = ?
                    )
                    SELECT effect_id, NULL, NULL, effect_kind, state, NULL,
                           target_folder_id, NULL, NULL, reason_code, retryable,
                           path_digest, parent_effect_id FROM ancestors""",
                (session_id, *chunk, session_id),
            ).fetchall()
            folder_rows_by_id.update((str(row[0]), row) for row in rows)
        for chunk in _sql_chunks(folder_effect_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""SELECT effect_id, NULL, NULL, effect_kind, state, NULL,
                    target_folder_id, NULL, NULL, reason_code, retryable,
                    path_digest, parent_effect_id FROM import_folder_effects
                    WHERE session_id = ? AND effect_id IN ({placeholders})""",
                (session_id, *chunk),
            ).fetchall()
            folder_rows_by_id.update((str(row[0]), row) for row in rows)
        folder_rows = list(folder_rows_by_id.values())
        records = NoteImportReceiptRepository._item_records(item_rows)
        total = sum(item.outcome_count for item in records)
        return ImportSessionSnapshot(
            approval_id=approval_id,
            session_id=session_id,
            plan_digest="0" * 64,
            state=state,
            batch_size=1,
            total=total,
            reason_code=None,
            items=records,
            payload_effects=NoteImportReceiptRepository._effect_records(
                _PAYLOAD_TABLE, payload_rows
            ),
            folder_effects=NoteImportReceiptRepository._effect_records(
                _FOLDER_TABLE, folder_rows
            ),
            membership_effects=NoteImportReceiptRepository._effect_records(
                _MEMBERSHIP_TABLE, membership_rows
            ),
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
        with self.transaction(immediate=True) as connection:
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
                self._validate_completion(connection, session_id, approval_id)
            connection.execute(
                """
                UPDATE import_sessions SET state = ?, reason_code = ?, updated_at = ?
                WHERE approval_id = ?
                """,
                (state.value, validated_reason, _now(), approval_id),
            )
            return self._load_snapshot(connection, approval_id)

    def resume_cancelled(
        self,
        approved: ApprovedNoteImportPlan,
        *,
        batch_size: int,
    ) -> ImportSessionSnapshot:
        """Resume cancelled pending work under the exact original authority."""

        validated_batch_size = _validate_batch_size(batch_size)
        if type(approved) is not ApprovedNoteImportPlan:
            raise TypeError("approved must be an ApprovedNoteImportPlan.")
        plan_digest = approved._private_plan_digest()
        if _DIGEST_PATTERN.fullmatch(plan_digest) is None:
            raise ValueError("The approved plan digest is invalid.")
        with self.transaction(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT plan_digest, batch_size, state FROM import_sessions
                WHERE approval_id = ?
                """,
                (approved.approval_id,),
            ).fetchone()
            if row is None:
                raise KeyError("Import receipt session was not found.")
            if row[:2] != (plan_digest, validated_batch_size):
                raise ImportReceiptConflictError(
                    "The approval is bound to different receipt authority."
                )
            if ImportSessionState(row[2]) is not ImportSessionState.CANCELLED:
                raise ImportReceiptTransitionError(
                    "Only a cancelled receipt session may use cancelled resume."
                )
            connection.execute(
                """
                UPDATE import_sessions
                SET state = ?, reason_code = NULL, updated_at = ?
                WHERE approval_id = ? AND state = ?
                """,
                (
                    ImportSessionState.RUNNING.value,
                    _now(),
                    approved.approval_id,
                    ImportSessionState.CANCELLED.value,
                ),
            )
            return self._load_snapshot(connection, approved.approval_id)

    @staticmethod
    def _validate_completion(
        connection: sqlite3.Connection,
        session_id: str,
        approval_id: str,
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
        snapshot = NoteImportReceiptRepository._load_snapshot(connection, approval_id)
        counts, _retryable = NoteImportReceiptRepository._derive_receipt_counts(
            snapshot
        )
        if (
            counts[ImportItemOutcome.PENDING]
            or counts[ImportItemOutcome.FAILED]
            or sum(
                counts[outcome]
                for outcome in (
                    ImportItemOutcome.IMPORTED,
                    ImportItemOutcome.UPDATED,
                    ImportItemOutcome.SKIPPED,
                )
            )
            != snapshot.total
        ):
            raise ImportReceiptTransitionError(
                "A session can complete only with successful durable note units."
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
            (effect.category, effect.effect_id): effect for effect in snapshot.effects
        }
        return tuple(
            by_key[(transition.category, transition.effect_id)] for transition in copied
        )

    def transition_batch(
        self,
        approval_id: str,
        *,
        item_transitions: Sequence[ItemTransition] = (),
        effect_transitions: Sequence[EffectTransition] = (),
    ) -> ImportBatchResult:
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
        with self.transaction(immediate=True) as connection:
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
            affected_item_ids = self._affected_item_ids(
                connection, session_id, items, effects
            )
            folder_effect_ids = {
                transition.effect_id
                for transition in effects
                if transition.category is ImportEffectCategory.FOLDER
            }
            self._validate_transition_identity_points(connection, session_id, effects)
            snapshot = self._load_dependency_snapshot(
                connection,
                approval_id,
                session_id,
                ImportSessionState.RUNNING,
                affected_item_ids,
                folder_effect_ids,
            )
            self._derive_receipt_counts(snapshot)
            changed_item_ids = {transition.item_id for transition in items}
            changed_effect_keys = {
                (transition.category, transition.effect_id) for transition in effects
            }
            result = ImportBatchResult(
                items=tuple(
                    item for item in snapshot.items if item.item_id in changed_item_ids
                ),
                effects=tuple(
                    effect
                    for effect in (
                        *snapshot.payload_effects,
                        *snapshot.folder_effects,
                        *snapshot.membership_effects,
                    )
                    if (effect.category, effect.effect_id) in changed_effect_keys
                ),
            )
            return result

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
                ImportAction.SKIP: {ImportItemOutcome.SKIPPED},
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
            if action is ImportAction.SKIP and (
                transition.target_note_id is not None
                or transition.observed_version is not None
            ):
                raise ValueError("Skip outcomes cannot bind reconciliation metadata.")
            if action is ImportAction.CREATE_NEW and (
                transition.target_note_id is not None
                or transition.observed_version is not None
            ):
                raise ValueError(
                    "Create item summaries cannot bind reconciliation metadata."
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
            table = _table_for_category(transition.category)
            effect_id = _validate_id(transition.effect_id, field_name="effect_id")
            key = (transition.category, effect_id)
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
                table=table,
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
            if table == _PAYLOAD_TABLE:
                if transition.target_folder_id is not None:
                    raise ValueError("Payload effects cannot bind a folder identifier.")
            elif table == _FOLDER_TABLE:
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
                    table == _PAYLOAD_TABLE
                    and (final_note_id is None or final_version is None)
                ) or (table == _FOLDER_TABLE and final_folder_id is None)
                missing_identity = missing_identity or (
                    table == _MEMBERSHIP_TABLE
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
        table = _table_for_category(transition.category)
        reason_code, retryable = _validate_transition_metadata(
            failed=transition.state is ImportEffectState.FAILED,
            reason_code=transition.reason_code,
            retryable=transition.retryable,
        )
        if table == _PAYLOAD_TABLE:
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
        elif table == _FOLDER_TABLE:
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
        record = self._reset_retryable(
            approval_id,
            table="import_items",
            key_value=item_id,
        )
        if not isinstance(record, ImportItemRecord):
            raise ImportReceiptError("The reset receipt row category is inconsistent.")
        return record

    def reset_retryable_effect(
        self,
        approval_id: str,
        *,
        category: ImportEffectCategory,
        effect_id: str,
    ) -> ImportEffectRecord:
        """Explicitly reset one retryable failed effect to pending."""

        table = _table_for_category(category)
        _validate_id(effect_id, field_name="effect_id")
        record = self._reset_retryable(
            approval_id,
            table=table,
            key_value=effect_id,
        )
        if (
            not isinstance(record, ImportEffectRecord)
            or record.category is not category
        ):
            raise ImportReceiptError("The reset receipt row category is inconsistent.")
        return record

    def annotate_applied_payload_reconciliation_conflict(
        self,
        approval_id: str,
        *,
        effect_id: str,
    ) -> ImportEffectRecord:
        """Record later target divergence without rewriting the applied mutation."""

        _validate_id(approval_id, field_name="approval_id")
        _validate_id(effect_id, field_name="effect_id")
        with self.transaction(immediate=True) as connection:
            session_row = connection.execute(
                "SELECT session_id, state FROM import_sessions WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError("Import receipt session was not found.")
            session_id = session_row[0]
            if ImportSessionState(session_row[1]) is not ImportSessionState.RUNNING:
                raise ImportReceiptTransitionError(
                    "Payload reconciliation annotations require a running session."
                )
            row = connection.execute(
                """
                SELECT state, target_note_id, observed_version, reason_code, retryable
                FROM import_payload_effects
                WHERE session_id = ? AND effect_id = ?
                """,
                (session_id, effect_id),
            ).fetchone()
            if row is None:
                raise KeyError("Import receipt effect was not found.")
            state = ImportEffectState(row[0])
            target_note_id = row[1]
            observed_version = row[2]
            reason_code = row[3]
            retryable = row[4]
            if (
                state is not ImportEffectState.APPLIED
                or _SAFE_ID_PATTERN.fullmatch(target_note_id or "") is None
                or isinstance(observed_version, bool)
                or not isinstance(observed_version, int)
                or observed_version < 1
                or retryable != 0
                or reason_code not in {None, "note_conflict"}
            ):
                raise ImportReceiptTransitionError(
                    "Only an exact applied payload may record reconciliation conflict."
                )
            connection.execute(
                """
                UPDATE import_payload_effects
                SET reason_code = ?, updated_at = ?
                WHERE session_id = ? AND effect_id = ? AND state = ?
                """,
                (
                    "note_conflict",
                    _now(),
                    session_id,
                    effect_id,
                    ImportEffectState.APPLIED.value,
                ),
            )
            record = self._load_effect_record(
                connection,
                table=_PAYLOAD_TABLE,
                session_id=session_id,
                effect_id=effect_id,
            )
            self._derive_receipt_counts(self._load_snapshot(connection, approval_id))
            return record

    def _reset_retryable(
        self,
        approval_id: str,
        *,
        table: str,
        key_value: str,
    ) -> ImportItemRecord | ImportEffectRecord:
        _validate_id(approval_id, field_name="approval_id")
        with self.transaction(immediate=True) as connection:
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
            if table in {_PAYLOAD_TABLE, _MEMBERSHIP_TABLE}:
                parent_outcome = self._select_parent_item_outcome(
                    connection,
                    table=table,
                    session_id=session_id,
                    effect_id=key_value,
                )
                if parent_outcome is not ImportItemOutcome.PENDING:
                    raise ImportReceiptTransitionError(
                        "The parent item must be reset before its retryable effect."
                    )
            if table == _FOLDER_TABLE and self._folder_has_failed_dependents(
                connection,
                session_id=session_id,
                effect_id=key_value,
            ):
                raise ImportReceiptTransitionError(
                    "Every dependent item must be reset before its folder effect."
                )
            self._update_retryable_row(
                connection,
                table=table,
                session_id=session_id,
                key_value=key_value,
            )
            if table == "import_items":
                item_row = connection.execute(
                    """SELECT item_id, source_locator_digest, selected_action,
                        outcome_count, outcome, target_note_id, expected_version,
                        observed_version, reason_code, retryable FROM import_items
                        WHERE session_id = ? AND item_id = ?""",
                    (session_id, key_value),
                ).fetchone()
                if item_row is None:
                    raise KeyError("Import receipt item was not found.")
                record: ImportItemRecord | ImportEffectRecord = self._item_records(
                    (item_row,)
                )[0]
            else:
                record = self._load_effect_record(
                    connection,
                    table=table,
                    session_id=session_id,
                    effect_id=key_value,
                )
            return record

    @staticmethod
    def _folder_has_failed_dependents(
        connection: sqlite3.Connection,
        *,
        session_id: str,
        effect_id: str,
    ) -> bool:
        row = connection.execute(
            """
            WITH RECURSIVE descendants(effect_id, path_digest) AS (
                SELECT effect_id, path_digest FROM import_folder_effects
                WHERE session_id = ? AND effect_id = ?
                UNION ALL
                SELECT child.effect_id, child.path_digest
                FROM import_folder_effects AS child
                JOIN descendants AS parent
                  ON child.parent_effect_id = parent.effect_id
                WHERE child.session_id = ?
            )
            SELECT 1 FROM descendants AS folder
            JOIN import_membership_effects AS membership
              ON membership.session_id = ?
             AND membership.folder_path_digest = folder.path_digest
            JOIN import_items AS item
              ON item.session_id = membership.session_id
             AND item.item_id = membership.item_id
            WHERE item.outcome = 'failed' LIMIT 1
            """,
            (session_id, effect_id, session_id, session_id),
        ).fetchone()
        return row is not None

    @staticmethod
    def _load_effect_record(
        connection: sqlite3.Connection,
        *,
        table: str,
        session_id: str,
        effect_id: str,
    ) -> ImportEffectRecord:
        if table == _PAYLOAD_TABLE:
            row = connection.execute(
                """SELECT effect_id, item_id, payload_index, effect_kind, state,
                    target_note_id, NULL, expected_version, observed_version,
                    reason_code, retryable, NULL, NULL FROM import_payload_effects
                    WHERE session_id = ? AND effect_id = ?""",
                (session_id, effect_id),
            ).fetchone()
        elif table == _FOLDER_TABLE:
            row = connection.execute(
                """SELECT effect_id, NULL, NULL, effect_kind, state, NULL,
                    target_folder_id, NULL, NULL, reason_code, retryable,
                    path_digest, parent_effect_id
                    FROM import_folder_effects WHERE session_id = ? AND effect_id = ?""",
                (session_id, effect_id),
            ).fetchone()
        else:
            row = connection.execute(
                """SELECT effect_id, item_id, payload_index, effect_kind, state,
                    target_note_id, target_folder_id, NULL, NULL, reason_code,
                    retryable, folder_path_digest, NULL FROM import_membership_effects
                    WHERE session_id = ? AND effect_id = ?""",
                (session_id, effect_id),
            ).fetchone()
        if row is None:
            raise KeyError("Import receipt effect was not found.")
        return NoteImportReceiptRepository._effect_records(table, (row,))[0]

    @staticmethod
    def _select_parent_item_outcome(
        connection: sqlite3.Connection,
        *,
        table: str,
        session_id: str,
        effect_id: str,
    ) -> ImportItemOutcome:
        if table == _PAYLOAD_TABLE:
            row = connection.execute(
                """
                SELECT item.outcome FROM import_payload_effects AS effect
                JOIN import_items AS item
                  ON item.session_id = effect.session_id
                 AND item.item_id = effect.item_id
                WHERE effect.session_id = ? AND effect.effect_id = ?
                """,
                (session_id, effect_id),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT item.outcome FROM import_membership_effects AS effect
                JOIN import_items AS item
                  ON item.session_id = effect.session_id
                 AND item.item_id = effect.item_id
                WHERE effect.session_id = ? AND effect.effect_id = ?
                """,
                (session_id, effect_id),
            ).fetchone()
        if row is None:
            raise KeyError("Import receipt parent item was not found.")
        return ImportItemOutcome(row[0])

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
                SET outcome = ?, reason_code = ?, retryable = ?, updated_at = ?,
                    observed_version = NULL
                WHERE session_id = ? AND item_id = ?
                """,
                parameters,
            )
        elif table == _PAYLOAD_TABLE:
            connection.execute(
                """
                UPDATE import_payload_effects
                SET state = ?, reason_code = ?, retryable = ?, updated_at = ?,
                    observed_version = NULL
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
        with self.transaction() as connection:
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
            return self._build_receipt(snapshot, payload_digests)

    def prior_observations_for_plan(
        self,
        plan: NoteImportPlan,
    ) -> tuple[PriorImportObservation, ...]:
        """Return latest exact single-note observations for current plan sources."""

        if type(plan) is not NoteImportPlan:
            raise TypeError("plan must be a NoteImportPlan.")
        try:
            digest_items: dict[str, list[ImportPreviewItem]] = {}
            for item in plan.items:
                digest = _private_source_locator_digest(item)
                digest_items.setdefault(digest, []).append(item)
        except Exception:  # noqa: BLE001 - source locator material is private
            raise ImportReceiptError(
                "Prior import observations could not be matched safely."
            ) from None
        if not digest_items:
            return ()

        latest: dict[
            str,
            tuple[tuple[int, int], str, list[tuple[object, ...]]],
        ] = {}
        with self.transaction() as connection:
            digests = tuple(digest_items)
            chunk_size = _prior_observation_chunk_size(connection)
            for offset in range(0, len(digests), chunk_size):
                chunk = digests[offset : offset + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""
                    SELECT item.source_locator_digest, session.session_id,
                           item.outcome_count, item.outcome,
                           (SELECT COUNT(*) FROM import_payload_effects AS counted
                            WHERE counted.session_id = item.session_id
                              AND counted.item_id = item.item_id) AS payload_count,
                           payload.payload_digest, payload.target_note_id,
                           payload.observed_version, payload.state,
                           payload.reason_code, payload.retryable,
                           session.updated_at, session.rowid, item.rowid
                    FROM import_items AS item
                    JOIN import_sessions AS session
                      ON session.session_id = item.session_id
                    LEFT JOIN import_payload_effects AS payload
                      ON payload.session_id = item.session_id
                     AND payload.item_id = item.item_id
                     AND payload.payload_index = 0
                    WHERE item.source_locator_digest IN ({placeholders})
                      AND session.state = ?
                    """,
                    (*chunk, ImportSessionState.COMPLETED.value),
                ).fetchall()
                for row in rows:
                    digest = str(row[0])
                    session_id = str(row[1])
                    updated_at = row[11]
                    session_rowid = row[12]
                    if type(updated_at) is not int or type(session_rowid) is not int:
                        raise ImportReceiptError(
                            "Prior import observations could not be ordered safely."
                        )
                    ordering = (updated_at, session_rowid)
                    existing = latest.get(digest)
                    if existing is None or ordering > existing[0]:
                        latest[digest] = (ordering, session_id, [tuple(row)])
                    elif ordering == existing[0] and existing[1] == session_id:
                        existing[2].append(tuple(row))

        observations: list[PriorImportObservation] = []
        for digest, items in digest_items.items():
            if len(items) != 1:
                continue
            latest_group = latest.get(digest)
            if latest_group is None or len(latest_group[2]) != 1:
                continue
            row = latest_group[2][0]
            (
                _source_digest,
                _session_id,
                outcome_count,
                outcome,
                payload_count,
                payload_digest,
                target_note_id,
                observed_version,
                payload_state,
                payload_reason_code,
                payload_retryable,
                *_ordering,
            ) = row
            if (
                outcome_count != 1
                or outcome
                not in {
                    ImportItemOutcome.IMPORTED.value,
                    ImportItemOutcome.UPDATED.value,
                }
                or payload_count != 1
                or payload_state != ImportEffectState.APPLIED.value
                or payload_reason_code is not None
                or payload_retryable != 0
                or _DIGEST_PATTERN.fullmatch(payload_digest or "") is None
                or _SAFE_ID_PATTERN.fullmatch(target_note_id or "") is None
                or isinstance(observed_version, bool)
                or not isinstance(observed_version, int)
                or observed_version < 1
            ):
                continue
            item = items[0]
            observations.append(
                _PrivatePriorImportObservation(
                    display_path=item.source.display_path,
                    match_kind=ImportMatchKind.EXACT,
                    note_id=target_note_id,
                    note_version=observed_version,
                    payload_fingerprint=payload_digest,
                )
            )
        return tuple(observations)

    @staticmethod
    def _validate_create_note_identities(snapshot: ImportSessionSnapshot) -> None:
        create_items = tuple(
            item
            for item in snapshot.items
            if item.selected_action is ImportAction.CREATE_NEW
        )
        for item in create_items:
            if (
                item.target_note_id is not None
                or item.expected_version is not None
                or item.observed_version is not None
            ):
                raise ImportReceiptConflictError(
                    "A Create item summary cannot bind reconciliation metadata."
                )
        create_item_ids = {item.item_id for item in create_items}
        payloads: dict[tuple[str, int], ImportEffectRecord] = {}
        target_note_ids: list[str] = []
        for effect in snapshot.payload_effects:
            if effect.item_id not in create_item_ids or effect.payload_index is None:
                continue
            if effect.expected_version is not None:
                raise ImportReceiptConflictError(
                    "A Create payload cannot bind expected version authority."
                )
            key = (effect.item_id, effect.payload_index)
            payloads[key] = effect
            if effect.target_note_id is not None:
                target_note_ids.append(effect.target_note_id)
        if len(target_note_ids) != len(set(target_note_ids)):
            raise ImportReceiptConflictError(
                "Create payload note identities must be unique per note unit."
            )
        update_target_note_ids = {
            item.target_note_id
            for item in snapshot.items
            if item.selected_action is ImportAction.UPDATE_EXISTING
            and item.target_note_id is not None
        }
        if any(
            target_note_id in update_target_note_ids
            for target_note_id in target_note_ids
        ):
            raise ImportReceiptConflictError(
                "Create payload note identities must be unique per note unit."
            )
        for effect in snapshot.membership_effects:
            if (
                effect.item_id not in create_item_ids
                or effect.payload_index is None
                or effect.target_note_id is None
            ):
                continue
            payload = payloads.get((effect.item_id, effect.payload_index))
            if payload is None or payload.target_note_id is None:
                raise ImportReceiptConflictError(
                    "A Create membership cannot bind before its payload target."
                )
            if effect.target_note_id != payload.target_note_id:
                raise ImportReceiptConflictError(
                    "A Create membership note identity conflicts with its payload."
                )
            if (
                effect.state is ImportEffectState.APPLIED
                and payload.state is not ImportEffectState.APPLIED
            ):
                raise ImportReceiptConflictError(
                    "An applied Create membership requires an applied payload."
                )

    @staticmethod
    def _validate_update_reconciliation(snapshot: ImportSessionSnapshot) -> None:
        update_items = {
            item.item_id: item
            for item in snapshot.items
            if item.selected_action is ImportAction.UPDATE_EXISTING
        }
        payloads_by_item: dict[str, list[ImportEffectRecord]] = {
            item_id: [] for item_id in update_items
        }
        memberships_by_item: dict[str, list[ImportEffectRecord]] = {
            item_id: [] for item_id in update_items
        }
        for effect in snapshot.payload_effects:
            if effect.item_id in payloads_by_item:
                payloads_by_item[effect.item_id].append(effect)
        for effect in snapshot.membership_effects:
            if effect.item_id not in memberships_by_item:
                continue
            item = update_items[effect.item_id]
            if (
                effect.target_note_id is not None
                and effect.target_note_id != item.target_note_id
            ):
                raise ImportReceiptConflictError(
                    "An Update membership conflicts with its approved target."
                )
            memberships_by_item[effect.item_id].append(effect)

        for item_id, item in update_items.items():
            payloads = payloads_by_item[item_id]
            for effect in payloads:
                if effect.expected_version != item.expected_version:
                    raise ImportReceiptConflictError(
                        "Update payload expected version authority conflicts with its item."
                    )
                if (
                    effect.target_note_id is not None
                    and effect.target_note_id != item.target_note_id
                ):
                    raise ImportReceiptConflictError(
                        "An Update payload conflicts with its approved target."
                    )
                if effect.state is ImportEffectState.APPLIED:
                    if effect.expected_version is None:
                        raise ImportReceiptConflictError(
                            "An applied replace payload requires expected version authority."
                        )
                    if effect.observed_version != effect.expected_version + 1:
                        raise ImportReceiptConflictError(
                            "An applied replace payload must observe exactly one version advance."
                        )
            if item.outcome is not ImportItemOutcome.UPDATED:
                continue
            if item.observed_version is None:
                raise ImportReceiptTransitionError(
                    "A terminal Update requires a final item observation."
                )
            required_effects = (*payloads, *memberships_by_item[item_id])
            if not required_effects or any(
                effect.state is not ImportEffectState.APPLIED
                for effect in required_effects
            ):
                raise ImportReceiptTransitionError(
                    "A terminal Update requires every approved effect to be applied."
                )
            if item.expected_version is None:
                raise ImportReceiptConflictError(
                    "A terminal Update requires expected version authority."
                )
            if payloads:
                successful_version = item.expected_version + 1
                if item.observed_version != successful_version or any(
                    effect.observed_version != successful_version for effect in payloads
                ):
                    raise ImportReceiptConflictError(
                        "A replace-content Update must observe exactly one version advance."
                    )
            elif item.observed_version != item.expected_version:
                raise ImportReceiptConflictError(
                    "A membership-only Update observation must exactly match its expected version."
                )

    @staticmethod
    def _validate_folder_identities(snapshot: ImportSessionSnapshot) -> None:
        folders_by_path: dict[str, ImportEffectRecord] = {}
        folders_by_id: dict[str, ImportEffectRecord] = {}
        target_paths: dict[str, str] = {}
        for effect in snapshot.folder_effects:
            if effect.folder_path_digest is None:
                raise ImportReceiptError(
                    "A durable folder effect has inconsistent path authority."
                )
            if effect.folder_path_digest in folders_by_path:
                raise ImportReceiptError(
                    "Durable folder effects have duplicate path authority."
                )
            folders_by_path[effect.folder_path_digest] = effect
            if effect.effect_id in folders_by_id:
                raise ImportReceiptError(
                    "Durable folder effects have duplicate effect authority."
                )
            folders_by_id[effect.effect_id] = effect
            if effect.target_folder_id is None:
                continue
            previous_path = target_paths.setdefault(
                effect.target_folder_id,
                effect.folder_path_digest,
            )
            if previous_path != effect.folder_path_digest:
                raise ImportReceiptConflictError(
                    "Distinct approved folder paths require distinct folder identities."
                )

        for effect in snapshot.folder_effects:
            visited: set[str] = set()
            current = effect
            while current.parent_effect_id is not None:
                if current.effect_id in visited:
                    raise ImportReceiptError(
                        "Durable folder effects have cyclic parent authority."
                    )
                visited.add(current.effect_id)
                parent = folders_by_id.get(current.parent_effect_id)
                if parent is None:
                    raise ImportReceiptError(
                        "A durable folder effect has missing parent authority."
                    )
                current = parent

        for effect in snapshot.membership_effects:
            if (
                effect.folder_path_digest is None
                or effect.folder_path_digest not in folders_by_path
            ):
                raise ImportReceiptError(
                    "A durable membership has inconsistent folder-path authority."
                )
            folder = folders_by_path[effect.folder_path_digest]
            required_folders = NoteImportReceiptRepository._required_folder_chain(
                effect, folders_by_path, folders_by_id
            )
            if effect.state is ImportEffectState.APPLIED and any(
                required.state is not ImportEffectState.APPLIED
                for required in required_folders
            ):
                raise ImportReceiptConflictError(
                    "An applied membership requires its applied folder authority."
                )
            if effect.target_folder_id is not None:
                if folder.target_folder_id is None:
                    raise ImportReceiptConflictError(
                        "A membership folder identity requires a known folder binding."
                    )
                if effect.target_folder_id != folder.target_folder_id:
                    raise ImportReceiptConflictError(
                        "A membership folder identity conflicts with its approved path."
                    )
            elif effect.state is ImportEffectState.APPLIED:
                raise ImportReceiptConflictError(
                    "An applied membership requires its applied folder authority."
                )

    @staticmethod
    def _required_folder_chain(
        membership: ImportEffectRecord,
        folders_by_path: Mapping[str | None, ImportEffectRecord],
        folders_by_id: Mapping[str, ImportEffectRecord],
    ) -> tuple[ImportEffectRecord, ...]:
        folder = folders_by_path.get(membership.folder_path_digest)
        if folder is None:
            raise ImportReceiptError(
                "A durable membership has inconsistent folder-path authority."
            )
        chain: list[ImportEffectRecord] = []
        visited: set[str] = set()
        current = folder
        while True:
            if current.effect_id in visited:
                raise ImportReceiptError(
                    "Durable folder effects have cyclic parent authority."
                )
            visited.add(current.effect_id)
            chain.append(current)
            if current.parent_effect_id is None:
                return tuple(chain)
            parent = folders_by_id.get(current.parent_effect_id)
            if parent is None:
                raise ImportReceiptError(
                    "A durable folder effect has missing parent authority."
                )
            current = parent

    @staticmethod
    def _validate_reconciliation_authority(snapshot: ImportSessionSnapshot) -> None:
        NoteImportReceiptRepository._validate_create_note_identities(snapshot)
        NoteImportReceiptRepository._validate_update_reconciliation(snapshot)
        NoteImportReceiptRepository._validate_folder_identities(snapshot)

    @staticmethod
    def _classify_required_effects(
        required_effects: tuple[ImportEffectRecord, ...],
        *,
        success_outcome: ImportItemOutcome,
        item: ImportItemRecord,
        include_item_retryable: bool,
    ) -> tuple[ImportItemOutcome, bool]:
        malformed_applied_metadata = any(
            effect.state is ImportEffectState.APPLIED
            and (
                effect.retryable
                or (
                    effect.reason_code is not None
                    and not (
                        effect.category is ImportEffectCategory.PAYLOAD
                        and effect.reason_code == "note_conflict"
                    )
                )
            )
            for effect in required_effects
        )
        if malformed_applied_metadata:
            raise ImportReceiptError(
                "Applied effect reconciliation metadata is inconsistent."
            )
        conflict_payloads = tuple(
            effect
            for effect in required_effects
            if effect.category is ImportEffectCategory.PAYLOAD
            and effect.state is ImportEffectState.APPLIED
            and effect.reason_code == "note_conflict"
        )
        if conflict_payloads:
            if len(conflict_payloads) != 1:
                raise ImportReceiptError(
                    "A note unit has inconsistent reconciliation conflicts."
                )
            if any(
                effect.state is ImportEffectState.PENDING for effect in required_effects
            ):
                return ImportItemOutcome.PENDING, False
            return ImportItemOutcome.FAILED, False
        failed_effects = tuple(
            effect
            for effect in required_effects
            if effect.state is ImportEffectState.FAILED
        )
        if failed_effects:
            retryable = any(effect.retryable for effect in failed_effects) or (
                include_item_retryable
                and item.outcome is ImportItemOutcome.FAILED
                and item.retryable
            )
            return ImportItemOutcome.FAILED, retryable
        if required_effects and all(
            effect.state is ImportEffectState.APPLIED for effect in required_effects
        ):
            return success_outcome, False
        return ImportItemOutcome.PENDING, False

    @staticmethod
    def _validate_terminal_item_summary(
        item: ImportItemRecord,
        unit_outcomes: tuple[ImportItemOutcome, ...],
        *,
        success_outcome: ImportItemOutcome,
        unit_retryable: bool,
    ) -> None:
        if item.outcome is ImportItemOutcome.PENDING:
            return
        consistent = False
        if item.outcome is success_outcome:
            consistent = all(outcome is success_outcome for outcome in unit_outcomes)
        elif item.outcome is ImportItemOutcome.FAILED:
            consistent = (
                ImportItemOutcome.FAILED in unit_outcomes
                and ImportItemOutcome.PENDING not in unit_outcomes
                and item.retryable is unit_retryable
            )
        if not consistent:
            raise ImportReceiptError(
                "The terminal item summary is inconsistent with durable effects."
            )

    @staticmethod
    def _derive_receipt_counts(
        snapshot: ImportSessionSnapshot,
    ) -> tuple[dict[ImportItemOutcome, int], int]:
        NoteImportReceiptRepository._validate_reconciliation_authority(snapshot)
        counts = {outcome: 0 for outcome in ImportItemOutcome}
        retryable_count = 0
        items_by_id = {item.item_id: item for item in snapshot.items}
        payloads_by_item: dict[str, list[ImportEffectRecord]] = {
            item_id: [] for item_id in items_by_id
        }
        memberships_by_item: dict[str, list[ImportEffectRecord]] = {
            item_id: [] for item_id in items_by_id
        }
        folders_by_path = {
            effect.folder_path_digest: effect for effect in snapshot.folder_effects
        }
        folders_by_id = {effect.effect_id: effect for effect in snapshot.folder_effects}
        if None in folders_by_path or len(folders_by_path) != len(
            snapshot.folder_effects
        ):
            raise ImportReceiptError(
                "Durable folder effects have inconsistent path authority."
            )
        for effect in snapshot.payload_effects:
            if effect.item_id not in payloads_by_item:
                raise ImportReceiptError(
                    "A durable payload effect has inconsistent item authority."
                )
            payloads_by_item[effect.item_id].append(effect)
        for effect in snapshot.membership_effects:
            if effect.item_id not in memberships_by_item:
                raise ImportReceiptError(
                    "A durable membership effect has inconsistent item authority."
                )
            memberships_by_item[effect.item_id].append(effect)

        for item in snapshot.items:
            payloads = payloads_by_item[item.item_id]
            memberships = memberships_by_item[item.item_id]
            unit_results: list[tuple[ImportItemOutcome, bool]] = []
            success_outcome: ImportItemOutcome
            if item.selected_action is ImportAction.SKIP:
                if item.outcome_count != 1 or payloads or memberships:
                    raise ImportReceiptError(
                        "A durable Skip item has inconsistent mutation effects."
                    )
                success_outcome = ImportItemOutcome.SKIPPED
                outcome = (
                    ImportItemOutcome.SKIPPED
                    if item.outcome is ImportItemOutcome.SKIPPED
                    else ImportItemOutcome.PENDING
                )
                unit_results.append((outcome, False))
            elif item.selected_action is ImportAction.CREATE_NEW:
                success_outcome = ImportItemOutcome.IMPORTED
                payload_by_index = {effect.payload_index: effect for effect in payloads}
                expected_indexes = set(range(item.outcome_count))
                if (
                    len(payload_by_index) != len(payloads)
                    or set(payload_by_index) != expected_indexes
                    or any(effect.effect_kind != "create_note" for effect in payloads)
                ):
                    raise ImportReceiptError(
                        "A durable Create item has inconsistent payload effects."
                    )
                memberships_by_index: dict[int, list[ImportEffectRecord]] = {
                    index: [] for index in expected_indexes
                }
                for effect in memberships:
                    if effect.payload_index not in memberships_by_index:
                        raise ImportReceiptError(
                            "A durable membership effect has an inconsistent payload."
                        )
                    memberships_by_index[effect.payload_index].append(effect)
                for payload_index in range(item.outcome_count):
                    payload = payload_by_index[payload_index]
                    unit_memberships = memberships_by_index[payload_index]
                    required_folders = tuple(
                        folder
                        for effect in unit_memberships
                        for folder in NoteImportReceiptRepository._required_folder_chain(
                            effect, folders_by_path, folders_by_id
                        )
                    )
                    required = (
                        payload,
                        *unit_memberships,
                        *required_folders,
                    )
                    unit_results.append(
                        NoteImportReceiptRepository._classify_required_effects(
                            required,
                            success_outcome=success_outcome,
                            item=item,
                            include_item_retryable=False,
                        )
                    )
            else:
                success_outcome = ImportItemOutcome.UPDATED
                if (
                    item.outcome_count != 1
                    or len(payloads) > 1
                    or any(
                        effect.payload_index != 0
                        or effect.effect_kind != "replace_content"
                        for effect in payloads
                    )
                    or any(effect.payload_index != 0 for effect in memberships)
                ):
                    raise ImportReceiptError(
                        "A durable Update item has inconsistent mutation effects."
                    )
                required_folders = tuple(
                    folder
                    for effect in memberships
                    for folder in NoteImportReceiptRepository._required_folder_chain(
                        effect, folders_by_path, folders_by_id
                    )
                )
                required = (*payloads, *memberships, *required_folders)
                if required:
                    unit_results.append(
                        NoteImportReceiptRepository._classify_required_effects(
                            required,
                            success_outcome=success_outcome,
                            item=item,
                            include_item_retryable=True,
                        )
                    )
                elif item.outcome is ImportItemOutcome.UPDATED:
                    unit_results.append((success_outcome, False))
                elif item.outcome is ImportItemOutcome.FAILED:
                    unit_results.append((ImportItemOutcome.FAILED, item.retryable))
                else:
                    unit_results.append((ImportItemOutcome.PENDING, False))

            unit_outcomes = tuple(outcome for outcome, _retryable in unit_results)
            NoteImportReceiptRepository._validate_terminal_item_summary(
                item,
                unit_outcomes,
                success_outcome=success_outcome,
                unit_retryable=any(
                    retryable
                    for outcome, retryable in unit_results
                    if outcome is ImportItemOutcome.FAILED
                ),
            )
            for outcome, retryable in unit_results:
                counts[outcome] += 1
                retryable_count += int(
                    outcome is ImportItemOutcome.FAILED and retryable
                )

        if sum(item.outcome_count for item in snapshot.items) != snapshot.total:
            raise ImportReceiptError(
                "The durable session total is inconsistent with approved work."
            )
        return counts, retryable_count

    @staticmethod
    def _build_receipt(
        snapshot: ImportSessionSnapshot,
        payload_digests: tuple[str, ...],
    ) -> ImportExecutionReceipt:
        counts, retryable_count = NoteImportReceiptRepository._derive_receipt_counts(
            snapshot
        )
        completed = snapshot.total - counts[ImportItemOutcome.PENDING]
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
        if reason_code is None:
            reason_code = next(
                (
                    effect.reason_code
                    for effect in effects
                    if effect.state is ImportEffectState.FAILED
                    and effect.reason_code is not None
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
            retryable=retryable_count,
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
                            if item.selected_action is ImportAction.UPDATE_EXISTING
                            and item.target_note_id is not None
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

        with self.transaction() as connection:
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
            return ReceiptSchemaSnapshot(
                user_version=user_version,
                tables=tables,
                columns=MappingProxyType(columns),
            )


__all__ = [
    "EFFECT_STATE_TRANSITIONS",
    "ITEM_OUTCOME_TRANSITIONS",
    "SESSION_STATE_TRANSITIONS",
    "EffectTransition",
    "ImportBatchResult",
    "ImportEffectCategory",
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
