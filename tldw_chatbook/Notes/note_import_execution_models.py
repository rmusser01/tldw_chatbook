"""Private approval boundary and redacted projections for Notes import execution.

The approved plan is intentionally opaque in ordinary representations. Execution
receipts may retain private reconciliation material, but their supported public
projection contains only bounded state and outcome counts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from uuid import UUID, uuid4

from tldw_chatbook.Notes.note_import_plan_models import (
    MAX_IMPORT_ENTRIES,
    ImportAction,
    ImportPreviewItem,
    NoteImportPlan,
    ParsedNotePayload,
    RootCollisionState,
)

MAX_IMPORT_REASON_CODE_LENGTH = 64
"""Absolute ceiling for a public execution reason code."""

MAX_PRIVATE_IMPORT_COLLECTION_ITEMS = MAX_IMPORT_ENTRIES
"""Absolute item ceiling for each private execution receipt collection."""

MAX_RECEIPT_LEDGER_ROWS = MAX_IMPORT_ENTRIES
"""Absolute row ceiling for one durable import receipt ledger session."""

_MAX_PRIVATE_IMPORT_ID_LENGTH = 256
_MAX_PRIVATE_IMPORT_ERROR_LENGTH = 4_096

_SAFE_REASON_CODE = re.compile(
    rf"[a-z][a-z0-9_]{{0,{MAX_IMPORT_REASON_CODE_LENGTH - 1}}}\Z"
)
_SAFE_PRIVATE_ID = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9_.:-]{{0,{_MAX_PRIVATE_IMPORT_ID_LENGTH - 1}}}\Z"
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ImportApprovalError(ValueError):
    """Raised when a preview plan cannot cross the execution boundary safely."""


class ImportSessionState(str, Enum):
    """Durable lifecycle state of one import execution session."""

    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NEEDS_ATTENTION = "needs_attention"


class ImportItemOutcome(str, Enum):
    """Terminal or pending result for one approved preview item."""

    PENDING = "pending"
    IMPORTED = "imported"
    UPDATED = "updated"
    SKIPPED = "skipped"
    FAILED = "failed"


class ImportEffectState(str, Enum):
    """Durable state of one independently replayable import effect."""

    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"


def _validate_uuid_text(value: object, *, approval_boundary: bool) -> str:
    """Return canonical UUID text without including rejected input in errors."""
    error_type = ImportApprovalError if approval_boundary else ValueError
    if type(value) is not str:
        raise error_type("approval_id must be canonical UUID text.")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError):
        parsed = None
    if parsed is None or str(parsed) != value:
        raise error_type("approval_id must be canonical UUID text.")
    return value


def _canonical_json_digest(value: object) -> str:
    """Hash one internal canonical JSON value without returning its encoding."""
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _private_payload_fingerprint(payloads: tuple[ParsedNotePayload, ...]) -> str:
    """Return private matching material for an immutable payload collection."""
    return _canonical_json_digest(
        {
            "payloads": [
                {
                    "content": payload.content,
                    "keywords": list(payload.keywords),
                    "template_name": payload.template_name,
                    "title": payload.title,
                    "type": "parsed_note_payload",
                }
                for payload in payloads
            ],
            "type": "tldw_note_import_payload_set",
            "version": 1,
        }
    )


def _private_source_locator_digest(item: ImportPreviewItem) -> str:
    """Return a private digest for one preview item's complete source locator."""
    source = item.source
    return _canonical_json_digest(
        {
            "display_path": source.display_path,
            "kind": source.kind.value,
            "source_path": str(source.source_path),
            "type": "tldw_note_import_source_locator",
            "version": 1,
        }
    )


def _private_plan_digest(plan: NoteImportPlan) -> str:
    """Bind every authority-bearing plan field into one private digest."""
    collision = plan.root_collision
    canonical_plan = {
        "bounds": {
            "max_depth": plan.bounds.max_depth,
            "max_entries": plan.bounds.max_entries,
            "max_file_bytes": plan.bounds.max_file_bytes,
            "max_files": plan.bounds.max_files,
            "max_keywords_per_note": plan.bounds.max_keywords_per_note,
            "max_notes_per_file": plan.bounds.max_notes_per_file,
            "max_reason_length": plan.bounds.max_reason_length,
            "max_total_bytes": plan.bounds.max_total_bytes,
        },
        "items": [
            {
                "effects": {
                    "add_membership": item.add_membership,
                    "replace_content": item.replace_content,
                },
                "item_id": item.item_id,
                "match": (
                    {
                        "kind": item.match.kind.value,
                        "note_id": item.match.note_id,
                        "note_version": item.match.note_version,
                    }
                    if item.match is not None
                    else None
                ),
                "memberships": [
                    {
                        "folder_segments": list(membership.folder_segments),
                        "payload_index": membership.payload_index,
                    }
                    for membership in item.memberships
                ],
                "payload_fingerprint": _private_payload_fingerprint(item.payloads),
                "selected_action": item.selected_action.value,
                "source_locator_digest": _private_source_locator_digest(item),
            }
            for item in plan.items
        ],
        "proposed_folder_paths": [
            list(folder_path) for folder_path in plan.proposed_folder_paths
        ],
        "root_collision": (
            {
                "choice": collision.choice.value if collision.choice else None,
                "collides": collision.collides,
                "proposed_label": collision.proposed_label,
                "resolved_label": collision.resolved_label,
            }
            if collision is not None
            else None
        ),
        "type": "tldw_approved_note_import_plan",
        "version": 1,
    }
    return _canonical_json_digest(canonical_plan)


@dataclass(frozen=True, slots=True, repr=False, init=False)
class ApprovedNoteImportPlan:
    """Opaque immutable authority to execute one exact import plan."""

    approval_id: str
    plan: NoteImportPlan
    __plan_digest: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ImportApprovalError(
            "Approved plans must be created by approve_note_import_plan()."
        )

    def _private_plan_digest(self) -> str:
        """Return private receipt-binding material for the executor."""
        return self.__plan_digest

    def __repr__(self) -> str:
        """Return an opaque representation that cannot disclose plan content."""
        return "ApprovedNoteImportPlan(<private>)"


def _validate_note_import_plan_for_approval(plan: object) -> NoteImportPlan:
    """Return one fully resolved real plan without echoing rejected contents."""
    if type(plan) is not NoteImportPlan:
        raise ImportApprovalError("plan must be a NoteImportPlan.")
    collision = plan.root_collision
    if collision is not None and type(collision) is not RootCollisionState:
        raise ImportApprovalError("plan must contain a validated root collision state.")
    if collision is not None and collision.collides and collision.choice is None:
        raise ImportApprovalError(
            "Every colliding import root must be explicitly resolved before approval."
        )
    if _receipt_ledger_row_count(plan) > MAX_RECEIPT_LEDGER_ROWS:
        raise ImportApprovalError(
            "The import plan exceeds the durable receipt ledger ceiling."
        )
    return plan


def _receipt_ledger_row_count(plan: NoteImportPlan) -> int:
    """Return the exact durable row count required to seed one plan."""

    required_folder_paths: set[tuple[str, ...]] = set()
    payload_effect_count = 0
    membership_effect_count = 0
    for item in plan.items:
        if item.selected_action is ImportAction.CREATE_NEW or (
            item.selected_action is ImportAction.UPDATE_EXISTING
            and item.replace_content
        ):
            payload_effect_count += len(item.payloads)
        if item.selected_action is ImportAction.SKIP or not item.add_membership:
            continue
        membership_effect_count += len(item.memberships)
        for membership in item.memberships:
            path = tuple(membership.folder_segments)
            required_folder_paths.update(
                path[:depth] for depth in range(1, len(path) + 1)
            )
    return (
        1
        + len(plan.items)
        + payload_effect_count
        + len(required_folder_paths)
        + membership_effect_count
    )


def _create_approved_note_import_plan(
    plan: NoteImportPlan,
    approval_id: str,
) -> ApprovedNoteImportPlan:
    """Create an approved plan only after repeating the full approval checks."""
    validated_plan = _validate_note_import_plan_for_approval(plan)
    validated_approval_id = _validate_uuid_text(
        approval_id,
        approval_boundary=True,
    )
    try:
        plan_digest = _private_plan_digest(validated_plan)
        approved = object.__new__(ApprovedNoteImportPlan)
        object.__setattr__(approved, "approval_id", validated_approval_id)
        object.__setattr__(approved, "plan", validated_plan)
        object.__setattr__(
            approved,
            "_ApprovedNoteImportPlan__plan_digest",
            plan_digest,
        )
        return approved
    except Exception:  # noqa: BLE001 - private canonicalization boundary
        sanitized_error = ImportApprovalError(
            "The import plan could not be validated safely for approval."
        )
    raise sanitized_error from None


def approve_note_import_plan(
    plan: NoteImportPlan,
    *,
    approval_id: str | None = None,
) -> ApprovedNoteImportPlan:
    """Cross the explicit approval boundary for one fully resolved plan.

    Args:
        plan: Immutable planner result containing the exact selected effects.
        approval_id: Optional canonical UUID text supplied by the caller.

    Returns:
        An opaque approved plan bound to its private canonical digest.

    Raises:
        ImportApprovalError: If the plan, approval identifier, or root collision
            is not safe to execute.
    """
    resolved_approval_id = str(uuid4()) if approval_id is None else approval_id
    return _create_approved_note_import_plan(
        plan,
        resolved_approval_id,
    )


def _validate_reason_code(reason_code: object) -> str | None:
    """Return one bounded public machine token without echoing rejected input."""
    if reason_code is None:
        return None
    if type(reason_code) is not str:
        raise TypeError("reason_code must be text when provided.")
    if not _SAFE_REASON_CODE.fullmatch(reason_code):
        raise ValueError("reason_code must be a bounded lowercase ASCII machine token.")
    return reason_code


def _validate_execution_counts(
    *,
    state: object,
    total: object,
    completed: object,
    imported: object,
    updated: object,
    skipped: object,
    failed: object,
    retryable: object,
    reason_code: object,
) -> None:
    """Validate one public execution projection at its construction boundary."""
    if type(state) is not ImportSessionState:
        raise TypeError("state must be an ImportSessionState.")
    counts = {
        "total": total,
        "completed": completed,
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "retryable": retryable,
    }
    for field_name, value in counts.items():
        if type(value) is not int:
            raise TypeError(f"{field_name} must be an integer.")
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative.")
    if completed != imported + updated + skipped + failed:
        raise ValueError("completed must equal all terminal outcome counts.")
    if completed > total:
        raise ValueError("completed cannot exceed total.")
    if retryable > failed:
        raise ValueError("retryable cannot exceed failed.")
    if state is ImportSessionState.PENDING and completed != 0:
        raise ValueError("Pending sessions cannot report completed outcomes.")
    if state is ImportSessionState.COMPLETED and (
        completed != total or failed != 0 or retryable != 0
    ):
        raise ValueError("Completed sessions must finish every item without failures.")
    _validate_reason_code(reason_code)


def _copy_private_collection(
    values: object,
    *,
    validator: Callable[[str], bool],
) -> tuple[str, ...]:
    """Copy and validate private values without disclosing them in errors."""
    if issubclass(type(values), (str, bytes)):
        raise TypeError("Private receipt data must be a collection.")
    copied: tuple[object, ...] | None
    try:
        copied = tuple(
            islice(
                values,  # type: ignore[arg-type]
                MAX_PRIVATE_IMPORT_COLLECTION_ITEMS + 1,
            )
        )
    except Exception:  # noqa: BLE001 - validation boundary must redact iterator errors
        copied = None
    if copied is None:
        raise ValueError("The private collection could not be read safely.")
    if len(copied) > MAX_PRIVATE_IMPORT_COLLECTION_ITEMS:
        raise ValueError("Private receipt data exceeds its safety ceiling.")
    try:
        valid = all(type(value) is str and validator(value) for value in copied)
    except Exception:  # noqa: BLE001 - validation boundary must redact validator errors
        valid = False
    if not valid:
        raise ValueError("Private receipt data contains an invalid value.")
    return copied  # type: ignore[return-value]


def _valid_private_id(value: str) -> bool:
    return _SAFE_PRIVATE_ID.fullmatch(value) is not None


def _valid_private_digest(value: str) -> bool:
    return _LOWER_SHA256.fullmatch(value) is not None


def _valid_private_error(value: str) -> bool:
    return (
        bool(value)
        and len(value) <= _MAX_PRIVATE_IMPORT_ERROR_LENGTH
        and "\x00" not in value
    )


@dataclass(frozen=True, slots=True)
class ImportExecutionProgress:
    """Immutable public progress snapshot emitted during execution."""

    state: ImportSessionState
    total: int
    completed: int
    imported: int
    updated: int
    skipped: int
    failed: int
    retryable: int
    reason_code: str | None = None

    def __post_init__(self) -> None:
        _validate_execution_counts(
            **{
                field_name: getattr(self, field_name)
                for field_name in (
                    "state",
                    "total",
                    "completed",
                    "imported",
                    "updated",
                    "skipped",
                    "failed",
                    "retryable",
                    "reason_code",
                )
            }
        )


@dataclass(frozen=True, slots=True)
class ImportExecutionDiagnostic:
    """Only supported redacted serialization of an execution receipt."""

    state: ImportSessionState
    total: int
    completed: int
    imported: int
    updated: int
    skipped: int
    failed: int
    retryable: int
    reason_code: str | None = None

    def __post_init__(self) -> None:
        _validate_execution_counts(
            **{
                field_name: getattr(self, field_name)
                for field_name in (
                    "state",
                    "total",
                    "completed",
                    "imported",
                    "updated",
                    "skipped",
                    "failed",
                    "retryable",
                    "reason_code",
                )
            }
        )


@dataclass(frozen=True, slots=True)
class ImportExecutionReceipt:
    """Immutable result with repr-hidden private reconciliation material."""

    approval_id: str = field(repr=False)
    state: ImportSessionState
    total: int
    completed: int
    imported: int
    updated: int
    skipped: int
    failed: int
    retryable: int
    reason_code: str | None = None
    _note_ids: tuple[str, ...] = field(default=(), repr=False)
    _folder_ids: tuple[str, ...] = field(default=(), repr=False)
    _source_locator_digests: tuple[str, ...] = field(default=(), repr=False)
    _payload_fingerprints: tuple[str, ...] = field(default=(), repr=False)
    _raw_errors: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        _validate_uuid_text(self.approval_id, approval_boundary=False)
        _validate_execution_counts(
            **{
                field_name: getattr(self, field_name)
                for field_name in (
                    "state",
                    "total",
                    "completed",
                    "imported",
                    "updated",
                    "skipped",
                    "failed",
                    "retryable",
                    "reason_code",
                )
            }
        )
        private_fields = (
            ("_note_ids", _valid_private_id),
            ("_folder_ids", _valid_private_id),
            ("_source_locator_digests", _valid_private_digest),
            ("_payload_fingerprints", _valid_private_digest),
            ("_raw_errors", _valid_private_error),
        )
        for field_name, validator in private_fields:
            object.__setattr__(
                self,
                field_name,
                _copy_private_collection(
                    getattr(self, field_name),
                    validator=validator,
                ),
            )

    def to_diagnostic(self) -> ImportExecutionDiagnostic:
        """Return a count-only projection safe for logs and serialization."""
        return ImportExecutionDiagnostic(
            state=self.state,
            total=self.total,
            completed=self.completed,
            imported=self.imported,
            updated=self.updated,
            skipped=self.skipped,
            failed=self.failed,
            retryable=self.retryable,
            reason_code=self.reason_code,
        )
