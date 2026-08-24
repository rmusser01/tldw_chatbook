"""Pure, privacy-bounded contracts for reviewed Notes sync conflicts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import unified_diff
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING

from tldw_chatbook.Notes.notes_sync_models import (
    normalize_notes_sync_relative_path,
    validate_notes_sync_digest,
    validate_notes_sync_opaque_id,
    validate_notes_sync_reason_code,
)
from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationPlan

if TYPE_CHECKING:
    from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutionResult


COMPARISON_MAX_INPUT_CHARS = 200_000
COMPARISON_MAX_INPUT_LINES = 10_000
COMPARISON_MAX_OUTPUT_CHARS = 120_000
COMPARISON_MAX_OUTPUT_LINES = 2_000

ELIGIBLE_CONFLICT_REASONS = frozenset({"both_sides_changed", "out_of_direction_change"})

_DISPLAY_LABEL_MAX_CHARS = 160
_OUTPUT_ELISION_MARKER = "… comparison output elided at the bounded display limit."


class NotesSyncConflictChoice(StrEnum):
    """One explicitly reviewed conflict choice."""

    KEEP_FILE = "keep_file"
    KEEP_NOTE = "keep_note"
    KEEP_BOTH = "keep_both"
    SKIP = "skip"


def eligible_conflict_reason(reason_code: str, *, managed: bool) -> bool:
    """Return whether one unmanaged reason supports inline resolution.

    Args:
        reason_code: Stable reconciliation reason code to evaluate.
        managed: Whether the binding is managed by a lasting-sync root.

    Returns:
        ``True`` when the reason is eligible and the binding is unmanaged.

    Raises:
        TypeError: If ``managed`` is not a boolean.
    """

    if type(managed) is not bool:
        raise TypeError("managed must be a boolean")
    return reason_code in ELIGIBLE_CONFLICT_REASONS and not managed


@dataclass(frozen=True, slots=True, repr=False)
class ConflictSelection:
    """One typed choice staged for an opaque binding."""

    binding_id: str
    choice: NotesSyncConflictChoice

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.binding_id, field_name="binding_id")
        if type(self.choice) is not NotesSyncConflictChoice:
            raise TypeError("choice must be a NotesSyncConflictChoice")

    def __repr__(self) -> str:
        return "ConflictSelection(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class ConflictComparison:
    """Bounded Note-to-File comparison without private authority."""

    binding_id: str
    note_title: str
    relative_path: str
    note_version: int
    note_updated_at: str | None
    file_modified_ns: int
    note_character_count: int
    note_line_count: int
    file_character_count: int
    file_line_count: int
    diff: str
    input_elided: bool
    output_elided: bool

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.binding_id, field_name="binding_id")
        if (
            type(self.note_title) is not str
            or not self.note_title
            or len(self.note_title) > _DISPLAY_LABEL_MAX_CHARS
            or "\n" in self.note_title
        ):
            raise ValueError("note_title must be a bounded single-line label")
        object.__setattr__(
            self,
            "relative_path",
            normalize_notes_sync_relative_path(self.relative_path),
        )
        if type(self.note_version) is not int or self.note_version < 0:
            raise ValueError("note_version must be a non-negative integer")
        _validate_timestamp(self.note_updated_at, optional=True)
        if type(self.file_modified_ns) is not int or self.file_modified_ns < 0:
            raise ValueError("file_modified_ns must be a non-negative integer")
        for field_name in (
            "note_character_count",
            "note_line_count",
            "file_character_count",
            "file_line_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if type(self.diff) is not str or len(self.diff) > COMPARISON_MAX_OUTPUT_CHARS:
            raise ValueError("diff exceeds the bounded output limit")
        if len(self.diff.splitlines()) > COMPARISON_MAX_OUTPUT_LINES:
            raise ValueError("diff exceeds the bounded output line limit")
        if type(self.input_elided) is not bool or type(self.output_elided) is not bool:
            raise TypeError("comparison elision flags must be booleans")

    @property
    def note_updated_label(self) -> str:
        """Return a bounded timestamp label for the note side."""

        return self.note_updated_at or "Unavailable"

    def __repr__(self) -> str:
        return (
            "ConflictComparison("
            f"note_chars={self.note_character_count}, "
            f"file_chars={self.file_character_count}, "
            f"input_elided={self.input_elided}, "
            f"output_elided={self.output_elided})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ConflictApplyResult:
    """Bounded result of one reviewed subset apply."""

    results: tuple[NotesSyncExecutionResult, ...]
    safe_completed: int
    conflicts_resolved: int
    unresolved_conflicts: int
    attention_remains: bool
    partial: bool
    needs_recovery: bool
    fresh_plan: ReconciliationPlan | None

    def __post_init__(self) -> None:
        if type(self.results) is not tuple:
            raise TypeError("results must be a tuple of NotesSyncExecutionResult")
        if self.results:
            from tldw_chatbook.Notes.notes_sync_executor import (
                NotesSyncExecutionResult,
            )

            if any(
                type(result) is not NotesSyncExecutionResult for result in self.results
            ):
                raise TypeError("results must be a tuple of NotesSyncExecutionResult")
        for field_name in (
            "safe_completed",
            "conflicts_resolved",
            "unresolved_conflicts",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if any(
            type(value) is not bool
            for value in (self.attention_remains, self.partial, self.needs_recovery)
        ):
            raise TypeError("apply result flags must be booleans")
        if (
            self.fresh_plan is not None
            and type(self.fresh_plan) is not ReconciliationPlan
        ):
            raise TypeError("fresh_plan must be a ReconciliationPlan or None")
        if (self.partial or self.needs_recovery) and self.fresh_plan is not None:
            raise ValueError("non-terminal apply results cannot carry a fresh plan")

    def __repr__(self) -> str:
        return (
            "ConflictApplyResult("
            f"safe_completed={self.safe_completed}, "
            f"conflicts_resolved={self.conflicts_resolved}, "
            f"unresolved_conflicts={self.unresolved_conflicts}, "
            f"partial={self.partial}, needs_recovery={self.needs_recovery})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ConflictReceipt:
    """One bounded in-process conflict-resolution receipt."""

    operation_id: str
    choice: NotesSyncConflictChoice
    state: str
    undo_available: bool
    undo_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_resolution_projection(
            operation_id=self.operation_id,
            choice=self.choice,
            state=self.state,
            undo_available=self.undo_available,
            undo_reason=self.undo_reason,
        )

    def __repr__(self) -> str:
        return "ConflictReceipt(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class ConflictHistoryRow:
    """One bounded durable resolution-history projection."""

    operation_id: str
    choice: NotesSyncConflictChoice
    state: str
    completed_at: str | None
    updated_at: str
    undo_available: bool
    undo_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_resolution_projection(
            operation_id=self.operation_id,
            choice=self.choice,
            state=self.state,
            undo_available=self.undo_available,
            undo_reason=self.undo_reason,
        )
        _validate_timestamp(self.completed_at, optional=True)
        _validate_timestamp(self.updated_at, optional=False)

    def __repr__(self) -> str:
        return "ConflictHistoryRow(<private>)"


def build_conflict_comparison(
    *,
    binding_id: str,
    title: str,
    relative_path: str,
    note_text: str,
    file_text: str,
    note_version: int,
    note_updated_at: str | None,
    file_modified_ns: int,
) -> ConflictComparison:
    """Build one bounded Note-to-File unified comparison.

    Args:
        binding_id: Opaque identifier of the reviewed binding.
        title: Note title used only for the bounded display label.
        relative_path: Normalized path of the bound file.
        note_text: Complete note content to compare.
        file_text: Complete file content to compare.
        note_version: Version of the observed note.
        note_updated_at: Optional ISO-8601 note update timestamp.
        file_modified_ns: File modification timestamp in nanoseconds.

    Returns:
        A validated, privacy-bounded comparison projection.

    Raises:
        TypeError: If comparison content or projection fields have invalid
            types.
        ValueError: If an identifier, path, version, timestamp, or display
            field is invalid.
    """

    if type(note_text) is not str or type(file_text) is not str:
        raise TypeError("comparison inputs must be strings")
    note_character_count = len(note_text)
    note_line_count = len(note_text.splitlines())
    file_character_count = len(file_text)
    file_line_count = len(file_text.splitlines())
    input_elided = (
        note_character_count > COMPARISON_MAX_INPUT_CHARS
        or note_line_count > COMPARISON_MAX_INPUT_LINES
        or file_character_count > COMPARISON_MAX_INPUT_CHARS
        or file_line_count > COMPARISON_MAX_INPUT_LINES
    )
    if input_elided:
        diff = (
            "Diff omitted because one or both complete inputs exceed the bounded "
            "comparison limit. "
            f"Note: {note_character_count:,} characters, {note_line_count:,} lines. "
            f"File: {file_character_count:,} characters, {file_line_count:,} lines."
        )
        output_elided = False
    else:
        diff_lines = list(
            unified_diff(
                note_text.splitlines(),
                file_text.splitlines(),
                fromfile="Note",
                tofile="File",
                lineterm="",
            )
        )
        full_diff = "\n".join(diff_lines)
        if diff_lines:
            full_diff += "\n"
        diff, output_elided = _bound_diff_output(full_diff)

    return ConflictComparison(
        binding_id=binding_id,
        note_title=_bounded_title(title),
        relative_path=relative_path,
        note_version=note_version,
        note_updated_at=note_updated_at,
        file_modified_ns=file_modified_ns,
        note_character_count=note_character_count,
        note_line_count=note_line_count,
        file_character_count=file_character_count,
        file_line_count=file_line_count,
        diff=diff,
        input_elided=input_elided,
        output_elided=output_elided,
    )


def conflict_copies_folder_id(note_scope_id: str) -> str:
    """Return the stable top-level conflict-copies folder ID.

    Args:
        note_scope_id: Opaque identifier of the note scope.

    Returns:
        A deterministic opaque folder identifier.

    Raises:
        ValueError: If ``note_scope_id`` is invalid.
    """

    validate_notes_sync_opaque_id(note_scope_id, field_name="note_scope_id")
    return _canonical_id("conflict_copies_folder_v1", note_scope_id)


def conflict_root_folder_id(note_scope_id: str, root_id: str) -> str:
    """Return the stable root child folder ID."""

    validate_notes_sync_opaque_id(note_scope_id, field_name="note_scope_id")
    validate_notes_sync_opaque_id(root_id, field_name="root_id")
    return _canonical_id("conflict_root_folder_v1", note_scope_id, root_id)


def conflict_copy_note_id(
    root_id: str,
    binding_id: str,
    observation_token: str,
) -> str:
    """Return the stable preserved-copy note ID for one review."""

    validate_notes_sync_opaque_id(root_id, field_name="root_id")
    validate_notes_sync_opaque_id(binding_id, field_name="binding_id")
    validate_notes_sync_digest(observation_token, field_name="observation_token")
    return _canonical_id(
        "conflict_copy_note_v1", root_id, binding_id, observation_token
    )


def conflict_resolution_operation_id(
    root_id: str,
    binding_id: str,
    observation_token: str,
    choice: NotesSyncConflictChoice,
) -> str:
    """Return the stable journal ID for one reviewed resolution."""

    validate_notes_sync_opaque_id(root_id, field_name="root_id")
    validate_notes_sync_opaque_id(binding_id, field_name="binding_id")
    validate_notes_sync_digest(observation_token, field_name="observation_token")
    if type(choice) is not NotesSyncConflictChoice:
        raise TypeError("choice must be a NotesSyncConflictChoice")
    if choice is NotesSyncConflictChoice.SKIP:
        raise ValueError("Skip does not create a conflict operation")
    return _canonical_id(
        "conflict_resolution_v1",
        root_id,
        binding_id,
        observation_token,
        choice.value,
    )


def linked_undo_operation_id(root_id: str, source_operation_id: str) -> str:
    """Return the stable linked Undo operation ID."""

    validate_notes_sync_opaque_id(root_id, field_name="root_id")
    validate_notes_sync_opaque_id(source_operation_id, field_name="source_operation_id")
    return _canonical_id("undo_resolution_v1", root_id, source_operation_id)


def _canonical_id(domain: str, *parts: str) -> str:
    return sha256("\0".join((domain, *parts)).encode("utf-8")).hexdigest()


def _bounded_title(value: str) -> str:
    if type(value) is not str:
        raise TypeError("title must be a string")
    return (" ".join(value.split()) or "Untitled")[:_DISPLAY_LABEL_MAX_CHARS]


def _bound_diff_output(diff: str) -> tuple[str, bool]:
    if (
        len(diff) <= COMPARISON_MAX_OUTPUT_CHARS
        and len(diff.splitlines()) <= COMPARISON_MAX_OUTPUT_LINES
    ):
        return diff, False

    retained: list[str] = []
    source_lines = diff.splitlines()
    for line in source_lines[: COMPARISON_MAX_OUTPUT_LINES - 1]:
        prefix = "\n".join(retained)
        separator_chars = 1 if retained else 0
        available = (
            COMPARISON_MAX_OUTPUT_CHARS
            - len(prefix)
            - separator_chars
            - 1
            - len(_OUTPUT_ELISION_MARKER)
        )
        if available <= 0:
            break
        if len(line) > available:
            retained.append(line[:available])
            break
        retained.append(line)
    return "\n".join((*retained, _OUTPUT_ELISION_MARKER)), True


def _validate_timestamp(value: str | None, *, optional: bool) -> None:
    if value is None:
        if optional:
            return
        raise ValueError("timestamp is required")
    if type(value) is not str or not value or len(value) > 64 or "\n" in value:
        raise ValueError("timestamp must be bounded ISO-8601 text")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("timestamp must be bounded ISO-8601 text") from None


def _validate_resolution_projection(
    *,
    operation_id: str,
    choice: NotesSyncConflictChoice,
    state: str,
    undo_available: bool,
    undo_reason: str | None,
) -> None:
    validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
    if (
        type(choice) is not NotesSyncConflictChoice
        or choice is NotesSyncConflictChoice.SKIP
    ):
        raise TypeError("choice must be a mutating NotesSyncConflictChoice")
    if type(state) is not str:
        raise TypeError("state must be a bounded reason code")
    validate_notes_sync_reason_code(state)
    if type(undo_available) is not bool:
        raise TypeError("undo_available must be a boolean")
    if undo_reason is not None and (
        type(undo_reason) is not str
        or not undo_reason
        or len(undo_reason) > _DISPLAY_LABEL_MAX_CHARS
        or "\n" in undo_reason
    ):
        raise ValueError("undo_reason must be a bounded single-line label")


__all__ = [
    "COMPARISON_MAX_INPUT_CHARS",
    "COMPARISON_MAX_INPUT_LINES",
    "COMPARISON_MAX_OUTPUT_CHARS",
    "COMPARISON_MAX_OUTPUT_LINES",
    "ELIGIBLE_CONFLICT_REASONS",
    "ConflictApplyResult",
    "ConflictComparison",
    "ConflictHistoryRow",
    "ConflictReceipt",
    "ConflictSelection",
    "NotesSyncConflictChoice",
    "build_conflict_comparison",
    "conflict_copies_folder_id",
    "conflict_copy_note_id",
    "conflict_resolution_operation_id",
    "conflict_root_folder_id",
    "eligible_conflict_reason",
    "linked_undo_operation_id",
]
