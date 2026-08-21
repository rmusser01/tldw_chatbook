"""Frozen, validated models shared by lasting Notes sync components."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath


_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def validate_notes_sync_opaque_id(value: object, *, field_name: str) -> str:
    """Return one bounded identifier that cannot carry a path or free text."""

    if type(value) is not str or _OPAQUE_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a bounded opaque identifier.")
    return value


def validate_notes_sync_digest(value: object, *, field_name: str) -> str:
    """Return one lowercase SHA-256 digest."""

    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def validate_notes_sync_reason_code(value: object) -> str | None:
    """Return a bounded machine reason code without accepting diagnostic text."""

    if value is None:
        return None
    if type(value) is not str or _REASON_CODE.fullmatch(value) is None:
        raise ValueError("reason code must be a bounded lowercase identifier.")
    return value


def normalize_notes_sync_relative_path(value: object) -> str:
    """Normalize a root-relative POSIX path and reject aliases or traversal."""

    if type(value) is not str or not value or "\\" in value:
        raise ValueError("relative path must be a normalized root-relative path.")
    windows_path = PureWindowsPath(value)
    if (
        PurePosixPath(value).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
    ):
        raise ValueError("relative path must be a normalized root-relative path.")
    parts = tuple(part for part in value.split("/") if part)
    if (
        not parts
        or len(value) > 4096
        or any(
            part in {".", ".."}
            or len(part) > 255
            or any(ord(character) < 32 or ord(character) == 127 for character in part)
            for part in parts
        )
    ):
        raise ValueError("relative path must be a normalized root-relative path.")
    return "/".join(parts)


class NotesSyncDirection(StrEnum):
    """Configured authority direction for one lasting root."""

    BIDIRECTIONAL = "bidirectional"
    FOLDER_TO_NOTES = "folder_to_notes"
    NOTES_TO_FOLDER = "notes_to_folder"


class NotesSyncRootState(StrEnum):
    """Durable root lifecycle state; runtime display status remains separate."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    DISCONNECTED = "disconnected"


class NotesSyncBindingState(StrEnum):
    """Durable ownership state for one note/file binding."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    PAUSED = "paused"
    NEEDS_ATTENTION = "needs_attention"
    DISCONNECTED = "disconnected"


class NotesSyncOperationState(StrEnum):
    """Durable journal stages advanced by the later executor task."""

    PENDING = "pending"
    RECOVERY_ADMITTED = "recovery_admitted"
    FIRST_AUTHORITY_APPLIED = "first_authority_applied"
    SECOND_AUTHORITY_APPLIED = "second_authority_applied"
    BINDING_UPDATED = "binding_updated"
    VERIFIED = "verified"
    NEEDS_ATTENTION = "needs_attention"
    COMPLETED = "completed"


class NotesSyncActionKind(StrEnum):
    """Pure reconciliation action classification."""

    NO_CHANGE = "no_change"
    CREATE_NOTE = "create_note"
    UPDATE_NOTE = "update_note"
    CREATE_FILE = "create_file"
    UPDATE_FILE = "update_file"
    MOVE_FILE = "move_file"
    CONFLICT = "conflict"
    DELETION_REVIEW = "deletion_review"
    PAUSE = "pause"


@dataclass(frozen=True, slots=True)
class NotesSyncSerializationProfile:
    """Byte-representation facts required for a lossless text round trip."""

    utf8_bom: bool
    newline: str
    final_newline: bool
    mode: int

    def __post_init__(self) -> None:
        if type(self.utf8_bom) is not bool or type(self.final_newline) is not bool:
            raise TypeError("serialization flags must be booleans.")
        if self.newline not in {"lf", "crlf"}:
            raise ValueError("newline must be 'lf' or 'crlf'.")
        if type(self.mode) is not int or not 0 <= self.mode <= 0o7777:
            raise ValueError("mode must be a supported non-negative file mode.")


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncFileIdentity:
    """Private stable identity for one admitted regular file."""

    device: int
    inode: int
    link_count: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0 for value in (self.device, self.inode)
        ):
            raise ValueError("file identity fields must be non-negative integers.")
        if type(self.link_count) is not int or self.link_count != 1:
            raise ValueError("a writable file identity must have exactly one link.")

    def __repr__(self) -> str:
        return "NotesSyncFileIdentity(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncFileObservation:
    """One immutable private filesystem observation."""

    relative_path: str
    identity: NotesSyncFileIdentity
    content_digest: str
    size_bytes: int
    serialization: NotesSyncSerializationProfile

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            normalize_notes_sync_relative_path(self.relative_path),
        )
        if type(self.identity) is not NotesSyncFileIdentity:
            raise TypeError("identity must be a NotesSyncFileIdentity.")
        validate_notes_sync_digest(self.content_digest, field_name="content_digest")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative.")
        if type(self.serialization) is not NotesSyncSerializationProfile:
            raise TypeError("serialization must be a NotesSyncSerializationProfile.")

    def __repr__(self) -> str:
        return "NotesSyncFileObservation(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncNoteObservation:
    """One immutable private note-authority observation."""

    note_scope_id: str
    note_id: str
    version: int
    content_digest: str

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.note_scope_id, field_name="note_scope_id")
        validate_notes_sync_opaque_id(self.note_id, field_name="note_id")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("version must be non-negative.")
        validate_notes_sync_digest(self.content_digest, field_name="content_digest")

    def __repr__(self) -> str:
        return "NotesSyncNoteObservation(<private>)"


@dataclass(frozen=True, slots=True)
class NotesSyncAction:
    """One privacy-safe action projection from pure reconciliation."""

    action_id: str
    kind: NotesSyncActionKind
    binding_id: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.action_id, field_name="action_id")
        if type(self.kind) is not NotesSyncActionKind:
            raise TypeError("kind must be a NotesSyncActionKind.")
        if self.binding_id is not None:
            validate_notes_sync_opaque_id(self.binding_id, field_name="binding_id")
        validate_notes_sync_reason_code(self.reason_code)


@dataclass(frozen=True, slots=True)
class NotesSyncPlan:
    """Immutable, path-free reconciliation plan authority."""

    root_id: str
    observation_token: str
    actions: tuple[NotesSyncAction, ...]

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        validate_notes_sync_opaque_id(
            self.observation_token,
            field_name="observation_token",
        )
        if type(self.actions) is not tuple or any(
            type(action) is not NotesSyncAction for action in self.actions
        ):
            raise TypeError("actions must be a tuple of NotesSyncAction values.")


@dataclass(frozen=True, slots=True)
class NotesSyncRecoveryAdmission:
    """Bounded, privacy-safe recovery-capacity decision."""

    admitted: bool
    reason_code: str | None = None
    required_bytes: int = 0
    available_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.admitted) is not bool:
            raise TypeError("admitted must be a boolean.")
        validate_notes_sync_reason_code(self.reason_code)
        for name, value in (
            ("required_bytes", self.required_bytes),
            ("available_bytes", self.available_bytes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.admitted and self.reason_code is not None:
            raise ValueError(
                "an admitted recovery decision cannot carry a reason code."
            )


__all__ = [
    "NotesSyncAction",
    "NotesSyncActionKind",
    "NotesSyncBindingState",
    "NotesSyncDirection",
    "NotesSyncFileIdentity",
    "NotesSyncFileObservation",
    "NotesSyncNoteObservation",
    "NotesSyncOperationState",
    "NotesSyncPlan",
    "NotesSyncRecoveryAdmission",
    "NotesSyncRootState",
    "NotesSyncSerializationProfile",
    "normalize_notes_sync_relative_path",
    "validate_notes_sync_digest",
    "validate_notes_sync_opaque_id",
    "validate_notes_sync_reason_code",
]
