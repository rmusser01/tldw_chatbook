"""Typed contracts for normalized Database Note folder operations."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote

FolderOwnership = Literal["manual", "managed"]
FolderCapabilityName = Literal[
    "list", "create", "rename", "move", "delete", "restore", "membership"
]


class FolderValidationError(ValueError):
    """Raised when a folder name does not form a valid path segment."""


class FolderCollisionError(RuntimeError):
    """Raised when a folder operation would create a normalized collision."""


class FolderConflictError(RuntimeError):
    """Raised when a folder operation has a version conflict."""


class FolderCapabilityError(RuntimeError):
    """Raised when a folder source does not support an operation.

    Args:
        reason_code: Stable machine-readable reason for the capability failure.
        user_message: User-facing explanation of the unavailable operation.
    """

    def __init__(self, *, reason_code: str, user_message: str) -> None:
        super().__init__(user_message)
        self.reason_code = reason_code
        self.user_message = user_message


@dataclass(frozen=True)
class NormalizedFolderName:
    """A display folder name and its normalized collision key."""

    display: str
    key: str


@dataclass(frozen=True)
class NoteFolder:
    """A folder snapshot in a normalized folder tree."""

    folder_id: str
    parent_id: str | None
    name: str
    path: str
    normalized_path: str
    version: int
    deleted: bool


@dataclass(frozen=True)
class NoteFolderMembership:
    """A note placement within a folder and its ownership state."""

    membership_id: str
    folder_id: str
    note_id: str
    ownership: FolderOwnership
    owner_id: str
    owner_active: bool
    version: int


@dataclass(frozen=True)
class NoteFolderCapability:
    """Availability information for one folder operation."""

    operation: FolderCapabilityName
    supported: bool
    reason_code: str = ""
    user_message: str = ""


@dataclass(frozen=True)
class NoteFolderPage:
    """A bounded folder-tree page with related note data."""

    folders: tuple[NoteFolder, ...]
    memberships: tuple[NoteFolderMembership, ...]
    notes: tuple[Mapping[str, Any], ...]
    total_folders: int
    total_notes: int
    next_offset: int | None
    next_folder_offset: int | None = None
    total_memberships: int = 0
    next_membership_offset: int | None = None
    managed_folder_ids: tuple[str, ...] = ()
    inactive_managed_folder_ids: tuple[str, ...] = ()
    unfiled_note_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class FolderMutationResult:
    """The folder returned by a mutation and every affected folder identifier."""

    folder: NoteFolder
    affected_folder_ids: tuple[str, ...]


@dataclass(frozen=True)
class RestoredManagedMembershipReview:
    """Inactive managed memberships needing restored-owner review."""

    owner_id: str
    membership_ids: tuple[str, ...]
    note_count: int
    folder_count: int


class FolderPlacementId:
    """Builds placement-aware tree identifiers for folders and notes."""

    @staticmethod
    def folder(folder_id: str) -> str:
        """Return the tree placement identifier for a folder."""
        return f"folder:{quote(folder_id, safe='')}"

    @staticmethod
    def note(folder_id: str, note_id: str, membership_id: str | None = None) -> str:
        """Return a folder/note identity, optionally for one exact membership.

        Args:
            folder_id: Folder containing the note placement.
            note_id: Note represented by the placement.
            membership_id: Exact membership identity for duplicate placements.

        Returns:
            A URL-escaped, stable tree placement identifier.
        """
        placement = f"note:{quote(folder_id, safe='')}:{quote(note_id, safe='')}"
        if membership_id is None:
            return placement
        return f"{placement}:{quote(membership_id, safe='')}"

    @staticmethod
    def unfiled(note_id: str) -> str:
        """Return the tree placement identifier for an unfiled note."""
        return f"unfiled:{quote(note_id, safe='')}"


def normalize_folder_name(name: str) -> NormalizedFolderName:
    """Validate and normalize a single folder-name segment.

    Args:
        name: Unnormalized folder name supplied by a caller.

    Returns:
        The trimmed display name and an NFKC/casefold collision key.

    Raises:
        FolderValidationError: If the value is not a valid folder-name segment.
    """
    if not isinstance(name, str):
        raise FolderValidationError("Folder name must be a string.")

    display = name.strip()
    if (
        not display
        or display in {".", ".."}
        or len(display) > 255
        or "/" in display
        or "\\" in display
        or "\x00" in display
    ):
        raise FolderValidationError("Folder name is not a valid path segment.")

    key = unicodedata.normalize("NFKC", display).casefold()
    if (
        not key
        or key in {".", ".."}
        or "/" in key
        or "\\" in key
        or "\x00" in key
    ):
        raise FolderValidationError("Folder name is not a valid path segment.")

    return NormalizedFolderName(display=display, key=key)


def join_normalized_folder_path(parent_path: str, child_key: str) -> str:
    """Join a parent normalized path and child normalized key.

    Args:
        parent_path: Existing normalized parent path, or an empty root path.
        child_key: Normalized key for the child segment.

    Returns:
        The normalized path for the child.

    Raises:
        FolderValidationError: If either value is not a safe normalized path.
    """
    if not isinstance(parent_path, str) or not isinstance(child_key, str):
        raise FolderValidationError("Normalized folder paths must be text.")
    if (
        not child_key
        or child_key in {".", ".."}
        or "/" in child_key
        or "\\" in child_key
        or "\x00" in child_key
    ):
        raise FolderValidationError("Child key is not a normalized path segment.")

    parent = parent_path.rstrip("/")
    if parent:
        if not parent.startswith("/"):
            raise FolderValidationError("Parent path must be absolute.")
        components = parent[1:].split("/")
        if any(
            not component
            or component in {".", ".."}
            or "\\" in component
            or "\x00" in component
            for component in components
        ):
            raise FolderValidationError("Parent path is not normalized.")
    elif parent_path not in {"", "/"}:
        raise FolderValidationError("Parent path is not normalized.")

    return f"{parent}/{child_key}" if parent else f"/{child_key}"
