"""Typed contracts for normalized Database Note folder operations."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import quote

FolderOwnership = Literal["manual", "managed"]
FolderManagedState = Literal["normal", "protected", "inactive_managed"]
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
class NoteFolderManagedStatus:
    """Authoritative managed-ownership status for one folder."""

    folder_id: str
    state: FolderManagedState

    def __post_init__(self) -> None:
        if not isinstance(self.folder_id, str) or not self.folder_id:
            raise ValueError("Folder status requires a folder identifier.")
        if self.state not in ("normal", "protected", "inactive_managed"):
            raise ValueError("Unknown managed folder status.")


def _validate_page_metadata(
    total: int,
    start_offset: int,
    previous_offset: int | None,
    next_offset: int | None,
    item_count: int,
) -> None:
    """Validate exact page counts and cursors shared by Notes tree envelopes."""
    values = (total, start_offset, previous_offset, next_offset)
    if any(value is not None and value < 0 for value in values):
        raise ValueError("Page totals and offsets must be nonnegative.")
    if next_offset is not None and next_offset > total:
        raise ValueError("Next offset cannot exceed the exact total.")
    if previous_offset is not None and previous_offset >= start_offset:
        raise ValueError("Previous offset must precede the page start.")
    if next_offset is not None and next_offset <= start_offset:
        raise ValueError("Next offset must follow the page start.")
    if start_offset == 0 and previous_offset is not None:
        raise ValueError("The first page cannot have a previous offset.")
    if start_offset > 0 and previous_offset is None:
        raise ValueError("A nonfirst page requires a previous offset.")
    if start_offset > total:
        if item_count or next_offset is not None or previous_offset > total:
            raise ValueError(
                "An out-of-range page must be empty and point back in range."
            )
        return

    end_offset = start_offset + item_count
    if end_offset > total:
        raise ValueError("Page items cannot extend beyond the exact total.")
    if not item_count and start_offset < total:
        raise ValueError("An in-range page cannot be empty before the exact total.")
    if next_offset is None and end_offset < total:
        raise ValueError("A nonfinal page requires a next offset.")
    if next_offset is not None and end_offset >= total:
        raise ValueError("A final page cannot have a next offset.")


@dataclass(frozen=True)
class NoteFolderChildPage:
    """One exact page of direct child folders."""

    folders: tuple[NoteFolder, ...]
    total_folders: int
    start_offset: int
    previous_offset: int | None
    next_offset: int | None
    folder_statuses: tuple[NoteFolderManagedStatus, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "folder_statuses", tuple(self.folder_statuses))
        _validate_page_metadata(
            self.total_folders,
            self.start_offset,
            self.previous_offset,
            self.next_offset,
            len(self.folders),
        )


@dataclass(frozen=True)
class NotePlacementRecord:
    """One note placement with its duplicate-safe membership identity."""

    note: Mapping[str, Any]
    folder_id: str | None
    membership: NoteFolderMembership | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "note", MappingProxyType(dict(self.note)))


@dataclass(frozen=True)
class NotePlacementPage:
    """One exact page of note placements beneath a tree parent."""

    placements: tuple[NotePlacementRecord, ...]
    total_placements: int
    start_offset: int
    previous_offset: int | None
    next_offset: int | None
    ancestor_folders: tuple[NoteFolder, ...] = ()
    folder_statuses: tuple[NoteFolderManagedStatus, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "folder_statuses", tuple(self.folder_statuses))
        _validate_page_metadata(
            self.total_placements,
            self.start_offset,
            self.previous_offset,
            self.next_offset,
            len(self.placements),
        )


@dataclass(frozen=True)
class NoteTreePathStep:
    """A folder and its parent-relative location along a tree path."""

    folder_id: str
    parent_id: str | None
    containing_offset: int

    def __post_init__(self) -> None:
        if self.containing_offset < 0:
            raise ValueError("Containing offset must be nonnegative.")


@dataclass(frozen=True)
class NoteTreeLocation:
    """A duplicate-safe folder or note location in the paged tree."""

    placement_id: str
    note_id: str | None
    membership_id: str | None
    path: tuple[NoteTreePathStep, ...]
    placement_offset: int | None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for index, step in enumerate(self.path):
            if step.folder_id in seen:
                raise ValueError("Tree location paths cannot repeat a folder.")
            if index == 0 and step.parent_id is not None:
                raise ValueError("Tree location paths must begin at the root.")
            if index and step.parent_id != self.path[index - 1].folder_id:
                raise ValueError("Tree location paths must form one connected chain.")
            seen.add(step.folder_id)

        is_folder = self.note_id is None
        if is_folder:
            if self.membership_id is not None or self.placement_offset is not None:
                raise ValueError(
                    "Folder locations cannot contain note placement fields."
                )
            if not self.path or self.placement_id != FolderPlacementId.folder(
                self.path[-1].folder_id
            ):
                raise ValueError(
                    "Folder locations must end at their stable placement ID."
                )
            return
        if self.placement_offset is None or self.placement_offset < 0:
            raise ValueError("Note locations require a nonnegative placement offset.")
        if self.membership_id is None:
            if self.path:
                raise ValueError("Root note locations cannot contain a folder path.")
            if self.placement_id != FolderPlacementId.unfiled(self.note_id):
                raise ValueError("Root note locations require an unfiled placement ID.")
            return
        if not self.path or self.placement_id != FolderPlacementId.note(
            self.path[-1].folder_id,
            self.note_id,
            self.membership_id,
        ):
            raise ValueError(
                "Filed note locations require their exact membership path."
            )


@dataclass(frozen=True)
class NoteTreeMutationContext:
    """Folder and placement parents affected by a tree mutation."""

    folder_ids: tuple[str, ...]
    parent_ids: tuple[str | None, ...]
    ancestor_ids: tuple[str, ...]
    placement_parent_ids: tuple[str, ...]


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
    """A mutation result separating its explicit target from derived effects."""

    folder: NoteFolder
    affected_folder_ids: tuple[str, ...]
    explicit_folder_id: str | None = None


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
    if not key or key in {".", ".."} or "/" in key or "\\" in key or "\x00" in key:
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
