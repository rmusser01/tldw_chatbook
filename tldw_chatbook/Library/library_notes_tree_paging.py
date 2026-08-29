"""Pure paging state for independently loaded Database Notes tree branches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import quote

from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NoteFolder,
    NoteFolderChildPage,
    NotePlacementPage,
    NotePlacementRecord,
)

NotesSliceKind = Literal["folders", "placements"]
NotesLoadDirection = Literal["replace", "more", "previous", "target"]
NotesSliceFreshness = Literal["uninitialized", "fresh", "stale"]
NotesApplyKind = Literal["applied", "ignored", "drift", "failed"]
NotesRecovery = Literal["reset_first", "reset_target"]
NotesSliceItem = NoteFolder | NotePlacementRecord


@dataclass(frozen=True)
class NotesBranchKey:
    """Stable identity for one folder or root child slice."""

    parent_id: str | None
    slice_kind: NotesSliceKind

    def __post_init__(self) -> None:
        if self.slice_kind not in ("folders", "placements"):
            raise ValueError("slice_kind must be folders or placements")
        if self.parent_id is not None and not self.parent_id:
            raise ValueError("parent_id cannot be empty")

    @property
    def pager_id(self) -> str:
        """Return the stable UI pager identifier for this slice."""
        parent = (
            "root"
            if self.parent_id is None
            else f"folder:{quote(self.parent_id, safe='')}"
        )
        return f"notes-tree:{parent}:{self.slice_kind}"


@dataclass(frozen=True)
class NotesBranchSliceState:
    """One immutable contiguous window loaded for a branch slice."""

    key: NotesBranchKey
    items: tuple[NotesSliceItem, ...]
    item_ids: tuple[str, ...]
    total: int | None
    start_offset: int
    previous_offset: int | None
    next_offset: int | None
    generation: int
    topology_epoch: int
    freshness: NotesSliceFreshness
    loading: bool = False
    recovery_attempted: bool = False
    requested_direction: NotesLoadDirection | None = None
    requested_offset: int | None = None
    requested_limit: int | None = None
    request_is_recovery: bool = False
    error: str = ""

    @property
    def pager_id(self) -> str:
        """Return the slice's stable pager identifier."""
        return self.key.pager_id


@dataclass(frozen=True)
class NotesSliceApplyResult:
    """Outcome of applying one asynchronous page response."""

    kind: NotesApplyKind
    state: NotesBranchSliceState
    recovery: NotesRecovery | None = None
    reason: str = ""


@dataclass(frozen=True)
class _PageData:
    items: tuple[NotesSliceItem, ...]
    item_ids: tuple[str, ...]
    total: int
    start: int
    previous: int | None
    next: int | None


def empty_notes_slice(
    key: NotesBranchKey, *, topology_epoch: int = 0
) -> NotesBranchSliceState:
    """Return an unloaded immutable slice for a branch key."""
    if topology_epoch < 0:
        raise ValueError("topology_epoch must be nonnegative")
    return NotesBranchSliceState(
        key=key,
        items=(),
        item_ids=(),
        total=None,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
        generation=0,
        topology_epoch=topology_epoch,
        freshness="uninitialized",
    )


def begin_notes_slice_load(
    state: NotesBranchSliceState,
    *,
    generation: int,
    direction: NotesLoadDirection,
    requested_offset: int,
    requested_limit: int,
    recovering: bool = False,
) -> NotesBranchSliceState:
    """Retain an exact immutable request contract while a page is loading.

    Args:
        state: Last state for the exact branch slice.
        generation: Generation assigned to the new request.
        direction: Requested replacement or continuation direction.
        requested_offset: Exact repository offset requested.
        requested_limit: Maximum number of requested records.
        recovering: Whether this is the slice's one recovery attempt.

    Returns:
        A new loading state retaining the request contract.
    """
    if generation < 0:
        raise ValueError("generation must be nonnegative")
    if direction not in ("replace", "more", "previous", "target"):
        raise ValueError("direction is not supported")
    if requested_offset < 0:
        raise ValueError("requested_offset must be nonnegative")
    if requested_limit < 1:
        raise ValueError("requested_limit must be positive")
    if recovering and not state.recovery_attempted:
        raise ValueError("recovery must follow a drift result")
    return replace(
        state,
        generation=generation,
        loading=True,
        requested_direction=direction,
        requested_offset=requested_offset,
        requested_limit=requested_limit,
        request_is_recovery=recovering,
        error="",
    )


def invalidate_notes_slice(
    state: NotesBranchSliceState, *, topology_epoch: int
) -> NotesBranchSliceState:
    """Clear a slice and advance its epoch so prior responses are ignored."""
    if topology_epoch <= state.topology_epoch:
        raise ValueError("topology_epoch must advance")
    return replace(
        state,
        items=(),
        item_ids=(),
        total=None,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
        generation=state.generation + 1,
        topology_epoch=topology_epoch,
        freshness="uninitialized",
        loading=False,
        recovery_attempted=False,
        requested_direction=None,
        requested_offset=None,
        requested_limit=None,
        request_is_recovery=False,
        error="",
    )


def apply_notes_slice_page(
    current: NotesBranchSliceState,
    incoming: NoteFolderChildPage | NotePlacementPage,
    *,
    direction: NotesLoadDirection,
    request_generation: int,
    topology_epoch: int,
) -> NotesSliceApplyResult:
    """Apply a page or report an ignored race/drift without raising.

    Args:
        current: Last reducer state for this exact branch slice.
        incoming: Typed repository page returned by the request.
        direction: Whether the response replaces, appends, or prepends.
        request_generation: Generation captured when the request began.
        topology_epoch: Tree topology epoch captured by the request.

    Returns:
        An explicit applied, ignored, or drift transition.
    """
    if (
        request_generation != current.generation
        or topology_epoch != current.topology_epoch
    ):
        return NotesSliceApplyResult("ignored", current, reason="obsolete request")
    if (
        not current.loading
        or current.requested_direction is None
        or current.requested_offset is None
        or current.requested_limit is None
    ):
        return NotesSliceApplyResult("ignored", current, reason="no active request")
    if direction not in ("replace", "more", "previous", "target"):
        return _drift(current, direction="replace", reason="unknown direction")
    if direction != current.requested_direction:
        return _drift(current, direction=direction, reason="request direction changed")

    try:
        page = _page_data(current.key, incoming)
    except (TypeError, ValueError, KeyError):
        return _drift(current, direction=direction, reason="invalid page identity")

    if page.start != current.requested_offset:
        return _drift(
            current,
            direction=direction,
            reason="response offset differs from request",
        )
    if not _coherent_page(
        page,
        requested_offset=current.requested_offset,
        requested_limit=current.requested_limit,
    ):
        return _drift(current, direction=direction, reason="incoherent page metadata")

    continuation = direction in ("more", "previous")
    if continuation and (current.freshness != "fresh" or current.total is None):
        return _drift(
            current, direction=direction, reason="continuation has no exact base"
        )
    if continuation and page.total != current.total:
        return _drift(current, direction=direction, reason="exact total changed")
    if continuation and set(current.item_ids).intersection(page.item_ids):
        return _drift(current, direction=direction, reason="stable identity overlap")

    if direction in ("replace", "target"):
        return NotesSliceApplyResult(
            "applied",
            _replace_window(current, page),
        )

    if direction == "more":
        expected_start = current.start_offset + len(current.items)
        if current.requested_offset != expected_start:
            return _drift(current, direction=direction, reason="nonadjacent append")
        state = replace(
            current,
            items=current.items + page.items,
            item_ids=current.item_ids + page.item_ids,
            next_offset=page.next,
            loading=False,
            recovery_attempted=False,
            requested_direction=None,
            requested_offset=None,
            requested_limit=None,
            request_is_recovery=False,
            error="",
        )
    else:
        if current.requested_offset + len(page.items) != current.start_offset:
            return _drift(current, direction=direction, reason="nonadjacent prepend")
        if page.next != current.start_offset:
            return _drift(
                current, direction=direction, reason="incoherent prepend cursor"
            )
        state = replace(
            current,
            items=page.items + current.items,
            item_ids=page.item_ids + current.item_ids,
            start_offset=page.start,
            previous_offset=page.previous,
            loading=False,
            recovery_attempted=False,
            requested_direction=None,
            requested_offset=None,
            requested_limit=None,
            request_is_recovery=False,
            error="",
        )
    return NotesSliceApplyResult("applied", state)


def _replace_window(
    current: NotesBranchSliceState, page: _PageData
) -> NotesBranchSliceState:
    return replace(
        current,
        items=page.items,
        item_ids=page.item_ids,
        total=page.total,
        start_offset=page.start,
        previous_offset=page.previous,
        next_offset=page.next,
        freshness="fresh",
        loading=False,
        recovery_attempted=False,
        requested_direction=None,
        requested_offset=None,
        requested_limit=None,
        request_is_recovery=False,
        error="",
    )


def fail_notes_slice_load(
    current: NotesBranchSliceState,
    *,
    request_generation: int,
    topology_epoch: int,
    error: str,
) -> NotesSliceApplyResult:
    """Finish a failed request while preserving the last visible rows.

    Obsolete failures are ignored. A normal failure retains authoritative
    metadata for retry; a recovery failure makes that metadata stale.

    Args:
        current: Loading state for the exact branch slice.
        request_generation: Generation captured by the failed request.
        topology_epoch: Topology epoch captured by the failed request.
        error: Recoverable user-facing or diagnostic failure text.

    Returns:
        An ignored or failed immutable transition.
    """
    if (
        request_generation != current.generation
        or topology_epoch != current.topology_epoch
        or not current.loading
    ):
        return NotesSliceApplyResult("ignored", current, reason="obsolete request")
    if not isinstance(error, str) or not error.strip():
        raise ValueError("error must be nonempty text")
    if current.request_is_recovery:
        stale = replace(
            current,
            total=None,
            previous_offset=None,
            next_offset=None,
            freshness="stale",
            loading=False,
            requested_direction=None,
            requested_offset=None,
            requested_limit=None,
            request_is_recovery=False,
            error=error,
        )
        return NotesSliceApplyResult("failed", stale, reason=error)
    failed = replace(
        current,
        loading=False,
        requested_direction=None,
        requested_offset=None,
        requested_limit=None,
        request_is_recovery=False,
        error=error,
    )
    return NotesSliceApplyResult("failed", failed, reason=error)


def _drift(
    current: NotesBranchSliceState,
    *,
    direction: str,
    reason: str,
) -> NotesSliceApplyResult:
    if current.recovery_attempted:
        stale = replace(
            current,
            total=None,
            previous_offset=None,
            next_offset=None,
            freshness="stale",
            loading=False,
            requested_direction=None,
            requested_offset=None,
            requested_limit=None,
            request_is_recovery=False,
            error=reason,
        )
        return NotesSliceApplyResult("drift", stale, reason=reason)
    recovery: NotesRecovery = "reset_target" if direction == "target" else "reset_first"
    recovering = replace(
        current,
        loading=False,
        recovery_attempted=True,
        requested_direction=None,
        requested_offset=None,
        requested_limit=None,
        request_is_recovery=False,
        error="",
    )
    return NotesSliceApplyResult("drift", recovering, recovery=recovery, reason=reason)


def _page_data(
    key: NotesBranchKey,
    incoming: NoteFolderChildPage | NotePlacementPage,
) -> _PageData:
    if key.slice_kind == "folders" and isinstance(incoming, NoteFolderChildPage):
        items: tuple[NotesSliceItem, ...] = incoming.folders
        if any(folder.parent_id != key.parent_id for folder in incoming.folders):
            raise ValueError("folder page does not belong to its branch parent")
        ids = tuple(FolderPlacementId.folder(folder.folder_id) for folder in items)
        total = incoming.total_folders
    elif key.slice_kind == "placements" and isinstance(incoming, NotePlacementPage):
        items = incoming.placements
        if any(item.folder_id != key.parent_id for item in incoming.placements):
            raise ValueError("placement page does not belong to its branch parent")
        ids = tuple(_placement_id(item) for item in items)
        total = incoming.total_placements
    else:
        raise TypeError("page does not match branch slice kind")
    if len(ids) != len(set(ids)):
        raise ValueError("page contains duplicate identities")
    return _PageData(
        items=items,
        item_ids=ids,
        total=total,
        start=incoming.start_offset,
        previous=incoming.previous_offset,
        next=incoming.next_offset,
    )


def _placement_id(item: NotesSliceItem) -> str:
    if not isinstance(item, NotePlacementRecord):
        raise TypeError("placement page contains a folder")
    note_id = _note_id(item.note)
    if item.folder_id is None:
        if item.membership is not None:
            raise ValueError("unfiled placement cannot have a membership")
        return FolderPlacementId.unfiled(note_id)
    membership = item.membership
    if (
        membership is None
        or membership.folder_id != item.folder_id
        or membership.note_id != note_id
    ):
        raise ValueError("folder placement requires its exact membership")
    return FolderPlacementId.note(item.folder_id, note_id, membership.membership_id)


def _note_id(note: Mapping[str, Any]) -> str:
    value = note.get("id", note.get("note_id"))
    if value is None or not str(value):
        raise ValueError("note has no stable identity")
    return str(value)


def _coherent_page(
    page: _PageData,
    *,
    requested_offset: int,
    requested_limit: int,
) -> bool:
    count = len(page.items)
    end = page.start + count
    if page.total < 0 or page.start < 0 or end > page.total:
        return False
    expected_count = min(requested_limit, max(page.total - requested_offset, 0))
    if page.start != requested_offset or count != expected_count:
        return False
    if len(page.item_ids) != count:
        return False
    if page.next != (end if end < page.total else None):
        return False
    expected_previous = (
        None
        if page.start == 0
        else min(
            max(0, page.start - requested_limit),
            max(0, page.total - requested_limit),
        )
    )
    return page.previous == expected_previous
