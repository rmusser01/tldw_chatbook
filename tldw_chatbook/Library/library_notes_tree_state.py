"""Pure, placement-aware display state for the Database Notes folder tree."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NoteFolder,
    NoteFolderMembership,
    NotePlacementPage,
    NotePlacementRecord,
)
from tldw_chatbook.Library.library_notes_tree_paging import (
    LIBRARY_NOTES_TREE_PAGE_SIZE,
    NotesBranchKey,
    NotesBranchSliceState,
    NotesLoadDirection,
    NotesSliceKind,
    retitle_note_placements,
)

LibraryNotesTreeRowKind = Literal["folder", "note", "unfiled", "pager"]
LibraryNotesTreeSemanticStatus = Literal["normal", "connected", "needs_attention"]
LibraryNotesTreePagingAction = Literal["earlier", "more", "retry"]
LibraryNotesFilterApplyKind = Literal["applied", "ignored", "drift", "failed"]

UNFILED_PLACEMENT_ID = "virtual:unfiled"


@dataclass(frozen=True)
class LibraryNotesTreeRow:
    """One visible folder, Unfiled, note-placement, or branch-pager row."""

    placement_id: str
    kind: LibraryNotesTreeRowKind
    label: str
    depth: int
    note_id: str | None = None
    folder_id: str | None = None
    membership_id: str | None = None
    breadcrumb: str = ""
    ownership: Literal["manual", "managed"] | None = None
    owner_active: bool = True
    protected: bool = False
    semantic_status: LibraryNotesTreeSemanticStatus = "normal"
    status_text: str = ""
    expanded: bool = False
    version: int | None = None
    parent_folder_id: str | None = None
    content_kind: NotesSliceKind | None = None
    paging_action: LibraryNotesTreePagingAction | None = None
    retry_direction: NotesLoadDirection | None = None
    range_copy: str = ""
    action_copy: str = ""
    focus_id: str = ""
    loading: bool = False
    disabled: bool = False
    unsafe_mutation_disabled: bool = False


@dataclass(frozen=True)
class LibraryNotesTreeIdentity:
    """Stable navigator identity: placement first, underlying note second."""

    placement_id: str
    note_id: str | None = None


@dataclass(frozen=True)
class LibraryNotesTreeProjection:
    """Visible placement-aware rows."""

    rows: tuple[LibraryNotesTreeRow, ...]

    def row(self, placement_id: str) -> LibraryNotesTreeRow | None:
        """Return one visible row by exact placement identity."""
        return next(
            (row for row in self.rows if row.placement_id == placement_id), None
        )


@dataclass(frozen=True)
class LibraryNotesBranchRange:
    """Semantic descriptor for one exact contiguous browse window."""

    parent_id: str | None
    content_kind: NotesSliceKind
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("branch range offsets are invalid")

    @property
    def key(self) -> NotesBranchKey:
        """Return the exact branch identity without carrying its records."""
        return NotesBranchKey(self.parent_id, self.content_kind)


@dataclass(frozen=True)
class LibraryNotesFilterRange:
    """Semantic descriptor for one exact contiguous filter window."""

    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("filter range offsets are invalid")


@dataclass(frozen=True)
class LibraryNotesTreeReceipt:
    """Record-free semantic context retained across a focused Notes task."""

    selected_placement_id: str
    selected_note_id: str
    expanded_folder_ids: tuple[str, ...]
    branch_ranges: tuple[LibraryNotesBranchRange, ...]
    filter_query: str
    filter_range: LibraryNotesFilterRange | None
    focus_semantic_id: str
    focus_role: str
    scroll_offset: tuple[int, int] | None
    rail_scroll_offset: tuple[int, int] | None
    lifecycle_generation: int
    topology_epoch: int
    preferred_folder_id: str | None = None
    preferred_membership_id: str | None = None


@dataclass(frozen=True)
class LibraryNotesFilterState:
    """One independent exact placement window owned by the active query."""

    query: str
    placements: tuple[NotePlacementRecord, ...]
    ancestor_folders: tuple[NoteFolder, ...]
    total: int | None
    start_offset: int
    previous_offset: int | None
    next_offset: int | None
    generation: int
    topology_epoch: int
    loading: bool = False
    stale: bool = False
    recovery_attempted: bool = False
    requested_direction: NotesLoadDirection | None = None
    requested_offset: int | None = None
    requested_limit: int | None = None
    request_is_recovery: bool = False
    failed_direction: NotesLoadDirection | None = None
    failed_offset: int | None = None
    error: str = ""

    @classmethod
    def empty(
        cls, *, query: str, generation: int, topology_epoch: int
    ) -> LibraryNotesFilterState:
        """Return a record-free initial request owner for one query."""
        return cls(query, (), (), None, 0, None, None, generation, topology_epoch)

    @classmethod
    def from_page(
        cls,
        *,
        query: str,
        page: NotePlacementPage,
        generation: int,
        topology_epoch: int,
    ) -> LibraryNotesFilterState:
        """Create an exact fresh filter window from one repository page."""
        return cls(
            query=query,
            placements=page.placements,
            ancestor_folders=page.ancestor_folders,
            total=page.total_placements,
            start_offset=page.start_offset,
            previous_offset=page.previous_offset,
            next_offset=page.next_offset,
            generation=generation,
            topology_epoch=topology_epoch,
        )

    @property
    def range_descriptor(self) -> LibraryNotesFilterRange:
        """Return the record-free descriptor for this loaded window."""
        return LibraryNotesFilterRange(
            self.start_offset, self.start_offset + len(self.placements)
        )

    def begin(
        self,
        *,
        generation: int,
        offset: int,
        direction: NotesLoadDirection = "replace",
        limit: int = LIBRARY_NOTES_TREE_PAGE_SIZE,
        recovering: bool = False,
    ) -> LibraryNotesFilterState:
        """Retain last-good rows while beginning one exact request."""
        return begin_library_notes_filter_load(
            self,
            generation=generation,
            direction=direction,
            offset=offset,
            limit=limit,
            recovering=recovering,
        )

    def fail(self, *, direction: NotesLoadDirection) -> LibraryNotesFilterState:
        """Keep rows and make only this filter window retryable."""
        loading = replace(
            self,
            requested_direction=direction,
            requested_offset=self.requested_offset or 0,
            requested_limit=(
                self.requested_limit or LIBRARY_NOTES_TREE_PAGE_SIZE
            ),
            loading=True,
        )
        return fail_library_notes_filter_load(
            loading,
            request_generation=loading.generation,
            topology_epoch=loading.topology_epoch,
            error="Could not load filtered notes.",
        ).state


@dataclass(frozen=True)
class LibraryNotesFilterApplyResult:
    """Pure outcome of one exact filter page transition."""

    kind: LibraryNotesFilterApplyKind
    state: LibraryNotesFilterState
    recovery_offset: int | None = None
    reason: str = ""


def begin_library_notes_filter_load(
    state: LibraryNotesFilterState,
    *,
    generation: int,
    direction: NotesLoadDirection,
    offset: int,
    limit: int,
    recovering: bool = False,
) -> LibraryNotesFilterState:
    """Retain last-good filter rows and record one exact request contract."""
    if direction not in ("replace", "more", "previous", "target"):
        raise ValueError("unsupported filter direction")
    if generation < 0 or offset < 0 or limit < 1:
        raise ValueError("invalid filter request bounds")
    if recovering and not state.recovery_attempted:
        raise ValueError("filter recovery must follow drift")
    return replace(
        state,
        generation=generation,
        loading=True,
        requested_direction=direction,
        requested_offset=offset,
        requested_limit=limit,
        request_is_recovery=recovering,
        failed_direction=None,
        failed_offset=None,
        error="",
    )


def apply_library_notes_filter_page(
    current: LibraryNotesFilterState,
    incoming: NotePlacementPage,
    *,
    request_generation: int,
    topology_epoch: int,
) -> LibraryNotesFilterApplyResult:
    """Apply one coherent exact filter page or return drift without mutation."""
    if (
        request_generation != current.generation
        or topology_epoch != current.topology_epoch
        or not current.loading
        or current.requested_direction is None
        or current.requested_offset is None
        or current.requested_limit is None
    ):
        return LibraryNotesFilterApplyResult("ignored", current, reason="obsolete")
    direction = current.requested_direction
    try:
        incoming_ids = tuple(_filter_placement_id(item) for item in incoming.placements)
    except (TypeError, ValueError):
        return _filter_drift(current, incoming, reason="invalid placement identity")
    if len(incoming_ids) != len(set(incoming_ids)):
        return _filter_drift(current, incoming, reason="duplicate placement identity")
    if not _valid_filter_ancestor_topology(
        incoming.placements, incoming.ancestor_folders
    ):
        return _filter_drift(current, incoming, reason="invalid ancestor topology")
    if not _coherent_filter_page(
        incoming,
        requested_offset=current.requested_offset,
        requested_limit=current.requested_limit,
    ):
        return _filter_drift(current, incoming, reason="incoherent page metadata")

    continuation = direction in ("more", "previous")
    current_ids = tuple(_filter_placement_id(item) for item in current.placements)
    if continuation and current.total is None:
        return _filter_drift(current, incoming, reason="continuation has no exact base")
    if continuation and incoming.total_placements != current.total:
        return _filter_drift(current, incoming, reason="exact total changed")
    if continuation and set(current_ids).intersection(incoming_ids):
        return _filter_drift(current, incoming, reason="stable identity overlap")

    if direction in ("replace", "target"):
        placements = incoming.placements
        start = incoming.start_offset
        previous = incoming.previous_offset
        next_ = incoming.next_offset
        ancestor_candidates = incoming.ancestor_folders
    elif direction == "more":
        if current.requested_offset != current.start_offset + len(current.placements):
            return _filter_drift(current, incoming, reason="nonadjacent append")
        placements = current.placements + incoming.placements
        start = current.start_offset
        previous = current.previous_offset
        next_ = incoming.next_offset
        ancestor_candidates = (*current.ancestor_folders, *incoming.ancestor_folders)
    else:
        if current.requested_offset + len(incoming.placements) != current.start_offset:
            return _filter_drift(current, incoming, reason="nonadjacent prepend")
        if incoming.next_offset != current.start_offset:
            return _filter_drift(current, incoming, reason="incoherent prepend cursor")
        placements = incoming.placements + current.placements
        start = incoming.start_offset
        previous = incoming.previous_offset
        next_ = current.next_offset
        ancestor_candidates = (*incoming.ancestor_folders, *current.ancestor_folders)

    state = replace(
        current,
        placements=placements,
        ancestor_folders=_filter_ancestors_for(placements, ancestor_candidates),
        total=incoming.total_placements,
        start_offset=start,
        previous_offset=previous,
        next_offset=next_,
        loading=False,
        stale=False,
        recovery_attempted=False,
        requested_direction=None,
        requested_offset=None,
        requested_limit=None,
        request_is_recovery=False,
        failed_direction=None,
        failed_offset=None,
        error="",
    )
    return LibraryNotesFilterApplyResult("applied", state)


def fail_library_notes_filter_load(
    current: LibraryNotesFilterState,
    *,
    request_generation: int,
    topology_epoch: int,
    error: str,
) -> LibraryNotesFilterApplyResult:
    """Finish one exact filter failure while retaining its retry authority."""
    if (
        request_generation != current.generation
        or topology_epoch != current.topology_epoch
        or not current.loading
        or current.requested_direction is None
        or current.requested_offset is None
    ):
        return LibraryNotesFilterApplyResult("ignored", current, reason="obsolete")
    stale = current.stale or current.request_is_recovery
    state = replace(
        current,
        total=None if stale else current.total,
        previous_offset=None if stale else current.previous_offset,
        next_offset=None if stale else current.next_offset,
        loading=False,
        stale=stale,
        requested_direction=None,
        requested_offset=None,
        requested_limit=None,
        request_is_recovery=False,
        failed_direction=current.requested_direction,
        failed_offset=current.requested_offset,
        error=error,
    )
    return LibraryNotesFilterApplyResult("failed", state, reason=error)


def reconcile_library_notes_filter_commit(
    state: LibraryNotesFilterState,
    *,
    operation: str,
    folder: NoteFolder | None = None,
    affected_folder_ids: frozenset[str] = frozenset(),
    removed_folder_ids: frozenset[str] = frozenset(),
    note_id: str = "",
    source_placement_id: str = "",
    partial: bool = False,
) -> LibraryNotesFilterState:
    """Apply only deterministic committed truth before an exact filter refresh."""
    placements = state.placements
    if operation == "delete_folder" and removed_folder_ids:
        placements = tuple(
            item for item in placements if item.folder_id not in removed_folder_ids
        )
    elif operation in {"detach_placement", "move_placement"} and not partial:
        placements = tuple(
            item
            for item in placements
            if _filter_placement_id(item) != source_placement_id
        )
    elif operation == "note_delete":
        placements = tuple(
            item for item in placements if _record_id(item.note) != note_id
        )

    ancestors = state.ancestor_folders
    if removed_folder_ids:
        ancestors = tuple(
            item for item in ancestors if item.folder_id not in removed_folder_ids
        )
    if folder is not None and operation in {"rename_folder", "move_folder"}:
        old = next(
            (item for item in ancestors if item.folder_id == folder.folder_id), None
        )
        patched: list[NoteFolder] = []
        for item in ancestors:
            if item.folder_id == folder.folder_id:
                patched.append(folder)
            elif (
                old is not None
                and item.folder_id in affected_folder_ids
                and item.path.startswith(f"{old.path}/")
            ):
                patched.append(
                    replace(
                        item,
                        path=f"{folder.path}{item.path[len(old.path) :]}",
                        normalized_path=(
                            f"{folder.normalized_path}"
                            f"{item.normalized_path[len(old.normalized_path) :]}"
                        ),
                    )
                )
            else:
                patched.append(item)
        ancestors = tuple(patched)

    ancestors = _filter_ancestors_for(placements, ancestors)
    return replace(
        state,
        placements=placements,
        ancestor_folders=ancestors,
        total=None,
        previous_offset=None,
        next_offset=None,
        loading=False,
        stale=True,
        recovery_attempted=False,
        requested_direction=None,
        requested_offset=None,
        requested_limit=None,
        request_is_recovery=False,
        failed_direction="target",
        failed_offset=state.start_offset,
        error="Committed change needs refresh.",
    )


def _filter_drift(
    current: LibraryNotesFilterState,
    incoming: NotePlacementPage,
    *,
    reason: str,
) -> LibraryNotesFilterApplyResult:
    if current.request_is_recovery or current.recovery_attempted:
        stale = replace(
            current,
            total=None,
            previous_offset=None,
            next_offset=None,
            loading=False,
            stale=True,
            requested_direction=None,
            requested_offset=None,
            requested_limit=None,
            request_is_recovery=False,
            failed_direction=None,
            failed_offset=None,
            error="Filtered notes changed. Retry.",
        )
        return LibraryNotesFilterApplyResult("drift", stale, reason=reason)
    assert current.requested_offset is not None
    assert current.requested_limit is not None
    total = incoming.total_placements
    last_offset = (
        ((total - 1) // current.requested_limit) * current.requested_limit
        if total > 0
        else 0
    )
    recovery_offset = min(current.requested_offset, last_offset)
    recovering = replace(
        current,
        loading=False,
        recovery_attempted=True,
        requested_direction=None,
        requested_offset=None,
        requested_limit=None,
        request_is_recovery=False,
        failed_direction=None,
        failed_offset=None,
        error="",
    )
    return LibraryNotesFilterApplyResult(
        "drift", recovering, recovery_offset=recovery_offset, reason=reason
    )


def _filter_placement_id(item: NotePlacementRecord) -> str:
    note_id = _record_id(item.note)
    if not note_id:
        raise ValueError("filter placement has no note identity")
    if item.folder_id is None:
        if item.membership is not None:
            raise ValueError("unfiled placement cannot carry membership")
        return FolderPlacementId.unfiled(note_id)
    membership = item.membership
    if (
        membership is None
        or membership.folder_id != item.folder_id
        or membership.note_id != note_id
    ):
        raise ValueError("folder placement requires its exact membership")
    return FolderPlacementId.note(item.folder_id, note_id, membership.membership_id)


def _coherent_filter_page(
    page: NotePlacementPage, *, requested_offset: int, requested_limit: int
) -> bool:
    count = len(page.placements)
    end = page.start_offset + count
    total = page.total_placements
    if (
        total < 0
        or page.start_offset != requested_offset
        or end > total
        or requested_offset > 0
        and requested_offset >= total
    ):
        return False
    if count != min(requested_limit, max(total - requested_offset, 0)):
        return False
    if page.next_offset != (end if end < total else None):
        return False
    expected_previous = (
        None
        if page.start_offset == 0
        else min(
            max(0, page.start_offset - requested_limit),
            max(0, total - requested_limit),
        )
    )
    return page.previous_offset == expected_previous


def _filter_ancestors_for(
    placements: tuple[NotePlacementRecord, ...],
    candidates: tuple[NoteFolder, ...],
) -> tuple[NoteFolder, ...]:
    folders = {folder.folder_id: folder for folder in candidates}
    retained: set[str] = set()
    for placement in placements:
        folder_id = placement.folder_id
        seen: set[str] = set()
        while folder_id is not None and folder_id not in seen:
            seen.add(folder_id)
            folder = folders.get(folder_id)
            if folder is None:
                break
            retained.add(folder_id)
            folder_id = folder.parent_id
    return tuple(
        folder for folder_id, folder in folders.items() if folder_id in retained
    )


def _valid_filter_ancestor_topology(
    placements: tuple[NotePlacementRecord, ...],
    candidates: tuple[NoteFolder, ...],
) -> bool:
    folder_ids = tuple(folder.folder_id for folder in candidates)
    if len(folder_ids) != len(set(folder_ids)):
        return False
    folders = {folder.folder_id: folder for folder in candidates}
    for placement in placements:
        folder_id = placement.folder_id
        seen: set[str] = set()
        while folder_id is not None:
            if folder_id in seen:
                return False
            seen.add(folder_id)
            folder = folders.get(folder_id)
            if folder is None or folder.deleted:
                return False
            folder_id = folder.parent_id
    return True


def _record_id(note: Mapping[str, object]) -> str:
    return str(note.get("id", note.get("note_id", "")) or "")


def _record_title(note: Mapping[str, object]) -> str:
    return str(note.get("title", "") or "Untitled")


def _note_row(
    *,
    note: Mapping[str, object],
    folder: NoteFolder | None,
    membership: NoteFolderMembership | None,
    depth: int,
    unsafe_mutation_disabled: bool = False,
) -> LibraryNotesTreeRow:
    note_id = _record_id(note)
    title = _record_title(note)
    if folder is None:
        return LibraryNotesTreeRow(
            placement_id=FolderPlacementId.unfiled(note_id),
            kind="note",
            label=title,
            depth=depth,
            note_id=note_id,
            breadcrumb=f"Unfiled / {title}",
            unsafe_mutation_disabled=unsafe_mutation_disabled,
        )

    assert membership is not None
    managed = membership.ownership == "managed"
    active = membership.owner_active
    if managed and active:
        semantic_status: LibraryNotesTreeSemanticStatus = "connected"
        status_text = "⇄ Synced placement"
    elif managed:
        semantic_status = "needs_attention"
        status_text = "! Needs owner review"
    else:
        semantic_status = "normal"
        status_text = ""
    path = folder.path.strip("/").replace("/", " / ")
    return LibraryNotesTreeRow(
        placement_id=FolderPlacementId.note(
            folder.folder_id, note_id, membership.membership_id
        ),
        kind="note",
        label=title,
        depth=depth,
        note_id=note_id,
        folder_id=folder.folder_id,
        membership_id=membership.membership_id,
        breadcrumb=f"{path} / {title}" if path else title,
        ownership=membership.ownership,
        owner_active=active,
        protected=managed,
        semantic_status=semantic_status,
        status_text=status_text,
        version=membership.version,
        unsafe_mutation_disabled=unsafe_mutation_disabled,
    )


def _pager_boundary_identity(
    state: NotesBranchSliceState,
    action: LibraryNotesTreePagingAction,
    retry_direction: NotesLoadDirection | None,
) -> str:
    if action in ("earlier", "more"):
        return action
    direction = retry_direction or state.requested_direction or state.failed_direction
    if direction == "previous":
        return "earlier"
    if direction in ("more", "target"):
        return direction
    return "replace"


def _pager_focus_id(
    key: NotesBranchKey,
    boundary_identity: str,
) -> str:
    parent = (
        "root"
        if key.parent_id is None
        else f"folder-{key.parent_id.encode('utf-8').hex()}"
    )
    return f"library-notes-tree-pager-{parent}-{key.slice_kind}-{boundary_identity}"


def _pager_row(
    state: NotesBranchSliceState,
    *,
    action: LibraryNotesTreePagingAction,
    depth: int,
    range_copy: str = "",
    status_copy: str = "",
    action_copy: str = "",
    retry_direction: NotesLoadDirection | None = None,
    disabled: bool = False,
    loading: bool = False,
) -> LibraryNotesTreeRow:
    if range_copy and action_copy:
        label = f"{range_copy}  {action_copy}"
    elif status_copy and action_copy:
        label = f"{status_copy} · {action_copy}"
    else:
        label = range_copy or status_copy or action_copy
    boundary_identity = _pager_boundary_identity(state, action, retry_direction)
    return LibraryNotesTreeRow(
        placement_id=f"pager:{state.pager_id}:{boundary_identity}",
        kind="pager",
        label=label,
        depth=depth,
        parent_folder_id=state.key.parent_id,
        content_kind=state.key.slice_kind,
        paging_action=action,
        retry_direction=retry_direction,
        range_copy=range_copy,
        action_copy=action_copy,
        status_text=status_copy,
        focus_id=_pager_focus_id(state.key, boundary_identity),
        loading=loading,
        disabled=disabled,
    )


def _slice_range_copy(state: NotesBranchSliceState) -> str:
    if state.total is None or not state.items:
        return ""
    noun = "Folders" if state.key.slice_kind == "folders" else "Notes"
    first = state.start_offset + 1
    last = state.start_offset + len(state.items)
    return f"{noun} {first}–{last} of {state.total}"


def _slice_pager_rows(
    state: NotesBranchSliceState, *, depth: int
) -> tuple[LibraryNotesTreeRow, ...]:
    count = len(state.items)
    stale_noun = "folder" if state.key.slice_kind == "folders" else "placement"
    load_noun = "folders" if state.key.slice_kind == "folders" else "notes"
    range_copy = _slice_range_copy(state)

    if state.freshness == "stale":
        status = (
            f"{count} {stale_noun if count == 1 else stale_noun + 's'} loaded"
            " · May be out of date"
        )
        if state.loading:
            return (
                _pager_row(
                    state,
                    action="retry",
                    depth=depth,
                    status_copy=status,
                    action_copy="Loading…",
                    retry_direction=state.requested_direction,
                    disabled=True,
                    loading=True,
                ),
            )
        return (
            _pager_row(
                state,
                action="retry",
                depth=depth,
                status_copy=status,
                action_copy="Retry",
                retry_direction=state.failed_direction,
            ),
        )

    if state.recovery_attempted:
        return (
            _pager_row(
                state,
                action="retry",
                depth=depth,
                status_copy="Tree changed · Refreshing…",
                disabled=True,
                loading=True,
            ),
        )

    has_earlier = state.previous_offset is not None
    has_more = state.next_offset is not None
    failed_action: LibraryNotesTreePagingAction | None = None
    if state.error:
        if state.failed_direction == "previous":
            failed_action = "earlier"
        elif state.failed_direction == "more":
            failed_action = "more"
        else:
            failed_action = "retry"

    if state.error and failed_action == "retry":
        initial_copy = (
            "Couldn’t load contents"
            if state.key.parent_id is not None
            else "Couldn’t load folders"
            if state.key.slice_kind == "folders"
            else "Couldn’t load notes"
        )
        return (
            _pager_row(
                state,
                action="retry",
                depth=depth,
                status_copy=initial_copy,
                action_copy="Retry",
                retry_direction=state.failed_direction,
            ),
        )

    rows: list[LibraryNotesTreeRow] = []
    for action, available, action_copy in (
        ("earlier", has_earlier, "Load earlier"),
        ("more", has_more, f"Load more {load_noun}"),
    ):
        if not available:
            continue
        if state.error and action == failed_action:
            rows.append(
                _pager_row(
                    state,
                    action="retry",
                    depth=depth,
                    status_copy=(
                        "Couldn’t load more"
                        if action == "more"
                        else "Couldn’t load earlier"
                    ),
                    action_copy="Retry",
                    retry_direction=state.failed_direction,
                )
            )
            continue
        active_loading = state.loading and state.requested_direction == (
            "previous" if action == "earlier" else "more"
        )
        rows.append(
            _pager_row(
                state,
                action=action,  # type: ignore[arg-type]
                depth=depth,
                range_copy=range_copy,
                action_copy="Loading…" if active_loading else action_copy,
                disabled=active_loading,
                loading=active_loading,
            )
        )

    if state.loading and not rows:
        loading_copy = (
            f"{range_copy}  Loading…" if range_copy else f"Loading {load_noun}…"
        )
        return (
            _pager_row(
                state,
                action="retry",
                depth=depth,
                status_copy=loading_copy,
                disabled=True,
                loading=True,
            ),
        )
    return tuple(rows)


def build_paged_library_notes_tree(
    *,
    branch_states: Mapping[NotesBranchKey, NotesBranchSliceState],
    expanded_folder_ids: set[str] | frozenset[str],
    protected_folder_ids: frozenset[str] = frozenset(),
    inactive_managed_folder_ids: frozenset[str] = frozenset(),
) -> LibraryNotesTreeProjection:
    """Project independently loaded parent-keyed slices into one visible tree.

    Only supplied branch slices are projected. Expanded identities control
    recursion, and every continuation remains at the boundary it extends.
    """
    for key, state in branch_states.items():
        if key != state.key:
            raise ValueError("branch state key does not match its mapping key")

    folders = {
        folder.folder_id: folder
        for state in branch_states.values()
        if state.key.slice_kind == "folders"
        for folder in state.items
        if isinstance(folder, NoteFolder) and not folder.deleted
    }
    protected_folder_ids = protected_folder_ids.union(inactive_managed_folder_ids)

    def folder_row(
        folder: NoteFolder, *, depth: int, unsafe: bool
    ) -> LibraryNotesTreeRow:
        protected = folder.folder_id in protected_folder_ids
        owner_active = folder.folder_id not in inactive_managed_folder_ids
        if protected and owner_active:
            semantic_status: LibraryNotesTreeSemanticStatus = "connected"
            status_text = "⇄ Sync managed"
        elif protected:
            semantic_status = "needs_attention"
            status_text = "! Needs owner review"
        else:
            semantic_status = "normal"
            status_text = ""
        return LibraryNotesTreeRow(
            placement_id=FolderPlacementId.folder(folder.folder_id),
            kind="folder",
            label=folder.name,
            depth=depth,
            folder_id=folder.folder_id,
            breadcrumb=folder.path.strip("/").replace("/", " / "),
            ownership="managed" if protected else None,
            owner_active=owner_active,
            protected=protected,
            semantic_status=semantic_status,
            status_text=status_text,
            expanded=folder.folder_id in expanded_folder_ids,
            version=folder.version,
            unsafe_mutation_disabled=unsafe,
        )

    rows: list[LibraryNotesTreeRow] = []

    def append_branch(parent_id: str, depth: int) -> None:
        folder_state = branch_states.get(NotesBranchKey(parent_id, "folders"))
        if folder_state is not None:
            unsafe = folder_state.freshness == "stale"
            for item in folder_state.items:
                if not isinstance(item, NoteFolder) or item.deleted:
                    continue
                rows.append(folder_row(item, depth=depth, unsafe=unsafe))
                if item.folder_id in expanded_folder_ids:
                    append_branch(item.folder_id, depth + 1)
            rows.extend(_slice_pager_rows(folder_state, depth=depth))

        placement_state = branch_states.get(NotesBranchKey(parent_id, "placements"))
        if placement_state is None:
            return
        unsafe = placement_state.freshness == "stale"
        folder = folders.get(parent_id)
        if folder is not None:
            for item in placement_state.items:
                if not isinstance(item, NotePlacementRecord):
                    continue
                membership = item.membership
                assert membership is not None
                rows.append(
                    _note_row(
                        note=item.note,
                        folder=folder,
                        membership=membership,
                        depth=depth,
                        unsafe_mutation_disabled=unsafe,
                    )
                )
        rows.extend(_slice_pager_rows(placement_state, depth=depth))

    root_folders = branch_states.get(NotesBranchKey(None, "folders"))
    if root_folders is not None:
        unsafe = root_folders.freshness == "stale"
        for item in root_folders.items:
            if not isinstance(item, NoteFolder) or item.deleted:
                continue
            rows.append(folder_row(item, depth=0, unsafe=unsafe))
            if item.folder_id in expanded_folder_ids:
                append_branch(item.folder_id, 1)
        rows.extend(_slice_pager_rows(root_folders, depth=0))

    root_placements = branch_states.get(NotesBranchKey(None, "placements"))
    if root_placements is not None:
        placement_pagers = _slice_pager_rows(root_placements, depth=1)
        if root_placements.items or placement_pagers:
            rows.append(
                LibraryNotesTreeRow(
                    placement_id=UNFILED_PLACEMENT_ID,
                    kind="unfiled",
                    label="Unfiled",
                    depth=0,
                    breadcrumb="Unfiled",
                    expanded=True,
                    unsafe_mutation_disabled=(root_placements.freshness == "stale"),
                )
            )
        for item in root_placements.items:
            if isinstance(item, NotePlacementRecord) and item.folder_id is None:
                rows.append(
                    _note_row(
                        note=item.note,
                        folder=None,
                        membership=None,
                        depth=1,
                        unsafe_mutation_disabled=(root_placements.freshness == "stale"),
                    )
                )
        rows.extend(placement_pagers)

    return LibraryNotesTreeProjection(rows=tuple(rows))


def patch_notes_filter_state_title(
    state: LibraryNotesFilterState,
    *,
    note_id: str,
    title: str,
    modified_at: str | None = None,
) -> LibraryNotesFilterState:
    """Return ``state`` with any filter-window placement of ``note_id`` retitled.

    While a filter is active the tree renders from this independent window
    rather than the browse branches, so the save-time title patch must
    reach here too (task-31796). Returns the original state unchanged when
    nothing matched.
    """
    placements, changed = retitle_note_placements(
        state.placements, note_id=note_id, title=title, modified_at=modified_at
    )
    if not changed:
        return state
    return replace(state, placements=placements)


def build_filtered_library_notes_tree(
    state: LibraryNotesFilterState,
) -> LibraryNotesTreeProjection:
    """Project one exact filter page without touching browse branch state."""
    folders = {folder.folder_id: folder for folder in state.ancestor_folders}
    rows: list[LibraryNotesTreeRow] = []
    rendered_folders: set[str] = set()
    unfiled_rendered = False

    def folder_chain(folder_id: str) -> tuple[NoteFolder, ...]:
        chain: list[NoteFolder] = []
        seen: set[str] = set()
        current = folders.get(folder_id)
        while current is not None and current.folder_id not in seen:
            seen.add(current.folder_id)
            chain.append(current)
            current = (
                folders.get(current.parent_id)
                if current.parent_id is not None
                else None
            )
        return tuple(reversed(chain))

    for placement in state.placements:
        if placement.folder_id is None:
            if not unfiled_rendered:
                rows.append(
                    LibraryNotesTreeRow(
                        placement_id=UNFILED_PLACEMENT_ID,
                        kind="unfiled",
                        label="Unfiled",
                        depth=0,
                        breadcrumb="Unfiled",
                        expanded=True,
                        unsafe_mutation_disabled=state.stale,
                    )
                )
                unfiled_rendered = True
            rows.append(
                _note_row(
                    note=placement.note,
                    folder=None,
                    membership=None,
                    depth=1,
                    unsafe_mutation_disabled=state.stale,
                )
            )
            continue

        chain = folder_chain(placement.folder_id)
        for depth, folder in enumerate(chain):
            if folder.folder_id in rendered_folders:
                continue
            rendered_folders.add(folder.folder_id)
            rows.append(
                LibraryNotesTreeRow(
                    placement_id=FolderPlacementId.folder(folder.folder_id),
                    kind="folder",
                    label=folder.name,
                    depth=depth,
                    folder_id=folder.folder_id,
                    breadcrumb=folder.path.strip("/").replace("/", " / "),
                    expanded=True,
                    version=folder.version,
                    unsafe_mutation_disabled=state.stale,
                )
            )
        membership = placement.membership
        folder = folders.get(placement.folder_id)
        if membership is None or folder is None:
            continue
        rows.append(
            _note_row(
                note=placement.note,
                folder=folder,
                membership=membership,
                depth=max(1, len(chain)),
                unsafe_mutation_disabled=state.stale,
            )
        )

    pager_state = NotesBranchSliceState(
        key=NotesBranchKey(None, "placements"),
        items=state.placements,
        item_ids=tuple(row.placement_id for row in rows if row.kind == "note"),
        total=state.total,
        start_offset=state.start_offset,
        previous_offset=state.previous_offset,
        next_offset=state.next_offset,
        generation=state.generation,
        topology_epoch=state.topology_epoch,
        freshness="stale" if state.stale else "fresh",
        loading=state.loading,
        recovery_attempted=state.recovery_attempted,
        requested_direction=state.requested_direction,
        requested_offset=state.requested_offset,
        requested_limit=state.requested_limit,
        failed_direction=state.failed_direction,
        error=state.error,
    )
    rows.extend(_slice_pager_rows(pager_state, depth=0))
    return LibraryNotesTreeProjection(rows=tuple(rows))


def reconcile_library_notes_tree_identity(
    projection: LibraryNotesTreeProjection,
    identity: LibraryNotesTreeIdentity | None,
) -> LibraryNotesTreeIdentity | None:
    """Keep a placement, then its note, otherwise the first surviving row."""
    if identity is None:
        return None
    exact = projection.row(identity.placement_id)
    if exact is not None:
        return LibraryNotesTreeIdentity(exact.placement_id, exact.note_id)
    if identity.note_id:
        same_note = next(
            (row for row in projection.rows if row.note_id == identity.note_id), None
        )
        if same_note is not None:
            return LibraryNotesTreeIdentity(same_note.placement_id, same_note.note_id)
    fallback = next((row for row in projection.rows if row.kind == "note"), None)
    if fallback is None:
        return None
    return LibraryNotesTreeIdentity(fallback.placement_id, fallback.note_id)
