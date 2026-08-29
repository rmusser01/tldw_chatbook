"""Pure, placement-aware display state for the Database Notes folder tree."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NoteFolder,
    NoteFolderMembership,
    NoteFolderPage,
    NotePlacementRecord,
)
from tldw_chatbook.Library.library_notes_tree_paging import (
    NotesBranchKey,
    NotesBranchSliceState,
    NotesSliceKind,
)

LibraryNotesTreeRowKind = Literal["folder", "note", "unfiled", "pager"]
LibraryNotesTreeSemanticStatus = Literal["normal", "connected", "needs_attention"]
LibraryNotesTreePagingAction = Literal["earlier", "more", "retry"]

UNFILED_PLACEMENT_ID = "virtual:unfiled"


def empty_note_folder_page() -> NoteFolderPage:
    """Return an empty bounded page for an unloaded branch set."""
    return NoteFolderPage(
        folders=(),
        memberships=(),
        notes=(),
        total_folders=0,
        total_notes=0,
        next_offset=None,
    )


def merge_note_folder_pages(
    base: NoteFolderPage, incoming: NoteFolderPage
) -> NoteFolderPage:
    """Merge continued bounded pages by stable storage identity."""
    folders = {folder.folder_id: folder for folder in base.folders}
    folders.update({folder.folder_id: folder for folder in incoming.folders})
    memberships = {
        membership.membership_id: membership for membership in base.memberships
    }
    memberships.update(
        {membership.membership_id: membership for membership in incoming.memberships}
    )
    notes = {_record_id(note): note for note in base.notes}
    notes.update({_record_id(note): note for note in incoming.notes})
    return NoteFolderPage(
        folders=tuple(folders.values()),
        memberships=tuple(memberships.values()),
        notes=tuple(notes.values()),
        total_folders=max(base.total_folders, incoming.total_folders),
        total_notes=max(base.total_notes, incoming.total_notes),
        next_offset=incoming.next_offset,
        next_folder_offset=incoming.next_folder_offset,
        total_memberships=max(base.total_memberships, incoming.total_memberships),
        next_membership_offset=incoming.next_membership_offset,
        managed_folder_ids=tuple(
            sorted({*base.managed_folder_ids, *incoming.managed_folder_ids})
        ),
        inactive_managed_folder_ids=tuple(
            sorted(
                {
                    *base.inactive_managed_folder_ids,
                    *incoming.inactive_managed_folder_ids,
                }
            )
        ),
        unfiled_note_ids=(
            tuple(
                sorted(
                    {
                        *(base.unfiled_note_ids or ()),
                        *(incoming.unfiled_note_ids or ()),
                    }
                )
            )
            if base.unfiled_note_ids is not None
            or incoming.unfiled_note_ids is not None
            else None
        ),
    )


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
    """Visible rows plus legacy aggregate cursors retained through cutover."""

    rows: tuple[LibraryNotesTreeRow, ...]
    next_folder_offset: int | None = None
    next_note_offset: int | None = None
    next_membership_offset: int | None = None

    @property
    def has_more(self) -> bool:
        """Return whether any bounded page has another cursor."""
        return any(
            cursor is not None
            for cursor in (
                self.next_folder_offset,
                self.next_note_offset,
                self.next_membership_offset,
            )
        )

    def row(self, placement_id: str) -> LibraryNotesTreeRow | None:
        """Return one visible row by exact placement identity."""
        return next(
            (row for row in self.rows if row.placement_id == placement_id), None
        )


def _record_id(note: Mapping[str, object]) -> str:
    return str(note.get("id", note.get("note_id", "")) or "")


def _record_title(note: Mapping[str, object]) -> str:
    return str(note.get("title", "") or "Untitled")


def _effective_memberships(
    memberships: tuple[NoteFolderMembership, ...],
    folders: Mapping[str, NoteFolder],
) -> tuple[NoteFolderMembership, ...]:
    """Collapse generated ancestor placements for the same note and owner."""
    managed_folders_by_owner: dict[tuple[str, str], set[str]] = {}
    for membership in memberships:
        if membership.ownership == "managed":
            managed_folders_by_owner.setdefault(
                (membership.note_id, membership.owner_id), set()
            ).add(membership.folder_id)

    managed_folder_ids = (
        set().union(*managed_folders_by_owner.values())
        if (managed_folders_by_owner)
        else set()
    )
    ancestors_by_folder: dict[str, frozenset[str]] = {}
    for folder_id in managed_folder_ids:
        if folder_id in ancestors_by_folder:
            continue
        chain: list[tuple[str, str]] = []
        seen: set[str] = set()
        current_id = folder_id
        while current_id not in ancestors_by_folder:
            if current_id in seen:
                chain.clear()
                ancestors_by_folder[folder_id] = frozenset()
                break
            seen.add(current_id)
            current = folders.get(current_id)
            if current is None or current.parent_id is None:
                ancestors_by_folder[current_id] = frozenset()
                break
            chain.append((current_id, current.parent_id))
            current_id = current.parent_id
        ancestor_ids = ancestors_by_folder.get(current_id, frozenset())
        for child_id, parent_id in reversed(chain):
            ancestor_ids = ancestor_ids.union((parent_id,))
            ancestors_by_folder[child_id] = ancestor_ids

    shadowed_folders_by_owner: dict[tuple[str, str], set[str]] = {}
    for owner_key, folder_ids in managed_folders_by_owner.items():
        shadowed = shadowed_folders_by_owner.setdefault(owner_key, set())
        for folder_id in folder_ids:
            shadowed.update(ancestors_by_folder[folder_id].intersection(folder_ids))

    return tuple(
        membership
        for membership in memberships
        if membership.ownership != "managed"
        or membership.folder_id
        not in shadowed_folders_by_owner[(membership.note_id, membership.owner_id)]
    )


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


def _pager_focus_id(key: NotesBranchKey, action: LibraryNotesTreePagingAction) -> str:
    parent = (
        "root"
        if key.parent_id is None
        else f"folder-{key.parent_id.encode('utf-8').hex()}"
    )
    return f"library-notes-tree-pager-{parent}-{key.slice_kind}-{action}"


def _pager_row(
    state: NotesBranchSliceState,
    *,
    action: LibraryNotesTreePagingAction,
    depth: int,
    range_copy: str = "",
    status_copy: str = "",
    action_copy: str = "",
    disabled: bool = False,
    loading: bool = False,
) -> LibraryNotesTreeRow:
    if range_copy and action_copy:
        label = f"{range_copy}  {action_copy}"
    elif status_copy and action_copy:
        label = f"{status_copy} · {action_copy}"
    else:
        label = range_copy or status_copy or action_copy
    return LibraryNotesTreeRow(
        placement_id=f"pager:{state.pager_id}:{action}",
        kind="pager",
        label=label,
        depth=depth,
        parent_folder_id=state.key.parent_id,
        content_kind=state.key.slice_kind,
        paging_action=action,
        range_copy=range_copy,
        action_copy=action_copy,
        status_text=status_copy,
        focus_id=_pager_focus_id(state.key, action),
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
        return (
            _pager_row(
                state,
                action="retry",
                depth=depth,
                status_copy=status,
                action_copy="Retry",
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
        failed_action = "more" if has_more else "earlier" if has_earlier else "retry"

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

    if state.error and not rows:
        initial_copy = (
            "Couldn’t load folders"
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
            ),
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
    placement_records = tuple(
        item
        for state in branch_states.values()
        if state.key.slice_kind == "placements"
        for item in state.items
        if isinstance(item, NotePlacementRecord)
    )
    memberships = _effective_memberships(
        tuple(
            item.membership for item in placement_records if item.membership is not None
        ),
        folders,
    )
    effective_membership_ids = {membership.membership_id for membership in memberships}
    managed_folder_active: dict[str, bool] = {}
    for membership in memberships:
        if membership.ownership != "managed":
            continue
        folder_id: str | None = membership.folder_id
        seen: set[str] = set()
        while folder_id is not None and folder_id not in seen:
            seen.add(folder_id)
            managed_folder_active[folder_id] = (
                managed_folder_active.get(folder_id, True) and membership.owner_active
            )
            folder = folders.get(folder_id)
            folder_id = folder.parent_id if folder is not None else None

    def folder_row(
        folder: NoteFolder, *, depth: int, unsafe: bool
    ) -> LibraryNotesTreeRow:
        protected = folder.folder_id in managed_folder_active
        owner_active = managed_folder_active.get(folder.folder_id, True)
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
                if membership is None or (
                    membership.membership_id not in effective_membership_ids
                ):
                    continue
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


def build_library_notes_tree(
    *,
    root_page: NoteFolderPage,
    expanded_page: NoteFolderPage,
    expanded_folder_ids: set[str] | frozenset[str],
    filter_text: str = "",
    matched_note_ids: frozenset[str] | None = None,
) -> LibraryNotesTreeProjection:
    """Project bounded root and expanded batches into one lazy visible tree.

    Args:
        root_page: Bounded root-folder and unfiled-note page.
        expanded_page: Bounded children, memberships, and notes for expanded folders.
        expanded_folder_ids: Folder identifiers whose immediate contents are visible.
        filter_text: Optional case-insensitive folder/note filter.
        matched_note_ids: Optional authoritative note IDs from bounded search.

    Returns:
        Visible placement-aware rows and the cursors needed to continue loading.
    """
    query = filter_text.strip().casefold()
    folders = {
        folder.folder_id: folder
        for folder in (*root_page.folders, *expanded_page.folders)
        if not folder.deleted
    }
    notes = {
        _record_id(note): note
        for note in (*root_page.notes, *expanded_page.notes)
        if _record_id(note)
    }
    memberships = _effective_memberships(expanded_page.memberships, folders)
    memberships_by_folder: dict[str, list[NoteFolderMembership]] = {}
    for membership in memberships:
        memberships_by_folder.setdefault(membership.folder_id, []).append(membership)
    managed_ids = {
        *root_page.managed_folder_ids,
        *expanded_page.managed_folder_ids,
    }
    inactive_managed_ids = {
        *root_page.inactive_managed_folder_ids,
        *expanded_page.inactive_managed_folder_ids,
    }
    managed_folder_active: dict[str, bool] = {
        folder_id: folder_id not in inactive_managed_ids for folder_id in managed_ids
    }
    for membership in memberships:
        if membership.ownership != "managed":
            continue
        folder_id: str | None = membership.folder_id
        seen: set[str] = set()
        while folder_id is not None and folder_id not in seen:
            seen.add(folder_id)
            managed_folder_active[folder_id] = (
                managed_folder_active.get(folder_id, True) and membership.owner_active
            )
            folder = folders.get(folder_id)
            folder_id = folder.parent_id if folder is not None else None
    children: dict[str | None, list[NoteFolder]] = {}
    for folder in folders.values():
        children.setdefault(folder.parent_id, []).append(folder)
    for group in children.values():
        group.sort(key=lambda folder: (folder.normalized_path, folder.folder_id))
    for group in memberships_by_folder.values():
        group.sort(
            key=lambda membership: (
                _record_title(notes.get(membership.note_id, {})).casefold(),
                membership.note_id,
                membership.membership_id,
            )
        )

    rows: list[LibraryNotesTreeRow] = []

    def folder_rows(folder: NoteFolder, depth: int) -> list[LibraryNotesTreeRow]:
        expanded = folder.folder_id in expanded_folder_ids or bool(query)
        protected = folder.folder_id in managed_folder_active
        owner_active = managed_folder_active.get(folder.folder_id, True)
        semantic_status: LibraryNotesTreeSemanticStatus = "normal"
        status_text = ""
        if protected and owner_active:
            semantic_status = "connected"
            status_text = "⇄ Sync managed"
        elif protected:
            semantic_status = "needs_attention"
            status_text = "! Needs owner review"
        folder_row = LibraryNotesTreeRow(
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
            expanded=expanded,
            version=folder.version,
        )
        if not expanded:
            return [folder_row]
        descendant_rows: list[LibraryNotesTreeRow] = []
        for child in children.get(folder.folder_id, ()):
            descendant_rows.extend(folder_rows(child, depth + 1))
        for membership in memberships_by_folder.get(folder.folder_id, ()):
            note = notes.get(membership.note_id)
            if note is not None:
                note_row = _note_row(
                    note=note,
                    folder=folder,
                    membership=membership,
                    depth=depth + 1,
                )
                if not query or (
                    membership.note_id in matched_note_ids
                    if matched_note_ids is not None
                    else query in note_row.breadcrumb.casefold()
                ):
                    descendant_rows.append(note_row)
        folder_matches = (
            matched_note_ids is None and query in folder_row.breadcrumb.casefold()
        )
        if query and not descendant_rows and not folder_matches:
            return []
        return [folder_row, *descendant_rows]

    for root in children.get(None, ()):
        rows.extend(folder_rows(root, 0))

    unfiled_note_ids = (
        {_record_id(note) for note in root_page.notes}
        if root_page.unfiled_note_ids is None
        else set(root_page.unfiled_note_ids)
    )
    unfiled_notes = [
        note
        for note in sorted(
            root_page.notes,
            key=lambda note: (_record_title(note).casefold(), _record_id(note)),
        )
        if _record_id(note) in unfiled_note_ids
        if not query
        or (
            _record_id(note) in matched_note_ids
            if matched_note_ids is not None
            else query in f"Unfiled / {_record_title(note)}".casefold()
        )
    ]
    if unfiled_notes or root_page.next_offset is not None:
        rows.append(
            LibraryNotesTreeRow(
                placement_id=UNFILED_PLACEMENT_ID,
                kind="unfiled",
                label="Unfiled",
                depth=0,
                breadcrumb="Unfiled",
                expanded=True,
            )
        )
        rows.extend(
            _note_row(note=note, folder=None, membership=None, depth=1)
            for note in unfiled_notes
        )

    return LibraryNotesTreeProjection(
        rows=tuple(rows),
        next_folder_offset=(
            expanded_page.next_folder_offset or root_page.next_folder_offset
        ),
        next_note_offset=expanded_page.next_offset or root_page.next_offset,
        next_membership_offset=expanded_page.next_membership_offset,
    )


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
