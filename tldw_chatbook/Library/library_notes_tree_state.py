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
)

LibraryNotesTreeRowKind = Literal["folder", "note", "unfiled"]
LibraryNotesTreeSemanticStatus = Literal["normal", "connected", "needs_attention"]

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
    """One visible folder, virtual Unfiled, or note-placement row."""

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


@dataclass(frozen=True)
class LibraryNotesTreeIdentity:
    """Stable navigator identity: placement first, underlying note second."""

    placement_id: str
    note_id: str | None = None


@dataclass(frozen=True)
class LibraryNotesTreeProjection:
    """Visible rows plus the bounded cursors needed to continue loading."""

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
    )


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
