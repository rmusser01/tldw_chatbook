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


def _folder_is_ancestor(
    ancestor_id: str,
    descendant_id: str,
    folders: Mapping[str, NoteFolder],
) -> bool:
    current = folders.get(descendant_id)
    seen: set[str] = set()
    while current is not None and current.parent_id is not None:
        if current.parent_id == ancestor_id:
            return True
        if current.parent_id in seen:
            return False
        seen.add(current.parent_id)
        current = folders.get(current.parent_id)
    return False


def _effective_memberships(
    memberships: tuple[NoteFolderMembership, ...],
    folders: Mapping[str, NoteFolder],
) -> tuple[NoteFolderMembership, ...]:
    """Collapse generated ancestor placements for the same note and owner."""
    effective: list[NoteFolderMembership] = []
    for candidate in memberships:
        if candidate.ownership == "managed" and any(
            other.membership_id != candidate.membership_id
            and other.ownership == "managed"
            and other.note_id == candidate.note_id
            and other.owner_id == candidate.owner_id
            and _folder_is_ancestor(candidate.folder_id, other.folder_id, folders)
            for other in memberships
        ):
            continue
        effective.append(candidate)
    return tuple(effective)


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
) -> LibraryNotesTreeProjection:
    """Project bounded root/expanded batches into one lazy visible tree."""
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

    def add_folder(folder: NoteFolder, depth: int) -> None:
        expanded = folder.folder_id in expanded_folder_ids
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
        rows.append(
            LibraryNotesTreeRow(
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
        )
        if not expanded:
            return
        for child in children.get(folder.folder_id, ()):
            add_folder(child, depth + 1)
        for membership in memberships_by_folder.get(folder.folder_id, ()):
            note = notes.get(membership.note_id)
            if note is not None:
                note_row = _note_row(
                    note=note,
                    folder=folder,
                    membership=membership,
                    depth=depth + 1,
                )
                if not query or query in note_row.breadcrumb.casefold():
                    rows.append(note_row)

    for root in children.get(None, ()):
        add_folder(root, 0)

    unfiled_notes = [
        note
        for note in sorted(
            root_page.notes,
            key=lambda note: (_record_title(note).casefold(), _record_id(note)),
        )
        if not query or query in f"Unfiled / {_record_title(note)}".casefold()
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
