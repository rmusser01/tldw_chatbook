"""Pure Library shell rail state builders."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_chatbook.Constants import TAB_INGEST, TAB_MEDIA, TAB_NOTES

LIBRARY_CANVAS_LANDING_COPY = "Search, pick a content type, or ingest something new."

LIBRARY_ROW_BROWSE_CONVERSATIONS = "browse-conversations"


@dataclass(frozen=True)
class LibraryRailRow:
    """One selectable row in the Library shell rail."""

    row_id: str
    section_id: str
    title: str
    target_kind: str
    target_id: str
    count: int | None = None
    count_known: bool = True


@dataclass(frozen=True)
class LibraryRailSectionState:
    """One Library shell rail section with its rows."""

    section_id: str
    title: str
    rows: tuple[LibraryRailRow, ...]


@dataclass(frozen=True)
class LibraryShellInput:
    """Adapter-provided input for Library shell state building."""

    media_count: int | None = None
    media_known: bool = True
    conversations_count: int | None = None
    conversations_known: bool = True
    notes_count: int | None = None
    notes_known: bool = True
    collections_count: int | None = None
    collections_known: bool = True
    runtime_source: str = "local"
    server_label: str | None = None
    details_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class LibraryShellState:
    """Full Library shell display state: header, rail sections, canvas."""

    header_line: str
    sections: tuple[LibraryRailSectionState, ...]
    details_lines: tuple[str, ...]
    selected_row_id: str
    canvas_kind: str
    canvas_target: str
    canvas_empty_copy: str


def build_library_shell_state(
    state: LibraryShellInput, *, selected_row_id: str = ""
) -> LibraryShellState:
    """Build the Library shell rail + canvas display state.

    Args:
        state: Adapter-provided Library shell input.
        selected_row_id: Explicit row selection; defaults to empty (landing canvas).

    Returns:
        Immutable shell state: header line, rail sections, details lines,
        and the canvas for the selected row (or landing canvas when
        nothing is selectable).
    """
    # Build the fixed row table
    browse_rows = (
        LibraryRailRow(
            row_id="browse-media",
            section_id="browse",
            title="Media",
            target_kind="screen",
            target_id=TAB_MEDIA,
            count=state.media_count,
            count_known=state.media_known,
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_BROWSE_CONVERSATIONS,
            section_id="browse",
            title="Conversations",
            target_kind="canvas",
            target_id="conversations",
            count=state.conversations_count,
            count_known=state.conversations_known,
        ),
        LibraryRailRow(
            row_id="browse-notes",
            section_id="browse",
            title="Notes",
            target_kind="screen",
            target_id=TAB_NOTES,
            count=state.notes_count,
            count_known=state.notes_known,
        ),
        LibraryRailRow(
            row_id="browse-collections",
            section_id="browse",
            title="Collections",
            target_kind="mode",
            target_id="collections",
            count=state.collections_count,
            count_known=state.collections_known,
        ),
        LibraryRailRow(
            row_id="browse-search",
            section_id="browse",
            title="Search / RAG",
            target_kind="mode",
            target_id="search",
            count=None,
            count_known=True,
        ),
    )

    create_rows = (
        LibraryRailRow(
            row_id="create-note",
            section_id="create",
            title="New note",
            target_kind="screen",
            target_id=TAB_NOTES,
            count=None,
            count_known=True,
        ),
        LibraryRailRow(
            row_id="create-study",
            section_id="create",
            title="Study decks",
            target_kind="mode",
            target_id="study",
            count=None,
            count_known=True,
        ),
        LibraryRailRow(
            row_id="create-flashcards",
            section_id="create",
            title="Flashcards",
            target_kind="mode",
            target_id="flashcards",
            count=None,
            count_known=True,
        ),
        LibraryRailRow(
            row_id="create-quizzes",
            section_id="create",
            title="Quizzes",
            target_kind="mode",
            target_id="quizzes",
            count=None,
            count_known=True,
        ),
    )

    ingest_rows = (
        LibraryRailRow(
            row_id="ingest-import-media",
            section_id="ingest",
            title="Import media",
            target_kind="screen",
            target_id=TAB_INGEST,
            count=None,
            count_known=True,
        ),
        LibraryRailRow(
            row_id="ingest-import-export",
            section_id="ingest",
            title="Import / Export",
            target_kind="mode",
            target_id="import-export",
            count=None,
            count_known=True,
        ),
    )

    sections = (
        LibraryRailSectionState(section_id="browse", title="Browse", rows=browse_rows),
        LibraryRailSectionState(section_id="create", title="Create", rows=create_rows),
        LibraryRailSectionState(section_id="ingest", title="Ingest", rows=ingest_rows),
    )

    # Build header line
    runtime_source = str(state.runtime_source or "local").strip().lower()
    if runtime_source == "server":
        server_label = str(state.server_label or "unknown").strip()
        header_line = f"Library | Server: {server_label}"
    else:
        header_line = "Library | Local"

    # Resolve canvas by selection
    all_rows = {row.row_id: row for section in sections for row in section.rows}
    selected_row = all_rows.get(selected_row_id)

    if selected_row is None:
        # No valid selection; use landing canvas
        canvas_kind = "empty"
        canvas_target = ""
        canvas_empty_copy = LIBRARY_CANVAS_LANDING_COPY
    elif selected_row.target_kind == "canvas":
        # Canvas rows resolve to conversations canvas
        canvas_kind = "conversations"
        canvas_target = ""
        canvas_empty_copy = LIBRARY_CANVAS_LANDING_COPY
    elif selected_row.target_kind == "mode":
        # Mode rows resolve to mode canvas
        canvas_kind = "mode"
        canvas_target = selected_row.target_id
        canvas_empty_copy = LIBRARY_CANVAS_LANDING_COPY
    else:
        # Screen rows and others resolve to empty canvas
        canvas_kind = "empty"
        canvas_target = ""
        canvas_empty_copy = LIBRARY_CANVAS_LANDING_COPY

    return LibraryShellState(
        header_line=header_line,
        sections=sections,
        details_lines=state.details_lines,
        selected_row_id=selected_row_id if selected_row is not None else "",
        canvas_kind=canvas_kind,
        canvas_target=canvas_target,
        canvas_empty_copy=canvas_empty_copy,
    )
