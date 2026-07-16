"""Pure Library shell rail state builders."""

from __future__ import annotations

from dataclasses import dataclass

LIBRARY_CANVAS_LANDING_COPY = "Search, pick a content type, or ingest something new."

LIBRARY_ROW_BROWSE_CONVERSATIONS = "browse-conversations"
LIBRARY_ROW_BROWSE_MEDIA = "browse-media"
LIBRARY_ROW_BROWSE_NOTES = "browse-notes"
# Value follows the sibling "browse-*" convention: the rail widget renders
# the row's DOM id as LIBRARY_RAIL_ROW_PREFIX ("library-row-") + row_id,
# i.e. "#library-row-browse-prompts" -- the id the task brief names.
LIBRARY_ROW_BROWSE_PROMPTS = "browse-prompts"
# Skills sub-project Task 1: same "browse-*" convention, rendered right after
# Prompts in the Browse section -- "#library-row-browse-skills". Until the
# Skills canvas lands (Task 3), the row is inert-but-selectable: pressing it
# just selects the row and falls through to the shell's generic empty-canvas
# landing path (no ``elif shell.canvas_kind == "skills"`` branch exists yet
# in library_screen.py's compose).
LIBRARY_ROW_BROWSE_SKILLS = "browse-skills"
LIBRARY_ROW_BROWSE_SEARCH = "browse-search"
LIBRARY_ROW_BROWSE_COLLECTIONS = "browse-collections"
LIBRARY_ROW_CREATE_NOTE = "create-note"
# Task 8b D1: "New prompt" -- unlike LIBRARY_ROW_CREATE_NOTE (its own
# "notes-create" canvas kind, a landing chooser of Blank/template rows),
# this row's target_id is "prompts" itself: it reuses the SAME canvas kind
# Browse > Prompts targets. The screen distinguishes "opened via Browse" vs
# "opened via New prompt" by view/selection state
# (`_library_prompts_view == "editor"` plus a `prompt_id=None` sentinel),
# not by a separate canvas kind -- see library_screen.py's
# `_enter_library_prompt_create_editor`.
LIBRARY_ROW_CREATE_PROMPT = "create-prompt"
# Skills sub-project (skills-200 spec, "Create > New skill"): same shape as
# LIBRARY_ROW_CREATE_PROMPT above -- its target_id is "skills" itself (the
# SAME canvas kind Browse > Skills targets), not a dedicated "skills-create"
# canvas kind. The screen distinguishes "opened via Browse" vs "opened via
# New skill" by ``_selected_skill_name`` being empty (the same sentinel
# ``_save_library_skill``'s ``is_create`` already reads) -- see
# library_screen.py's ``_enter_library_skill_create_editor``.
LIBRARY_ROW_CREATE_SKILL = "create-skill"
LIBRARY_ROW_INGEST_MEDIA = "ingest-import-media"
LIBRARY_ROW_INGEST_EXPORT = "ingest-export"

# Export packages local DB content directly (the chatbook creator reads
# local DBs, never a server) -- when a server source is active, the row
# renders disabled with this tooltip rather than offering a control that
# would silently export nothing (or the wrong content). Mirrors the
# scope-service gating pattern (F4 design spec, "Entry points").
LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP = "Export packages local content only."


@dataclass(frozen=True)
class LibraryRailRow:
    """One selectable row in the Library shell rail.

    Attributes:
        disabled: Whether this row's rail button should render disabled
            (unclickable). Only the Export row uses this today (server-
            mode gating) -- every other row is always enabled.
        disabled_tooltip: The tooltip shown while ``disabled`` is
            ``True``, overriding the row's normal title-as-tooltip.
            Ignored when ``disabled`` is ``False``.
    """

    row_id: str
    section_id: str
    title: str
    target_kind: str
    target_id: str
    count: int | None = None
    count_known: bool = True
    count_display: str = ""
    count_emphasis: str = ""
    disabled: bool = False
    disabled_tooltip: str = ""


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
    prompts_count: int | None = None
    prompts_known: bool = True
    skills_count: int | None = None
    skills_known: bool = True
    collections_count: int | None = None
    collections_known: bool = True
    runtime_source: str = "local"
    server_label: str | None = None
    details_lines: tuple[str, ...] = ()
    study_decks_count: int | None = None
    flashcards_due_count: int | None = None
    quizzes_count: int | None = None


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
            row_id=LIBRARY_ROW_BROWSE_MEDIA,
            section_id="browse",
            title="Media",
            target_kind="canvas",
            target_id="media",
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
            row_id=LIBRARY_ROW_BROWSE_NOTES,
            section_id="browse",
            title="Notes",
            target_kind="canvas",
            target_id="notes",
            count=state.notes_count,
            count_known=state.notes_known,
        ),
        LibraryRailRow(
            # Row click resolves target_id "prompts" as its canvas_kind
            # below; the screen's compose_content (Task 3) renders
            # LibraryPromptsListCanvas for that kind -- no registry change
            # needed here, the row -> canvas_kind mapping already existed
            # from Task 1.
            row_id=LIBRARY_ROW_BROWSE_PROMPTS,
            section_id="browse",
            title="Prompts",
            target_kind="canvas",
            target_id="prompts",
            count=state.prompts_count,
            count_known=state.prompts_known,
        ),
        LibraryRailRow(
            # Task 1: row exists and is selectable now; its canvas (Task 3)
            # does not exist yet, so selecting it falls through to the
            # shell's generic empty-canvas landing path -- see
            # ``LIBRARY_ROW_BROWSE_SKILLS``'s comment above.
            row_id=LIBRARY_ROW_BROWSE_SKILLS,
            section_id="browse",
            title="Skills",
            target_kind="canvas",
            target_id="skills",
            count=state.skills_count,
            count_known=state.skills_known,
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_BROWSE_COLLECTIONS,
            section_id="browse",
            title="Collections",
            target_kind="canvas",
            target_id="collections",
            count=state.collections_count,
            count_known=state.collections_known,
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_BROWSE_SEARCH,
            section_id="browse",
            title="Search / RAG",
            target_kind="canvas",
            target_id="search",
            count=None,
            count_known=True,
        ),
    )

    create_rows = (
        LibraryRailRow(
            row_id=LIBRARY_ROW_CREATE_NOTE,
            section_id="create",
            title="New note",
            target_kind="canvas",
            target_id="notes-create",
            count=None,
            count_known=True,
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_CREATE_PROMPT,
            section_id="create",
            title="New prompt",
            target_kind="canvas",
            target_id="prompts",
            count=None,
            count_known=True,
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_CREATE_SKILL,
            section_id="create",
            title="New skill",
            target_kind="canvas",
            target_id="skills",
            count=None,
            count_known=True,
        ),
        LibraryRailRow(
            row_id="create-study",
            section_id="create",
            title="Study decks",
            target_kind="handoff",
            target_id="study",
            count=state.study_decks_count,
            count_known=True,
        ),
        LibraryRailRow(
            row_id="create-flashcards",
            section_id="create",
            title="Flashcards",
            target_kind="handoff",
            target_id="flashcards",
            count=None,
            count_known=True,
            count_display=(
                f" due: {state.flashcards_due_count}"
                if state.flashcards_due_count is not None
                else ""
            ),
            count_emphasis=(
                (
                    "bright"
                    if state.flashcards_due_count > 0
                    else "dim"
                )
                if state.flashcards_due_count is not None
                else ""
            ),
        ),
        LibraryRailRow(
            row_id="create-quizzes",
            section_id="create",
            title="Quizzes",
            target_kind="handoff",
            target_id="quizzes",
            count=state.quizzes_count,
            count_known=True,
        ),
    )

    # Computed up front (not just in the header-line block below) since
    # the Export row's server-mode gating also depends on it.
    runtime_source = str(state.runtime_source or "local").strip().lower()
    export_row_disabled = runtime_source == "server"

    ingest_rows = (
        LibraryRailRow(
            row_id=LIBRARY_ROW_INGEST_MEDIA,
            section_id="ingest",
            title="Import media",
            target_kind="canvas",
            target_id="ingest-media",
            count=None,
            count_known=True,
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_INGEST_EXPORT,
            section_id="ingest",
            title="Export",
            target_kind="canvas",
            target_id="export",
            count=None,
            count_known=True,
            disabled=export_row_disabled,
            disabled_tooltip=(
                LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP if export_row_disabled else ""
            ),
        ),
    )

    sections = (
        LibraryRailSectionState(section_id="browse", title="Browse", rows=browse_rows),
        LibraryRailSectionState(section_id="create", title="Create", rows=create_rows),
        LibraryRailSectionState(section_id="ingest", title="Import / Export", rows=ingest_rows),
    )

    # Build header line
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
        # Canvas rows resolve to their target canvas
        canvas_kind = selected_row.target_id
        canvas_target = ""
        canvas_empty_copy = LIBRARY_CANVAS_LANDING_COPY
    elif selected_row.target_kind == "handoff":
        # Handoff rows (study/flashcards/quizzes) resolve to the handoff
        # canvas: a Library-owned trio plus the Study handoff detail widget.
        canvas_kind = "handoff"
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
