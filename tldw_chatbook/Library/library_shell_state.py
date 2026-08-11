"""Pure Library shell rail state builders."""

from __future__ import annotations

from dataclasses import dataclass

# task-4023 AC#7: no "on the left" -- at ≤100 columns the shell shows one
# pane at a time (the rail fills the width and this canvas is hidden), so
# spatial copy was width-dependent nonsense. The copy now holds at every
# width the layout can take.
LIBRARY_CANVAS_LANDING_COPY = (
    "Search everything, pick a section, or add something new."
)

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
LIBRARY_CANVAS_KIND_NOTES_CREATE = "notes-create"
# The three Study staging (handoff) rows -- consumers that need "is this a
# study handoff row" import these rather than repeating the literals
# (Qodo PR-1488 #2; the study_rows definitions below are the canonical use).
LIBRARY_ROW_CREATE_STUDY = "create-study"
LIBRARY_ROW_CREATE_FLASHCARDS = "create-flashcards"
LIBRARY_ROW_CREATE_QUIZZES = "create-quizzes"
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

# F-018: every disabled Library action says why. The three canvas
# "Export selected" actions share this pair -- reason while disabled,
# action description once a selection exists (the workspaces handoff
# button's pattern, library_screen.py `_workspace_handoff_action_state`).
LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP = "Select one or more items to export them."
LIBRARY_EXPORT_SELECTED_TOOLTIP = "Export the selected items."

# task-2853: the Media canvas's "Delete selected" bulk action follows the
# same disabled/enabled tooltip pair as "Export selected" above.
LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP = "Select one or more items to delete them."
LIBRARY_DELETE_SELECTED_TOOLTIP = "Move the selected items to trash."

# task-4023 AC#1 (RC-07): disabled state finally joins the product's
# non-colour vocabulary. Every disabled Library action label carries a
# leading "○" -- the existing ✓/○ pair's neutral glyph (ingest option
# toggles, sync status, RAG scope chips), extended rather than a new
# invention -- so the state survives monochrome rendering, colour-blind
# users, and low-contrast themes. The colour half of the fix (the 3:1
# Legible Disabled floor) is app-tier CSS in css/components/
# _agentic_terminal.tcss; this marker is the structural half.
LIBRARY_DISABLED_ACTION_MARKER = "○"

# F-018 reason for the list canvases' Select toggle while the rendered
# list is empty -- previously the only disabled Library action with no
# reason anywhere at the control ("click does nothing, says nothing",
# re-critique RC-07).
LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP = "Nothing here to select yet."


def library_disabled_action_label(label: str, disabled: bool) -> str:
    """Prefix ``label`` with the non-colour disabled marker when disabled.

    Args:
        label: The action's plain enabled label.
        disabled: Whether the control renders disabled.

    Returns:
        ``"○ <label>"`` while disabled, ``label`` unchanged otherwise.
    """
    return f"{LIBRARY_DISABLED_ACTION_MARKER} {label}" if disabled else label


# task-4023 AC#5: "▸" carried three meanings on one screen -- selected-row
# prefix ("▸ Media"), collapsed-disclosure suffix ("Details ▸"), AND the
# silent value-cycler suffix ("type: All ▸"), where it looked like a
# disclosure but silently advanced a hidden option set. Convention after
# that task, extended by task-14902: leading "▸ " marks the selected row
# of a list; a trailing "▸/▾" pair on a section HEADER is disclosure
# state; "⇄" is a cycle control (press to advance -- since task-14902 the
# only surviving cyclers are genuine two-option TOGGLES, and the glyph
# sits BETWEEN the two options with the "✓" marker on the active one:
# "mode: ✓ Search ⇄ RAG Answer"); a plain "name: value" Button with no
# glyph is a CHOOSER-OPENER (the Notes Sort precedent -- press swaps in a
# one-row choice strip, "✓ " prefixing the active option, Escape/second
# press cancels).
LIBRARY_CYCLE_MARKER = "⇄"

#: The non-colour active-option marker shared by every choice strip and
#: kept toggle (extends the product's existing ``✓/○`` vocabulary -- the
#: Notes Sort strip and the sync panel's direction/conflict groups used
#: it first; task-14902 makes it the one marker for "this option is the
#: active one").
LIBRARY_CHOICE_ACTIVE_MARKER = "✓"


def library_choice_label(name: str, value: str) -> str:
    """Build a chooser-opener Button label: ``"{name}: {value}"``.

    task-14902: one source for every Library control whose press OPENS a
    direct-pick surface (the choice strips on media type / prompts sort /
    skills sort / export quality, and the prompt-collection manager
    modal). Deliberately glyph-free: ``⇄`` means "press to advance", and
    these controls no longer advance.

    Args:
        name: The control's subject (e.g. ``"type"``).
        value: The currently active option, already display-safe.

    Returns:
        The chooser Button's label text.
    """
    return f"{name}: {value}"


def library_choice_tooltip(subject: str, options: "tuple[str, ...] | list[str]") -> str:
    """Build a chooser-opener's tooltip naming the pick interaction.

    Replaces ``library_cycle_tooltip``'s "Cycles ..." copy on the
    converged controls -- a press shows the options to pick from, it does
    not cycle, so the tooltip must not claim it does.

    Args:
        subject: What is being picked, e.g. ``"media type"``.
        options: The full option set, in display order.

    Returns:
        ``"Press to pick {subject}: A · B · C."`` (or the generic line
        when the option set is empty/dynamic).
    """
    listing = " · ".join(str(option) for option in options if str(option))
    if not listing:
        return f"Press to pick {subject}."
    return f"Press to pick {subject}: {listing}."


def library_toggle_label(
    name: str, options: "tuple[str, str]", active_index: int
) -> str:
    """Build a kept one-press toggle's label with the FULL option set.

    task-14902 AC#1: the surviving cyclers are genuine two-option toggles
    (Search/RAG mode, the skill editor's yes/no + inline/fork switches) --
    a choice strip would add a press to the most common action for zero
    information, so instead the whole option space moves onto the label:
    both options in stable order, ``✓`` on the active one, ``⇄`` between
    them keeping its press-advances meaning. One press IS a direct pick
    of the only other option.

    Args:
        name: The control's subject (e.g. ``"mode"``).
        options: Both options in canonical order (never reordered by
            activation, so the label stays spatially stable).
        active_index: Index (0 or 1) of the currently active option.

    Returns:
        e.g. ``"mode: ✓ Search ⇄ RAG Answer"``.
    """
    rendered = tuple(
        f"{LIBRARY_CHOICE_ACTIVE_MARKER} {option}" if index == active_index else option
        for index, option in enumerate(options)
    )
    return f"{name}: {rendered[0]} {LIBRARY_CYCLE_MARKER} {rendered[1]}"


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
    # F-014: one-line plain-language gloss for jargon rows (e.g. "search
    # everything" under "Search / RAG"), rendered by the rail as a dim
    # em-dash suffix on the same line. Empty on already-plain rows so the
    # rail doesn't stutter.
    subtitle: str = ""
    # F-014: True while this row's count is still being fetched -- the
    # rail renders a dim "(…)" placeholder (one count policy: placeholder
    # while loading, count when known, no suffix when the source is off).
    count_loading: bool = False
    # LIB-15: True for a row whose count is not fetched yet but WILL be
    # (e.g. Collections, whose count only loads on first canvas visit --
    # see ``count_loading``'s own docstring for why it is NOT used for this
    # row: showing a "(…)" placeholder before any fetch has even started
    # would misrepresent an idle row as actively loading). Distinct from a
    # row like Search/RAG, whose ``count`` is ``None`` FOREVER by design
    # (an "off" source, not a pending one) -- without this flag the two
    # cases are indistinguishable from ``count``/``count_known`` alone.
    # ``_row_label``'s gloss-fit check reserves a stable minimum width for
    # this row's eventual count so the gloss's visibility does not flip the
    # instant the count arrives (the observed "Collections — item sets" ->
    # "(0)", gloss silently dropping on the SAME terminal width just
    # because the count went from absent to a single digit).
    count_pending: bool = False
    # LIB-18: an abbreviated fallback for a single-word ``title`` that would
    # otherwise need a mid-word ellipsis cut at the rail's real minimum
    # width (120/100/80 columns all pin the rail to the same ~17-cell row
    # budget via its own ``min_width``). Empty for every row whose title
    # already fits comfortably. Used INSTEAD OF ellipsizing the full title
    # when the full title does not fit but ``short_title`` does -- never
    # itself further truncated except as an absolute last resort.
    short_title: str = ""


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
    # F-014: True while the local source snapshot is still in flight --
    # every row whose count rides that snapshot renders the dim "(…)"
    # placeholder instead of a misleading "(0)" or a blank. Collections
    # is excluded (its count is fetched lazily on the first canvas visit).
    counts_loading: bool = False


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
            # task-2236 (R2): glosses fit the rail's realistic
            # width budget (<=25 content cells with title+count).
            subtitle="your files",
            count_loading=state.counts_loading,
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_BROWSE_CONVERSATIONS,
            section_id="browse",
            title="Conversations",
            target_kind="canvas",
            target_id="conversations",
            count=state.conversations_count,
            count_known=state.conversations_known,
            count_loading=state.counts_loading,
            # LIB-18: "Conversations" (13 cells) does not fit the rail's
            # real ~17-cell row budget alongside a count without ellipsis-
            # cutting mid-word ("Conversa..."). "Chats" is the app's own
            # existing short label for this concept (see workspace.py's
            # DEFAULT_WORKSPACE_ID label and conversation_browser_state.py).
            short_title="Chats",
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_BROWSE_NOTES,
            section_id="browse",
            title="Notes",
            target_kind="canvas",
            target_id="notes",
            count=state.notes_count,
            count_known=state.notes_known,
            count_loading=state.counts_loading,
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
            # task-2859 item 2: "AI asks" (jargon noun for "the things you
            # ask the AI") read as cryptic in UAT -- "reuse" is plain
            # language. Live-verified at 170x50 (the plan's required
            # verification width): two longer drafts ("saved
            # instructions", then "reuse text") both silently dropped at
            # this exact width even though a hand-computed budget check
            # said they should fit -- "Prompts" is one cell longer than
            # "Skills", and the F-015 "gloss renders whole or not at all"
            # rule is unforgiving of being even one cell over. Kept short
            # with margin rather than re-deriving the rail's exact
            # available width precisely.
            subtitle="reuse",
            count_loading=state.counts_loading,
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
            subtitle="AI add-ons",
            count_loading=state.counts_loading,
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_BROWSE_COLLECTIONS,
            section_id="browse",
            title="Collections",
            target_kind="canvas",
            target_id="collections",
            count=state.collections_count,
            count_known=state.collections_known,
            subtitle="item sets",
            # LIB-15: Collections' count is fetched lazily (on first canvas
            # visit), unlike every other counts_loading row -- flagging it
            # pending (rather than "off", Search/RAG's case) lets the
            # gloss-fit check reserve stable width for the count that is
            # coming, so the gloss does not silently drop the instant the
            # count arrives at the same terminal width.
            count_pending=True,
            # LIB-18: "Collections" (11 cells) fits the ~17-cell row budget
            # only while its count stays single-digit; "Sets" (matching
            # this row's own "item sets" gloss) is the fallback once a
            # longer count would otherwise force a mid-word cut.
            short_title="Sets",
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_BROWSE_SEARCH,
            section_id="browse",
            title="Search / RAG",
            target_kind="canvas",
            target_id="search",
            count=None,
            count_known=True,
            subtitle="find all",
        ),
    )

    create_rows = (
        LibraryRailRow(
            row_id=LIBRARY_ROW_CREATE_NOTE,
            section_id="create",
            title="New note",
            target_kind="canvas",
            target_id=LIBRARY_CANVAS_KIND_NOTES_CREATE,
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
    )

    # F-017: the Study rows are handoffs, not creation verbs -- every one
    # opens the Study destination ("Continue in Study" per
    # LIBRARY_STUDY_HANDOFF_MODES), so they group under their own "Study"
    # section between Create and Import / Export. Row ids stay
    # "create-*": they are long-published DOM ids (tests and deep links
    # press them); the section_id carries the regrouping.
    study_rows = (
        LibraryRailRow(
            row_id=LIBRARY_ROW_CREATE_STUDY,
            section_id="study",
            title="Study decks",
            target_kind="handoff",
            target_id="study",
            count=state.study_decks_count,
            count_known=True,
            count_loading=state.counts_loading,
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_CREATE_FLASHCARDS,
            section_id="study",
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
                ("bright" if state.flashcards_due_count > 0 else "dim")
                if state.flashcards_due_count is not None
                else ""
            ),
            count_loading=state.counts_loading,
            # LIB-18: "Flashcards" (10 cells) plus its own longer
            # " due: N" count display (vs. the default " (N)") leaves less
            # budget than any other row -- "Cards" is the common shorthand
            # (matches Anki's own terminology) and reads fine paired with
            # the row's own "due: N" suffix for context.
            short_title="Cards",
        ),
        LibraryRailRow(
            row_id=LIBRARY_ROW_CREATE_QUIZZES,
            section_id="study",
            title="Quizzes",
            target_kind="handoff",
            target_id="quizzes",
            count=state.quizzes_count,
            count_known=True,
            count_loading=state.counts_loading,
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
            # task-2857 (Library UAT 2026-08-06, LIB-10): one canonical verb
            # ("Import") across rail, canvas header, empty states, buttons
            # and toasts -- supersedes task-2235 (R2)'s "Add content…",
            # which disagreed with the canvas it opened ("Import media"),
            # the Start button, and the completion toast, all of which
            # already said "ingest"/"import" inconsistently.
            title="Import…",
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
        LibraryRailSectionState(section_id="study", title="Study", rows=study_rows),
        LibraryRailSectionState(
            section_id="ingest", title="Import / Export", rows=ingest_rows
        ),
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
