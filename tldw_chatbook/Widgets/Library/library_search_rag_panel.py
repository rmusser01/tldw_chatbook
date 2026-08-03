"""Textual widget for Library-native Search/RAG."""

from __future__ import annotations

from rich.markup import escape as escape_markup

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Static
from textual.widget import Widget

from ...Library.library_rag_state import (
    LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES,
    LibraryRagPanelState,
    LibraryRagQueryState,
    LibraryRagResultRow,
    LibraryRagScopeState,
    LibraryRagSourceOption,
    library_rag_empty_state_quiet_copy,
    library_rag_score_suffix,
    library_rag_scope_summary,
    searching_status_line,
)


class LibrarySearchRagPanel(VerticalScroll):
    """Display the source scope, query controls, and evidence results."""

    def __init__(self, state: LibraryRagPanelState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(
            "Library Search/RAG",
            id="library-rag-panel-title",
            classes="destination-section",
        )
        with Vertical(
            id="library-rag-query-controls",
            classes=_query_region_classes(self.state),
        ):
            yield Button(
                _mode_toggle_label(self.state),
                id="library-rag-mode-toggle",
                tooltip=_mode_toggle_tooltip(self.state),
            )
            yield Input(
                value=self.state.query_state.query,
                placeholder="Ask or search Library sources",
                id="library-rag-query-input",
            )
            for child in library_rag_query_status_children(self.state):
                yield child
            yield Button(
                self.state.query_state.run_action.label,
                id=self.state.query_state.run_action.widget_id,
                disabled=not self.state.query_state.run_action.enabled,
                tooltip=self.state.query_state.run_action.tooltip,
            )

        with Vertical(
            id="library-rag-source-scope",
            classes=_scope_region_classes(self.state),
        ):
            yield Static(
                "Sources",
                id="library-rag-scope-heading",
                classes="destination-section",
            )
            yield Static(
                _scope_summary(self.state),
                id="library-rag-scope-summary",
            )
            for toggle in library_rag_scope_toggle_children(self.state):
                yield toggle
            for child in library_rag_scope_recovery_children(self.state):
                yield child

        with Vertical(id="library-rag-results", classes="library-rag-region"):
            yield Static(
                results_heading_text(self.state),
                id="library-rag-results-heading",
                classes="destination-section",
            )
            for child in library_rag_results_body_children(self.state):
                yield child

        with Collapsible(
            title="Recent searches",
            collapsed=self.state.history_collapsed,
            id="library-rag-history",
        ):
            for child in library_rag_history_children(self.state):
                yield child


def _scope_summary(state: LibraryRagPanelState) -> str:
    """Return the source scope line for the main Search/RAG work lane."""
    return library_rag_scope_summary(state.scope)


def scope_toggle_label(option: LibraryRagSourceOption) -> str:
    """Return a toggle Button's visible label for one scope source option.

    Public (RAG-27 fix-review): also imported by the screen's snapshot-
    driven in-place refresh (`LibraryScreen._sync_library_rag_scope_toggle_and_run_gate_widgets`)
    so a background ingest's fresh counts can update each toggle's ``(N)``
    suffix without going through `library_rag_scope_toggle_children`'s
    full Button rebuild (a mount/remove sequence unsafe to run
    concurrently with the other refresh callers -- see that method's
    docstring).
    """
    marker = "✓" if option.selected else "○"
    return f"{marker} {option.label} ({option.count})"


def library_rag_scope_toggle_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return one full-width toggle `Button` per real source type (B2).

    Shared by the panel's own `compose()` and the screen's incremental
    refresh so both build identical toggles from the same state. Only
    `LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES` (notes/media/conversations) get
    a toggle -- workspaces/collections have no retrieval seam of their own.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        One toggle `Button` per real source type, disabled when that
        source's count is 0.
    """
    return [
        Button(
            scope_toggle_label(option),
            id=f"library-rag-scope-toggle-{option.source_type}",
            classes="library-rag-scope-toggle",
            disabled=not option.available,
            tooltip=f"Toggle {option.label} in the retrieval scope.",
        )
        for option in state.scope.options
        if option.source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
    ]


def library_rag_scope_shows_recovery(scope: LibraryRagScopeState) -> bool:
    """True when the scope region should render its recovery dump + Import media.

    Only the genuinely-empty-library case (no sources available at all)
    gets the full recovery presentation; a user deselecting every scope
    toggle with sources still available is covered by the query region's
    quiet line instead (A1/B2) -- an Import media button would not fix
    that case.
    """
    return bool(scope.recovery_copy) and not scope.has_available_sources


def library_rag_scope_recovery_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the scope region's no-sources gate line + Import media button, or none.

    Shared by `compose()` and the screen's incremental refresh. The gate
    copy is `LIBRARY_RAG_NO_SOURCES_GATE_COPY` -- one quiet muted line, not
    the retired Unavailable/Why/Next/Recovery/Owner dump -- rendered with
    the same quiet-line styling as the query gate lines.
    """
    if not library_rag_scope_shows_recovery(state.scope):
        return []
    return [
        Static(
            state.scope.recovery_copy,
            id="library-rag-scope-recovery",
            classes="library-rag-quiet-line",
        ),
        Button(
            "Open Import media",
            id="library-rag-open-import-export",
            classes="library-rag-recovery-action",
            tooltip="Open Library Import media to add sources.",
        ),
    ]


def _query_blocked_is_quiet(query_state: LibraryRagQueryState) -> bool:
    """True when the run gate's blocker renders as a single quiet line (A1)."""
    return query_state.blocked_is_empty_query or query_state.blocked_is_no_scope


def library_rag_query_shows_full_recovery(query_state: LibraryRagQueryState) -> bool:
    """True when the query region should render the callout + recovery dump.

    Reserved for real failures (unsafe query, missing dependencies/index, no
    provider for RAG mode) -- the empty-query and no-scope gates render a
    single quiet line instead (A1), and the ready/searching states render
    neither.
    """
    return bool(query_state.recovery_copy) and not _query_blocked_is_quiet(query_state)


def library_rag_query_status_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the query region's status widgets (A1/A2).

    Shared by `compose()` and the screen's incremental refresh. The quiet
    gate line is ALWAYS returned -- with empty text in the ready/searching
    states, and a fixed one-row height so the Run button below it never
    shifts vertically when a gate's copy appears or disappears (2026-07
    UAT: the button jumped ~2 rows on valid input, breaking muscle
    memory). The no-scope gate stays quiet-but-empty when the Library has
    no sources at all: the scope region's single no-sources gate line +
    "Open Import media" action own that state, so a second "Select at
    least one source." line would just re-stack guidance. Real failures
    (unsafe query, missing dependencies/index, no provider) additionally
    render the callout + recovery-copy block.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        The quiet-line `Static` (always), plus the callout + recovery
        widgets for full-recovery failures.
    """
    query_state = state.query_state
    quiet_text = ""
    if query_state.blocked_is_empty_query:
        quiet_text = "Enter a question or search query."
    elif query_state.blocked_is_no_scope and state.scope.has_available_sources:
        quiet_text = "Select at least one source."
    quiet_line = Static(
        quiet_text,
        id="library-rag-query-quiet-line",
        classes="library-rag-quiet-line",
    )
    quiet_line.styles.height = 1
    children: list[Widget] = [quiet_line]
    if library_rag_query_shows_full_recovery(query_state):
        reason = query_state.run_action.disabled_reason
        children.extend(
            (
                Static(
                    f"Blocked | {reason}",
                    id="library-rag-query-blocked-callout",
                    classes="library-rag-callout is-blocked",
                ),
                Static(query_state.recovery_copy, id="library-rag-query-recovery"),
            )
        )
    return children


def _mode_toggle_label(state: LibraryRagPanelState) -> str:
    """Return the visible mode-cycle button label."""
    return f"mode: {state.query_state.mode_label} ▸"


def _other_mode_label(state: LibraryRagPanelState) -> str:
    """Return the label of the mode a toggle press would switch TO.

    The cycle only ever has two states (`rag`/`search`), so the "other"
    mode is simply whichever one isn't current -- see `_mode_toggle_tooltip`.
    """
    return "Search" if state.query_state.mode == "rag" else "RAG Answer"


def _mode_toggle_tooltip(state: LibraryRagPanelState) -> str:
    """Return the mode-cycle button's tooltip, naming the next mode (RAG-39).

    A bare "Cycle Search/RAG mode." tooltip gives no hint how many modes
    exist or what a press does -- a two-state cycle looks identical to a
    five-state one. Naming the next mode makes the button's effect legible
    before the user presses it, and stays honest across a mode flip because
    it reads `state.query_state.mode` fresh on every build (recompose is
    the only path that rebuilds this button -- see the mode-toggle
    `Button.Pressed` handler in `library_screen.py`).
    """
    return f"Cycle Search/RAG mode. Next: {_other_mode_label(state)}."


def results_heading_text(state: LibraryRagPanelState) -> str:
    """Return the Evidence region heading, surfacing top-k (A3).

    Public (Task 8): shared by the panel's own `compose()` and the screen's
    incremental refresh (`_refresh_library_rag_results_widgets`), mirroring
    every other body/heading builder in this module.

    "Per source" is only true for keyword mode: `_search_keyword` fans out
    one query per selected source and caps each independently at `top_k`.
    Rag mode's semantic leg is ONE store query (or one per allowlisted
    source type under an active scope, still merged by score) trimmed to a
    single `top_k` overall -- so the suffix is dropped there rather than
    making a claim that live UAT showed was false (RAG-29/scout item 3).
    """
    suffix = "" if state.query_state.mode == "rag" else " per source"
    return f"Evidence · top {state.query_state.top_k}{suffix}"


def library_rag_coverage_note_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the Evidence region's semantic coverage-note `Static`, or none.

    Shared by `compose()` and the screen's incremental refresh (folded into
    `library_rag_results_body_children` below, so both paths get it for
    free). Reuses the existing `library-rag-quiet-line` styling -- no new
    CSS. Empty (`[]`) whenever `state.coverage_note` has nothing to say
    (everything the query's semantic leg was asked to cover came back
    covered, and no result banded weak) -- see `library_rag_coverage_note`.
    """
    if not state.coverage_note:
        return []
    return [
        Static(
            state.coverage_note,
            id="library-rag-coverage-note",
            classes="library-rag-quiet-line",
        )
    ]


def library_rag_result_row_children(
    row: LibraryRagResultRow,
    index: int,
    selected_result_id: str,
) -> list[Widget]:
    """Return one evidence row as a single focusable card (C1/Task 12).

    Shared by the panel's own `compose()` and the screen's incremental DOM
    refresh (`_refresh_library_rag_results_widgets`) so both build identical
    rows from the same state.

    RAG-36 (live UAT, keyboard-only persona Sam): evidence rows used to be a
    flat list of sibling Statics plus a per-row `Horizontal` of buttons,
    mounted directly into the results container -- Tab only ever reached
    the buttons, with no row-level cursor and nothing indicating which row
    keyboard focus was "on". Every row's children are now wrapped in one
    `Vertical` card (`.library-rag-result-card`, `#library-rag-result-card-
    {index}`) that is itself a Tab stop; `LibraryScreen`'s Enter/`o`
    handlers resolve this card's index the same way the button handlers do
    (`_trailing_index` on the id) and call the exact same underlying
    selection/open methods -- no duplicated logic.

    Args:
        row: The evidence row to render.
        index: The row's position among the currently rendered results,
            used to build stable per-row widget ids.
        selected_result_id: The panel's currently selected result id, if any.

    Returns:
        A single-element list holding the row's card: title -> badges ->
        snippet -> citations (when present) -> an action row with Open
        first (primary emphasis, when the row is openable) then Select
        evidence.
    """
    selected = row.result_id == selected_result_id
    score = library_rag_score_suffix(row.score)
    card_children: list[Widget] = [
        Static(
            f"{index + 1}. {row.title}{score}",
            id=f"library-rag-result-{index}",
            classes=(
                "library-rag-result-row is-selected"
                if selected
                else "library-rag-result-row"
            ),
        ),
        Static(
            row.row_badge_label,
            id=f"library-rag-result-badges-{index}",
            classes="library-rag-result-badges",
        ),
        Static(
            row.display_snippet,
            id=f"library-rag-result-snippet-{index}",
            classes="library-rag-result-snippet",
        ),
    ]
    if row.citation_labels:
        card_children.append(
            Static(
                f"Citations: {', '.join(row.citation_labels)}",
                id=f"library-rag-result-citations-{index}",
            )
        )
    actions: list[Widget] = []
    if row.can_open:
        actions.append(
            Button(
                "Open",
                id=f"library-rag-open-result-{index}",
                classes="library-rag-result-open console-action-primary",
                tooltip="Open this result's source in its Library editor/viewer.",
            )
        )
    actions.append(
        Button(
            "Selected evidence" if selected else "Select evidence",
            id=f"library-rag-select-result-{index}",
            classes="library-rag-result-action",
            tooltip="Select this evidence result for Console handoff.",
        )
    )
    card_children.append(Horizontal(*actions, classes="library-rag-result-actions"))
    card = Vertical(
        *card_children,
        id=f"library-rag-result-card-{index}",
        classes="library-rag-result-card",
    )
    # `Vertical.__init__` has no `can_focus` kwarg (only
    # `VerticalScroll`/`ScrollableContainer` accept it) -- set the instance
    # attribute directly, the same idiom already used elsewhere in this
    # screen (e.g. `left_rail.can_focus = True` in `library_screen.py`).
    card.can_focus = True
    return [card]


def library_rag_results_body_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the Evidence region's body widgets below the heading.

    Shared by `compose()` and the screen's incremental refresh
    (`_refresh_library_rag_results_widgets`) so both render identically:
    exactly one of evidence rows (plus a per-row Console handoff button on
    the selected row), the in-flight searching line, explicit retrieval
    recovery copy, or empty-state guidance, depending on retrieval status
    and result count.

    The `retrieval_status == "empty"` case (RAG-33/Task 11: a routine
    "your library has nothing matching this query" search) renders the
    quiet two-line `library_rag_empty_state_quiet_copy` instead of
    `state.recovery_copy`'s full Unavailable/Why/Next/Recovery/Owner
    dump -- that dump is reserved for real failures (`"blocked"`/
    `"failed"`: missing dependencies, empty index, provider unavailable,
    policy denial), which still render it verbatim because the user
    genuinely has to act on infrastructure there. Both branches keep
    `state.recovery_selector` as the rendered `Static`'s id, so existing
    selectors (`#library-rag-empty-state`, `#library-rag-service-error`)
    are unaffected.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        The widgets to mount directly below the Evidence heading.
    """
    # Task 8: the coverage note, when there is one, is the very first thing
    # under the heading -- ahead of the row list. `state.coverage_note` is
    # only ever non-empty alongside `state.results` (see
    # `library_rag_coverage_note`'s empty-rows guard), so prepending it
    # unconditionally here is a no-op in every other branch below rather
    # than needing its own conditional per branch.
    if state.results:
        children: list[Widget] = list(library_rag_coverage_note_children(state))
        for index, result in enumerate(state.results):
            children.extend(
                library_rag_result_row_children(result, index, state.selected_result_id)
            )
            if result.result_id == state.selected_result_id:
                children.append(
                    Button(
                        state.use_in_console_action.label,
                        id="library-rag-use-selected-in-console",
                        classes=(
                            "library-rag-console-action "
                            "library-rag-center-console-action"
                        ),
                        disabled=not state.use_in_console_action.enabled,
                        tooltip=state.use_in_console_action.tooltip,
                    )
                )
        return children
    if state.retrieval_status == "searching":
        return [
            Static(
                searching_status_line(state.scope.selected_source_types),
                id="library-rag-searching-line",
            )
        ]
    if state.recovery_copy and state.recovery_selector:
        if state.retrieval_status == "empty":
            return [
                Static(
                    library_rag_empty_state_quiet_copy(
                        state.query_state.query, state.scope
                    ),
                    id=state.recovery_selector,
                    classes="library-rag-quiet-line",
                )
            ]
        return [Static(state.recovery_copy, id=state.recovery_selector)]
    if not state.scope.has_available_sources:
        # No Library sources at all: the scope region's single quiet gate
        # line + "Open Import media" action are the entire guidance for
        # this state -- repeating "No evidence yet"/"Add or import
        # sources…" here would re-stack the layered dump the quiet-gate
        # principle retired (2026-07 UAT).
        return []
    return [
        Static(
            "No evidence yet. Run Search/RAG to populate results.",
            id="library-rag-results-empty",
        ),
        Static(
            _evidence_empty_guidance(),
            id="library-rag-evidence-empty-guidance",
            classes="library-rag-empty-guidance",
        ),
    ]


def library_rag_history_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the `Recent searches` collapsible's child widgets (D1).

    Shared by the widget's own `compose` and the screen's incremental
    DOM refresh so both build identical rows from the same state.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        When history is empty, a single muted placeholder `Static`.
        Otherwise: a muted hint `Static` first, then one full-width
        `Button` per history entry (most recent first), then a
        `Clear history` `Button` last.
    """
    if not state.history:
        return [
            Static(
                "No recent searches.",
                id="library-rag-history-empty",
                classes="library-rag-history-empty",
            )
        ]
    children: list[Widget] = [
        Static(
            "Select an entry to run it again.",
            id="library-rag-history-hint",
            classes="library-rag-history-hint",
        )
    ]
    children.extend(
        Button(
            # Textual parses a plain string Button label as markup: an
            # unescaped stored entry like "docs [/archive] cleanup" raises
            # MarkupError at construction time -- and because history is
            # persisted before this rebuild, the crash would recur on every
            # Search-canvas entry after restart. Escaping mirrors the
            # `_sanitize_display_text(escape=True)` path result titles and
            # snippets already use.
            escape_markup(entry),
            id=f"library-rag-history-{index}",
            classes="library-rag-history-row",
            # RAG-38: history entries are bare strings -- no mode was
            # recorded when they ran -- so clicking one always re-runs
            # under the CURRENT mode, not necessarily the one it first ran
            # under. The tooltip says so honestly instead of implying an
            # exact replay, and stays truthful across a mode flip because
            # it reads `state.query_state.mode_label` fresh on every build.
            tooltip=(
                f"Re-runs under the current mode "
                f"({state.query_state.mode_label})."
            ),
        )
        for index, entry in enumerate(state.history)
    )
    children.append(
        Button(
            "Clear history",
            id="library-rag-history-clear",
            classes="library-rag-history-clear",
        )
    )
    return children


def _evidence_empty_guidance() -> str:
    """Return empty evidence workflow guidance."""
    return "Add or import sources, run a query, then select evidence for Console."


def _query_region_classes(state: LibraryRagPanelState) -> str:
    """Return query-region classes that reserve recovery height only when needed."""
    return (
        "library-rag-region has-recovery"
        if library_rag_query_shows_full_recovery(state.query_state)
        else "library-rag-region"
    )


def _scope_region_classes(state: LibraryRagPanelState) -> str:
    """Return source-scope classes that keep the ready state compact."""
    return (
        "library-rag-region has-recovery"
        if library_rag_scope_shows_recovery(state.scope)
        else "library-rag-region"
    )
