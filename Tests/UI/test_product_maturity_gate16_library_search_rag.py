"""Gate 1.6 Library-native Search/RAG mounted UI regressions."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow
from tldw_chatbook.Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_SEARCH,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_selector,
)


async def _wait_for_library_shell_ready(screen, pilot, *, timeout: float = 2.0) -> None:
    """Wait for the Library rail shell (not the retired 3-pane workbench).

    Mirrors ``Tests/UI/test_library_shell.py::_wait_for_library_shell`` for
    suites that use the generic ``DestinationHarness``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(screen, "_library_loaded", False) and screen.query("#library-rail"):
            await pilot.pause()
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Library shell never loaded. Visible text: {_visible_text(screen)}"
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE16_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/"
    "2026-05-07-gate-1-6-library-native-search-rag.md"
)
ROADMAP = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
TASK_10 = Path(
    "backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md"
)
TASK_10_8 = Path(
    "backlog/tasks/task-10.8 - "
    "Product-Maturity-Phase-3.8-Gate-1.6-Library-Native-Search-RAG.md"
)
TASK_10_8_5 = Path(
    "backlog/tasks/task-10.8.5 - Gate-1.6.5-Library-Search-RAG-QA-closeout.md"
)


def _repo_text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _seed_library_sources(app) -> None:
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )


class StaticLibraryRagSearchService:
    def __init__(self, result, *, log=None):
        self.result = result
        self.calls = []
        # PR-3 Task 4: optional shared ordering log. Passing the same list to
        # this fake and to `RecordingAnswerChat` is what lets a test assert
        # retrieval happened BEFORE generation (`log == ["search", "answer"]`)
        # rather than merely that both happened.
        self.log = log if log is not None else []

    async def search(self, query, scope, mode, **kwargs):
        self.calls.append(
            {
                "query": query,
                "scope": scope,
                "mode": mode,
                **kwargs,
            }
        )
        self.log.append("search")
        return self.result


class DelayedLibraryRagSearchService(StaticLibraryRagSearchService):
    async def search(self, query, scope, mode, **kwargs):
        self.calls.append(
            {
                "query": query,
                "scope": scope,
                "mode": mode,
                **kwargs,
            }
        )
        await asyncio.sleep(0.05)
        return self.result


async def _wait_for_query_ready(
    screen, pilot, query: str, *, timeout: float = 2.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        inputs = list(screen.query("#library-rag-query-input"))
        buttons = list(screen.query("#library-rag-run-query"))
        if inputs and buttons:
            input_widget = inputs[0]
            run_button = buttons[0]
            if input_widget.value == query and run_button.disabled is False:
                await pilot.pause()
                return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Timed out waiting for Library Search/RAG query readiness: {query!r}"
    )


async def _wait_for_evidence_selected(
    screen,
    pilot,
    title: str,
    *,
    timeout: float = 2.0,
) -> None:
    """Wait for a result row to be selected in-panel.

    The retired 3-pane inspector column (``#library-rag-selected-result``,
    ``#library-rag-use-in-console``) is never mounted by the new canvas;
    ``LibrarySearchRagPanel`` now surfaces selection directly on the result
    row (``is-selected`` class) and its own per-result Console action
    (``#library-rag-use-selected-in-console``).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        selected_rows = list(screen.query(".library-rag-result-row.is-selected"))
        console_buttons = list(screen.query("#library-rag-use-selected-in-console"))
        if selected_rows and console_buttons:
            if (
                title in str(selected_rows[0].renderable)
                and console_buttons[0].disabled is False
            ):
                await pilot.pause()
                return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Timed out waiting for Library Search/RAG selection: {title!r}"
    )


# --- Task 8: mode-aware Evidence heading + semantic coverage note ----------


def test_evidence_heading_and_coverage_note_are_mode_aware_and_conditional() -> None:
    """(Task 8) RAG mode's Evidence heading drops the false "per source"
    claim -- the semantic leg is one merged store query trimmed to top_k,
    not a fan-out per selected source the way the keyword leg is -- and the
    coverage-note Static mounts directly under the heading, only when there
    is a specific, honest thing to say."""
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import (
        library_rag_results_body_children,
        results_heading_text,
    )

    result = LibraryRagResultRow.from_result(
        {
            "title": "Media doc",
            "score": 0.6,
            "source_id": "media-1",
            "provenance": {"source_type": "media"},
        }
    )
    rag_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1, "media": 1},
        query="cake",
        mode="rag",
        results=(result,),
        diagnostics={
            "semantic_scope_coverage": {"covered": ["media"], "uncovered": ["notes"]}
        },
    )
    search_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1, "media": 1},
        query="cake",
        mode="search",
        results=(result,),
    )

    # A3's "top_k" claim is only accurate for keyword mode's per-source
    # fan-out; rag mode drops the "per source" suffix outright.
    assert results_heading_text(rag_state) == "Evidence · top 5"
    assert results_heading_text(search_state) == "Evidence · top 5 per source"

    rag_children = library_rag_results_body_children(rag_state)
    coverage_statics = [
        child
        for child in rag_children
        if getattr(child, "id", None) == "library-rag-coverage-note"
    ]
    assert len(coverage_statics) == 1
    assert coverage_statics[0].has_class("library-rag-quiet-line")
    assert (
        str(coverage_statics[0].renderable)
        == "Semantic search found nothing from: Notes."
    )
    # It renders directly under the heading -- the first body child.
    assert rag_children[0] is coverage_statics[0]

    # Keyword mode's diagnostics never carry `semantic_scope_coverage` (no
    # coverage claim to make) -> no widget mounted at all.
    search_children = library_rag_results_body_children(search_state)
    assert not any(
        getattr(child, "id", None) == "library-rag-coverage-note"
        for child in search_children
    )


# --- Task 11: quiet no-match state (RAG-33) ---------------------------------


def test_empty_status_renders_quiet_two_line_state_not_full_dump() -> None:
    """(RAG-33/Task 11) A routine "your library has nothing matching this
    query" search (`retrieval_status == "empty"`) renders the quiet-line
    idiom (`.library-rag-quiet-line`, already used by the empty-query and
    no-scope gates) instead of the retired Unavailable/Why/Next/Recovery/
    Owner dump -- but a REAL failure at the same render seam
    (`retrieval_status == "failed"`: missing dependencies, empty index,
    provider unavailable, policy denial) still renders that dump verbatim,
    because the user genuinely has to act on infrastructure there."""
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import library_rag_results_body_children

    empty_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1, "media": 1},
        query="unicorn migration guide",
        retrieval_status="empty",
    )
    empty_children = library_rag_results_body_children(empty_state)
    assert len(empty_children) == 1
    quiet_static = empty_children[0]
    assert quiet_static.id == "library-rag-empty-state"
    assert quiet_static.has_class("library-rag-quiet-line")
    quiet_text = str(quiet_static.renderable)
    assert quiet_text == (
        "No evidence matched 'unicorn migration guide'.\nTry broader terms."
    )
    for jargon in ("Owner:", "Unavailable:", "Why:", "Next:", "Recovery:", "No results"):
        assert jargon not in quiet_text

    # Real failure: same render seam, different retrieval_status -> the
    # full recovery dump is unchanged.
    failed_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="unicorn migration guide",
        retrieval_status="failed",
    )
    failed_children = library_rag_results_body_children(failed_state)
    assert len(failed_children) == 1
    dump_static = failed_children[0]
    assert dump_static.id == "library-rag-service-error"
    assert not dump_static.has_class("library-rag-quiet-line")
    dump_text = str(dump_static.renderable)
    assert "Owner: Library retrieval." in dump_text
    assert "Unavailable: Library Search/RAG retrieval." in dump_text


# --- Task 13: honest re-run hint on history rows (RAG-38) ------------------


def test_history_row_tooltip_names_current_mode_and_tracks_mode_changes() -> None:
    """(RAG-38) Clicking a search-history row re-runs the query under the
    CURRENT mode, not the mode it originally ran under -- history entries
    are bare strings, no mode is recorded per entry (an honest fix here is
    a tooltip, not a persistence/data-model change). Each row's tooltip
    must therefore name whichever mode is presently selected, and change
    when the mode does -- a tooltip that always said e.g. "RAG Answer"
    would go stale the moment the user flipped modes and become exactly
    the kind of silent lie this task exists to remove."""
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import library_rag_history_children

    search_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        mode="search",
        history=("cats", "dogs"),
    )
    rag_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        mode="rag",
        history=("cats", "dogs"),
    )

    search_rows = {
        child.id: child for child in library_rag_history_children(search_state)
    }
    rag_rows = {child.id: child for child in library_rag_history_children(rag_state)}

    search_row = search_rows["library-rag-history-0"]
    rag_row = rag_rows["library-rag-history-0"]
    assert search_row.tooltip == "Re-runs under the current mode (Search)."
    assert rag_row.tooltip == "Re-runs under the current mode (RAG Answer)."
    # The dynamic pin: same history entry, same index, different mode ->
    # different tooltip text. A hardcoded string could not satisfy this.
    assert search_row.tooltip != rag_row.tooltip

    # Second row gets the same treatment, not just the first.
    assert (
        search_rows["library-rag-history-1"].tooltip
        == "Re-runs under the current mode (Search)."
    )


# --- PR-3 Task 3: answer region ---------------------------------------------
#
# `library_rag_answer_children` builds `Vertical#library-rag-answer`, mounted
# in `LibrarySearchRagPanel.compose()` as its own sibling BETWEEN
# `#library-rag-source-scope` and `#library-rag-results` -- never inside the
# latter, whose own teardown/remount loop
# (`LibraryScreen._refresh_library_rag_results_widgets`) only ever touches
# ITS OWN children (skip `LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS`, tear down
# the rest, remount from `library_rag_results_body_children`) and would
# otherwise destroy an answer region mounted inside it on every refresh. This
# also keeps `library_rag_results_body_children` itself unchanged, so
# `test_evidence_heading_and_coverage_note_are_mode_aware_and_conditional`
# above -- whose `rag_children[0] is coverage_statics[0]` assertion pins the
# coverage note as the FIRST results-body child -- stays true.


def _answer_region_children(region) -> list:
    """Return `region`'s constructor-time children without mounting it.

    `Widget.__init__(*children, ...)` stores its positional `children` args
    as `_pending_children` -- `.children` (`_nodes`) stays empty until the
    real Textual mount cycle processes them, which these pure
    builder-function tests never trigger (matching the unmounted style
    already used by every other test in this "Task N" block, e.g.
    `test_evidence_heading_and_coverage_note_are_mode_aware_and_conditional`
    above). This reads the widgets straight back from the constructor input
    `library_rag_answer_children` built them with, instead.
    """
    return list(region._pending_children)


def test_answer_region_ready_text_is_escaped_against_markup_injection() -> None:
    """PR-2's app-crash class (`[*/etc/hosts*]`) plus a plain `[bold]x`
    payload: the model's answer text is untrusted output like any other
    Library Search/RAG display text, rendered through a `Static` -- i.e.
    TEXTUAL's own markup tokenizer, which (unlike Rich's narrower
    `escape_markup`) opens a tag on ANY unescaped `[`. Escaping must be the
    TERMINAL step (no transform runs after it); this pins both halves: the
    payload parses as no markup at all, and the words a user needs to read
    the answer survive."""
    from rich.text import Text

    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_READY,
        LibraryRagAnswer,
    )
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import library_rag_answer_children

    for payload, preserved in (
        ("[bold]x", ("bold", "x")),
        ("An expired credential [*/etc/hosts*] caused the incident.", ("etc", "hosts")),
    ):
        answer = LibraryRagAnswer(
            status=ANSWER_STATUS_READY,
            text=payload,
            citation_status="validated",
            citation_recovery="",
        )
        state = LibraryRagPanelState.from_values(
            source_counts={"notes": 1},
            query="Why did the incident happen?",
            mode="rag",
            answer=answer,
        )
        region = library_rag_answer_children(state)[0]
        text_static = next(
            child
            for child in _answer_region_children(region)
            if child.id == "library-rag-answer-text"
        )
        rendered = Text.from_markup(str(text_static.renderable))
        assert rendered.spans == []
        for word in preserved:
            assert word in rendered.plain


def test_answer_region_ready_status_branches_on_citation_status_before_clean_render() -> (
    None
):
    """Carried ruling (Task 1 review): `status == "ready"` must NEVER render
    as a clean grounded answer without branching on `citation_status` --
    `uncited`/`unverified` show their `citation_recovery` copy at least as
    prominently as the answer text (a bordered callout ABOVE it, not a
    footnote below). `validated` gets a neutral note that never claims the
    citation is "verified" -- `build_answer_citation_validation` only checks
    that a citation label RESOLVES to a staged reference, never that the
    snippet actually supports the claim."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_READY,
        LibraryRagAnswer,
    )
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import library_rag_answer_children

    for citation_status, recovery in (
        ("uncited", "The answer does not cite available staged evidence."),
        (
            "unverified",
            "Some citation markers do not match available staged evidence.",
        ),
    ):
        answer = LibraryRagAnswer(
            status=ANSWER_STATUS_READY,
            text="An expired credential caused the incident.",
            citation_status=citation_status,
            citation_recovery=recovery,
        )
        state = LibraryRagPanelState.from_values(
            source_counts={"notes": 1},
            query="Why did the incident happen?",
            mode="rag",
            answer=answer,
        )
        region = library_rag_answer_children(state)[0]
        region_children = _answer_region_children(region)
        ids_in_order = [child.id for child in region_children]
        assert ids_in_order.index("library-rag-answer-caution") < ids_in_order.index(
            "library-rag-answer-text"
        )
        caution = next(
            child
            for child in region_children
            if child.id == "library-rag-answer-caution"
        )
        assert caution.has_class("library-rag-callout")
        assert caution.has_class("is-caution")
        assert str(caution.renderable) == recovery

    clean_answer = LibraryRagAnswer(
        status=ANSWER_STATUS_READY,
        text="An expired credential caused the incident [S1].",
        citation_status="validated",
        citation_recovery="",
    )
    clean_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Why did the incident happen?",
        mode="rag",
        answer=clean_answer,
    )
    clean_region = library_rag_answer_children(clean_state)[0]
    clean_region_children = _answer_region_children(clean_region)
    assert not any(
        child.id == "library-rag-answer-caution" for child in clean_region_children
    )
    note = next(
        child
        for child in clean_region_children
        if child.id == "library-rag-answer-citation-note"
    )
    note_text = str(note.renderable).lower()
    assert "verified" not in note_text
    assert "resolve" in note_text


def test_answer_region_abstained_and_no_evidence_render_quiet_register() -> None:
    """Carried ruling (Task 1 review): abstention -- and the no-evidence
    path, which never called a provider at all -- are NOT errors. Both
    render in the same `.library-rag-quiet-line` register this panel
    already uses elsewhere for "nothing went wrong, there's just nothing to
    show" (RAG-29/33), not a bordered callout or a recovery dump."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_ABSTAINED,
        ANSWER_STATUS_NO_EVIDENCE,
        LIBRARY_RAG_NO_EVIDENCE_TEXT,
        LibraryRagAnswer,
    )
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import library_rag_answer_children

    for status, text in (
        (ANSWER_STATUS_ABSTAINED, "I can't answer that from the evidence given."),
        (ANSWER_STATUS_NO_EVIDENCE, LIBRARY_RAG_NO_EVIDENCE_TEXT),
    ):
        answer = LibraryRagAnswer(status=status, text=text)
        state = LibraryRagPanelState.from_values(
            source_counts={"notes": 1},
            query="Why did the incident happen?",
            mode="rag",
            answer=answer,
        )
        region = library_rag_answer_children(state)[0]
        region_children = _answer_region_children(region)
        text_static = next(
            child for child in region_children if child.id == "library-rag-answer-text"
        )
        assert text_static.has_class("library-rag-quiet-line")
        assert str(text_static.renderable) == text
        assert not any(
            child.has_class("library-rag-callout") for child in region_children
        )


def test_answer_region_failed_status_renders_quiet_error_and_retry_hint() -> None:
    """A failed generation attempt is not an error dump either (carried
    ruling) -- one quiet line naming what went wrong, plus a retry hint.
    Task 3 ships hint copy rather than a bespoke retry Button: the existing
    Run button already re-triggers retrieval + answer generation, so a
    second, not-yet-wired button would be a dead click until Task 4 lands --
    worse than no button at all."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_FAILED,
        LibraryRagAnswer,
    )
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import library_rag_answer_children

    answer = LibraryRagAnswer(
        status=ANSWER_STATUS_FAILED,
        text="",
        error="Provider timed out after 30s",
    )
    state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Why did the incident happen?",
        mode="rag",
        answer=answer,
    )
    region = library_rag_answer_children(state)[0]
    region_children = _answer_region_children(region)
    error_static = next(
        child for child in region_children if child.id == "library-rag-answer-error"
    )
    assert error_static.has_class("library-rag-quiet-line")
    assert "Provider timed out after 30s" in str(error_static.renderable)
    assert not any(child.has_class("library-rag-callout") for child in region_children)

    retry_hint = next(
        child
        for child in region_children
        if child.id == "library-rag-answer-retry-hint"
    )
    assert retry_hint.has_class("library-rag-quiet-line")
    assert "again" in str(retry_hint.renderable).lower()


def test_answer_region_shows_generating_indicator_while_answering() -> None:
    """The in-flight "answering" retrieval_status (Task 3's normalizer
    addition) renders a quiet generating-indicator line under the same
    "Answer" heading -- distinct from the final answer/abstention/failure
    presentations, and gone once one of those lands."""
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import library_rag_answer_children

    state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Why did the incident happen?",
        mode="rag",
        retrieval_status="answering",
    )
    region = library_rag_answer_children(state)[0]
    assert region.id == "library-rag-answer"
    status_static = next(
        child
        for child in _answer_region_children(region)
        if child.id == "library-rag-answer-status"
    )
    assert str(status_static.renderable) == "Generating answer…"


def test_answer_region_absent_outside_rag_mode_and_before_any_answer() -> None:
    """Keyword (search) mode never shows an answer region at all, even if a
    stale `LibraryRagAnswer` is still sitting on state (e.g. a mode flip
    after a landed rag answer) -- this is deliberately NOT keyed off
    `state.results`/`state.retrieval_status` alone. The idle canvas (rag
    mode, no query run yet) also renders nothing: there is no answering
    status and no answer to show."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_READY,
        LibraryRagAnswer,
    )
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import library_rag_answer_children

    stale_answer = LibraryRagAnswer(
        status=ANSWER_STATUS_READY,
        text="grounded answer",
        citation_status="validated",
    )
    search_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Why did the incident happen?",
        mode="search",
        answer=stale_answer,
    )
    assert library_rag_answer_children(search_state) == []

    idle_rag_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        mode="rag",
    )
    assert library_rag_answer_children(idle_rag_state) == []


@pytest.mark.asyncio
async def test_library_search_rag_empty_results_render_quiet_state_end_to_end() -> (
    None
):
    """(RAG-33/Task 11) Full plumbing: a real zero-row service outcome
    renders the quiet two-line no-match state, not the six-line dump the
    2026-07 UAT flagged (critique RAG-33)."""
    app = _build_test_app()
    _seed_library_sources(app)
    service = StaticLibraryRagSearchService([])
    app.library_rag_search_service = service
    host = DestinationHarness(app, "library")
    query = "unicorn migration guide"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-empty-state")

        quiet_line = screen.query_one("#library-rag-empty-state", Static)
        assert quiet_line.has_class("library-rag-quiet-line")

        visible_text = _visible_text(screen)
        assert "No evidence matched 'unicorn migration guide'." in visible_text
        assert "Try broader terms" in visible_text
        assert "Owner:" not in visible_text
        assert "Unavailable:" not in visible_text
        assert len(screen.query("#library-rag-service-error")) == 0


def test_library_search_rag_provenance_labels_escape_rich_markup() -> None:
    result = LibraryRagResultRow.from_result(
        {
            "document_title": "Markup Attempt",
            "snippet": "Adapter provenance should render literally.",
            "source_id": "note-markup",
            "chunk_id": "chunk-markup",
            "provenance": {
                "source_type": "[bold]spoof[/]",
                "workspace_ids": ("[red]workspace[/]",),
                "authority_label": "[green]trusted[/]",
                "eligibility_reason": "[blink]blocked[/]",
            },
        }
    )

    combined = " ".join(
        (
            result.row_badge_label,
            result.authority_display_label,
            result.eligibility_label,
        )
    )
    assert "[bold]spoof[/]" not in combined
    assert "[red]workspace[/]" not in combined
    assert "[green]trusted[/]" not in combined
    assert r"\[bold]spoof\[/]" in combined
    assert r"\[red]workspace\[/]" in combined
    assert r"\[green]trusted\[/]" in combined


@pytest.mark.asyncio
async def test_library_search_rag_mode_mounts_native_panel_without_leaving_library() -> (
    None
):
    """Selecting the Search rail row mounts ``LibrarySearchRagPanel`` inside
    the Library canvas without navigating away. The retired 3-pane workbench
    panes (``#library-source-browser/-detail/-inspector``) and the inspector
    column (``#library-rag-inspector``, ``#library-rag-use-in-console``) are
    never mounted by the new shell; the canvas host is ``#library-canvas``."""
    app = _build_test_app()
    _seed_library_sources(app)
    seen_routes: list[str] = []
    host = DestinationHarness(app, "library", seen_routes=seen_routes)

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        assert len(screen.query("#library-search-rag-panel")) == 0

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        assert _active_destination_screen(host) is screen
        assert seen_routes == []
        assert len(screen.query("#search-rag-container")) == 0

        canvas = screen.query_one("#library-canvas")
        for selector in (
            "#library-search-rag-panel",
            "#library-rag-source-scope",
            "#library-rag-query-input",
            "#library-rag-run-query",
            "#library-rag-results",
        ):
            assert canvas.query_one(selector)

        active_row_button = screen.query_one("#library-row-browse-search", Button)
        assert active_row_button.tooltip == "Search / RAG"
        assert active_row_button.has_class("library-rail-row-selected")


@pytest.mark.asyncio
async def test_library_search_rag_panel_exposes_blocked_recovery_for_empty_query() -> (
    None
):
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        query_input = screen.query_one("#library-rag-query-input", Input)
        run_button = screen.query_one("#library-rag-run-query", Button)
        visible_text = _visible_text(screen)

        assert query_input.value == ""
        assert str(run_button.label) == "Run"
        assert run_button.disabled is True
        assert "Enter a question or search query" in str(run_button.tooltip)
        assert not screen.query(".library-rag-result-action")
        # A1: the empty-query gate is a single quiet line now -- no summary
        # Static, callout box, "Run disabled:" reason, or recovery dump.
        assert screen.query_one("#library-rag-query-quiet-line", Static)
        assert "Enter a question or search query." in visible_text
        assert not screen.query("#library-rag-query-blocked-callout")
        assert not screen.query("#library-rag-query-recovery")
        assert not screen.query("#library-rag-run-disabled-reason")
        assert "Blocked: enter a question or search query." not in visible_text
        assert (
            "Blocked | Enter a question before running retrieval." not in visible_text
        )
        assert "Scope: all local sources" in visible_text
        # A4: the retired-workbench shortcuts line is gone; Enter-to-run
        # keeps working (covered by the keyboard-enter pilot below).
        assert not screen.query("#library-rag-query-shortcuts")
        assert "Tab: move panes" not in visible_text


@pytest.mark.asyncio
async def test_library_search_rag_task_loop_orders_query_before_scope_and_results() -> (
    None
):
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        panel = screen.query_one("#library-search-rag-panel")
        child_ids = [child.id for child in panel.children]
        assert child_ids.index("library-rag-query-controls") < child_ids.index(
            "library-rag-source-scope"
        )
        assert child_ids.index("library-rag-source-scope") < child_ids.index(
            "library-rag-results"
        )
        assert "Scope: all local sources" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_search_rag_empty_sources_has_mode_local_blocked_status() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        visible_text = _visible_text(screen)
        # (task-185) The no-sources state is ONE quiet gate line plus the
        # single Open Import media action -- the old 8-line recovery dump,
        # its checklist, the "Select at least one source." query line, and
        # the Evidence empty-state hints must not stack on top of it.
        gate_line = screen.query_one("#library-rag-scope-recovery", Static)
        assert (
            str(gate_line.renderable)
            == "No Library sources yet — import media or create notes, then search."
        )
        assert "No Library sources yet" in visible_text
        assert "Recovery checklist" not in visible_text
        assert "Owner: Library source index." not in visible_text
        assert "Unavailable: Library Search/RAG." not in visible_text
        assert "1. Import Library sources." not in visible_text
        assert "Select at least one source." not in visible_text
        assert "Why: Enter a question or search query." not in visible_text
        assert (
            "No evidence yet. Run Search/RAG to populate results." not in visible_text
        )
        assert (
            "Add or import sources, run a query, then select evidence for Console."
            not in visible_text
        )
        assert not screen.query("#library-rag-results-empty")
        assert not screen.query("#library-rag-evidence-empty-guidance")
        # The quiet-line slot stays mounted (empty) so the Run button's
        # position is stable, but it carries no second guidance layer.
        quiet_line = screen.query_one("#library-rag-query-quiet-line", Static)
        assert str(quiet_line.renderable) == ""
        recovery_button = screen.query_one("#library-rag-open-import-export", Button)
        assert str(recovery_button.label) == "Open Import media"
        assert recovery_button.tooltip == "Open Library Import media to add sources."
        # Pressing this button drives the shell selection to the Ingest ▸
        # Import media canvas row (the Import/Export row/mode it used to
        # target is retired); the canvas-switch behavior itself is covered
        # by test_library_shell.py.


@pytest.mark.asyncio
async def test_library_search_rag_query_updates_action_and_survives_recompose() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")
    query = "What does the research note say?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False
        assert str(run_button.tooltip) == ""
        assert len(screen.query("#library-rag-query-recovery")) == 0
        # (task-185) The gate helper's one-row slot stays mounted (empty)
        # once the query is valid, so the Run button never shifts when the
        # "Enter a question or search query." line clears.
        quiet_line = screen.query_one("#library-rag-query-quiet-line", Static)
        assert str(quiet_line.renderable) == ""

        screen.refresh(recompose=True)
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        assert len(screen.query("#library-search-rag-panel")) == 1
        assert screen.query_one("#library-rag-query-input", Input).value == query
        assert screen.query_one("#library-rag-run-query", Button).disabled is False


@pytest.mark.asyncio
async def test_library_search_rag_run_query_renders_service_results_and_calls_scope() -> (
    None
):
    app = _build_test_app()
    _seed_library_sources(app)
    service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "score": "0.93",
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                    "runtime_backend": "local-fts",
                    "citations": [{"label": "Incident Review p.2"}],
                    "provenance": {
                        "source_type": "note",
                        "workspace_ids": ("workspace-a",),
                        "active_workspace_id": "workspace-a",
                    },
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    app.library_rag_search_service = service
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        # Default canvas mode is "search" (keyword); RAG-mode gating and
        # dispatch are covered separately by the mode-toggle pilots.
        assert service.calls == [
            {
                "query": query,
                "scope": ("notes", "media", "conversations"),
                "mode": "search",
                "top_k": 5,
                "include_citations": True,
            }
        ]
        visible_text = _visible_text(screen)
        # (RAG-34) Evidence rows render an honest match band, not the raw
        # cosine score -- 0.93 lands in the "strong" band.
        assert "Incident Review | match: strong" in visible_text
        assert "Expired credential caused the incident." in visible_text
        assert "Incident Review p.2" in visible_text
        assert len(screen.query("#library-rag-service-error")) == 0


@pytest.mark.asyncio
async def test_library_search_rag_rag_mode_renders_coverage_note_end_to_end() -> None:
    """(Task 8) Full plumbing: a rag-mode outcome whose `diagnostics` carry
    `semantic_scope_coverage` renders the honest heading and coverage note
    on screen -- not just at the pure display-state layer."""
    app = _build_test_app()
    _seed_library_sources(app)
    service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Fixture doc",
                    "snippet": "Unrelated fixture content.",
                    "score": "0.08",
                    "source_id": "media-1",
                    "runtime_backend": "rag-semantic",
                    "provenance": {"source_type": "media"},
                }
            ],
            "runtime_backend": "rag-semantic",
            "diagnostics": {
                "semantic_scope_coverage": {
                    "covered": ["media"],
                    "uncovered": ["notes", "conversations"],
                }
            },
        }
    )
    app.library_rag_search_service = service
    host = DestinationHarness(app, "library")
    query = "cake"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-mode-toggle")
        screen.query_one("#library-rag-mode-toggle", Button).press()
        for _ in range(150):
            toggles = list(screen.query("#library-rag-mode-toggle"))
            if toggles and str(toggles[0].label) == "mode: RAG Answer ▸":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Mode toggle never switched to RAG Answer.")

        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-coverage-note")

        visible_text = _visible_text(screen)
        # Scout item 3: the semantic leg is one merged query, not a
        # per-source fan-out -- the heading must not claim "per source".
        assert "Evidence · top 5" in visible_text
        assert "per source" not in visible_text
        assert (
            "Semantic search found nothing from: Notes, Conversations."
            in visible_text
        )


@pytest.mark.asyncio
async def test_library_search_rag_keyword_mode_never_renders_coverage_note() -> None:
    """Keyword mode's per-source fan-out keeps the "per source" heading
    claim (it is true there), and no coverage note is ever attached --
    `_search_semantic` is the only diagnostics producer for this slot."""
    app = _build_test_app()
    _seed_library_sources(app)
    service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "source_id": "note-42",
                    "runtime_backend": "local-fts",
                    "provenance": {"source_type": "note"},
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    app.library_rag_search_service = service
    host = DestinationHarness(app, "library")
    query = "incident"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        visible_text = _visible_text(screen)
        assert "Evidence · top 5 per source" in visible_text
        assert not screen.query("#library-rag-coverage-note")


@pytest.mark.asyncio
async def test_library_search_rag_selected_result_launches_console_live_work() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "score": 0.93,
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                    "runtime_backend": "local-fts",
                    "citations": [{"label": "Incident Review p.2"}],
                    "provenance": {
                        "source_type": "note",
                        "workspace_ids": ("workspace-a",),
                        "active_workspace_id": "workspace-a",
                    },
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        assert not screen.query(".library-rag-console-action")

        screen.query_one("#library-rag-select-result-0", Button).press()
        await _wait_for_evidence_selected(screen, pilot, "Incident Review")

        assert (
            screen.query_one("#library-rag-use-selected-in-console", Button).disabled
            is False
        )
        assert "Incident Review" in str(
            screen.query_one("#library-rag-result-0").renderable
        )

        screen.query_one("#library-rag-use-selected-in-console", Button).press()
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    launch_kwargs = app.open_console_for_live_work.call_args.kwargs
    assert launch_kwargs["source"] == "Library Search/RAG"
    assert launch_kwargs["title"] == "Incident Review"
    assert launch_kwargs["status"] == "staged"
    assert launch_kwargs["recovery"] == "Review citations before sending."
    assert launch_kwargs["action_label"] == "Review evidence in Console"
    payload = launch_kwargs["payload"]
    assert payload["target_id"] == "local:library-rag:note-42:chunk-7"
    assert payload["snippet"] == "Expired credential caused the incident."
    evidence_bundle = payload["evidence_bundle"]
    evidence_reference = evidence_bundle["references"][0]
    assert evidence_bundle["query"] == query
    assert evidence_bundle["status"] == "available"
    assert evidence_reference["evidence_id"] == "S1"
    assert evidence_reference["source_id"] == "note-42"
    assert evidence_reference["source_type"] == "note"
    assert evidence_reference["snippet"] == "Expired credential caused the incident."
    assert evidence_reference["authority_label"] == "Workspace: workspace-a"
    assert evidence_reference["metadata"]["active_context_eligible"] is True
    assert evidence_reference["metadata"]["global_browse_visible"] is True
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_library_search_rag_selected_result_evidence_metadata() -> None:
    """``LibrarySearchRagPanel`` surfaces the row provenance badge, snippet,
    and citations for a result unconditionally (not gated on selection). The
    retired inspector column's structured evidence breakdown (Source/Score/
    Runtime lines, "Authority & eligibility" and "Console Handoff" headings,
    the eligibility/handoff sentences) has no successor in the single-pane
    canvas."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "score": 0.93,
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                    "runtime_backend": "local-fts",
                    "citations": [{"label": "Incident Review p.2"}],
                    "provenance": {
                        "source_type": "note",
                        "workspace_ids": ("workspace-a",),
                        "active_workspace_id": "workspace-a",
                        "active_context_eligible": True,
                        "eligibility_reason": "active_workspace_match",
                    },
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        screen.query_one("#library-rag-select-result-0", Button).press()
        await _wait_for_evidence_selected(screen, pilot, "Incident Review")

        visible_text = _visible_text(screen)
        # UX wave M5: humanized badge composition -- "eligible" contributes
        # nothing (only a "blocked" -> "excluded from context" badge is
        # shown), joined with " · " instead of "|".
        assert "note · workspace-a · 1 citation" in visible_text
        assert (
            screen.query_one("#library-rag-use-selected-in-console", Button).disabled
            is False
        )
        assert "Expired credential caused the incident." in visible_text
        assert "Citations: Incident Review p.2" in visible_text


@pytest.mark.asyncio
async def test_library_search_rag_keyboard_enter_runs_query_and_handoff_button() -> (
    None
):
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Keyboard Evidence",
                    "snippet": "Keyboard-only users can run and stage evidence.",
                    "source_id": "note-keyboard",
                    "chunk_id": "chunk-1",
                }
            ],
        }
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")
    query = "Can keyboard-only users stage evidence?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = query
        await _wait_for_query_ready(screen, pilot, query)

        query_input.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        select_button = screen.query_one("#library-rag-select-result-0", Button)
        select_button.focus()
        await pilot.press("enter")
        await _wait_for_evidence_selected(screen, pilot, "Keyboard Evidence")

        console_button = screen.query_one(
            "#library-rag-use-selected-in-console", Button
        )
        console_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    payload = app.open_console_for_live_work.call_args.kwargs["payload"]
    assert payload["query"] == query
    assert payload["source_id"] == "note-keyboard"
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_library_search_rag_keyboard_u_shortcut_uses_selected_evidence() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Shortcut Evidence",
                    "snippet": "The u shortcut stages selected evidence.",
                    "source_id": "note-shortcut",
                    "chunk_id": "chunk-u",
                }
            ],
        }
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")
    query = "Can the u shortcut stage evidence?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = query
        await _wait_for_query_ready(screen, pilot, query)

        query_input.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        select_button = screen.query_one("#library-rag-select-result-0", Button)
        select_button.focus()
        await pilot.press("enter")
        await _wait_for_evidence_selected(screen, pilot, "Shortcut Evidence")

        await pilot.press("u")
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    payload = app.open_console_for_live_work.call_args.kwargs["payload"]
    assert payload["query"] == query
    assert payload["source_id"] == "note-shortcut"
    assert payload["chunk_id"] == "chunk-u"
    app.open_chat_with_handoff.assert_not_called()


# --- Task 12/RAG-36: keyboard-traversable evidence cards --------------------


@pytest.mark.asyncio
async def test_library_search_rag_tab_reaches_and_focuses_evidence_card() -> None:
    """Tab from the query input eventually lands on the first evidence
    card, and the card itself (not just its buttons) is the focus target --
    the flat Static+Button rows evidence used to render gave keyboard users
    no row-level cursor at all (RAG-36)."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "First result",
                    "snippet": "alpha evidence",
                    "source_id": "note-1",
                },
                {
                    "document_title": "Second result",
                    "snippet": "beta evidence",
                    "source_id": "note-2",
                },
            ],
        }
    )
    host = DestinationHarness(app, "library")
    query = "alpha or beta"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = query
        await _wait_for_query_ready(screen, pilot, query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-card-0")

        query_input.focus()
        await pilot.pause()

        for _ in range(40):
            focused = screen.focused
            if focused is not None and focused.id == "library-rag-result-card-0":
                break
            await pilot.press("tab")
        else:
            raise AssertionError(
                "Tab from the query input never reached the first evidence card."
            )

        card = screen.query_one("#library-rag-result-card-0")
        assert card.has_focus is True
        assert "focus" in card.pseudo_classes


@pytest.mark.asyncio
async def test_library_search_rag_enter_on_focused_card_selects_evidence() -> None:
    """Enter on a focused evidence card selects it, same as clicking its
    "Select evidence" button -- and the incremental results refresh that
    selection triggers must not steal focus off the card the user just
    acted on (the refresh-path lockstep hazard the largest part of this
    change has to get right)."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "First result",
                    "snippet": "alpha evidence",
                    "source_id": "note-1",
                },
                {
                    "document_title": "Second result",
                    "snippet": "beta evidence",
                    "source_id": "note-2",
                },
            ],
        }
    )
    host = DestinationHarness(app, "library")
    query = "alpha or beta"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = query
        await _wait_for_query_ready(screen, pilot, query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-card-1")

        card = screen.query_one("#library-rag-result-card-1")
        card.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_evidence_selected(screen, pilot, "Second result")

        assert screen.query_one("#library-rag-result-1").has_class("is-selected")
        # The refresh triggered by selection remounts the results region --
        # the SAME card index must still hold keyboard focus afterward.
        assert screen.query_one("#library-rag-result-card-1").has_focus is True


@pytest.mark.asyncio
async def test_library_search_rag_o_on_focused_card_opens_like_button() -> None:
    """`o` on a focused evidence card routes to the same open path as the
    row's "Open" button (mirrors ``test_library_shell.py``'s
    Open-button-lands-in-viewer pilots)."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "source_id": "media-1",
                    "document_title": "Transcript A",
                    "snippet": "audio transcript",
                    "provenance": {"source_type": "media"},
                }
            ],
        }
    )
    host = DestinationHarness(app, "library")
    query = "transcript"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = query
        await _wait_for_query_ready(screen, pilot, query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-open-result-0")

        card = screen.query_one("#library-rag-result-card-0")
        card.focus()
        await pilot.pause()
        await pilot.press("o")

        for _ in range(120):
            if (
                screen._selected_media_id == "media-1"
                and screen._library_media_view == "viewer"
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "`o` on the focused card never opened the media evidence item."
            )

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA


@pytest.mark.asyncio
async def test_library_search_rag_u_on_focused_card_selects_then_stages() -> None:
    """The focused-card fast path: `u` on a focused-but-not-yet-selected
    card selects that evidence AND stages it in one keystroke, instead of
    requiring Enter (select) then u (stage) as two separate actions."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Fast-path Evidence",
                    "snippet": "u selects then stages",
                    "source_id": "note-fastpath",
                    "chunk_id": "chunk-fp",
                }
            ],
        }
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")
    query = "Does u select then stage?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = query
        await _wait_for_query_ready(screen, pilot, query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-card-0")

        # No prior Enter/select -- the card is only focused, never activated.
        assert not screen.query(".library-rag-result-row.is-selected")

        card = screen.query_one("#library-rag-result-card-0")
        card.focus()
        await pilot.pause()
        await pilot.press("u")
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    payload = app.open_console_for_live_work.call_args.kwargs["payload"]
    assert payload["source_id"] == "note-fastpath"
    assert payload["chunk_id"] == "chunk-fp"
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_library_search_rag_server_result_launches_server_console_live_work() -> (
    None
):
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Server Incident Review",
                    "snippet": "Server retrieval found the authoritative incident record.",
                    "score": 0.88,
                    "source_id": "server-note-42",
                    "chunk_id": "chunk-9",
                    "runtime_backend": "server-rag",
                    "citations": [{"label": "Server Incident Review p.4"}],
                }
            ],
            "runtime_backend": "server-rag",
        }
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")
    query = "What did the server evidence say?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        screen.query_one("#library-rag-select-result-0", Button).press()
        await _wait_for_evidence_selected(screen, pilot, "Server Incident Review")

        screen.query_one("#library-rag-use-selected-in-console", Button).press()
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    launch_kwargs = app.open_console_for_live_work.call_args.kwargs
    assert launch_kwargs["source"] == "Library Search/RAG"
    assert launch_kwargs["title"] == "Server Incident Review"
    assert launch_kwargs["status"] == "staged"
    assert launch_kwargs["recovery"] == "Review citations before sending."
    assert launch_kwargs["action_label"] == "Review evidence in Console"
    payload = launch_kwargs["payload"]
    assert payload["target_id"] == "server:library-rag:server-note-42:chunk-9"
    assert payload["result_id"] == "server-note-42:chunk-9"
    assert payload["query"] == query
    assert payload["title"] == "Server Incident Review"
    assert payload["source_id"] == "server-note-42"
    assert payload["chunk_id"] == "chunk-9"
    assert (
        payload["snippet"]
        == "Server retrieval found the authoritative incident record."
    )
    assert payload["citations"] == ["Server Incident Review p.4"]
    assert payload["score"] == 0.88
    assert payload["runtime_backend"] == "server-rag"
    assert payload["source_authority"] == "server"
    assert payload["source_selector_state"] == "server"
    evidence_reference = payload["evidence_bundle"]["references"][0]
    assert evidence_reference["source_owner"] == "server"
    assert evidence_reference["authority_label"] == "Source authority: server"
    assert evidence_reference["snippet"] == (
        "Server retrieval found the authoritative incident record."
    )
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_library_search_rag_run_query_renders_persistent_recovery_without_service(
    monkeypatch,
) -> None:
    """RAG mode without embeddings support runs the query and renders the
    retrieval service's persistent "RAG unavailable" recovery state (task-249).

    The L3a provider gate that pre-disabled Run when ``app._rag_service`` was
    unset is retired: the runtime now initializes lazily at query time, so
    the Run button stays enabled and the real production service
    (``LibraryLocalRagSearchService``, wired by ``_build_test_app``'s real
    ``TldwCli``) owns the recovery copy routing the user to setup. The deps
    probe is pinned False so the lazy path never constructs a real embedding
    runtime inside a UI test. The default canvas mode is "search" (keyword,
    always available given seeded local seams), so reaching this state
    requires explicitly cycling to RAG mode first.
    """
    from tldw_chatbook.Library import library_local_rag_search_service

    monkeypatch.setattr(
        library_local_rag_search_service,
        "embeddings_rag_deps_installed",
        lambda: False,
    )
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")
    query = "What policy applies?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-mode-toggle")
        screen.query_one("#library-rag-mode-toggle", Button).press()

        # Mode toggle drives a full-screen recompose: poll for the new mode
        # label rather than assuming a fixed number of pauses is enough.
        for _ in range(150):
            toggles = list(screen.query("#library-rag-mode-toggle"))
            if toggles and str(toggles[0].label) == "mode: RAG Answer ▸":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Mode toggle never switched to RAG Answer.")

        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False

        run_button.press()
        await _wait_for_selector(screen, pilot, "#library-rag-service-error")

        visible_text = _visible_text(screen)
        assert "RAG unavailable" in visible_text
        # (Task-14 enabler) the recovery copy now names the pip extra. The
        # display sanitizer's `escape_markup` pass backslash-escapes the
        # opening "[" (same reason `library-rag-history-*` labels escape
        # entries -- unescaped, Rich would try to parse "[embeddings_rag]"
        # as a style tag); the backslash resolves away at real paint time
        # and is not visible to the user, but `_visible_text` reads
        # `.renderable` directly, before that resolution.
        assert (
            'Install RAG support: pip install "tldw_chatbook\\[embeddings_rag]", '
            "then restart, or switch mode to Search." in visible_text
        )
        # (2026-08-03 task-15 finding-1 fix) The display sanitizer no longer
        # HTML-entity-escapes plain text for display -- a Rich `Static`
        # never decodes "&gt;" back to ">", so re-encoding here was itself
        # the over-escaping bug finding 1 fixed (for "&" in evidence
        # snippets; ">" in this recovery copy is the same class of bug).
        # The recovery route's ">" now renders as the literal character.
        assert "Recovery: Settings > RAG." in visible_text


@pytest.mark.asyncio
async def test_library_search_rag_run_query_preserves_panel_instances_during_updates() -> (
    None
):
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = DelayedLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                }
            ],
        }
    )
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)
        panel = screen.query_one("#library-search-rag-panel")

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-searching-line")

        assert screen.query_one("#library-search-rag-panel") is panel

        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        assert screen.query_one("#library-search-rag-panel") is panel


@pytest.mark.asyncio
async def test_library_search_rag_worker_completion_ignores_unmounted_screen(
    monkeypatch,
) -> None:
    app = _build_test_app()
    screen = LibraryScreen(app)
    # Put the screen in the state where every guard in
    # _apply_library_rag_search_outcome passes EXCEPT is_mounted, so that the
    # mount guard is the sole thing preventing the DOM refresh. The code under
    # test refreshes via _refresh_search_rag_panel_state_widgets (not the stale
    # _sync_search_rag_panel), so that is the method the test must poison.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SEARCH
    screen._library_rag_query = "Find evidence"
    monkeypatch.setattr(screen, "query", lambda *args, **kwargs: [object()])

    async def fail_refresh() -> None:
        raise AssertionError("unmounted worker completion should not touch the DOM")

    monkeypatch.setattr(screen, "_refresh_search_rag_panel_state_widgets", fail_refresh)

    assert screen.is_mounted is False
    await screen._apply_library_rag_search_outcome(
        LibraryRagSearchRequest(
            query="Find evidence",
            source_types=("notes",),
        ),
        LibraryRagSearchOutcome(status="ready"),
    )


# --- PR-3 Task 4: two-phase wiring (retrieve, THEN generate) ---------------
#
# Retrieval and generation are two workers in two DIFFERENT exclusive groups
# (`library_rag_search` / `library_rag_answer`). Sharing one group would let a
# newly-started search silently cancel an in-flight answer without ever
# resolving its "answering" status -- the dangling-status hazard this file's
# `test_library_shell_search_outcome_resolves_status_after_leaving_canvas`
# sibling already pins for retrieval.
#
# The provider seam is `app.library_rag_answer_chat`, resolved by
# `LibraryScreen._library_rag_answer_chat_kwargs`. `Tests/UI/app_factory.py`
# sets it to `None` on every test app, which DISABLES generation -- so no
# pre-existing pilot can reach a provider, and a test that wants generation
# opts in by assigning a fake (as every test below does). Nothing here can
# reach a network.

GROUNDED_ANSWER = "An expired credential caused the incident [S1]."

# Bounds a gated fake's wait so a forgotten release can't wedge pytest
# shutdown (mirrors `_GATED_RELEASE_TIMEOUT_SECONDS` in test_library_shell.py).
_ANSWER_RELEASE_TIMEOUT_SECONDS = 5.0


class RecordingAnswerChat:
    """The one faked provider seam for Library RAG answer generation.

    Deliberately a SYNC callable: the real `chat_api_call` is synchronous and
    `library_rag_answer_service._invoke_chat` offloads it with
    `asyncio.to_thread`, so a sync fake exercises that same path -- and lets
    `release_event` block a worker thread instead of the event loop.
    """

    def __init__(self, *, replies=None, log=None, gated=False):
        self.replies = list(replies) if replies is not None else None
        self.calls: list[dict] = []
        self.log = log if log is not None else []
        self.release_event = threading.Event() if gated else None

    def __call__(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        self.log.append("answer")
        if self.release_event is not None:
            self.release_event.wait(_ANSWER_RELEASE_TIMEOUT_SECONDS)
        if self.replies is None:
            return GROUNDED_ANSWER
        return self.replies[min(index, len(self.replies) - 1)]


def _rag_result_fixture(*, with_coverage_gap: bool = False) -> dict:
    """One retrievable row, in the shape the local retrieval seam returns."""
    result = {
        "results": [
            {
                "document_title": "Incident Review",
                "snippet": "Expired credential caused the incident.",
                "score": "0.93",
                "source_id": "note-42",
                "chunk_id": "chunk-7",
                "runtime_backend": "rag-semantic",
                "provenance": {"source_type": "note"},
            }
        ],
        "runtime_backend": "rag-semantic",
    }
    if with_coverage_gap:
        result["diagnostics"] = {
            "semantic_scope_coverage": {
                "covered": ["notes"],
                "uncovered": ["media", "conversations"],
            }
        }
    return result


async def _switch_to_rag_mode(screen, pilot) -> None:
    """Cycle the Search canvas from its default keyword mode into RAG Answer."""
    await _wait_for_selector(screen, pilot, "#library-rag-mode-toggle")
    screen.query_one("#library-rag-mode-toggle", Button).press()
    for _ in range(150):
        toggles = list(screen.query("#library-rag-mode-toggle"))
        if toggles and str(toggles[0].label) == "mode: RAG Answer ▸":
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError("Mode toggle never switched to RAG Answer.")


async def _wait_until(pilot, predicate, message: str, *, attempts: int = 200) -> None:
    for _ in range(attempts):
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError(message)


def _spy_panel_statuses(monkeypatch, screen) -> list[str]:
    """Record the `retrieval_status` of every panel state built from here on.

    Every render path (compose and each incremental refresh) goes through
    `_library_rag_panel_state`, so this is how a test can assert a transient
    status was -- or was never -- entered, instead of only inspecting the
    settled end state.
    """
    statuses: list[str] = []
    original = screen._library_rag_panel_state

    def spy():
        state = original()
        statuses.append(state.retrieval_status)
        return state

    monkeypatch.setattr(screen, "_library_rag_panel_state", spy)
    return statuses


def test_answer_region_flags_a_non_validated_status_with_no_recovery_copy() -> None:
    """(Task 3 review finding, folded into Task 4) `citation_status` and
    `citation_recovery` are set together by `build_answer_citation_validation`,
    but that invariant lives in another module and is unenforced at the
    dataclass level. If a non-`validated` status ever arrives with empty
    recovery copy, the answer must still be flagged -- rendering it as a clean
    grounded answer is exactly the plausible-looking-but-unverified failure
    this whole feature exists to prevent."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_READY,
        LibraryRagAnswer,
    )
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library import library_rag_answer_children

    answer = LibraryRagAnswer(
        status=ANSWER_STATUS_READY,
        text="An expired credential caused the incident.",
        citation_status="unknown_future_status",
        citation_recovery="",
    )
    state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Why did the incident happen?",
        mode="rag",
        answer=answer,
    )
    region = library_rag_answer_children(state)[0]
    region_children = _answer_region_children(region)
    ids_in_order = [child.id for child in region_children]
    assert ids_in_order.index("library-rag-answer-caution") < ids_in_order.index(
        "library-rag-answer-text"
    )
    caution = next(
        child for child in region_children if child.id == "library-rag-answer-caution"
    )
    assert caution.has_class("library-rag-callout")
    assert caution.has_class("is-caution")
    caution_text = str(caution.renderable)
    assert caution_text.strip()
    assert "verified" not in caution_text.lower()
    # The clean-answer note must NOT also render -- it would contradict the
    # caution standing right above it.
    assert not any(
        child.id == "library-rag-answer-citation-note" for child in region_children
    )


@pytest.mark.asyncio
async def test_library_search_rag_rag_mode_generates_answer_after_results_land(
    monkeypatch,
) -> None:
    """(a) The two phases, in order: a rag-mode query retrieves evidence and
    only THEN calls the provider, with the retrieved snippets and the panel's
    own retrieval-coverage note in the prompt. The in-flight "answering"
    status is really entered (the run gate says so) and is settled by the
    answer's arrival."""
    log: list[str] = []
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        _rag_result_fixture(with_coverage_gap=True), log=log
    )
    chat = RecordingAnswerChat(log=log)
    app.library_rag_answer_chat = chat
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _switch_to_rag_mode(screen, pilot)
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        statuses = _spy_panel_statuses(monkeypatch, screen)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-answer-text")

        # Retrieval first, generation second -- one provider call, no more.
        assert log == ["search", "answer"]
        assert len(chat.calls) == 1

        user_message = chat.calls[0]["messages_payload"][0]["content"]
        assert "Expired credential caused the incident." in user_message
        # Coverage-note pass-through: what retrieval did NOT reach travels
        # into the prompt, so "your media says nothing" can never be
        # confused with "your media was never searched".
        assert "Semantic search found nothing from: Media, Conversations." in (
            user_message
        )

        # Evidence stayed on screen alongside the answer.
        assert screen.query("#library-rag-result-0")
        answer_text = str(screen.query_one("#library-rag-answer-text").renderable)
        assert "An expired credential caused the incident" in answer_text

        assert screen._library_rag_answer is not None
        assert screen._library_rag_answer.status == "ready"
        assert screen._library_rag_answer_query == query
        assert screen._library_rag_answer_mode == "rag"
        assert screen._library_rag_answer_in_flight is False

        # The in-flight status was really entered, and really settled.
        assert "answering" in statuses
        assert statuses[-1] == "ready"
        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False
        assert str(run_button.label) != "Answering…"


@pytest.mark.asyncio
async def test_library_search_rag_mode_toggle_mid_answer_discards_stale_answer() -> (
    None
):
    """(b) Toggling out of RAG Answer mode while the provider call is still
    in flight must discard that answer when it lands -- it belongs to the mode
    the user has since left -- and must leave no dangling "answering" status.
    Mirrors `test_library_shell_search_mode_toggle_mid_flight_discards_wrong_mode_outcome`
    for the second phase."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_READY,
        LibraryRagAnswer,
    )

    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        _rag_result_fixture()
    )
    chat = RecordingAnswerChat(gated=True)
    app.library_rag_answer_chat = chat
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _switch_to_rag_mode(screen, pilot)
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        try:
            screen.query_one("#library-rag-run-query", Button).press()
            await _wait_until(
                pilot,
                lambda: bool(chat.calls),
                "Generation never reached the gated chat seam.",
            )
            assert screen._library_rag_answer_in_flight is True
            assert screen.query("#library-rag-answer-status")

            # The mode guard, exercised directly while the fields are still
            # set: an outcome carrying the mode the user has left is dropped
            # even though its query still matches.
            await screen._apply_library_rag_answer(
                LibraryRagSearchRequest(
                    query=query,
                    source_types=("notes",),
                    mode="search",
                ),
                LibraryRagAnswer(status=ANSWER_STATUS_READY, text="wrong-mode answer"),
            )
            assert screen._library_rag_answer is None

            screen.query_one("#library-rag-mode-toggle", Button).press()
            await _wait_until(
                pilot,
                lambda: screen._library_rag_mode == "search",
                "Mode toggle never switched back to keyword Search.",
            )
        finally:
            chat.release_event.set()

        for _ in range(30):
            await pilot.pause(0.02)

        assert screen._library_rag_answer is None
        assert screen._library_rag_answer_query == ""
        assert screen._library_rag_answer_mode == ""
        assert screen._library_rag_answer_in_flight is False
        assert not screen.query("#library-rag-answer")
        assert screen._library_rag_panel_state().retrieval_status != "answering"


@pytest.mark.asyncio
async def test_library_search_rag_new_search_mid_answer_leaves_no_dangling_status() -> (
    None
):
    """(c) A new search started while an answer is in flight must not leave
    the panel stuck on "answering", and the stale answer must never overwrite
    the new one. The answer worker lives in its OWN exclusive group precisely
    so a new SEARCH cannot cancel it out from under its own status."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_READY,
        LibraryRagAnswer,
    )

    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        _rag_result_fixture()
    )
    chat = RecordingAnswerChat(
        gated=True,
        replies=[
            "Stale answer from the first query [S1].",
            "Fresh answer from the second query [S1].",
        ],
    )
    app.library_rag_answer_chat = chat
    host = DestinationHarness(app, "library")
    first_query = "Why did the incident happen?"
    second_query = "What changed after the incident?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _switch_to_rag_mode(screen, pilot)
        screen.query_one("#library-rag-query-input", Input).value = first_query
        await _wait_for_query_ready(screen, pilot, first_query)

        try:
            screen.query_one("#library-rag-run-query", Button).press()
            await _wait_until(
                pilot,
                lambda: bool(chat.calls),
                "The first generation never reached the gated chat seam.",
            )
            assert screen._library_rag_answer_in_flight is True

            # Typing un-sticks the run gate (the same escape hatch the
            # retrieval phase already offers), so a new search can start
            # while the first answer is still blocked in the provider.
            query_input = screen.query_one("#library-rag-query-input", Input)
            query_input.value = second_query
            await screen.update_library_rag_query(
                Input.Changed(query_input, second_query)
            )
            await _wait_for_query_ready(screen, pilot, second_query)
            assert screen._library_rag_answer_in_flight is False

            screen.query_one("#library-rag-run-query", Button).press()
            await _wait_until(
                pilot,
                lambda: len(chat.calls) == 2,
                "The second generation never reached the gated chat seam.",
            )

            # The query guard, exercised directly: the first query's answer
            # is dropped now that a newer search owns the panel.
            await screen._apply_library_rag_answer(
                LibraryRagSearchRequest(
                    query=first_query,
                    source_types=("notes",),
                    mode="rag",
                ),
                LibraryRagAnswer(status=ANSWER_STATUS_READY, text="stale answer"),
            )
            assert screen._library_rag_answer is None
        finally:
            chat.release_event.set()

        await _wait_until(
            pilot,
            lambda: screen._library_rag_answer is not None,
            "The second query's answer never landed.",
        )
        for _ in range(20):
            await pilot.pause(0.02)

        assert "Fresh answer from the second query" in screen._library_rag_answer.text
        assert screen._library_rag_answer_query == second_query
        assert screen._library_rag_answer_in_flight is False
        assert screen._library_rag_panel_state().retrieval_status == "ready"
        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False
        assert str(run_button.label) != "Answering…"


@pytest.mark.asyncio
async def test_library_search_rag_keyword_mode_never_calls_the_answer_seam() -> None:
    """(d) Keyword Search mode answers nothing and calls no provider, ever --
    it is a retrieval mode. Asserted on the fake itself, not on the absence of
    a widget."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        _rag_result_fixture()
    )
    chat = RecordingAnswerChat()
    app.library_rag_answer_chat = chat
    host = DestinationHarness(app, "library")
    query = "incident"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        assert screen._library_rag_mode == "search"
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        for _ in range(25):
            await pilot.pause(0.02)

        assert chat.calls == []
        assert screen._library_rag_answer is None
        assert screen._library_rag_answer_in_flight is False
        assert not screen.query("#library-rag-answer")


@pytest.mark.asyncio
async def test_library_search_rag_zero_results_answer_needs_no_provider(
    monkeypatch,
) -> None:
    """(e) A rag query whose retrieval found nothing still gets an answer --
    the honest one -- and it costs no provider call: handing a model an
    evidence block with nothing citable can only produce an abstention or a
    guess. The quiet no-match state that
    `test_empty_status_renders_quiet_two_line_state_not_full_dump` pins is
    untouched: it explains what retrieval did and how to widen it, while the
    answer says what the library can tell you. The panel never passes through
    "answering" here either -- an in-flight indicator for a call that is never
    made would replace the quiet no-match line with the idle
    "No evidence yet" line for no reason."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService({"results": []})
    chat = RecordingAnswerChat()
    app.library_rag_answer_chat = chat
    host = DestinationHarness(app, "library")
    query = "unicorn migration guide"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _switch_to_rag_mode(screen, pilot)
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        statuses = _spy_panel_statuses(monkeypatch, screen)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-empty-state")
        await _wait_until(
            pilot,
            lambda: screen._library_rag_answer is not None,
            "The no-evidence answer never landed.",
        )

        assert chat.calls == []
        assert screen._library_rag_answer.status == "no_evidence"
        answer_text = str(screen.query_one("#library-rag-answer-text").renderable)
        assert answer_text == "Nothing in your library supports an answer to that."

        # The quiet no-match state is exactly as Task 11 pinned it.
        quiet_line = screen.query_one("#library-rag-empty-state", Static)
        assert quiet_line.has_class("library-rag-quiet-line")
        assert str(quiet_line.renderable) == (
            "No evidence matched 'unicorn migration guide'.\nTry broader terms."
        )
        assert not screen.query("#library-rag-results-empty")
        assert "answering" not in statuses
        assert screen._library_rag_retrieval_status == "empty"
        assert screen._library_rag_answer_in_flight is False


def test_library_rag_answer_state_survives_restore_without_dangling_status() -> None:
    """Save/restore carries the answer with the results it belongs to (the
    same lifecycle `_library_rag_diagnostics`/`_library_rag_searched_query`
    already follow) -- but NEVER the in-flight flag: the restored screen is a
    brand-new instance with no worker running, so a restored "answering"
    status could never be resolved by anything."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_READY,
        LibraryRagAnswer,
    )

    app = _build_test_app()
    screen = LibraryScreen(app)
    row = LibraryRagResultRow.from_result(
        {
            "document_title": "Incident Review",
            "snippet": "Expired credential caused the incident.",
            "source_id": "note-42",
        }
    )
    answer = LibraryRagAnswer(
        status=ANSWER_STATUS_READY,
        text="An expired credential caused the incident [S1].",
        citation_status="validated",
    )
    screen._library_rag_mode = "rag"
    screen._library_rag_query = "Why did the incident happen?"
    screen._library_rag_searched_query = "Why did the incident happen?"
    screen._library_rag_results = (row,)
    screen._library_rag_retrieval_status = "ready"
    screen._library_rag_answer = answer
    screen._library_rag_answer_query = "Why did the incident happen?"
    screen._library_rag_answer_mode = "rag"
    # Navigating away mid-generation is exactly when this matters.
    screen._library_rag_answer_in_flight = True

    restored = LibraryScreen(app)
    restored.restore_state(screen.save_state())

    assert restored._library_rag_answer == answer
    assert restored._library_rag_answer_query == "Why did the incident happen?"
    assert restored._library_rag_answer_mode == "rag"
    assert restored._library_rag_answer_in_flight is False
    # A restored instance has not run `_refresh_local_source_snapshot` yet,
    # so its scope counts are all 0 and the run gate would read "blocked";
    # seed them so the assertion below is about the answer status overlay
    # rather than about an unloaded scope.
    restored._local_source_counts = {"notes": 1, "media": 1, "conversations": 1}
    assert restored._library_rag_panel_state().retrieval_status == "ready"


@pytest.mark.asyncio
async def test_library_search_rag_disabled_answer_seam_generates_nothing() -> None:
    """The network-safety contract, pinned rather than assumed: an app whose
    `library_rag_answer_chat` is present but not callable generates nothing
    at all -- no worker, no answer, no "answering" status. That is the state
    `Tests/UI/app_factory.py` leaves EVERY test app in, which is what keeps
    the pre-existing rag-mode pilots (which never opted into a fake chat) off
    a real provider. An app with no such attribute at all -- the shipping
    `TldwCli` -- gets the service's own `chat_api_call` default instead."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        _rag_result_fixture()
    )
    assert app.library_rag_answer_chat is None  # what the factory installs
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _switch_to_rag_mode(screen, pilot)
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        for _ in range(25):
            await pilot.pause(0.02)

        assert screen._library_rag_answer is None
        assert screen._library_rag_answer_in_flight is False
        assert not screen.query("#library-rag-answer")
        assert screen._library_rag_panel_state().retrieval_status == "ready"


class GatedLibraryRagSearchService(StaticLibraryRagSearchService):
    """A `search` the test releases, so the window between "a new search
    started" and "its outcome landed" can be held open and inspected."""

    def __init__(self, result, *, log=None):
        super().__init__(result, log=log)
        self.release_event = threading.Event()

    async def search(self, query, scope, mode, **kwargs):
        self.calls.append({"query": query, "scope": scope, "mode": mode, **kwargs})
        self.log.append("search")
        await asyncio.to_thread(
            self.release_event.wait, _ANSWER_RELEASE_TIMEOUT_SECONDS
        )
        return self.result


@pytest.mark.asyncio
async def test_library_search_rag_starting_a_new_search_drops_the_previous_answer() -> (
    None
):
    """A landed answer belongs to the results it was grounded in, so starting
    a new search must drop it IMMEDIATELY -- not once the new outcome
    arrives. The window in between (retrieval running, results already
    cleared) is the one where a leftover answer would be displayed against
    evidence that is no longer on screen; clearing the guard fields is also
    what makes an older generation's arrival inside that window a discarded
    no-op."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_READY,
        LibraryRagAnswer,
    )

    app = _build_test_app()
    _seed_library_sources(app)
    service = GatedLibraryRagSearchService(_rag_result_fixture())
    service.release_event.set()  # the first query runs straight through
    app.library_rag_search_service = service
    app.library_rag_answer_chat = RecordingAnswerChat()
    host = DestinationHarness(app, "library")
    first_query = "Why did the incident happen?"
    second_query = "What changed after the incident?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _switch_to_rag_mode(screen, pilot)
        screen.query_one("#library-rag-query-input", Input).value = first_query
        await _wait_for_query_ready(screen, pilot, first_query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-answer-text")
        assert screen._library_rag_answer is not None

        # Hold the SECOND retrieval open.
        service.release_event.clear()
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = second_query
        await screen.update_library_rag_query(Input.Changed(query_input, second_query))
        await _wait_for_query_ready(screen, pilot, second_query)

        try:
            screen.query_one("#library-rag-run-query", Button).press()
            await _wait_until(
                pilot,
                lambda: len(service.calls) == 2,
                "The second search never reached the gated service.",
            )

            # Mid-search: the previous answer is already gone, together with
            # the guards that would otherwise let a late arrival re-apply it.
            assert screen._library_rag_answer is None
            assert screen._library_rag_answer_query == ""
            assert screen._library_rag_answer_mode == ""
            assert not screen.query("#library-rag-answer")
            await screen._apply_library_rag_answer(
                LibraryRagSearchRequest(
                    query=first_query,
                    source_types=("notes",),
                    mode="rag",
                ),
                LibraryRagAnswer(status=ANSWER_STATUS_READY, text="stale answer"),
            )
            assert screen._library_rag_answer is None
        finally:
            service.release_event.set()

        await _wait_until(
            pilot,
            lambda: screen._library_rag_answer is not None,
            "The second query's answer never landed.",
        )
        assert screen._library_rag_answer_query == second_query
        assert screen._library_rag_answer_in_flight is False


def test_library_rag_answer_seam_absent_resolves_to_the_shipping_default() -> None:
    """(PR-3 Task 4 review) The path PRODUCTION takes, which no other test
    exercises: the shipping `TldwCli` carries no `library_rag_answer_chat`
    attribute at all, so the screen sends no `chat=` override and the answer
    service resolves its own `chat_api_call`. Every other test here runs
    against the factory's `None` (generation disabled), so without this pin a
    future `library_rag_answer_chat = None` class attribute on `TldwCli`
    would silently kill the whole feature with a green suite.

    The resolution is asserted, never invoked -- calling it would be the very
    live provider call the rest of this section exists to prevent."""
    from tldw_chatbook.Library.library_rag_answer_service import (
        _resolve_answer_chat,
        chat_api_call,
    )

    app = _build_test_app()
    # Undo the factory's test-only disable to recreate the shipping shape.
    del app.library_rag_answer_chat
    assert not hasattr(app, "library_rag_answer_chat")

    screen = LibraryScreen(app)
    chat_kwargs = screen._library_rag_answer_chat_kwargs()

    # No override -> the service's own default...
    assert chat_kwargs == {}
    # ...which resolves to the real provider entry point.
    assert _resolve_answer_chat(chat_kwargs.get("chat")) is chat_api_call

    # And the two other cases still hold on the same helper.
    app.library_rag_answer_chat = None
    assert screen._library_rag_answer_chat_kwargs() is None  # disabled
    fake = RecordingAnswerChat()
    app.library_rag_answer_chat = fake
    assert screen._library_rag_answer_chat_kwargs() == {"chat": fake}


@pytest.mark.asyncio
async def test_panel_refresh_rechecks_the_panel_after_taking_the_lock() -> None:
    """(PR-3 Task 4 review) The panel-presence check happens before the
    refresh lock, and the lock inserts an unbounded wait between that check
    and the `query_one` calls that act on it -- so the panel can be gone by
    the time a queued refresh actually runs (a rail switch, or a recompose,
    while a prior refresh held the lock). Without a re-check inside the lock
    every `query_one` raises `NoMatches` out of whichever worker called it.

    Forced deterministically here: hold the lock, queue a refresh behind it,
    tear the panel out, then release."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        _rag_result_fixture()
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        async with screen._library_rag_panel_refresh_lock:
            queued = asyncio.create_task(
                screen._refresh_search_rag_panel_state_widgets()
            )
            # Let it clear the pre-lock check and park on the lock.
            for _ in range(5):
                await asyncio.sleep(0)
            # The panel goes away while the refresh is parked.
            await screen.query_one("#library-search-rag-panel").remove()

        await queued  # must return quietly, not raise NoMatches
        assert queued.done() and queued.exception() is None
