"""Library shell (L1) rail + conversations canvas pilot contracts."""

import asyncio
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Input, Markdown, Static, TextArea

from tldw_chatbook.app import LibraryIngestQueueMixin
from Tests.Library.test_library_ingest_runner import _FakeIngestParsePool
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_INGEST,
    LIBRARY_NAV_CONTEXT_NOTE_ID,
    LIBRARY_NAV_CONTEXT_NOTES_CREATE,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.Library.library_rag_state import LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_export_state import EMPTY_SCOPE_COPY
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP,
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_BROWSE_SEARCH,
    LIBRARY_ROW_CREATE_NOTE,
    LIBRARY_ROW_INGEST_EXPORT,
    LIBRARY_ROW_INGEST_MEDIA,
)
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Study_Interop.local_quiz_service import LocalQuizService
from tldw_chatbook.Study_Interop.local_study_service import LocalStudyService
from tldw_chatbook.Study_Interop.quiz_scope_service import QuizScopeService
from tldw_chatbook.Study_Interop.study_scope_service import StudyScopeService
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, FileSave
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_ingest_canvas import LibraryIngestCanvas
from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesListScopeService,
    StaticLibraryNotesScopeService,
    _link_library_items_to_active_workspace,
)
from Tests.UI.test_library_content_hub import StaticLibraryCollectionsService
from Tests.UI.test_screen_navigation import _build_test_app

LIBRARY_TEST_SIZE = (170, 48)


# --- D1: capped, markup-escaped carries-forward line (pure logic) ----------


def test_library_carries_forward_line_lists_up_to_three_titles_with_no_cap_suffix():
    line = library_screen_module._library_carries_forward_line(
        ["Research Note", "Transcript A", "Planning Chat"]
    )

    assert line == "Carries forward: Research Note, Transcript A, Planning Chat"


def test_library_carries_forward_line_caps_at_three_and_counts_the_rest():
    line = library_screen_module._library_carries_forward_line(
        ["Research Note", "Transcript A", "Planning Chat", "Design Doc", "Roadmap"]
    )

    assert line == "Carries forward: Research Note, Transcript A, Planning Chat and 2 more."


def test_library_carries_forward_line_escapes_markup_in_titles():
    line = library_screen_module._library_carries_forward_line(["[bold]Unsafe[/bold] title"])

    assert line == r"Carries forward: \[bold]Unsafe\[/bold] title"


# Gated fakes block a real executor thread on a threading.Event until a test
# releases it. A test that fails (or forgets to release) before that point
# would leave the thread parked on an unbounded ``Event.wait()`` -- and the
# asyncio default executor joins its threads at loop/interpreter shutdown, so
# one un-released gate wedges the WHOLE pytest process at exit with no output
# (observed: three full-suite runs hung in ``wait_for_thread_shutdown``). A
# generous bound lets the thread free itself so shutdown always completes;
# passing tests release within milliseconds, far inside this window.
_GATED_RELEASE_TIMEOUT_SECONDS = 30.0


@pytest.fixture(autouse=True)
def _stub_library_search_history_cli_fallback(monkeypatch):
    """Isolate ``LibraryScreen`` construction from the real on-disk CLI config.

    ``_load_library_search_history`` falls back to ``get_cli_setting`` when
    ``app_config`` has no in-memory history yet (the Issue 1 fix: recover
    persisted history after a restart). Tests share one real ``HOME`` /
    ``config.toml`` across the whole pytest session -- other tests'
    ``_record_library_search_history`` calls persist to that same file via
    a background ``save_setting_to_cli_config`` worker -- so without this
    stub, a freshly constructed screen would non-deterministically inherit
    whatever ``[library.search] history`` a prior test (or prior session)
    happened to leave on disk instead of starting clean. Tests that want to
    exercise the CLI-config fallback itself re-patch
    ``library_screen_module.get_cli_setting`` after this fixture runs, which
    takes precedence for the remainder of the test.

    This blanket stub (it returns ``None`` for *any* ``get_cli_setting``
    call, not just ``"library.search"``) also isolates
    ``_library_rail_preferences``'s own ``get_cli_setting("library.rail_state")``
    fallback (C4) from the same on-disk leakage, for the same reason --
    tests that exercise *that* fallback specifically also re-patch
    ``library_screen_module.get_cli_setting`` after this fixture runs.
    """
    monkeypatch.setattr(
        library_screen_module, "get_cli_setting", lambda *args, **kwargs: None
    )


class LibraryHarness(App):
    """Mount a single LibraryScreen with the real app stylesheet."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self, app_instance, seen_routes=None, screen=None):
        super().__init__()
        self.app_instance = app_instance
        self.seen_routes = seen_routes if seen_routes is not None else []
        self.seen_contexts = []
        self._screen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self._screen or LibraryScreen(self.app_instance))

    def on_navigate_to_screen(self, message) -> None:
        self.seen_routes.append(message.screen_name)
        self.seen_contexts.append(dict(message.screen_context or {}))


def _active_library_screen(host: LibraryHarness):
    return host.screen_stack[-1]


def _visible_text(screen) -> str:
    chunks = []
    for widget in screen.query("Static"):
        renderable = getattr(widget, "renderable", "")
        chunks.append(getattr(renderable, "plain", str(renderable)))
    for widget in screen.query("Button"):
        label = getattr(widget, "label", "")
        chunks.append(str(label) if label is not None else "")
    return " ".join(chunks)


def _seed_conversations(app, conversations, *, notes=None, media=None, highlights=None):
    app.notes_scope_service = StaticLibraryNotesScopeService(notes or [])
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        media or [], highlights=highlights
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        conversations
    )


async def _wait_for_library_shell(screen, pilot, *, attempts=120):
    for _ in range(attempts):
        if getattr(screen, "_library_loaded", False) and screen.query("#library-rail"):
            await pilot.pause()
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError(
        f"Library shell never loaded. Visible text: {_visible_text(screen)}"
    )


async def _wait_for_selector(screen, pilot, selector, *, attempts=120):
    for _ in range(attempts):
        matches = list(screen.query(selector))
        if matches:
            await pilot.pause()
            return matches[0]
        await pilot.pause(0.02)
    raise AssertionError(
        f"{selector} never mounted. Visible text: {_visible_text(screen)}"
    )


def _two_conversations():
    return [
        {
            "title": "Quarterly planning sync",
            "conversation_id": "chat-1",
            "message_count": 8,
            "updated_at": "2026-06-01T10:00:00Z",
        },
        {
            "title": "Design review notes",
            "conversation_id": "chat-2",
            "message_count": 3,
            "updated_at": "2026-06-02T09:30:00Z",
        },
    ]


def _two_media_items():
    return [
        {
            "id": "media-1",
            "title": "Interview Recording",
            "type": "audio",
            "last_modified": "2026-07-06T08:00:00Z",
            "author": "Jordan Lee",
            "keywords": ["interview", "audio"],
            "content": "Full transcript: the interview recording covers the quarterly roadmap.",
            "version": 1,
        },
        {
            "id": "media-2",
            "title": "Product Demo Video",
            "type": "video",
            "last_modified": "2026-07-06T10:00:00Z",
            "author": "Morgan Lee",
            "keywords": ["demo", "video"],
            "content": "Full transcript: the product demo video walks through the new dashboard.",
            "version": 2,
        },
    ]


@pytest.mark.asyncio
async def test_library_shell_renders_rail_sections_and_landing_canvas():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        header = str(screen.query_one("#library-header-line").renderable)
        assert header == "Library | Local"

        for selector in (
            "#library-rail-section-header-browse",
            "#library-rail-section-header-create",
            "#library-rail-section-header-ingest",
            "#library-rail-section-header-details",
        ):
            assert screen.query_one(selector)

        visible = _visible_text(screen)
        assert "Conversations (2)" in visible
        assert "Search, pick a content type, or ingest something new." in visible
        assert screen.query_one("#library-canvas-landing")

        assert not screen.query("#library-mode-bar")
        assert not screen.query("#library-contract-grid")
        assert not screen.query("#library-notes-summary")


@pytest.mark.asyncio
async def test_library_shell_browse_conversations_renders_canvas():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")

        rows = list(screen.query(".library-conversation-row"))
        assert len(rows) == 2
        first_label = str(rows[0].label)
        assert first_label.startswith("▸")

        preview = str(screen.query_one("#library-conversation-preview-lines").renderable)
        assert "Messages:" in preview
        assert screen.query_one("#library-conversation-open-console")


@pytest.mark.asyncio
async def test_library_shell_conversation_row_switches_selection():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-1")

        # Rows sort newest-first: chat-2 (06-02) is row 0, chat-1 (06-01) is row 1.
        preview_before = str(
            screen.query_one("#library-conversation-preview-lines").renderable
        )
        assert "Design review notes" in preview_before
        assert screen._selected_conversation_id == "chat-2"

        screen.query_one("#library-conversation-row-1").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._selected_conversation_id == "chat-1"
        preview_after = str(
            screen.query_one("#library-conversation-preview-lines").renderable
        )
        assert "Quarterly planning sync" in preview_after


@pytest.mark.asyncio
async def test_library_shell_open_in_console_triggers_handoff():
    app = _build_test_app()
    _seed_conversations(
        app,
        [
            {
                "title": "Planning Chat",
                "conversation_id": "chat-1",
                "message_count": 7,
                "updated_at": "2026-06-03T10:15:00Z",
            }
        ],
    )
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("conversation", "chat-1", "Planning Chat"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-open-console")

        screen.query_one("#library-conversation-open-console").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "library"
    assert payload.item_type == "conversation"
    assert payload.source_id == "chat-1"


@pytest.mark.asyncio
async def test_library_shell_flashcards_row_renders_handoff_canvas():
    """Create > Flashcards is a "handoff" rail row (L3b Task 8, not the
    retired "mode" kind): pressing it renders the consolidated handoff
    canvas (UX wave D1) -- one header (the row's own title), one purpose
    line, the capped carries-forward line, one ownership line, the ready
    snapshot line (plain, D2), and the primary "Open Flashcards" action
    button (D3). The duplicated mode/purpose lines, the "Primary action:"
    line, the "Flashcards handoff" sub-header, and the WIP roadmap callout
    from the pre-D1 layout are gone.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-flashcards").press()
        await _wait_for_selector(screen, pilot, "#library-study-handoff-detail")

        canvas = screen.query_one("#library-canvas")
        detail = screen.query_one("#library-study-handoff-detail")
        assert canvas in detail.ancestors

        # Header: the row's own title, not a second "X mode" restatement.
        title = screen.query_one("#library-active-mode-title", Static)
        assert str(title.renderable) == "Flashcards"
        assert title.has_class("destination-section")

        # Removed duplicated mode/purpose lines and the WIP/primary-action
        # lines: they no longer render at all.
        assert not screen.query("#library-active-mode-description")
        assert not screen.query("#library-active-mode-next-action")
        assert not screen.query("#library-study-handoff-primary-action")
        assert not screen.query("#library-study-handoff-wip")

        purpose = screen.query_one("#library-study-handoff-purpose", Static)
        assert str(purpose.renderable) == "Generate or review cards from Library sources."
        visible = _visible_text(screen)
        assert "Flashcards handoff" not in visible
        assert "Primary action:" not in visible
        assert "WIP:" not in visible

        context = screen.query_one("#library-study-handoff-context", Static)
        assert str(context.renderable) == (
            "Carries forward: Quarterly planning sync, Design review notes"
        )

        owner = screen.query_one("#library-study-handoff-owner", Static)
        assert str(owner.renderable) == "Generation and review run in Study."

        # D2: ready state is a plain line, no warning-callout classes.
        recovery = screen.query_one("#library-study-handoff-recovery", Static)
        assert str(recovery.renderable) == "Source snapshot is ready."
        assert not recovery.has_class("ds-recovery-callout")
        assert not recovery.has_class("is-blocked")

        # D3: the Open action carries primary emphasis.
        open_button = screen.query_one("#library-open-flashcards", Button)
        assert canvas in open_button.ancestors
        assert open_button.has_class("console-action-primary")


@pytest.mark.asyncio
async def test_library_shell_handoff_canvas_button_reads_continue_in_study():
    """UX wave L2: the handoff canvas action button reads as a verb
    ("Continue in Study") for every study handoff kind, instead of
    restating the destination's own name a second time -- the header
    already says "Flashcards"/"Study decks"/"Quizzes". Header/purpose
    still use the mode's own copy (unchanged); button ids are unchanged.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        for row_id, button_id, header in (
            ("#library-row-create-flashcards", "#library-open-flashcards", "Flashcards"),
            ("#library-row-create-study", "#library-open-study", "Study decks"),
            ("#library-row-create-quizzes", "#library-open-quizzes", "Quizzes"),
        ):
            screen.query_one(row_id).press()
            await _wait_for_selector(screen, pilot, button_id)

            title = screen.query_one("#library-active-mode-title", Static)
            assert str(title.renderable) == header
            button = screen.query_one(button_id, Button)
            assert str(button.label) == "Continue in Study"


@pytest.mark.asyncio
async def test_library_shell_search_row_renders_first_class_canvas():
    """Browse ▸ Search/RAG is a first-class canvas row now, not a legacy
    "mode" row: pressing it mounts ``LibrarySearchRagPanel`` directly (no
    ``_compose_mode_canvas`` indirection), and -- unlike a handoff row
    (e.g. Flashcards, see the sibling test above) -- the shared mode-title
    block never renders for it.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        canvas = screen.query_one("#library-canvas")
        panel = screen.query_one("#library-search-rag-panel")
        assert canvas in panel.ancestors
        assert not screen.query("#library-active-mode-title")


@pytest.mark.asyncio
async def test_library_shell_rag_open_import_export_switches_canvas_and_selection():
    app = _build_test_app()
    # Empty sources so the Search/RAG scope recovery button renders.
    _seed_conversations(app, [])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-open-import-export")

        screen.query_one("#library-rag-open-import-export").press()
        await _wait_for_selector(
            screen, pilot, "#library-ingest-canvas"
        )

        # The canvas now renders the real Ingest canvas, driven by the shell
        # selection rather than a bare _active_mode flip. The Import/Export
        # mode this recovery button used to drive is retired -- the removed
        # row/mode's only surviving successor is the Ingest ▸ Import media
        # canvas row.
        canvas = screen.query_one("#library-canvas")
        ingest_canvas = screen.query_one("#library-ingest-canvas")
        assert canvas in ingest_canvas.ancestors
        assert not screen.query("#library-search-rag-panel")

        # ...and the rail selection marker moved to the Import media row.
        assert screen._library_selected_row_id == "ingest-import-media"
        row = screen.query_one("#library-row-ingest-import-media")
        assert row.has_class("library-rail-row-selected")


class _StaticLibraryRagSearchService:
    """Minimal recording fake for ``app.library_rag_search_service``."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    async def search(self, query, scope, mode, **kwargs):
        self.calls.append({"query": query, "scope": scope, "mode": mode, **kwargs})
        return self.result


class _GatedLibraryRagSearchService(_StaticLibraryRagSearchService):
    """A ``search`` that blocks until the test releases it, so the pilot can
    observe the in-flight ``searching`` status line before results land.

    Uses ``asyncio.to_thread`` over a ``threading.Event`` -- mirrors
    ``_GatedSearchLibraryNotesScopeService`` below; ``_GATED_RELEASE_TIMEOUT_SECONDS``
    bounds the wait so a failed/forgotten release can't wedge pytest shutdown.
    """

    def __init__(self, result):
        super().__init__(result)
        self.release_event = threading.Event()

    async def search(self, query, scope, mode, **kwargs):
        self.calls.append({"query": query, "scope": scope, "mode": mode, **kwargs})
        await asyncio.to_thread(self.release_event.wait, _GATED_RELEASE_TIMEOUT_SECONDS)
        return self.result


async def _wait_for_library_rag_query_ready(screen, pilot, query, *, attempts=150):
    for _ in range(attempts):
        inputs = list(screen.query("#library-rag-query-input"))
        buttons = list(screen.query("#library-rag-run-query"))
        if inputs and buttons and inputs[0].value == query and buttons[0].disabled is False:
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError(f"Library Search/RAG query never became ready: {query!r}")


@pytest.mark.asyncio
async def test_library_shell_search_mode_toggle_cycles_mode():
    """Pressing the mode-cycle button flips Search <-> RAG Answer, and the
    default mode on a fresh canvas is ``search``.

    A3: the toggle button label is the single mode surface now --
    ``#library-rag-query-status`` (the old "Mode: {label} | Top {k}" Static)
    is retired, so this asserts against the button label directly.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-mode-toggle")

        assert str(screen.query_one("#library-rag-mode-toggle", Button).label) == (
            "mode: Search ▸"
        )
        assert not screen.query("#library-rag-query-status")

        screen.query_one("#library-rag-mode-toggle", Button).press()
        for _ in range(120):
            toggles = list(screen.query("#library-rag-mode-toggle"))
            if toggles and str(toggles[0].label) == "mode: RAG Answer ▸":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Mode toggle never switched to RAG Answer.")

        screen.query_one("#library-rag-mode-toggle", Button).press()
        for _ in range(120):
            toggles = list(screen.query("#library-rag-mode-toggle"))
            if toggles and str(toggles[0].label) == "mode: Search ▸":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Mode toggle never switched back to Search.")


@pytest.mark.asyncio
async def test_library_shell_search_rag_mode_blocks_run_without_provider():
    """Default ``search`` mode keeps Run enabled without a provider; cycling
    to ``rag`` mode blocks Run behind the provider gate, and cycling back
    re-enables it -- the app fake here has no ``_rag_service`` attribute.
    """
    app = _build_test_app()
    assert getattr(app, "_rag_service", None) is None
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "policy question"
        await _wait_for_library_rag_query_ready(screen, pilot, "policy question")
        assert screen.query_one("#library-rag-run-query", Button).disabled is False

        screen.query_one("#library-rag-mode-toggle", Button).press()
        for _ in range(120):
            run_buttons = list(screen.query("#library-rag-run-query"))
            if run_buttons and run_buttons[0].disabled:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("RAG mode never disabled Run without a provider.")
        assert "Select a provider/model" in _visible_text(screen)

        screen.query_one("#library-rag-mode-toggle", Button).press()
        for _ in range(120):
            run_buttons = list(screen.query("#library-rag-run-query"))
            if run_buttons and not run_buttons[0].disabled:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Search mode never re-enabled Run.")


@pytest.mark.asyncio
async def test_library_shell_search_history_records_submitted_queries():
    """Running a query records it into the Recent searches collapsible and
    persists it into the in-memory ``app_config`` (most recent first).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "alpha"
        await _wait_for_library_rag_query_ready(screen, pilot, "alpha")
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-history-0")
        assert str(screen.query_one("#library-rag-history-0", Button).label) == "alpha"

        screen.query_one("#library-rag-query-input", Input).value = "beta"
        await _wait_for_library_rag_query_ready(screen, pilot, "beta")
        screen.query_one("#library-rag-run-query", Button).press()

        labels: list[str] = []
        for _ in range(150):
            rows = list(screen.query(".library-rag-history-row"))
            labels = [str(row.label) for row in rows]
            if labels == ["beta", "alpha"]:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(f"History rows never became [beta, alpha]: {labels}")

        assert app.app_config["library"]["search"]["history"] == ["beta", "alpha"]


@pytest.mark.asyncio
async def test_library_shell_search_history_loads_from_cli_config_fallback(monkeypatch):
    """Issue: a restarted app always showed empty Search/RAG history even
    though history was persisted to ``config.toml``, because ``app_config``
    (from ``load_settings()``) can come back without a ``library`` section
    at all while the on-disk CLI config still has one. ``_build_test_app``'s
    fake ``app_config`` reproduces that exact shape (no ``library`` key), so
    the screen must fall back to ``get_cli_setting`` to recover history.
    """
    app = _build_test_app()
    assert "library" not in app.app_config
    _seed_conversations(app, _two_conversations())

    calls: list[tuple] = []

    def fake_get_cli_setting(section, key=None, default=None):
        calls.append((section, key, default))
        if section == "library.search" and key is None:
            return {"history": ["alpha", "bravo"]}
        return default

    monkeypatch.setattr(library_screen_module, "get_cli_setting", fake_get_cli_setting)

    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen._library_search_history == ("alpha", "bravo")
        assert calls, "get_cli_setting fallback was never consulted"

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-history-0")
        assert str(screen.query_one("#library-rag-history-0", Button).label) == "alpha"
        assert str(screen.query_one("#library-rag-history-1", Button).label) == "bravo"


@pytest.mark.asyncio
async def test_library_shell_search_history_prefers_app_config_over_cli_config(monkeypatch):
    """Precedence: when ``app_config`` already carries a history list, the
    ``get_cli_setting`` fallback must never be consulted.
    """
    app = _build_test_app()
    app.app_config["library"] = {"search": {"history": ["from-app-config"]}}
    _seed_conversations(app, _two_conversations())

    def raising_get_cli_setting(*args, **kwargs):
        raise AssertionError(
            "get_cli_setting should not be called when app_config already has history"
        )

    monkeypatch.setattr(library_screen_module, "get_cli_setting", raising_get_cli_setting)

    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen._library_search_history == ("from-app-config",)


@pytest.mark.asyncio
async def test_library_shell_rail_preferences_loads_from_cli_config_fallback(monkeypatch):
    """(C4) Same restart-persistence gap as search history: ``app_config``
    (from ``load_settings()``) can come back without a ``library`` section
    at all even when ``config.toml`` has persisted ``[library.rail_state]``
    sections on disk -- so a freshly started app would otherwise always
    reopen every rail section at its hardcoded default instead of the
    user's last-chosen open/collapsed state. Mirrors
    ``_load_library_search_history``'s fallback template exactly (1-arg
    dotted ``get_cli_setting`` call, ``sections`` sub-key extracted from
    the returned ``rail_state`` dict).
    """
    app = _build_test_app()
    assert "library" not in app.app_config
    _seed_conversations(app, _two_conversations())

    calls: list[tuple] = []

    def fake_get_cli_setting(section, key=None, default=None):
        calls.append((section, key, default))
        if section == "library.rail_state" and key is None:
            return {"sections": {"details_open": True, "browse_open": False}}
        return default

    monkeypatch.setattr(library_screen_module, "get_cli_setting", fake_get_cli_setting)

    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        preferences = screen._library_rail_preferences()
        assert preferences.details_open is True
        assert preferences.browse_open is False
        assert calls, "get_cli_setting fallback was never consulted"


@pytest.mark.asyncio
async def test_library_shell_rail_preferences_prefers_app_config_over_cli_config(monkeypatch):
    """Precedence: when ``app_config`` already carries rail-state sections,
    the ``get_cli_setting`` fallback must never be consulted.
    """
    app = _build_test_app()
    app.app_config["library"] = {"rail_state": {"sections": {"details_open": True}}}
    _seed_conversations(app, _two_conversations())

    def raising_get_cli_setting(*args, **kwargs):
        raise AssertionError(
            "get_cli_setting should not be called when app_config already has rail state"
        )

    monkeypatch.setattr(library_screen_module, "get_cli_setting", raising_get_cli_setting)

    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen._library_rail_preferences().details_open is True


@pytest.mark.asyncio
async def test_library_shell_search_history_row_reruns_query():
    """Clicking a history row re-runs that prior query against the service."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "alpha"
        await _wait_for_library_rag_query_ready(screen, pilot, "alpha")
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-history-0")

        screen.query_one("#library-rag-query-input", Input).value = "beta"
        await _wait_for_library_rag_query_ready(screen, pilot, "beta")
        screen.query_one("#library-rag-run-query", Button).press()

        for _ in range(150):
            rows = list(screen.query(".library-rag-history-row"))
            labels = [str(row.label) for row in rows]
            if labels == ["beta", "alpha"]:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(f"History rows never became [beta, alpha]: {labels}")

        # (C5a) History recording happens synchronously the instant Run is
        # pressed, but the search-service call itself is dispatched to an
        # async worker -- the rows above can already read [beta, alpha]
        # before the "beta" search has actually reached the service. Wait
        # for it explicitly before capturing `calls_before`; otherwise a
        # late-landing "beta" call can itself satisfy the "count
        # increased" check below and leave `service.calls[-1]` reading
        # "beta" instead of the history row's "alpha" rerun.
        for _ in range(150):
            if service.calls and service.calls[-1]["query"] == "beta":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The 'beta' search never reached the search service.")

        calls_before = len(service.calls)
        screen.query_one("#library-rag-history-1", Button).press()

        for _ in range(150):
            if len(service.calls) > calls_before:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("History row press never re-ran the search service.")

        assert service.calls[-1]["query"] == "alpha"
        # Minor #5: the visible query input must show the re-run entry too,
        # not the "beta" text it held before the history row was clicked.
        # (C5a) The history-row press's query-input update lands via the
        # same recompose/refresh path as the service call above, but
        # isn't guaranteed to have settled by the instant the service call
        # is observed -- bounded-poll instead of a single immediate assert.
        for _ in range(150):
            if screen.query_one("#library-rag-query-input", Input).value == "alpha":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Query input never showed the re-run entry's text (still "
                f"{screen.query_one('#library-rag-query-input', Input).value!r})."
            )


@pytest.mark.asyncio
async def test_library_shell_search_history_row_survives_bracketed_query():
    """C1: a history entry containing bracket-like text must not crash the
    Search canvas.

    Textual parses a plain string ``Button`` label as Rich markup: an
    unescaped stored entry like "docs [/archive] cleanup" raises
    ``MarkupError`` at construction time inside
    ``library_rag_history_children`` -- and because the query is recorded
    into history *before* that rebuild runs, the crash would recur on
    every Search-canvas entry after restart. This exercises both call
    sites that build history rows from the same state: the live refresh
    triggered by submitting the query, and a fresh ``compose()`` reached by
    leaving and re-entering the Search canvas.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    query = "docs [/archive] cleanup"

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_library_rag_query_ready(screen, pilot, query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-history-0")

        history_row = screen.query_one("#library-rag-history-0", Button)
        assert str(history_row.label) == query

        # Leave and re-enter the Search canvas: this rebuilds the history
        # rows via the widget's own compose(), not the live-refresh path
        # the submit above already exercised -- both must survive the same
        # unescaped bracket entry.
        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-history-0")
        assert str(screen.query_one("#library-rag-history-0", Button).label) == query

        # Re-running from the history row must still work end to end.
        calls_before = len(service.calls)
        screen.query_one("#library-rag-history-0", Button).press()
        for _ in range(150):
            if len(service.calls) > calls_before:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("History row press never re-ran the search service.")
        assert service.calls[-1]["query"] == query


@pytest.mark.asyncio
async def test_library_shell_search_searching_line_shows_while_gated():
    """The searching status line renders the selected source scope while a
    gated fake service still holds the search open.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
        media=_two_media_items(),
    )
    service = _GatedLibraryRagSearchService({"results": []})
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "policy"
        await _wait_for_library_rag_query_ready(screen, pilot, "policy")
        screen.query_one("#library-rag-run-query", Button).press()

        try:
            await _wait_for_selector(screen, pilot, "#library-rag-searching-line")
            line = str(screen.query_one("#library-rag-searching-line").renderable)
            assert line == "searching · notes, media, conversations…"
        finally:
            service.release_event.set()


@pytest.mark.asyncio
async def test_library_shell_search_outcome_resolves_status_after_leaving_canvas():
    """I1(b): an outcome that lands while the user has left the Search
    canvas (switched to Media mid-flight) must still resolve the dangling
    "searching" retrieval status into settled state -- not leave it stuck,
    which would render a stale "searching" line again on re-entry.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
        media=_two_media_items(),
    )
    service = _GatedLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        search_input = screen.query_one("#library-search-input", Input)
        search_input.value = "policy"
        search_input.focus()
        await pilot.pause()
        await pilot.press("enter")

        for _ in range(150):
            if service.calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Rail submit never reached the gated search service.")

        assert screen._library_rag_retrieval_status == "searching"

        # Leave the Search canvas while the gated fake is still in flight.
        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        assert screen._library_selected_row_id == "browse-media"

        service.release_event.set()

        # The outcome must resolve _library_rag_retrieval_status even
        # though the panel is unmounted right now (user is on Media).
        for _ in range(150):
            if screen._library_rag_retrieval_status not in ("", "searching"):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Retrieval status was never resolved off-canvas.")

        assert screen._library_rag_retrieval_status == "ready"
        assert screen._library_rag_results

        # Re-entering the Search canvas must compose from the settled
        # state -- no stale "searching" line.
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-select-result-0")

        assert not screen.query("#library-rag-searching-line")


@pytest.mark.asyncio
async def test_library_shell_rail_search_submit_renders_every_result_row():
    """RED regression pilot: a rail-top search that matches multiple
    sources (one note, one media item, one conversation -- mirroring the
    live QA repro for query "research") must render ALL result rows in
    the Evidence region, not just the first, and every row must actually
    be reachable on screen (not merely present in the widget tree).

    Live QA on the served app found only result 1 rendering for a
    multi-result query after commits e308a71f/ec1a207c; single-result
    queries rendered fine. This submits through the rail search box (the
    path live QA used) so it also covers the rail's
    ``_select_library_rail_row`` -> recompose -> ``_start_library_rag_query``
    sequence, not just the in-panel Run button.

    Root cause note: the live truncation was NOT a mid-rebuild exception
    dropping rows from the DOM -- ``screen.query("#library-rag-result-N")``
    finds every row's widgets even on the buggy build (compose()/the live
    refresh both iterate every result without raising). The actual bug is
    that ``LibrarySearchRagPanel`` (and its ``#library-rag-results``
    sub-region) never scrolled: ``#library-rag-query-controls`` switched
    from a hand-counted fixed height to ``height: auto`` in ec1a207c, which
    (correctly) fixed that region's own internal overlap but also let it
    consume more of the fixed, non-scrolling canvas box, leaving less room
    for Evidence -- and anything past that both silently clipped AND was
    permanently unreachable, no matter how a user tried to scroll. So a
    query that only checks widget existence passes on both the broken and
    fixed builds; this asserts each row becomes visible in an actual
    rendered screenshot after ``scroll_visible()``, which only succeeds if
    some ancestor in the chain is actually scrollable.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Tides research",
                    "snippet": "Tide charts for the coastal survey.",
                    "source_id": "note-1",
                    "chunk_id": "chunk-1",
                    "provenance": {"source_type": "note"},
                },
                {
                    "document_title": "Ocean survey transcript",
                    "snippet": "Recorded interview about tide research.",
                    "source_id": "media-1",
                    "chunk_id": "chunk-2",
                    "provenance": {"source_type": "media"},
                },
                {
                    "document_title": "Draft quarterly research digest",
                    "snippet": "Conversation drafting the research digest.",
                    "source_id": "chat-1",
                    "chunk_id": "chunk-3",
                    "provenance": {"source_type": "conversation"},
                },
            ]
        }
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        search_input = screen.query_one("#library-search-input", Input)
        search_input.value = "research"
        search_input.focus()
        await pilot.pause()
        await pilot.press("enter")

        for _ in range(150):
            if service.calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Rail submit never reached the search service.")

        await _wait_for_selector(screen, pilot, "#library-rag-result-2")

        for index in range(3):
            assert screen.query(f"#library-rag-result-{index}"), (
                f"Result row {index} never rendered. Visible text: "
                f"{_visible_text(screen)}"
            )
            assert screen.query(f"#library-rag-select-result-{index}"), (
                f"Select-evidence button {index} never rendered."
            )
            assert screen.query(f"#library-rag-open-result-{index}"), (
                f"Open button {index} never rendered (all three results are "
                "openable: note, media, conversation)."
            )

        # Existence alone doesn't catch the real regression (see the
        # docstring): every row must also be reachable on screen. Scroll
        # each one into view individually and confirm its title text
        # actually appears in a rendered screenshot -- on the broken build
        # this fails for every row (nothing in the ancestor chain scrolls,
        # so ``scroll_visible()`` is a no-op and clipped content never
        # becomes visible no matter what).
        titles = ("Tides research", "Ocean survey transcript", "Draft quarterly research digest")
        for index, title in enumerate(titles):
            result_widget = screen.query_one(f"#library-rag-result-{index}")
            result_widget.scroll_visible(animate=False)
            await pilot.pause()
            for _ in range(10):
                await pilot.pause(0.02)
            screenshot = pilot.app.export_screenshot()
            # Match a distinctive word rather than the full title: Rich's
            # SVG export can render a run-together phrase as separate
            # per-style text nodes (e.g. a non-breaking space between
            # words), which would make a full-phrase substring check flaky.
            distinctive_word = title.split()[0]
            assert distinctive_word in screenshot, (
                f"Result row {index} ({title!r}) was never reachable on "
                f"screen after scroll_visible()."
            )


@pytest.mark.asyncio
async def test_library_shell_rag_results_arrival_scrolls_evidence_heading_into_view():
    """(C2) Results landing -- and ONLY results landing -- must scroll the
    Evidence heading back into view.

    The query controls and source-scope regions sit above Evidence in
    ``LibrarySearchRagPanel`` (a ``VerticalScroll``) and can grow tall
    enough (recovery callouts, many source toggles) to push Evidence past
    the fold, and a results-heavy Evidence region can itself do the same.
    Spies on the heading's own ``scroll_visible`` rather than asserting
    the settled scroll geometry: Textual's ``Collapsible`` widget (the
    "Recent searches" collapsible directly below Evidence) fires its own
    *animated* ``scroll_visible()`` on itself whenever its ``collapsed``
    reactive flips -- which D1 does the moment results land -- and that
    competing ~1s animation can outlast and override any assertion made
    against the panel's final scroll offset shortly after. Spying on the
    call is deterministic and directly proves the gating logic: called
    with ``animate=False`` when results land, not called for an unrelated
    refresh (typing the query) that never reaches results-arrival at all.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        heading = screen.query_one("#library-rag-results-heading", Static)
        heading.scroll_visible = Mock()

        screen.query_one("#library-rag-query-input", Input).value = "alpha"
        await _wait_for_library_rag_query_ready(screen, pilot, "alpha")
        assert heading.scroll_visible.call_count == 0, (
            "Typing the query alone (no results yet) must not scroll Evidence."
        )

        screen.query_one("#library-rag-run-query", Button).press()
        for _ in range(150):
            if heading.scroll_visible.call_count:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Results landing never scrolled the Evidence heading into view."
            )
        _, kwargs = heading.scroll_visible.call_args
        assert kwargs.get("animate") is False


@pytest.mark.asyncio
async def test_library_shell_rail_search_submit_renders_every_result_row_post_mount():
    """Variation: force the outcome to land AFTER the Search canvas has
    already recomposed and mounted, so resolution must go through the
    incremental ``_refresh_library_rag_results_widgets`` DOM-mutation path
    instead of a fresh ``compose()`` picking up already-set state.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _GatedLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Tides research",
                    "snippet": "Tide charts for the coastal survey.",
                    "source_id": "note-1",
                    "chunk_id": "chunk-1",
                    "provenance": {"source_type": "note"},
                },
                {
                    "document_title": "Ocean survey transcript",
                    "snippet": "Recorded interview about tide research.",
                    "source_id": "media-1",
                    "chunk_id": "chunk-2",
                    "provenance": {"source_type": "media"},
                },
                {
                    "document_title": "Draft quarterly research digest",
                    "snippet": "Conversation drafting the research digest.",
                    "source_id": "chat-1",
                    "chunk_id": "chunk-3",
                    "provenance": {"source_type": "conversation"},
                },
            ]
        }
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        search_input = screen.query_one("#library-search-input", Input)
        search_input.value = "research"
        search_input.focus()
        await pilot.pause()
        await pilot.press("enter")

        for _ in range(150):
            if service.calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Rail submit never reached the search service.")

        # Make sure the Search canvas has actually recomposed and mounted
        # before releasing the gated fake -- forces outcome resolution
        # through the live incremental refresh, not a fresh compose().
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        for _ in range(10):
            await pilot.pause(0.02)

        service.release_event.set()

        await _wait_for_selector(screen, pilot, "#library-rag-result-2")

        for index in range(3):
            assert screen.query(f"#library-rag-result-{index}"), (
                f"Result row {index} never rendered. Visible text: "
                f"{_visible_text(screen)}"
            )
            assert screen.query(f"#library-rag-select-result-{index}"), (
                f"Select-evidence button {index} never rendered."
            )
            assert screen.query(f"#library-rag-open-result-{index}"), (
                f"Open button {index} never rendered."
            )


@pytest.mark.asyncio
async def test_library_shell_search_run_button_renders_every_result_row():
    """Variation: use the in-panel Run button (no rail recompose at all) so
    resolution always goes through the incremental
    ``_refresh_library_rag_results_widgets`` DOM-mutation path.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Tides research",
                    "snippet": "Tide charts for the coastal survey.",
                    "source_id": "note-1",
                    "chunk_id": "chunk-1",
                    "provenance": {"source_type": "note"},
                },
                {
                    "document_title": "Ocean survey transcript",
                    "snippet": "Recorded interview about tide research.",
                    "source_id": "media-1",
                    "chunk_id": "chunk-2",
                    "provenance": {"source_type": "media"},
                },
                {
                    "document_title": "Draft quarterly research digest",
                    "snippet": "Conversation drafting the research digest.",
                    "source_id": "chat-1",
                    "chunk_id": "chunk-3",
                    "provenance": {"source_type": "conversation"},
                },
            ]
        }
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "research"
        await _wait_for_library_rag_query_ready(screen, pilot, "research")
        screen.query_one("#library-rag-run-query", Button).press()

        await _wait_for_selector(screen, pilot, "#library-rag-result-2")

        for index in range(3):
            assert screen.query(f"#library-rag-result-{index}"), (
                f"Result row {index} never rendered. Visible text: "
                f"{_visible_text(screen)}"
            )
            assert screen.query(f"#library-rag-select-result-{index}"), (
                f"Select-evidence button {index} never rendered."
            )
            assert screen.query(f"#library-rag-open-result-{index}"), (
                f"Open button {index} never rendered."
            )


@pytest.mark.asyncio
async def test_library_shell_rail_search_submit_aborts_on_note_conflict():
    """I1(a): a rail-top search submit while a dirty note sits in an
    unresolved save conflict must not run the query or record history.

    ``_select_library_rail_row`` aborts the row switch until the conflict
    is resolved; the rail submit handler must bail out too, instead of
    running the search against a canvas the user never actually reached.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        notes_service = app.notes_scope_service
        _bump_note_version_externally(notes_service, "n-1")

        screen.query_one("#library-note-body", TextArea).text = "kept text that must survive"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        history_before = screen._library_search_history
        search_input = screen.query_one("#library-search-input", Input)
        search_input.value = "zeta"
        search_input.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert service.calls == []
        assert screen._library_search_history == history_before
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen._library_note_autosave_state == "conflict"


@pytest.mark.asyncio
async def test_library_shell_search_mode_toggle_mid_flight_discards_wrong_mode_outcome():
    """I2: toggling Search <-> RAG Answer mode while a query is still in
    flight must not apply that query's outcome once it lands -- the result
    belongs to the mode the user has since left.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _GatedLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "policy"
        await _wait_for_library_rag_query_ready(screen, pilot, "policy")
        assert screen._library_rag_mode == "search"
        screen.query_one("#library-rag-run-query", Button).press()

        for _ in range(150):
            if service.calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Run never reached the gated search service.")

        # Toggle mode mid-flight. This resets the in-flight request's own
        # display state via _reset_library_rag_retrieval_state -- what
        # this test guards is the STALE outcome re-populating it once the
        # gate releases.
        screen.query_one("#library-rag-mode-toggle", Button).press()
        for _ in range(120):
            if screen._library_rag_mode == "rag":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Mode toggle never switched to RAG Answer.")

        service.release_event.set()
        for _ in range(20):
            await pilot.pause(0.02)

        assert screen._library_rag_results == ()
        assert screen._library_rag_retrieval_status == ""


@pytest.mark.asyncio
async def test_library_shell_media_row_selects_canvas_without_navigating():
    """Browse ▸ Media is a canvas row now, not a screen-route shortcut.

    Re-anchors the old screen-route contract: pressing the rail row must
    render the media canvas in place (selecting it, like Conversations)
    rather than firing ``NavigateToScreen("media")``. The media SCREEN
    route now lives behind the canvas's own "Open in Media" action.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    seen = []
    host = LibraryHarness(app, seen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        await pilot.pause()

        assert seen == []
        assert screen._library_selected_row_id == "browse-media"
        assert screen.query_one("#library-media-canvas")


@pytest.mark.asyncio
async def test_library_shell_browse_media_renders_canvas_with_rows_and_preview():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-canvas")

        title = str(screen.query_one("#library-media-title").renderable)
        assert title == "Media (2)"

        filter_button = screen.query_one("#library-media-type-filter", Button)
        assert str(filter_button.label) == "type: All ▸"

        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 2
        first_label = str(rows[0].label)
        assert first_label.startswith("▸")
        # Rows sort newest-first: media-2 (10:00) before media-1 (08:00).
        assert "Product Demo Video" in first_label
        assert "video" in first_label

        preview = str(screen.query_one("#library-media-preview-lines").renderable)
        assert "Product Demo Video" in preview
        # UX wave M2: names the real destination -- "Open in Media" read
        # like a no-op from a screen already showing media.
        assert str(screen.query_one("#library-media-open", Button).label) == "Open in Media manager"


@pytest.mark.asyncio
async def test_library_shell_media_type_filter_narrows_list():
    """Pressing the cycling filter button advances through type_options.

    The media fixture seeds two types ("audio", "video"), so
    ``type_options`` resolves to ``("All", "audio", "video")``. One press
    from the "All" default advances to "audio"; a second press advances to
    "video"; a third press wraps back around to "All".
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-canvas")

        filter_button = screen.query_one("#library-media-type-filter", Button)

        filter_button.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_type_filter == "audio"
        filter_button = screen.query_one("#library-media-type-filter", Button)
        assert str(filter_button.label) == "type: audio ▸"
        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 1
        assert "Interview Recording" in str(rows[0].label)

        status = str(screen.query_one("#library-media-status").renderable)
        assert "type: audio" in status

        filter_button.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_type_filter == "video"
        filter_button = screen.query_one("#library-media-type-filter", Button)
        assert str(filter_button.label) == "type: video ▸"
        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 1
        assert "Product Demo Video" in str(rows[0].label)

        filter_button.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_type_filter == "All"
        filter_button = screen.query_one("#library-media-type-filter", Button)
        assert str(filter_button.label) == "type: All ▸"
        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 2


@pytest.mark.asyncio
async def test_library_shell_media_row_switches_selection():
    """Selecting a different media row updates ``_selected_media_id``.

    Re-anchored for the in-canvas viewer rebuild: pressing a media row now
    replaces the list with the full ``LibraryMediaViewer`` (no more inline
    ``#library-media-preview-lines``), so selection is asserted against the
    viewer's title/metadata instead of the old stub preview.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")

        preview_before = str(
            screen.query_one("#library-media-preview-lines").renderable
        )
        assert "Product Demo Video" in preview_before
        assert screen._selected_media_id == "media-2"

        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")

        assert screen._selected_media_id == "media-1"
        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording"


@pytest.mark.asyncio
async def test_library_shell_media_row_opens_full_viewer_with_content():
    """Pressing a media row switches to the in-canvas viewer.

    The viewer must show the title, Type/Author metadata lines, and the
    full stored content text (not just a 3-line preview stub), replacing
    the list (``#library-media-list`` no longer present).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")

        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")

        assert screen._library_media_view == "viewer"
        assert not screen.query("#library-media-list")

        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording"

        meta = str(screen.query_one("#library-media-viewer-meta").renderable)
        assert "Type: audio" in meta
        assert "Author: Jordan Lee" in meta

        content_container = screen.query_one("#library-media-viewer-content")
        content_text = str(content_container.query_one(Static).renderable)
        assert "Full transcript: the interview recording" in content_text


@pytest.mark.asyncio
async def test_library_shell_media_viewer_uses_destination_honest_labels():
    """UX wave M2: the full viewer's Open/Use-in actions name their real
    destinations. "Open in Media" read like a no-op from a screen already
    showing media; "Use in Chat" is inaccurate once staged as Console live
    work (the same handoff every other Library "Use in Console" action --
    notes, conversations -- already uses). Button ids are unchanged.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-use-in-chat")

        assert str(screen.query_one("#library-media-open", Button).label) == "Open in Media manager"
        assert str(screen.query_one("#library-media-use-in-chat", Button).label) == "Use in Console"


@pytest.mark.asyncio
async def test_library_shell_media_analysis_button_reads_add_when_no_analysis():
    """UX wave L1: the analysis toggle reads "Add analysis" when the viewer
    has no analysis text yet (mirroring the Read-it-later conditional), and
    "Edit analysis" once analysis exists -- covered by the existing
    ``test_library_shell_media_analysis_edit_shows_prefilled_textarea``.
    """
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["versions"] = []
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")

        assert str(screen.query_one("#library-media-analysis-edit", Button).label) == "Add analysis"


@pytest.mark.asyncio
async def test_library_shell_media_use_in_chat_triggers_handoff():
    """``#library-media-use-in-chat`` stages the open media item as Console context.

    Uses the same clean ``ChatHandoffPayload`` + ``open_chat_with_handoff``
    handoff the Library conversation "Use in Console" action already uses.
    Seeds a single media item (rather than the usual two-item fixture) so
    the workspace context-handoff gate -- which is computed across *every*
    visible Library source, not just the one item under test -- isn't
    tripped up by an unlinked sibling row.
    """
    app = _build_test_app()
    media_items = _two_media_items()[:1]
    _seed_conversations(app, [], media=media_items)
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("media", "media-1", "Interview Recording"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        screen.query_one("#library-media-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-media-use-in-chat")

        screen.query_one("#library-media-use-in-chat").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "library"
    assert payload.item_type == "media"
    assert payload.source_id == "media-1"
    assert payload.title == "Interview Recording"
    assert "Full transcript: the interview recording" in payload.body


@pytest.mark.asyncio
async def test_library_shell_media_use_in_chat_without_open_item_notifies():
    """No handoff fires when no media item is currently loaded in the viewer.

    The "Use in Chat" button only mounts once a media item's detail has
    loaded into the viewer, so this exercises the same guard directly:
    calling the handler with ``_library_media_detail`` still at its
    freshly-mounted ``None`` must notify instead of staging anything.
    """
    app = _build_test_app()
    _seed_conversations(app, [], media=_two_media_items())
    app.open_chat_with_handoff = Mock()
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen._library_media_detail is None
        screen._open_selected_media_handoff()
        await pilot.pause()

    app.open_chat_with_handoff.assert_not_called()
    app.notify.assert_called_once()
    message = app.notify.call_args.args[0]
    assert "Open a media item" in message


@pytest.mark.asyncio
async def test_library_shell_media_back_returns_to_list():
    """``#library-media-back`` returns the media canvas to its list view."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-back")

        screen.query_one("#library-media-back").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_view == "list"
        assert screen.query_one("#library-media-list")
        assert not screen.query("#library-media-viewer-title")


@pytest.mark.asyncio
async def test_library_shell_media_rail_reentry_resets_to_list():
    """Re-entering Browse Media from the rail must show the list, not a stale viewer.

    A rail-row press is always a fresh entry into a content type. If the
    media viewer was left open on a previous visit, navigating away via
    another rail row and then pressing "Browse Media" again must land on
    the media list -- not resume the previously opened item's viewer.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")

        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")

        screen.query_one("#library-row-browse-conversations").press()
        await pilot.pause()
        await pilot.pause()

        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        await pilot.pause()

        assert screen.query_one("#library-media-list")
        assert not screen.query("#library-media-viewer-title")


@pytest.mark.asyncio
async def test_library_shell_media_viewer_shows_loading_before_detail_loads(monkeypatch):
    """The viewer shows a loading line until ``_library_media_detail`` arrives."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    monkeypatch.setattr(
        LibraryScreen, "_refresh_library_media_detail", _media_detail_never_loads
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")

        screen.query_one("#library-media-row-1").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_detail is None
        assert screen.query_one("#library-media-viewer-loading")
        assert not screen.query("#library-media-viewer-title")


async def _media_detail_never_loads(self, media_id: str) -> None:
    """Stand in for ``_refresh_library_media_detail`` that never resolves.

    Used to freeze the viewer in its loading state deterministically,
    instead of racing the real (async) detail fetch worker.
    """


@pytest.mark.asyncio
async def test_library_shell_media_open_posts_navigate_to_screen():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    seen = []
    host = LibraryHarness(app, seen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-open")

        screen.query_one("#library-media-open").press()
        await pilot.pause()
        await pilot.pause()

    assert seen[-1] == "media"


@pytest.mark.asyncio
async def test_library_shell_media_viewer_open_posts_navigate_to_screen():
    """``#library-media-open`` inside the full viewer also hands off to Media."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    seen = []
    host = LibraryHarness(app, seen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-open")

        screen.query_one("#library-media-open").press()
        await pilot.pause()
        await pilot.pause()

    assert seen[-1] == "media"


@pytest.mark.asyncio
async def test_library_shell_media_edit_shows_prefilled_form():
    """Pressing ``Edit`` swaps the metadata block for prefilled edit inputs."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit")

        screen.query_one("#library-media-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-title")

        assert screen._library_media_editing is True
        assert screen.query_one("#library-media-edit-title", Input).value == "Interview Recording"
        assert screen.query_one("#library-media-edit-author", Input).value == "Jordan Lee"
        assert screen.query_one("#library-media-edit-url", Input).value == ""
        assert (
            screen.query_one("#library-media-edit-keywords", Input).value
            == "interview, audio"
        )
        assert screen.query_one("#library-media-edit-save")
        assert screen.query_one("#library-media-edit-cancel")


@pytest.mark.asyncio
async def test_library_shell_media_edit_save_persists_and_exits_edit_mode():
    """Saving the edit form calls ``update_media_item`` and refreshes the viewer."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit")

        screen.query_one("#library-media-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-title")

        screen.query_one("#library-media-edit-title", Input).value = (
            "Interview Recording (Revised)"
        )

        screen.query_one("#library-media-edit-save").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.update_calls and not screen._library_media_editing:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Save never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert service.update_calls, "update_media_item was never called"
        call = service.update_calls[-1]
        assert call["media_id"] == "media-1"
        assert call["title"] == "Interview Recording (Revised)"
        assert call["author"] == "Jordan Lee"
        assert call["url"] == ""
        assert call["keywords"] == ["interview", "audio"]
        # `version` is NOT a supported local metadata field (see
        # LocalMediaReadingService._SUPPORTED_METADATA_FIELDS) -- sending it
        # makes every real save raise ValueError. Assert it is absent so a
        # future regression trips this test instead of silently failing in
        # production.
        assert "version" not in call

        assert screen._library_media_editing is False
        assert not screen.query("#library-media-edit-title")
        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording (Revised)"


async def _open_media_edit_and_save_title(screen, pilot, new_title):
    """Open media-1's edit form, set the title, save, and wait for completion."""
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-row-1")
    screen.query_one("#library-media-row-1").press()
    await _wait_for_selector(screen, pilot, "#library-media-edit")

    screen.query_one("#library-media-edit").press()
    await _wait_for_selector(screen, pilot, "#library-media-edit-title")
    screen.query_one("#library-media-edit-title", Input).value = new_title
    screen.query_one("#library-media-edit-save").press()

    service = screen.app_instance.media_reading_scope_service
    for _ in range(150):
        if service.update_calls and not screen._library_media_editing:
            break
        await pilot.pause(0.02)
    else:
        raise AssertionError(f"Save never completed. Visible: {_visible_text(screen)}")
    await pilot.pause()


@pytest.mark.asyncio
async def test_library_shell_media_edit_save_updates_list_snapshot_cache():
    """Saving a metadata edit updates the cached list snapshot, so the media
    list shows the new title immediately instead of the pre-edit value."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_edit_and_save_title(screen, pilot, "Renamed In List")

        # The cached list snapshot record now carries the new title.
        cached = {
            screen._source_record_id(r): r
            for r in screen._local_source_records.get("media", ())
        }
        assert cached["media-1"]["title"] == "Renamed In List"

        # ...and the list view reflects it after navigating back.
        screen.query_one("#library-media-back").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        assert "Renamed In List" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_shell_media_edit_save_sanitizes_user_input():
    """User-entered edit fields are sanitized at the UI boundary, so HTML/script
    markup never reaches the persistence service."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_edit_and_save_title(
            screen, pilot, "Clean<script>alert(1)</script>Title"
        )

        sent_title = screen.app_instance.media_reading_scope_service.update_calls[-1]["title"]
        assert "<script" not in sent_title.lower()
        assert "</script" not in sent_title.lower()
        # ...but the surrounding legitimate text is preserved.
        assert "Clean" in sent_title
        assert "Title" in sent_title


@pytest.mark.asyncio
async def test_library_shell_media_detail_race_discards_stale_fetch():
    """A detail fetch that completes for a no-longer-selected media id must be
    discarded instead of overwriting the currently-open viewer."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")
        for _ in range(150):
            detail = screen._library_media_detail
            if isinstance(detail, dict) and str(detail.get("id")) == "media-1":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("media-1 detail never loaded.")
        assert screen._selected_media_id == "media-1"

        # Simulate a slower in-flight fetch for the previously-selected media-2
        # completing now: it must not overwrite media-1's detail.
        await screen._refresh_library_media_detail("media-2")

        detail = screen._library_media_detail
        assert isinstance(detail, dict)
        assert str(detail.get("id")) == "media-1"


@pytest.mark.asyncio
async def test_library_shell_open_deleted_media_notifies_and_falls_back_to_list():
    """(A3) Opening a media item whose backing record was deleted between
    the id being captured (e.g. a stale Search/RAG "Open" result) and the
    click must notify the user and fall back to the list view instead of
    leaving an empty/stuck viewer -- mirrors the existing "Conversation is
    unavailable." notify ``_open_library_item_by_id`` already gives its
    conversations branch for the equivalent out-of-snapshot case.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    notifications = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Delete the backing record "between done and click": remove it
        # from the fake service's store so a subsequent get_media_item
        # resolves to None, the same way the real local backend does for a
        # deleted/never-existed id.
        service = screen.app_instance.media_reading_scope_service
        service.media_items = tuple(
            item for item in service.media_items if str(item.get("id")) != "media-1"
        )

        await screen._open_library_item_by_id("media", "media-1")
        for _ in range(150):
            if screen._library_media_view != "viewer":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Deleted-media open never fell back to the list view.")

        assert screen._library_media_view == "list"
        assert screen._library_media_detail is None
        assert notifications
        assert notifications[-1][0] == "Media item is unavailable."
        assert notifications[-1][1].get("severity") == "warning"
        # No empty/stuck viewer left mounted once the canvas recomposes.
        assert not screen.query("#library-media-viewer-title")


@pytest.mark.asyncio
async def test_library_shell_snapshot_replace_carries_over_out_of_page_selection():
    """(C3) A wholesale ``_local_source_records`` replace (the periodic
    background refresh) must not silently drop the currently-open
    conversation when it isn't part of the freshly-fetched page.

    Mirrors the out-of-snapshot open flow ``_open_library_item_by_id``
    already handles (fetch-and-prepend) -- this closes the same gap for the
    *next* background snapshot refresh, which would otherwise wholesale
    ``self._local_source_records = records`` over the prepended record and
    silently reset the selection back to the first row the next time
    something reads ``_selected_conversation_id``.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Simulate having opened an out-of-snapshot conversation the same
        # way `_open_library_item_by_id` does: prepend the fetched record
        # into `_local_source_records["conversations"]` and select it.
        out_of_snapshot_record = {
            "title": "Out of page conversation",
            "conversation_id": "chat-3",
            "message_count": 1,
            "updated_at": "2026-06-03T00:00:00Z",
        }
        screen._local_source_records["conversations"] = (
            out_of_snapshot_record,
            *screen._local_source_records.get("conversations", ()),
        )
        screen._selected_conversation_id = "chat-3"

        # Force a wholesale snapshot apply -- e.g. the periodic background
        # refresh -- whose freshly-fetched page does NOT include chat-3.
        screen._apply_local_source_snapshot(
            {"notes": (), "media": (), "conversations": tuple(_two_conversations())},
            {"notes": 0, "media": 0, "conversations": 2},
            {"notes": True, "media": True, "conversations": True},
        )

        conversation_ids = [
            screen._source_record_id(record)
            for record in screen._local_source_records["conversations"]
        ]
        assert "chat-3" in conversation_ids, (
            "The out-of-page conversation record was dropped by the "
            f"snapshot replace: {conversation_ids}"
        )
        assert screen._selected_conversation_id == "chat-3"
        selected = screen._selected_conversation_record()
        assert selected is not None
        _, selected_record = selected
        assert screen._source_record_id(selected_record) == "chat-3"


@pytest.mark.asyncio
async def test_library_shell_media_edit_cancel_discards():
    """Cancelling the edit form discards changes without calling the service."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit")

        screen.query_one("#library-media-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-title")

        screen.query_one("#library-media-edit-title", Input).value = "Should not persist"

        screen.query_one("#library-media-edit-cancel").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_editing is False
        assert not screen.query("#library-media-edit-title")
        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording"

        service = app.media_reading_scope_service
        assert service.update_calls == []


@pytest.mark.asyncio
async def test_library_shell_media_delete_shows_inline_confirm_without_deleting():
    """Pressing ``Delete`` shows the inline confirm affordance, not an immediate delete."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete")

        screen.query_one("#library-media-delete").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete-confirm")

        assert screen._library_media_confirming_delete is True
        assert screen.query_one("#library-media-delete-cancel")

        service = app.media_reading_scope_service
        assert service.delete_calls == []
        assert screen._library_media_view == "viewer"


@pytest.mark.asyncio
async def test_library_shell_media_delete_confirm_removes_item_and_returns_to_list():
    """Confirming the delete trashes the item and drops it from the list view."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete")

        screen.query_one("#library-media-delete").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete-confirm")

        screen.query_one("#library-media-delete-confirm").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.delete_calls and screen._library_media_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Delete never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert service.delete_calls, "delete_media_item was never called"
        assert service.delete_calls[-1]["media_id"] == "media-1"
        assert screen._library_media_confirming_delete is False
        assert screen._library_media_view == "list"
        assert not screen.query("#library-media-viewer")
        # media-1 ("Interview Recording") is deleted; only media-2 ("Product
        # Demo Video") remains, re-indexed to row-0 by the sorted rebuild.
        assert not any(
            "Interview Recording" in str(getattr(button, "label", ""))
            for button in screen.query(".library-media-row")
        )
        assert screen.query_one("#library-media-row-0")


@pytest.mark.asyncio
async def test_library_shell_media_delete_cancel_leaves_item_intact():
    """Cancelling the inline confirm discards it without calling the service."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete")

        screen.query_one("#library-media-delete").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete-confirm")

        screen.query_one("#library-media-delete-cancel").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_confirming_delete is False
        assert not screen.query("#library-media-delete-confirm")
        assert screen.query_one("#library-media-delete")
        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording"

        service = app.media_reading_scope_service
        assert service.delete_calls == []


@pytest.mark.asyncio
async def test_library_shell_media_viewer_renders_seeded_highlight():
    """A seeded highlight's quote renders in the viewer's highlights section."""
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        media=_two_media_items(),
        highlights=[("media-1", "Important sentence", "Check this", "yellow")],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-highlight-0")

        visible = _visible_text(screen)
        assert "Important sentence" in visible
        assert "Check this" in visible
        # A renderable color is conveyed by a tinted "●" swatch on the quote,
        # not by the literal word "yellow".
        quote = screen.query_one("#library-media-highlight-0").renderable
        assert "●" in quote.plain
        assert any("yellow" in str(span.style).lower() for span in quote.spans)
        assert screen.query_one("#library-media-highlight-delete-0")


@pytest.mark.asyncio
async def test_library_shell_media_highlight_non_rich_color_does_not_crash_render():
    """A Textual-valid but Rich-invalid color (e.g. 'transparent') must not crash.

    The swatch color is consumed as a Rich style; validating with Textual's
    (superset) parser let 'transparent'/'hsl(...)'/'ansi_*' through and then
    crashed at render (rich.errors.MissingStyle). Such colors must fall back
    to plain text instead.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        media=_two_media_items(),
        highlights=[("media-1", "Fragile quote", None, "transparent")],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        # Reaching this selector means the viewer rendered without raising.
        await _wait_for_selector(screen, pilot, "#library-media-highlight-0")

        quote = screen.query_one("#library-media-highlight-0").renderable
        # No swatch marker (color not renderable); it survives as plain text.
        assert "●" not in quote.plain
        assert "transparent" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_shell_media_highlight_add_creates_and_renders_new_highlight():
    """Filling the quote input and pressing Add calls ``create_highlight`` and renders it."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-highlight-add-collapsible")

        assert not screen.query(".library-media-highlight-delete")

        # The add-highlight form starts collapsed; expand it before typing
        # into its inputs, mirroring how a real user would reveal the form.
        collapsible = screen.query_one(
            "#library-media-highlight-add-collapsible", Collapsible
        )
        assert collapsible.collapsed
        collapsible.collapsed = False
        await pilot.pause()

        screen.query_one("#library-media-highlight-quote", Input).value = "New highlight quote"
        screen.query_one("#library-media-highlight-add").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.create_highlight_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Add never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()
        await pilot.pause()

        assert service.create_highlight_calls[-1]["item_id"] == "media-1"
        assert service.create_highlight_calls[-1]["quote"] == "New highlight quote"
        await _wait_for_selector(screen, pilot, "#library-media-highlight-0")
        assert "New highlight quote" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_shell_media_highlight_delete_removes_it():
    """Pressing a highlight's delete button calls ``delete_highlight`` and removes the row."""
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        media=_two_media_items(),
        highlights=[("media-1", "Doomed highlight", None, None)],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-highlight-delete-0")

        screen.query_one("#library-media-highlight-delete-0").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.delete_highlight_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Delete never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()
        await pilot.pause()

        assert service.delete_highlight_calls[-1]["highlight_id"] == "1"
        assert "Doomed highlight" not in _visible_text(screen)
        assert not screen.query(".library-media-highlight-delete")


@pytest.mark.asyncio
async def test_library_shell_media_read_later_saves_and_flips_button_label():
    """Pressing "Read it later" calls save_to_read_it_later and flips the label."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-read-later")

        assert str(screen.query_one("#library-media-read-later", Button).label) == "Read it later"

        screen.query_one("#library-media-read-later").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.read_it_later_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Toggle never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()
        await pilot.pause()

        assert service.read_it_later_calls[-1] == {"action": "save", "media_id": "media-1"}
        assert screen._library_media_detail["is_read_it_later"] is True
        assert (
            str(screen.query_one("#library-media-read-later", Button).label)
            == "Remove from read-it-later"
        )


@pytest.mark.asyncio
async def test_library_shell_media_read_later_removes_when_already_saved():
    """Pressing the toggle on an already-saved item calls remove_from_read_it_later."""
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["is_read_it_later"] = True
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-read-later")

        assert (
            str(screen.query_one("#library-media-read-later", Button).label)
            == "Remove from read-it-later"
        )

        screen.query_one("#library-media-read-later").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.read_it_later_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Toggle never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()
        await pilot.pause()

        assert service.read_it_later_calls[-1] == {"action": "remove", "media_id": "media-1"}
        assert "is_read_it_later" not in screen._library_media_detail
        assert str(screen.query_one("#library-media-read-later", Button).label) == "Read it later"


@pytest.mark.asyncio
async def test_library_shell_media_viewer_shows_analysis_from_latest_version():
    """Analysis text from the newest DocumentVersions row renders in the viewer.

    Local media details never carry top-level analysis_content (it lives on
    DocumentVersions only) -- this proves the viewer surfaces it from the
    ``versions`` list returned by ``get_media_item``.
    """
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["versions"] = [
        {"version_number": 2, "analysis_content": "Roadmap analysis"},
        {"version_number": 1, "analysis_content": None},
    ]
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-analysis-text")

        analysis_text = str(screen.query_one("#library-media-viewer-analysis-text").renderable)
        assert analysis_text == "Roadmap analysis"


@pytest.mark.asyncio
async def test_library_shell_media_analysis_edit_shows_prefilled_textarea():
    """Pressing "Edit analysis" swaps the analysis text for a prefilled TextArea."""
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["versions"] = [{"version_number": 1, "analysis_content": "Existing analysis"}]
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")

        screen.query_one("#library-media-analysis-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit-text")

        assert screen._library_media_editing_analysis is True
        text_area = screen.query_one("#library-media-analysis-edit-text", TextArea)
        assert text_area.text == "Existing analysis"
        assert screen.query_one("#library-media-analysis-save")
        assert screen.query_one("#library-media-analysis-cancel")


@pytest.mark.asyncio
async def test_library_shell_media_analysis_save_persists_and_exits_edit_mode():
    """Saving the analysis edit form calls save_analysis_version and refreshes the viewer."""
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["versions"] = [{"version_number": 1, "analysis_content": "Existing analysis"}]
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")

        screen.query_one("#library-media-analysis-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit-text")

        screen.query_one("#library-media-analysis-edit-text", TextArea).text = (
            "Revised analysis"
        )

        screen.query_one("#library-media-analysis-save").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.analysis_calls and not screen._library_media_editing_analysis:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Save never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert service.analysis_calls, "save_analysis_version was never called"
        call = service.analysis_calls[-1]
        assert call["media_id"] == "media-1"
        assert call["analysis_content"] == "Revised analysis"
        assert call["content"] == "Full transcript: the interview recording covers the quarterly roadmap."

        assert screen._library_media_editing_analysis is False
        assert not screen.query("#library-media-analysis-edit-text")
        analysis_text = str(screen.query_one("#library-media-viewer-analysis-text").renderable)
        assert analysis_text == "Revised analysis"


@pytest.mark.asyncio
async def test_library_shell_media_analysis_cancel_discards():
    """Cancelling the analysis edit form discards changes without calling the service."""
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["versions"] = [{"version_number": 1, "analysis_content": "Existing analysis"}]
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")

        screen.query_one("#library-media-analysis-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit-text")

        screen.query_one("#library-media-analysis-edit-text", TextArea).text = "Should not persist"

        screen.query_one("#library-media-analysis-cancel").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_editing_analysis is False
        assert not screen.query("#library-media-analysis-edit-text")
        analysis_text = str(screen.query_one("#library-media-viewer-analysis-text").renderable)
        assert analysis_text == "Existing analysis"

        service = app.media_reading_scope_service
        assert service.analysis_calls == []


def _media_item_with_multiline_content():
    """A media item whose content has two lines containing "budget"."""
    items = _two_media_items()
    items[0]["content"] = "\n".join(
        [
            "Intro paragraph about the roadmap.",
            "The budget forecast is discussed here.",
            "Another neutral line.",
            "Second budget line appears here too.",
        ]
    )
    return items


def _media_item_with_two_hits_on_one_line():
    """A media item whose only matching line contains "budget" twice."""
    items = _two_media_items()
    items[0]["content"] = "The budget and the budget again on one single line."
    return items


async def _open_media_viewer(screen, pilot):
    """Navigate to the media list and open the first row's viewer."""
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-row-1")
    screen.query_one("#library-media-row-1").press()
    await _wait_for_selector(screen, pilot, "#library-media-content-search")


async def _submit_content_search_query(screen, pilot, query):
    """Type ``query`` into the open viewer's content search box and press Enter."""
    search_input = screen.query_one("#library-media-content-search", Input)
    search_input.value = query
    search_input.focus()
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


async def _open_media_viewer_and_submit_content_search(screen, pilot, query):
    """Open the first media row's viewer and submit a content-search query."""
    await _open_media_viewer(screen, pilot)
    await _submit_content_search_query(screen, pilot, query)


@pytest.mark.asyncio
async def test_library_shell_media_content_search_no_query_hides_status_and_nav():
    """With no active search, only the search box renders -- no orphaned status/prev/next."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer(screen, pilot)

        assert screen.query_one("#library-media-content-search")
        assert not screen.query("#library-media-content-search-status")
        assert not screen.query("#library-media-content-search-prev")
        assert not screen.query("#library-media-content-search-next")


@pytest.mark.asyncio
async def test_library_shell_media_content_search_shows_match_count():
    """Submitting a query shows the match count and starts at the first match."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")

        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 1 of 2 matches"
        assert screen._library_media_content_query == "budget"
        assert screen._library_media_content_match_index == 0


@pytest.mark.asyncio
async def test_library_shell_media_content_search_highlights_matches_in_body():
    """While searching, each query occurrence in the content body is styled."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")

        content = screen.query_one("#library-media-viewer-content-text").renderable
        # Rendered as a Rich Text with a styled span over each "budget".
        highlighted = [
            content.plain[span.start : span.end]
            for span in content.spans
            if str(span.style)
        ]
        assert highlighted.count("budget") == 2
        assert all(part.lower() == "budget" for part in highlighted)


@pytest.mark.asyncio
async def test_library_shell_media_content_search_one_mark_per_matching_line():
    """A line with two hits is one match and gets one mark (count == visible marks)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_two_hits_on_one_line())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")

        # One matching line -> "Match 1 of 1", even though "budget" appears twice.
        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 1 of 1 matches"
        # ...and exactly one styled mark in the body, so count == visible marks.
        content = screen.query_one("#library-media-viewer-content-text").renderable
        marks = [span for span in content.spans if str(span.style)]
        assert len(marks) == 1
        assert content.plain[marks[0].start : marks[0].end] == "budget"


@pytest.mark.asyncio
async def test_library_shell_media_content_search_no_matches_shows_status():
    """A query with no hits in the content shows a "No matches" status."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "nonexistent-term")

        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "No matches"


@pytest.mark.asyncio
async def test_library_shell_media_content_search_empty_query_hides_status_and_nav():
    """Submitting a blank query clears the query and hides the status/prev/next row."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")
        assert screen.query_one("#library-media-content-search-status")

        await _submit_content_search_query(screen, pilot, "")

        # With no active query, the status line and prev/next toolbar are not
        # rendered at all -- they aren't just blank, they're gone -- so the
        # orphaned nav doesn't linger under the search box.
        assert not screen.query("#library-media-content-search-status")
        assert not screen.query("#library-media-content-search-prev")
        assert not screen.query("#library-media-content-search-next")
        assert screen._library_media_content_query == ""


@pytest.mark.asyncio
async def test_library_shell_media_content_search_next_prev_advances_match_index():
    """Next/Prev cycle the match index (wrapping) and update the status line."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")
        assert screen._library_media_content_match_index == 0

        screen.query_one("#library-media-content-search-next").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_content_match_index == 1
        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 2 of 2 matches"

        # Next wraps back around to the first match.
        screen.query_one("#library-media-content-search-next").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_content_match_index == 0
        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 1 of 2 matches"

        # Prev wraps backwards to the last match.
        screen.query_one("#library-media-content-search-prev").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_content_match_index == 1
        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 2 of 2 matches"


@pytest.mark.asyncio
async def test_library_shell_media_content_search_resets_on_back():
    """Returning to the media list clears the in-content search state."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")
        assert screen._library_media_content_query == "budget"

        screen.query_one("#library-media-back").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_content_query == ""
        assert screen._library_media_content_match_index == 0


@pytest.mark.asyncio
async def test_library_shell_media_canvas_shows_loading_before_snapshot_loads(monkeypatch):
    """Mirrors the conversations loading-gate contract for the media canvas."""
    app = _build_test_app()
    _seed_conversations(app, [])

    monkeypatch.setattr(LibraryScreen, "_refresh_local_source_snapshot", _never_loads)

    screen = LibraryScreen(app)
    screen._library_selected_row_id = "browse-media"
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await pilot.pause()
        await pilot.pause()

        assert active_screen._library_loaded is False
        assert active_screen.query_one("#library-canvas-loading")
        assert not active_screen.query("#library-media-canvas")


@pytest.mark.asyncio
async def test_library_shell_conversations_filter_filters_canvas():
    """The in-canvas filter (``#library-conversations-filter``) narrows the
    loaded conversations snapshot client-side; the rail-top box no longer
    does this (it feeds the Search canvas instead -- see the rail-submit
    pilots below).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-filter")

        filter_input = screen.query_one("#library-conversations-filter")
        filter_input.value = "quarterly"
        filter_input.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-conversations-status")

        status = str(screen.query_one("#library-conversations-status").renderable)
        assert status == "1 match for 'quarterly'"
        rows = list(screen.query(".library-conversation-row"))
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_library_shell_conversations_filter_retains_value_after_submit():
    """Submitting a filter recomposes the shell; the box must keep the value.

    The submit handler recomposes the whole screen (``refresh(recompose=True)``),
    which remounts a brand-new ``#library-conversations-filter``. Regression
    guard: that new input must be seeded with the active filter instead of
    showing empty text while the filter is silently active.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-filter")

        filter_input = screen.query_one("#library-conversations-filter")
        filter_input.value = "quarterly"
        filter_input.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-conversations-status")

        recomposed_input = screen.query_one("#library-conversations-filter")
        assert recomposed_input.value == "quarterly"
        assert recomposed_input.has_focus


@pytest.mark.asyncio
async def test_library_shell_rail_search_submit_runs_search_canvas_query():
    """Submitting the rail-top search box feeds the promoted Search canvas
    (single query truth = ``_library_rag_query``): it selects the Search
    row, runs the fast ``search`` mode query against the recording fake
    service, and returns focus to the rail box (which remains mounted --
    it is not torn down when leaving the conversations canvas).
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
        media=_two_media_items(),
    )
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        search_input = screen.query_one("#library-search-input", Input)
        search_input.value = "zeta"
        search_input.focus()
        await pilot.pause()
        await pilot.press("enter")

        for _ in range(150):
            if screen.query("#library-search-rag-panel") and service.calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Rail search submit never mounted the Search canvas / called "
                f"the service. Visible text: {_visible_text(screen)}"
            )

        assert screen._library_selected_row_id == "browse-search"
        assert service.calls == [
            {
                "query": "zeta",
                "scope": ("notes", "media", "conversations"),
                "mode": "search",
                "top_k": 5,
                "include_citations": True,
            }
        ]

        recomposed_input = screen.query_one("#library-search-input", Input)
        assert recomposed_input.value == "zeta"
        assert recomposed_input.has_focus


@pytest.mark.asyncio
async def test_library_shell_rail_search_placeholder_is_unconditional():
    """The rail placeholder always reads "Search Library..." now -- it no
    longer flips to "Search conversations..." while Browse Conversations is
    selected, because the rail box no longer filters conversations.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert (
            str(screen.query_one("#library-search-input", Input).placeholder)
            == "Search Library…"
        )

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")

        assert (
            str(screen.query_one("#library-search-input", Input).placeholder)
            == "Search Library…"
        )


@pytest.mark.asyncio
async def test_library_shell_rail_search_empty_submit_selects_without_service_call():
    """An empty rail-top submit still selects the Search canvas (so the user
    lands somewhere sensible on a bare Enter) but must not invoke the search
    service -- there is nothing to search for.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        search_input = screen.query_one("#library-search-input", Input)
        search_input.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        assert screen._library_selected_row_id == "browse-search"
        assert service.calls == []


@pytest.mark.asyncio
async def test_library_shell_scope_toggle_deselect_sends_only_selected_types():
    """B2: deselecting a scope toggle removes that source type from the
    retrieval request; deselecting every toggle blocks the run gate with
    the A1 quiet line instead of running the query.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
        media=_two_media_items(),
    )
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-scope-toggle-media")

        assert str(
            screen.query_one("#library-rag-scope-toggle-media", Button).label
        ).startswith("✓")

        screen.query_one("#library-rag-scope-toggle-media", Button).press()
        for _ in range(120):
            toggles = list(screen.query("#library-rag-scope-toggle-media"))
            if toggles and str(toggles[0].label).startswith("○"):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Media toggle never deselected.")

        screen.query_one("#library-rag-query-input", Input).value = "policy"
        await _wait_for_library_rag_query_ready(screen, pilot, "policy")
        screen.query_one("#library-rag-run-query", Button).press()

        for _ in range(150):
            if service.calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Run never reached the search service.")

        assert service.calls[-1]["scope"] == ("notes", "conversations")

        # Deselect-all: the run gate blocks with the A1 quiet line, not the
        # old scope-table recovery dump, and the service is not re-invoked.
        screen.query_one("#library-rag-scope-toggle-notes", Button).press()
        for _ in range(120):
            toggles = list(screen.query("#library-rag-scope-toggle-notes"))
            if toggles and str(toggles[0].label).startswith("○"):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Notes toggle never deselected.")

        screen.query_one("#library-rag-scope-toggle-conversations", Button).press()
        for _ in range(120):
            run_buttons = list(screen.query("#library-rag-run-query"))
            if run_buttons and run_buttons[0].disabled:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Deselect-all never blocked the run gate.")

        assert screen.query_one("#library-rag-query-quiet-line", Static)
        assert "Select at least one source." in _visible_text(screen)

        calls_before = len(service.calls)
        screen.query_one("#library-rag-run-query", Button).press()
        await pilot.pause()
        assert len(service.calls) == calls_before


@pytest.mark.asyncio
async def test_library_shell_search_scope_strip_refresh_path_uses_shared_copy():
    """Both scope-strip builders read LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY.

    The "#library-rag-scope-summary" text has two independent builders:
    the panel's own compose() (pinned by the gate16 asserts) and the
    screen's incremental refresh path (``_refresh_search_rag_panel_state_
    widgets``, driven by Input.Changed on the query field), which was
    previously a second hardcoded literal kept in sync only by comments.
    This exercises the refresh path specifically: overwrite the strip with
    a sentinel, type into the query input, and require the refresh to
    rewrite it to the shared constant -- so re-inlining a drifting literal
    at either site fails a test instead of drifting silently.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
        media=_two_media_items(),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-scope-summary")

        strip = screen.query_one("#library-rag-scope-summary", Static)
        # Compose path (panel-side builder).
        assert str(strip.renderable) == LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY

        # Sentinel-overwrite, then drive the screen-side refresh path via
        # a query edit (Input.Changed -> _refresh_search_rag_panel_state_widgets).
        strip.update("SENTINEL-SCOPE-DRIFT-CHECK")
        screen.query_one("#library-rag-query-input", Input).value = "policy"

        for _ in range(150):
            strips = list(screen.query("#library-rag-scope-summary"))
            if strips and str(strips[0].renderable) != "SENTINEL-SCOPE-DRIFT-CHECK":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "The refresh path never rewrote the scope strip. Visible "
                f"text: {_visible_text(screen)}"
            )

        assert str(
            screen.query_one("#library-rag-scope-summary", Static).renderable
        ) == LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY


@pytest.mark.asyncio
async def test_library_shell_search_run_button_shows_searching_while_gated():
    """C2: while a query is in flight, the Run button itself carries the
    in-flight state -- label "Searching…", disabled -- and returns to the
    normal enabled Run label once the search settles. Exercises the
    incremental (non-recompose) refresh path, not just a fresh compose().
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _GatedLibraryRagSearchService({"results": []})
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "policy"
        await _wait_for_library_rag_query_ready(screen, pilot, "policy")
        screen.query_one("#library-rag-run-query", Button).press()

        try:
            for _ in range(150):
                run_buttons = list(screen.query("#library-rag-run-query"))
                if (
                    run_buttons
                    and str(run_buttons[0].label) == "Searching…"
                    and run_buttons[0].disabled is True
                ):
                    break
                await pilot.pause(0.02)
            else:
                raise AssertionError(
                    f"Run button never showed the Searching… label. Visible "
                    f"text: {_visible_text(screen)}"
                )
        finally:
            service.release_event.set()

        for _ in range(150):
            run_buttons = list(screen.query("#library-rag-run-query"))
            if (
                run_buttons
                and str(run_buttons[0].label) == "Run"
                and run_buttons[0].disabled is False
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Run button never returned to the enabled Run label.")


@pytest.mark.asyncio
async def test_library_shell_history_clear_button_empties_history():
    """D1: `Clear history` empties both in-memory and persisted history, and
    (alongside the hint line) only renders once history is non-empty.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        assert not screen.query("#library-rag-history-clear")
        assert not screen.query("#library-rag-history-hint")

        screen.query_one("#library-rag-query-input", Input).value = "alpha"
        await _wait_for_library_rag_query_ready(screen, pilot, "alpha")
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-history-clear")

        assert screen.query_one("#library-rag-history-hint", Static)
        assert "Select an entry to run it again." in _visible_text(screen)

        screen.query_one("#library-rag-history-clear", Button).press()
        for _ in range(150):
            if screen._library_search_history == ():
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Clear history never emptied in-memory history.")

        await _wait_for_selector(screen, pilot, "#library-rag-history-empty")
        assert not screen.query("#library-rag-history-clear")
        assert not screen.query("#library-rag-history-hint")
        assert not screen.query(".library-rag-history-row")
        assert app.app_config["library"]["search"]["history"] == []


@pytest.mark.asyncio
async def test_library_shell_history_manual_expand_survives_unrelated_refresh():
    """D1: a manual expand of `Recent searches` must survive an unrelated
    refresh (editing the query text) -- only the results-arrival transition
    in `_apply_library_rag_search_outcome` is allowed to force it collapsed.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "alpha"
        await _wait_for_library_rag_query_ready(screen, pilot, "alpha")
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-history-0")

        # Results just landed: the collapsible is force-collapsed.
        assert screen.query_one("#library-rag-history", Collapsible).collapsed is True

        # Mirror a user click on the collapsible header.
        screen.query_one("#library-rag-history", Collapsible).collapsed = False

        # An unrelated refresh (editing the query text) must not re-collapse it.
        screen.query_one("#library-rag-query-input", Input).value = "alpha b"
        await _wait_for_library_rag_query_ready(screen, pilot, "alpha b")

        assert screen.query_one("#library-rag-history", Collapsible).collapsed is False


@pytest.mark.asyncio
async def test_library_shell_history_manual_expand_survives_scope_toggle_recompose():
    """D1: a manual expand of `Recent searches` must survive a scope-toggle
    recompose (`refresh(recompose=True)`).

    Unlike a query edit (which only refreshes widgets in place), a scope
    toggle tears down and rebuilds the whole canvas via `compose()`, which
    reads `_library_rag_history_collapsed`. The live `Collapsible.collapsed`
    reactive must be synced back into that field on user interaction, or the
    recompose reads the stale (force-collapsed) field and silently discards
    the manual expand.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
        media=_two_media_items(),
    )
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "alpha"
        await _wait_for_library_rag_query_ready(screen, pilot, "alpha")
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-history-0")

        # Results just landed: the collapsible is force-collapsed.
        assert screen.query_one("#library-rag-history", Collapsible).collapsed is True

        # Mirror a user click on the collapsible header.
        screen.query_one("#library-rag-history", Collapsible).collapsed = False
        for _ in range(120):
            if screen._library_rag_history_collapsed is False:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Manual expand never synced back to _library_rag_history_collapsed."
            )

        # A scope toggle triggers a full `refresh(recompose=True)`, unlike a
        # query edit -- this is the transition the field-sync must survive.
        screen.query_one("#library-rag-scope-toggle-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-history-0")

        assert screen.query_one("#library-rag-history", Collapsible).collapsed is False


def _never_loads(self) -> None:
    """Stand in for ``_refresh_local_source_snapshot`` that never resolves.

    Used to freeze ``LibraryScreen`` in its pre-load state deterministically,
    instead of racing the real snapshot worker to assert on a narrow timing
    window.
    """
    return None


@pytest.mark.asyncio
async def test_library_shell_shows_loading_state_before_snapshot_loads(monkeypatch):
    """Before the local source snapshot loads, the canvas must show a loading
    indicator instead of the false "no conversations" empty state, and the
    rail must not claim a known zero count for Conversations.
    """
    app = _build_test_app()
    _seed_conversations(app, [])

    monkeypatch.setattr(LibraryScreen, "_refresh_local_source_snapshot", _never_loads)

    screen = LibraryScreen(app)
    screen.apply_navigation_context({"mode": "conversations"})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await pilot.pause()
        await pilot.pause()

        assert active_screen._library_loaded is False
        assert active_screen.query_one("#library-canvas-loading")
        assert not active_screen.query("#library-conversations-canvas")

        visible = _visible_text(active_screen)
        assert "Conversations (0)" not in visible


@pytest.mark.asyncio
async def test_library_shell_shows_lookup_error_in_canvas(monkeypatch):
    """A local-source lookup error must surface in the canvas, not only in
    the (possibly collapsed) Details disclosure.
    """
    app = _build_test_app()
    _seed_conversations(app, [])

    monkeypatch.setattr(LibraryScreen, "_refresh_local_source_snapshot", _never_loads)

    screen = LibraryScreen(app)
    screen.apply_navigation_context({"mode": "conversations"})
    screen._library_lookup_error = "Library sources are unavailable right now."
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await pilot.pause()
        await pilot.pause()

        error_static = active_screen.query_one("#library-canvas-error")
        assert "unavailable" in str(error_static.renderable).lower()
        assert not active_screen.query("#library-canvas-loading")


@pytest.mark.asyncio
async def test_library_shell_details_toggle_persists():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert (
            screen.query_one("#library-rail-section-body-details").styles.display
            == "none"
        )
        toggle = screen.query_one("#console-rail-section-toggle-library-details")
        toggle.press()
        await pilot.pause()
        await pilot.pause()

        assert (
            screen.query_one("#library-rail-section-body-details").styles.display
            != "none"
        )

    sections = (
        app.app_config.get("library", {}).get("rail_state", {}).get("sections", {})
    )
    assert sections.get("details_open") is True


@pytest.mark.asyncio
async def test_library_shell_collections_row_loads_seeded_records():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_collections_service = StaticLibraryCollectionsService(
        [
            {
                "collection_id": "collection-1",
                "name": "Launch Evidence",
                "description": "Sources for release review.",
                "item_count": 3,
                "source_authority": "local",
                "sync_status": "local-only",
                "updated_at": "2026-06-09T12:00:00Z",
            }
        ]
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Entering Collections via the rail row must load the snapshot with no
        # prior create/rename action; the seeded record renders in the canvas.
        screen.query_one("#library-row-browse-collections").press()
        select_button = await _wait_for_selector(
            screen, pilot, "#library-collection-select-0"
        )

        canvas = screen.query_one("#library-canvas")
        assert canvas in select_button.ancestors
        assert "Launch Evidence" in str(select_button.label)


@pytest.mark.asyncio
async def test_library_shell_collections_deeplink_loads_before_mount():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_collections_service = StaticLibraryCollectionsService(
        [
            {
                "collection_id": "collection-1",
                "name": "Launch Evidence",
                "description": "Sources for release review.",
                "item_count": 3,
                "source_authority": "local",
                "sync_status": "local-only",
                "updated_at": "2026-06-09T12:00:00Z",
            }
        ]
    )
    screen = LibraryScreen(app)

    # Mirrors the real app.py ordering: handle_screen_navigation calls
    # apply_navigation_context BEFORE switch_screen mounts the destination
    # screen, so the screen is never mounted at this point.
    assert screen.is_mounted is False
    screen.apply_navigation_context({"mode": "collections"})

    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Deep-linking straight into Collections (no prior rail-row press)
        # must still load the snapshot; the seeded record renders in the
        # canvas without requiring any additional user interaction.
        select_button = await _wait_for_selector(
            screen, pilot, "#library-collection-select-0"
        )

        canvas = screen.query_one("#library-canvas")
        assert canvas in select_button.ancestors
        assert "Launch Evidence" in str(select_button.label)


@pytest.mark.asyncio
async def test_library_shell_notes_create_deeplink_lands_on_create_view():
    """The retired Notes tab's "new note" deep link now re-points into
    Library: a ``notes_create`` navigation context must land the shell on
    the in-canvas Create > New note view, mirroring how pressing the
    "New note" rail row does (``LIBRARY_ROW_CREATE_NOTE`` / canvas kind
    ``notes-create``).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    screen = LibraryScreen(app)

    # Mirrors the real app.py ordering: handle_screen_navigation calls
    # apply_navigation_context BEFORE switch_screen mounts the destination
    # screen (see test_library_shell_collections_deeplink_loads_before_mount).
    assert screen.is_mounted is False
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_NOTES_CREATE: True})

    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_NOTE
        assert screen.query_one("#library-notes-create-blank")


@pytest.mark.asyncio
async def test_library_shell_notes_create_deeplink_reentry_resets_stale_editor_state():
    """(A4) A cached ``LibraryScreen`` re-entered via the ``notes_create``
    deep link must never carry over a previously opened note's editor state
    -- ``_select_library_rail_row`` (the "New note" rail row's own entry
    path) already resets the note editor on every switch via
    ``_reset_library_note_editor_state``, but the ``notes_create`` deep-link
    branch in ``_apply_navigation_context_state`` skipped that call, so a
    post-mount re-entry through the deep link (after the user had already
    opened an existing note in the editor) kept that note's id/detail/
    version around instead of landing on a clean create-note slate. Mirrors
    ``test_library_shell_ingest_nav_context_deeplink_reentry_resets_stale_form``.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Open an existing note in the editor, as if the user had already
        # visited it on a previous Notes visit -- no edits made, so the
        # editor is clean (not dirty), and ``apply_navigation_context``
        # takes its synchronous (no-flush-needed) path.
        await _open_note_editor(screen, pilot)
        assert screen._selected_note_id == "n-1"
        assert screen._library_notes_view == "editor"
        assert screen._library_note_detail is not None
        assert screen._library_note_dirty is False

        # The retired Notes tab's "new note" deep link re-enters via this
        # same navigation context on an already-mounted (cached) screen.
        screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_NOTES_CREATE: True})
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_NOTE
        assert screen._selected_note_id == ""
        assert screen._library_notes_view == "list"
        assert screen._library_note_detail is None
        assert screen._library_note_version is None
        assert screen._library_note_dirty is False
        assert screen._library_note_autosave_state == "idle"


@pytest.mark.asyncio
async def test_library_shell_note_id_deeplink_opens_note_editor():
    """The retired Notes tab's chat-sidebar deep link now re-points into
    Library: a ``note_id`` navigation context must open that note's
    in-canvas editor directly, without requiring a prior rail-row press.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    screen = LibraryScreen(app)

    assert screen.is_mounted is False
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_NOTE_ID: "n-1"})

    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-note-title")

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen._selected_note_id == "n-1"
        title = screen.query_one("#library-note-title", Input)
        assert title.value == "Q3 retro"


@pytest.mark.asyncio
async def test_library_shell_ingest_nav_context_deeplink_lands_on_ingest_canvas():
    """Home's ingest-jobs ``Open details`` control re-points here (L3b Task
    6): a ``LIBRARY_NAV_CONTEXT_INGEST`` navigation context must land the
    shell on the in-canvas Ingest > Import media view, mirroring how
    pressing the Ingest rail row does (``LIBRARY_ROW_INGEST_MEDIA`` / canvas
    kind ``ingest-media``). Unlike the collections/note_id deep links, the
    ingest canvas needs no async data fetch, so setting the selected row id
    pre-mount is sufficient -- there is no on_mount deferral to add.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    screen = LibraryScreen(app)

    # Mirrors the real app.py ordering: handle_screen_navigation calls
    # apply_navigation_context BEFORE switch_screen mounts the destination
    # screen (see test_library_shell_collections_deeplink_loads_before_mount).
    assert screen.is_mounted is False
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})

    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
        assert screen.query_one("#library-ingest-path")


@pytest.mark.asyncio
async def test_library_shell_ingest_nav_context_deeplink_reentry_resets_stale_form():
    """(Minor, L3b Task 6 fix wave) A cached ``LibraryScreen`` re-entered via
    Home's ingest-jobs ``Open details`` deep link must never show a stale
    half-filled Import media form left over from a previous Ingest visit --
    ``_select_library_rail_row`` (the rail-row entry path) already resets
    the form on every switch via ``_reset_library_ingest_transient_state``,
    but the ``LIBRARY_NAV_CONTEXT_INGEST`` deep-link branch in
    ``_apply_navigation_context_state`` skipped that call, so a post-mount
    re-entry through the deep link (unlike the pre-mount case covered by
    ``test_library_shell_ingest_nav_context_deeplink_lands_on_ingest_canvas``)
    kept whatever the user had typed on their prior visit.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Pre-fill the ingest form programmatically, as if the user had
        # already typed into it on a previous Ingest visit.
        screen._library_ingest_form = LibraryIngestFormState(
            path="/tmp/stale-upload.txt",
            title="Stale title",
            author="Stale author",
            keywords="stale, keywords",
        )

        # Home's "Open details" control re-enters via this same navigation
        # context on an already-mounted (cached) screen.
        screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
        assert screen._library_ingest_form == LibraryIngestFormState()
        path_input = screen.query_one("#library-ingest-path", Input)
        assert path_input.value == ""


@pytest.mark.asyncio
async def test_library_shell_unknown_nav_context_mode_degrades_quietly():
    """A retired/unknown navigation-context ``mode`` (e.g. the removed
    Import/Export placeholder row's old mode value) must not raise and must
    leave the current rail selection untouched -- carried Minor from L3b
    Task 3.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        selected_before = screen._library_selected_row_id

        screen.apply_navigation_context({"mode": "import-export"})
        await pilot.pause()

        assert screen._library_selected_row_id == selected_before


@pytest.mark.asyncio
async def test_library_shell_workspaces_body_lives_under_details():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        toggle = screen.query_one("#console-rail-section-toggle-library-details")
        toggle.press()
        await pilot.pause()
        await pilot.pause()

        details_body = screen.query_one("#library-rail-section-body-details")
        create_button = screen.query_one("#library-create-local-workspace")
        assert details_body in create_button.ancestors


def test_generated_stylesheet_includes_library_shell_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        "#library-shell-grid",
        "#library-header-line",
        "#library-rail",
        "#library-canvas",
        ".library-rail-empty-copy",
        ".library-rail-row",
        ".library-rail-row-selected",
        ".library-conversation-row",
        ".library-conversation-row-selected",
        ".library-canvas-action",
        "#library-search-input",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector
    for stale in ("#library-mode-bar", "#library-contract-grid"):
        assert stale not in component_css, stale


def test_generated_stylesheet_includes_library_media_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        "#library-media-title",
        ".library-media-row",
        ".library-media-row-selected",
        "#library-media-type-filter",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector
    assert "#library-media-header" not in component_css


def _css_rule_body(css_text: str, selector: str) -> str:
    """Return the declaration body of the first ``selector { ... }`` rule.

    Args:
        css_text: Full stylesheet text to search.
        selector: A literal selector (e.g. ``"#library-rail"``) that opens a
            rule, possibly as part of a comma-separated selector list.

    Returns:
        The text between the rule's opening ``{`` and its matching ``}``.

    Raises:
        AssertionError: If the selector never opens a rule block.
    """
    pattern = re.compile(re.escape(selector) + r"\s*[,{]")
    match = pattern.search(css_text)
    assert match, f"{selector} does not open a CSS rule"
    brace_start = css_text.index("{", match.start())
    brace_end = css_text.index("}", brace_start)
    return css_text[brace_start + 1 : brace_end]


def test_library_rail_css_scrolls_vertically_with_scrollbar_styling():
    """The rail must scroll so Details content past the viewport stays reachable.

    Regression for the live-QA overflow: the rail is a plain ``Vertical``
    (Textual default ``overflow: hidden hidden``), so once every section is
    open the Details body and its workspace action buttons render past the
    pane's bottom and are neither visible nor reachable. Scoping to the
    literal ``#library-rail`` rule (not just "selector appears somewhere in
    the file") keeps this from passing on an unrelated rule that happens to
    mention the same properties elsewhere.
    """
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    for css_path in (
        root / "components" / "_agentic_terminal.tcss",
        root / "tldw_cli_modular.tcss",
    ):
        body = _css_rule_body(css_path.read_text(), "#library-rail")
        assert "overflow-y: auto" in body, css_path
        assert "scrollbar-background: $ds-surface-panel" in body, css_path
        assert "scrollbar-color: $ds-grid-line" in body, css_path


@pytest.mark.asyncio
async def test_library_shell_rail_scrolls_to_reveal_workspace_actions():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        for section in ("browse", "create", "ingest", "details"):
            body = screen.query_one(f"#library-rail-section-body-{section}")
            if body.styles.display == "none":
                screen.query_one(f"#console-rail-section-toggle-library-{section}").press()
                await pilot.pause()
                await pilot.pause()

        rail = screen.query_one("#library-rail")
        assert str(rail.styles.overflow_y) == "auto"

        rail.scroll_end(animate=False)
        await pilot.pause()
        await pilot.pause()

        for selector in ("#library-create-local-workspace", "#library-use-in-console"):
            button = screen.query_one(selector)
            assert rail.region.y <= button.region.y < rail.region.y + rail.region.height, (
                f"{selector} region {button.region} is not within the scrolled rail "
                f"viewport {rail.region}"
            )


@pytest.mark.asyncio
async def test_library_shell_details_body_has_no_duplicate_collapse_control():
    """The moved Workspaces body must not carry its own collapse affordance.

    The Details section already owns exactly one collapse/expand toggle (the
    rail's ``ConsoleRailSectionHeader``, a sibling of the body, tooltipped
    "Collapse Details"/"Expand Details"). Nothing inside the body itself
    should duplicate that affordance.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        toggle = screen.query_one("#console-rail-section-toggle-library-details")
        toggle.press()
        await pilot.pause()
        await pilot.pause()

        details_body = screen.query_one("#library-rail-section-body-details")
        for button in details_body.query(Button):
            label = str(button.label)
            tooltip = button.tooltip or ""
            assert "Collapse Details" not in label
            assert "Collapse Details" not in tooltip


@pytest.mark.asyncio
async def test_library_shell_media_use_in_chat_body_truncated_for_long_content():
    """``body_truncated`` is True when media content exceeds 500 chars.

    The handoff excerpt is capped at LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS (500),
    so long content must set body_truncated=True. This test seeds media with
    600 chars and asserts both the flag and that the excerpt is bounded.
    """
    app = _build_test_app()
    media_items = _two_media_items()[:1]
    # Seed with content > 500 chars: use a marker string repeated to exceed 500.
    long_content = "This is long content. " * 35  # ~770 chars
    media_items[0]["content"] = long_content
    _seed_conversations(app, [], media=media_items)
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("media", "media-1", "Interview Recording"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        screen.query_one("#library-media-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-media-use-in-chat")

        screen.query_one("#library-media-use-in-chat").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "library"
    assert payload.item_type == "media"
    assert payload.source_id == "media-1"
    assert payload.title == "Interview Recording"
    assert payload.body_truncated is True
    # The payload.body includes metadata headers plus the excerpt.
    # Verify the excerpt is bounded: a marker string only in the first
    # 500 chars should appear, but content after position 500 should not.
    assert "This is long content." in payload.body
    # Verify truncation: content after char 500 should not be present.
    # The full content string repeated would produce a pattern that continues,
    # so if truncated, it will end mid-word or mid-pattern.
    assert "Content excerpt:" in payload.body
    # Count the actual content after "Content excerpt:" header
    excerpt_start = payload.body.find("Content excerpt:")
    content_part = payload.body[excerpt_start:]
    # The excerpt part should be much shorter than the original content
    assert len(content_part) < len(long_content)


@pytest.mark.asyncio
async def test_library_shell_media_use_in_chat_body_not_truncated_for_short_content():
    """``body_truncated`` is False when media content is <= 500 chars.

    Short content does not set the truncated flag. This test seeds media with
    300 chars and asserts body_truncated=False and that the full content
    is included in the excerpt.
    """
    app = _build_test_app()
    media_items = _two_media_items()[:1]
    # Seed with content < 500 chars: a 300-char realistic string
    short_content = (
        "This is a medium-length transcript excerpt that stays well under "
        "the 500-character limit. It contains enough text to be substantial "
        "but short enough to pass through without truncation. This demonstrates "
        "that the body_truncated flag correctly handles content that fits "
        "entirely within the handoff excerpt boundary."
    )
    media_items[0]["content"] = short_content
    _seed_conversations(app, [], media=media_items)
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("media", "media-1", "Interview Recording"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        screen.query_one("#library-media-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-media-use-in-chat")

        screen.query_one("#library-media-use-in-chat").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "library"
    assert payload.item_type == "media"
    assert payload.source_id == "media-1"
    assert payload.title == "Interview Recording"
    assert payload.body_truncated is False
    # Full content should be present (not truncated) in the body payload.
    assert short_content in payload.body
    # Verify the structure includes the excerpt section.
    assert "Content excerpt:" in payload.body


def _two_notes():
    return [
        {"id": "n-1", "title": "Q3 retro", "content": "alpha budget line",
         "last_modified": "2026-07-07T11:57:00+00:00", "version": 2},
        {"id": "n-2", "title": "Reading list", "content": "bravo",
         "last_modified": "2026-07-06T12:00:00+00:00", "version": 1},
    ]


class _CountSeamLibraryNotesScopeService(StaticLibraryNotesListScopeService):
    """Local notes fake mirroring the real production shape: ``list_notes``
    returns a bare list with no total (like ``NotesInteropService.list_notes``),
    but ``count_notes`` (the new seam under test) gives the exact total.

    Used to prove the Library screen's rail badge renders an exact count
    even though the paginated ``list_notes`` response alone cannot supply
    one -- unlike ``StaticLibraryNotesScopeService``, whose ``list_notes``
    already carries a ``pagination.total`` and so cannot distinguish
    "badge is exact because of ``count_notes``" from "badge is exact
    because ``list_notes`` said so".
    """

    def __init__(self, notes):
        super().__init__(notes)
        self.count_calls = []

    async def count_notes(self, *, scope, user_id=None, **kwargs):
        self.count_calls.append({"scope": scope, "user_id": user_id, **kwargs})
        return len(self.notes)


@pytest.mark.asyncio
async def test_library_shell_notes_rail_badge_shows_exact_count_via_count_seam():
    """The rail badge renders an exact ``(2)`` -- no "+" sample-cap suffix
    -- once ``count_notes`` is wired into the Library screen's local-source
    snapshot fetch, even though the underlying ``list_notes`` response
    carries no total of its own (the real production shape)."""
    app = _build_test_app()
    app.notes_scope_service = _CountSeamLibraryNotesScopeService(_two_notes())
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        rail_label = str(screen.query_one("#library-row-browse-notes").label)
        assert "(2)" in rail_label
        assert "(2+)" not in rail_label
    assert app.notes_scope_service.count_calls


@pytest.mark.asyncio
async def test_library_shell_notes_rail_badge_degrades_without_count_seam():
    """When the local notes service exposes no ``count_notes`` seam (the
    plain-list fake never defines one, mirroring a backend that hasn't
    adopted the new seam), the rail badge keeps today's "showing up to N"
    sample-cap contract (``(N+)``) instead of claiming an exact total it
    cannot verify -- and the snapshot fetch does not error out."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService(_two_notes())
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        rail_label = str(screen.query_one("#library-row-browse-notes").label)
        assert "(2+)" in rail_label


class _FakeStudyScopeService:
    """Minimal study-scope fake exposing only the count seams under test.

    Mirrors the real ``StudyScopeService``'s ``count_decks``/
    ``count_due_flashcards`` shape (async, no required args) without going
    through the local/server routing -- same spirit as the Static* fakes
    above for notes/media/conversations.
    """

    def __init__(self, *, decks, due_flashcards):
        self._decks = decks
        self._due_flashcards = due_flashcards
        self.count_decks_calls = []
        self.count_due_flashcards_calls = []

    async def count_decks(self, **kwargs):
        self.count_decks_calls.append(kwargs)
        return self._decks

    async def count_due_flashcards(self, **kwargs):
        self.count_due_flashcards_calls.append(kwargs)
        return self._due_flashcards


class _FakeQuizScopeService:
    """Minimal quiz-scope fake exposing only the ``count_quizzes`` seam."""

    def __init__(self, *, quizzes):
        self._quizzes = quizzes
        self.count_quizzes_calls = []

    async def count_quizzes(self, **kwargs):
        self.count_quizzes_calls.append(kwargs)
        return self._quizzes


@pytest.mark.asyncio
async def test_library_shell_create_rail_shows_study_and_quiz_counts():
    """The Create rail renders exact live counts once the study/quiz
    scope-service count seams (Task 8) are wired into the Library screen's
    local-source snapshot fetch: flashcards due gets the special
    "Flashcards due: N" copy (bright emphasis when N > 0), decks/quizzes
    render the ordinary "(N)" suffix."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.study_scope_service = _FakeStudyScopeService(decks=3, due_flashcards=7)
    app.study_quiz_scope_service = _FakeQuizScopeService(quizzes=2)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        flashcards_button = screen.query_one("#library-row-create-flashcards", Button)
        decks_label = str(screen.query_one("#library-row-create-study").label)
        quizzes_label = str(screen.query_one("#library-row-create-quizzes").label)

        assert "Flashcards due: 7" in str(flashcards_button.label)
        assert "Study decks (3)" in decks_label
        assert "Quizzes (2)" in quizzes_label
        assert flashcards_button.has_class("library-rail-row-due-bright")
        assert not flashcards_button.has_class("library-rail-row-due-dim")

    assert app.study_scope_service.count_decks_calls
    assert app.study_scope_service.count_due_flashcards_calls
    assert app.study_quiz_scope_service.count_quizzes_calls


@pytest.mark.asyncio
async def test_library_shell_create_rail_flashcards_due_zero_renders_dim():
    """Zero due flashcards still renders the exact "due: 0" copy, but with
    the dim (not bright) emphasis class -- there is nothing to act on."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.study_scope_service = _FakeStudyScopeService(decks=3, due_flashcards=0)
    app.study_quiz_scope_service = _FakeQuizScopeService(quizzes=2)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        flashcards_button = screen.query_one("#library-row-create-flashcards", Button)
        assert "Flashcards due: 0" in str(flashcards_button.label)
        assert flashcards_button.has_class("library-rail-row-due-dim")
        assert not flashcards_button.has_class("library-rail-row-due-bright")


@pytest.mark.asyncio
async def test_library_shell_create_rail_degrades_without_study_count_seams():
    """When the study/quiz services expose neither count seam (mirroring a
    runtime where those services are absent or unwired), the Create rows
    render uncounted -- no crash, no error copy leaking from the decorative
    study counts into the Library lookup-error state that the three browse
    sources use."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.study_scope_service = object()
    app.study_quiz_scope_service = object()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        flashcards_button = screen.query_one("#library-row-create-flashcards", Button)
        decks_label = str(screen.query_one("#library-row-create-study").label)
        quizzes_label = str(screen.query_one("#library-row-create-quizzes").label)

        assert "due:" not in str(flashcards_button.label)
        assert "(" not in decks_label
        assert "(" not in quizzes_label
        assert not flashcards_button.has_class("library-rail-row-due-bright")
        assert not flashcards_button.has_class("library-rail-row-due-dim")
        # Study-count degrade must never surface as a Library lookup error --
        # that error state is reserved for the three browse sources.
        assert screen._library_lookup_error is None


@pytest.mark.asyncio
async def test_library_shell_create_rail_flashcards_due_count_reads_in_memory_db():
    """F4a (PR #590 review, Qodo): study/quiz counts must not be forced onto
    a worker thread when ChaChaNotes is an in-memory SQLite DB. SQLite
    ``:memory:`` connections are thread-local (``threading.local``), so a
    worker thread invoking ``count_due_flashcards``/``count_decks`` would
    open a brand-new, unmigrated in-memory connection and the count query
    would fail -- degrading the badge to uncounted even though real data
    exists. Mirrors the ``is_memory_db`` guard already used by
    ``LibraryLocalRagSearchService._search_conversations`` and
    ``LibraryScreen._fetch_library_conversation_by_id``. Uses real
    ``StudyScopeService``/``QuizScopeService`` wrapping real
    ``LocalStudyService``/``LocalQuizService`` over a real in-memory
    ``CharactersRAGDB`` (not fakes) so the thread-locality is genuinely
    exercised.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    db = CharactersRAGDB(":memory:", client_id="test-client")
    try:
        deck_id = db.create_deck("Biology")
        db.create_flashcard(
            {"deck_id": deck_id, "front": "ATP", "back": "Energy", "tags": "", "type": "basic"}
        )
        app.chachanotes_db = db
        app.study_scope_service = StudyScopeService(
            local_service=LocalStudyService(db), server_service=None
        )
        app.study_quiz_scope_service = QuizScopeService(
            local_service=LocalQuizService(db), server_service=None
        )
        host = LibraryHarness(app)

        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)

            flashcards_button = screen.query_one("#library-row-create-flashcards", Button)
            decks_label = str(screen.query_one("#library-row-create-study").label)

            assert "Flashcards due: 1" in str(flashcards_button.label)
            assert "Study decks (1)" in decks_label
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_library_shell_notes_row_opens_notes_list_canvas():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        header = str(screen.query_one("#library-notes-header").renderable)
        assert header == "Notes (2)"
        assert screen.query_one("#library-notes-filter")
        assert screen.query_one("#library-notes-sort")


@pytest.mark.asyncio
async def test_library_shell_notes_sort_button_cycles():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-sort")
        assert screen._library_notes_sort == "newest"
        screen.query_one("#library-notes-sort").press()
        await pilot.pause()
        assert screen._library_notes_sort == "oldest"


@pytest.mark.asyncio
async def test_library_shell_notes_filter_queries_search_seam():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-filter")
        box = screen.query_one("#library-notes-filter", Input)
        box.value = "retro"
        box.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(); await pilot.pause()
        service = app.notes_scope_service
        assert service.search_calls[-1]["query"] == "retro"
        assert screen._library_notes_filter == "retro"


class _GatedSearchLibraryNotesScopeService(StaticLibraryNotesScopeService):
    """A ``search_notes`` that blocks until the test releases it, so a
    filter test can submit, observe the seam call start, mutate screen
    state while that call is still in flight, and only then let the stale
    response resolve on demand.

    Uses a ``threading.Event`` rather than an ``asyncio.Event``:
    ``_run_library_service_call`` runs this seam with
    ``isolate_in_worker=True``, which executes it in a *different* thread
    under a freshly-created event loop (``asyncio.run`` inside
    ``asyncio.to_thread``) -- an ``asyncio.Event`` set from the test's own
    thread/loop would not safely wake a waiter parked on that other loop.
    """

    def __init__(self, notes):
        super().__init__(notes)
        self.release_event = threading.Event()

    async def search_notes(self, **kwargs):
        self.search_calls.append(kwargs)
        await asyncio.to_thread(self.release_event.wait, _GATED_RELEASE_TIMEOUT_SECONDS)
        return await super().search_notes(
            scope=kwargs.get("scope"),
            query=kwargs.get("query"),
            limit=kwargs.get("limit"),
            user_id=kwargs.get("user_id"),
            offset=kwargs.get("offset", 0),
        )


@pytest.mark.asyncio
async def test_library_shell_notes_filter_clears_before_stale_response_lands():
    """Clearing the filter while a slower ``search_notes`` call is still in
    flight must not let that stale response overwrite the cleared filter
    state once it resolves -- the same stale-result discipline the note
    detail fetch and save already apply to their own out-of-order results.
    """
    app = _build_test_app()
    service = _GatedSearchLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-filter")

        box = screen.query_one("#library-notes-filter", Input)
        box.value = "retro"
        box.focus()
        await pilot.pause()
        await pilot.press("enter")

        for _ in range(150):
            if service.search_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Filter submit never called search_notes.")

        assert screen._library_notes_filter == "retro"

        # Clear the filter while the "retro" search is still gated in flight.
        clear_box = screen.query_one("#library-notes-filter", Input)
        clear_box.value = ""
        clear_box.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert screen._library_notes_filter == ""
        assert screen._library_notes_filter_records is None

        # Now release the stale "retro" response.
        service.release_event.set()
        for _ in range(20):
            await pilot.pause(0.02)

        assert screen._library_notes_filter_records is None, (
            "The stale in-flight filter response overwrote the cleared filter state."
        )
        header = str(screen.query_one("#library-notes-header").renderable)
        assert header == "Notes (2)"


@pytest.mark.asyncio
async def test_library_shell_notes_row_opens_editor_with_detail():
    """Pressing a note row fetches the full detail and renders the in-canvas
    editor prefilled with the note's title, body, and version."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        title = screen.query_one("#library-note-title", Input)
        assert title.value == "Q3 retro"
        body = screen.query_one("#library-note-body", TextArea)
        assert body.text == "alpha budget line"
        meta = str(screen.query_one("#library-note-meta").renderable)
        assert "v2" in meta
        assert screen._library_notes_view == "editor"
        assert screen._selected_note_id == "n-1"


class StaticLibraryNotesKeywordsService:
    """Fake ``app.notes_service`` stand-in for the keywords-enrichment seam
    ``_refresh_library_note_detail`` uses (mirroring
    ``NotesInteropService.get_keywords_for_note``'s real
    ``(user_id, note_id) -> list[{"keyword": ...}]`` shape -- see
    ``ChaChaNotes_DB.get_keywords_for_note``, whose rows always carry a
    ``keyword`` column).
    """

    def __init__(self, keywords_by_note_id):
        self.keywords_by_note_id = dict(keywords_by_note_id)
        self.calls = []

    def get_keywords_for_note(self, user_id, note_id):
        self.calls.append({"user_id": user_id, "note_id": note_id})
        return [
            {"id": index, "keyword": keyword}
            for index, keyword in enumerate(
                self.keywords_by_note_id.get(str(note_id), []), start=1
            )
        ]


@pytest.mark.asyncio
async def test_library_shell_note_editor_shows_enriched_keywords():
    """``notes_scope_service.get_note_detail``'s local-scope shape is the
    raw ``notes`` table row and never carries a ``keywords`` field (see
    ``NotesInteropService.get_note_by_id``), so the in-canvas editor's
    Keywords input would otherwise always render empty even for a note
    with persisted keywords. ``_refresh_library_note_detail`` must enrich
    the fetched detail via the ``app.notes_service.get_keywords_for_note``
    seam before building editor state.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notes_service = StaticLibraryNotesKeywordsService({"n-1": ["kw1", "kw2"]})
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        keywords_input = screen.query_one("#library-note-keywords", Input)
        for _ in range(150):
            if keywords_input.value == "kw1, kw2":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Keywords input never showed enriched keywords: {keywords_input.value!r}"
            )

        assert app.notes_service.calls
        assert app.notes_service.calls[-1]["note_id"] == "n-1"


@pytest.mark.asyncio
async def test_library_shell_note_editor_keywords_empty_without_notes_service():
    """Absent ``app.notes_service`` (or a service missing the method), the
    editor still opens normally with an empty Keywords input -- the
    enrichment is quiet-best-effort, never a hard requirement to open a
    note (matches the current/pre-enrichment behavior).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        assert screen.query_one("#library-note-keywords", Input).value == ""


@pytest.mark.asyncio
async def test_library_shell_note_back_returns_to_list():
    """The editor's Back action returns to the notes list and clears the
    selected note/detail state."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        screen.query_one("#library-note-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")

        assert screen._library_notes_view == "list"
        assert screen._selected_note_id == ""
        assert screen._library_note_detail is None
        assert not screen.query("#library-note-title")


@pytest.mark.asyncio
async def test_library_shell_notes_rail_reentry_resets_to_list():
    """Leaving the notes editor for another rail row and returning to
    Browse > Notes must land on the list, not the previously-open editor."""
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=_two_notes(), media=_two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")

        assert not screen.query("#library-note-title")
        assert screen._library_notes_view == "list"
        assert screen._selected_note_id == ""
        assert screen._library_note_detail is None


@pytest.mark.asyncio
async def test_library_shell_note_detail_race_discards_stale_fetch():
    """A detail fetch that completes for a no-longer-selected note id must be
    discarded instead of overwriting the currently-open editor."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        for _ in range(150):
            detail = screen._library_note_detail
            if isinstance(detail, dict) and str(detail.get("id")) == "n-1":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("n-1 detail never loaded.")
        assert screen._selected_note_id == "n-1"

        # Simulate a slower in-flight fetch for the previously-selected n-2
        # completing now: it must not overwrite n-1's detail.
        await screen._refresh_library_note_detail("n-2")

        detail = screen._library_note_detail
        assert isinstance(detail, dict)
        assert str(detail.get("id")) == "n-1"


class _DelayedSaveLibraryNotesScopeService(StaticLibraryNotesScopeService):
    """A ``save_note`` that stalls before delegating, so a test can observe
    "the seam call started" as an event distinct from "the save resolved" --
    proving the caller actually awaited the save before doing anything
    else, rather than merely calling it eventually.

    Two stall mechanisms are supported: a fixed ``delay`` (the default --
    good enough when a test only needs "this save takes a moment"), or an
    explicit ``release_event`` (a ``threading.Event``) when a test needs to
    hold a save open indefinitely and release multiple concurrent calls
    together on demand -- e.g. proving an autosave and a flush's save don't
    race each other. The blocking ``.wait()`` is dispatched via
    ``asyncio.to_thread``, mirroring ``_GatedSearchLibraryNotesScopeService``:
    ``save_note`` runs with ``isolate_in_worker=True``, i.e. inside its own
    thread with its own event loop, so an ``asyncio.Event`` set from the
    test's thread/loop would not safely wake a waiter parked on that other
    loop.
    """

    def __init__(self, notes, *, delay: float = 0.15, release_event: threading.Event | None = None):
        super().__init__(notes)
        self.delay = delay
        self.release_event = release_event
        self.save_started: list[str] = []

    async def save_note(self, **kwargs):
        self.save_started.append(str(kwargs.get("note_id") or ""))
        if self.release_event is not None:
            await asyncio.to_thread(self.release_event.wait, _GATED_RELEASE_TIMEOUT_SECONDS)
        else:
            await asyncio.sleep(self.delay)
        return await super().save_note(**kwargs)


async def _open_note_editor(screen, pilot, note_id_suffix: str = "n-1"):
    """Drive the shared path to the in-canvas note editor for ``_two_notes()``'s
    first row, then let the mount-time armed-flag callback settle.
    """
    screen.query_one("#library-row-browse-notes").press()
    await _wait_for_selector(screen, pilot, "#library-notes-row-0")
    screen.query_one("#library-notes-row-0").press()
    await _wait_for_selector(screen, pilot, "#library-note-title")
    await pilot.pause()
    await pilot.pause()


def _bump_note_version_externally(service, note_id: str, **field_overrides) -> None:
    """Simulate another writer changing a note: bump its stored version (and
    optionally other fields) on the fake, out from under a screen that still
    has the old version cached -- the setup for every conflict pilot below.
    """
    service.notes = tuple(
        dict(note, version=int(note["version"]) + 1, **field_overrides)
        if note.get("id") == note_id
        else note
        for note in service.notes
    )


def _remove_note_externally(service, note_id: str) -> None:
    """Simulate the note being deleted elsewhere entirely (not just a
    version bump): a subsequent ``get_note_detail`` for it resolves to
    ``None`` on the fake, mirroring a real deleted/never-existed note.
    """
    service.notes = tuple(note for note in service.notes if note.get("id") != note_id)


@pytest.mark.asyncio
async def test_library_shell_note_explicit_save_calls_seam_and_bumps_version():
    """Pressing Save calls the seam with the sanitized fields/keywords list,
    bumps the in-memory version from the seam's dict result, updates the
    meta line to show "saved", and never recomposes the editor -- the
    ``TextArea`` instance must be the exact same object before and after.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        body_before_save = screen.query_one("#library-note-body", TextArea)
        body_before_save.text = "alpha budget line, edited"
        screen.query_one("#library-note-keywords", Input).value = "alpha, beta"
        await pilot.pause()

        screen.query_one("#library-note-save").press()

        service = app.notes_scope_service
        for _ in range(150):
            if service.save_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Save never called the save_note seam.")

        call = service.save_calls[-1]
        assert call["scope"] == "local_note"
        assert call["note_id"] == "n-1"
        assert call["version"] == 2
        assert call["title"] == "Q3 retro"
        assert call["content"] == "alpha budget line, edited"
        assert call["keywords"] == ["alpha", "beta"]

        for _ in range(150):
            if screen._library_note_version == 3:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(f"Version never bumped (still {screen._library_note_version!r}).")

        assert screen.query_one("#library-note-body", TextArea) is body_before_save
        meta = str(screen.query_one("#library-note-meta").renderable)
        assert "saved" in meta


@pytest.mark.asyncio
async def test_library_shell_note_save_patches_detail_title_and_content_for_later_recompose():
    """A save only ever updates the ``#library-note-meta`` ``Static`` in
    place -- it never recomposes the editor itself. But a *later*
    recompose (entering the delete-confirm state, in this test) rebuilds
    the editor's ``TextArea``/``Input`` values fresh from
    ``_library_note_detail``. If Save only patched the cached ``version``
    there and left ``title``/``content`` stale, that later recompose would
    silently revert the just-saved edit back to the pre-save text on
    screen.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "edited body text"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "saved":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Save never completed.")

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm-copy")
        screen.query_one("#library-note-delete-cancel").press()
        await pilot.pause()

        assert screen.query_one("#library-note-body", TextArea).text == "edited body text", (
            "A later recompose reverted the saved edit -- the detail mirror "
            "wasn't patched with the saved title/content."
        )


@pytest.mark.asyncio
async def test_library_shell_note_autosave_fires_after_debounce(monkeypatch):
    """Editing the body (no Save press) arms the autosave debounce; once it
    fires, ``save_note`` was called and the meta line shows "saved".
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.05)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "alpha budget line, autosaved"
        await pilot.pause()

        service = app.notes_scope_service
        for _ in range(150):
            if service.save_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Autosave never called save_note without a Save press.")

        assert service.save_calls[-1]["content"] == "alpha budget line, autosaved"

        meta = ""
        for _ in range(150):
            meta = str(screen.query_one("#library-note-meta").renderable)
            if "saved" in meta:
                break
            await pilot.pause(0.02)
        assert "saved" in meta


@pytest.mark.asyncio
async def test_library_flush_pending_work_saves_dirty_note_and_reports_conflicts():
    """``flush_pending_work`` (the app's nav-away hook) must persist a dirty
    note before the screen instance is discarded, and must return False when
    the flush surfaces an unresolved save conflict so the app vetoes the
    navigation instead of destroying the editor (and the user's edits) with
    the outgoing screen.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "edited mid-tab-switch"
        await pilot.pause()

        allowed = await screen.flush_pending_work()

        service = app.notes_scope_service
        assert allowed is True
        assert service.save_calls, "flush_pending_work never persisted the dirty note"
        assert service.save_calls[-1]["content"] == "edited mid-tab-switch"

        # Unsaved edits surviving the flush must veto the navigation: a
        # FAILED save leaves the edits only in this screen instance, so
        # discarding it would lose them. Force a genuine save failure
        # through the real seam and re-dirty the editor.
        screen.query_one("#library-note-body", TextArea).text = "edit that must not be lost"
        await pilot.pause()

        async def _failing_save(**kwargs):
            raise RuntimeError("simulated save failure")

        service.save_note = _failing_save
        assert await screen.flush_pending_work() is False
        assert screen._library_note_autosave_state == "error"
        assert screen._library_note_dirty is True


@pytest.mark.asyncio
async def test_library_shell_note_flush_on_back_saves_before_view_switches():
    """Editing the body then immediately pressing Back must flush (await)
    the pending save before the canvas leaves the editor -- proven by
    observing the view is still "editor" while the (deliberately slow)
    save is in flight, and only becomes "list" once it resolves.
    """
    app = _build_test_app()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "alpha budget line, flushed"
        await pilot.pause()

        screen.query_one("#library-note-back").press()
        for _ in range(50):
            if service.save_started:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("Back never triggered the flush save.")

        # The save is still sleeping: the view must not have switched yet.
        assert screen._library_notes_view == "editor"

        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Back never completed once the flush resolved.")

        assert service.save_calls, "The flushed save never actually completed."
        assert service.save_calls[-1]["content"] == "alpha budget line, flushed"


@pytest.mark.asyncio
async def test_library_shell_note_flush_on_rail_switch_saves_before_switching():
    """Editing the body then immediately switching rail rows must flush the
    pending save before the rail switch takes effect, mirroring the
    flush-on-Back contract for the ``_select_library_rail_row`` exit path.
    """
    app = _build_test_app()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService(_two_media_items())
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "alpha budget line, rail-flushed"
        await pilot.pause()

        screen.query_one("#library-row-browse-media").press()
        for _ in range(50):
            if service.save_started:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("Rail switch never triggered the flush save.")

        # The save is still sleeping: the rail switch must not have applied yet.
        assert screen._library_selected_row_id == "browse-notes"

        for _ in range(150):
            if screen._library_selected_row_id == "browse-media":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Rail switch never completed once the flush resolved.")

        assert service.save_calls, "The flushed save never actually completed."
        assert service.save_calls[-1]["content"] == "alpha budget line, rail-flushed"


@pytest.mark.asyncio
async def test_library_shell_note_flush_on_notes_create_deeplink_saves_before_switching():
    """A ``notes_create`` deep link arriving on an already-mounted Library
    screen that holds a dirty note editor must flush the pending save before
    tearing the editor down -- the same flush-then-apply contract the Back
    and rail-switch exits honour.

    ``apply_navigation_context`` is the retired Notes tab's re-pointed entry;
    navigation composes fresh screens, but this defense-in-depth branch
    covers any direct caller that runs it on a mounted, dirty editor
    mid-edit. Without the flush the recompose to
    the create view destroys the ``#library-note-body`` the debounced
    autosave would have read, silently dropping the last edits. Proven by
    the (deliberately slow) save being in flight while the selected row is
    still ``browse-notes`` and only becoming ``create-note`` once it
    resolves.
    """
    app = _build_test_app()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService(_two_media_items())
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "alpha budget line, deeplink-flushed"
        await pilot.pause()

        screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_NOTES_CREATE: True})
        for _ in range(50):
            if service.save_started:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("notes_create deep link never triggered the flush save.")

        # The save is still sleeping: the create view must not have applied yet.
        assert screen._library_selected_row_id == "browse-notes"

        for _ in range(150):
            if screen._library_selected_row_id == "create-note":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("notes_create deep link never applied once the flush resolved.")

        assert service.save_calls, "The flushed save never actually completed."
        assert service.save_calls[-1]["content"] == "alpha budget line, deeplink-flushed"


@pytest.mark.asyncio
async def test_library_shell_flush_waits_for_inflight_autosave(monkeypatch):
    """Back's exit-flush must wait for an in-flight autosave save instead of
    racing its own inline save against it with the same stale version.

    Without the wait, the debounced autosave's ``save_note`` call and the
    flush's inline ``save_note`` call both carry the same
    (not-yet-bumped) version -- an optimistic-lock version conflict that
    the local backend would only ever raise for *another* writer now fires
    against nobody but the note's own autosave, popping a spurious
    "changed elsewhere" conflict banner and aborting the Back navigation.

    Uses ``_DelayedSaveLibraryNotesScopeService``'s ``release_event`` gate
    so the autosave's ``save_note`` call can be held open indefinitely:
    Back is pressed while it is still in flight, proving the flush does
    not issue a second concurrent ``save_note`` call while waiting, then
    the event is released and the flush is proven to complete cleanly.
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.05)
    app = _build_test_app()
    release_event = threading.Event()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes(), release_event=release_event)
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "alpha budget line, autosave-inflight"
        await pilot.pause()

        # Let the autosave debounce fire; its save_note blocks on
        # release_event (in its own worker thread) so it is genuinely
        # in-flight when Back is pressed.
        for _ in range(150):
            if service.save_started:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Autosave never called save_note.")
        assert len(service.save_started) == 1

        # The Back handler's flush awaits the in-flight save worker, which
        # blocks the message pump -- so ``pilot.pause`` cannot observe the
        # intermediate "flush is waiting" state (it waits for an idle pump
        # that only arrives once the wait completes). Release the gate from
        # a separate thread, independent of the loop, so the in-flight save
        # can finish and the whole Back->flush->navigate sequence settles;
        # then assert on the OUTCOME.
        threading.Timer(0.2, release_event.set).start()
        screen.query_one("#library-note-back").press()

        for _ in range(300):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Back never completed after the in-flight autosave resolved "
                f"(view={screen._library_notes_view!r}, "
                f"autosave_state={screen._library_note_autosave_state!r}, "
                f"save_started={service.save_started!r})."
            )

        # No spurious self-conflict: nobody else touched the note, so the
        # flush must not have raced its own save against the autosave with a
        # stale version.
        assert not screen.query("#library-note-conflict-overwrite"), (
            "A spurious self-conflict was raised even though nobody else "
            "touched the note."
        )
        assert screen._library_note_autosave_state != "conflict"
        # The autosave's save completed; the flush then saw dirty cleared and
        # skipped its own redundant save -> exactly one save_note call, at the
        # note's actual stored version (2), never a stale-version second call.
        assert service.save_calls, "The in-flight autosave's save never completed."
        assert len(service.save_started) == 1, (
            "The flush raced a second concurrent save_note against the "
            f"in-flight autosave (save_started={service.save_started!r})."
        )
        assert service.save_calls[0]["version"] == 2


@pytest.mark.asyncio
async def test_library_shell_note_save_result_after_switch_is_discarded():
    """A save started for note A that only resolves after the user has
    already switched to note B's editor must not land its version bump or
    "saved" meta status on B -- the same stale-result discipline
    ``_refresh_library_note_detail`` and ``_resolve_library_note_conflict``
    already apply to their own out-of-order results.
    """
    app = _build_test_app()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)  # opens n-1 (seeded at version 2)

        assert screen._selected_note_id == "n-1"
        assert screen._library_note_version == 2

        # Arm a save for n-1 whose fake response is deliberately delayed, but
        # never dirty the editor -- the save is triggered purely by the Save
        # button so ``_library_note_dirty`` stays False and a subsequent row
        # switch's flush is a no-op, letting the switch complete immediately
        # while this save is still in flight.
        screen.query_one("#library-note-save").press()
        for _ in range(50):
            if service.save_started:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("Save never called the save_note seam.")

        # Switch to n-2's editor while n-1's save is still sleeping. The
        # notes canvas only renders row buttons in its list view, so this
        # goes via Back (a no-op flush, since nothing is dirty) then a row
        # press -- the same "row press" switch path a user takes.
        screen.query_one("#library-note-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-1")
        screen.query_one("#library-notes-row-1").press()
        for _ in range(150):
            if screen._selected_note_id == "n-2":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Row switch to n-2 never completed.")

        for _ in range(150):
            detail = screen._library_note_detail
            if isinstance(detail, dict) and str(detail.get("id")) == "n-2":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("n-2 detail never loaded.")

        assert screen._library_note_version == 1  # n-2's own seeded version
        n2_meta_before = str(screen.query_one("#library-note-meta").renderable)
        assert "saved" not in n2_meta_before

        # Now let the stale n-1 save resolve and give the (buggy) mutation a
        # few pause cycles to land, if it were going to.
        for _ in range(150):
            if service.save_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The stale n-1 save never resolved.")
        for _ in range(10):
            await pilot.pause(0.02)

        assert screen._selected_note_id == "n-2"
        assert screen._library_note_version == 1, (
            "n-1's stale save result clobbered n-2's version: "
            f"{screen._library_note_version!r}"
        )
        n2_meta_after = str(screen.query_one("#library-note-meta").renderable)
        assert "saved" not in n2_meta_after, (
            f"n-1's stale save result leaked into n-2's meta line: {n2_meta_after!r}"
        )


@pytest.mark.asyncio
async def test_library_shell_note_conflict_shows_overwrite_reload_and_keeps_user_text():
    """A version conflict (the seam returns ``False``) stops the autosave
    timer, shows the Overwrite/Reload actions, and re-seeds the editor from
    the user's own kept text -- never the stale server-side detail.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-body", TextArea).text = "kept text that must survive"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        assert screen.query("#library-note-conflict-overwrite")
        assert screen.query("#library-note-conflict-reload")
        meta = str(screen.query_one("#library-note-meta").renderable)
        assert "changed elsewhere" in meta
        assert screen.query_one("#library-note-body", TextArea).text == (
            "kept text that must survive"
        )


@pytest.mark.asyncio
async def test_library_shell_note_conflict_during_preview_reads_live_text():
    """A save conflict raised while Preview is toggled on must not leave a
    stale ``_library_note_preview``/``_library_note_preview_snapshot`` behind:
    the conflict UI always renders the live ``TextArea`` (``compose`` never
    threads ``preview`` through for the conflict branch), so
    ``_read_library_note_editor_fields`` still reading through the old
    preview snapshot would send stale, pre-preview text to the seam on a
    save from the conflict UI -- and visibly revert the user's on-screen
    edits on the next recompose.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service

        screen.query_one("#library-note-body", TextArea).text = "T1"
        await pilot.pause()

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-preview-body")
        assert screen._library_note_preview is True

        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        # The conflict UI always shows the live TextArea, never the
        # read-only Markdown preview, regardless of the Preview flag.
        conflict_body = screen.query_one("#library-note-body", TextArea)
        assert conflict_body.text == "T1"

        conflict_body.text = "T2"
        await pilot.pause()

        calls_before_second_save = len(service.save_calls)
        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if len(service.save_calls) > calls_before_second_save:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The second Save press never called the seam.")
        # Give the resulting conflict recompose a few cycles to settle.
        for _ in range(10):
            await pilot.pause(0.02)

        assert service.save_calls[-1]["content"] == "T2", (
            "The conflict-UI save sent stale pre-preview text to the seam: "
            f"{service.save_calls[-1]['content']!r}"
        )
        assert screen.query_one("#library-note-body", TextArea).text == "T2", (
            "The second conflict recompose reverted the user's on-screen edit."
        )


@pytest.mark.asyncio
async def test_library_shell_note_conflict_overwrite_resaves_with_fresh_version():
    """Overwrite re-fetches the note silently, takes only the fresh
    version, and re-saves the user's kept text at that version -- ending
    with the fake holding the user's text at the newest version.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-body", TextArea).text = "kept text that must survive"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        (await _wait_for_selector(screen, pilot, "#library-note-conflict-overwrite")).press()
        for _ in range(150):
            if screen._library_note_autosave_state == "saved":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Overwrite never completed.")

        assert not screen.query("#library-note-conflict-overwrite")
        stored = next(note for note in service.notes if note["id"] == "n-1")
        assert stored["content"] == "kept text that must survive"
        assert stored["version"] == 4  # v2 -> externally bumped to v3 -> overwrite saves to v4
        assert screen._library_note_version == 4


@pytest.mark.asyncio
async def test_library_shell_note_conflict_reload_discards_local_edits():
    """Reload discards the local edit and recomposes the editor from the
    freshly re-fetched server-side detail.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(
            service, "n-1", title="Server-side title", content="Server-side content"
        )

        screen.query_one("#library-note-body", TextArea).text = "local edits to discard"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        (await _wait_for_selector(screen, pilot, "#library-note-conflict-reload")).press()
        for _ in range(150):
            if (
                screen._library_note_autosave_state == "idle"
                and screen._library_notes_view == "editor"
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Reload never completed.")

        assert not screen.query("#library-note-conflict-reload")
        assert screen.query_one("#library-note-body", TextArea).text == "Server-side content"
        assert screen.query_one("#library-note-title", Input).value == "Server-side title"


@pytest.mark.asyncio
async def test_library_shell_note_conflict_reload_falls_back_to_list_when_note_missing():
    """A Reload from the conflict UI whose silent re-fetch discovers the
    note was deleted elsewhere entirely (not just version-bumped again)
    must fall back to the list view with a warning -- not strand the
    canvas on a permanent "Loading note…" placeholder, mirroring the
    missing-note fallback ``_refresh_library_note_detail`` already applies.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-body", TextArea).text = "kept text"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        # The note is now gone entirely (not just bumped again) before
        # Reload's silent re-fetch runs.
        _remove_note_externally(service, "n-1")

        (await _wait_for_selector(screen, pilot, "#library-note-conflict-reload")).press()
        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Reload never fell back to the list view for a missing note "
                f"(stuck: view={screen._library_notes_view!r}, "
                f"autosave_state={screen._library_note_autosave_state!r})."
            )

        assert screen._selected_note_id == ""
        assert screen._library_note_detail is None
        assert screen._library_note_autosave_state == "idle"
        assert not screen.query("#library-note-conflict-reload")


@pytest.mark.asyncio
async def test_library_shell_note_conflict_overwrite_falls_back_to_list_when_note_missing():
    """Same fallback as the Reload case above, but for Overwrite: it also
    runs the same silent re-fetch first, so a note deleted elsewhere must
    not leave Overwrite stuck silently no-op-ing against a conflict UI that
    can never be resolved.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-body", TextArea).text = "kept text"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        _remove_note_externally(service, "n-1")

        (await _wait_for_selector(screen, pilot, "#library-note-conflict-overwrite")).press()
        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Overwrite never fell back to the list view for a missing note "
                f"(stuck: view={screen._library_notes_view!r}, "
                f"autosave_state={screen._library_note_autosave_state!r})."
            )

        assert screen._selected_note_id == ""
        assert screen._library_note_detail is None
        assert screen._library_note_autosave_state == "idle"
        assert not screen.query("#library-note-conflict-overwrite")


@pytest.mark.asyncio
async def test_library_shell_note_delete_shows_inline_confirm_without_deleting():
    """Pressing Delete shows the inline confirm affordance, not an immediate delete."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm-copy")

        assert screen._library_note_confirming_delete is True
        assert screen.query_one("#library-note-delete-confirm")
        assert screen.query_one("#library-note-delete-cancel")
        assert not screen.query("#library-note-save")
        assert not screen.query("#library-note-delete")

        service = app.notes_scope_service
        assert service.delete_calls == []
        assert screen._library_notes_view == "editor"


@pytest.mark.asyncio
async def test_library_shell_note_delete_confirm_removes_note_and_returns_to_list():
    """Confirming the delete calls the seam with the current version, drops
    the note from the list view, and resets the editor state."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")

        screen.query_one("#library-note-delete-confirm").press()

        service = app.notes_scope_service
        for _ in range(150):
            if service.delete_calls and screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Delete never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert service.delete_calls, "delete_note was never called"
        assert service.delete_calls[-1]["scope"] == "local_note"
        assert service.delete_calls[-1]["note_id"] == "n-1"
        assert service.delete_calls[-1]["version"] == 2
        assert screen._library_note_confirming_delete is False
        assert screen._selected_note_id == ""
        assert screen._library_notes_view == "list"
        assert not screen.query("#library-note-title")
        assert not any(
            "Q3 retro" in str(getattr(button, "label", ""))
            for button in screen.query(".library-notes-row")
        )


@pytest.mark.asyncio
async def test_library_shell_note_delete_cancel_leaves_note_intact():
    """Cancelling the inline confirm discards it without calling the service."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")

        screen.query_one("#library-note-delete-cancel").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_note_confirming_delete is False
        assert not screen.query("#library-note-delete-confirm")
        assert not screen.query("#library-note-delete-confirm-copy")
        assert screen.query_one("#library-note-delete")
        assert screen.query_one("#library-note-title", Input).value == "Q3 retro"

        service = app.notes_scope_service
        assert service.delete_calls == []


@pytest.mark.asyncio
async def test_library_shell_note_delete_confirm_does_not_arm_autosave(monkeypatch):
    """Entering (and leaving) the delete-confirmation state recomposes the
    editor, which remounts the Input/TextArea and re-fires their mount-time
    ``Changed`` events. Those must never be mistaken for a real edit and
    fire a background autosave while a destructive-action confirmation is
    on screen -- the same armed-flag discipline every other note-editor
    recompose site already follows (see ``_refresh_library_note_detail``,
    ``_save_library_note``'s conflict branch, and
    ``_resolve_library_note_conflict``).
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.05)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        calls_before_delete = len(service.save_calls)

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm-copy")

        # Wait well past the (monkeypatched) debounce window -- long enough
        # for a spurious autosave timer to have fired if the armed flag
        # were left set across the recompose.
        for _ in range(150):
            await pilot.pause(0.02)

        assert len(service.save_calls) == calls_before_delete, (
            "Entering delete-confirm must not trigger an autosave."
        )
        assert screen._library_note_confirming_delete is True
        assert screen.query_one("#library-note-delete-confirm")
        assert screen.query_one("#library-note-delete-cancel")

        screen.query_one("#library-note-delete-cancel").press()
        await pilot.pause()

        for _ in range(150):
            await pilot.pause(0.02)

        assert len(service.save_calls) == calls_before_delete, (
            "Cancelling delete-confirm must not trigger an autosave either."
        )
        assert screen._library_note_confirming_delete is False
        assert not screen.query("#library-note-delete-confirm")
        assert screen.query_one("#library-note-delete")


@pytest.mark.asyncio
async def test_library_shell_note_delete_stale_version_rearms_the_editor():
    """Task 7 regression rider: the delete confirm's stale-version failure
    path must re-arm dirty-tracking around its recompose, the same
    ``_library_note_editor_armed = False`` +
    ``call_after_refresh(_arm_library_note_editor)`` dance every other
    notes-editor recompose already uses (before this rider fix, this one
    call site skipped it). Proven deterministically via a spy on
    ``_arm_library_note_editor`` rather than by observing autosave timing:
    a worker-triggered recompose racing its own mount-time
    ``Input``/``TextArea`` ``Changed`` events against the re-arm is a
    pre-existing characteristic of this mechanism (independently
    reproducible today on the already-shipped save-conflict recompose in
    ``_save_library_note``), so asserting "no autosave ever fires" around
    the recompose itself would be asserting a guarantee this codebase
    doesn't actually make -- not something introduced or fixable within
    this rider's narrow scope (adding the same dance already used
    elsewhere to 3 more call sites).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        # Another writer deletes/updates n-1 out from under the open editor,
        # bumping its stored version so the confirmed delete below is stale.
        _bump_note_version_externally(service, "n-1")

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")
        await pilot.pause()

        # Spy on the re-arm hook from here on, so only calls triggered by
        # the confirmed (stale) delete below count -- entering confirm mode
        # already re-arms once on its own (a separate, already-covered path).
        rearm_calls: list[bool] = []
        original_arm = screen._arm_library_note_editor

        def _spy_arm() -> None:
            rearm_calls.append(True)
            original_arm()

        screen._arm_library_note_editor = _spy_arm

        screen.query_one("#library-note-delete-confirm").press()

        for _ in range(150):
            if service.delete_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Delete confirm never called the delete_note seam.")

        # The delete lost the optimistic-lock race (falsy result): the note
        # must still be open in the editor, not removed.
        for _ in range(150):
            if not screen._library_note_confirming_delete:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Stale-version delete never returned to the normal action row.")
        await pilot.pause()

        assert screen._library_notes_view == "editor"
        assert screen._selected_note_id == "n-1"
        assert screen.query_one("#library-note-delete")
        assert rearm_calls, (
            "The stale-version delete failure path never re-armed the "
            "editor's dirty-tracking after its recompose."
        )
        assert screen._library_note_editor_armed is True


@pytest.mark.asyncio
async def test_library_shell_filtered_delete_refreshes_list_without_ghost():
    """Deleting a note opened from a filtered list must clear the active
    filter and refresh from the full snapshot, so the deleted note does
    not linger as a ghost row in the (now-stale) filtered results
    (final-review finding, part A).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-filter")
        box = screen.query_one("#library-notes-filter", Input)
        box.value = "retro"
        box.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")

        # The filter narrows the list to just "Q3 retro" (n-1).
        assert not any(
            "Reading list" in str(getattr(button, "label", ""))
            for button in screen.query(".library-notes-row")
        )
        assert screen._library_notes_filter == "retro"

        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        assert screen._selected_note_id == "n-1"

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")
        screen.query_one("#library-note-delete-confirm").press()

        service = app.notes_scope_service
        for _ in range(150):
            if service.delete_calls and screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Delete never completed. Visible text: {_visible_text(screen)}"
            )

        rows_text: list[str] = []
        for _ in range(150):
            rows_text = [
                str(getattr(button, "label", ""))
                for button in screen.query(".library-notes-row")
            ]
            if any("Reading list" in text for text in rows_text) and not any(
                "Q3 retro" in text for text in rows_text
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"List never settled on the refreshed (unfiltered) snapshot: {rows_text}"
            )

        assert screen._library_notes_filter == ""
        assert screen._library_notes_filter_records is None
        filter_box = screen.query_one("#library-notes-filter", Input)
        assert filter_box.value == ""
        assert not any("Q3 retro" in text for text in rows_text), (
            f"Deleted note still rendered as a ghost row: {rows_text}"
        )
        assert any("Reading list" in text for text in rows_text), (
            f"Remaining note missing from the refreshed list: {rows_text}"
        )


@pytest.mark.asyncio
async def test_library_shell_opening_missing_note_falls_back_to_list():
    """Pressing a row whose note has since vanished (a ghost row, or a note
    deleted elsewhere) must fall back to the list view instead of stranding
    the canvas on the "Loading note…" placeholder forever (final-review
    finding, part B).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        row_button = screen.query_one("#library-notes-row-0")
        assert row_button.note_id == "n-1"

        # Simulate the note vanishing out from under the still-rendered row
        # (deleted elsewhere, or simply a stale/ghost row): the fake's
        # ``get_note_detail`` now resolves to ``None`` for "n-1".
        service = app.notes_scope_service
        service.notes = tuple(
            note for note in service.notes if note.get("id") != "n-1"
        )

        row_button.press()

        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Never fell back to the list. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert not screen.query("#library-note-loading"), (
            "Stuck on the permanent 'Loading note…' placeholder."
        )
        assert not screen.query("#library-note-title")
        assert screen._library_note_detail is None
        assert screen._selected_note_id == ""
        app.notify.assert_called_once()
        assert app.notify.call_args.kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_shell_note_preview_toggle_shows_markdown_and_restores_edit():
    """Preview swaps the body TextArea for a read-only Markdown widget
    rendering the *current* (unsaved) text; toggling back to Edit restores
    the TextArea with that same edit intact.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        body = screen.query_one("#library-note-body", TextArea)
        body.text = "alpha budget line, unsaved preview edit"
        await pilot.pause()

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-preview-body")

        assert not screen.query("#library-note-body")
        preview = screen.query_one("#library-note-preview-body", Markdown)
        assert "alpha budget line, unsaved preview edit" in preview.source
        assert str(screen.query_one("#library-note-preview").label) == "Edit"
        # Title/keywords stay live Inputs -- only the body swaps.
        assert screen.query_one("#library-note-title", Input).value == "Q3 retro"

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-body")

        assert not screen.query("#library-note-preview-body")
        restored_body = screen.query_one("#library-note-body", TextArea)
        assert restored_body.text == "alpha budget line, unsaved preview edit"
        assert str(screen.query_one("#library-note-preview").label) == "Preview"


@pytest.mark.asyncio
async def test_library_shell_note_save_works_while_previewing():
    """Pressing Save while Preview is on must still persist the edit.

    While Preview is showing, ``#library-note-body`` is a read-only
    ``Markdown`` widget -- not the ``TextArea`` -- so a naive
    ``query_one("#library-note-body", TextArea)`` raises ``NoMatches``.
    ``_save_library_note`` must instead read through the preview-aware
    ``_read_library_note_editor_fields`` helper (the same one Export/Copy/
    Use-in-Console already rely on) so Save is not a silent no-op while
    previewing.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = (
            "alpha budget line, edited while about to preview"
        )
        await pilot.pause()

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-preview-body")
        assert not screen.query("#library-note-body")

        screen.query_one("#library-note-save").press()

        service = app.notes_scope_service
        for _ in range(150):
            if service.save_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Save never called the save_note seam while previewing."
            )

        call = service.save_calls[-1]
        assert call["content"] == "alpha budget line, edited while about to preview"

        for _ in range(150):
            if not screen._library_note_dirty:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Save while previewing never cleared the dirty flag.")


@pytest.mark.asyncio
async def test_library_shell_note_back_flushes_while_previewing():
    """Back must flush a pending edit even while Preview is still on.

    Regression pilot for the "silent data loss" finding: edit the body,
    toggle Preview on (without toggling back to Edit), then press Back
    immediately -- the edit must reach the seam before the view switches
    to the notes list, mirroring the flush-on-Back contract already
    proven for the non-preview case.
    """
    app = _build_test_app()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = (
            "alpha budget line, edited then previewed then back"
        )
        await pilot.pause()

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-preview-body")
        assert not screen.query("#library-note-body")

        screen.query_one("#library-note-back").press()
        for _ in range(50):
            if service.save_started:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError(
                "Back never triggered the flush save while previewing."
            )

        # The save is still sleeping: the view must not have switched yet.
        assert screen._library_notes_view == "editor"

        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Back never completed once the previewing flush resolved."
            )

        assert service.save_calls, (
            "The flushed save while previewing never actually completed."
        )
        assert service.save_calls[-1]["content"] == (
            "alpha budget line, edited then previewed then back"
        )


@pytest.mark.asyncio
async def test_library_shell_note_export_markdown_pushes_file_save_dialog():
    """Export .md pushes a ``FileSave`` dialog pre-filled with a sanitized
    default filename derived from the note's current title."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-export-md").press()
        for _ in range(150):
            if isinstance(host.screen_stack[-1], FileSave):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export .md never pushed a FileSave dialog.")

        dialog = host.screen_stack[-1]
        assert dialog._default_file == "Q3 retro.md"

        await host.pop_screen()
        await pilot.pause()


@pytest.mark.asyncio
async def test_library_shell_note_write_export_file_writes_expected_content(tmp_path):
    """The export write-path (bypassing the dialog UI, which is exercised
    separately above) writes the same content ``build_note_export_content``
    produces and notifies on success -- the "pure part" the brief calls out
    as pilot-testable independent of driving the real file dialog.

    ``_write_library_note_export_file`` is a plain (non-async) method --
    it awaits nothing -- so this calls it directly rather than awaiting it.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        destination = tmp_path / "export.md"
        screen._write_library_note_export_file(
            destination, "markdown", "Q3 retro", "alpha budget line", "alpha, beta", "n-1"
        )

        written = destination.read_text(encoding="utf-8")
        assert "title: Q3 retro" in written
        assert "keywords: alpha, beta" in written
        assert "note_id: n-1" in written
        assert written.endswith("alpha budget line")
        app.notify.assert_called_once()
        assert "exported successfully" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_library_shell_note_write_export_file_rejects_invalid_path(tmp_path, monkeypatch):
    """A ``FileSave``-returned path that fails ``validate_path_simple``
    must be rejected with a quiet warning notice -- no write, no crash --
    rather than trusting the dialog's returned path unconditionally.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notify = Mock()
    host = LibraryHarness(app)

    def _reject_path(*_args, **_kwargs):
        raise ValueError("rejected for test")

    monkeypatch.setattr(library_screen_module, "validate_path_simple", _reject_path)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        destination = tmp_path / "export.md"
        screen._write_library_note_export_file(
            destination, "markdown", "Q3 retro", "alpha budget line", "alpha, beta", "n-1"
        )

        assert not destination.exists()
        app.notify.assert_called_once()
        args, kwargs = app.notify.call_args
        assert "Rejected export path" in args[0]
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_shell_note_copy_calls_clipboard_seam():
    """Copy calls the app's ``copy_to_clipboard`` seam with the markdown
    export shape and notifies on success."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.copy_to_clipboard = Mock()
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-copy").press()
        await pilot.pause()

    app.copy_to_clipboard.assert_called_once()
    copied = app.copy_to_clipboard.call_args.args[0]
    assert copied.startswith("---\n")
    assert "title: Q3 retro" in copied
    assert "alpha budget line" in copied
    app.notify.assert_called_once()
    assert "copied to clipboard" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_library_shell_note_use_in_console_triggers_handoff():
    """"Use in Console" stages the open note as Console context, mirroring
    the media viewer's "Use in Chat" handoff test.
    """
    app = _build_test_app()
    note_items = _two_notes()[:1]
    _seed_conversations(app, [], notes=note_items)
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("note", "n-1", "Q3 retro"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-use-in-console").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "notes"
    assert payload.item_type == "note"
    assert payload.source_id == "n-1"
    assert payload.title == "Q3 retro"
    assert "alpha budget line" in payload.body
    assert payload.metadata["note_version"] == 2


@pytest.mark.asyncio
async def test_library_shell_note_use_in_console_without_open_note_notifies():
    """No handoff fires when no note is currently open in the editor."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.open_chat_with_handoff = Mock()
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen._selected_note_id == ""
        screen._open_selected_library_note_handoff()
        await pilot.pause()

    app.open_chat_with_handoff.assert_not_called()
    app.notify.assert_called_once()
    message = app.notify.call_args.args[0]
    assert "Open a note" in message


def test_library_note_css_bounds_editor_body_and_mutes_meta():
    """Generated-CSS presence check (house pattern: see
    ``test_library_source_actions_use_console_text_control_style``) --
    the note editor's ``TextArea`` must never be left at Textual's default
    ``height: 1fr`` inside the scrolling Library canvas, and the meta line
    must use the same muted tone as the media viewer's.
    """
    source_css = Path("tldw_chatbook/css/components/_agentic_terminal.tcss").read_text(encoding="utf-8")
    bundled_css = Path("tldw_chatbook/css/tldw_cli_modular.tcss").read_text(encoding="utf-8")
    for css in (source_css, bundled_css):
        assert "#library-note-body {" in css
        body_block = css[css.index("#library-note-body {"):]
        body_block = body_block[: body_block.index("}")]
        assert "height: auto;" in body_block
        assert "min-height: 12;" in body_block
        assert "max-height: 20;" in body_block

        assert "#library-note-meta {" in css
        meta_block = css[css.index("#library-note-meta {"):]
        meta_block = meta_block[: meta_block.index("}")]
        assert "color: $ds-text-muted;" in meta_block


@pytest.mark.asyncio
async def test_library_shell_create_note_row_renders_blank_and_template_rows():
    """Selecting the rail's Create > New note row renders the Blank note
    action plus at least one template row, carrying a ``template_key``."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        assert screen.query_one("#library-notes-create-blank")
        template_row = screen.query_one("#library-notes-template-0")
        assert getattr(template_row, "template_key", None)


@pytest.mark.asyncio
async def test_library_shell_create_blank_note_lands_in_editor():
    """Pressing Blank note calls ``save_note`` with ``note_id=None`` and
    ``title="Untitled"``, then opens the in-canvas editor on the newly
    created note and refreshes the notes rail count."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        screen.query_one("#library-notes-create-blank").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        service = app.notes_scope_service
        assert service.save_calls, "Blank note create never called the seam."
        call = service.save_calls[-1]
        assert call["note_id"] is None
        assert call["title"] == "Untitled"
        assert call["content"] == ""
        assert call["keywords"] is None

        assert screen._library_notes_view == "editor"
        assert screen._library_selected_row_id == "browse-notes"
        created_note = next(n for n in service.notes if n["title"] == "Untitled")
        assert screen._selected_note_id == created_note["id"]
        assert screen.query_one("#library-note-title", Input).value == "Untitled"

        for _ in range(150):
            if screen._local_source_counts.get("notes") == 3:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The notes snapshot/rail count never refreshed after create.")
        rail_label = str(screen.query_one("#library-row-browse-notes").label)
        assert "(3)" in rail_label


@pytest.mark.asyncio
async def test_library_shell_create_from_template_uses_template_fields():
    """Pressing a template row creates a note using that template's title/
    content with ``{date}``/``{time}``/``{datetime}`` placeholders resolved
    to the current date/time -- mirroring the standalone Notes screen's
    ``_create_local_note_from_template`` substitution semantics."""
    from datetime import datetime

    from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        template_button = next(
            button
            for button in screen.query(".library-notes-template-row")
            if getattr(button, "template_key", None) == "meeting"
        )
        expected = NOTE_TEMPLATES["meeting"]
        # Capture the date both before and after pressing the row so a
        # midnight rollover mid-test can't make this flaky.
        before = datetime.now().strftime("%Y-%m-%d")
        template_button.press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        after = datetime.now().strftime("%Y-%m-%d")

        service = app.notes_scope_service
        assert service.save_calls, "Template create never called the seam."
        call = service.save_calls[-1]
        assert call["note_id"] is None
        assert "{date}" not in call["title"]
        assert "{date}" not in call["content"]
        assert "{time}" not in call["content"]
        assert call["title"] in (
            expected["title"].format(date=before),
            expected["title"].format(date=after),
        )
        assert re.search(r"\*\*Date:\*\* \d{4}-\d{2}-\d{2}", call["content"])
        assert re.search(r"\*\*Time:\*\* \d{2}:\d{2}", call["content"])
        # Template keywords ride along on create -- the standalone screen
        # applies them, so Library parity requires them at the seam too.
        assert call["keywords"] == ["meeting", "notes"]
        assert screen._library_notes_view == "editor"


def _fake_import_dialog_result(screen, selected_path):
    """Monkeypatch ``screen.app.push_screen`` so the Import note dialog's
    callback fires immediately with ``selected_path`` instead of driving
    the real ``FileOpen`` file-browser UI (which needs keystrokes into a
    path input and is unrelated to what this handler contract tests: that
    a *resolved* dialog result reaches ``_import_library_note_from_path``).
    """
    calls = []

    def _fake_push_screen(dialog, callback=None):
        calls.append(dialog)
        if callback is not None:
            screen.run_worker(callback(selected_path))
        return None

    screen.app.push_screen = _fake_push_screen
    return calls


@pytest.mark.asyncio
async def test_library_shell_import_note_lands_in_editor(tmp_path):
    """Pressing Import note, with the dialog resolving to a real file, reads
    that file's title/content and creates a note through the same
    ``_create_library_note`` seam the Blank note/template rows use --
    landing in the editor with the snapshot/count refresh that seam already
    performs."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    note_file = tmp_path / "imported.json"
    note_file.write_text(
        json.dumps({"title": "Imported from disk", "content": "body from disk"}),
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import")

        push_calls = _fake_import_dialog_result(screen, note_file)

        screen.query_one("#library-notes-import").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        assert push_calls and isinstance(push_calls[0], FileOpen)

        service = app.notes_scope_service
        assert service.save_calls, "Import never called the create seam."
        call = service.save_calls[-1]
        assert call["note_id"] is None
        assert call["title"] == "Imported from disk"
        assert call["content"] == "body from disk"

        assert screen._library_notes_view == "editor"
        created_note = next(n for n in service.notes if n["title"] == "Imported from disk")
        assert screen._selected_note_id == created_note["id"]

        for _ in range(150):
            if screen._local_source_counts.get("notes") == 3:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The notes snapshot/rail count never refreshed after import.")


@pytest.mark.asyncio
async def test_library_shell_import_note_oversize_file_rejected(tmp_path, monkeypatch):
    """A file larger than ``LIBRARY_NOTE_CONTENT_MAX_CHARS`` is rejected with
    a quiet warning notice -- no note is created."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notify = Mock()
    host = LibraryHarness(app)

    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTE_CONTENT_MAX_CHARS", 10)
    note_file = tmp_path / "too_big.txt"
    note_file.write_text("x" * 50, encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import")

        _fake_import_dialog_result(screen, note_file)

        service = app.notes_scope_service
        screen.query_one("#library-notes-import").press()
        for _ in range(60):
            await pilot.pause(0.02)

        assert not service.save_calls, "Oversize import must not create a note."
        assert screen._library_notes_view != "editor"
        app.notify.assert_called()
        args, kwargs = app.notify.call_args
        assert "import" in args[0].lower()
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_shell_import_note_huge_file_rejected_without_reading(tmp_path, monkeypatch):
    """A file whose on-disk SIZE already proves it exceeds the char cap
    (UTF-8 chars are at most 4 bytes, so ``st_size > 4 * cap`` guarantees
    over-cap) must be rejected before ``read_text`` is ever called -- the
    char-level check alone would first slurp an arbitrarily large file into
    memory (the PR reviewer's OOM finding).
    """
    import pathlib

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notify = Mock()
    host = LibraryHarness(app)

    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTE_CONTENT_MAX_CHARS", 10)
    note_file = tmp_path / "way_too_big.txt"
    note_file.write_text("x" * 50, encoding="utf-8")  # 50 bytes > 4 * 10

    read_calls: list[pathlib.Path] = []
    real_read_text = pathlib.Path.read_text

    def _recording_read_text(self, *args, **kwargs):
        read_calls.append(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", _recording_read_text)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import")

        _fake_import_dialog_result(screen, note_file)

        service = app.notes_scope_service
        screen.query_one("#library-notes-import").press()
        for _ in range(60):
            await pilot.pause(0.02)

        assert not any(p.name == "way_too_big.txt" for p in read_calls), (
            "The oversized file must be rejected by the pre-read size guard, "
            f"not read into memory first (read_text calls: {read_calls!r})."
        )
        assert not service.save_calls, "Huge import must not create a note."
        assert screen._library_notes_view != "editor"
        app.notify.assert_called()
        args, kwargs = app.notify.call_args
        assert "import" in args[0].lower()
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_shell_import_note_md_without_title_uses_filename_stem(tmp_path):
    """A ``.md`` file (which never carries a JSON/YAML "title" key in this
    parser's contract) falls back to the filename stem as the note title."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    note_file = tmp_path / "My Great Notes.md"
    note_file.write_text("# Just a heading\n\nSome body text.", encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import")

        _fake_import_dialog_result(screen, note_file)

        screen.query_one("#library-notes-import").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        service = app.notes_scope_service
        call = service.save_calls[-1]
        assert call["title"] == "My Great Notes"
        assert call["content"] == "# Just a heading\n\nSome body text."


@pytest.mark.asyncio
async def test_library_shell_import_note_cancelled_dialog_is_noop():
    """Cancelling the ``FileOpen`` dialog (callback fired with ``None``)
    creates no note and does not crash."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import")

        _fake_import_dialog_result(screen, None)

        service = app.notes_scope_service
        screen.query_one("#library-notes-import").press()
        for _ in range(30):
            await pilot.pause(0.02)

        assert not service.save_calls
        assert screen._library_notes_view != "editor"


@pytest.mark.asyncio
async def test_library_shell_create_view_groups_templates_without_blank_duplicate():
    """The create view separates Blank note from a "From a template" group,
    excludes the redundant blank template row, and shows each template's
    resolved title as a secondary line."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        # Group label between the Blank action and the template rows.
        section = screen.query_one("#library-notes-template-section")
        assert str(section.renderable) == "From a template"

        template_rows = list(screen.query(".library-notes-template-row"))
        assert template_rows, "No template rows rendered."
        # The blank template row is excluded -- Blank note is the one
        # canonical empty path (it also had a different default title).
        assert all(
            getattr(button, "template_key", "") != "blank" for button in template_rows
        )
        assert "Empty note" not in _visible_text(screen)
        # Rows show the resolved title (date substituted) as a secondary line.
        meeting = next(
            button
            for button in template_rows
            if getattr(button, "template_key", None) == "meeting"
        )
        label = str(meeting.label)
        assert "\n" in label
        assert re.search(r"Meeting Notes - \d{4}-\d{2}-\d{2}", label)
        assert "{date}" not in label


def test_library_note_template_fields_malformed_placeholder_degrades_to_raw_text():
    """A template with an unknown ``{placeholder}`` or an unbalanced brace
    must not raise -- the create flow falls back to the raw (unsubstituted)
    text instead of crashing, mirroring the guarded ``.format()`` call the
    standalone Notes screen uses but degrading gracefully instead of
    aborting the create."""
    template = {
        "title": "Notes - {unknown_key}",
        "content": "Stray brace ahead: { oops",
    }

    title, content = LibraryScreen._library_note_template_fields(template)

    assert title == "Notes - {unknown_key}"
    assert content == "Stray brace ahead: { oops"


# ----- Notes sync panel --------------------------------------------------

class _RecordingSyncResults:
    def __init__(self):
        self.created_notes = []
        self.updated_notes = []
        self.created_files = []
        self.updated_files = []
        self.conflicts = []
        self.errors = []


class _RecordingNotesSyncService:
    """Records the exact args ``sync_folder`` was called with, and lets the
    test control the returned (session_id, results) tuple. The progress
    callback is fired from a real background thread (mirroring the real
    engine's worker-thread callback) and gated on a ``threading.Event`` so
    the test can observe the mid-run targeted status update before letting
    the run complete -- proving the update lands without a recompose.
    """

    instances = []

    def __init__(self, notes_service, db):
        self.notes_service = notes_service
        self.db = db
        self.calls = []
        self.progress_fired = threading.Event()
        self.release_event = threading.Event()
        _RecordingNotesSyncService.instances.append(self)

    async def sync_folder(self, *, root_folder, user_id, direction,
                           conflict_resolution, progress_callback=None, extensions=None):
        self.calls.append({
            "root_folder": root_folder,
            "user_id": user_id,
            "direction": direction,
            "conflict_resolution": conflict_resolution,
        })
        if progress_callback is not None:
            from tldw_chatbook.Notes.sync_engine import SyncProgress
            progress_callback(SyncProgress(total_files=2, processed_files=1))
            self.progress_fired.set()
        # ``_run_library_service_call(..., isolate_in_worker=True)`` already
        # runs this coroutine on a worker thread with no event loop of its
        # own (see library_screen.py); blocking here on a plain
        # threading.Event (not asyncio.sleep) is what actually holds up
        # that worker thread until the test lets it proceed.
        await asyncio.to_thread(self.release_event.wait, _GATED_RELEASE_TIMEOUT_SECONDS)
        results = _RecordingSyncResults()
        results.created_notes = ["n-new"]
        return ("session-1", results)


def _prepare_library_notes_sync_app(app, *, notes=None):
    _seed_conversations(app, _two_conversations(), notes=notes or _two_notes())
    app.notes_service = Mock()
    app.chachanotes_db = Mock()
    # The sync direction/conflict/auto-sync config keys live in the real,
    # session-shared CLI config file (under the isolated test HOME) -- a
    # previous test in this same session may have persisted a non-default
    # value there. Reset to known starting values so every sync test's
    # cycling assertions are deterministic regardless of run order.
    from tldw_chatbook.config import save_setting_to_cli_config

    save_setting_to_cli_config("notes", "sync_direction", "bidirectional")
    save_setting_to_cli_config("notes", "sync_conflict_resolution", "newer_wins")
    save_setting_to_cli_config("notes", "auto_sync", False)


async def _open_library_notes_sync_panel(screen, pilot):
    screen.query_one("#library-row-browse-notes").press()
    await _wait_for_selector(screen, pilot, "#library-notes-sync-open")
    screen.query_one("#library-notes-sync-open").press()
    await _wait_for_selector(screen, pilot, "#library-notes-sync-back")


@pytest.mark.asyncio
async def test_library_shell_notes_sync_button_opens_sync_mode():
    """Pressing Sync on the notes list header opens the sync panel with all
    of its widgets present, and Back returns to the notes list."""
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_view == "sync"
        for selector in (
            "#library-notes-sync-back",
            "#library-notes-sync-header",
            "#library-notes-sync-purpose",
            "#library-notes-sync-folder-label",
            "#library-notes-sync-folder",
            "#library-notes-sync-browse",
            "#library-notes-sync-direction",
            "#library-notes-sync-conflict",
            "#library-notes-sync-auto",
            "#library-notes-sync-run",
            "#library-notes-sync-status",
            "#library-notes-sync-activity",
        ):
            assert screen.query_one(selector), f"{selector} missing from sync panel"

        # C4: auto-sync cadence is spelled out in the toggle's own label.
        auto_toggle = screen.query_one("#library-notes-sync-auto", Button)
        assert "every 5m" in str(auto_toggle.label)

        # A3: Sync now starts out enabled with the idle "Sync now" label.
        run_button = screen.query_one("#library-notes-sync-run", Button)
        assert "Sync now" in str(run_button.label)
        assert not run_button.disabled

        screen.query_one("#library-notes-sync-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-sync-open")
        assert screen._library_notes_view == "list"
        assert not screen.query("#library-notes-sync-back")


@pytest.mark.asyncio
async def test_library_shell_notes_sync_direction_cycles_and_persists():
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_sync_direction == "bidirectional"
        button = screen.query_one("#library-notes-sync-direction", Button)
        assert "Bidirectional" in str(button.label)

        button.press()
        await pilot.pause()
        assert screen._library_notes_sync_direction == "disk_to_db"
        from tldw_chatbook.config import get_cli_setting
        assert get_cli_setting("notes", "sync_direction", None) == "disk_to_db"
        button = screen.query_one("#library-notes-sync-direction", Button)
        assert "Disk" in str(button.label)


@pytest.mark.asyncio
async def test_library_shell_notes_sync_conflict_cycles_and_persists():
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_sync_conflict == "newer_wins"
        screen.query_one("#library-notes-sync-conflict").press()
        await pilot.pause()
        assert screen._library_notes_sync_conflict == "disk_wins"
        from tldw_chatbook.config import get_cli_setting
        assert get_cli_setting("notes", "sync_conflict_resolution", None) == "disk_wins"

        # A1: the cycle is 3 modes long with "ask" removed entirely -- one
        # more press reaches "Library wins" (B2's de-jargoned "db_wins"
        # label), and a third press wraps back to "newer_wins" rather than
        # ever landing on "ask". Each press recomposes the panel (see
        # handle_library_notes_sync_conflict), so the Button must be
        # re-queried after every press rather than reusing a stale
        # reference to the pre-recompose instance.
        screen.query_one("#library-notes-sync-conflict").press()
        await pilot.pause()
        assert screen._library_notes_sync_conflict == "db_wins"
        button = screen.query_one("#library-notes-sync-conflict", Button)
        assert "Library wins" in str(button.label)

        screen.query_one("#library-notes-sync-conflict").press()
        await pilot.pause()
        assert screen._library_notes_sync_conflict == "newer_wins"


@pytest.mark.asyncio
async def test_library_shell_notes_sync_auto_toggle_flips_and_persists():
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_sync_auto is False
        toggle = screen.query_one("#library-notes-sync-auto", Button)
        # C4: the cadence is spelled out in the label itself, not just left
        # implicit behind a bare toggle glyph.
        assert "auto-sync: every 5m ○" in str(toggle.label)

        toggle.press()
        await pilot.pause()
        assert screen._library_notes_sync_auto is True
        assert screen._library_notes_auto_sync_timer is not None
        from tldw_chatbook.config import get_cli_setting
        assert get_cli_setting("notes", "auto_sync", None) is True
        toggle = screen.query_one("#library-notes-sync-auto", Button)
        assert "auto-sync: every 5m ✓" in str(toggle.label)

        toggle.press()
        await pilot.pause()
        assert screen._library_notes_sync_auto is False
        assert screen._library_notes_auto_sync_timer is None
        assert get_cli_setting("notes", "auto_sync", None) is False


@pytest.mark.asyncio
async def test_library_shell_notes_sync_folder_typing_does_not_write_config(tmp_path):
    """Typing in the sync folder box must NOT rewrite the config file per
    keystroke (a full TOML write + cache reload per character -- the PR
    reviewer's IO-thrash finding). The typed value lives in screen state
    (surviving the panel's recomposes) and persists only on explicit
    commit: Enter in the box, or a validated Sync now run.
    """
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config

    save_setting_to_cli_config("notes", "sync_directory", "~/Documents/Notes")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        folder_input = screen.query_one("#library-notes-sync-folder", Input)
        folder_input.value = str(tmp_path)
        folder_input.focus()
        await pilot.pause()

        # Typed but not committed: config untouched...
        assert get_cli_setting("notes", "sync_directory", None) == "~/Documents/Notes"
        # ...but the typed value survives a recompose (cycling any setting
        # rebuilds the canvas) instead of snapping back to the config value.
        screen.query_one("#library-notes-sync-direction").press()
        await pilot.pause()
        folder_input = screen.query_one("#library-notes-sync-folder", Input)
        assert folder_input.value == str(tmp_path)
        assert get_cli_setting("notes", "sync_directory", None) == "~/Documents/Notes"

        # Enter commits the typed folder to config.
        folder_input.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert get_cli_setting("notes", "sync_directory", None) == str(tmp_path)


@pytest.mark.asyncio
async def test_library_shell_notes_sync_run_persists_validated_folder(monkeypatch, tmp_path):
    """A validated Sync now run commits the (typed-but-unsubmitted) folder to
    config -- the other explicit commit point besides Enter/Browse -- so the
    folder a run actually used is always the one that persists.
    """
    from tldw_chatbook.Notes import sync_service as sync_service_module

    _RecordingNotesSyncService.instances.clear()
    monkeypatch.setattr(sync_service_module, "NotesSyncService", _RecordingNotesSyncService)

    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config

    save_setting_to_cli_config("notes", "sync_directory", "~/Documents/Notes")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        folder_input = screen.query_one("#library-notes-sync-folder", Input)
        folder_input.value = str(tmp_path)
        folder_input.focus()
        await pilot.pause()
        assert get_cli_setting("notes", "sync_directory", None) == "~/Documents/Notes"

        screen.query_one("#library-notes-sync-run").press()
        for _ in range(150):
            if _RecordingNotesSyncService.instances and _RecordingNotesSyncService.instances[0].calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Sync-now never reached the recording service.")

        # The run validated the typed folder and committed it to config.
        assert get_cli_setting("notes", "sync_directory", None) == str(tmp_path)

        # Release the gated fake so the run (and teardown) can finish.
        service = _RecordingNotesSyncService.instances[0]
        service.release_event.set()
        for _ in range(150):
            if not screen._library_notes_sync_running:
                break
            await pilot.pause(0.02)


@pytest.mark.asyncio
async def test_library_shell_notes_sync_now_calls_recording_service_with_chosen_enums(
    monkeypatch, tmp_path
):
    """Sync-now with a valid folder must call the sync seam with the chosen
    direction/conflict enums, and update status/activity without ever
    recomposing (same Static widget instance) mid-run."""
    from tldw_chatbook.Notes.sync_engine import ConflictResolution, SyncDirection
    from tldw_chatbook.Notes import sync_service as sync_service_module
    from tldw_chatbook.Library.library_notes_sync_state import (
        sync_conflict_label,
        sync_direction_label,
    )

    _RecordingNotesSyncService.instances.clear()
    monkeypatch.setattr(sync_service_module, "NotesSyncService", _RecordingNotesSyncService)

    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        folder_input = screen.query_one("#library-notes-sync-folder", Input)
        folder_input.value = str(tmp_path)
        folder_input.focus()
        await pilot.pause()

        # Cycle direction/conflict once each so the recorded call proves the
        # *chosen* enums (not just the defaults) are threaded through. Each
        # press triggers a full canvas recompose (`refresh(recompose=True)`)
        # that re-mounts these two toggle buttons -- poll for the
        # re-mounted button to actually show the newly-chosen enum's label
        # before the next press, instead of a fixed pause, so the second
        # press can't land mid-recompose and get lost/target a stale
        # instance.
        expected_direction_label = (
            f"direction: {sync_direction_label(SyncDirection.DISK_TO_DB.value)} ▸"
        )
        screen.query_one("#library-notes-sync-direction").press()
        for _ in range(150):
            direction_buttons = screen.query("#library-notes-sync-direction")
            if direction_buttons and str(direction_buttons.first().label) == (
                expected_direction_label
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Direction toggle never re-mounted with the cycled label "
                f"(wanted {expected_direction_label!r})."
            )

        expected_conflict_label = (
            f"conflicts: {sync_conflict_label(ConflictResolution.DISK_WINS.value)} ▸"
        )
        screen.query_one("#library-notes-sync-conflict").press()
        for _ in range(150):
            conflict_buttons = screen.query("#library-notes-sync-conflict")
            if conflict_buttons and str(conflict_buttons.first().label) == (
                expected_conflict_label
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Conflict toggle never re-mounted with the cycled label "
                f"(wanted {expected_conflict_label!r})."
            )

        screen.query_one("#library-notes-sync-run").press()
        for _ in range(150):
            if screen._library_notes_sync_running:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Sync-now never started.")

        # A3: the run handler flips the running flag then recomposes; the
        # flag is set synchronously at the top of the worker coroutine, so
        # poll until the start-of-run recompose has landed the disabled
        # "Syncing…" button (querying the instant the flag flips can race
        # the mid-recompose teardown -> NoMatches).
        for _ in range(150):
            run_buttons = screen.query("#library-notes-sync-run")
            if run_buttons and run_buttons.first().disabled:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Sync-now never rendered the disabled Syncing… state.")
        run_button_mid_run = screen.query_one("#library-notes-sync-run", Button)
        assert "Syncing…" in str(run_button_mid_run.label)
        assert run_button_mid_run.disabled

        # Captured only once the start-of-sync recompose has already
        # happened, so this is the one-and-only Static instance the
        # progress callback's targeted update (and the finish handler)
        # must reuse -- proving no recompose happens mid-run.
        status_widget_before = screen.query_one("#library-notes-sync-status", Static)

        for _ in range(150):
            if _RecordingNotesSyncService.instances and _RecordingNotesSyncService.instances[0].calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Sync-now never reached the recording service.")

        service = _RecordingNotesSyncService.instances[0]
        call = service.calls[0]
        assert call["root_folder"] == tmp_path
        assert call["direction"] == SyncDirection.DISK_TO_DB
        assert call["conflict_resolution"] == ConflictResolution.DISK_WINS

        for _ in range(150):
            if service.progress_fired.is_set():
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Progress callback never fired.")
        for _ in range(150):
            if "syncing" in str(
                screen.query_one("#library-notes-sync-status", Static).renderable
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Progress callback never landed on the status widget.")

        # The mid-run progress update must have landed on the very same
        # Static instance captured before the callback fired -- no
        # recompose happened in between.
        status_widget_mid_run = screen.query_one("#library-notes-sync-status", Static)
        assert status_widget_before is status_widget_mid_run
        assert "1/2" in str(status_widget_mid_run.renderable)

        # Now let the gated fake service return so the run completes.
        service.release_event.set()
        for _ in range(150):
            if not screen._library_notes_sync_running:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Sync run never completed.")

        status_widget_after = screen.query_one("#library-notes-sync-status", Static)
        assert "done" in str(status_widget_after.renderable)

        # A3: the finish-of-run recompose restores "Sync now", re-enabled.
        run_button_after = screen.query_one("#library-notes-sync-run", Button)
        assert "Sync now" in str(run_button_after.label)
        assert not run_button_after.disabled


@pytest.mark.asyncio
async def test_library_shell_notes_sync_invalid_folder_notifies_quietly_no_run(monkeypatch):
    from tldw_chatbook.Notes import sync_service as sync_service_module

    _RecordingNotesSyncService.instances.clear()
    monkeypatch.setattr(sync_service_module, "NotesSyncService", _RecordingNotesSyncService)

    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    notifications = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        folder_input = screen.query_one("#library-notes-sync-folder", Input)
        folder_input.value = "/definitely/does/not/exist/anywhere"
        folder_input.focus()
        await pilot.pause()

        screen.query_one("#library-notes-sync-run").press()
        await pilot.pause()
        await pilot.pause()

        assert not _RecordingNotesSyncService.instances or not _RecordingNotesSyncService.instances[0].calls
        assert notifications
        assert notifications[-1][1].get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_shell_notes_sync_rail_reentry_resets_transient_state():
    """Leaving the sync panel for another rail row and returning to Browse >
    Notes must clear stale status/activity from the previous visit."""
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        screen._library_notes_sync_status = "failed · disk full"
        screen._library_notes_sync_activity = ("some previous line",)

        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-sync-open")

        assert screen._library_notes_sync_status == "idle"
        assert screen._library_notes_sync_activity == ()
        assert screen._library_notes_view == "list"


@pytest.mark.asyncio
async def test_library_shell_notes_sync_stale_ask_conflict_coerces_to_newer_wins():
    """A1: an old config still holding the removed "ask" conflict value must
    coerce to "newer_wins" on panel entry -- "ask" can never render or reach
    the sync engine from this panel."""
    from tldw_chatbook.config import save_setting_to_cli_config

    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    # Seed a stale "ask" value the same way _prepare_library_notes_sync_app
    # seeds its known-good defaults, simulating a config written before
    # "ask" was removed from this panel's cycle.
    save_setting_to_cli_config("notes", "sync_conflict_resolution", "ask")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_sync_conflict == "newer_wins"
        button = screen.query_one("#library-notes-sync-conflict", Button)
        assert "Newer wins" in str(button.label)


@pytest.mark.asyncio
async def test_library_shell_notes_sync_stale_direction_coerces_to_bidirectional():
    """A1: an unrecognized persisted direction must coerce to
    "bidirectional" on panel entry, mirroring the conflict coercion."""
    from tldw_chatbook.config import save_setting_to_cli_config

    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    save_setting_to_cli_config("notes", "sync_direction", "some_removed_mode")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_sync_direction == "bidirectional"
        button = screen.query_one("#library-notes-sync-direction", Button)
        assert "Bidirectional" in str(button.label)


@pytest.mark.asyncio
async def test_library_shell_notes_sync_conflicts_get_honest_resolved_copy(monkeypatch, tmp_path):
    """A2/B1: the activity line for recorded conflicts must state the
    resolved policy (truthful: this panel never offers a review surface)
    and pluralize correctly."""
    from tldw_chatbook.Notes import sync_service as sync_service_module

    class _ConflictResults:
        def __init__(self):
            self.created_notes = []
            self.updated_notes = []
            self.created_files = []
            self.updated_files = []
            self.conflicts = ["c-1"]
            self.errors = []

    class _ConflictSyncService:
        def __init__(self, notes_service, db):
            pass

        async def sync_folder(self, *, root_folder, user_id, direction,
                               conflict_resolution, progress_callback=None, extensions=None):
            return ("session-conflict", _ConflictResults())

    monkeypatch.setattr(sync_service_module, "NotesSyncService", _ConflictSyncService)

    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        folder_input = screen.query_one("#library-notes-sync-folder", Input)
        folder_input.value = str(tmp_path)
        folder_input.focus()
        await pilot.pause()

        screen.query_one("#library-notes-sync-run").press()
        for _ in range(150):
            if not screen._library_notes_sync_running and screen._library_notes_sync_status != "idle":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Sync run never completed.")

        # Honest copy: names the resolved policy instead of promising a
        # "review" surface that doesn't exist in this panel.
        assert any(
            "1 conflict resolved (Newer wins)" in line
            for line in screen._library_notes_sync_activity
        ), screen._library_notes_sync_activity
        assert not any(
            "recorded for review" in line for line in screen._library_notes_sync_activity
        )


async def _run_library_search_and_wait_for_open_result(
    screen, pilot, query: str, *, index: int = 0
):
    """Run a Library Search/RAG query and wait for its Open result button."""
    screen.query_one("#library-row-browse-search").press()
    await _wait_for_selector(screen, pilot, "#library-rag-query-input")

    screen.query_one("#library-rag-query-input", Input).value = query
    await _wait_for_library_rag_query_ready(screen, pilot, query)
    screen.query_one("#library-rag-run-query", Button).press()
    await _wait_for_selector(screen, pilot, f"#library-rag-open-result-{index}")


@pytest.mark.asyncio
async def test_library_shell_search_result_open_note_lands_in_editor():
    """Pressing Open on a note evidence result jumps straight to that note's
    in-canvas editor, fetching its full detail by id -- not via the notes
    list/row-selection path.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    service = _StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "source_id": "n-1",
                    "title": "Q3 retro",
                    "snippet": "alpha budget line",
                    "provenance": {"source_type": "note"},
                }
            ]
        }
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _run_library_search_and_wait_for_open_result(screen, pilot, "retro")

        screen.query_one("#library-rag-open-result-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        for _ in range(120):
            if screen._selected_note_id == "n-1" and screen._library_notes_view == "editor":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Open never landed on the note editor.")
        await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        title = screen.query_one("#library-note-title", Input)
        assert title.value == "Q3 retro"
        assert any(
            call["note_id"] == "n-1" for call in app.notes_scope_service.detail_calls
        )


@pytest.mark.asyncio
async def test_library_shell_search_result_open_media_switches_to_viewer():
    """Pressing Open on a media evidence result flips the canvas to the
    in-canvas media viewer and fetches that item's detail by id.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    service = _StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "source_id": "media-1",
                    "title": "Interview Recording",
                    "snippet": "audio transcript",
                    "provenance": {"source_type": "media"},
                }
            ]
        }
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _run_library_search_and_wait_for_open_result(screen, pilot, "interview")

        screen.query_one("#library-rag-open-result-0").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")
        for _ in range(120):
            if screen._selected_media_id == "media-1" and screen._library_media_view == "viewer":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Open never landed on the media viewer.")
        await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording"
        assert any(
            call["media_id"] == "media-1"
            for call in app.media_reading_scope_service.detail_calls
        )


@pytest.mark.asyncio
async def test_library_shell_search_result_open_conversation_fetches_missing_id():
    """Pressing Open on a conversation evidence result whose id is NOT in the
    loaded snapshot fetches it directly from ChaChaNotes and selects it --
    closing the deep-link caveat where an unknown id silently fell back to
    the snapshot's first row.

    Uses a real in-memory ``CharactersRAGDB`` (not a fake) per the task
    brief: the fetch path checks ``is_memory_db`` and must call the DB
    directly rather than via ``asyncio.to_thread`` for in-memory connections.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    db = CharactersRAGDB(":memory:", client_id="test-client")
    off_snapshot_id = db.add_conversation({"title": "Off-snapshot chat"})
    app.chachanotes_db = db
    try:
        service = _StaticLibraryRagSearchService(
            {
                "results": [
                    {
                        "source_id": off_snapshot_id,
                        "title": "Off-snapshot chat",
                        "snippet": "not part of the loaded snapshot",
                        "provenance": {"source_type": "conversation"},
                    }
                ]
            }
        )
        app.library_rag_search_service = service
        host = LibraryHarness(app)

        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            assert off_snapshot_id not in {
                screen._conversation_record_id(record, index)
                for index, record in enumerate(screen._conversation_records())
            }
            await _run_library_search_and_wait_for_open_result(screen, pilot, "snapshot")

            screen.query_one("#library-rag-open-result-0").press()
            for _ in range(150):
                if screen._selected_conversation_id == off_snapshot_id:
                    break
                await pilot.pause(0.02)
            else:
                raise AssertionError("Open never fetched/selected the off-snapshot conversation.")
            await pilot.pause()
            await pilot.pause()

            assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
            preview = str(
                screen.query_one("#library-conversation-preview-lines").renderable
            )
            assert "Off-snapshot chat" in preview
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_library_shell_search_result_without_provenance_has_no_open_button():
    """A Search/RAG result lacking resolvable provenance (no known source
    type/id) renders no Open button -- ``LibraryRagResultRow.can_open`` is
    False, so the Open action must not appear (the row remains selectable
    for Console handoff via ``Select evidence``).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "title": "Unattributed result",
                    "snippet": "no provenance",
                }
            ]
        }
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        screen.query_one("#library-rag-query-input", Input).value = "unattributed"
        await _wait_for_library_rag_query_ready(screen, pilot, "unattributed")
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-select-result-0")

        assert not screen.query(".library-rag-result-open")
        assert not list(screen.query("#library-rag-open-result-0"))


# --- Ingest canvas (L3b Task 4) --------------------------------------------
#
# Unlike ``LibraryHarness`` (whose ``app_instance`` is a separate, never-run
# ``TldwCli``), submitting a real ingest job needs ``self.app_instance`` to
# literally be the running app: ``submit_library_ingest_job`` starts an
# ``@work``-decorated background worker and marshals every registry
# mutation back via ``call_from_thread`` -- neither works without a live
# message pump. ``_LibraryIngestCanvasHarness`` mixes the real
# ``LibraryIngestQueueMixin`` (app.py) straight into a minimal running
# ``App`` and pushes ``LibraryScreen(self)`` -- mirroring
# ``_IngestRunnerHarness`` (Tests/Library/test_library_ingest_runner.py)
# combined with ``LibraryHarness``'s screen-hosting shape. Notes/
# conversations use the existing static empty fakes purely so
# ``_list_local_source_snapshot`` doesn't bail into its lookup-error state
# (which would otherwise blank out the media viewer on Open in Library);
# only Media needs to be real here, wired to the SAME db the ingest
# queue-runner writes to.

_INGEST_POLL_ATTEMPTS = 500
_INGEST_POLL_INTERVAL = 0.02


class _LibraryIngestCanvasHarness(LibraryIngestQueueMixin, App):
    """Runs a real LibraryScreen against a running app mixing the real
    ingest coordinator + writer + registry + an optional real file-backed
    MediaDatabase.

    Defaults to an auto-run ``_FakeIngestParsePool`` (F3) -- real
    ``run_parse_job``/``persist_parsed_media``, fake pool -- so the ~20
    ingest pilots below never spawn real OS processes. Pass
    ``pool_factory``/``worker_count`` for pilots needing manual completion
    control or a specific backpressure cap (mirrors
    ``Tests/Library/test_library_ingest_runner.py``'s
    ``_IngestRunnerHarness``)."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self, media_db, *, pool_factory=None, worker_count=None):
        super().__init__()
        self.library_ingest_jobs = LibraryIngestJobRegistry()
        self.media_db = media_db
        self._ingest_parse_pool = None
        self._ingest_parsed_payloads = {}
        self._ingest_shutdown = False
        self._pool_factory = pool_factory or (lambda: _FakeIngestParsePool())
        self._worker_count_override = worker_count
        self.notes_scope_service = StaticLibraryNotesScopeService([])
        self.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
        if media_db is not None:
            self.media_reading_scope_service = MediaReadingScopeService(
                LocalMediaReadingService(media_db), None
            )

    def _create_ingest_parse_pool(self):
        return self._pool_factory()

    def _ingest_parse_worker_count(self) -> int:
        if self._worker_count_override is not None:
            return self._worker_count_override
        return super()._ingest_parse_worker_count()

    async def on_mount(self) -> None:
        await self.push_screen(LibraryScreen(self))


async def _open_library_ingest_canvas(screen, pilot):
    screen.query_one("#library-row-ingest-import-media").press()
    await _wait_for_selector(screen, pilot, "#library-ingest-path")


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_happy_path_open_in_library(tmp_path):
    """(a) Full happy path: a real tmp .txt through a real file-backed
    MediaDatabase -- type the path, Start, poll until the row reaches
    done, then Open in Library lands in the media viewer on the item that
    was actually ingested."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    source = tmp_path / "tides.txt"
    source.write_text("Tides are driven by the moon's gravity.", encoding="utf-8")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        await _open_library_ingest_canvas(screen, pilot)

        path_input = screen.query_one("#library-ingest-path", Input)
        path_input.value = str(source)
        await pilot.pause()

        start_button = screen.query_one("#library-ingest-start", Button)
        assert start_button.disabled is False
        start_button.press()
        await pilot.pause()

        # Path clears immediately on submit (metadata fields would persist).
        assert screen.query_one("#library-ingest-path", Input).value == ""

        # Task 4 has no live-update listener yet (that's Task 5) -- the
        # canvas only re-renders on a user-triggered recompose, so poll the
        # registry directly, then force one recompose once the job is done.
        for _ in range(_INGEST_POLL_ATTEMPTS):
            jobs = harness.library_ingest_jobs.jobs()
            if jobs and jobs[0].state == IngestJobState.DONE:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError(f"Job never reached DONE: {harness.library_ingest_jobs.jobs()}")

        screen.refresh(recompose=True)
        await pilot.pause()

        row_text = str(screen.query_one("#library-ingest-row-0").renderable)
        assert row_text.startswith("✓ done · tides.txt")
        # Row-action buttons are keyed by job_id, not row index (PR #591
        # review, F1) -- this is the first (only) job submitted against a
        # fresh registry, so its id is deterministically "ingest-job-1".
        assert screen.query_one("#library-ingest-open-ingest-job-1")

        screen.query_one("#library-ingest-open-ingest-job-1").press()
        for _ in range(_INGEST_POLL_ATTEMPTS):
            if screen._library_media_detail is not None:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError("Media detail never loaded after Open in Library.")

        assert screen._library_media_view == "viewer"
        viewer_title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert "tides" in viewer_title.lower()


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_invalid_path_notifies_and_submits_nothing(tmp_path):
    """(b) An invalid path (rejected by validate_path_simple) is a quiet
    warning notice -- no job is ever submitted."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    harness = _LibraryIngestCanvasHarness(db)
    harness.notify = Mock()

    def _reject_path(*_args, **_kwargs):
        raise ValueError("rejected for test")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(library_screen_module, "validate_path_simple", _reject_path)

        async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = harness.screen_stack[-1]
            await _wait_for_library_shell(screen, pilot)
            await _open_library_ingest_canvas(screen, pilot)

            screen.query_one("#library-ingest-path", Input).value = "/tmp/whatever.txt"
            await pilot.pause()
            screen.query_one("#library-ingest-start", Button).press()
            await pilot.pause()

            harness.notify.assert_called_once()
            args, kwargs = harness.notify.call_args
            assert kwargs.get("severity") == "warning"
            assert harness.library_ingest_jobs.jobs() == ()


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_failed_job_renders_retry_and_requeues(tmp_path):
    """(c) A failed (non-permanent) job renders a failed row with Retry;
    pressing Retry appends a fresh queued job and (L3b AB wave, B1)
    supersedes the original -- the queue shows exactly ONE row for the
    retried file, not two.

    (M4 re-anchor, fix batch F1b) A *missing-file* failure is now
    classified permanent and withholds Retry entirely by design (see
    ``test_library_shell_ingest_canvas_permanent_failure_has_no_retry_button``
    below), so this test drives a non-permanent failure directly through
    the registry instead -- keeping its original focus on the canvas's
    Retry-button-press -> requeue -> single-row-collapse flow, independent
    of the runner's own failure classification (covered separately in
    ``Tests/Library/test_library_ingest_runner.py``).
    """
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    source = tmp_path / "flaky.txt"
    source.write_text("Retried content.", encoding="utf-8")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)

        failing_job = harness.library_ingest_jobs.submit(source_path=str(source))
        harness.library_ingest_jobs.mark_parsing(failing_job.job_id)
        harness.library_ingest_jobs.mark_writing(failing_job.job_id)
        harness.library_ingest_jobs.mark_failed(
            failing_job.job_id, error="boom", permanent=False
        )

        await _open_library_ingest_canvas(screen, pilot)
        # Row-action buttons are keyed by job_id, not row index (PR #591
        # review, F1) -- this is the only job in a fresh registry, so its
        # id is deterministically "ingest-job-1".
        await _wait_for_selector(screen, pilot, "#library-ingest-retry-ingest-job-1")
        row_text = str(screen.query_one("#library-ingest-row-0").renderable)
        assert row_text.startswith("✗ failed · flaky.txt")

        jobs_before = len(harness.library_ingest_jobs.jobs())
        screen.query_one("#library-ingest-retry-ingest-job-1", Button).press()
        await pilot.pause()

        jobs_after = harness.library_ingest_jobs.jobs()
        # B1: the original failed job is superseded, not kept alongside the
        # fresh copy -- net job count is unchanged (one hidden, one added).
        assert len(jobs_after) == jobs_before
        newest = jobs_after[0]
        assert newest.job_id != failing_job.job_id
        assert newest.source_path == str(source)

        # The canvas itself must show exactly one row -- not the retried
        # QUEUED copy sitting alongside a still-visible failed original.
        await pilot.pause()
        assert len(list(screen.query(".library-ingest-row"))) == 1


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_permanent_failure_has_no_retry_button(tmp_path):
    """(M4, fix batch F1b; F3 re-anchor) A missing-file failure is
    classified permanent inside the real parse worker (``run_parse_job``,
    driven here by the fake-pool seam) -- the canvas withholds Retry
    entirely (but still offers Dismiss)."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    missing = tmp_path / "does-not-exist.txt"
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)

        failing_job = harness.submit_library_ingest_job(source_path=str(missing))
        for _ in range(_INGEST_POLL_ATTEMPTS):
            jobs = {j.job_id: j for j in harness.library_ingest_jobs.jobs()}
            if jobs[failing_job.job_id].state == IngestJobState.FAILED:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError("Job never reached FAILED.")
        assert jobs[failing_job.job_id].permanent is True

        await _open_library_ingest_canvas(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-dismiss-ingest-job-1")

        assert not list(screen.query("#library-ingest-retry-ingest-job-1"))
        assert screen.query_one("#library-ingest-dismiss-ingest-job-1", Button)
        # Defense in depth (registry-level guard) even if something bypasses
        # the canvas's own button gating.
        assert harness.retry_library_ingest_job(failing_job.job_id) is None


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_failed_row_actions_share_one_line(tmp_path):
    """(L5, fix batch F1b) Retry and Dismiss render on ONE line -- wrapped
    in a shared ``Horizontal`` -- rather than stacking vertically."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)

        job = harness.library_ingest_jobs.submit(source_path="/tmp/broken.pdf")
        harness.library_ingest_jobs.mark_parsing(job.job_id)
        harness.library_ingest_jobs.mark_writing(job.job_id)
        harness.library_ingest_jobs.mark_failed(job.job_id, error="boom", permanent=False)

        await _open_library_ingest_canvas(screen, pilot)
        await _wait_for_selector(screen, pilot, f"#library-ingest-retry-{job.job_id}")

        retry_button = screen.query_one(f"#library-ingest-retry-{job.job_id}", Button)
        dismiss_button = screen.query_one(f"#library-ingest-dismiss-{job.job_id}", Button)
        assert retry_button.region.y == dismiss_button.region.y
        assert retry_button.parent is dismiss_button.parent


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_submit_clears_path_and_title_keeps_metadata(
    tmp_path,
):
    """(A1) On successful submit, both the path AND title fields clear --
    title is per-file -- while author/keywords/toggles persist so a batch
    of files sharing metadata doesn't need to be retyped for every
    submission."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    source = tmp_path / "tides.txt"
    source.write_text("Tides are driven by the moon's gravity.", encoding="utf-8")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        await _open_library_ingest_canvas(screen, pilot)

        screen.query_one("#library-ingest-path", Input).value = str(source)
        screen.query_one("#library-ingest-title", Input).value = "Tides 101"
        screen.query_one("#library-ingest-author", Input).value = "Jane Doe"
        screen.query_one("#library-ingest-keywords", Input).value = "ocean, moon"
        await pilot.pause()

        screen.query_one("#library-ingest-start", Button).press()
        await pilot.pause()

        assert screen._library_ingest_form.path == ""
        assert screen._library_ingest_form.title == ""
        assert screen._library_ingest_form.author == "Jane Doe"
        assert screen._library_ingest_form.keywords == "ocean, moon"
        assert screen.query_one("#library-ingest-path", Input).value == ""
        assert screen.query_one("#library-ingest-title", Input).value == ""
        assert screen.query_one("#library-ingest-author", Input).value == "Jane Doe"
        assert screen.query_one("#library-ingest-keywords", Input).value == "ocean, moon"


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_dismiss_button_removes_failed_row(tmp_path):
    """(B2) Pressing Dismiss on a failed row removes it from the canvas AND
    the registry -- the row is gone, not just visually hidden."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    missing = tmp_path / "does-not-exist.txt"
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)

        failing_job = harness.submit_library_ingest_job(source_path=str(missing))
        for _ in range(_INGEST_POLL_ATTEMPTS):
            jobs = {j.job_id: j for j in harness.library_ingest_jobs.jobs()}
            if jobs[failing_job.job_id].state == IngestJobState.FAILED:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError("Job never reached FAILED.")

        await _open_library_ingest_canvas(screen, pilot)
        # Row-action buttons are keyed by job_id, not row index (PR #591
        # review, F1) -- this is the only job in a fresh registry, so its
        # id is deterministically "ingest-job-1".
        await _wait_for_selector(screen, pilot, "#library-ingest-dismiss-ingest-job-1")

        screen.query_one("#library-ingest-dismiss-ingest-job-1", Button).press()
        await pilot.pause()

        assert harness.library_ingest_jobs.jobs() == ()
        assert not list(screen.query(".library-ingest-row"))
        assert screen.query_one("#library-ingest-queue-empty")


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_dismiss_targets_correct_job_across_stale_render(
    tmp_path,
):
    """(F1 regression, PR #591 review) A Dismiss button pressed AFTER the
    queue has reordered underneath it -- but BEFORE the canvas has
    re-rendered to reflect that reorder -- must still act on the job it
    was rendered for, never on whatever job happens to now sit at the
    same row index.

    Sequence: render the queue with exactly one FAILED job ("the
    target"), capture a direct reference to ITS Dismiss button, then
    freeze the canvas's own ``refresh(recompose=True)`` (a no-op stand-in)
    to hold the DOM in that exact "just rendered, not yet caught up"
    state -- reproducing the real production window between a registry
    mutation and Textual actually applying the scheduled recompose on a
    later event-loop turn (see
    ``LibraryScreen._handle_library_ingest_registry_changed``'s
    docstring). Only then is a second, newer job appended straight to the
    registry -- it sorts first in the newest-first ``jobs()`` snapshot,
    so "row index 0" now means something different than it did when the
    captured button was built -- and only then is the captured (frozen,
    still-mounted) button actually pressed.

    Job_id-keyed button ids resolve the ORIGINAL target job regardless of
    that index shift (dismissed; the new job is untouched and still
    queued). The index-keyed scheme this replaces instead re-derives row
    0 fresh at press time -- landing on the new QUEUED job, whose
    ``can_dismiss`` is False -- and silently no-ops, leaving the failed
    row the user was looking at un-dismissed (confirmed by running this
    exact test against the pre-fix handlers: it fails with the target job
    still present/un-dismissed).
    """
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    missing = tmp_path / "does-not-exist.txt"
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)

        target_job = harness.submit_library_ingest_job(source_path=str(missing))
        for _ in range(_INGEST_POLL_ATTEMPTS):
            jobs = {j.job_id: j for j in harness.library_ingest_jobs.jobs()}
            if jobs[target_job.job_id].state == IngestJobState.FAILED:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError("Job never reached FAILED.")

        await _open_library_ingest_canvas(screen, pilot)
        await _wait_for_selector(screen, pilot, f"#library-ingest-dismiss-{target_job.job_id}")
        stale_dismiss_button = screen.query_one(
            f"#library-ingest-dismiss-{target_job.job_id}", Button
        )

        # Freeze the canvas mid-render: block the registry-listener-driven
        # recompose so the DOM stays exactly as it was when the button
        # above was captured, rather than racing Textual's own recompose
        # scheduling (unreliable to depend on from a test).
        screen.refresh = lambda *args, **kwargs: None

        # Append a newer job DIRECTLY on the registry (not through
        # ``submit_library_ingest_job``) so it stays QUEUED forever --
        # nothing ever claims it, so no background-thread timing is
        # involved.
        newer_job = harness.library_ingest_jobs.submit(source_path=str(tmp_path / "newer.txt"))
        assert newer_job.job_id != target_job.job_id
        # Confirm the reorder actually happened underneath the frozen DOM.
        assert harness.library_ingest_jobs.jobs()[0].job_id == newer_job.job_id

        # Press the button captured BEFORE the mutation above -- this is
        # the "stale render, but the click still lands" race from F1.
        stale_dismiss_button.press()
        await pilot.pause()

        jobs_by_id = {j.job_id: j for j in harness.library_ingest_jobs.jobs()}
        assert target_job.job_id not in jobs_by_id, (
            "Dismiss must remove the job the button was rendered for, "
            "even after the queue reordered underneath it."
        )
        assert newer_job.job_id in jobs_by_id
        assert jobs_by_id[newer_job.job_id].state == IngestJobState.QUEUED


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_clear_finished_empties_done_and_failed(
    tmp_path,
):
    """(B2) Pressing "Clear finished" removes every done+failed job from
    the registry in one shot, leaving the queue empty when nothing else is
    queued/running."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    source = tmp_path / "tides.txt"
    source.write_text("Tides are driven by the moon's gravity.", encoding="utf-8")
    missing = tmp_path / "does-not-exist.txt"
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)

        done_job = harness.submit_library_ingest_job(source_path=str(source))
        for _ in range(_INGEST_POLL_ATTEMPTS):
            jobs = {j.job_id: j for j in harness.library_ingest_jobs.jobs()}
            if jobs.get(done_job.job_id) and jobs[done_job.job_id].state == IngestJobState.DONE:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError("Job never reached DONE.")

        failing_job = harness.submit_library_ingest_job(source_path=str(missing))
        for _ in range(_INGEST_POLL_ATTEMPTS):
            jobs = {j.job_id: j for j in harness.library_ingest_jobs.jobs()}
            if jobs.get(failing_job.job_id) and jobs[failing_job.job_id].state == IngestJobState.FAILED:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError("Job never reached FAILED.")

        await _open_library_ingest_canvas(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-clear-finished")

        screen.query_one("#library-ingest-clear-finished", Button).press()
        await pilot.pause()

        counts = harness.library_ingest_jobs.counts()
        assert counts["done"] == 0
        assert counts["failed"] == 0
        assert not list(screen.query(".library-ingest-row"))
        assert screen.query_one("#library-ingest-queue-empty")
        assert not list(screen.query("#library-ingest-clear-finished"))


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_quiet_line_toggles_live_while_typing(tmp_path):
    """(A4, live-QA repro) ``handle_library_ingest_path_changed`` deliberately
    avoids a full canvas recompose while typing (to preserve the Input's
    cursor position) -- it must still keep the quiet line in sync via the
    same kind of targeted, no-recompose update it already does for the
    Start button's ``disabled`` flag. Before the fix, the quiet line stayed
    stuck showing (from the initial mount) even after a path was typed and
    Start had already gone enabled."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-canvas")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        await _open_library_ingest_canvas(screen, pilot)

        assert screen.query_one("#library-ingest-start-quiet-line", Static)
        assert screen.query_one("#library-ingest-start", Button).disabled is True

        screen.query_one("#library-ingest-path", Input).value = str(tmp_path / "a.txt")
        await pilot.pause()

        assert not list(screen.query("#library-ingest-start-quiet-line"))
        assert screen.query_one("#library-ingest-start", Button).disabled is False

        # Clearing the path back out must bring the quiet line back too --
        # the same targeted update in reverse.
        screen.query_one("#library-ingest-path", Input).value = ""
        await pilot.pause()

        assert screen.query_one("#library-ingest-start-quiet-line", Static)
        assert screen.query_one("#library-ingest-start", Button).disabled is True


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_db_unavailable_disables_start(tmp_path):
    """(d) A missing media DB disables Start with the exact blocked copy."""
    harness = _LibraryIngestCanvasHarness(None)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        await _open_library_ingest_canvas(screen, pilot)

        unavailable_line = screen.query_one("#library-ingest-unavailable-line")
        assert str(unavailable_line.renderable) == "Media database is unavailable."

        screen.query_one("#library-ingest-path", Input).value = str(tmp_path / "a.txt")
        await pilot.pause()
        assert screen.query_one("#library-ingest-start", Button).disabled is True


@pytest.mark.asyncio
async def test_library_shell_ingest_advanced_expand_survives_toggle_and_listener_recompose(tmp_path):
    """(F1, whole-branch review fix) The Advanced options collapsible must
    stay expanded across BOTH kinds of recompose the ingest canvas can hit
    while the panel is open: the analyze/chunk toggle handlers' own
    ``refresh(recompose=True)``, and the registry listener's recompose on a
    job transition (``_handle_library_ingest_registry_changed``). Before the
    fix, the widget hardcoded ``collapsed=True`` on every compose, so
    pressing "Analyze after ingest" INSIDE the panel closed it out from
    under the user (mirrors
    ``test_library_shell_history_manual_expand_survives_scope_toggle_recompose``).
    """
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-advanced")
    source = tmp_path / "note.txt"
    source.write_text("Advanced options must survive a recompose.", encoding="utf-8")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        await _open_library_ingest_canvas(screen, pilot)

        # Starts collapsed.
        assert screen.query_one("#library-ingest-advanced", Collapsible).collapsed is True

        # Mirror a user click on the collapsible header (expand).
        screen.query_one("#library-ingest-advanced", Collapsible).collapsed = False
        for _ in range(_INGEST_POLL_ATTEMPTS):
            if screen._library_ingest_form.advanced_open is True:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError("Manual expand never synced back to advanced_open.")

        # Pressing the analyze toggle (INSIDE the panel) recomposes the canvas.
        screen.query_one("#library-ingest-analyze-toggle", Button).press()
        await _wait_for_selector(screen, pilot, "#library-ingest-advanced")

        assert screen.query_one("#library-ingest-advanced", Collapsible).collapsed is False

        # A registry-listener-driven recompose (job transition) must also
        # leave the panel expanded -- submit programmatically, exactly like
        # the queue-runner's own call_from_thread-marshaled transitions.
        harness.submit_library_ingest_job(source_path=str(source))
        for _ in range(_INGEST_POLL_ATTEMPTS):
            jobs = harness.library_ingest_jobs.jobs()
            if jobs and jobs[0].state == IngestJobState.DONE:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError(f"Job never reached DONE: {harness.library_ingest_jobs.jobs()}")
        await pilot.pause()

        assert screen.query_one("#library-ingest-advanced", Collapsible).collapsed is False


class _IngestCanvasWidgetHost(App):
    """Bare host for mounting ``LibraryIngestCanvas`` directly with a
    hand-built state -- used only to exercise the widget's markup-escaping
    in isolation, without a full Library screen."""

    def __init__(self, state):
        super().__init__()
        self._state = state

    def compose(self):
        yield LibraryIngestCanvas(self._state, id="library-ingest-canvas")


@pytest.mark.asyncio
async def test_library_ingest_canvas_renders_markup_hostile_filename_without_crash():
    """(e) A filename containing Rich-markup-like syntax (brackets that
    look like closing tags) must render without raising MarkupError --
    mirrors the ``docs [/archive] cleanup`` lesson already fixed for the
    Search/RAG history rows."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="weird [/bracket] name.txt",
        state=IngestJobState.FAILED,
        error="failed near [bold]tag[/bold] marker",
        submitted_at=1.0,
        started_at=1.0,
        finished_at=2.0,
    )
    state = build_library_ingest_state((job,), form=LibraryIngestFormState())
    host = _IngestCanvasWidgetHost(state)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        row = host.query_one("#library-ingest-row-0", Static)
        # Reaching this line at all is the core assertion: a MarkupError
        # during compose/mount would have raised before ``run_test``'s
        # context manager ever returned control here.
        assert "bracket" in str(row.renderable)
        assert "tag" in str(row.renderable)
        # Row-action buttons are keyed by job_id, not row index (PR #591
        # review, F1).
        assert host.query_one("#library-ingest-retry-ingest-job-1")


# --- L3b AB wave: widget-level (A4/A5/A6/B2) --------------------------------


@pytest.mark.asyncio
async def test_library_ingest_canvas_metadata_placeholders_are_optional_labeled():
    """(A5) Title/Author/Keywords placeholders spell out "(optional)" so
    the form doesn't read as if every field were required."""
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    host = _IngestCanvasWidgetHost(state)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        assert (
            host.query_one("#library-ingest-title", Input).placeholder
            == "Title (optional)"
        )
        assert (
            host.query_one("#library-ingest-author", Input).placeholder
            == "Author (optional)"
        )
        assert (
            host.query_one("#library-ingest-keywords", Input).placeholder
            == "Keywords, comma-separated (optional)"
        )


@pytest.mark.asyncio
async def test_library_ingest_canvas_chunk_size_input_labeled_and_disable_follows_toggle():
    """(A6) The chunk-size Input gets a "Chunk size (words)" placeholder and
    is visually disabled whenever "Chunk content" is toggled off (submit
    already ignores it when disabled; this only adds the visual affordance)."""
    off_state = build_library_ingest_state(
        (), form=LibraryIngestFormState(chunk=False)
    )
    host = _IngestCanvasWidgetHost(off_state)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        chunk_size_input = host.query_one("#library-ingest-chunk-size", Input)
        assert chunk_size_input.placeholder == "Chunk size (words)"
        assert chunk_size_input.disabled is True

    on_state = build_library_ingest_state(
        (), form=LibraryIngestFormState(chunk=True)
    )
    host2 = _IngestCanvasWidgetHost(on_state)
    async with host2.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        assert host2.query_one("#library-ingest-chunk-size", Input).disabled is False


@pytest.mark.asyncio
async def test_library_ingest_canvas_start_quiet_line_renders_when_path_blank():
    state = build_library_ingest_state((), form=LibraryIngestFormState(path=""))
    host = _IngestCanvasWidgetHost(state)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        line = host.query_one("#library-ingest-start-quiet-line", Static)
        assert str(line.renderable) == "Enter a file path to start."


@pytest.mark.asyncio
async def test_library_ingest_canvas_start_quiet_line_absent_when_path_typed():
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="/tmp/a.txt")
    )
    host = _IngestCanvasWidgetHost(state)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        assert not list(host.query("#library-ingest-start-quiet-line"))


@pytest.mark.asyncio
async def test_library_ingest_canvas_counts_line_hidden_when_no_jobs():
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    host = _IngestCanvasWidgetHost(state)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        assert not list(host.query("#library-ingest-queue-counts"))
        assert host.query_one("#library-ingest-queue-empty", Static)


@pytest.mark.asyncio
async def test_library_ingest_canvas_counts_line_shown_when_jobs_present():
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.QUEUED,
        submitted_at=1.0,
    )
    state = build_library_ingest_state((job,), form=LibraryIngestFormState())
    host = _IngestCanvasWidgetHost(state)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        counts_line = host.query_one("#library-ingest-queue-counts", Static)
        assert str(counts_line.renderable) == "1 queued"
        assert not list(host.query("#library-ingest-queue-empty"))


@pytest.mark.asyncio
async def test_library_ingest_canvas_failed_row_renders_dismiss_next_to_retry():
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/broken.pdf",
        state=IngestJobState.FAILED,
        error="unsupported format",
        submitted_at=1.0,
        started_at=1.0,
        finished_at=2.0,
    )
    state = build_library_ingest_state((job,), form=LibraryIngestFormState())
    host = _IngestCanvasWidgetHost(state)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        # Row-action buttons are keyed by job_id, not row index (PR #591
        # review, F1).
        assert host.query_one("#library-ingest-retry-ingest-job-1", Button)
        dismiss_button = host.query_one("#library-ingest-dismiss-ingest-job-1", Button)
        assert "library-ingest-row-action" in dismiss_button.classes


@pytest.mark.asyncio
async def test_library_ingest_canvas_clear_finished_absent_with_no_finished_jobs():
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.QUEUED,
        submitted_at=1.0,
    )
    state = build_library_ingest_state((job,), form=LibraryIngestFormState())
    host = _IngestCanvasWidgetHost(state)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        assert not list(host.query("#library-ingest-clear-finished"))


@pytest.mark.asyncio
async def test_library_ingest_canvas_clear_finished_present_with_a_failed_job():
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/broken.pdf",
        state=IngestJobState.FAILED,
        error="unsupported format",
        submitted_at=1.0,
        started_at=1.0,
        finished_at=2.0,
    )
    state = build_library_ingest_state((job,), form=LibraryIngestFormState())
    host = _IngestCanvasWidgetHost(state)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await pilot.pause()
        clear_button = host.query_one("#library-ingest-clear-finished", Button)
        # Plain (not primary/accented like #library-ingest-start), and not a
        # per-row action (it sits below all queue rows, not next to one).
        assert "library-canvas-action" in clear_button.classes
        assert "library-ingest-row-action" not in clear_button.classes


# --- Live updates: registry listener -> canvas refresh + count poke
# (L3b Task 5) --------------------------------------------------------------
#
# Task 4's tests above (e.g. the "happy path" test's comment) deliberately
# forced a manual ``screen.refresh(recompose=True)`` once a job reached
# DONE, because no live-update listener existed yet. These pilots assert
# the opposite: NO manual recompose anywhere in the test body -- the
# registry listener registered in ``LibraryScreen.on_mount`` must be the
# only thing driving the row/rail updates onto screen.


class _DummyReplacementScreen(Screen):
    """Minimal screen used only to replace ``LibraryScreen`` on the stack.

    ``App.switch_screen`` pops the previous top screen and -- since it is
    not installed/present in any other stack -- awaits its ``.remove()``,
    which actually unmounts it (firing ``on_unmount``). Merely *pushing* a
    screen on top would only suspend the one underneath, which is exactly
    the lifecycle event ``LibraryScreen.on_unmount`` documents itself as
    deliberately NOT keying off of.
    """

    def compose(self) -> ComposeResult:
        yield Static("replacement", id="dummy-replacement-static")


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_live_updates_without_manual_recompose(tmp_path):
    """(a) With the ingest canvas open, a *programmatic* submit (calling
    the app seam directly, exactly like the queue-runner's own
    ``call_from_thread``-marshaled transitions, and unlike a button press
    whose handler does its own trailing recompose) must still make the
    queue row appear, flip to done, and grow the rail ``Media (N)`` count
    -- all without this test ever calling ``refresh(recompose=True)``
    itself."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-live")
    source = tmp_path / "river.txt"
    source.write_text("Rivers carve valleys over millennia.", encoding="utf-8")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        await _open_library_ingest_canvas(screen, pilot)

        media_button = screen.query_one("#library-row-browse-media", Button)
        assert "Media (0)" in str(media_button.label)

        harness.submit_library_ingest_job(source_path=str(source))

        for _ in range(_INGEST_POLL_ATTEMPTS):
            if screen.query("#library-ingest-row-0"):
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"Ingest row never appeared without a manual recompose. "
                f"Visible text: {_visible_text(screen)}"
            )

        for _ in range(_INGEST_POLL_ATTEMPTS):
            rows = list(screen.query("#library-ingest-row-0"))
            if rows and str(rows[0].renderable).startswith("✓ done"):
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"Row never reached done without a manual recompose: "
                f"{harness.library_ingest_jobs.jobs()}"
            )

        for _ in range(_INGEST_POLL_ATTEMPTS):
            media_button = screen.query_one("#library-row-browse-media", Button)
            if "Media (1)" in str(media_button.label):
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"Rail Media count never incremented after completion. "
                f"Label: {media_button.label!r}"
            )


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_registry_listener_removed_on_unmount(tmp_path):
    """(b) The registry listener registered in ``on_mount`` is removed in
    ``on_unmount``: replacing ``LibraryScreen`` on the stack (real
    unmount, not a suspend-only push) drops the registry's listener count
    to zero, and a subsequent mutation neither raises nor resurrects the
    removed screen."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-unmount")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        assert screen in harness.screen_stack
        assert len(harness.library_ingest_jobs._listeners) == 1

        await harness.switch_screen(_DummyReplacementScreen())
        await pilot.pause()
        await pilot.pause()

        # Note: Textual's ``Widget.is_mounted`` tracks "has been mounted at
        # least once" (flipped True on first mount, never reset), so it is
        # NOT the right signal for "was later removed" -- stack membership
        # is (mirrors how ``App._replace_screen`` itself decides whether
        # to actually call ``.remove()``: not installed + not present in
        # any screen stack).
        assert screen not in harness.screen_stack
        assert len(harness.library_ingest_jobs._listeners) == 0

        # Must not raise, and must not resurrect/recompose the removed
        # screen -- the queue-runner will run this (missing) file to a
        # FAILED transition on a background thread, exercising the exact
        # call_from_thread-marshaled notify path against zero listeners.
        harness.submit_library_ingest_job(source_path=str(tmp_path / "ghost.txt"))
        for _ in range(_INGEST_POLL_ATTEMPTS):
            jobs = harness.library_ingest_jobs.jobs()
            if jobs and jobs[0].state == IngestJobState.FAILED:
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError("Ghost job never reached FAILED.")

        assert screen not in harness.screen_stack


@pytest.mark.asyncio
async def test_library_shell_ingest_canvas_different_canvas_isolation(tmp_path):
    """(c) Completing a job while a DIFFERENT canvas (Notes) is selected
    must not yank the user onto the ingest canvas -- the selected row and
    composed widgets stay on Notes -- but the rail ``Media (N)`` count
    still updates once the job completes."""
    db = MediaDatabase(tmp_path / "ingest-canvas.db", client_id="l3b-ingest-isolation")
    source = tmp_path / "delta.txt"
    source.write_text("Deltas form where rivers meet the sea.", encoding="utf-8")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await pilot.pause()
        await pilot.pause()
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES

        harness.submit_library_ingest_job(source_path=str(source))

        for _ in range(_INGEST_POLL_ATTEMPTS):
            media_button = screen.query_one("#library-row-browse-media", Button)
            if "Media (1)" in str(media_button.label):
                break
            await pilot.pause(_INGEST_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"Rail Media count never incremented while Notes was open. "
                f"Label: {media_button.label!r}"
            )

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen._library_selected_row_id != LIBRARY_ROW_INGEST_MEDIA
        assert not screen.query("#library-ingest-path")
        assert not list(screen.query(".library-ingest-row"))


# --- Cross-visit state persistence (save_state/restore_state) -------------
#
# Screens are never cached/reused across a navigation (a fresh instance is
# constructed every visit); continuity is entirely the app's
# ``_screen_states`` dict, keyed by screen name and populated/consumed by
# ``save_state``/``restore_state`` (see app.py's ``handle_screen_navigation``).
# These tests exercise ``LibraryScreen``'s real overrides directly (unit
# style) plus the on_mount interaction the restored viewer/editor state
# depends on (pilot style, via ``LibraryHarness``).


@pytest.mark.asyncio
async def test_library_shell_save_state_captures_selection_and_rag_state():
    """``save_state`` is the ``_screen_states`` producer -- assert it
    actually carries the selection/view attrs the restore contract
    promises, and never leaks a bulk fetched snapshot.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        screen.query_one("#library-rag-query-input", Input).value = "roadmap"
        await pilot.pause()
        await pilot.pause()

        state = screen.save_state()

    assert state["library_selected_row_id"] == LIBRARY_ROW_BROWSE_SEARCH
    assert state["library_rag_query"] == "roadmap"
    assert state["library_rag_mode"] == "search"
    assert state["library_rag_scope_deselected"] == set()
    assert state["library_rag_results"] == ()
    assert state["library_rag_selected_result_id"] == ""
    # Bulk fetched snapshots must never leak into the persisted dict --
    # screens re-fetch fresh on the next mount, and a restored id may be
    # stale by then (see the stale-id tests below).
    assert "local_source_records" not in state
    assert "library_note_detail" not in state
    assert "library_media_detail" not in state


def test_library_shell_restore_state_sets_attrs_on_fresh_unmounted_instance():
    """``restore_state`` runs on a freshly-constructed, not-yet-mounted
    instance (see app.py's ``handle_screen_navigation``): construct one
    exactly that way and assert every attr lands, mirroring the base
    ``test_screen_state_preservation`` contract test but for Library's real
    override.
    """
    app = _build_test_app()
    original = LibraryScreen(app)
    original._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    original._selected_media_id = "media-42"
    original._library_media_view = "viewer"
    original._library_rag_query = "alpha"
    original._library_rag_mode = "rag"
    original._library_rag_scope_deselected = {"notes"}
    state = original.save_state()

    restored = LibraryScreen(app)
    restored.restore_state(state)

    assert restored._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
    assert restored._selected_media_id == "media-42"
    assert restored._library_media_view == "viewer"
    assert restored._library_rag_query == "alpha"
    assert restored._library_rag_mode == "rag"
    assert restored._library_rag_scope_deselected == {"notes"}

    # The restore must not alias the saved dict's mutable set -- mutating
    # the restored instance's copy must never bleed back into the saved
    # state dict (a shallow-copied structure the app's runtime-policy
    # reconciliation touches on every navigation).
    restored._library_rag_scope_deselected.add("media")
    assert state["library_rag_scope_deselected"] == {"notes"}


def test_library_shell_restore_state_degrades_editor_view_without_matching_id():
    """A corrupted/foreign saved-state dict (``library_notes_view`` says
    "editor" but carries no note id) must not leave the screen pointed at a
    permanent "Loading note..." placeholder -- fall back to the list view.
    Same guard, mirrored for the media viewer.
    """
    app = _build_test_app()

    notes_screen = LibraryScreen(app)
    notes_screen.restore_state({"library_notes_view": "editor", "selected_note_id": ""})
    assert notes_screen._library_notes_view == "list"
    assert notes_screen._selected_note_id == ""

    media_screen = LibraryScreen(app)
    media_screen.restore_state({"library_media_view": "viewer", "selected_media_id": ""})
    assert media_screen._library_media_view == "list"
    assert media_screen._selected_media_id == ""


def test_library_shell_restore_state_tolerates_garbage_values():
    """``restore_state`` must never crash on a saved-state dict from a
    different build/shape -- e.g. a non-dict ``library_rag_results``, or a
    ``library_rag_mode`` outside the two known values.
    """
    app = _build_test_app()
    screen = LibraryScreen(app)

    screen.restore_state(
        {
            "library_rag_mode": "not-a-real-mode",
            "library_rag_results": "not-a-tuple",
            "library_rag_scope_deselected": "not-a-set",
            "library_rag_recovery_state": "not-a-dataclass",
        }
    )

    assert screen._library_rag_mode == "search"
    assert screen._library_rag_results == ()
    assert screen._library_rag_scope_deselected == set()
    assert screen._library_rag_recovery_state is None

    # A completely non-dict payload (e.g. a runtime-policy mismatch that
    # ``reconcile_saved_screen_state`` failed to catch) must be a no-op,
    # not a crash.
    screen.restore_state(None)


@pytest.mark.asyncio
async def test_library_shell_restored_media_viewer_fetches_detail_on_mount():
    """Mirrors the notes-editor nav-context deep link ``on_mount`` already
    handled before this task: a restored ``_library_media_view == "viewer"``
    with a ``_selected_media_id`` must kick the detail fetch itself, since
    nothing else does for a restore (unlike a live row click, which calls
    ``_refresh_library_media_detail`` directly from
    ``handle_library_media_row``).
    """
    app = _build_test_app()
    _seed_conversations(app, [], media=_two_media_items())

    screen = LibraryScreen(app)
    screen.restore_state(
        {
            "library_selected_row_id": LIBRARY_ROW_BROWSE_MEDIA,
            "selected_media_id": "media-1",
            "library_media_view": "viewer",
        }
    )
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_selector(screen, pilot, "#library-media-viewer")

        assert screen._library_media_detail is not None
        assert screen._library_media_detail.get("id") == "media-1"
        assert screen._library_media_view == "viewer"


@pytest.mark.asyncio
async def test_library_shell_restored_export_canvas_rekicks_counts_worker_on_mount():
    """REVIEW FIX (F4 Task 3): a cross-visit ``restore_state`` (or a tab
    round-trip whose ``save_state`` persisted ``_library_selected_row_id ==
    LIBRARY_ROW_INGEST_EXPORT``) lands a fresh instance ON the export
    canvas with ``_library_export_counts is None`` -- so the scope line
    renders "Counting…" and Export is disabled. The counts worker is only
    kicked from the two LIVE entry points, never from a restore, so without
    an ``on_mount`` re-kick the form stays stuck "Counting…" with Export
    permanently disabled until the user clicks another rail row and back.
    Same restored-placeholder class already handled for the media
    viewer/notes editor (see
    ``test_library_shell_restored_media_viewer_fetches_detail_on_mount``).

    RED-verified: fails without the ``on_mount`` re-kick block (counts
    never land -> the bounded poll below raises).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-restore-media")
    app.media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-restore-ccn")
    app.chachanotes_db.add_conversation({"title": "Conv"})
    app.chachanotes_db.add_note("N1", "content")

    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_INGEST_EXPORT})
    # Precondition: the restore alone leaves counts unresolved -- the
    # canvas would render "Counting…" forever without the on_mount kick.
    assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT
    assert screen._library_export_counts is None
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_selector(screen, pilot, "#library-export-header")

        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Restored export canvas never re-kicked its counts worker."
            )

        screen.refresh(recompose=True)
        await pilot.pause()

        scope_line = str(screen.query_one("#library-export-scope-line").renderable)
        assert scope_line == "Everything: 1 media · 1 conversations · 1 notes"
        # Non-empty scope + counts landed: Export is no longer stuck
        # disabled by a permanent "Counting…" (only the missing
        # destination keeps it disabled now, which is correct).
        assert screen.query_one("#library-export-empty-line", Static).display is False


@pytest.mark.asyncio
async def test_library_shell_restored_media_viewer_with_deleted_item_falls_back_to_list():
    """Stale-id safety: the media record backing a restored viewer selection
    was deleted while the user was elsewhere. The existing
    ``_refresh_library_media_detail`` unavailable-notify fallback must fire
    the same way it does for a live click on a since-deleted row.
    """
    app = _build_test_app()
    _seed_conversations(app, [], media=[])  # nothing resolves "media-ghost"
    notified = []
    app.notify = lambda message, **kwargs: notified.append(message)

    screen = LibraryScreen(app)
    screen.restore_state(
        {
            "library_selected_row_id": LIBRARY_ROW_BROWSE_MEDIA,
            "selected_media_id": "media-ghost",
            "library_media_view": "viewer",
        }
    )
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        for _ in range(120):
            if screen._library_media_view == "list":
                break
            await pilot.pause(0.02)

        assert screen._library_media_view == "list"
        assert any("unavailable" in message.lower() for message in notified)


@pytest.mark.asyncio
async def test_library_shell_restored_notes_editor_with_deleted_note_falls_back_to_list():
    """Same stale-id contract as the media viewer above, but for notes --
    exercised through the pre-existing ``on_mount`` deep-link fetch (this
    task's restore just feeds it the same inputs a ``note_id`` nav-context
    deep link always could)."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    notified = []
    app.notify = lambda message, **kwargs: notified.append(message)

    screen = LibraryScreen(app)
    screen.restore_state(
        {
            "library_selected_row_id": LIBRARY_ROW_BROWSE_NOTES,
            "selected_note_id": "note-ghost",
            "library_notes_view": "editor",
        }
    )
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        for _ in range(120):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)

        assert screen._library_notes_view == "list"
        assert notified  # _notify_library_note_missing_warning fired


# --- Export canvas (F4 Task 2) -----------------------------------------------
#
# Real in-memory ``MediaDatabase``/``CharactersRAGDB`` handles (not fakes):
# the counts worker's ``is_memory_db`` guard (mirrors
# ``_fetch_library_conversation_by_id``'s) runs the count inline on the UI
# thread for these, exercising the exact same code path a real file-backed
# deployment would exercise on a genuine worker thread -- only the
# thread-vs-inline dispatch differs, not the query/marshal/gate logic.


@pytest.mark.asyncio
async def test_library_shell_export_rail_row_opens_everything_scope_and_counts_land():
    """Pressing the Export rail row opens the export canvas scoped to
    Everything; the counts worker lands a real full-query result (never
    the rendered/capped snapshot) within a bounded poll.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    media_db = MediaDatabase(":memory:", client_id="export-pilot-everything-media")
    media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.media_db = media_db
    chachanotes_db = CharactersRAGDB(":memory:", client_id="export-pilot-everything-ccn")
    chachanotes_db.add_conversation({"title": "Conv"})
    chachanotes_db.add_note("N1", "content")
    app.chachanotes_db = chachanotes_db
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-header")

        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT
        assert screen._library_export_scope == ExportScope(kind="everything")

        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed.")

        screen.refresh(recompose=True)
        await pilot.pause()

        scope_line = str(screen.query_one("#library-export-scope-line").renderable)
        assert scope_line == "Everything: 1 media · 1 conversations · 1 notes"
        submit = screen.query_one("#library-export-submit", Button)
        # Counts landed with a positive total, but no destination chosen yet.
        assert submit.disabled is True


@pytest.mark.asyncio
async def test_library_shell_media_export_action_carries_type_filter_into_scope():
    """The media canvas's "Export…" action opens the export canvas
    pre-scoped to Media with the canvas's CURRENT type filter -- never
    Everything, and never the filter's default."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    app.media_db = MediaDatabase(":memory:", client_id="export-pilot-media-scope-media")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-pilot-media-scope-ccn")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_BROWSE_MEDIA}").press()
        await _wait_for_selector(screen, pilot, "#library-media-type-filter")

        # Cycle the type filter off "All" onto a concrete type before
        # opening Export -- the scope must carry THIS filter, not the
        # canvas's default.
        screen.query_one("#library-media-type-filter").press()
        await pilot.pause()
        active_type = screen._library_media_type_filter
        assert active_type != "All"

        screen.query_one("#library-media-export").press()
        await _wait_for_selector(screen, pilot, "#library-export-header")

        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT
        assert screen._library_export_scope == ExportScope(kind="media", media_type=active_type)


@pytest.mark.asyncio
async def test_library_shell_export_empty_scope_disables_export_and_shows_helper():
    """An empty-everywhere scope disables Export with the exact helper copy."""
    app = _build_test_app()
    _seed_conversations(app, [])
    app.media_db = MediaDatabase(":memory:", client_id="export-pilot-empty-media")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-pilot-empty-ccn")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-header")

        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed.")

        screen.refresh(recompose=True)
        await pilot.pause()

        submit = screen.query_one("#library-export-submit", Button)
        assert submit.disabled is True
        empty_line = str(screen.query_one("#library-export-empty-line").renderable)
        assert empty_line == EMPTY_SCOPE_COPY


@pytest.mark.asyncio
async def test_library_shell_export_choose_destination_pushes_file_save_dialog_with_sanitized_name():
    """"Choose destination…" pushes a ``FileSave`` dialog pre-filled from the
    export name field, mirroring ``_export_library_note``'s dialog flow."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-pilot-fs-media")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-pilot-fs-ccn")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")

        expected_name = screen._library_export_form["name"]
        screen.query_one("#library-export-destination").press()
        for _ in range(150):
            if isinstance(host.screen_stack[-1], FileSave):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Choose destination… never pushed a FileSave dialog.")

        dialog = host.screen_stack[-1]
        assert dialog._default_file == f"{expected_name}.zip"

        await host.pop_screen()
        await pilot.pause()


@pytest.mark.asyncio
async def test_library_shell_export_destination_normalizes_missing_suffix_to_zip(tmp_path):
    """A ``FileSave``-returned path with no ``.zip`` suffix (e.g. "foo") is
    normalized to "foo.zip" -- both in the stored form field and the
    rendered destination line -- BEFORE any overwrite check runs. Bypasses
    the dialog UI itself (exercised separately above), mirroring
    ``_write_library_note_export_file``'s direct-call pilot style for the
    "pure part" of the write path.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-pilot-norm-media")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-pilot-norm-ccn")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")

        screen._apply_library_export_destination(tmp_path / "foo")
        await pilot.pause()

        expected = str(tmp_path / "foo.zip")
        assert screen._library_export_form["destination"] == expected
        destination_line = str(screen.query_one("#library-export-destination-line").renderable)
        assert destination_line == expected
        # The freshly-normalized path does not exist yet -- no overwrite line.
        assert not screen.query("#library-export-overwrite-line")


@pytest.mark.asyncio
async def test_library_shell_export_destination_existing_file_shows_overwrite_line(tmp_path):
    """When the ``.zip``-normalized destination already exists on disk, the
    form shows an "Overwrites …" line naming the NORMALIZED file -- purely
    informational (Export stays enabled, not blocked) per the design
    spec's explicit "normalize before confirming overwrite" ordering."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-pilot-ow-media")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-pilot-ow-ccn")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")

        existing = tmp_path / "already-there.zip"
        existing.write_bytes(b"")
        screen._apply_library_export_destination(existing)
        await pilot.pause()

        overwrite_line = str(screen.query_one("#library-export-overwrite-line").renderable)
        assert overwrite_line == "Overwrites already-there.zip"


@pytest.mark.asyncio
async def test_library_shell_export_row_disabled_with_tooltip_in_server_mode():
    """A server-active runtime disables the Export rail row (tooltip
    explains why) -- pressing it is a no-op, since a disabled Textual
    Button never dispatches ``Pressed``."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.runtime_policy = SimpleNamespace(
        state=RuntimeSourceState(
            active_source="server", server_configured=True, active_server_id="srv-1"
        ),
        persist=lambda: None,
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        export_row = screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}", Button)
        assert export_row.disabled is True
        assert export_row.tooltip == LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP

        export_row.press()
        await pilot.pause()

        assert screen._library_selected_row_id != LIBRARY_ROW_INGEST_EXPORT
        assert not screen.query("#library-export-header")


@pytest.mark.asyncio
async def test_library_shell_section_export_action_refuses_in_server_mode():
    """The media canvas's "Export…" action bypasses the rail row's own
    server-disabled gate, so it must re-check runtime mode itself (Qodo
    review): in server mode it warns and never opens the export canvas --
    export reads the LOCAL DBs, so running it would package the wrong
    dataset while the user views server-scoped content."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    app.media_db = MediaDatabase(":memory:", client_id="export-pilot-server-media")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-pilot-server-ccn")
    app.runtime_policy = SimpleNamespace(
        state=RuntimeSourceState(
            active_source="server", server_configured=True, active_server_id="srv-1"
        ),
        persist=lambda: None,
    )
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Drive the section action directly (in server mode the browse
        # canvas may not surface the media list the same way; the handler
        # is the gate under test).
        await screen._open_library_export_canvas(ExportScope(kind="media"))
        await pilot.pause()

        assert screen._library_selected_row_id != LIBRARY_ROW_INGEST_EXPORT
        assert not screen.query("#library-export-header")
        app.notify.assert_called_once()
        assert app.notify.call_args.args[0] == LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP


@pytest.mark.asyncio
async def test_library_shell_conversations_export_action_opens_conversations_scope():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-pilot-conv-media")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-pilot-conv-ccn")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_BROWSE_CONVERSATIONS}").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-export")

        screen.query_one("#library-conversations-export").press()
        await _wait_for_selector(screen, pilot, "#library-export-header")

        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT
        assert screen._library_export_scope == ExportScope(kind="conversations")
        # Conversations-only scope never touches media -- the quality
        # control has nothing to control.
        assert not screen.query("#library-export-quality")


@pytest.mark.asyncio
async def test_library_shell_notes_export_action_opens_notes_scope():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.media_db = MediaDatabase(":memory:", client_id="export-pilot-notes-media")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-pilot-notes-ccn")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_BROWSE_NOTES}").press()
        await _wait_for_selector(screen, pilot, "#library-notes-export")

        screen.query_one("#library-notes-export").press()
        await _wait_for_selector(screen, pilot, "#library-export-header")

        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT
        assert screen._library_export_scope == ExportScope(kind="notes")


@pytest.mark.asyncio
async def test_library_shell_export_counts_worker_uses_real_thread_for_file_backed_dbs(tmp_path):
    """Both export DBs are real, FILE-backed (not ``:memory:``) here --
    unlike every other export pilot above, this exercises the actual
    ``@work(thread=True, exclusive=True, group="library_export_counts")``
    worker-thread dispatch branch (``_start_library_export_counts_worker``'s
    ``is_memory_db`` guard only takes the inline UI-thread path for
    in-memory connections)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    media_db = MediaDatabase(tmp_path / "export-thread.db", client_id="export-pilot-thread-media")
    media_db.add_media_with_keywords(title="M1", content="c1", media_type="article")
    app.media_db = media_db
    chachanotes_db = CharactersRAGDB(
        tmp_path / "export-thread-ccn.db", client_id="export-pilot-thread-ccn"
    )
    chachanotes_db.add_conversation({"title": "Conv"})
    app.chachanotes_db = chachanotes_db
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert not bool(getattr(media_db, "is_memory_db", False))
        assert not bool(getattr(chachanotes_db, "is_memory_db", False))

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-header")

        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed via the worker thread.")

        screen.refresh(recompose=True)
        await pilot.pause()

        scope_line = str(screen.query_one("#library-export-scope-line").renderable)
        assert scope_line == "Everything: 1 media · 1 conversations · 0 notes"


class _GatedExportCountMediaDB:
    """A media-id source whose count blocks until the test releases it.

    ``is_memory_db = False`` deliberately routes
    ``_start_library_export_counts_worker`` onto the real worker-thread
    path, so the counts stay in-flight ("Counting…") for as long as the
    gate is held -- the window in which a user can be mid-keystroke in
    the form when the counts land. The wait is bounded (30.0s) so a
    failing test can never wedge the worker thread past the suite's own
    timeouts (the gated-fake convention used throughout this file).
    """

    is_memory_db = False

    def __init__(self) -> None:
        self.release = threading.Event()

    def get_all_active_media_ids(self, media_type=None):
        assert self.release.wait(timeout=30.0), "count gate never released"
        return [1]


class _StaticExportCountChaChaDB:
    """A fixed-id ChaChaNotes source for the gated counts pilot."""

    is_memory_db = False

    def get_all_conversation_ids(self):
        return ["c-1"]

    def get_all_note_ids(self):
        return []


@pytest.mark.asyncio
async def test_library_shell_export_counts_landing_preserves_input_focus_and_text():
    """REGRESSION (F4 Task 2 review): counts landing must update the form
    IN PLACE -- never recompose the canvas. A recompose destroys and
    rebuilds the name/description ``Input`` mid-keystroke: the typed text
    survives (via the form dict) but keyboard focus does not. Gate the
    counts, focus the name field, type, release the gate, and assert the
    SAME ``Input`` instance still holds focus and the typed text -- and
    that the scope line/Export gate still landed their updates."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    media_db = _GatedExportCountMediaDB()
    app.media_db = media_db
    app.chachanotes_db = _StaticExportCountChaChaDB()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        name_input = await _wait_for_selector(screen, pilot, "#library-export-name")

        # The gate is held: counts are still in flight.
        assert screen._library_export_counts is None
        scope_line = screen.query_one("#library-export-scope-line", Static)
        assert str(scope_line.renderable) == "Counting…"

        name_input.focus()
        await pilot.pause()
        assert screen.focused is name_input
        await pilot.press("h", "i")
        assert name_input.value.endswith("hi")

        media_db.release.set()
        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed after gate release.")
        await pilot.pause()
        await pilot.pause()

        # The SAME widget instances survived the landing (no recompose)...
        assert screen.query_one("#library-export-name", Input) is name_input
        assert screen.focused is name_input
        assert name_input.value.endswith("hi")
        # ...and the targeted updates landed on them.
        assert screen.query_one("#library-export-scope-line", Static) is scope_line
        assert (
            str(scope_line.renderable)
            == "Everything: 1 media · 1 conversations · 0 notes"
        )
        # Positive total, but still no destination -- Export stays disabled.
        assert screen.query_one("#library-export-submit", Button).disabled is True
        # Non-empty scope: the (always-mounted) helper stays hidden.
        assert screen.query_one("#library-export-empty-line", Static).display is False


@pytest.mark.asyncio
async def test_library_shell_export_counts_landing_at_zero_reveals_empty_helper_in_place():
    """The empty-scope helper is display-toggled (not conditionally
    composed): counts landing at zero must reveal it -- text and
    visibility -- without a recompose."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())

    class _EmptyGatedMediaDB(_GatedExportCountMediaDB):
        def get_all_active_media_ids(self, media_type=None):
            assert self.release.wait(timeout=30.0), "count gate never released"
            return []

    class _EmptyChaChaDB(_StaticExportCountChaChaDB):
        def get_all_conversation_ids(self):
            return []

    media_db = _EmptyGatedMediaDB()
    app.media_db = media_db
    app.chachanotes_db = _EmptyChaChaDB()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-header")

        empty_line = screen.query_one("#library-export-empty-line", Static)
        assert empty_line.display is False  # still Counting…

        media_db.release.set()
        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed after gate release.")
        await pilot.pause()

        assert screen.query_one("#library-export-empty-line", Static) is empty_line
        assert empty_line.display is True
        assert str(empty_line.renderable) == EMPTY_SCOPE_COPY
        assert screen.query_one("#library-export-submit", Button).disabled is True


# --- Export canvas: execution worker (F4 Task 3) -----------------------------


class _FakeLibraryExportService:
    """A fake ``local_chatbook_service`` double: async-signature, records calls.

    Mirrors ``LocalChatbookService``'s async-signature/sync-body contract
    (methods run through ``asyncio.run`` inside the worker's real OS
    thread) closely enough to exercise the worker end-to-end, without
    touching any real DB, zip file, or on-disk registry. ``gate`` (when
    given) blocks INSIDE the ``export_chatbook`` coroutine -- safe because
    it runs on its own throwaway event loop on a genuine background thread,
    never the UI thread -- so a test can hold the export "in flight" for as
    long as it needs, bounded by ``_GATED_RELEASE_TIMEOUT_SECONDS`` (the
    gated-fake convention used throughout this file).
    """

    def __init__(
        self,
        *,
        export_result=None,
        gate: "threading.Event | None" = None,
        create_error: "Exception | None" = None,
    ):
        self.export_calls: list[dict] = []
        self.create_calls: list[dict] = []
        self._export_result = (
            export_result
            if export_result is not None
            else {"success": True, "message": "", "path": "", "dependency_info": {}}
        )
        self._gate = gate
        self._create_error = create_error

    async def export_chatbook(self, request_data):
        if self._gate is not None:
            assert self._gate.wait(
                timeout=_GATED_RELEASE_TIMEOUT_SECONDS
            ), "export gate never released"
        self.export_calls.append(dict(request_data))
        result = dict(self._export_result)
        result.setdefault("path", request_data.get("output_path"))
        return result

    async def create_chatbook(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._create_error is not None:
            raise self._create_error
        return {"chatbook_id": 1, **kwargs}


@pytest.mark.asyncio
async def test_library_shell_export_submit_single_flight_and_notifies_on_success(
    monkeypatch, tmp_path
):
    """Pressing Export: (1) enters ``running`` via a recompose (button
    disabled, quiet status line shown) -- acceptable per
    ``handle_library_export_submit``'s docstring, since the user's last
    action was clicking, not typing; (2) a second attempt -- both via the
    now-disabled button AND a direct re-entrant handler call -- is a no-op
    (single-flight: the button's own disabled gate PLUS the handler's own
    ``running`` guard, on top of the worker's
    ``group="library_export"``/``exclusive=True``); (3) the user can keep
    typing in the name field while the export is in flight; (4) on
    completion, a TARGETED update (not a recompose) clears ``running``,
    notifies (with the auto-included-characters suffix), and preserves the
    SAME ``Input`` instance + its typed text -- mirroring Task 2's own
    counts-landing regression pilot for the reverse transition.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-run-media")
    app.media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-run-ccn")
    app.chachanotes_db.add_conversation({"title": "Conv"})

    gate = threading.Event()
    service = _FakeLibraryExportService(
        export_result={
            "success": True,
            "message": "ok",
            "path": "",
            "dependency_info": {"auto_included": [1, 2, 3]},
        },
        gate=gate,
    )
    app.local_chatbook_service = service
    notified = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kwargs: notified.append((message, kwargs))
    )

    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")
        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed.")
        screen.refresh(recompose=True)
        await pilot.pause()

        screen._apply_library_export_destination(tmp_path / "out")
        await pilot.pause()

        submit = screen.query_one("#library-export-submit", Button)
        assert submit.disabled is False
        submit.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_export_running is True
        submit_running = screen.query_one("#library-export-submit", Button)
        assert submit_running.disabled is True
        status_widget = screen.query_one("#library-export-status-line", Static)
        assert status_widget.display is True
        assert str(status_widget.renderable) == "Exporting… (2 items)"

        # Second attempt #1: through the button itself -- Textual refuses
        # to dispatch Pressed for a disabled Button, so this is a no-op.
        submit_running.press()
        await pilot.pause()
        # Second attempt #2: a direct re-entrant handler call, bypassing
        # the button entirely -- the handler's own ``running`` guard blocks
        # it independently of the button's disabled state.
        screen.handle_library_export_submit(Mock())
        await pilot.pause()

        name_input = screen.query_one("#library-export-name", Input)
        name_input.focus()
        await pilot.pause()
        await pilot.press("!")
        assert name_input.value.endswith("!")

        gate.set()
        for _ in range(150):
            if screen._library_export_running is False:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export run never completed after gate release.")
        await pilot.pause()
        await pilot.pause()

        # Single-flight: exactly one export call landed despite two extra
        # press attempts while running.
        assert len(service.export_calls) == 1
        assert service.export_calls[0]["include_media"] is True
        assert service.export_calls[0]["output_path"] == str(tmp_path / "out.zip")
        assert len(service.create_calls) == 1  # zip succeeded -> registry recorded

        # Targeted update (not recompose): the SAME Input instance/typed
        # text survived the running -> done transition.
        assert screen.query_one("#library-export-name", Input) is name_input
        assert name_input.value.endswith("!")
        assert screen.query_one("#library-export-status-line", Static) is status_widget
        assert status_widget.display is False
        assert screen.query_one("#library-export-submit", Button) is submit_running
        assert submit_running.disabled is False
        assert screen._library_export_error == ""

        assert len(notified) == 1
        message, kwargs = notified[0]
        # Exact match (not just a prefix check): pins the REAL exported
        # path flowing all the way back into the notification, not a
        # stale/wrong/empty value.
        assert message == (
            f"Exported chatbook to {tmp_path / 'out.zip'} (3 characters auto-included)"
        )
        assert kwargs.get("severity") == "information"


@pytest.mark.asyncio
async def test_library_shell_export_submit_failure_shows_escaped_error_and_reenables_form(
    tmp_path,
):
    """A failed export renders the (escaped) error message via a TARGETED
    update, clears ``running``, and re-enables Export -- the destination
    and counts are still valid, so the user can retry without re-picking
    anything. The registry is never touched (zip-first, registry-only-on-
    success). Also asserts, symmetrically with the success-path pilot
    above, that the SAME ``Input``/``Button`` instances (and typed text)
    survive the running -> failed transition -- a targeted-update
    regression that broke ONLY the failure path (e.g. an accidental
    ``refresh(recompose=True)`` inside ``_apply_library_export_failure``)
    would otherwise go undetected."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-fail-media")
    app.media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-fail-ccn")

    gate = threading.Event()
    service = _FakeLibraryExportService(
        export_result={
            "success": False,
            "message": "Destination [bold]not[/bold] writable.",
            "path": "",
            "dependency_info": {},
        },
        gate=gate,
    )
    app.local_chatbook_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")
        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed.")
        screen.refresh(recompose=True)
        await pilot.pause()

        screen._apply_library_export_destination(tmp_path / "out")
        await pilot.pause()

        submit = screen.query_one("#library-export-submit", Button)
        submit.press()
        await pilot.pause()
        await pilot.pause()

        # Re-query AFTER the press-triggered recompose (the acceptable
        # recompose transition INTO ``running`` -- see
        # ``handle_library_export_submit``'s docstring): ``submit`` above
        # is now a stale, unmounted reference, so the identity check below
        # must be against the post-recompose instance, not the pre-press
        # one.
        submit_running = screen.query_one("#library-export-submit", Button)
        assert submit_running.disabled is True

        name_input = screen.query_one("#library-export-name", Input)
        name_input.focus()
        await pilot.pause()
        await pilot.press("?")
        assert name_input.value.endswith("?")

        gate.set()
        for _ in range(150):
            if screen._library_export_running is False:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export run never completed.")
        await pilot.pause()

        assert len(service.create_calls) == 0  # registry skipped on failure

        # Targeted update (not recompose): the SAME Input/Button instances
        # and the typed text survived the running -> failed transition.
        assert screen.query_one("#library-export-name", Input) is name_input
        assert name_input.value.endswith("?")
        submit_after = screen.query_one("#library-export-submit", Button)
        assert submit_after is submit_running

        error_widget = screen.query_one("#library-export-error-line", Static)
        assert error_widget.display is True
        assert (
            str(error_widget.renderable)
            == "Destination \\[bold]not\\[/bold] writable."
        )
        assert screen._library_export_error == "Destination \\[bold]not\\[/bold] writable."
        status_widget = screen.query_one("#library-export-status-line", Static)
        assert status_widget.display is False
        assert submit_after.disabled is False
        assert screen._library_export_running is False


@pytest.mark.asyncio
async def test_library_shell_export_orphaned_run_completion_cannot_corrupt_a_later_visit(
    tmp_path,
):
    """REGRESSION (code review finding, F4 Task 3): a real OS worker thread
    cannot be preempted mid-``asyncio.run`` by ``Worker.cancel()`` --
    navigating away from the Export canvas while a run is in flight resets
    ``_library_export_running`` for whatever the user does NEXT, but the
    abandoned worker keeps running regardless. Before the ``run_id``
    staleness guard, that orphaned worker's LATE completion would
    unconditionally stomp ``_library_export_running``/``_error``/
    ``_status`` (and the canvas DOM) out from under a completely different,
    later visit to the Export canvas -- silently re-enabling/disabling the
    button or showing a stray error for a run the user has long since
    forgotten about. This pilot: starts an export, navigates away mid-run,
    navigates BACK to a fresh Export visit, THEN releases the orphaned
    run -- and asserts the fresh visit's state is completely undisturbed by
    the orphaned run's completion (which still fires its notification --
    the export genuinely happened -- but nothing else)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-orphan-media")
    app.media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-orphan-ccn")
    app.chachanotes_db.add_conversation({"title": "Conv"})

    gate = threading.Event()
    service = _FakeLibraryExportService(
        export_result={
            "success": True,
            "message": "ok",
            "path": "",
            "dependency_info": {},
        },
        gate=gate,
    )
    app.local_chatbook_service = service
    notified = []
    app.notify = lambda message, **kwargs: notified.append((message, kwargs))

    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Visit 1: start an export, then navigate away while it's in flight.
        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")
        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed.")
        screen.refresh(recompose=True)
        await pilot.pause()

        screen._apply_library_export_destination(tmp_path / "orphaned_dest")
        await pilot.pause()

        screen.query_one("#library-export-submit", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen._library_export_running is True
        orphaned_run_id = screen._library_export_run_id

        screen.query_one(f"#library-row-{LIBRARY_ROW_BROWSE_CONVERSATIONS}").press()
        await pilot.pause()
        await pilot.pause()
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
        # The navigation reset ``running`` for THIS (non-Export) visit --
        # the orphaned worker is still executing regardless.
        assert screen._library_export_running is False
        assert screen._library_export_run_id != orphaned_run_id

        # Visit 2: back to a completely FRESH Export visit -- new scope/
        # counts/form, not touching the destination the orphaned run is
        # still writing to.
        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")
        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed on the fresh visit.")
        screen.refresh(recompose=True)
        await pilot.pause()

        fresh_run_id = screen._library_export_run_id
        assert fresh_run_id != orphaned_run_id
        assert screen._library_export_running is False
        assert screen._library_export_form["destination"] == ""
        fresh_submit = screen.query_one("#library-export-submit", Button)
        assert fresh_submit.disabled is True  # no destination chosen on this visit

        # NOW let the orphaned run finish.
        gate.set()
        for _ in range(150):
            if notified:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Orphaned export run's notification never landed.")
        await pilot.pause()
        await pilot.pause()

        # The orphaned export genuinely completed -- still notified...
        assert len(notified) == 1
        assert notified[0][0] == f"Exported chatbook to {tmp_path / 'orphaned_dest.zip'}"
        # ...but the CURRENT (fresh, second) visit is completely
        # undisturbed: still not running, still no error/status, the
        # SAME submit button instance, still correctly disabled (no
        # destination on THIS visit).
        assert screen._library_export_running is False
        assert screen._library_export_error == ""
        assert screen.query_one("#library-export-submit", Button) is fresh_submit
        assert fresh_submit.disabled is True
        assert screen.query_one("#library-export-status-line", Static).display is False
        assert screen.query_one("#library-export-error-line", Static).display is False


@pytest.mark.asyncio
async def test_library_shell_export_stale_run_completion_never_clears_a_newer_runs_flags(
    tmp_path,
):
    """REGRESSION (code review finding, F4 Task 3), narrower/deterministic
    variant of the pilot above: directly proves the ``run_id`` staleness
    guard, independent of any second real threaded worker's own timing.
    Starts run R1 (gated), then bumps ``_library_export_run_id`` in place
    (exactly what ``_reset_library_export_transient_state`` does when the
    user navigates away and back) while manually marking a DIFFERENT
    error/running state as though a newer run R2 now owns the canvas --
    then releases R1 and asserts its completion did NOT overwrite R2's
    state, even though it still fires its own notification.

    RED-verified: temporarily neutering ``_apply_library_export_success``'s
    ``if run_id != self._library_export_run_id: return`` guard (replacing
    it with ``if False:``) made this test fail exactly on the
    ``_library_export_running is True`` / ``_library_export_error ==
    "unrelated newer error"`` assertions below (the stale R1 completion
    flipped ``running`` back to ``False`` and clobbered the error text);
    reverted, test passes.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-stale-media")
    app.media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-stale-ccn")

    gate = threading.Event()
    service = _FakeLibraryExportService(
        export_result={
            "success": True,
            "message": "ok",
            "path": "",
            "dependency_info": {},
        },
        gate=gate,
    )
    app.local_chatbook_service = service
    notified = []
    app.notify = lambda message, **kwargs: notified.append((message, kwargs))

    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")
        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed.")
        screen.refresh(recompose=True)
        await pilot.pause()

        screen._apply_library_export_destination(tmp_path / "stale_dest")
        await pilot.pause()

        screen.query_one("#library-export-submit", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen._library_export_running is True

        # Simulate a newer run superseding this one (what
        # ``_reset_library_export_transient_state`` does on a real
        # navigate-away-and-back) WITHOUT touching the still-gated worker
        # thread itself, so this test's timing is fully deterministic.
        screen._library_export_run_id += 1
        screen._library_export_running = True  # pretend R2 is now running
        screen._library_export_error = "unrelated newer error"

        gate.set()
        for _ in range(150):
            if notified:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Stale export run's notification never landed.")
        await pilot.pause()

        # R1 genuinely finished -- still notified...
        assert len(notified) == 1
        # ...but R2's state (running=True, a different error string) must
        # survive completely untouched.
        assert screen._library_export_running is True
        assert screen._library_export_error == "unrelated newer error"


@pytest.mark.asyncio
async def test_library_shell_export_submit_missing_service_surfaces_error_and_reenables(
    tmp_path,
):
    """``app_instance.local_chatbook_service`` missing entirely (``None``)
    is a guarded failure inside the real worker thread, not a crash or a
    silently-stuck ``running`` state -- the closest-to-production shape of
    "the service wiring failed", covered nowhere else in this suite."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-noservice-media")
    app.media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-noservice-ccn")
    app.local_chatbook_service = None
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")
        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed.")
        screen.refresh(recompose=True)
        await pilot.pause()

        screen._apply_library_export_destination(tmp_path / "out")
        await pilot.pause()

        screen.query_one("#library-export-submit", Button).press()
        await pilot.pause()
        await pilot.pause()

        for _ in range(150):
            if screen._library_export_running is False:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export run never completed.")
        await pilot.pause()

        assert screen._library_export_error == "Chatbook export service unavailable."
        error_widget = screen.query_one("#library-export-error-line", Static)
        assert error_widget.display is True
        assert str(error_widget.renderable) == "Chatbook export service unavailable."
        assert screen.query_one("#library-export-submit", Button).disabled is False


@pytest.mark.asyncio
async def test_library_shell_export_registry_failure_warns_it_wont_appear_in_artifacts(
    tmp_path,
):
    """REVIEW FIX (F4 Task 3): a successful zip whose ``create_chatbook``
    registry step fails is still an overall SUCCESS (the artifact exists
    on disk -- zip-first semantics), but the user must be TOLD the
    bookkeeping failed, or the export silently never appears under
    Artifacts/Home with no explanation. Asserts BOTH notifications fire
    in order -- the primary success info, then the registry-failure
    warning -- and that the form still lands in the clean success state
    (no error line: the export itself did not fail)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-regfail-media")
    app.media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-regfail-ccn")

    service = _FakeLibraryExportService(
        export_result={
            "success": True,
            "message": "ok",
            "path": "",
            "dependency_info": {},
        },
        create_error=RuntimeError("registry disk full"),
    )
    app.local_chatbook_service = service
    notified = []
    app.notify = lambda message, **kwargs: notified.append((message, kwargs))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")
        for _ in range(150):
            if screen._library_export_counts is not None:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export counts never landed.")
        screen.refresh(recompose=True)
        await pilot.pause()

        screen._apply_library_export_destination(tmp_path / "out")
        await pilot.pause()

        screen.query_one("#library-export-submit", Button).press()
        await pilot.pause()
        await pilot.pause()

        for _ in range(150):
            if screen._library_export_running is False:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export run never completed.")
        await pilot.pause()

        # The registry step was genuinely attempted (zip-first ordering)...
        assert len(service.create_calls) == 1
        # ...and BOTH notifications fired, in order: the primary success
        # info, then the registry-failure warning.
        assert len(notified) == 2
        assert notified[0][0] == f"Exported chatbook to {tmp_path / 'out.zip'}"
        assert notified[0][1].get("severity") == "information"
        assert notified[1][0] == (
            "Export saved, but couldn't be registered — it won't appear under Artifacts."
        )
        assert notified[1][1].get("severity") == "warning"
        # Still an overall success: no error line, form back to clean state.
        assert screen._library_export_error == ""
        assert screen.query_one("#library-export-error-line", Static).display is False
        assert screen.query_one("#library-export-submit", Button).disabled is False
