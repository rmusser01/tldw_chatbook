import dataclasses
from types import SimpleNamespace

import pytest
# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Input, Static

from tldw_chatbook.UI.Screens.library_screen import (
    LibraryScreen,
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
)
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_conversations_state import (
    LibraryConversationsCanvasState,
    LibraryConversationRow,
    build_library_conversations_state,
)
from tldw_chatbook.Widgets.Library.library_conversations_canvas import (
    LibraryConversationsCanvas,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _conversation_records,
    _seed_conversations,
    _wait_for_condition,
    _two_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _fake(select_mode):
    return SimpleNamespace(
        _library_conversation_freshness="fresh",
        _library_conversations_select_mode=select_mode,
        _library_conversations_row_selection=RowSelection("conversations"),
        _selected_conversation_id="",
        _library_selected_row_id="",
        _refreshed=0,
        _opened=[],
    )


def test_convo_row_select_mode_toggles():
    fake = _fake(True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    ev = SimpleNamespace(
        button=SimpleNamespace(conversation_id="c5"), stop=lambda: None
    )
    LibraryScreen.handle_library_conversation_row(fake, ev)
    assert fake._library_conversations_row_selection.is_selected("c5")
    assert fake._selected_conversation_id == ""  # did NOT open/select the detail
    assert fake._refreshed == 1


def test_convo_row_normal_mode_selects():
    fake = _fake(False)
    fake.refresh = lambda **k: None
    ev = SimpleNamespace(
        button=SimpleNamespace(conversation_id="c5"), stop=lambda: None
    )
    LibraryScreen.handle_library_conversation_row(fake, ev)
    assert fake._selected_conversation_id == "c5"
    assert fake._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS


@pytest.mark.asyncio
async def test_convo_export_selected_scope():
    fake = _fake(True)
    fake._library_conversations_row_selection.select_all(["c2", "c1"])

    async def _open(s):
        fake._opened.append(s)

    fake._open_library_export_canvas = _open
    await LibraryScreen.handle_library_conversations_export_selected(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert fake._opened == [ExportScope(kind="conversations", ids=("c1", "c2"))]


def _select_mode_canvas_state() -> LibraryConversationsCanvasState:
    rows = (
        LibraryConversationRow(
            conversation_id="c1",
            title="First conversation",
            secondary="today",
            checked=False,
        ),
        LibraryConversationRow(
            conversation_id="c2",
            title="Second conversation",
            secondary="today",
            checked=False,
        ),
    )
    return LibraryConversationsCanvasState(
        rows=rows,
        query="",
        status_copy="",
        empty_copy="No conversations in your Library yet.",
        selected_id="",
        preview_lines=(),
        select_mode=True,
        selected_count=0,
    )


class _ConversationsCanvasApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryConversationsCanvas(
            canvas=_select_mode_canvas_state(), id="library-conversations-canvas"
        )


@pytest.mark.asyncio
async def test_canvas_select_mode_renders_action_row_and_disables_export():
    app = _ConversationsCanvasApp()
    async with app.run_test() as pilot:
        select_all_btn = pilot.app.query_one(
            "#library-conversations-select-all", Button
        )
        assert select_all_btn is not None
        assert "2 shown" in str(select_all_btn.label)
        export_selected_btn = pilot.app.query_one(
            "#library-conversations-export-selected", Button
        )
        assert export_selected_btn.disabled is True


class _ConversationsCanvasSelectedApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryConversationsCanvas(
            canvas=dataclasses.replace(_select_mode_canvas_state(), selected_count=1),
            id="library-conversations-canvas",
        )


class _FreshEmptyConversationsCanvasApp(ConsolidatedCSSApp):
    def __init__(
        self, *, query: str, loading: bool = False, error_copy: str = ""
    ) -> None:
        super().__init__()
        self._query = query
        self.loading = loading
        self.error_copy = error_copy

    def compose(self):
        yield LibraryConversationsCanvas(
            build_library_conversations_state(
                (),
                query=self._query,
                total_count=0,
                total_known=True,
                freshness="fresh",
                loading=self.loading,
                error_copy=self.error_copy,
            ),
            id="library-conversations-canvas",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "action_id", "label", "filter_visible"),
    [
        ("", "library-conversations-empty-console", "Start in Console", False),
        ("needle", "library-conversations-empty-clear-filter", "Clear filter", True),
    ],
    ids=["source-empty", "filtered-zero"],
)
async def test_conversations_fresh_zero_distills_to_one_recovery_action(
    query: str, action_id: str, label: str, filter_visible: bool
):
    app = _FreshEmptyConversationsCanvasApp(query=query)

    async with app.run_test() as pilot:
        action = pilot.app.query_one(f"#{action_id}", Button)
        assert str(action.label) == label
        assert action.disabled is False
        assert action in pilot.app.screen.focus_chain
        assert bool(pilot.app.query("#library-conversations-filter")) is filter_visible
        if filter_visible:
            assert pilot.app.query_one("#library-conversations-filter", Input).value == query
        assert "No conversations" in str(
            pilot.app.query_one("#library-conversations-status", Static).renderable
        )
        assert not pilot.app.query("#library-conversations-pager")
        assert not pilot.app.query("#library-conversations-select-toggle")
        assert not pilot.app.query("#library-conversations-export")
        assert len(pilot.app.query(".library-canvas-action")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("loading", "error_copy", "retry_visible"),
    [
        (True, "", False),
        (False, "Filter wasn't applied; showing previous results.", True),
    ],
    ids=["loading", "error"],
)
async def test_conversations_retained_zero_keeps_request_recovery_authority(
    loading: bool, error_copy: str, retry_visible: bool
):
    app = _FreshEmptyConversationsCanvasApp(
        query="",
        loading=loading,
        error_copy=error_copy,
    )

    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-conversations-pager")
        assert bool(pilot.app.query("#library-conversations-retry")) is retry_visible
        assert not pilot.app.query("#library-conversations-empty-console")


def test_conversations_empty_console_uses_existing_live_work_route():
    calls = []
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            open_console_for_live_work=lambda **kwargs: calls.append(kwargs)
        )
    )

    LibraryScreen.handle_library_conversations_empty_console(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert calls == [
        {
            "source": "library-conversations-empty",
            "title": "Start a conversation",
            "action_label": "Start in Console",
        }
    ]


def test_conversations_empty_clear_filter_requests_unfiltered_page_one():
    calls = []
    fake = SimpleNamespace(
        _library_conversation_loading=False,
        _start_library_conversation_page_request=lambda page, query, **kwargs: calls.append(
            (page, query, kwargs)
        ),
    )

    LibraryScreen.handle_library_conversations_empty_clear_filter(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert calls == [(1, "", {"refocus_filter": True})]


@pytest.mark.asyncio
async def test_export_selected_tooltip_follows_its_disabled_state():
    """F-018: "Export selected" disabled with zero selection says WHY;
    with a selection the tooltip describes the action."""
    app = _ConversationsCanvasApp()
    async with app.run_test() as pilot:
        export_btn = pilot.app.query_one(
            "#library-conversations-export-selected", Button
        )
        assert export_btn.disabled is True
        assert "select" in str(export_btn.tooltip).lower()

    app_with_selection = _ConversationsCanvasSelectedApp()
    async with app_with_selection.run_test() as pilot:
        export_btn = pilot.app.query_one(
            "#library-conversations-export-selected", Button
        )
        assert export_btn.disabled is False
        assert "export" in str(export_btn.tooltip).lower()


@pytest.mark.asyncio
async def test_conversations_toolbar_count_static_stays_bounded_width_with_real_css():
    """task-2853 review round 2: the SAME unbounded-width defect proved
    live in the Media canvas's identical "N selected" counter (see
    library_media_canvas.py's compose()) also affects this canvas's own
    counter -- both are fixed by the SAME shared ``library-toolbar-count``
    CSS class (css/components/_agentic_terminal.tcss's ``width: auto``),
    not a per-canvas Python one-off, so one declaration covers both.

    Mounts the REAL ``LibraryScreen`` with the REAL generated CSS bundle
    (``LibraryHarness``, not a bare canvas-only ``App`` the way the other
    tests in this file do) so the assertions below reflect the actual
    cascade a live terminal sees -- a bare-App mount never reproduced this
    bug (Button's own ``DEFAULT_CSS`` alone was enough to keep it
    visible), only the full app bundle's stylesheet did. Before the fix
    this Static's rendered region width was ~1700 columns on a
    170-column simulated terminal and every sibling Button was pushed
    entirely off-screen (present in the DOM, invisible on screen); this
    pins both symptoms as regression guards.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")

        screen.query_one("#library-conversations-select-toggle", Button).press()
        count_static = await _wait_for_selector(
            screen, pilot, "#library-conversations-selected-count"
        )

        # Bounded to its own content ("0 selected" is 10 characters) --
        # NOT the ~1700-column runaway the unbounded-width bug produced.
        assert count_static.region.width < 30

        select_all_btn = screen.query_one(
            "#library-conversations-select-all", Button
        )
        # Genuinely on-screen (within the simulated terminal's own
        # width), not pushed past the visible viewport the way the
        # unbounded Static's sibling Buttons were before the fix.
        assert 0 < select_all_btn.region.x < LIBRARY_TEST_SIZE[0]
        assert select_all_btn.region.width > 0


@pytest.mark.asyncio
async def test_library_conversation_selection_clears_on_page_exit_and_cannot_export():
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(45))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-19")
        screen.query_one("#library-conversations-select-toggle", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_conversations_select_mode
            and any(
                str(row.label).startswith("☐")
                for row in screen.query("#library-conversation-row-0")
            ),
            message="Conversation select-mode rows never recomposed.",
        )
        screen.query_one("#library-conversation-row-0", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_conversations_row_selection.count == 1,
            message="Conversation row was not selected.",
        )

        screen.query_one("#library-conversations-next", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_conversation_page == 2,
            message="Conversation page exit never applied.",
        )

        assert screen._library_conversations_select_mode is False
        assert screen._library_conversations_row_selection.count == 0
        assert screen._library_conversation_selection_notice == "Selection cleared."
        assert (
            screen._build_library_conversations_state().selection_notice
            == "Selection cleared."
        )
        opened = []

        async def record_open(scope):
            opened.append(scope)

        screen._open_library_export_canvas = record_open
        await screen.handle_library_conversations_export_selected(
            SimpleNamespace(stop=lambda: None)
        )
        assert opened == []


@pytest.mark.asyncio
async def test_library_conversation_stale_state_disables_actions_but_allows_recovery():
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(25))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-19")
        selected_before = screen._selected_conversation_id
        screen._library_conversation_freshness = "stale"
        screen._library_conversation_total_known = False
        screen._library_conversation_stale_copy = "Source changed again; try again."
        screen._library_conversation_error = ""
        screen._sync_library_conversation_canvas()
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-conversation-row-0", Button).disabled,
            message="Stale Conversation row never disabled.",
        )

        state = screen._build_library_conversations_state()
        assert state.actions_disabled is True
        assert state.pager is not None and state.pager.retry_visible is True
        assert screen.query_one("#library-conversations-previous", Button).disabled
        assert screen.query_one("#library-conversations-next", Button).disabled
        assert screen.query_one("#library-conversations-select-toggle", Button).disabled
        assert screen.query_one("#library-conversations-filter").disabled is False

        event = SimpleNamespace(
            button=SimpleNamespace(conversation_id="chat-002"),
            stop=lambda: None,
        )
        LibraryScreen.handle_library_conversation_row(screen, event)
        assert screen._selected_conversation_id == selected_before
