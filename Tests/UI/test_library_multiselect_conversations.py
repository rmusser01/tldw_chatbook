import ast
import dataclasses
import time
from pathlib import Path
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
from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationMessageView,
    ConversationReaderState,
)
from tldw_chatbook.Widgets.Library.library_conversations_canvas import (
    LibraryConversationsCanvas,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _build_test_app,
    _active_library_screen,
    _conversation_records,
    _painted_text,
    _seed_conversations,
    _wait_for_condition,
    _two_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _fake(select_mode):
    loaded_message = ConversationMessageView(
        "message-1", "user", "now", "revision-1", 5, "hello"
    )
    fake = SimpleNamespace(
        _conversations_state=SimpleNamespace(
            freshness="fresh",
            select_mode=select_mode,
            row_selection=RowSelection("conversations"),
            reader_state=ConversationReaderState(
                selected_id="c1",
                selected_version=1,
                loaded_id="c1",
                loaded_version=1,
                loaded_generation=1,
                generation=1,
                messages=(loaded_message,),
                message_total=1,
                complete=True,
            ),
        ),
        _selected_conversation_id="",
        _library_selected_row_id="",
        _acknowledge_library_destination_change=lambda: None,
        _refreshed=0,
        _opened=[],
        _reader_synced=0,
        _reader_started=[],
        _sync_library_conversation_reader=lambda: None,
        _start_library_conversation_reader_selection=lambda conversation_id: None,
    )
    fake._library_conversation_loaded_preview_selected = lambda: (
        LibraryScreen._library_conversation_loaded_preview_selected(fake)
    )
    return fake


def test_convo_row_select_mode_toggles():
    fake = _fake(True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    ev = SimpleNamespace(
        button=SimpleNamespace(conversation_id="c5"), stop=lambda: None
    )
    LibraryScreen.handle_library_conversation_row(fake, ev)
    assert fake._conversations_state.row_selection.is_selected("c5")
    assert fake._selected_conversation_id == ""  # did NOT open/select the detail
    assert fake._refreshed == 1
    assert fake._conversations_state.reader_state.bulk_selected_count == 1
    assert fake._conversations_state.reader_state.loaded_id == "c1"
    assert fake._conversations_state.reader_state.messages[0].text == "hello"
    assert fake._conversations_state.reader_state.bulk_loaded_preview_selected is False
    assert fake._conversations_state.reader_state.loaded_actions_eligible is False


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
    fake._conversations_state.row_selection.select_all(["c2", "c1"])

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
            assert (
                pilot.app.query_one("#library-conversations-filter", Input).value
                == query
            )
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
        _conversations_state=SimpleNamespace(loading=False),
        _start_library_conversation_page_request=lambda page, query, **kwargs: (
            calls.append((page, query, kwargs))
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

        select_all_btn = screen.query_one("#library-conversations-select-all", Button)
        # Genuinely on-screen (within the simulated terminal's own
        # width), not pushed past the visible viewport the way the
        # unbounded Static's sibling Buttons were before the fix.
        assert 0 < select_all_btn.region.x < LIBRARY_TEST_SIZE[0]
        assert select_all_btn.region.width > 0


@pytest.mark.asyncio
async def test_zero_checked_select_mode_keeps_reader_read_only_until_done() -> None:
    app = _build_test_app()
    conversations = _two_conversations()
    _seed_conversations(app, conversations)
    # (task-32056) The header action is now also gated on workspace
    # eligibility, so link these rows into the active workspace -- this test
    # is about the bulk-selection fence, not the workspace one.
    registry = app.workspace_registry_service
    for record in conversations:
        registry.link_membership(
            registry.ensure_default_workspace().workspace_id,
            item_type="conversation",
            item_id=str(record["conversation_id"]),
            title=str(record["title"]),
        )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        loaded_message = ConversationMessageView(
            "message-loaded", "user", "now", "revision-loaded", 5, "hello"
        )
        screen._conversations_state.reader_state = ConversationReaderState(
            selected_id="chat-1",
            selected_version=1,
            loaded_id="chat-1",
            loaded_version=1,
            loaded_generation=5,
            generation=5,
            messages=(loaded_message,),
            message_total=1,
            complete=True,
        )
        screen._sync_library_conversation_reader()
        await pilot.pause()
        transcript = screen._conversations_state.reader_state.messages
        open_console = screen.query_one("#library-conversation-open-console", Button)
        assert not open_console.disabled

        screen.query_one("#library-conversations-select-toggle", Button).press()
        await pilot.pause()
        state = screen._conversations_state.reader_state
        assert state.bulk_active and state.bulk_selected_count == 0
        assert not state.loaded_actions_eligible and open_console.disabled
        assert state.messages == transcript

        screen.query_one("#library-conversation-row-0", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._conversations_state.row_selection.count == 1,
            message="Conversation checkbox did not settle.",
        )
        screen.query_one("#library-conversations-select-clear", Button).press()
        await pilot.pause()
        state = screen._conversations_state.reader_state
        assert state.bulk_active and state.bulk_selected_count == 0
        assert not state.loaded_actions_eligible and open_console.disabled
        assert state.messages == transcript

        screen.query_one("#library-conversations-select-toggle", Button).press()
        await pilot.pause()
        state = screen._conversations_state.reader_state
        assert not state.bulk_active and state.loaded_actions_eligible
        assert not open_console.disabled


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
            lambda: (
                screen._conversations_state.select_mode
                and any(
                    str(row.label).startswith("☐")
                    for row in screen.query("#library-conversation-row-0")
                )
            ),
            message="Conversation select-mode rows never recomposed.",
        )
        screen.query_one("#library-conversation-row-0", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._conversations_state.row_selection.count == 1,
            message="Conversation row was not selected.",
        )

        screen.query_one("#library-conversations-next", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._conversations_state.page == 2,
            message="Conversation page exit never applied.",
        )

        assert screen._conversations_state.select_mode is False
        assert screen._conversations_state.row_selection.count == 0
        assert screen._conversations_state.selection_notice == "Selection cleared."
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
        screen._conversations_state.freshness = "stale"
        screen._conversations_state.total_known = False
        screen._conversations_state.stale_copy = "Source changed again; try again."
        screen._conversations_state.error = ""
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


# ---------------------------------------------------------------------------
# task-31959: the select-mode "Export selected" label must not move when the
# first selection enables it. The "○ " disabled marker is part of the label,
# so crossing 0 -> 1 selected shifted the word two cells left, right under
# the row the user had just checked (PR J padded Media's bulk row only).
# ---------------------------------------------------------------------------


def _painted_word_column(host, button, word: str) -> int:
    """Absolute column where ``word`` is painted inside ``button`` (task-31959).

    Mirrors the Notes/Prompts sibling helper: measures the WORD, not the
    first painted glyph. task-32042 gave the Conversations select toolbar
    Media's full-label treatment, so "Export selected" no longer clips at
    the pane's width -- the invariant this pins is that the WORD "Export"
    holds its column across the disabled flip (the "○ " marker paints two
    cells to its left while disabled, the reserved pad holds it while
    enabled), exactly as Notes measures it.
    """
    painted = _painted_text(host, button.region)
    assert word in painted, (word, painted)
    return button.region.x + painted.index(word)


@pytest.mark.asyncio
async def test_conversations_export_label_holds_its_column_across_the_first_selection():
    """The painted "Export" word holds its column across the first selection.

    Drives the REAL screen, because the shift lives on the in-place
    ``_apply_library_row_toggle`` patch a row press takes -- not on
    compose. The word stays put ("○ " prefixes it while disabled, the
    reserved pad holds it while enabled).
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
        export = await _wait_for_selector(
            screen, pilot, "#library-conversations-export-selected"
        )
        assert export.disabled
        before = _painted_word_column(host, export, "Export")

        screen.query_one("#library-conversation-row-0", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query_one(
                "#library-conversations-export-selected", Button
            ).disabled,
            message="The row press never enabled Export selected.",
        )
        await pilot.pause()

        export = screen.query_one("#library-conversations-export-selected", Button)
        after = _painted_word_column(host, export, "Export")
        assert after == before, (before, after)


# ---------------------------------------------------------------------------
# task-32042 (critique #7 P1): the Conversations select toolbar was a degraded
# copy of Media's -- all four actions (count + Select all + Clear + Export
# selected) shared ONE ds-toolbar row at width:1fr, so at the narrow
# conversations list pane they split the row evenly and every label truncated
# ("Selec", "Exp") while "0 selected" wrapped. Media (task-30043) keeps each
# action at its content width across a multi-row toolbar; this pins the
# Conversations toolbar to that same treatment at both supported sizes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30)])
async def test_conversations_select_labels_paint_in_full_like_media(size):
    """Every Conversations select action paints its full word, unwrapped count."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        screen.query_one("#library-conversations-select-toggle", Button).press()

        count = await _wait_for_selector(
            screen, pilot, "#library-conversations-selected-count"
        )
        select_all = screen.query_one("#library-conversations-select-all", Button)
        clear = screen.query_one("#library-conversations-select-clear", Button)
        export = screen.query_one("#library-conversations-export-selected", Button)

        # Full, untruncated action labels (no "Selec"/"Exp" mid-word cuts).
        assert "Select all 2 shown" in _painted_text(host, select_all.region)
        assert "Clear" in _painted_text(host, clear.region)
        assert "Export selected" in _painted_text(host, export.region)

        # "0 selected" reads on one line, not the awkward wrap critique saw.
        painted_count = _painted_text(host, count.region)
        assert "0 selected" in painted_count
        assert count.region.height == 1, painted_count


# ---------------------------------------------------------------------------
# task-31945 AC#1/#3: a sibling canvas's row survives a fast second click.
#
# Textual's ``Button._on_click`` DROPS any click landing while the previous
# press's 0.2s ``-active`` flash is still on the widget (``if not
# self.has_class("-active"): self.press()``). PR F cleared that flash on the
# MEDIA rows only; the conversations/notes/prompts rows kept the default, so
# clicking ☐ and then the same row's title -- what a reviewer does -- lost
# the second click and the row read as a one-cell target.
#
# Driven with REAL mouse events: a ``Button.press()`` call bypasses
# ``_on_click`` entirely and can never see this bug.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_click_on_a_conversation_row_toggles_it_in_select_mode():
    """task-31945: marker cell and title cells are the same target."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        screen.query_one("#library-conversations-select-toggle", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._conversations_state.select_mode,
            message="Conversations select mode did not open.",
        )

        row = screen.query_one("#library-conversation-row-0", Button)
        marker_x = row.region.x
        # Past the marker cell and its padding: inside the title text.
        title_x = row.region.x + 6
        row_y = row.region.y
        assert title_x < row.region.right, row.region

        # Two assertions on purpose, because the wall-clock one cannot be
        # relied on alone: ``Pilot.click`` awaits its own pause, and on a
        # loaded box two of those already cost ~0.28s -- past the window,
        # where the behavioural assertion below would pass for the WRONG
        # reason (the flash simply expired). The ``-active`` check is the
        # load-independent core: that class is what makes
        # ``Button._on_click`` drop the next click, and with the flash off
        # it is never set at all. ``elapsed`` says whether the second
        # click really landed inside the window, and rides into the
        # failure message either way (PR L review item 5).
        started = time.monotonic()
        await pilot.click(offset=(marker_x, row_y))
        await pilot.pause()
        assert screen._conversations_state.row_selection.count == 1, "marker click"
        # The same widget instance survives the toggle (the label is
        # rewritten in place), which is exactly why the flash can swallow
        # the next click -- and what makes this assertion meaningful
        # rather than a check on a fresh widget that never flashed.
        assert screen.query_one("#library-conversation-row-0", Button) is row
        assert not row.has_class("-active"), (
            "the row still flashes -active after a press, so Button._on_click "
            "will drop the next click that lands on it"
        )

        # The title, immediately after -- the click the flash swallowed.
        await pilot.click(offset=(title_x, row_y))
        elapsed = time.monotonic() - started
        await pilot.pause()
        await pilot.pause()
        inside_window = elapsed < _ACTIVE_EFFECT_WINDOW
        assert screen._conversations_state.row_selection.count == 0, (
            "a title click right after a marker click did not toggle the row "
            f"(second click landed {elapsed:.3f}s after the first, "
            f"{'inside' if inside_window else 'OUTSIDE'} the "
            f"{_ACTIVE_EFFECT_WINDOW}s active-effect window)"
        )


#: Textual's ``Button.active_effect_duration`` default -- the window in
#: which ``Button._on_click`` drops the next click on the same widget.
_ACTIVE_EFFECT_WINDOW = 0.2

#: task-31945 AC#2: every Library LIST row is built by one helper, so the
#: press behaviour cannot drift back apart one canvas at a time. Keyed by
#: the row's DOM id prefix (the ``classes=`` argument is a variable at
#: three of these sites, the id is a literal f-string at all of them).
_LIBRARY_ROW_ID_PREFIXES = {
    "library-media-row-",
    "library-conversation-row-",
    "library-notes-row-",
    "library-notes-tree-note-",
    "library-notes-tree-folder-",
    "library-prompt-row-",
    "library-skill-row-",
}


def test_every_library_row_button_is_built_by_the_shared_helper():
    """task-31945 AC#2: one press behaviour, enforced at the source.

    A census, not a spot check: a NEW row canvas (or a revert of one of
    the converted sites to a bare ``Button``) fails here rather than
    silently shipping a row that swallows every fast second click. The
    expected-prefix set is asserted too, so a rename cannot make this
    pass vacuously by finding nothing.
    """
    canvas_dir = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Widgets" / "Library"
    found: dict[str, set[str]] = {}
    for path in sorted(canvas_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "id":
                    continue
                rendered = ast.unparse(keyword.value)
                prefix = next(
                    (p for p in _LIBRARY_ROW_ID_PREFIXES if p in rendered), None
                )
                if prefix is None:
                    continue
                callee = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", ""
                )
                if callee not in {"Button", "library_row_button"}:
                    # Containers keyed off the same id stem (the media
                    # rows' own scroll host) are not row buttons.
                    continue
                found.setdefault(prefix, set()).add(f"{path.name}:{callee}")

    assert set(found) == _LIBRARY_ROW_ID_PREFIXES, (
        "row-button census found the wrong set of construction sites "
        f"(a rename or a new canvas?): {sorted(found)}"
    )
    offenders = {
        prefix: sorted(sites)
        for prefix, sites in found.items()
        if any(not site.endswith(":library_row_button") for site in sites)
    }
    assert not offenders, (
        "these row buttons bypass library_row_button(), so Textual's 0.2s "
        f"active-effect flash will swallow their next click: {offenders}"
    )
