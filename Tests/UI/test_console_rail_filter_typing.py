"""TASK-33622.6: the rail's 'Filter visible titles' input keeps what is typed.

The Conversations tray rebuilt its whole subtree -- the filter input included
-- on every sync that changed anything, and a search changes the rows, so
every debounced search replaced the input under the user's fingers. The
replacement was seeded with the stripped, lowercased query ('OK' came back as
'ok'); focus went to whatever Textual's reset picked (the out-of-band 'New
conversation' button, the section toggle), or back to the new input with
its text selected. The next keys then replaced the query, were swallowed by
a button, or reached a screen binding: typing 'Say' opened Trace.

These tests drive the real ChatScreen rail with real Pilot key presses and
cross the 0.2 s debounce on purpose. Each asserts the exact text, that the
SAME input instance still holds keyboard focus, and that no screen action
fired.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input
from textual.widgets.input import Selection

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)

pytestmark = pytest.mark.bootstrap_profile

APP_SIZE = (160, 48)
SEARCH = "#console-workspace-conversation-search"
STATUS = "#console-workspace-conversation-search-status"
# Longer than the rail search's 0.2 s debounce, so the debounced search (and
# the tray rebuild its results cause) lands between two keys.
PAST_DEBOUNCE_S = 0.45


def _seed_conversations(app, tmp_path) -> None:
    """Give the app real conversation storage with a few titled chats."""
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "rail-filter.db", client_id="rail-filter")
    app.chachanotes_db = db
    service = app.local_chat_conversation_service = ChatConversationService(db)
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=service, server_service=None
    )
    for title, content in (
        ("OK 9 rollout plan", "first message"),
        ("Okra recipes", "second message"),
        ("Say hello draft", "third message"),
        ("Quarterly notes", "the zebra crossing outside"),
    ):
        conversation_id = service.create_conversation(title=title)
        db.add_message(
            {"conversation_id": conversation_id, "sender": "user", "content": content}
        )


class _ScreenActionSpy:
    """Count Trace opens and new-chat creations while the user types."""

    def __init__(self) -> None:
        self.trace_opens = 0
        self._original = ChatScreen.action_open_trajectory_view

    def __enter__(self) -> "_ScreenActionSpy":
        spy = self
        original = self._original

        def counting(screen):
            spy.trace_opens += 1
            return original(screen)

        ChatScreen.action_open_trajectory_view = counting
        return self

    def __exit__(self, *_exc) -> None:
        ChatScreen.action_open_trajectory_view = self._original


async def _focused_filter(host, pilot):
    """Return the settled Console screen and its focused rail filter input."""
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, SEARCH)
    for _ in range(6):
        await pilot.pause()
    search = console.query_one(SEARCH, Input)
    search.focus()
    for _ in range(40):
        if console.focused is search:
            break
        await pilot.pause(0.05)
    assert console.focused is search, getattr(console.focused, "id", None)
    return console, search


async def _wait_for_search_to_settle(console, pilot, query: str) -> bool:
    """Wait (bounded) for the debounced search for ``query`` to reach the tray.

    Returns whether it did. Callers assert the input's text, identity and
    focus FIRST, so a regression fails on what the user sees rather than on
    this wait (a clobbered query never settles at all).
    """

    def settled() -> bool:
        lane = console._workspace._flat_conversation_search
        tray = next(
            (
                ancestor
                for ancestor in console.query_one(SEARCH).ancestors
                if isinstance(ancestor, ConsoleWorkspaceContextTray)
            ),
            None,
        )
        browser = getattr(getattr(tray, "state", None), "conversation_browser", None)
        # (The debounce timer attribute is not reset after it fires, so it
        # says nothing about whether the search ran.)
        return (
            lane.settled_query == query
            and tray is not None
            and not tray.recompose_in_flight
            and browser is not None
            and browser.query.strip().lower() == query.strip().lower()
        )

    for _ in range(80):
        if settled():
            break
        await pilot.pause(0.05)
    # A few more turns so the rebuild those results scheduled has run.
    for _ in range(4):
        await pilot.pause()
    return settled()


@pytest.mark.asyncio
async def test_typing_across_the_debounce_keeps_case_text_and_focus(tmp_path):
    """AC#1/#2/#5: 'OK', a pause past the debounce, then ' 9' reads 'OK 9'."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    _seed_conversations(app, tmp_path)
    host = ConsoleHarness(app)

    with _ScreenActionSpy() as spy:
        async with host.run_test(size=APP_SIZE) as pilot:
            console, search = await _focused_filter(host, pilot)
            stack = list(host.screen_stack)

            await pilot.press("O", "K")
            await pilot.pause(PAST_DEBOUNCE_S)
            ok_settled = await _wait_for_search_to_settle(console, pilot, "OK")

            live = console.query_one(SEARCH, Input)
            assert live.value == "OK"
            assert console.focused is live, getattr(console.focused, "id", None)
            assert live is search, "the search rebuilt the filter input"
            # Nothing selected: the next key appends instead of replacing.
            assert live.selection.is_empty
            assert live.cursor_position == len("OK")
            assert ok_settled, "the search for 'OK' never reached the tray"

            await pilot.press("space", "9")
            await pilot.pause(PAST_DEBOUNCE_S)
            ok9_settled = await _wait_for_search_to_settle(console, pilot, "OK 9")

            live = console.query_one(SEARCH, Input)
            assert live.value == "OK 9"
            assert console.focused is live, getattr(console.focused, "id", None)
            assert live is search, "the search rebuilt the filter input"
            assert list(host.screen_stack) == stack
            assert spy.trace_opens == 0
            assert ok9_settled, "the search for 'OK 9' never reached the tray"


@pytest.mark.asyncio
async def test_binding_letters_typed_around_rebuilds_stay_in_the_filter(tmp_path):
    """AC#3/#5: 'y' (Trace) and space typed into the filter never act on the screen.

    The keys are paced so the debounced search and its rebuild land between
    them at several offsets, not just once.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    _seed_conversations(app, tmp_path)
    host = ConsoleHarness(app)
    typed = ""

    with _ScreenActionSpy() as spy:
        async with host.run_test(size=APP_SIZE) as pilot:
            console, search = await _focused_filter(host, pilot)
            stack = list(host.screen_stack)
            sessions_before = len(console._ensure_console_chat_store().sessions())

            for key, char, pause_after in (
                ("S", "S", 0.0),
                ("a", "a", PAST_DEBOUNCE_S),
                ("y", "y", 0.25),
                ("space", " ", 0.21),
                ("y", "y", 0.05),
                ("e", "e", PAST_DEBOUNCE_S),
                ("s", "s", 0.0),
            ):
                await pilot.press(key)
                typed += char
                if pause_after:
                    await pilot.pause(pause_after)
                live = console.query_one(SEARCH, Input)
                assert live.value == typed
                assert console.focused is live, getattr(console.focused, "id", None)
                assert live is search, "the search rebuilt the filter input"

            say_settled = await _wait_for_search_to_settle(console, pilot, "Say yes")
            assert search.value == "Say yes"
            assert console.focused is search
            assert say_settled, "the search for 'Say yes' never reached the tray"
            assert spy.trace_opens == 0
            assert list(host.screen_stack) == stack
            # A space that reached the 'New conversation' button would have
            # created a chat.
            assert len(console._ensure_console_chat_store().sessions()) == (
                sessions_before
            )


@pytest.mark.asyncio
async def test_clear_empties_the_filter_the_rebuild_kept(tmp_path):
    """Clear still empties the field now that a rebuild keeps the input.

    The rebuild never writes state into the kept input, so without the
    tray's own Clear handling the search state would reset while the field
    still showed the old text.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    _seed_conversations(app, tmp_path)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, search = await _focused_filter(host, pilot)
        await pilot.press("O", "K")
        await pilot.pause(PAST_DEBOUNCE_S)
        assert await _wait_for_search_to_settle(console, pilot, "OK")

        await pilot.click("#console-workspace-conversation-search-clear")
        for _ in range(40):
            if not console._console_conversation_browser_query:
                break
            await pilot.pause(0.05)
        for _ in range(6):
            await pilot.pause()

        live = console.query_one(SEARCH, Input)
        assert live.value == ""
        assert console._console_conversation_browser_query == ""
        assert console.focused is live, getattr(console.focused, "id", None)
        assert not list(console.query("#console-workspace-conversation-search-status"))


def _shown(console) -> tuple[str, list[str]]:
    """The status line and the conversation rows the rail paints."""
    statuses = list(console.query(STATUS))
    status = str(statuses[0].render()) if statuses else ""
    rows = [
        str(row.label).splitlines()[0]
        for row in console.query(".console-workspace-conversation-row")
    ]
    return status, rows


@pytest.mark.asyncio
async def test_the_match_count_and_rows_follow_one_title_rule(tmp_path):
    """AC#4 (owner decision 2026-10-27): the filter counts what it shows -- titles.

    'Quarterly notes' mentions zebra only in a message. The database used to
    count it while the rail hid it: '1 match. Showing 0 of 1' over an empty
    list. Message-content search stays in 'Search all chats…' / Ctrl+K.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    _seed_conversations(app, tmp_path)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, search = await _focused_filter(host, pilot)
        seen = {}
        for query in ("zebra", "notes", "ok"):
            if search.value:
                await pilot.click("#console-workspace-conversation-search-clear")
                for _ in range(20):
                    await pilot.pause(0.05)
                    if not search.value and not console.query(STATUS):
                        break
                search.focus()
                await pilot.pause()
            await pilot.press(*query)
            await pilot.pause(PAST_DEBOUNCE_S)
            assert await _wait_for_search_to_settle(console, pilot, query)
            seen[query] = _shown(console)

        assert seen["zebra"] == ("0 matches", [])
        status, rows = seen["notes"]
        assert status == "1 match"
        assert len(rows) == 1 and "Quarterly notes" in rows[0]
        status, rows = seen["ok"]
        assert status == "2 matches"
        assert len(rows) == 2
        assert any("OK 9 rollout plan" in row for row in rows)
        assert any("Okra recipes" in row for row in rows)


def _unpainted_rows(console) -> list[str]:
    """Matched rows whose cell is painted by something else (clipped/covered)."""
    hidden = []
    for row in console.query(".console-workspace-conversation-row"):
        region = row.region
        try:
            hit, _ = console.get_widget_at(region.x + 1, region.y)
        except Exception:  # noqa: BLE001 - off-screen counts as unpainted
            hit = None
        if hit is not row and row not in getattr(hit, "ancestors", ()):
            hidden.append(f"{str(row.label).splitlines()[0]!r} under {getattr(hit, 'id', hit)}")
    return hidden


@pytest.mark.asyncio
async def test_every_counted_row_is_painted_after_each_filter(tmp_path):
    """AC#4: the rows the count promises are ones the user can SEE.

    Keeping the filter row across a rebuild leaves it laid out while the new
    children are not yet, and a height fit that ran in that window pinned the
    tray short (the TASK-37160 mechanism): the matched row sat under the next
    rail section's header while the status said '1 match'.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    _seed_conversations(app, tmp_path)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, search = await _focused_filter(host, pilot)
        typed = ""
        for chunk in ("OK", " 9", "", "Q", "u", "a"):
            if chunk == "":
                await pilot.click("#console-workspace-conversation-search-clear")
                for _ in range(20):
                    await pilot.pause(0.05)
                    if not search.value and not console.query(STATUS):
                        break
                search.focus()
                await pilot.pause()
                typed = ""
                continue
            await pilot.press(*("space" if char == " " else char for char in chunk))
            typed += chunk
            await pilot.pause(PAST_DEBOUNCE_S)
            assert await _wait_for_search_to_settle(console, pilot, typed)
            for _ in range(6):
                await pilot.pause()
            status, rows = _shown(console)
            assert rows, f"{typed!r}: no rows for {status!r}"
            assert _unpainted_rows(console) == [], f"{typed!r}: {status!r}"


# ---------------------------------------------------------------------------
# The tray itself: what a rebuild may and may not do to a kept filter row.
# ---------------------------------------------------------------------------


def _tray_state(query: str, titles: tuple[str, ...], *, browser: bool = True):
    """A Conversations-tray snapshot filtered by ``query`` over ``titles``."""
    from dataclasses import replace

    from Tests.UI.test_console_rail_reconciliation import _workspace_state
    from tldw_chatbook.Workspaces.conversation_browser_state import (
        ConsoleConversationBrowserInputRow,
        build_console_conversation_browser_state,
    )

    rows = tuple(
        ConsoleConversationBrowserInputRow(
            row_key=f"c{index}",
            conversation_id=f"c{index}",
            native_session_id=None,
            title=title,
            scope_type="global",
            workspace_id=None,
            workspace_label="Chats",
            updated_sort=f"2026-09-{28 - index:02d}T00:00:00",
        )
        for index, title in enumerate(titles)
    )
    return replace(
        _workspace_state(),
        conversation_browser=(
            build_console_conversation_browser_state(
                rows=rows,
                active_workspace_id=None,
                group_collapse_preferences={"section:chats": False},
                query=query,
            )
            if browser
            else None
        ),
    )


def _tray_app(state):
    from textual.app import App, ComposeResult

    class _TrayApp(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.changed: list[str] = []

        def compose(self) -> ComposeResult:
            yield ConsoleWorkspaceContextTray(
                state,
                show_heading=False,
                content="conversations",
                id="console-workspace-context",
            )

        def on_input_changed(self, event: Input.Changed) -> None:
            self.changed.append(str(event.value))

    return _TrayApp()


async def _settled_tray(app, pilot) -> ConsoleWorkspaceContextTray:
    """The tray once its first-width relabel rebuild (after mount) has run.

    Grab the input only after this: a reference taken before a rebuild that
    replaces it is a silent key sink.
    """
    tray = app.query_one(ConsoleWorkspaceContextTray)
    for _ in range(40):
        await pilot.pause()
        if tray._row_width_measured and not tray.recompose_in_flight:
            break
    for _ in range(3):
        await pilot.pause()
    return tray


async def _rebuild(tray, state, pilot) -> None:
    tray.sync_state(state)
    for _ in range(40):
        await pilot.pause()
        if not tray.recompose_in_flight and tray.state is state:
            break
    for _ in range(3):
        await pilot.pause()


_TITLES = ("OK 9 rollout plan", "Okra recipes", "Quarterly notes")


@pytest.mark.asyncio
async def test_a_rebuild_never_writes_state_into_the_kept_input():
    """AC#2: typing ahead of a slower sync is never overwritten.

    A sync built before the latest keystrokes carries an older query and
    different rows. The rebuild changes the rows; the input keeps its
    instance, the user's text, its cursor and its focus, and posts nothing.
    """
    app = _tray_app(_tray_state("O", _TITLES))
    async with app.run_test(size=(60, 30)) as pilot:
        tray = await _settled_tray(app, pilot)
        search = app.query_one(SEARCH, Input)
        search.focus()
        await pilot.pause()
        # Type at the end of 'O' whatever focusing did to the selection: this
        # test is about the rebuild, not about select-on-focus.
        search.selection = Selection.cursor(len(search.value))
        await pilot.press("K", "space", "9")
        assert search.value == "OK 9"
        cursor = search.cursor_position
        app.changed.clear()

        await _rebuild(tray, _tray_state("OK", _TITLES[:1]), pilot)

        live = app.query_one(SEARCH, Input)
        assert live is search
        assert live.value == "OK 9"
        assert live.cursor_position == cursor
        assert live.selection.is_empty
        assert app.focused is live
        assert app.changed == []
        rows = [str(row.row_key) for row in app.query(".console-workspace-conversation-row")]
        assert rows == ["c0"], "the rebuild did not apply the new rows"


@pytest.mark.asyncio
async def test_the_kept_row_sits_where_a_fresh_compose_puts_it():
    """The rebuild mounts the new children around the kept row, in order."""
    state = _tray_state("ok", _TITLES)
    app = _tray_app(_tray_state("o", _TITLES))
    async with app.run_test(size=(60, 30)) as pilot:
        tray = await _settled_tray(app, pilot)
        await _rebuild(tray, state, pilot)
        rebuilt = [child.id for child in tray.children]

    fresh_app = _tray_app(state)
    async with fresh_app.run_test(size=(60, 30)) as pilot:
        fresh = [child.id for child in (await _settled_tray(fresh_app, pilot)).children]

    assert "console-workspace-conversation-search-row" in rebuilt
    assert rebuilt == fresh


@pytest.mark.asyncio
async def test_clear_empties_the_kept_input_without_an_echo():
    """Clear is the one programmatic write: it empties the kept input silently."""
    app = _tray_app(_tray_state("ok", _TITLES))
    async with app.run_test(size=(60, 30)) as pilot:
        tray = await _settled_tray(app, pilot)
        search = app.query_one(SEARCH, Input)
        assert search.value == "ok"
        app.changed.clear()

        # The bare harness has no rail CSS, so press rather than click.
        app.query_one("#console-workspace-conversation-search-clear").press()
        await pilot.pause()
        assert search.value == ""
        assert app.changed == [], "Clear posted a Changed echo"

        await _rebuild(tray, _tray_state("", _TITLES), pilot)
        assert app.query_one(SEARCH, Input) is search
        assert search.value == ""


@pytest.mark.asyncio
async def test_a_state_without_a_browser_drops_the_kept_row():
    """A rebuild whose state renders no browser removes the filter row too."""
    app = _tray_app(_tray_state("ok", _TITLES))
    async with app.run_test(size=(60, 30)) as pilot:
        tray = await _settled_tray(app, pilot)
        await _rebuild(tray, _tray_state("", _TITLES, browser=False), pilot)
        assert not app.query(SEARCH)
        assert not app.query("#console-workspace-conversation-search-row")
