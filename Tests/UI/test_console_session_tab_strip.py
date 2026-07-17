"""Console session tab strip: scroll overflow, streaming glyph, middle-click close."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.containers import HorizontalScroll
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Console.console_session_surface import ConsoleSessionSurface


class TabStripHost(App[None]):
    def __init__(self):
        super().__init__()
        self.pressed_ids: list[str] = []

    def compose(self):
        yield ConsoleSessionSurface(SimpleNamespace(notify=MagicMock()))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.pressed_ids.append(event.button.id or "")


def _sessions(count: int) -> list[ConsoleChatSession]:
    return [ConsoleChatSession(title=f"Session {i}", id=f"s{i}") for i in range(1, count + 1)]


@pytest.mark.asyncio
async def test_tab_strip_is_horizontally_scrollable() -> None:
    app = TabStripHost()
    async with app.run_test(size=(80, 24)):
        strip = app.query_one("#console-native-tab-strip", HorizontalScroll)
        assert isinstance(strip, HorizontalScroll)


@pytest.mark.asyncio
async def test_streaming_session_tab_shows_run_glyph() -> None:
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        sessions = _sessions(2)
        await surface.sync_sessions(sessions=sessions, active_session_id="s1")
        await pilot.pause()

        await surface.sync_sessions(
            sessions=sessions,
            active_session_id="s1",
            streaming_session_id="s2",
        )
        await pilot.pause()

        streaming_tab = app.query_one("#console-session-tab-s2", Button)
        idle_tab = app.query_one("#console-session-tab-s1", Button)
        assert str(streaming_tab.label).startswith("●")
        assert not str(idle_tab.label).startswith("●")
        assert "Run in progress" in (streaming_tab.tooltip or "")

        # Glyph clears when the run ends.
        await surface.sync_sessions(
            sessions=sessions,
            active_session_id="s1",
            streaming_session_id=None,
        )
        await pilot.pause()
        assert not str(streaming_tab.label).startswith("●")


@pytest.mark.asyncio
async def test_middle_click_on_tab_presses_its_close_button() -> None:
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        await surface.sync_sessions(sessions=_sessions(2), active_session_id="s1")
        await pilot.pause()

        clicked = await pilot.click("#console-session-tab-s2", button=2)
        await pilot.pause()

        assert clicked
        assert "console-close-session-tab-s2" in app.pressed_ids


@pytest.mark.asyncio
async def test_active_tab_is_scrolled_into_view() -> None:
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        sessions = _sessions(8)
        await surface.sync_sessions(sessions=sessions, active_session_id="s8")
        await pilot.pause()
        await pilot.pause()

        strip = app.query_one("#console-native-tab-strip", HorizontalScroll)
        assert strip.scroll_x > 0
