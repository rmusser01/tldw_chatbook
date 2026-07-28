"""Console session tab strip: scroll overflow, streaming glyph, middle-click close."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.containers import HorizontalScroll
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
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
    return [
        ConsoleChatSession(title=f"Session {i}", id=f"s{i}")
        for i in range(1, count + 1)
    ]


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
        # TASK-1233: the tooltip decodes the ● glyph in context instead of
        # a bare "Run in progress." fragment.
        assert "agent running" in (streaming_tab.tooltip or "")

        # Glyph clears when the run ends.
        await surface.sync_sessions(
            sessions=sessions,
            active_session_id="s1",
            streaming_session_id=None,
        )
        await pilot.pause()
        assert not str(streaming_tab.label).startswith("●")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("marker", "expected_fragment"),
    [
        (ConsoleRunMarker.NONE, None),
        (ConsoleRunMarker.RUNNING, "— agent running."),
        (ConsoleRunMarker.NEEDS_APPROVAL, "— waiting for approval."),
        (ConsoleRunMarker.FINISHED_OK, "— finished — unseen."),
        (ConsoleRunMarker.FINISHED_FAILED, "— failed — unseen."),
    ],
)
async def test_tab_tooltip_decodes_marker_meaning(
    marker: ConsoleRunMarker, expected_fragment: str | None
) -> None:
    """TASK-1233 AC#1: a tab's tooltip decodes its fleet run-marker glyph in
    context instead of leaving the reader to infer meaning from shape alone.

    ``ConsoleRunMarker.NONE`` is the pinned no-regression case: the tooltip
    must be byte-for-byte the pre-task-1233 copy (no dangling suffix, no
    stray em dash).
    """
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        sessions = _sessions(2)
        await surface.sync_sessions(
            sessions=sessions,
            active_session_id="s1",
            run_markers={"s2": marker},
        )
        await pilot.pause()

        tab = app.query_one("#console-session-tab-s2", Button)
        tooltip = tab.tooltip or ""
        if expected_fragment is None:
            assert tooltip == "Switch to Console tab: Session 2."
        else:
            assert expected_fragment in tooltip


@pytest.mark.asyncio
async def test_tab_tooltip_escapes_markup_in_title() -> None:
    """A session title containing bracket tokens must render literally in
    the tooltip, not be interpreted as Rich markup (Textual's Tooltip is a
    Static with markup parsing on)."""
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        sessions = [ConsoleChatSession(title="[red]Alarm[/red]", id="s1")]
        await surface.sync_sessions(
            sessions=sessions,
            active_session_id="s1",
            run_markers={"s1": ConsoleRunMarker.NEEDS_APPROVAL},
        )
        await pilot.pause()

        tab = app.query_one("#console-session-tab-s1", Button)
        assert tab.tooltip == (
            r"Active Console tab: \[red]Alarm\[/red] — waiting for approval."
            " Click again to rename."
        )


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
