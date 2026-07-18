"""Console left-rail staged context tray tests."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleStagedContextState,
)
from tldw_chatbook.Widgets.Console.console_staged_context import (
    ConsoleStagedContextTray,
)


@pytest.mark.asyncio
async def test_staged_context_renders_source_count() -> None:
    """The tray header shows the number of staged sources."""

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(
                    heading="Context",
                    summary="",
                    rows=(
                        ConsoleDisplayRow("Source", "readme.md", status="ready"),
                    ),
                )
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        count = tray.query_one("#console-staged-context-count", Static)
        assert str(count.renderable) == "1"


@pytest.mark.asyncio
async def test_staged_context_omits_attach_button() -> None:
    """The redesign removes the Attach button from the empty tray."""

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(heading="Context", summary="", rows=())
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        assert not list(tray.query("#console-staged-context-attach"))


@pytest.mark.asyncio
async def test_staged_context_empty_shows_guidance() -> None:
    """An empty tray prompts the user to stage sources from the Library."""

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(heading="Context", summary="", rows=())
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        empty = tray.query_one("#console-staged-context-empty", Static)
        assert "Stage sources from Library" in str(empty.renderable)


@pytest.mark.asyncio
async def test_staged_context_row_renders_name_and_normalized_status() -> None:
    """Each source renders its value and a normalized status line."""

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(
                    heading="Context",
                    summary="",
                    rows=(
                        ConsoleDisplayRow("Source", "readme.md", status="available"),
                        ConsoleDisplayRow("Source", "missing.txt", status="missing"),
                    ),
                )
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        name = tray.query_one("#console-staged-source-name-0", Static)
        status = tray.query_one("#console-staged-source-status-0", Static)
        assert str(name.renderable) == "readme.md"
        assert str(status.renderable) == "ready"
        assert status.has_class("ready")

        blocked_status = tray.query_one("#console-staged-source-status-1", Static)
        assert str(blocked_status.renderable) == "blocked"
        assert blocked_status.has_class("blocked")
