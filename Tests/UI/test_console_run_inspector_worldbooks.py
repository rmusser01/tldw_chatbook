"""Roleplay P2g-2: the inspector renders a World Books block from state (no I/O)."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static, Button

from tldw_chatbook.Chat.console_display_state import (
    ConsoleInspectorState,
    ConsoleDisplayRow,
    ConsoleInspectorAction,
)
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector

pytestmark = pytest.mark.asyncio


def _state(**kw):
    return ConsoleInspectorState.from_values(**kw)


class _Host(App):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleRunInspector(self._state)


async def test_worldbooks_block_renders_rows_and_actions():
    from dataclasses import replace

    state = replace(
        _state(),
        world_book_rows=(ConsoleDisplayRow("Alpha", "2 entries"),),
        world_book_actions=(
            ConsoleInspectorAction(
                "console-inspector-worldbooks-attach", "Attach world book…", True
            ),
        ),
    )
    async with _Host(state).run_test(size=(120, 50)) as pilot:
        texts = [str(s.renderable) for s in pilot.app.query(Static)]
        assert any("World Books" in t for t in texts)
        row = pilot.app.query_one("#console-inspector-worldbooks-row-0", Static)
        assert str(row.renderable) == "Alpha: 2 entries"
        assert pilot.app.query_one("#console-inspector-worldbooks-attach", Button)


async def test_worldbooks_block_absent_when_empty():
    async with _Host(_state()).run_test(size=(120, 50)) as pilot:
        texts = [str(s.renderable) for s in pilot.app.query(Static)]
        assert not any("World Books" in t for t in texts)


async def test_dictionaries_block_still_renders_no_regression():
    from dataclasses import replace

    state = replace(
        _state(),
        dictionary_rows=(ConsoleDisplayRow("Slang", "from conversation"),),
        dictionary_actions=(
            ConsoleInspectorAction(
                "console-inspector-dictionaries-attach", "Attach dictionary…", True
            ),
        ),
    )
    async with _Host(state).run_test(size=(120, 50)) as pilot:
        texts = [str(s.renderable) for s in pilot.app.query(Static)]
        assert any("Chat Dictionaries" in t for t in texts)
        assert any("Slang" in t for t in texts)
        assert pilot.app.query_one("#console-inspector-dictionaries-attach", Button)
        assert not any("World Books" in t for t in texts)
