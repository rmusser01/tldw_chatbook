"""P1g: the inspector renders a Chat Dictionaries block from state (no I/O)."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static, Button

from tldw_chatbook.Chat.console_display_state import (
    ConsoleInspectorState, ConsoleDisplayRow, ConsoleInspectorAction,
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


async def test_dictionaries_block_renders_rows_and_actions():
    from dataclasses import replace
    state = replace(
        _state(),
        dictionary_rows=(
            ConsoleDisplayRow("Slang", "from conversation"),
            ConsoleDisplayRow("Period", "from character (shadowed)"),
        ),
        dictionary_actions=(
            ConsoleInspectorAction("console-inspector-dictionaries-attach", "Attach dictionary…", True),
            ConsoleInspectorAction("console-inspector-dictionaries-detach", "Detach dictionary…", True),
        ),
    )
    async with _Host(state).run_test(size=(120, 50)) as pilot:
        texts = [str(s.renderable) for s in pilot.app.query(Static)]
        assert any("Chat Dictionaries" in t for t in texts)
        assert any("Slang" in t for t in texts) and any("shadowed" in t for t in texts)
        assert pilot.app.query_one("#console-inspector-dictionaries-attach", Button)
        assert pilot.app.query_one("#console-inspector-dictionaries-detach", Button)


async def test_dictionaries_block_absent_when_empty():
    async with _Host(_state()).run_test(size=(120, 50)) as pilot:
        texts = [str(s.renderable) for s in pilot.app.query(Static)]
        assert not any("Chat Dictionaries" in t for t in texts)
