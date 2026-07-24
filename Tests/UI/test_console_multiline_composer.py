"""TASK-381: a terminal-portable newline chord + composer capabilities in help.

Shift+Enter inserts a newline but terminals deliver it as a plain CR (send), so
the composer also needs Ctrl+J, which survives any terminal. The F1 help must
document the newline chords plus the attach / paste-path / cap capabilities that
were previously undiscoverable.
"""

import pytest
from textual.events import Key

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_console_native_chat_flow import (
    ConsoleHarness,
    _configure_native_ready_console,
)
from tldw_chatbook.UI.Screens.chat_screen import CONSOLE_WORKBENCH_SHORTCUT_GROUPS
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


def test_composer_help_documents_newline_attach_and_cap():
    """AC#2: the Composer help group covers the real composer capabilities."""
    groups = dict(CONSOLE_WORKBENCH_SHORTCUT_GROUPS)
    assert "Composer" in groups
    composer = groups["Composer"]
    flat = " ".join(f"{key} {label}" for key, label in composer).lower()

    # AC#1: a terminal-safe newline chord is documented.
    assert any("ctrl+j" in key.lower() for key, _ in composer)
    assert "newline" in flat
    # AC#2: attach, paste-path, and the per-message cap are covered.
    assert "attach" in flat
    assert "paste" in flat
    assert "5" in flat


@pytest.mark.asyncio
async def test_ctrl_j_inserts_a_newline_in_the_composer():
    """AC#1: Ctrl+J inserts a newline (Shift+Enter is swallowed as CR by terminals)."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("line one")

        console.on_key(Key(key="ctrl+j", character="\n"))
        composer.insert_text("line two")

        assert composer.draft_text() == "line one\nline two"
