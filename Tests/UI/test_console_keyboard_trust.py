"""Keyboard-trust cluster from the Console UX review (tasks 345/348/358/359).

- 345: the non-obscuring focus contract's own mechanism was nullified —
  ``$ds-focus-bg`` resolved to the surface color, so every conforming
  control rendered a ~1.1:1 focus shift (an invisible-focus Tab+Enter
  activated 'Save as…' live).
- 348: transcript scrollback was mouse-gated — PageUp from the composer
  did nothing.
- 358: switcher arrows were dead (results are plain Buttons; only
  Tab/Shift+Tab moved).
- 359: the F6 rail stop painted no visible indicator (the region border
  is INLINE-styled, so CSS :focus-within cannot win).
"""

import re
from pathlib import Path

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_switcher_state import (
    ConsoleConversationBrowserInputRow,
)
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_FOCUS_FRAME_BORDER,
    CONSOLE_FRAME_BORDER,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)

CSS_DIR = Path("tldw_chatbook/css")


def _variables_text() -> str:
    return (CSS_DIR / "core/_variables.tcss").read_text()


def _agentic_text() -> str:
    return (CSS_DIR / "components/_agentic_terminal.tcss").read_text()


def test_focus_bg_token_is_visibly_raised_not_surface():
    """TASK-345: $ds-focus-bg = $ds-surface-raised = $surface rendered the
    contract's focus background invisible by definition."""
    text = _variables_text()
    match = re.search(r"^\$ds-focus-bg:\s*(?P<value>[^;]+);", text, re.M)
    assert match, "$ds-focus-bg must be defined"
    value = match.group("value").strip()
    assert value not in {"$ds-surface-raised", "$surface", "$ds-surface-panel"}, (
        "the focus background must be visibly distinct from resting surfaces"
    )


@pytest.mark.parametrize(
    "selector",
    [
        ".console-rail-collapse-button:focus",
        ".console-switcher-result:focus",
        ".console-workspace-conversation-row Button:focus",
    ],
)
def test_console_focus_gaps_have_contract_style_rules(selector):
    """TASK-345/358: controls with NO :focus rule fell back to Textual's
    imperceptible default shift; new rules must follow the non-obscuring
    contract vocabulary (ds tokens + bold underline, no accent/reverse)."""
    text = _agentic_text()
    index = text.find(selector)
    assert index != -1, f"missing focus rule for {selector}"
    block = text[index : text.index("}", index)]
    assert "background: $ds-focus-bg;" in block
    assert "color: $ds-focus-fg;" in block
    assert "text-style: bold underline;" in block
    assert "reverse" not in block
    assert "$accent" not in block


@pytest.mark.asyncio
async def test_composer_pageup_scrolls_transcript_keyboard_only():
    """TASK-348: scrollback must be reachable without the mouse."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        for n in range(30):
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER if n % 2 else ConsoleMessageRole.ASSISTANT,
                content=f"history row {n} " + "x" * 40,
            )
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        transcript = console.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        assert transcript.max_scroll_y > 0
        bottom = transcript.scroll_y

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        await pilot.press("pageup")
        await pilot.pause()
        assert transcript.scroll_y < bottom, (
            "PageUp from the composer must scroll the transcript"
        )
        await pilot.press("pagedown")
        await pilot.pause()
        assert transcript.scroll_y == pytest.approx(bottom, abs=1)


def _switcher_rows() -> tuple[ConsoleConversationBrowserInputRow, ...]:
    return tuple(
        ConsoleConversationBrowserInputRow(
            row_key=f"conv-{n}",
            conversation_id=f"conv-{n}",
            native_session_id=None,
            title=f"Conversation {n}",
            scope_type="workspace",
            workspace_id="ws-1",
            workspace_label="Workspace 1",
            updated_sort="2026-07-04T10:00:00+00:00",
        )
        for n in range(4)
    )


@pytest.mark.asyncio
async def test_switcher_arrow_keys_move_focus_through_results():
    """TASK-358: ArrowDown/ArrowUp must navigate results (quick-switcher
    idiom), from the search field and between results."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        modal = ConsoleSessionSwitcherModal(rows=_switcher_rows())
        host.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        assert isinstance(host.focused, Input)

        await pilot.press("down")
        focused = host.focused
        assert isinstance(focused, Button)
        assert focused.id == "console-switcher-result-0"

        await pilot.press("down")
        assert host.focused.id == "console-switcher-result-1"

        await pilot.press("up")
        assert host.focused.id == "console-switcher-result-0"

        # Up from the first result returns to the search field.
        await pilot.press("up")
        assert isinstance(host.focused, Input)


@pytest.mark.asyncio
async def test_f6_rail_stop_paints_accent_frame_and_restores_on_leave():
    """TASK-359: the rail F6 stop must be as visible as the other two —
    the inline region border swaps to the focus accent while focus is
    inside the rail, and restores when it leaves."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        rail = console.query_one("#console-left-rail")

        from textual.color import Color

        console._focus_console_workbench_target("console-left-rail")
        await pilot.pause()
        assert rail.styles.border_top[1] == Color.parse(
            CONSOLE_FOCUS_FRAME_BORDER[1]
        )

        console._focus_console_workbench_target("console-native-composer")
        await pilot.pause()
        assert rail.styles.border_top[1] == Color.parse(CONSOLE_FRAME_BORDER[1])
