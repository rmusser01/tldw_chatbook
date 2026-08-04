"""ConsoleCommandPopup widget behavior; ChatScreen integration (Tasks 3-4)."""

from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_command_suggestions import CommandSuggestion
from tldw_chatbook.Chat.console_skill_resolver import SkillCommandCandidate
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_command_popup import ConsoleCommandPopup

SUGGESTIONS = [
    CommandSuggestion(insert_text="/a ", label="/a", description="first"),
    CommandSuggestion(insert_text="/b ", label="/b", description="second"),
]


class _PopupApp(App):
    def compose(self) -> ComposeResult:
        # The popup repositions against whatever carries this id; a Static
        # suffices for widget-level tests.
        yield Static("anchor", id="console-native-composer")
        yield ConsoleCommandPopup()


@pytest.mark.asyncio
async def test_popup_show_highlight_accept_hide():
    app = _PopupApp()
    async with app.run_test(size=(80, 24)) as pilot:
        popup = app.screen.query_one(ConsoleCommandPopup)
        assert not popup.is_open

        popup.show_suggestions(SUGGESTIONS)
        await pilot.pause()
        assert popup.is_open
        assert popup.accept_selected().label == "/a"

        popup.move_highlight(1)
        assert popup.accept_selected().label == "/b"

        popup.move_highlight(1)  # wraps
        assert popup.accept_selected().label == "/a"

        popup.hide()
        await pilot.pause()
        assert not popup.is_open
        assert popup.accept_selected() is None


def _popup_labels(popup) -> list[str]:
    return [s.label for s in popup._suggestions]


@pytest.mark.asyncio
async def test_slash_opens_popup_and_typing_filters():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)
        assert not popup.is_open

        await pilot.press("/")
        await pilot.pause()
        assert popup.is_open
        assert _popup_labels(popup) == [
            "/prompt",
            "/system",
            "/skills",
            "/prefill",
            "/generate-image",
            "/rewind",
        ]

        await pilot.press("s", "y", "s")
        await pilot.pause()
        assert _popup_labels(popup) == ["/system"]


@pytest.mark.asyncio
async def test_enter_accepts_and_inserts_without_sending():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)

        await pilot.press("/", "s", "k")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert composer.draft_text() == "/skills "
        # No skill candidates configured -> arg-mode list is empty -> hidden.
        assert not popup.is_open


@pytest.mark.asyncio
async def test_down_up_navigates_and_tab_accepts():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.press("/")
        await pilot.pause()
        await pilot.press("down")  # -> "/system"
        await pilot.press("down")  # -> "/skills"
        await pilot.press("up")  # back to "/system"
        await pilot.press("tab")
        await pilot.pause()
        assert composer.draft_text() == "/system "


@pytest.mark.asyncio
async def test_escape_closes_popup_and_keeps_draft():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)

        await pilot.press("/")
        await pilot.pause()
        assert popup.is_open
        await pilot.press("escape")
        await pilot.pause()
        assert not popup.is_open
        assert composer.draft_text() == "/"


@pytest.mark.asyncio
async def test_skill_entries_and_skills_arg_mode():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)
        # Test-level snapshot injection — production refreshes candidates via
        # _refresh_console_skill_candidates; setting the tuple directly is
        # deliberate here, not the production path.
        await pilot.app.workers.wait_for_complete()
        console._console_skill_candidates = (
            SkillCommandCandidate(name="web-search", description="Search the web"),
        )

        await pilot.press("/", "w")
        await pilot.pause()
        # Bare `/skill-name` is not dispatchable on dev (fallback resolver
        # removed), so the entry completes to the `/skills <name> ` form.
        assert _popup_labels(popup) == ["/skills web-search"]
        await pilot.press("enter")
        await pilot.pause()
        assert composer.draft_text() == "/skills web-search "

        composer.load_draft("/skills w")
        console._sync_console_command_popup()
        await pilot.pause()
        assert popup.is_open
        assert _popup_labels(popup) == ["web-search"]
        await pilot.press("enter")
        await pilot.pause()
        assert composer.draft_text() == "/skills web-search "


async def _spy_submit_draft(console) -> AsyncMock:
    """Wrap the active controller's ``submit_draft`` so real sends still work."""
    controller = console._ensure_console_chat_controller()
    spy = AsyncMock(wraps=controller.submit_draft)
    controller.submit_draft = spy
    return spy


@pytest.mark.asyncio
async def test_enter_with_popup_closed_sends_normally():
    """Popup-hidden Enter must reach the normal send path (spec contract)."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)
        composer.load_draft("hello")
        await pilot.pause()
        assert not popup.is_open
        submit_spy = await _spy_submit_draft(console)

        console.query_one("#console-send-message", Button).press()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        submit_spy.assert_awaited()
