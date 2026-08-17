"""ConsoleCommandPopup widget behavior; ChatScreen integration (Tasks 3-4)."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Static

import tldw_chatbook
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_command_grammar import default_console_registry
from tldw_chatbook.Chat.console_command_suggestions import CommandSuggestion
from tldw_chatbook.Chat.console_skill_resolver import SkillCommandCandidate
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_command_popup import ConsoleCommandPopup

SUGGESTIONS = [
    CommandSuggestion(insert_text="/a ", label="/a", description="first"),
    CommandSuggestion(insert_text="/b ", label="/b", description="second"),
]


class _PopupApp(ConsolidatedCSSApp):
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
        # The offered rows ARE the registered commands, in registry order.
        # Pinning the list literally is what broke this test: /generate-video
        # (task-3401.5) and /stream-video (task-3401.11) registered two more
        # built-ins and the six-item literal here sat red until something ran
        # the file whole. The claim was never "there are six" -- it was "the
        # popup offers the registered commands" -- so assert that, plus the
        # six this test was written for, and let honest additions through.
        labels = _popup_labels(popup)
        assert labels == [
            f"/{name}" for name in default_console_registry().available_names()
        ]
        assert {
            "/prompt",
            "/system",
            "/skills",
            "/prefill",
            "/generate-image",
            "/rewind",
        } <= set(labels)

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


class _StyledConsoleHarness(ConsoleHarness):
    """ConsoleHarness with the real bundled stylesheet loaded.

    The bare harness App has no CSS_PATH, so the popup's TCSS rules never
    apply in the other tests — fine for behavior assertions, but positioning
    must be verified with real CSS.
    """

    CSS_PATH = str(
        Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
    )


@pytest.mark.asyncio
async def test_popup_does_not_reclaim_workspace_grid_rows():
    """Opening the popup must not shrink the workspace grid.

    Regression (Textual 8.x vertical layout): a ``position: absolute``
    child still has its height deducted from the container's fr pool during
    box-model resolution, so opening the popup used to shrink the workspace
    grid by the popup's 8 rows — the transcript jumped up, a dead band
    opened under the status row, and on short terminals the anchor clamped
    to the shell top. The popup now places itself via ``overlay: screen``,
    which is exempt from that deduction: the grid must hold its height while
    autocomplete is open, and the popup must still clear the composer.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _StyledConsoleHarness(app)

    async with host.run_test(size=(150, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        grid = console.query_one("#console-workspace-grid")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        closed_grid_height = grid.region.height
        closed_composer_y = composer.region.y

        await pilot.press("/")
        await pilot.pause()
        await pilot.pause(0.2)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)
        assert popup.is_open
        # The transcript neither jumps nor yields rows to the open popup.
        assert grid.region.height == closed_grid_height
        assert composer.region.y == closed_composer_y
        popup_bottom = popup.region.y + popup.region.height
        assert popup_bottom <= composer.region.y, (
            f"popup {popup.region} overlaps composer {composer.region}"
        )


@pytest.mark.asyncio
async def test_popup_anchors_above_composer_with_real_css():
    """The popup's bottom edge sits at the composer's top edge, no overlap."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)

        await pilot.press("/")
        await pilot.pause()
        # Let the call_after_refresh re-anchor land and layout settle.
        await pilot.pause(0.2)
        assert popup.is_open
        assert popup.region.height > 0
        popup_bottom = popup.region.y + popup.region.height
        assert popup_bottom <= composer.region.y, (
            f"popup {popup.region} overlaps composer {composer.region}"
        )
        assert popup.region.y >= 0


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


@pytest.mark.asyncio
async def test_popup_anchor_clears_composer_with_chips_below():
    """DS-09 (TASK-2154.15), post-swap geometry: the status chip strip now
    sits BELOW the composer as the shell's bottom row, so an anchor that
    chases the chips would drop the popup over the input row. The popup's
    bottom edge must clear the composer's top edge, which also keeps the
    chips (further down) out of the popup's reach."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        chips = console.query_one("#console-status-chips")
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)

        await pilot.press("/")
        await pilot.pause()
        # Let the call_after_refresh re-anchor land and layout settle.
        await pilot.pause(0.2)
        assert popup.is_open
        assert chips.display and chips.region.height > 0
        # The chips are the bottom row, below the composer.
        assert composer.region.y + composer.region.height <= chips.region.y
        popup_bottom = popup.region.y + popup.region.height
        assert popup_bottom <= composer.region.y, (
            f"popup {popup.region} overlaps composer {composer.region}"
        )
        assert popup_bottom <= chips.region.y, (
            f"popup {popup.region} overlaps status chips {chips.region}"
        )
