"""TASK-33008.4: the Get started card's plain backdrop, its Tab stops and its copy.

Absorbs TASK-32572: Tab and Shift+Tab cycle every action the card shows, and a
guidance sync no longer pulls focus back to the first one. The snow field that
used to sit behind the card is gone; the dim is the modal's own flat fill.
The mounted tests are ``private_profile_test`` cases: a factory-built app
needs its own profile.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp, app_css_text
from Tests.UI.test_console_local_server_discovery_card import (
    _BLOCKED_CARD_STATE,
    _DETECTED_SERVER,
    ConsoleHarness,
    _blocked_console_app,
    _detected_action_focused,
)
from Tests.UI.test_console_switch_model_entry_points import _switcher_ready, _until
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_CARD_SUBTITLE,
    build_console_detected_server_action,
)
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_WORKBENCH_SHORTCUTS,
    CONSOLE_WORKBENCH_SHORTCUTS_SETUP_BLOCKED,
)
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_ACTION_ID,
    CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID,
    CONSOLE_SETUP_MODAL_NOTES_ACTION_ID,
    ConsoleSetupModal,
)

_PRIMARY = f"#{CONSOLE_SETUP_MODAL_ACTION_ID}"
_NOTES = f"#{CONSOLE_SETUP_MODAL_NOTES_ACTION_ID}"
_DETECTED = f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}"


class _StyledCardApp(ConsolidatedCSSApp):
    """The card under the app's own stylesheets, so the focus cue is real.

    The Input stands in for the workbench beneath the card: the app's own
    Tab would reach it, so it must never take focus while the card blocks.
    """

    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, *, detected: bool) -> None:
        super().__init__()
        self._detected = detected

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")
        yield Input(id="beneath")

    async def on_mount(self) -> None:
        modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        modal.sync_card_state(
            _BLOCKED_CARD_STATE,
            action_label="Set up provider",
            action_tooltip="Open provider settings.",
        )
        if self._detected:
            modal.sync_detected_server_action(
                build_console_detected_server_action(_DETECTED_SERVER, card_mode="card")
            )
        modal.focus_primary_action()


def _cue(button: Button) -> str:
    """The heavy side rail that task-32555's ``:focus`` rule draws."""
    return button.styles.outline_left[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detected", "stops"),
    [
        (False, [_PRIMARY, _NOTES]),
        (True, [_PRIMARY, _NOTES, _DETECTED]),
    ],
    ids=["two-actions", "with-detected-server"],
)
async def test_tab_and_shift_tab_cycle_every_card_action_with_the_focus_cue(
    detected: bool, stops: list[str]
) -> None:
    """AC#3 (TASK-32572 AC#1, AC#2): every shown action is a Tab stop, the
    order wraps both ways, and only the focused action draws the cue."""
    app = _StyledCardApp(detected=detected)

    async with app.run_test(size=(211, 44)) as pilot:
        buttons = [app.query_one(selector, Button) for selector in stops]
        # First focus: the detected action when offered (TASK-33008.3).
        start = len(stops) - 1 if detected else 0
        await _until(pilot, lambda: app.focused is buttons[start], "first focus")

        def assert_stop(index: int) -> None:
            assert app.focused is buttons[index], (
                f"expected {stops[index]}, got {app.focused!r}"
            )
            assert [_cue(button) for button in buttons] == [
                "heavy" if position == index else "" for position in range(len(stops))
            ]

        assert_stop(start)
        for step in range(1, len(stops) + 1):
            await pilot.press("tab")
            await pilot.pause()
            assert_stop((start + step) % len(stops))
        for step in range(1, len(stops) + 1):
            await pilot.press("shift+tab")
            await pilot.pause()
            assert_stop((start - step) % len(stops))


@pytest.mark.asyncio
async def test_a_sync_under_a_dialog_keeps_the_cards_tab_stop() -> None:
    """TASK-33008 review: the guard read ``app.focused``, which under a dialog
    is the dialog's widget, so a sync moved the card's remembered Tab stop
    back to the primary action and closing the dialog landed there."""
    from textual.screen import ModalScreen

    app = _StyledCardApp(detected=False)

    async with app.run_test(size=(211, 44)) as pilot:
        primary = app.query_one(_PRIMARY, Button)
        await _until(pilot, lambda: app.focused is primary, "first focus")
        await pilot.press("tab")
        notes = app.query_one(_NOTES, Button)
        await _until(pilot, lambda: app.focused is notes, "Tab to Notes")

        await app.push_screen(ModalScreen())
        await pilot.pause()
        app.query_one("#console-setup-modal", ConsoleSetupModal).focus_primary_action()
        app.pop_screen()
        await pilot.pause()
        assert app.focused is notes


@pytest.mark.asyncio
async def test_card_draws_on_a_plain_dim_fill_with_no_backdrop_widget() -> None:
    """AC#1: nothing but the card is drawn on the dim, and the dim is the
    modal's own flat token fill; the snow backdrop and its CSS are gone."""
    app = _StyledCardApp(detected=False)

    async with app.run_test(size=(211, 44)) as pilot:
        # The stand-in has no shell layers to sit under here; the real
        # workbench does (#console-shell's console-setup-overlay layer).
        app.query_one("#beneath").display = False
        await pilot.pause()
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        assert [child.id for child in modal.children] == ["console-setup-modal-card"]
        background = modal.styles.background
        assert 0 < background.a < 1, background

        card = app.query_one("#console-setup-modal-card")
        strips = app.screen._compositor.render_strips()
        assert any("Get started" in strip.text for strip in strips)
        for y, strip in enumerate(strips):
            text = strip.text
            if card.region.y <= y < card.region.bottom:
                text = text[: card.region.x] + text[card.region.right :]
            assert not text.strip(), f"row {y} draws outside the card: {text!r}"

    css = app_css_text()
    for selector in ("console-setup-modal-backdrop-snow", "console-setup-snow"):
        assert selector not in css


def test_card_copy_names_the_settings_round_trip_and_only_footer_keys() -> None:
    """AC#4: cloud setup opens Settings and comes back here, and the only key
    named is Enter, the one the blocked footer teaches (ADR-031 rule 5).

    TASK-33008.2 (R35): the line promises no return for a custom endpoint.
    Settings offers none there, on the card's own step or from a NEEDS SETUP
    row in Switch model."""
    assert ("Enter", "continue setup") in CONSOLE_WORKBENCH_SHORTCUTS_SETUP_BLOCKED
    assert "Settings" in CONSOLE_SETUP_CARD_SUBTITLE
    assert "back here, except for custom endpoints." in CONSOLE_SETUP_CARD_SUBTITLE
    assert "Enter" in CONSOLE_SETUP_CARD_SUBTITLE
    for key in ("Tab", "Esc", "Shift", "Ctrl", "Alt", "Space", "F1", "F6"):
        assert key not in CONSOLE_SETUP_CARD_SUBTITLE, key


@pytest.mark.asyncio
@private_profile_test
async def test_the_blocked_footer_names_only_keys_that_act_under_the_card(
    request,
) -> None:
    """Capture finding 3 (ADR-031 rule 5): under the card the footer showed
    "Alt+M switch model" and "Ctrl+O chat settings", and both did nothing (R10
    keeps Alt+M off while the card blocks). Every workbench key the blocked
    footer leaves out is pressed here and changes nothing; each one it names
    opens what it says."""
    app = _blocked_console_app(())
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        primary = await _card_focus_on(host, console, pilot, _PRIMARY)
        footer = console.query_one(AppFooterStatus)
        drawn = footer.query_one("#footer-key-quit", Static)
        await _until(
            pilot, lambda: "continue setup" in str(drawn.render()), "blocked footer"
        )
        # Round 2 finding 1: read the footer as drawn, global cluster included
        # ("F6 next pane · Ctrl+Q quit" is one " | " part, so split on both).
        drawn_text = str(drawn.render())
        hinted = {
            hint.split(" ", 1)[0]
            for part in drawn_text.split(" | ")
            for hint in part.split(" · ")
        }
        assert not {"Alt+M", "Ctrl+O", "F6"} & hinted, drawn_text

        for key, *_label in CONSOLE_WORKBENCH_SHORTCUTS:
            if key in hinted:
                continue
            await pilot.press(key.lower())
            await pilot.pause(0.2)
            assert host.screen is console and host.focused is primary, key

        for key, *_label in CONSOLE_WORKBENCH_SHORTCUTS_SETUP_BLOCKED:
            if key == "Enter" or key not in hinted:
                continue
            await pilot.press(key.lower())
            await _until(pilot, lambda: host.screen is not console, f"{key} to act")
            host.pop_screen()
            await _until(pilot, lambda: host.screen is console, f"{key}'s screen to go")


@pytest.mark.asyncio
@private_profile_test
async def test_a_dialog_over_the_card_leaves_the_cards_enter_out_of_the_footer(
    request,
) -> None:
    """Capture fix 3 finding 3 (02b): with Switch model open over the card the
    footer still read "Enter continue setup", while Enter there applies the
    dialog's highlighted model. While a dialog covers the card the footer
    names only the keys that act under it (a re-registration included):
    each key it leaves out is pressed and changes nothing, the palette's
    opens. The card's keys are back when the dialog closes."""
    app = _blocked_console_app((_DETECTED_SERVER,))
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _detected_action_focused(host, console, pilot)
        drawn = console.query_one(AppFooterStatus).query_one("#footer-key-quit", Static)
        await _until(
            pilot, lambda: "continue setup" in str(drawn.render()), "blocked footer"
        )

        await pilot.press("enter")
        switcher = await _switcher_ready(host, pilot)
        await _until(
            pilot,
            lambda: "continue setup" not in str(drawn.render()),
            "the card's Enter to leave the footer",
        )
        console._register_console_footer_shortcuts()  # a sync while covered
        await pilot.pause()
        drawn_text = str(drawn.render())
        hinted = {
            hint.split(" ", 1)[0]
            for part in drawn_text.split(" | ")
            for hint in part.split(" · ")
        }
        assert hinted == {"Ctrl+P", "Ctrl+Q"}, drawn_text

        for key in ("f1", "f6", "ctrl+shift+f"):
            focused = host.focused
            await pilot.press(key)
            await pilot.pause(0.2)
            assert host.screen is switcher and host.focused is focused, key
        await pilot.press("ctrl+p")
        await _until(pilot, lambda: host.screen is not switcher, "the palette")
        host.pop_screen()
        await _until(pilot, lambda: host.screen is switcher, "the palette to go")

        await pilot.press("escape")
        await _until(pilot, lambda: host.screen is console, "the dialog to close")
        await _until(
            pilot,
            lambda: "continue setup" in str(drawn.render()),
            "the card's Enter to come back",
        )


async def _card_focus_on(host, console, pilot, selector: str) -> Button:
    button = console.query_one(selector, Button)
    await _until(pilot, lambda: host.focused is button, f"focus on {selector}")
    return button


@pytest.mark.asyncio
@private_profile_test
async def test_a_guidance_sync_keeps_focus_on_the_action_tab_reached(request) -> None:
    """AC#3, R24 cause 2: the guidance sync used to pull focus back to the
    first action, so a Tab stop never held. Esc does not dismiss (AC#5)."""
    app = _blocked_console_app(())
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        await _card_focus_on(host, console, pilot, _PRIMARY)

        await pilot.press("tab")
        notes = await _card_focus_on(host, console, pilot, _NOTES)
        console._sync_console_control_bar()
        await pilot.pause()
        await pilot.pause(0.5)  # several of the 0.2 s guidance ticks
        assert host.focused is notes

        await pilot.press("escape")
        await pilot.pause()
        assert modal.is_blocking and modal.display
        assert host.focused is notes

        await pilot.press("shift+tab")
        await _card_focus_on(host, console, pilot, _PRIMARY)
        await pilot.press("shift+tab")
        await _card_focus_on(host, console, pilot, _NOTES)


@pytest.mark.asyncio
@private_profile_test
async def test_the_detected_action_takes_focus_once_and_every_stop_then_holds(
    request,
) -> None:
    """AC#3 with the detected-server action: it takes focus once when it
    appears (TASK-33008.3), and after that a sync leaves each Tab stop alone."""
    app = _blocked_console_app((_DETECTED_SERVER,))
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _detected_action_focused(host, console, pilot)

        for selector in (_PRIMARY, _NOTES, _DETECTED, _PRIMARY):
            await pilot.press("tab")
            button = await _card_focus_on(host, console, pilot, selector)
            console._sync_console_control_bar()
            await pilot.pause(0.5)
            assert host.focused is button, selector

        await pilot.press("shift+tab")
        await _card_focus_on(host, console, pilot, _DETECTED)
