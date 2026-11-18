"""TASK-2154.10: reduced-motion option for the startup splash (FR-08, AC-04).

`appearance.reduce_motion = true` must render the startup splash as a static
frame (no animation timer); the default (False) keeps the animated behavior.
The Console setup backdrop this option also covered was a still frame for
everyone after TASK-23021, and TASK-33008.4 removed it: the card now sits on
a plain dim fill, which draws nothing that could move.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal
from tldw_chatbook.Widgets.splash_screen import SplashScreen


def _fake_cli_setting(section, key=None, default=None):
    """Offline config reader: every key resolves to its code default."""
    return default


class ModalHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")


class SplashHarness(ConsolidatedCSSApp):
    def __init__(self, **splash_kwargs):
        super().__init__()
        self._splash_kwargs = splash_kwargs

    def compose(self) -> ComposeResult:
        yield SplashScreen(duration=0, show_progress=False, **self._splash_kwargs)


def _blocking_state() -> ConsoleSetupCardState:
    return ConsoleSetupCardState(
        mode="card",
        steps=(ConsoleSetupStep(state="active", label="Add an API key"),),
    )


# ---------------------------------------------------------------------------
# ConsoleSetupModal: a guidance sync reads no config.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guidance_sync_needs_no_config_read_while_backup_pauses_storage():
    """Rendering a mounted modal needs no new disk admission during backup.

    Rewritten on purpose (TASK-33008.4) from
    ``test_guidance_uses_accepted_appearance_while_backup_pauses_config``,
    which also pinned the ``reduce_motion`` value the sync copied onto the
    modal for the removed backdrop. The contract kept: with no app config
    on the screen and every CLI-setting read refused, the sync still shows
    the card and its steps.
    """
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app = ModalHarness()
    screen = SimpleNamespace(
        query_one=app.query_one,
        _pending_console_launch_context=None,
        _console_detected_local_server=None,
        _apply_console_setup_block=lambda blocking: None,
        _maybe_start_console_local_discovery=lambda: None,
        _resume_navigation_startup_in_progress=True,
    )
    async with app.run_test(size=(80, 24)):
        with patch(
            "tldw_chatbook.UI.Screens.chat_screen.get_cli_setting",
            side_effect=RecoveryRequired("storage_locally_paused"),
        ):
            ChatScreen._sync_console_setup_modal(
                screen, _blocking_state(), action_label="Set up", action_tooltip="Set up"
            )
            modal = app.query_one(ConsoleSetupModal)
            assert modal.is_blocking and modal.display
            step = app.query_one("#console-setup-step-1", Static)
            assert "Add an API key" in str(step.render())


# ---------------------------------------------------------------------------
# SplashScreen
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_splash_reduced_motion_renders_card_statically_without_timer():
    with patch(
        "tldw_chatbook.Widgets.splash_screen.get_cli_setting",
        side_effect=_fake_cli_setting,
    ):
        app = SplashHarness(card_name="matrix", reduced_motion=True)

        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            splash = app.query_one(SplashScreen)
            assert splash.reduced_motion is True
            # No animation timer was started...
            assert splash.animation_timer is None
            # ...and static, readable content is on screen instead.
            display = app.query_one("#splash-display", Static)
            assert str(display.render()).strip()


@pytest.mark.asyncio
async def test_splash_default_motion_starts_animation_timer():
    with patch(
        "tldw_chatbook.Widgets.splash_screen.get_cli_setting",
        side_effect=_fake_cli_setting,
    ):
        app = SplashHarness(card_name="matrix", reduced_motion=False)

        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            splash = app.query_one(SplashScreen)
            assert splash.reduced_motion is False
            assert splash.animation_timer is not None
