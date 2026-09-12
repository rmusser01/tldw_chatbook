"""TASK-2154.10: reduced-motion option for the setup backdrop and splash (FR-08, AC-04).

`appearance.reduce_motion = true` must render the Console setup backdrop and
the startup splash as static frames (no animation timers). For the SPLASH the
default (False) keeps the animated behavior. For the setup BACKDROP the
static frame became the only presentation in TASK-23021 (the animation's
whole-screen repaint cost ~4% of a core at idle on every new user's first
screen), so both settings must now produce the frozen field -- the setting's
promise holds trivially, and these tests keep it pinned.
"""

import random
from unittest.mock import patch

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Static

from tldw_chatbook.Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_BACKDROP_ID,
    ConsoleSetupBackdrop,
    ConsoleSetupModal,
)
from tldw_chatbook.Widgets.splash_screen import SplashScreen


def _fake_cli_setting(section, key=None, default=None):
    """Offline config reader: every key resolves to its code default."""
    return default


class BackdropHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleSetupBackdrop(rng=random.Random(7))


class ModalHarness(ConsolidatedCSSApp):
    def __init__(self, state: ConsoleSetupCardState, *, reduced_motion: bool):
        super().__init__()
        self._state = state
        self._reduced_motion = reduced_motion

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        modal.reduced_motion = self._reduced_motion
        modal.sync_card_state(self._state)


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
# ConsoleSetupBackdrop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backdrop_renders_static_frame_and_arms_no_timer():
    """TASK-23021 retired the snow animation for everyone: the field is one
    still frame per (re)size, with no timers -- which is exactly what AC-04's
    reduced-motion presentation always was. The setting's promise (this
    backdrop never animates under reduce_motion) therefore holds trivially,
    and is pinned here so a re-animated backdrop cannot ship without
    re-answering both this test and the reduce_motion contract."""
    app = BackdropHarness()

    async with app.run_test(size=(80, 24)):
        backdrop = app.query_one(ConsoleSetupBackdrop)
        # The static frame is seeded on mount/resize...
        assert backdrop.flake_count > 0
        rendered = str(backdrop.render())
        assert any(glyph in rendered for glyph in ("·", "•", "*"))

        # ...and no timer exists to ever advance it.
        assert len(backdrop._timers) == 0


# ---------------------------------------------------------------------------
# ConsoleSetupModal: the reduce_motion preference is still recorded, and the
# backdrop stays frozen in BOTH settings.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("reduced_motion", [True, False])
async def test_modal_records_preference_and_backdrop_stays_frozen(
    reduced_motion,
):
    app = ModalHarness(_blocking_state(), reduced_motion=reduced_motion)

    async with app.run_test(size=(80, 24)):
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.is_blocking
        # The ChatScreen writes this on every guidance sync; the recorded
        # preference must round-trip even though the backdrop no longer
        # branches on it.
        assert modal.reduced_motion is reduced_motion
        backdrop = app.query_one(
            f"#{CONSOLE_SETUP_MODAL_BACKDROP_ID}", ConsoleSetupBackdrop
        )
        assert len(backdrop._timers) == 0
        rendered = str(backdrop.render())
        assert any(glyph in rendered for glyph in ("·", "•", "*"))


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
