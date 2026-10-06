"""Recovery geometry stays correct without restyling an unchanged control bar."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_control_bar import ConsoleControlBar

# Real config readers in CompactModelBar keep their source-bound sandbox for
# this interpreter. No case edits config or accesses the real user profile.
pytestmark = pytest.mark.bootstrap_profile


class ControlBarHarness(App[None]):
    """Mount the complete production control bar with no provider selected."""

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self) -> None:
        super().__init__()
        self.app_config = {}
        self.bar = ConsoleControlBar(
            ConsoleControlState.from_values(), self, classes="retained-class"
        )

    def compose(self) -> ComposeResult:
        yield self.bar


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
async def test_unchanged_speech_recovery_does_not_restyle_the_bar(
    monkeypatch: pytest.MonkeyPatch, paused: bool
) -> None:
    """Catch remove/re-add of the same height class on stable refreshes."""
    app = ControlBarHarness()
    async with app.run_test(size=(120, 10)) as pilot:
        app.bar.sync_auto_speak(enabled=True, paused=paused, retry_available=paused)
        await pilot.pause()
        original = app.bar.update_node_styles
        restyles = []

        def observe_restyle() -> None:
            restyles.append(app.bar.classes)
            original()

        monkeypatch.setattr(app.bar, "update_node_styles", observe_restyle)
        for _ in range(5):
            app.bar.sync_auto_speak(enabled=True, paused=paused, retry_available=paused)
        assert restyles == []
        assert app.bar.has_class("retained-class")
        assert (
            app.bar.query_one("#console-auto-speak-row", Horizontal).display is paused
        )


@pytest.mark.asyncio
async def test_recovery_transitions_replace_height_in_one_restyle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch transient height removal while preserving real recovery controls."""
    app = ControlBarHarness()
    async with app.run_test(size=(120, 10)) as pilot:
        await pilot.pause()
        original = app.bar.update_node_styles
        restyles = []

        def observe_restyle() -> None:
            restyles.append(app.bar.classes)
            original()

        monkeypatch.setattr(app.bar, "update_node_styles", observe_restyle)
        app.bar.sync_auto_speak(enabled=True, paused=True, retry_available=True)
        assert restyles == [frozenset({"retained-class", "h-2"})]
        await pilot.pause()
        assert app.bar.region.height == 2
        assert app.bar.query_one("#console-auto-speak-resume", Button).display is True
        assert app.bar.query_one("#console-auto-speak-retry", Button).display is True

        app.bar.sync_auto_speak(enabled=True, paused=True, retry_available=False)
        assert len(restyles) == 1
        assert app.bar.query_one("#console-auto-speak-retry", Button).display is False

        app.bar.sync_auto_speak(enabled=False, paused=True)
        assert restyles == [
            frozenset({"retained-class", "h-2"}),
            frozenset({"retained-class", "h-1"}),
        ]
        await pilot.pause()
        assert app.bar.region.height == 1
        assert app.bar.query_one("#console-auto-speak-row", Horizontal).display is False


@pytest.mark.asyncio
@pytest.mark.parametrize("paused, height", [(False, 1), (True, 2)])
async def test_recovery_repairs_conflicting_height_and_inline_override(
    paused: bool, height: int
) -> None:
    """Catch skipping real class/inline repairs just because state is unchanged."""
    app = ControlBarHarness()
    async with app.run_test(size=(120, 10)) as pilot:
        app.bar.add_class("h-0", "h-3", "another-retained-class")
        app.bar.styles.height = 8
        app.bar.styles.min_height = 8
        app.bar.styles.max_height = 8
        app.bar.sync_auto_speak(enabled=True, paused=paused)
        await pilot.pause()
        assert app.bar.classes == frozenset(
            {"retained-class", "another-retained-class", f"h-{height}"}
        )
        assert app.bar.styles.inline.has_rule("height") is False
        assert app.bar.styles.min_height.value == height
        assert app.bar.styles.max_height.value == height
        assert app.bar.region.height == height


@pytest.mark.asyncio
async def test_newly_mounted_bar_applies_recovery_after_previous_bar_removal() -> None:
    """Catch a cross-widget memo hiding recovery on the replacement bar."""
    app = ControlBarHarness()
    async with app.run_test(size=(120, 10)) as pilot:
        app.bar.sync_auto_speak(enabled=True, paused=True)
        await app.bar.remove()
        app.bar = ConsoleControlBar(
            ConsoleControlState.from_values(), app, classes="retained-class"
        )
        await app.mount(app.bar)
        app.bar.sync_auto_speak(enabled=True, paused=True)
        await pilot.pause()
        assert app.bar.region.height == 2
        assert app.bar.query_one("#console-auto-speak-resume", Button).display is True
