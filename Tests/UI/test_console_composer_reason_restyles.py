"""Send guidance retains its geometry without redundant size-class restyles."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

import tldw_chatbook.app  # noqa: F401 -- bind config under the collection bootstrap.
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile


class ComposerHarness(App[None]):
    """Mount the production composer without provider or profile mutations."""

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self) -> None:
        super().__init__()
        self.app_config = {}
        self.composer = ConsoleComposerBar(id="console-native-composer")

    def compose(self) -> ComposeResult:
        yield self.composer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "width, reason, visible",
    [
        (160, "Send disabled: type a message", True),
        (80, "Send disabled: type a message", False),
        (160, "", False),
    ],
)
async def test_unchanged_reason_does_not_restyle_its_size(
    monkeypatch: pytest.MonkeyPatch, width: int, reason: str, visible: bool
) -> None:
    """Catch removal/re-addition of unchanged width/height classes."""
    app = ComposerHarness()
    async with app.run_test(size=(width, 20)) as pilot:
        composer = app.composer
        strip = composer.query_one("#console-send-disabled-reason", Static)
        strip.add_class("retained-class")
        composer._sync_send_disabled_reason(reason, muted=True)
        await pilot.pause()
        original = strip.update_node_styles
        restyles = []

        def observe_restyle() -> None:
            restyles.append(strip.classes)
            original()

        monkeypatch.setattr(strip, "update_node_styles", observe_restyle)
        for _ in range(5):
            composer._sync_send_disabled_reason(reason, muted=True)
        assert restyles == []
        await pilot.pause()
        assert strip.display is visible
        assert strip.content.plain == reason
        assert strip.has_class("retained-class")
        if visible:
            assert strip.region.height == 1
            assert 0 < strip.region.width <= 52
            assert (
                composer.query_one("#console-command-visible-text").region.width >= 32
            )
        else:
            assert strip.region.area == 0


@pytest.mark.asyncio
async def test_reason_size_transition_is_atomic_and_resize_rebudgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch transient size-class removal or a stale narrow-row budget."""
    app = ComposerHarness()
    async with app.run_test(size=(160, 20)) as pilot:
        composer = app.composer
        strip = composer.query_one("#console-send-disabled-reason", Static)
        composer._sync_send_disabled_reason("", muted=False)
        await pilot.pause()
        original = strip.update_node_styles
        restyles = []

        def observe_restyle() -> None:
            restyles.append(strip.classes)
            original()

        monkeypatch.setattr(strip, "update_node_styles", observe_restyle)
        composer._sync_send_disabled_reason("Blocked", muted=False)
        assert len(restyles) == 1
        assert restyles[0] >= {"w-auto", "h-1"}
        assert not (restyles[0] & {"w-0", "h-0"})
        await pilot.pause()
        assert strip.display and strip.region.height == 1

        # Resize uses the same shared owner; cached copy still yields to the draft.
        composer._send_disabled_reason = "Blocked"
        await pilot.resize_terminal(80, 20)
        await pilot.pause()
        assert not strip.display
        assert strip.region.area == 0
        assert composer.query_one("#console-command-visible-text").region.width >= 8
        await pilot.resize_terminal(160, 20)
        await pilot.pause()
        assert strip.display and strip.region.height == 1
        assert strip.region.width <= 52


@pytest.mark.asyncio
@pytest.mark.parametrize("reason, visible", [("Blocked", True), ("", False)])
async def test_reason_repairs_conflicting_sizes_without_losing_other_classes(
    reason: str, visible: bool
) -> None:
    """Catch a state memo skipping real repairs to the currently mounted node."""
    app = ComposerHarness()
    async with app.run_test(size=(160, 20)) as pilot:
        composer = app.composer
        strip = composer.query_one("#console-send-disabled-reason", Static)
        strip.add_class("retained-class", "w-3", "w-full", "h-3", "h-full")
        strip.set_styles(width=8, height=8)
        composer._sync_send_disabled_reason(reason, muted=False)
        await pilot.pause()
        assert strip.has_class("retained-class")
        assert not (strip.classes & {"w-3", "w-full", "h-3", "h-full"})
        assert strip.styles.inline.has_rule("width") is False
        assert strip.styles.inline.has_rule("height") is False
        assert strip.display is visible
        assert strip.region.height == (1 if visible else 0)
        assert strip.content.plain == reason


@pytest.mark.asyncio
async def test_full_width_voice_keeps_reason_cached_but_out_of_layout() -> None:
    """Retain the existing voice-preparation suppression and return path."""
    app = ComposerHarness()
    async with app.run_test(size=(160, 20)) as pilot:
        composer = app.composer
        strip = composer.query_one("#console-send-disabled-reason", Static)
        composer._send_disabled_reason = "Blocked"
        composer._sync_full_width_voice_presentation(True)
        await pilot.pause()
        assert strip.content.plain == "Blocked"
        assert not strip.display and strip.region.area == 0
        composer._sync_full_width_voice_presentation(False)
        await pilot.pause()
        assert strip.display and strip.region.height == 1
        assert strip.region.width <= 52
