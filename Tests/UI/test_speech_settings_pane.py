"""The assembled Settings view: Save reachable, configured providers open."""

from __future__ import annotations

import pathlib

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible

from tldw_chatbook.UI.Speech.speech_settings_model import (
    SETTINGS_ACTIONS,
    SETTINGS_PROVIDER_ORDER,
)
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane

_BUNDLE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = _BUNDLE

    def __init__(self, values=None):
        super().__init__()
        self._values = values or {}

    def compose(self) -> ComposeResult:
        yield SpeechSettingsPane(values=self._values, id="speech-settings-pane")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 60), (80, 24)])
async def test_save_is_reachable_without_scrolling(size):
    """The defect this phase exists to fix.

    `save-settings-btn` measured at y=102 in a 26-row viewport -- the primary
    action of a settings screen, four screens below where you land.
    """
    app = _Harness()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.pause()
        pane = app.query_one("#speech-settings-pane")
        save = app.query_one("#save-settings-btn", Button)
        assert pane.region.contains_region(save.region), (
            f"Save below the fold at {size}: y={save.region.y}"
        )


@pytest.mark.asyncio
async def test_a_configured_provider_opens_and_an_untouched_one_does_not():
    """The spec's rule: one block per provider, only the configured ones
    expanded. Opening all eight is the legacy wall of forms again; opening
    none makes the user hunt for the one they set up."""
    app = _Harness({"openai-api-key-input": "sk-live"})
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        by_provider = {
            group.provider: group for group in app.query(Collapsible).results()
        }
        assert by_provider["openai"].collapsed is False
        assert by_provider["elevenlabs"].collapsed is True


@pytest.mark.asyncio
async def test_an_incomplete_provider_opens_too():
    """Half-configured is the state that needs attention most, so it must
    not be the one hidden behind a closed disclosure."""
    app = _Harness({"elevenlabs-stability-input": "0.7"})
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        group = next(
            g for g in app.query(Collapsible).results() if g.provider == "elevenlabs"
        )
        assert group.collapsed is False


@pytest.mark.asyncio
async def test_every_provider_gets_a_group():
    app = _Harness()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        providers = {g.provider for g in app.query(Collapsible).results()}
        assert providers == set(SETTINGS_PROVIDER_ORDER)


@pytest.mark.asyncio
async def test_the_actions_keep_their_own_ids():
    """`CommandStrip` rewrites every action id it is given, so
    `save-settings-btn` would mount as `workbench-action-save-settings-btn`
    while the handler matches the bare id -- a button that renders and can
    never fire."""
    app = _Harness()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        for action in SETTINGS_ACTIONS:
            if action.startswith(("audio-cpp-", "chatterbox-", "higgs-", "kokoro-")):
                continue  # provider-scoped, mounted inside their groups
            assert app.query(f"#{action}"), f"{action} not mounted under its own id"
