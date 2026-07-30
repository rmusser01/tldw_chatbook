"""Nothing TTS Settings offered may go missing in the rebuild.

Provider-aware by necessity. Settings are scoped to their provider's group,
so a flat "all 79 mounted" assertion cannot pass -- and unlike the
Playground, every group is mounted at once here, so what varies is not
existence but which are expanded. The union check still matters: a setting
dropped from the model disappears from the screen with nothing to say so.
"""

from __future__ import annotations

import pathlib

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Speech.speech_settings_model import (
    ALL_SETTINGS_CONTROLS,
    NON_SETTING_IDS,
    SETTINGS_CONTAINERS,
    PROVIDER_SETTINGS,
    SETTINGS_ACTIONS,
    SETTINGS_STATUS,
)
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane

_BUNDLE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"
)

#: Legacy ids the rebuild deliberately does not mount, each with its reason.
#: Named rather than dropped from the inventory so removing one is a recorded
#: decision and removing anything else still fails.
NOT_REBUILT: dict[str, str] = {
    "audio-cpp-mode-value": "static readout folded into the group header",
    "audio-cpp-privacy-notice": "one-line notice, per the spec",
}


class _Harness(App[None]):
    CSS_PATH = _BUNDLE

    def compose(self) -> ComposeResult:
        yield SpeechSettingsPane(id="speech-settings-pane")


async def _mounted_ids() -> set[str]:
    app = _Harness()
    async with app.run_test(size=(200, 80)) as pilot:
        await pilot.pause()
        await pilot.pause()
        return {w.id for w in app.query("*") if w.id}


@pytest.mark.asyncio
async def test_every_setting_survives_the_rebuild():
    """The guard the phase rests on: 79 ids re-sited, none silently lost."""
    required = (
        {c for controls in PROVIDER_SETTINGS.values() for c in controls}
        - set(NOT_REBUILT)
    )
    missing = required - await _mounted_ids()
    assert not missing, f"lost in the rebuild: {sorted(missing)}"


@pytest.mark.asyncio
async def test_the_shared_actions_are_mounted():
    """Save and the blend commands are the view's own, not a provider's."""
    shared = {
        action
        for action in SETTINGS_ACTIONS
        if not action.startswith(
            ("audio-cpp-", "chatterbox-", "higgs-", "kokoro-", "elevenlabs-")
        )
    }
    missing = shared - await _mounted_ids()
    assert not missing, f"missing actions: {sorted(missing)}"


@pytest.mark.asyncio
async def test_what_is_not_rebuilt_is_declared():
    """Every legacy id is either mounted or named with a reason -- there is
    no third category."""
    accounted = (
        await _mounted_ids()
        | set(NOT_REBUILT)
        | set(NON_SETTING_IDS)
        | set(SETTINGS_CONTAINERS)
        | set(SETTINGS_ACTIONS)
        | set(SETTINGS_STATUS)
    )
    unaccounted = ALL_SETTINGS_CONTROLS - accounted
    assert not unaccounted, (
        f"neither mounted nor declared as dropped: {sorted(unaccounted)}"
    )
