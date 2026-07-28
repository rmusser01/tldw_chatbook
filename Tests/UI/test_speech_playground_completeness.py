"""Nothing the legacy Playground offered may go missing in the rebuild.

The check is deliberately provider-aware. A single "are all 57 ids mounted"
assertion cannot pass and never could: provider parameters are scoped to the
selected provider, which is the point of the redesign -- with audio.cpp
chosen, Chatterbox's knobs and Higgs' six sliders correctly do not exist. So
completeness is the union across providers, plus a per-provider check that
the always-present surface is there every time.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Speech.speech_playground_model import (
    PROVIDER_PARAMS,
    REPLACED_CONTAINERS,
    REQUIRED_PLAYGROUND_CONTROLS,
    params_for_provider,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane


class _Harness(App[None]):
    def __init__(self, provider: str) -> None:
        super().__init__()
        self._provider = provider

    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(provider=self._provider)


async def _ids_for(provider: str) -> set[str]:
    app = _Harness(provider)
    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        await pilot.pause()
        return {widget.id for widget in app.query("*") if widget.id}


@pytest.mark.asyncio
async def test_every_legacy_control_survives_somewhere():
    """The union across providers must cover every legacy control.

    This is the guard the whole phase rests on: the rebuild moved 57
    controls into a new grammar, and a control that quietly failed to make
    the journey is a feature the user silently lost.
    """
    mounted: set[str] = set()
    for provider in sorted(PROVIDER_PARAMS):
        mounted |= await _ids_for(provider)

    missing = REQUIRED_PLAYGROUND_CONTROLS - mounted
    assert not missing, f"lost in the rebuild: {sorted(missing)}"


@pytest.mark.asyncio
async def test_the_replaced_containers_are_genuinely_gone():
    """The containers the param group replaces must not linger.

    Keeping an empty legacy box alongside its replacement is how two
    competing layouts end up shipping at once.
    """
    mounted: set[str] = set()
    for provider in sorted(PROVIDER_PARAMS):
        mounted |= await _ids_for(provider)

    assert not (REPLACED_CONTAINERS & mounted)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(PROVIDER_PARAMS))
async def test_the_always_present_surface_is_present_for_every_provider(provider):
    """Text, actions, player, status and log do not depend on the provider.

    `_generate_tts` has no try/except anywhere in its 229 lines: it queries
    `#tts-text-input`, `#tts-generate-btn` and `#tts-generation-log`
    unguarded, so any of them missing for any provider is an uncaught
    `NoMatches` at the moment the user presses Generate.
    """
    always_present = REQUIRED_PLAYGROUND_CONTROLS - {
        control
        for other in PROVIDER_PARAMS
        for control in PROVIDER_PARAMS[other]
    } - {"reference-audio-btn", "clear-reference-audio-btn",
         "reference-audio-status", "higgs-voice-upload-btn",
         "higgs-clear-voice-btn", "higgs-voice-status",
         "higgs-voice-upload-row"}

    mounted = await _ids_for(provider)
    missing = always_present - mounted
    assert not missing, f"{provider} is missing {sorted(missing)}"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(PROVIDER_PARAMS))
async def test_only_the_selected_providers_knobs_are_mounted(provider):
    """A knob for another provider must not exist while this one is chosen.

    Not merely hidden: a hidden control is still focusable, so tabbing
    through the pane would walk into settings that do nothing.
    """
    mounted = await _ids_for(provider)
    foreign = {
        control
        for other, controls in PROVIDER_PARAMS.items()
        if other != provider
        for control in controls
    } - set(params_for_provider(provider))

    assert not (foreign & mounted), f"{provider} mounted {sorted(foreign & mounted)}"
