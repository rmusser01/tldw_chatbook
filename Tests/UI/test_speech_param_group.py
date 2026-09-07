"""Provider tuning knobs: scoped to the selection, collapsed by default."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Collapsible

from tldw_chatbook.UI.Speech.speech_param_group import PARAM_LABELS, SpeechParamGroup
from tldw_chatbook.UI.Speech.speech_playground_model import (
    PROVIDER_PARAMS,
    REQUEST_PARAMS,
    params_for_provider,
)


class _Harness(App[None]):
    def __init__(self, provider):
        super().__init__()
        self._provider = provider

    def compose(self) -> ComposeResult:
        yield SpeechParamGroup(provider=self._provider)


@pytest.mark.asyncio
async def test_only_the_selected_providers_knobs_are_mounted():
    """ElevenLabs' parameters must not exist in the DOM while Chatterbox is
    selected -- not merely be hidden, which would leave them focusable."""
    app = _Harness("chatterbox")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.query("#tts-exaggeration-input")
        assert not app.query("#tts-stability-input")


@pytest.mark.asyncio
async def test_the_group_starts_collapsed_and_actually_hides_its_contents():
    """Assert what renders, not just the attribute.

    Subclassing Collapsible and overriding compose() replaces its own
    composition -- the title row and the contents container it toggles -- so
    the group rendered fully expanded while still reporting
    `collapsed is True`. Checking the flag alone passed that bug straight
    through; the contents' visibility is the real oracle.
    """
    app = _Harness("chatterbox")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        group = app.query_one(Collapsible)
        assert group.collapsed is True
        control = app.query_one("#tts-exaggeration-input")
        assert not control.region.height, (
            "collapsed group is still rendering its controls"
        )


@pytest.mark.asyncio
async def test_a_provider_with_no_specific_params_still_gets_the_shared_ones():
    """audio.cpp has no provider-specific knobs but is not knob-less: every
    synthesis request carries download format and text normalisation."""
    app = _Harness("audio_cpp")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        for param in REQUEST_PARAMS:
            assert app.query(f"#{param}"), f"{param} missing for audio_cpp"


@pytest.mark.asyncio
async def test_an_unknown_provider_renders_rather_than_raising():
    """compose() must not raise on a provider the model has not been taught."""
    app = _Harness("nonexistent")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.query_one(Collapsible) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(PROVIDER_PARAMS))
async def test_every_parameter_gets_a_control_and_a_label(provider):
    """No knob may be listed by the model and then not rendered, and none may
    render without a human-readable label -- a bare id is not a control."""
    app = _Harness(provider)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for param in params_for_provider(provider):
            assert app.query(f"#{param}"), f"{provider}: {param} not rendered"
            assert param in PARAM_LABELS, f"{param} has no label"
