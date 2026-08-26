"""The axes must offer real choices, not render as empty selects.

This is the difference between a screen that looks finished and one that
works. The rebuilt pane can mount all 57 controls, put Generate above the
fold and wire the press to synthesis, and still be useless if nothing ever
fills the provider, model and voice lists.

The fake service and catalog come from the legacy playground's own tests, so
both panes are driven by identical inputs -- the point of sharing
`SpeechCatalogMixin` is that they behave the same.
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select

from Tests.UI.speech_playground_fixtures import (
    FakeTTSService,
    _resolved,
    _wait_until,
)
from tldw_chatbook.TTS.adapter_types import TTSVoiceDiscoveryResult
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(provider="audio_cpp")


def _option_values(select: Select[Any]) -> tuple[Any, ...]:
    return tuple(value for _label, value in select._options)


@pytest.fixture
def faked_service(monkeypatch):
    """Point the pane's service hook at the legacy tests' fake."""
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane, "_check_higgs_installation", lambda self: None
    )
    return service


@pytest.mark.asyncio
async def test_the_provider_axis_offers_real_providers(faked_service):
    """An empty provider list means nothing downstream can resolve.

    Before `on_mount` kicked off the catalog, this select held only
    `Select.NULL` -- a control with nothing to choose.
    """
    app = _Harness()
    async with app.run_test(size=(160, 60)) as pilot:
        select = app.query_one("#tts-provider-select", Select)
        await _wait_until(
            pilot,
            lambda: any(isinstance(v, str) for v in _option_values(select)),
        )
        values = _option_values(select)

    assert "audio_cpp" in values, f"provider axis never populated: {values}"


class _SeededOpenAIHarness(App[None]):
    """A pane opened with saved exact custom OpenAI axis values."""

    def compose(self) -> ComposeResult:
        """Mount the pane seeded like a saved custom-endpoint setup.

        Returns:
            A ``ComposeResult`` yielding one seeded ``SpeechPlaygroundPane``.
        """
        seeded = {
            "tts-provider-select": "openai",
            "tts-model-select": "pocket-tts-model",
            "tts-voice-select": "pocket-voice",
        }
        yield SpeechPlaygroundPane(
            provider="openai",
            axis_values=dict(seeded),
            axis_defaults=dict(seeded),
        )


@pytest.mark.asyncio
async def test_saved_exact_custom_openai_model_and_voice_reach_the_axes(
    faked_service,
):
    """Saved custom OpenAI names must appear selected, not silently swapped.

    Settings ▸ Speech & TTS lets a user save an exact model and voice for a
    custom OpenAI-compatible endpoint (TASK-2260); before TASK-15421 the
    pane discarded both in favour of the official catalog's first entries
    (`tts-1` / `alloy`), so Generate synthesized with values the user never
    chose.

    Args:
        faked_service: Pane service hook pointed at the shared fake.
    """
    app = _SeededOpenAIHarness()
    async with app.run_test(size=(160, 60)) as pilot:
        model_select = app.query_one("#tts-model-select", Select)
        await _wait_until(
            pilot,
            lambda: "pocket-tts-model" in _option_values(model_select),
        )
        assert model_select.value == "pocket-tts-model"
        voice_select = app.query_one("#tts-voice-select", Select)
        await _wait_until(
            pilot,
            lambda: "pocket-voice" in _option_values(voice_select),
        )
        assert voice_select.value == "pocket-voice"
        # The pinned selection must actually be generatable — the readiness
        # gate used to veto any model outside the official catalog, leaving
        # the pinned axes displayed but Generate permanently disabled.
        generate = app.query_one("#tts-generate-btn", Button)
        await _wait_until(pilot, lambda: not generate.disabled)


@pytest.mark.asyncio
async def test_audio_cpp_voice_axis_rejects_a_different_catalog_revision(
    faked_service,
):
    async def stale_observation(
        provider_id: str,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        faked_service.voice_observation_calls.append((provider_id, model_id, refresh))
        return TTSVoiceDiscoveryResult(
            provider_id=provider_id,
            model_id=model_id,
            catalog_revision=faked_service.catalogs[provider_id].revision - 1,
            voices=("stale-voice",),
            state="complete",
        )

    faked_service.observe_voices = stale_observation
    app = _Harness()
    async with app.run_test(size=(160, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: bool(faked_service.voice_observation_calls),
        )
        await pilot.pause()
        voices = _option_values(app.query_one("#tts-voice-select", Select))

    assert "stale-voice" not in voices
