"""The Playground's axis/knob classification.

Pure data, so it is testable without mounting `TTSPlaygroundWidget` -- a
5,900-line widget that builds a TTS playground on compose.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Speech.speech_playground_model import (
    ALL_PLAYGROUND_CONTROLS,
    AUDIO_PARAMS,
    AXIS_CONTROLS,
    LEGACY_PLAYGROUND_CONTROLS,
    NEW_PLAYGROUND_CONTROLS,
    PROVIDER_PARAMS,
    REQUEST_PARAMS,
    params_for_provider,
)


@pytest.mark.unit
def test_only_the_selected_providers_parameters_are_offered():
    """ElevenLabs' knobs must not appear while Chatterbox is selected.

    This is the whole point of the split: 26 provider parameters exist, but
    a user comparing voices should see only the ones that apply to what they
    have selected.
    """
    chatterbox = set(params_for_provider("chatterbox"))
    elevenlabs = set(params_for_provider("elevenlabs"))

    assert "tts-exaggeration-input" in chatterbox
    assert "tts-stability-input" not in chatterbox
    assert "tts-stability-input" in elevenlabs
    assert "tts-exaggeration-input" not in elevenlabs


@pytest.mark.unit
def test_audio_post_processing_applies_to_every_provider():
    """Normalisation is not provider-specific and must not vanish with a
    provider switch."""
    for provider in PROVIDER_PARAMS:
        assert "tts-normalize-audio-switch" in params_for_provider(provider)


@pytest.mark.unit
def test_an_unknown_provider_offers_only_the_shared_parameters():
    """A provider the model has not been taught about must degrade, not raise.

    `params_for_provider` is called from compose(); an exception there takes
    down the screen rather than showing a missing group.
    """
    assert set(params_for_provider("nonexistent")) == set(AUDIO_PARAMS) | set(
        REQUEST_PARAMS
    )


@pytest.mark.unit
def test_every_provider_has_knobs_including_the_ones_with_no_specific_params():
    """No provider is knob-less.

    audio.cpp, OpenAI and AllTalk have no provider-specific parameters, but
    every synthesis request carries `download_format` and six
    `normalization_options` fields -- none of which any screen surfaced
    before. They looked knob-less because the legacy Playground never
    offered the request options it was already sending.
    """
    for provider in PROVIDER_PARAMS:
        knobs = params_for_provider(provider)
        assert knobs, f"{provider} offers nothing to tune"
        assert set(REQUEST_PARAMS) <= set(knobs)

    for bare in ("audio_cpp", "openai", "alltalk"):
        assert PROVIDER_PARAMS[bare] == ()
        assert len(params_for_provider(bare)) == len(AUDIO_PARAMS) + len(
            REQUEST_PARAMS
        )


@pytest.mark.unit
def test_axes_and_parameters_do_not_overlap():
    """A control is an axis or a knob, never both.

    An overlap would render the control twice -- once always-visible and
    once inside the collapsed group -- with two sources of truth.
    """
    knobs = {c for params in PROVIDER_PARAMS.values() for c in params}
    assert set(AXIS_CONTROLS) & knobs == set()
    assert set(AXIS_CONTROLS) & set(AUDIO_PARAMS) == set()
    assert set(AXIS_CONTROLS) & set(REQUEST_PARAMS) == set()


@pytest.mark.unit
def test_every_classified_control_is_a_known_playground_control():
    """The classification may not invent ids the Playground does not have."""
    classified = (
        set(AXIS_CONTROLS)
        | set(AUDIO_PARAMS)
        | {c for params in PROVIDER_PARAMS.values() for c in params}
    )
    assert classified <= ALL_PLAYGROUND_CONTROLS


# `test_the_inventory_matches_the_live_widget` lived here. It parsed
# `TTSPlaygroundWidget`'s source and diffed the ids both ways, which is what
# kept the rebuild honest while the two playgrounds coexisted. That widget is
# now deleted, so there is no live source to diff against -- and the guard it
# provided has moved to `test_speech_playground_completeness.py`, which
# asserts the same thing against what the pane actually mounts, per provider.
#
# `LEGACY_PLAYGROUND_CONTROLS` stays as the frozen record of what the legacy
# screen offered. It is the yardstick that test measures against; without it
# "nothing was dropped" has nothing to mean.

