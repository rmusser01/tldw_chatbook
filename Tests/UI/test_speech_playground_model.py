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


@pytest.mark.unit
def test_the_inventory_matches_the_live_widget():
    """The model's inventory must equal what the widget actually composes.

    This is the guard against silently dropping a control in the rebuild: it
    reads the ids straight out of `TTSPlaygroundWidget` and diffs both ways,
    so a control that exists but is unclassified fails here rather than
    going missing on screen three phases later.
    """
    import re
    from pathlib import Path

    source = Path(
        "tldw_chatbook/UI/STTS_Window.py"
    ).read_text(encoding="utf-8").split("\n")
    start = next(
        i
        for i, line in enumerate(source, 1)
        if line.startswith("class TTSPlaygroundWidget")
    )
    end = next(
        i for i, line in enumerate(source, 1) if i > start and line.startswith("class ")
    )
    live = {
        match.group(1)
        for line in source[start - 1 : end - 1]
        for match in re.finditer(r'id="([a-z0-9_-]+)"', line)
    }

    assert live - LEGACY_PLAYGROUND_CONTROLS == set(), (
        "the widget composes controls the model does not know about"
    )
    assert LEGACY_PLAYGROUND_CONTROLS - live == set(), (
        "the model files a control as legacy that the widget does not compose "
        "-- if it is new, it belongs in NEW_PLAYGROUND_CONTROLS"
    )
    # Additions are deliberate and must not leak into the legacy set.
    assert NEW_PLAYGROUND_CONTROLS & live == set()
    assert ALL_PLAYGROUND_CONTROLS == LEGACY_PLAYGROUND_CONTROLS | NEW_PLAYGROUND_CONTROLS
