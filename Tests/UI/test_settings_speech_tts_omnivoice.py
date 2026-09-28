"""The speech-tts settings category projects [OmniVoiceSettings]."""

from __future__ import annotations

from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    GLOBAL_TTS_PROVIDER_ENVIRONMENT_FIELDS,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    TTS_PROVIDER_LABELS,
    load_global_speech_tts_state,
)


def test_omnivoice_in_provider_set() -> None:
    assert "omnivoice" in BUILT_IN_TTS_PROVIDER_ORDER
    assert TTS_PROVIDER_LABELS["omnivoice"] == "OmniVoice"


def test_omnivoice_fields_declared() -> None:
    fields = GLOBAL_TTS_PROVIDER_FIELD_IDS["omnivoice"]
    assert "model_root" in fields
    assert "num_steps" in fields
    assert "guidance_scale" in fields
    assert GLOBAL_TTS_PROVIDER_ENVIRONMENT_FIELDS["omnivoice"]["model_root"] == (
        "OMNIVOICE_MODEL_ROOT"
    )


def test_projection_reads_section() -> None:
    state = load_global_speech_tts_state(
        {"OmniVoiceSettings": {"num_steps": 16, "model_root": "/models/omnivoice"}},
        environment={},
    )
    omnivoice = state.providers["omnivoice"]

    assert omnivoice["num_steps"] == 16
    assert omnivoice["model_root"] == "/models/omnivoice"


def test_projection_defaults_without_section() -> None:
    state = load_global_speech_tts_state({}, environment={})
    omnivoice = state.providers["omnivoice"]

    assert omnivoice["model_root"] == ""
    assert omnivoice["num_steps"] == 32
    assert omnivoice["guidance_scale"] == 2.0
    assert omnivoice["max_reference_duration"] == 30
    assert omnivoice["language"] == "auto"
    assert omnivoice["voice_resource_directory"] == (
        "~/.config/tldw_cli/omnivoice_voices"
    )


def test_projection_environment_wins_for_declared_fields() -> None:
    """Env presence is surfaced via field sources; runtime override is the
    engine layer's job (covered by the provider-wiring tests)."""
    from tldw_chatbook.UI.Screens.settings_speech_tts import (
        GlobalSpeechTTSEffectiveSource,
    )

    state = load_global_speech_tts_state(
        {"OmniVoiceSettings": {"model_root": "/section/models"}},
        environment={"OMNIVOICE_MODEL_ROOT": "/env/models"},
    )

    assert state.provider_field_sources["omnivoice"]["model_root"] is (
        GlobalSpeechTTSEffectiveSource.ENVIRONMENT
    )
    assert state.provider_sources["omnivoice"] is (
        GlobalSpeechTTSEffectiveSource.ENVIRONMENT
    )
