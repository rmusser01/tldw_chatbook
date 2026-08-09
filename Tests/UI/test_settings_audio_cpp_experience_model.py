from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSNativeCapabilityObservation,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    AudioCppExactChoiceState,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    GlobalSpeechTTSValidationError,
    audio_cpp_transport_warning,
    build_global_speech_tts_save_proposal,
    load_global_speech_tts_state,
    project_audio_cpp_global_choices,
)


def test_slice_four_audio_cpp_settings_model_is_external_only() -> None:
    projected = AudioCppConfig().to_mapping()
    state = load_global_speech_tts_state({}, environment={})
    managed_fields = {
        "managed_binary_path",
        "managed_server_json_path",
        "managed_startup_timeout_seconds",
        "managed_health_check_interval_seconds",
        "managed_termination_grace_seconds",
    }

    assert projected["mode"] == "external"
    assert state.providers["audio_cpp"] == projected
    assert set(GLOBAL_TTS_PROVIDER_FIELD_IDS["audio_cpp"]) == (
        set(projected) - {"mode"}
    )
    assert managed_fields.isdisjoint(projected)
    assert managed_fields.isdisjoint(GLOBAL_TTS_PROVIDER_FIELD_IDS["audio_cpp"])


def _observation(
    *,
    configuration_revision: int = 4,
    catalog_revision: int = 7,
    fresh: bool = True,
    approximate: bool = False,
    voices: tuple[str, ...] | None = ("voice-a",),
    voice_state: str = "complete",
) -> TTSNativeCapabilityObservation:
    catalog = TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=catalog_revision,
        health=ProviderHealth(state="available", fresh=fresh),
        models=(
            TTSModelInfo(
                model_id="model-a",
                display_name="Model A",
                family="fake",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
            TTSModelInfo(
                model_id="model-b",
                display_name="Model B",
                family="fake",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
        ),
        approximate=approximate,
    )
    voice_results = {}
    if voices is not None:
        voice_results["model-a"] = TTSVoiceDiscoveryResult(
            provider_id="audio_cpp",
            model_id="model-a",
            catalog_revision=catalog_revision,
            voices=voices,
            state=voice_state,  # type: ignore[arg-type]
        )
    return TTSNativeCapabilityObservation(
        snapshot=TTSNativeCapabilitySnapshot(
            provider_id="audio_cpp",
            configuration_revision=configuration_revision,
            state="unverified",
            catalog=catalog,
            voice_results=voice_results,
        ),
        observed_at=datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    "origin",
    (
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
        "http://localhost:8080",
        "https://remote.example.test",
    ),
)
def test_safe_audio_cpp_origins_do_not_show_remote_plain_http_warning(
    origin: str,
) -> None:
    assert audio_cpp_transport_warning(origin) is None


def test_non_loopback_plain_http_has_fixed_non_echoing_transport_warning() -> None:
    origin = "http://private-remote.example.test:8080"

    warning = audio_cpp_transport_warning(origin)

    assert warning is not None
    assert "not transport-encrypted" in warning
    assert "submitted text" in warning
    assert "returned audio" in warning
    assert origin not in warning


def test_audio_cpp_global_save_persists_a_canonical_origin() -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.providers["audio_cpp"]["base_url"] = "HTTP://EXAMPLE.COM:80/"

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="audio_cpp",
    )

    assert proposal.settings["audio_cpp"]["base_url"] == "http://example.com"


@pytest.mark.parametrize(
    "field_id",
    (
        "connect_timeout_seconds",
        "synthesis_timeout_seconds",
        "max_input_characters",
        "max_response_bytes",
        "max_metadata_bytes",
        "max_catalog_models",
        "max_voices_per_model",
        "max_identifier_characters",
    ),
)
def test_each_audio_cpp_timeout_and_safety_bound_is_validated(
    field_id: str,
) -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.providers["audio_cpp"][field_id] = 0

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.provider_id == "audio_cpp"
    assert error.value.field_id == field_id


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "binary_path",
        "server_config_path",
        "bind_address",
        "auth_headers",
        "launch_policy",
        "restart",
        "supervision",
        "stop",
    ),
)
def test_global_audio_cpp_save_rejects_managed_and_authentication_values(
    forbidden_key: str,
) -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.providers["audio_cpp"][forbidden_key] = "synthetic-value"

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.provider_id == "audio_cpp"
    assert forbidden_key not in str(error.value)


def test_global_audio_cpp_save_rejects_managed_mode_without_echoing_it() -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.providers["audio_cpp"]["mode"] = "managed-synthetic-value"

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.provider_id == "audio_cpp"
    assert "managed-synthetic-value" not in str(error.value)


def test_first_run_without_observation_offers_only_dynamic_modes() -> None:
    defaults = load_global_speech_tts_state({}, environment={}).defaults
    defaults.provider_id = "audio_cpp"
    defaults.model_mode = "first_available"
    defaults.model_id = None
    defaults.voice_mode = "server_default"
    defaults.voice_id = None

    choices = project_audio_cpp_global_choices(
        defaults,
        observation=None,
        current_configuration_revision=1,
    )

    assert choices.model.options == ()
    assert choices.model.exact_allowed is False
    assert choices.model.state is AudioCppExactChoiceState.NOT_OBSERVED
    assert choices.voice.options == ()
    assert choices.voice.exact_allowed is False
    assert choices.voice.state is AudioCppExactChoiceState.NOT_OBSERVED


def test_saved_exact_values_stay_pinned_and_unverified_without_observation() -> None:
    defaults = load_global_speech_tts_state({}, environment={}).defaults
    defaults.provider_id = "audio_cpp"
    defaults.model_mode = "exact"
    defaults.model_id = "saved-model"
    defaults.voice_mode = "exact"
    defaults.voice_id = "saved-voice"

    choices = project_audio_cpp_global_choices(
        defaults,
        observation=None,
        current_configuration_revision=1,
    )

    assert choices.model.options[-1][1] == "saved-model"
    assert choices.model.state is AudioCppExactChoiceState.UNVERIFIED
    assert choices.voice.options[-1][1] == "saved-voice"
    assert choices.voice.state is AudioCppExactChoiceState.UNVERIFIED


def test_fresh_catalog_and_model_scoped_voice_observation_offer_exact_choices() -> None:
    defaults = load_global_speech_tts_state({}, environment={}).defaults
    defaults.provider_id = "audio_cpp"
    defaults.model_mode = "exact"
    defaults.model_id = "model-a"
    defaults.voice_mode = "exact"
    defaults.voice_id = "voice-a"

    choices = project_audio_cpp_global_choices(
        defaults,
        observation=_observation(),
        current_configuration_revision=4,
    )

    assert tuple(value for _label, value in choices.model.options) == (
        "model-a",
        "model-b",
    )
    assert choices.model.state is AudioCppExactChoiceState.FRESH
    assert choices.voice.options == (("voice-a", "voice-a"),)
    assert choices.voice.state is AudioCppExactChoiceState.FRESH
    assert choices.catalog_revision == 7


def test_prior_configuration_observation_is_stale_and_cannot_prove_absence() -> None:
    defaults = load_global_speech_tts_state({}, environment={}).defaults
    defaults.provider_id = "audio_cpp"
    defaults.model_mode = "exact"
    defaults.model_id = "saved-missing-model"
    defaults.voice_mode = "exact"
    defaults.voice_id = "saved-missing-voice"

    choices = project_audio_cpp_global_choices(
        defaults,
        observation=_observation(configuration_revision=3),
        current_configuration_revision=4,
    )

    assert choices.model.state is AudioCppExactChoiceState.STALE
    assert choices.model.options[-1][1] == "saved-missing-model"
    assert choices.voice.state is AudioCppExactChoiceState.STALE
    assert choices.voice.options[-1][1] == "saved-missing-voice"


def test_fresh_authoritative_catalog_keeps_missing_exact_values_visible() -> None:
    defaults = load_global_speech_tts_state({}, environment={}).defaults
    defaults.provider_id = "audio_cpp"
    defaults.model_mode = "exact"
    defaults.model_id = "saved-missing-model"
    defaults.voice_mode = "exact"
    defaults.voice_id = "saved-missing-voice"

    choices = project_audio_cpp_global_choices(
        defaults,
        observation=_observation(),
        current_configuration_revision=4,
    )

    assert choices.model.state is AudioCppExactChoiceState.MISSING
    assert choices.model.options[-1][1] == "saved-missing-model"
    assert choices.voice.state is AudioCppExactChoiceState.UNVERIFIED
    assert choices.voice.options[-1][1] == "saved-missing-voice"


def test_approximate_catalog_cannot_mark_an_absent_exact_model_missing() -> None:
    defaults = load_global_speech_tts_state({}, environment={}).defaults
    defaults.provider_id = "audio_cpp"
    defaults.model_mode = "exact"
    defaults.model_id = "saved-model"

    choices = project_audio_cpp_global_choices(
        defaults,
        observation=_observation(approximate=True),
        current_configuration_revision=4,
    )

    assert choices.model.state is AudioCppExactChoiceState.UNVERIFIED
    assert choices.model.options[-1] == ("saved-model (Unverified)", "saved-model")


def test_complete_voice_observation_missing_exact_voice_is_authoritative() -> None:
    defaults = load_global_speech_tts_state({}, environment={}).defaults
    defaults.provider_id = "audio_cpp"
    defaults.model_mode = "exact"
    defaults.model_id = "model-a"
    defaults.voice_mode = "exact"
    defaults.voice_id = "saved-missing-voice"

    choices = project_audio_cpp_global_choices(
        defaults,
        observation=_observation(voices=("voice-a",)),
        current_configuration_revision=4,
    )

    assert choices.voice.state is AudioCppExactChoiceState.MISSING
    assert choices.voice.options[-1][1] == "saved-missing-voice"


def test_voice_choices_do_not_cross_exact_model_scope() -> None:
    defaults = load_global_speech_tts_state({}, environment={}).defaults
    defaults.provider_id = "audio_cpp"
    defaults.model_mode = "exact"
    defaults.model_id = "model-b"
    defaults.voice_mode = "exact"
    defaults.voice_id = "saved-voice"

    choices = project_audio_cpp_global_choices(
        defaults,
        observation=_observation(),
        current_configuration_revision=4,
    )

    assert choices.voice.options == (("saved-voice (Unverified)", "saved-voice"),)
    assert choices.voice.state is AudioCppExactChoiceState.UNVERIFIED
