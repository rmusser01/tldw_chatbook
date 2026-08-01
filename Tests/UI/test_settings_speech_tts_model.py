from __future__ import annotations

from copy import deepcopy

import pytest

from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    CredentialIntent,
    CredentialSource,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    GlobalSpeechTTSEffectiveSource,
    GlobalSpeechTTSCredentialMutation,
    GlobalSpeechTTSValidationError,
    build_credential_mutation,
    build_global_speech_tts_save_proposal,
    global_speech_tts_provider_configuration_state,
    load_global_speech_tts_state,
    restore_non_secret_defaults,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConfigurationState,
)


def _settings() -> dict[str, object]:
    return {
        "COMPREHENSIVE_CONFIG_RAW": {
            "app_tts": {
                "default_provider": "audio_cpp",
                "default_model_mode": "first_available",
                "default_voice_mode": "server_default",
                "default_format": "wav",
                "default_speed": 1.0,
                "audio_cpp": {
                    **AudioCppConfig().to_mapping(),
                    "base_url": "http://127.0.0.1:18001",
                },
                "OPENAI_BASE_URL": "https://api.openai.com/v1/audio/speech",
                "OPENAI_ORG_ID": "org-local",
                "KOKORO_DEVICE_DEFAULT": "cpu",
                "KOKORO_USE_ONNX": True,
                "KOKORO_ONNX_MODEL_PATH_DEFAULT": "/models/kokoro.onnx",
                "KOKORO_ONNX_VOICES_JSON_DEFAULT": "/models/voices.json",
                "CHATTERBOX_DEVICE": "cpu",
                "CHATTERBOX_VOICE_DIR": "/voices/chatterbox",
                "ALLTALK_TTS_URL_DEFAULT": "http://127.0.0.1:7851",
            },
            "api_settings": {
                "openai": {"api_key": "local-openai-secret"},
                "elevenlabs": {"api_key": "local-eleven-secret"},
            },
            "HiggsSettings": {
                "model_path": "bosonai/higgs-audio-v2-generation-3B-base",
                "voice_samples_dir": "/voices/higgs",
                "device": "auto",
                "enable_flash_attn": True,
                "dtype": "bfloat16",
            },
        }
    }


def test_global_field_inventory_is_bounded_complete_and_has_no_managed_audio_cpp() -> (
    None
):
    assert BUILT_IN_TTS_PROVIDER_ORDER == (
        "audio_cpp",
        "openai",
        "elevenlabs",
        "kokoro",
        "chatterbox",
        "higgs",
        "alltalk",
    )
    assert set(GLOBAL_TTS_PROVIDER_FIELD_IDS) == set(BUILT_IN_TTS_PROVIDER_ORDER)

    assert {
        "base_url",
        "connect_timeout_seconds",
        "synthesis_timeout_seconds",
        "max_input_characters",
        "max_response_bytes",
        "max_metadata_bytes",
        "max_catalog_models",
        "max_voices_per_model",
        "max_identifier_characters",
    } <= set(GLOBAL_TTS_PROVIDER_FIELD_IDS["audio_cpp"])
    assert {"credential", "base_url", "organization_id"} <= set(
        GLOBAL_TTS_PROVIDER_FIELD_IDS["openai"]
    )
    assert {"credential"} <= set(GLOBAL_TTS_PROVIDER_FIELD_IDS["elevenlabs"])
    assert {"device", "use_onnx", "onnx_model_path", "voices_json_path"} <= set(
        GLOBAL_TTS_PROVIDER_FIELD_IDS["kokoro"]
    )
    assert {"device", "voice_resource_directory"} <= set(
        GLOBAL_TTS_PROVIDER_FIELD_IDS["chatterbox"]
    )
    assert {
        "model_path",
        "voice_resource_directory",
        "device",
        "enable_flash_attention",
        "dtype",
    } <= set(GLOBAL_TTS_PROVIDER_FIELD_IDS["higgs"])
    assert {"server_url"} <= set(GLOBAL_TTS_PROVIDER_FIELD_IDS["alltalk"])

    rendered_names = " ".join(GLOBAL_TTS_PROVIDER_FIELD_IDS["audio_cpp"]).lower()
    for forbidden in (
        "binary",
        "server.json",
        "bind",
        "launch",
        "adoption",
        "restart",
        "supervision",
        "stop",
    ):
        assert forbidden not in rendered_names


def test_load_state_keeps_credentials_out_of_editable_provider_values() -> None:
    state = load_global_speech_tts_state(
        _settings(),
        environment={"OPENAI_API_KEY": "environment-secret"},
    )

    assert state.defaults.provider_id == "audio_cpp"
    assert state.providers["audio_cpp"]["base_url"] == "http://127.0.0.1:18001"
    assert "credential" not in state.providers["openai"]
    assert "local-openai-secret" not in repr(state)
    assert state.credentials["openai"].source is CredentialSource.ENVIRONMENT
    assert state.credentials["openai"].environment_variable == "OPENAI_API_KEY"
    assert state.credentials["openai"].local_saved is True
    assert state.credentials["openai"].local_shadowed is True
    assert state.credentials["elevenlabs"].source is CredentialSource.SAVED_LOCAL


def test_load_state_marks_legacy_environment_owned_paths_without_copying_values() -> (
    None
):
    state = load_global_speech_tts_state(
        _settings(),
        environment={
            "KOKORO_MODEL_PATH": "/environment/private-kokoro-model",
            "KOKORO_VOICES_PATH": "/environment/private-kokoro-voices",
            "HIGGS_MODEL_PATH": "/environment/private-higgs-model",
        },
    )

    assert state.provider_sources["kokoro"] is (
        GlobalSpeechTTSEffectiveSource.ENVIRONMENT
    )
    assert state.provider_sources["higgs"] is (
        GlobalSpeechTTSEffectiveSource.ENVIRONMENT
    )
    assert state.provider_field_sources["kokoro"] == {
        "onnx_model_path": GlobalSpeechTTSEffectiveSource.ENVIRONMENT,
        "voices_json_path": GlobalSpeechTTSEffectiveSource.ENVIRONMENT,
    }
    assert state.provider_field_sources["higgs"] == {
        "model_path": GlobalSpeechTTSEffectiveSource.ENVIRONMENT,
    }
    rendered = repr(state)
    assert "/environment/private-kokoro-model" not in rendered
    assert "/environment/private-kokoro-voices" not in rendered
    assert "/environment/private-higgs-model" not in rendered


@pytest.mark.parametrize("provider_id", ("openai", "elevenlabs"))
def test_credential_provider_is_incomplete_until_a_safe_source_exists(
    provider_id: str,
) -> None:
    missing = load_global_speech_tts_state({}, environment={})
    environment = load_global_speech_tts_state(
        {},
        environment={
            missing.credentials[provider_id].environment_variable: "synthetic-secret"
        },
    )
    local = load_global_speech_tts_state(_settings(), environment={})

    assert (
        global_speech_tts_provider_configuration_state(
            missing,
            provider_id=provider_id,
        )
        is SpeechTTSConfigurationState.INCOMPLETE
    )
    assert (
        global_speech_tts_provider_configuration_state(
            environment,
            provider_id=provider_id,
        )
        is SpeechTTSConfigurationState.SAVED
    )
    assert (
        global_speech_tts_provider_configuration_state(
            local,
            provider_id=provider_id,
        )
        is SpeechTTSConfigurationState.SAVED
    )


@pytest.mark.parametrize(
    ("provider_id", "raw"),
    (
        ("openai", {"openai_api": {"api_key": "legacy-openai-secret"}}),
        ("openai", {"API": {"openai_api_key": "legacy-openai-secret"}}),
        (
            "elevenlabs",
            {"elevenlabs_api": {"api_key": "legacy-eleven-secret"}},
        ),
        (
            "elevenlabs",
            {"API": {"elevenlabs_api_key": "legacy-eleven-secret"}},
        ),
    ),
)
def test_legacy_credential_aliases_remain_readable_without_projecting_secrets(
    provider_id: str,
    raw: dict[str, object],
) -> None:
    state = load_global_speech_tts_state(
        {"COMPREHENSIVE_CONFIG_RAW": raw},
        environment={},
    )

    assert state.credentials[provider_id].source is CredentialSource.SAVED_LOCAL
    assert state.credentials[provider_id].local_saved is True
    assert "legacy-" not in repr(state)


def test_ordinary_save_excludes_credentials_and_targets_only_changed_provider() -> None:
    original = load_global_speech_tts_state(_settings(), environment={})
    draft = deepcopy(original)
    draft.providers["audio_cpp"]["synthesis_timeout_seconds"] = 321.0

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="audio_cpp",
    )

    assert proposal.changed_provider_ids == ("audio_cpp",)
    assert proposal.settings == {
        "audio_cpp": {
            **AudioCppConfig().to_mapping(),
            "base_url": "http://127.0.0.1:18001",
            "synthesis_timeout_seconds": 321.0,
        }
    }
    assert "openai_api_key" not in proposal.settings
    assert "elevenlabs_api_key" not in proposal.settings


def test_selection_only_save_has_no_adapter_affecting_payload() -> None:
    original = load_global_speech_tts_state(_settings(), environment={})
    draft = deepcopy(original)
    draft.defaults.provider_id = "openai"
    draft.defaults.model_mode = "exact"
    draft.defaults.model_id = "tts-1-hd"
    draft.defaults.voice_mode = "exact"
    draft.defaults.voice_id = "alloy"
    draft.defaults.response_format = "mp3"

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="audio_cpp",
    )

    assert proposal.settings == {}
    assert proposal.changed_provider_ids == ()
    assert proposal.preferences.provider_id == "openai"


@pytest.mark.parametrize("speed", (0.24, 4.01, float("inf")))
def test_global_default_speed_enforces_the_visible_range(speed: float) -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.defaults.speed = speed

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.provider_id == "defaults"
    assert error.value.field_id == "default_speed"
    assert str(speed) not in str(error.value)


@pytest.mark.parametrize(
    ("field", "value", "expected_field"),
    (
        ("provider_id", "unknown-provider", "provider_id"),
        ("response_format", "executable", "response_format"),
    ),
)
def test_global_default_choices_are_bounded(
    field: str,
    value: str,
    expected_field: str,
) -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    setattr(draft.defaults, field, value)

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.provider_id == "defaults"
    assert error.value.field_id == expected_field
    assert value not in str(error.value)


@pytest.mark.parametrize(
    ("field", "value", "expected_field"),
    (
        ("response_format", "mp3", "response_format"),
        ("speed", 2.0, "default_speed"),
    ),
)
def test_audio_cpp_default_constraints_report_the_responsible_field(
    field: str,
    value: object,
    expected_field: str,
) -> None:
    original = load_global_speech_tts_state(_settings(), environment={})
    draft = deepcopy(original)
    setattr(draft.defaults, field, value)

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.provider_id == "defaults"
    assert error.value.field_id == expected_field


@pytest.mark.parametrize(
    ("field", "value", "expected_field"),
    (
        ("model_id", " unsafe-model", "default_model"),
        ("model_id", "unsafe\nmodel", "default_model"),
        ("voice_id", "unsafe\x00voice", "default_voice"),
        ("voice_id", "v" * 513, "default_voice"),
    ),
)
def test_exact_global_identifiers_are_bounded_safe_and_non_echoing(
    field: str,
    value: str,
    expected_field: str,
) -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.defaults.provider_id = "openai"
    draft.defaults.model_mode = "exact"
    draft.defaults.model_id = "tts-1"
    draft.defaults.voice_mode = "exact"
    draft.defaults.voice_id = "alloy"
    draft.defaults.response_format = "mp3"
    setattr(draft.defaults, field, value)

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="openai",
        )

    assert error.value.provider_id == "defaults"
    assert error.value.field_id == expected_field
    assert value not in str(error.value)


def test_audio_cpp_exact_identifier_honors_the_configured_safety_bound() -> None:
    original = load_global_speech_tts_state(_settings(), environment={})
    draft = deepcopy(original)
    draft.defaults.model_mode = "exact"
    draft.defaults.model_id = "model-too-long"
    draft.providers["audio_cpp"]["max_identifier_characters"] = 8

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.provider_id == "defaults"
    assert error.value.field_id == "default_model"
    assert "model-too-long" not in str(error.value)


def test_clearing_optional_random_seed_emits_an_explicit_delete() -> None:
    settings = _settings()
    settings["COMPREHENSIVE_CONFIG_RAW"]["app_tts"][  # type: ignore[index]
        "CHATTERBOX_RANDOM_SEED"
    ] = 42
    original = load_global_speech_tts_state(settings, environment={})
    draft = deepcopy(original)
    draft.providers["chatterbox"]["random_seed"] = ""

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="chatterbox",
    )

    assert "CHATTERBOX_RANDOM_SEED" not in proposal.settings
    assert proposal.delete_setting_keys == ("CHATTERBOX_RANDOM_SEED",)
    assert proposal.changed_provider_ids == ("chatterbox",)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("base_url", "https://example.invalid/path"),
        ("connect_timeout_seconds", 0),
        ("max_response_bytes", -1),
    ),
)
def test_audio_cpp_validation_is_local_and_field_specific(
    field: str, value: object
) -> None:
    original = load_global_speech_tts_state(_settings(), environment={})
    draft = deepcopy(original)
    draft.providers["audio_cpp"][field] = value

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.provider_id == "audio_cpp"
    assert error.value.field_id == field
    assert str(value) not in str(error.value)


def test_path_syntax_validation_does_not_require_the_path_to_exist() -> None:
    original = load_global_speech_tts_state(_settings(), environment={})
    draft = deepcopy(original)
    draft.providers["higgs"]["model_path"] = "/not-installed-yet/model.gguf"

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="higgs",
    )

    assert proposal.changed_provider_ids == ("higgs",)


def test_openai_url_rejects_a_fragment_without_echoing_it() -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    rejected = "https://example.test/v1/audio/speech#not-sent"
    draft.providers["openai"]["base_url"] = rejected

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="openai",
        )

    assert error.value.provider_id == "openai"
    assert error.value.field_id == "base_url"
    assert rejected not in str(error.value)


def test_alltalk_server_url_rejects_a_query_without_echoing_it() -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    rejected = "http://127.0.0.1:7851?token=not-for-logs"
    draft.providers["alltalk"]["server_url"] = rejected

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="alltalk",
        )

    assert error.value.provider_id == "alltalk"
    assert error.value.field_id == "server_url"
    assert rejected not in str(error.value)


@pytest.mark.parametrize("provider_id", ("kokoro", "higgs"))
def test_existing_mps_device_values_round_trip(provider_id: str) -> None:
    settings = _settings()
    raw = settings["COMPREHENSIVE_CONFIG_RAW"]
    if provider_id == "kokoro":
        raw["app_tts"]["KOKORO_DEVICE_DEFAULT"] = "mps"  # type: ignore[index]
    else:
        raw["HiggsSettings"]["device"] = "mps"  # type: ignore[index]
    original = load_global_speech_tts_state(settings, environment={})
    draft = deepcopy(original)

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider=provider_id,
    )

    assert draft.providers[provider_id]["device"] == "mps"
    assert proposal.settings == {}


def test_restore_non_secret_defaults_never_changes_credential_state() -> None:
    state = load_global_speech_tts_state(_settings(), environment={})
    credentials = state.credentials

    restored = restore_non_secret_defaults(state, configure_provider="openai")

    assert restored.credentials == credentials
    assert restored.providers["openai"]["base_url"] == (
        "https://api.openai.com/v1/audio/speech"
    )
    assert restored.providers["openai"]["organization_id"] == ""


def test_credential_mutations_require_explicit_intent_and_never_accept_placeholders() -> (
    None
):
    state = load_global_speech_tts_state(_settings(), environment={})

    mutation = build_credential_mutation(
        state.credentials["openai"],
        CredentialIntent.REPLACE,
        "replacement-secret",
    )
    assert mutation == GlobalSpeechTTSCredentialMutation(
        provider_id="openai",
        setting_key="openai_api_key",
        value="replacement-secret",
        delete=False,
    )

    clear = build_credential_mutation(
        state.credentials["openai"],
        CredentialIntent.CLEAR,
        None,
    )
    assert clear.delete is True
    assert clear.value is None

    for placeholder in ("••••••••", "********", "<saved>", "", "x" * 4097):
        with pytest.raises(ValueError, match="credential value"):
            build_credential_mutation(
                state.credentials["openai"],
                CredentialIntent.REPLACE,
                placeholder,
            )
