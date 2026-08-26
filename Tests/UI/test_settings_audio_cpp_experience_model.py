from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os

import pytest

from tldw_chatbook.TTS import audio_cpp_guided_launch as guided_launch_module
from tldw_chatbook.UI.Screens import settings_speech_tts as settings_model
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSNativeCapabilityObservation,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppAcceptedPackage,
    AudioCppSafeModelProjection,
    AudioCppSettingsConfig,
)
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    AudioCppExactChoiceState,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    GlobalSpeechTTSValidationError,
    audio_cpp_transport_warning,
    build_global_speech_tts_save_proposal,
    load_global_speech_tts_state,
    project_audio_cpp_global_choices,
)


def test_audio_cpp_settings_inventory_exposes_both_explicit_modes() -> None:
    projected = AudioCppSettingsConfig().to_mapping()
    state = load_global_speech_tts_state({}, environment={})

    assert projected["mode"] == "external"
    assert state.providers["audio_cpp"] == projected
    assert set(GLOBAL_TTS_PROVIDER_FIELD_IDS["audio_cpp"]) == set(projected)
    assert "mode" in GLOBAL_TTS_PROVIDER_FIELD_IDS["audio_cpp"]


def _guided_package(root: str) -> AudioCppAcceptedPackage:
    return AudioCppAcceptedPackage(
        package_uuid="d3f6d610-6fd9-4cde-9ea7-cc5175ca445b",
        recipe_id="audio-cpp-0.5.1.supertonic.supertonic_3_orig",
        recipe_revision=1,
        package_variant="supertonic_3_orig",
        public_model_id="narrator",
        canonical_root=root,
        canonical_root_identity="1" * 64,
        configuration_identity="2" * 64,
        weight_identity="3" * 64,
        projection=AudioCppSafeModelProjection(
            family="supertonic",
            task="tts",
            mode="offline",
            model_relative_path="supertonic-3-orig.gguf",
        ),
    )


def test_guided_settings_load_save_and_reload_preserve_all_dormant_sources(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guided edits round-trip without erasing dormant External or JSON values."""

    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    package = _guided_package(str(tmp_path))
    raw = AudioCppSettingsConfig(
        mode="managed",
        base_url="https://external.example.test:8443",
        managed_setup_source="guided",
        managed_binary_path="/manual/audiocpp_server",
        managed_server_json_path="/manual/server.json",
        guided_binary_path=str(binary),
        guided_binary_source="manual",
        guided_packages=(package,),
        guided_default_model_id="narrator",
        guided_backend_preference="cpu",
        guided_device=0,
        guided_threads=4,
    ).to_mapping()
    settings = {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": raw}}}
    original = load_global_speech_tts_state(settings, environment={})
    draft = deepcopy(original)
    draft.providers["audio_cpp"]["guided_threads"] = 6
    manual_validator_calls: list[object] = []

    def reject_manual_validation(_config: object) -> None:
        manual_validator_calls.append(_config)
        raise AssertionError("Guided Save entered the user-JSON validator")

    monkeypatch.setattr(
        settings_model,
        "validate_audio_cpp_managed_launch",
        reject_manual_validation,
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="audio_cpp",
    )
    settings_model.validate_audio_cpp_managed_settings(draft.providers["audio_cpp"])
    reloaded = load_global_speech_tts_state(
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": proposal.settings}},
        environment={},
    )

    assert proposal.settings["audio_cpp"] == {
        **raw,
        "guided_threads": 6,
    }
    assert reloaded.providers["audio_cpp"] == proposal.settings["audio_cpp"]
    assert reloaded.providers["audio_cpp"]["base_url"] == (
        "https://external.example.test:8443"
    )
    assert reloaded.providers["audio_cpp"]["managed_server_json_path"] == (
        "/manual/server.json"
    )
    assert manual_validator_calls == []


def test_guided_save_rejects_backend_without_host_recipe_evidence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passive Save rejects a backend the exact host/package tuple cannot run."""

    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    package = _guided_package(str(tmp_path))
    values = AudioCppSettingsConfig(
        mode="managed",
        managed_setup_source="guided",
        guided_binary_path=str(binary),
        guided_packages=(package,),
        guided_default_model_id="narrator",
        guided_backend_preference="cuda",
    ).to_mapping()
    monkeypatch.setattr(guided_launch_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(guided_launch_module.platform, "machine", lambda: "arm64")

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        settings_model.validate_audio_cpp_managed_settings(values)

    assert error.value.field_id == "guided_backend_preference"
    assert str(error.value) == (
        "Choose Auto or a backend supported by every reviewed package on this host."
    )
    assert str(tmp_path) not in str(error.value)


def test_managed_load_retains_the_dormant_external_origin() -> None:
    raw = {
        "mode": "managed",
        "base_url": "https://external.example.test:8443",
        "managed_binary_path": "/opt/homebrew/bin/audiocpp_server",
        "managed_server_json_path": "/srv/audio/server.json",
        "managed_startup_timeout_seconds": 31.0,
        "managed_health_check_interval_seconds": 11.0,
        "managed_termination_grace_seconds": 6.0,
        "connect_timeout_seconds": 4.0,
        "synthesis_timeout_seconds": 90.0,
        "max_input_characters": 1001,
        "max_response_bytes": 1002,
        "max_metadata_bytes": 1003,
        "max_catalog_models": 1004,
        "max_voices_per_model": 1005,
        "max_identifier_characters": 1006,
    }

    state = load_global_speech_tts_state(
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": raw}}},
        environment={},
    )

    assert state.providers["audio_cpp"] == {
        **AudioCppSettingsConfig().to_mapping(),
        **raw,
    }


def test_managed_save_persists_the_external_origin_for_a_later_switch() -> None:
    original = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "audio_cpp": {
                        **AudioCppConfig().to_mapping(),
                        "base_url": "https://external.example.test:8443",
                    }
                }
            }
        },
        environment={},
    )
    draft = deepcopy(original)
    draft.providers["audio_cpp"].update(
        {
            "mode": "managed",
            "managed_binary_path": "/opt/homebrew/bin/audiocpp_server",
            "managed_server_json_path": "/srv/audio/server.json",
            "managed_startup_timeout_seconds": 30.0,
            "managed_health_check_interval_seconds": 10.0,
            "managed_termination_grace_seconds": 5.0,
        }
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="audio_cpp",
    )

    saved = proposal.settings["audio_cpp"]
    assert saved["mode"] == "managed"
    assert saved["base_url"] == "https://external.example.test:8443"
    assert saved["managed_binary_path"] == "/opt/homebrew/bin/audiocpp_server"
    assert saved["managed_server_json_path"] == "/srv/audio/server.json"


def test_external_save_retains_previously_saved_managed_values() -> None:
    managed = {
        **AudioCppConfig(
            mode="managed",
            managed_binary_path="/opt/homebrew/bin/audiocpp_server",
            managed_server_json_path="/srv/audio/server.json",
        ).to_mapping(),
        "base_url": "http://127.0.0.1:18080",
    }
    original = load_global_speech_tts_state(
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": managed}}},
        environment={},
    )
    draft = deepcopy(original)
    draft.providers["audio_cpp"].update(
        {"mode": "external", "base_url": "http://127.0.0.1:18081"}
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="audio_cpp",
    )

    saved = proposal.settings["audio_cpp"]
    assert saved["mode"] == "external"
    assert saved["base_url"] == "http://127.0.0.1:18081"
    assert saved["managed_binary_path"] == "/opt/homebrew/bin/audiocpp_server"
    assert saved["managed_server_json_path"] == "/srv/audio/server.json"


def test_invalid_dormant_managed_values_do_not_block_external_save() -> None:
    raw = {
        **AudioCppConfig().to_mapping(),
        "managed_binary_path": "relative/binary",
        "managed_server_json_path": "relative/server.json",
        "managed_startup_timeout_seconds": "not-a-number",
    }
    original = load_global_speech_tts_state(
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": raw}}},
        environment={},
    )
    draft = deepcopy(original)
    draft.providers["audio_cpp"]["base_url"] = "http://127.0.0.1:18082"

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="audio_cpp",
    )

    saved = proposal.settings["audio_cpp"]
    assert saved["managed_binary_path"] == "relative/binary"
    assert saved["managed_server_json_path"] == "relative/server.json"
    assert saved["managed_startup_timeout_seconds"] == "not-a-number"


def test_selecting_invalid_dormant_managed_values_reports_the_active_field() -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.providers["audio_cpp"].update(
        {
            "mode": "managed",
            "managed_binary_path": "/opt/homebrew/bin/audiocpp_server",
            "managed_server_json_path": "/srv/audio/server.json",
            "managed_startup_timeout_seconds": "not-a-number",
            "managed_health_check_interval_seconds": 10.0,
            "managed_termination_grace_seconds": 5.0,
        }
    )

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.provider_id == "audio_cpp"
    assert error.value.field_id == "managed_startup_timeout_seconds"
    assert "not-a-number" not in str(error.value)


def test_invalid_dormant_external_origin_does_not_block_managed_save() -> None:
    managed = {
        **AudioCppConfig(
            mode="managed",
            managed_binary_path="/opt/homebrew/bin/audiocpp_server",
            managed_server_json_path="/srv/audio/server.json",
        ).to_mapping(),
        "base_url": "https://[private-invalid.example.test",
    }
    original = load_global_speech_tts_state(
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": managed}}},
        environment={},
    )
    draft = deepcopy(original)
    draft.providers["audio_cpp"]["connect_timeout_seconds"] = 6.0

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="audio_cpp",
    )

    assert proposal.settings["audio_cpp"]["base_url"] == (
        "https://[private-invalid.example.test"
    )


def test_selecting_invalid_dormant_external_origin_reports_base_url() -> None:
    managed = {
        **AudioCppConfig(
            mode="managed",
            managed_binary_path="/opt/homebrew/bin/audiocpp_server",
            managed_server_json_path="/srv/audio/server.json",
        ).to_mapping(),
        "base_url": "https://[private-invalid.example.test",
    }
    original = load_global_speech_tts_state(
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": managed}}},
        environment={},
    )
    draft = deepcopy(original)
    draft.providers["audio_cpp"]["mode"] = "external"

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="audio_cpp",
        )

    assert error.value.field_id == "base_url"
    assert "private-invalid" not in str(error.value)


def test_detect_audio_cpp_binary_returns_the_exact_platform_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detected = "/opt/homebrew/bin/audiocpp_server"
    calls: list[str] = []

    def lookup(command: str) -> str:
        calls.append(command)
        return detected

    monkeypatch.setattr(settings_model.shutil, "which", lookup)

    assert settings_model.detect_audio_cpp_server_binary() == detected
    assert calls == ["audiocpp_server"]


def test_detect_audio_cpp_binary_returns_none_without_mutating_a_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_model.shutil, "which", lambda _command: None)

    assert settings_model.detect_audio_cpp_server_binary() is None


def _managed_values(binary_path: str, server_json_path: str) -> dict[str, object]:
    return {
        **AudioCppConfig(
            mode="managed",
            managed_binary_path=binary_path,
            managed_server_json_path=server_json_path,
        ).to_mapping(),
        "base_url": "http://127.0.0.1:8080",
    }


def test_managed_save_validation_reads_but_never_modifies_user_artifacts(
    tmp_path,
) -> None:
    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    server_json = tmp_path / "server.json"
    server_json.write_text('{"host":"127.0.0.1","port":19001}', encoding="utf-8")
    before_binary = binary.read_bytes()
    before_json = server_json.read_bytes()
    before_binary_mode = os.stat(binary).st_mode

    settings_model.validate_audio_cpp_managed_settings(
        _managed_values(str(binary), str(server_json))
    )

    assert binary.read_bytes() == before_binary
    assert server_json.read_bytes() == before_json
    assert os.stat(binary).st_mode == before_binary_mode


def test_managed_save_validation_maps_binary_failure_without_echoing_path(
    tmp_path,
) -> None:
    missing_binary = tmp_path / "private-missing-audiocpp_server"
    server_json = tmp_path / "server.json"
    server_json.write_text('{"host":"127.0.0.1","port":19002}', encoding="utf-8")

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        settings_model.validate_audio_cpp_managed_settings(
            _managed_values(str(missing_binary), str(server_json))
        )

    assert error.value.field_id == "managed_binary_path"
    assert str(error.value) == (
        "Choose an existing audiocpp_server file that is executable."
    )
    assert str(missing_binary) not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.parametrize(
    ("case", "expected_message"),
    (
        ("missing", "Choose an existing server.json file that is readable."),
        ("oversized", "server.json must be 1 MiB or smaller."),
        ("invalid_utf8", "server.json must use UTF-8 encoding."),
        (
            "invalid_json",
            "server.json must contain strict JSON with no duplicate keys or "
            "non-JSON values.",
        ),
        ("not_object", "server.json must contain one JSON object."),
        ("invalid_host", "server.json must set host exactly to 127.0.0.1."),
        (
            "invalid_port",
            "server.json must set port to a whole number from 1 through 65535.",
        ),
    ),
)
def test_managed_save_validation_reports_the_exact_server_json_failure(
    tmp_path,
    case: str,
    expected_message: str,
) -> None:
    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    server_json = tmp_path / "private-server.json"
    documents = {
        "invalid_utf8": b"\xff",
        "invalid_json": b'{"private-broken-json":',
        "not_object": b"[]",
        "invalid_host": b'{"host":"0.0.0.0","port":19003}',
        "invalid_port": b'{"host":"127.0.0.1","port":"private-port"}',
    }
    if case == "oversized":
        server_json.write_bytes(b"{" + (b" " * 1_048_576))
    elif case != "missing":
        server_json.write_bytes(documents[case])

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        settings_model.validate_audio_cpp_managed_settings(
            _managed_values(str(binary), str(server_json))
        )

    assert error.value.field_id == "managed_server_json_path"
    assert str(error.value) == expected_message
    assert str(server_json) not in str(error.value)
    assert "private-port" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.parametrize(
    "document",
    (
        '{"host":"0.0.0.0","port":19003}',
        '{"host":"127.0.0.1","port":"private-port"}',
        '{"private-broken-json":',
    ),
)
def test_managed_save_validation_maps_server_json_failure_without_echoing_content(
    tmp_path,
    document: str,
) -> None:
    binary = tmp_path / "audiocpp_server"
    binary.write_bytes(b"synthetic-binary")
    binary.chmod(0o700)
    server_json = tmp_path / "server.json"
    server_json.write_text(document, encoding="utf-8")

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        settings_model.validate_audio_cpp_managed_settings(
            _managed_values(str(binary), str(server_json))
        )

    assert error.value.field_id == "managed_server_json_path"
    assert document not in str(error.value)
    assert "private" not in str(error.value).lower()
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_external_save_validation_ignores_dormant_managed_artifacts() -> None:
    values = AudioCppConfig().to_mapping()
    values.update(
        {
            "managed_binary_path": "/private/missing/binary",
            "managed_server_json_path": "/private/missing/server.json",
            "managed_startup_timeout_seconds": "dormant-invalid",
        }
    )

    settings_model.validate_audio_cpp_managed_settings(values)


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
