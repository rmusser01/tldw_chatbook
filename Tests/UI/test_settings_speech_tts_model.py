from __future__ import annotations

import io
import wave
from copy import deepcopy

import pytest

from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_guided_config import AudioCppSettingsConfig
from tldw_chatbook.TTS.openai_compatible_config import (
    normalize_openai_compatible_endpoint,
    openai_destination_fingerprint,
)
from tldw_chatbook.UI.Screens import settings_speech_tts as settings_speech_tts_module
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    CredentialIntent,
    CredentialSource,
    GlobalSpeechTTSCredentialMutation,
    GlobalSpeechTTSEffectiveSource,
    GlobalSpeechTTSValidationError,
    ProcessProviderTestEvidenceStore,
    build_credential_mutation,
    build_global_speech_tts_save_proposal,
    build_provider_test_fingerprint,
    global_speech_tts_provider_configuration_state,
    load_global_speech_tts_state,
    restore_non_secret_defaults,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConfigurationState,
    SpeechTTSConfigurationValidity,
    SpeechTTSConnectionState,
    SpeechTTSTestOperation,
    combine_tts_readiness,
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


def _wav_sample(frames: bytes = b"\x00\x00\x01\x00") -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(frames)
    return output.getvalue()


def _ogg_page(payload: bytes, *, sequence: int, header_type: int = 0) -> bytes:
    return b"".join(
        (
            b"OggS",
            b"\x00",
            bytes((header_type,)),
            b"\x00" * 8,
            b"\x01\x00\x00\x00",
            sequence.to_bytes(4, "little"),
            b"\x00" * 4,
            b"\x01",
            bytes((len(payload),)),
            payload,
        )
    )


def _malformed_sized_compressed_sample(response_format: str) -> bytes:
    if response_format == "mp3":
        return b"\xff\xfb\x90\x64" + b"\x00" * 413
    if response_format == "opus":
        opus_head = (
            b"OpusHead\x01\x01\x00\x00\x80\xbb\x00\x00\x00\x00\x00"
        )
        return _ogg_page(opus_head, sequence=0, header_type=2) + _ogg_page(
            b"\x00" * 20,
            sequence=1,
        )
    if response_format == "flac":
        stream_info = bytearray(34)
        packed = (44_100 << 44) | (15 << 36) | 1
        stream_info[10:18] = packed.to_bytes(8, "big")
        return (
            b"fLaC\x80\x00\x00\x22"
            + bytes(stream_info)
            + b"\xff\xf8"
            + b"\x00" * 20
        )
    if response_format == "aac":
        return b"\xff\xf1\x50\x80\x02\x9f\xfc" + b"\x00" * 13
    raise AssertionError(f"Unsupported test format: {response_format}")


def _pyav_encoded_sample(response_format: str) -> bytes:
    av = pytest.importorskip("av")
    container_format, sample_rate, samples, codec_names = {
        "mp3": ("mp3", 44_100, 1_152, ("libmp3lame", "mp3")),
        "opus": ("ogg", 48_000, 960, ("libopus", "opus")),
        "flac": ("flac", 44_100, 1_024, ("flac",)),
        "aac": ("adts", 44_100, 1_024, ("aac",)),
    }[response_format]
    errors: list[str] = []
    for codec_name in codec_names:
        output = io.BytesIO()
        try:
            codec = av.Codec(codec_name, "w")
            sample_format = next(
                audio_format.name
                for preferred in ("s16", "s16p", "flt", "fltp")
                for audio_format in codec.audio_formats
                if audio_format.name == preferred
            )
            with av.open(output, mode="w", format=container_format) as container:
                stream = container.add_stream(codec_name, rate=sample_rate)
                stream.layout = "mono"
                stream.codec_context.format = sample_format
                frame = av.AudioFrame(
                    format=sample_format,
                    layout="mono",
                    samples=samples,
                )
                frame.sample_rate = sample_rate
                frame.pts = 0
                for plane in frame.planes:
                    plane.update(bytes(plane.buffer_size))
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            payload = output.getvalue()
            if payload:
                return payload
        except Exception as error:  # pragma: no cover - codec-build dependent
            errors.append(f"{codec_name}: {error}")
    pytest.skip(
        f"No usable PyAV {response_format} encoder in this build: {'; '.join(errors)}"
    )


def _wav_with_declared_data_size(size: int) -> bytes:
    body = bytearray(_wav_sample())
    data_size_offset = body.index(b"data") + 4
    body[data_size_offset : data_size_offset + 4] = size.to_bytes(4, "little")
    return bytes(body)


def test_successful_sample_outranks_unsupported_catalog() -> None:
    readiness = combine_tts_readiness(
        configuration="valid",
        catalog="unsupported",
        sample="success",
    )

    assert readiness.configuration is SpeechTTSConfigurationValidity.VALID
    assert readiness.connection is SpeechTTSConnectionState.REACHABLE
    assert readiness.catalog is SpeechTTSConnectionState.UNSUPPORTED
    assert readiness.sample is SpeechTTSConnectionState.REACHABLE


def test_readiness_keeps_configuration_and_connection_independent() -> None:
    readiness = combine_tts_readiness(
        configuration=SpeechTTSConfigurationState.INVALID,
        catalog=SpeechTTSConnectionState.REACHABLE,
        sample=SpeechTTSConnectionState.NOT_TESTED,
    )

    assert readiness.configuration is SpeechTTSConfigurationValidity.INVALID
    assert readiness.connection is SpeechTTSConnectionState.REACHABLE


@pytest.mark.parametrize(
    ("sample", "expected"),
    (
        ("failure", SpeechTTSConnectionState.UNREACHABLE),
        ("unsupported", SpeechTTSConnectionState.UNSUPPORTED),
    ),
)
def test_completed_sample_outcome_outranks_reachable_catalog(
    sample: str,
    expected: SpeechTTSConnectionState,
) -> None:
    readiness = combine_tts_readiness(
        configuration="valid",
        catalog="reachable",
        sample=sample,
    )

    assert readiness.connection is expected


def test_provider_test_fingerprint_is_secret_free_and_revision_bound() -> None:
    state = load_global_speech_tts_state(
        _settings(),
        environment={"OPENAI_API_KEY": "environment-secret"},
    )
    state.providers["openai"]["credential"] = "draft-secret"

    initial = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=7,
    )
    same = build_provider_test_fingerprint(
        deepcopy(state),
        provider_id="openai",
        saved_revision=7,
    )
    changed_revision = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=8,
    )
    state.providers["openai"]["base_url"] = "http://127.0.0.1:8765/v1"
    changed_endpoint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=7,
    )

    assert initial == same
    assert initial != changed_revision
    assert initial != changed_endpoint
    assert ("credential_present", "true") in initial.normalized_fields
    assert ("authentication_mode", "api_key") in initial.normalized_fields
    rendered = repr(initial)
    assert "environment-secret" not in rendered
    assert "draft-secret" not in rendered
    assert "credential" not in initial.digest
    assert len(initial.digest) == 64


def test_process_evidence_preserves_exact_unchanged_fingerprint_only() -> None:
    state = load_global_speech_tts_state(_settings(), environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=3,
    )
    store = ProcessProviderTestEvidenceStore()

    assert store.record_successful_sample(
        fingerprint,
        status_code=200,
        response_format="wav",
        body=_wav_sample(),
    )
    assert store.sample_state(fingerprint) is SpeechTTSConnectionState.REACHABLE
    unchanged = build_provider_test_fingerprint(
        deepcopy(state),
        provider_id="openai",
        saved_revision=3,
    )
    assert store.sample_state(unchanged) is SpeechTTSConnectionState.REACHABLE
    assert (
        store.sample_state(
            build_provider_test_fingerprint(
                state,
                provider_id="openai",
                saved_revision=4,
            )
        )
        is SpeechTTSConnectionState.NOT_TESTED
    )
    assert (
        ProcessProviderTestEvidenceStore().sample_state(fingerprint)
        is SpeechTTSConnectionState.NOT_TESTED
    )


def test_process_evidence_keeps_only_latest_fingerprint_per_provider() -> None:
    state = load_global_speech_tts_state(_settings(), environment={})
    first = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=1,
    )
    second = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=2,
    )
    store = ProcessProviderTestEvidenceStore()

    assert store.record_successful_sample(
        first,
        status_code=200,
        response_format="wav",
        body=_wav_sample(),
    )
    assert store.record_successful_sample(
        second,
        status_code=200,
        response_format="wav",
        body=_wav_sample(),
    )

    assert store.sample_state(first) is SpeechTTSConnectionState.NOT_TESTED
    assert store.sample_state(second) is SpeechTTSConnectionState.REACHABLE


def test_provider_fingerprint_tracks_authentication_and_credential_presence() -> None:
    missing = load_global_speech_tts_state({}, environment={})
    configured = load_global_speech_tts_state(
        {},
        environment={"OPENAI_API_KEY": "first-secret"},
    )
    replaced = load_global_speech_tts_state(
        {},
        environment={"OPENAI_API_KEY": "different-secret"},
    )
    none_auth = deepcopy(configured)
    none_auth.providers["openai"].update(
        {
            "base_url": "http://127.0.0.1:8765/v1/audio/speech",
            "authentication_mode": "none",
        }
    )

    missing_fingerprint = build_provider_test_fingerprint(
        missing,
        provider_id="openai",
        saved_revision=1,
    )
    configured_fingerprint = build_provider_test_fingerprint(
        configured,
        provider_id="openai",
        saved_revision=1,
    )
    replaced_fingerprint = build_provider_test_fingerprint(
        replaced,
        provider_id="openai",
        saved_revision=1,
    )
    none_fingerprint = build_provider_test_fingerprint(
        none_auth,
        provider_id="openai",
        saved_revision=1,
    )

    assert missing_fingerprint != configured_fingerprint
    assert configured_fingerprint == replaced_fingerprint
    assert configured_fingerprint != none_fingerprint
    assert "first-secret" not in repr(configured_fingerprint)
    assert "different-secret" not in repr(replaced_fingerprint)


@pytest.mark.parametrize(
    ("status_code", "response_format", "body", "max_bytes"),
    (
        (500, "wav", _wav_sample(), 1024),
        (200, "wav", b"not audio", 1024),
        (200, "wav", _wav_sample(), 8),
        (200, "unknown", _wav_sample(), 1024),
    ),
)
def test_sample_evidence_requires_bounded_successful_format_valid_response(
    status_code: int,
    response_format: str,
    body: bytes,
    max_bytes: int,
) -> None:
    state = load_global_speech_tts_state(_settings(), environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=3,
    )
    store = ProcessProviderTestEvidenceStore()

    assert not store.record_successful_sample(
        fingerprint,
        status_code=status_code,
        response_format=response_format,
        body=body,
        max_bytes=max_bytes,
    )
    assert store.sample_state(fingerprint) is SpeechTTSConnectionState.NOT_TESTED
    assert store.sample_operation(fingerprint) is SpeechTTSTestOperation.SAMPLE


@pytest.mark.parametrize(
    "body",
    (
        b"RIFF\x04\x00\x00\x00WAVE",
        _wav_sample(b""),
        _wav_sample()[:-1],
        _wav_with_declared_data_size(128),
    ),
    ids=("header-only", "no-data", "truncated", "malformed-data-size"),
)
def test_wav_sample_evidence_requires_complete_non_empty_audio(body: bytes) -> None:
    state = load_global_speech_tts_state(_settings(), environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=3,
    )
    store = ProcessProviderTestEvidenceStore()

    assert not store.record_successful_sample(
        fingerprint,
        status_code=200,
        response_format="wav",
        content_type="audio/wav",
        body=body,
    )
    assert store.sample_state(fingerprint) is SpeechTTSConnectionState.NOT_TESTED


def test_raw_pcm_sample_requires_authoritative_frame_metadata() -> None:
    state = load_global_speech_tts_state(_settings(), environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=3,
    )
    store = ProcessProviderTestEvidenceStore()
    one_frame = b"\x00\x00\x00\x00"

    assert not store.record_successful_sample(
        fingerprint,
        status_code=200,
        response_format="pcm",
        content_type="audio/pcm",
        body=one_frame,
    )
    assert not store.record_successful_sample(
        fingerprint,
        status_code=200,
        response_format="pcm",
        content_type="audio/pcm",
        body=one_frame[:-1],
        sample_rate_hz=24_000,
        channels=2,
        sample_width_bytes=2,
    )
    assert store.record_successful_sample(
        fingerprint,
        status_code=200,
        response_format="pcm",
        content_type="audio/pcm",
        body=one_frame,
        sample_rate_hz=24_000,
        channels=2,
        sample_width_bytes=2,
    )


@pytest.mark.parametrize(
    ("response_format", "content_type", "body"),
    (
        ("mp3", "audio/mpeg", b"ID3\x04\x00\x00\x00\x00\x00\x00"),
        ("mp3", "audio/mpeg", b"\xff\xfb"),
        ("flac", "audio/flac", b"fLaC"),
        ("opus", "audio/ogg", b"OggS" + b"\x00" * 23 + b"OpusHead"),
        ("aac", "audio/aac", b"\xff\xf1"),
    ),
)
def test_compressed_sample_evidence_rejects_magic_only_or_truncated_audio(
    response_format: str,
    content_type: str,
    body: bytes,
) -> None:
    state = load_global_speech_tts_state(_settings(), environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=3,
    )
    store = ProcessProviderTestEvidenceStore()

    assert not store.record_successful_sample(
        fingerprint,
        status_code=200,
        response_format=response_format,
        content_type=content_type,
        body=body,
    )
    assert store.sample_state(fingerprint) is SpeechTTSConnectionState.NOT_TESTED


@pytest.mark.parametrize(
    ("response_format", "content_type"),
    (
        ("mp3", "audio/mpeg"),
        ("opus", "audio/ogg"),
        ("flac", "audio/flac"),
        ("aac", "audio/aac"),
    ),
)
def test_compressed_sample_evidence_rejects_sized_but_undecodable_frames(
    response_format: str,
    content_type: str,
) -> None:
    state = load_global_speech_tts_state(_settings(), environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=3,
    )
    store = ProcessProviderTestEvidenceStore()

    assert not store.record_successful_sample(
        fingerprint,
        status_code=200,
        response_format=response_format,
        content_type=content_type,
        body=_malformed_sized_compressed_sample(response_format),
    )
    assert store.sample_state(fingerprint) is SpeechTTSConnectionState.NOT_TESTED


@pytest.mark.parametrize(
    ("response_format", "content_type"),
    (
        ("mp3", "audio/mpeg"),
        ("opus", "audio/ogg"),
        ("flac", "audio/flac"),
        ("aac", "audio/aac"),
    ),
)
def test_compressed_sample_evidence_accepts_a_decoder_produced_audio_frame(
    response_format: str,
    content_type: str,
) -> None:
    state = load_global_speech_tts_state(_settings(), environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=3,
    )
    store = ProcessProviderTestEvidenceStore()

    assert store.record_successful_sample(
        fingerprint,
        status_code=200,
        response_format=response_format,
        content_type=content_type,
        body=_pyav_encoded_sample(response_format),
    )
    assert store.sample_state(fingerprint) is SpeechTTSConnectionState.REACHABLE


def test_global_field_inventory_is_bounded_complete_and_includes_managed_audio_cpp() -> (
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
        "mode",
        "base_url",
        "managed_binary_path",
        "managed_server_json_path",
        "managed_startup_timeout_seconds",
        "managed_health_check_interval_seconds",
        "managed_termination_grace_seconds",
        "connect_timeout_seconds",
        "synthesis_timeout_seconds",
        "max_input_characters",
        "max_response_bytes",
        "max_metadata_bytes",
        "max_catalog_models",
        "max_voices_per_model",
        "max_identifier_characters",
    } <= set(GLOBAL_TTS_PROVIDER_FIELD_IDS["audio_cpp"])
    assert {
        "credential",
        "authentication_mode",
        "base_url",
        "organization_id",
    } <= set(GLOBAL_TTS_PROVIDER_FIELD_IDS["openai"])
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
        "bind",
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


@pytest.mark.parametrize("raw_mode", (None, "", "token", 7, False))
def test_missing_or_invalid_openai_authentication_loads_fail_closed(
    raw_mode: object,
) -> None:
    app_tts: dict[str, object] = {}
    if raw_mode is not None:
        app_tts["OPENAI_AUTH_MODE"] = raw_mode

    state = load_global_speech_tts_state(
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": app_tts}},
        environment={},
    )

    assert state.providers["openai"]["authentication_mode"] == "api_key"


def test_openai_none_auth_is_complete_without_a_credential_and_saves_mode() -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.providers["openai"].update(
        {
            "base_url": "http://127.0.0.1:8765/v1/audio/speech",
            "authentication_mode": "none",
        }
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="openai",
    )

    assert proposal.settings["OPENAI_AUTH_MODE"] == "none"
    assert "openai_api_key" not in proposal.settings
    assert (
        global_speech_tts_provider_configuration_state(draft, provider_id="openai")
        is not SpeechTTSConfigurationState.INCOMPLETE
    )


def test_official_openai_rejects_none_and_loads_existing_none_fail_closed() -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.providers["openai"]["authentication_mode"] = "none"

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="openai",
        )

    assert error.value.field_id == "authentication_mode"
    loaded = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": "https://api.openai.com/v1/audio/speech",
                    "OPENAI_AUTH_MODE": "none",
                }
            }
        },
        environment={},
    )
    assert loaded.providers["openai"]["authentication_mode"] == "api_key"


def test_plaintext_none_confirmation_is_origin_bound_and_non_secret() -> None:
    endpoint = normalize_openai_compatible_endpoint(
        "http://voice.example.test:8765/v1/audio/speech"
    )
    fingerprint = openai_destination_fingerprint("openai", endpoint)
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.providers["openai"].update(
        {"base_url": endpoint.speech_url, "authentication_mode": "none"}
    )

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider="openai",
        )
    assert error.value.field_id == "authentication_mode"

    draft.openai_plaintext_confirmation = (
        settings_speech_tts_module.OpenAIPlaintextConfirmation(fingerprint)
    )
    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="openai",
    )

    assert proposal.settings["OPENAI_NONE_HTTP_CONFIRMATION"] == fingerprint
    assert proposal.settings["OPENAI_NONE_HTTP_CONFIRMATION"] != endpoint.origin


def test_plaintext_none_confirmation_survives_path_change_but_not_origin_or_auth() -> (
    None
):
    endpoint = normalize_openai_compatible_endpoint("http://voice.example.test:8765/v1")
    fingerprint = openai_destination_fingerprint("openai", endpoint)
    settings = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "app_tts": {
                "OPENAI_BASE_URL": endpoint.speech_url,
                "OPENAI_AUTH_MODE": "none",
                "OPENAI_NONE_HTTP_CONFIRMATION": fingerprint,
            }
        }
    }
    original = load_global_speech_tts_state(settings, environment={})
    assert original.openai_plaintext_confirmation is not None
    assert original.openai_plaintext_confirmation_cleanup_needed is False
    path_draft = deepcopy(original)
    path_draft.providers["openai"]["base_url"] = (
        "http://voice.example.test:8765/custom/speech"
    )

    path_proposal = build_global_speech_tts_save_proposal(
        original,
        path_draft,
        configure_provider="openai",
    )
    assert "OPENAI_NONE_HTTP_CONFIRMATION" not in path_proposal.delete_setting_keys

    origin_draft = deepcopy(original)
    origin_draft.providers["openai"]["base_url"] = (
        "http://other.example.test:8765/v1/audio/speech"
    )
    with pytest.raises(GlobalSpeechTTSValidationError):
        build_global_speech_tts_save_proposal(
            original,
            origin_draft,
            configure_provider="openai",
        )

    auth_draft = deepcopy(original)
    auth_draft.providers["openai"]["authentication_mode"] = "api_key"
    auth_proposal = build_global_speech_tts_save_proposal(
        original,
        auth_draft,
        configure_provider="openai",
    )
    assert "OPENAI_NONE_HTTP_CONFIRMATION" in auth_proposal.delete_setting_keys


@pytest.mark.parametrize(
    ("authentication_mode", "base_url"),
    (
        ("api_key", "http://voice.example.test:8765/v1/audio/speech"),
        ("none", "http://other.example.test:8765/v1/audio/speech"),
        ("none", "https://voice.example.test:8765/v1/audio/speech"),
        ("none", "http://127.0.0.1:8765/v1/audio/speech"),
        ("none", "not-an-endpoint"),
    ),
)
def test_load_invalidates_confirmation_when_auth_or_origin_changes(
    authentication_mode: str,
    base_url: str,
) -> None:
    confirmed = normalize_openai_compatible_endpoint(
        "http://voice.example.test:8765/v1/audio/speech"
    )
    fingerprint = openai_destination_fingerprint("openai", confirmed)

    settings = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "app_tts": {
                "OPENAI_BASE_URL": base_url,
                "OPENAI_AUTH_MODE": authentication_mode,
                "OPENAI_NONE_HTTP_CONFIRMATION": fingerprint,
            }
        }
    }
    before_load = deepcopy(settings)

    state = load_global_speech_tts_state(settings, environment={})

    assert settings == before_load
    assert state.openai_plaintext_confirmation is None
    assert state.openai_plaintext_confirmation_cleanup_needed is True


@pytest.mark.parametrize(
    ("authentication_mode", "base_url", "stored_confirmation"),
    (
        (
            "api_key",
            "http://voice.example.test:8765/v1/audio/speech",
            "confirmed-origin",
        ),
        (
            "none",
            "https://voice.example.test:8765/v1/audio/speech",
            "confirmed-origin",
        ),
        (
            "none",
            "http://127.0.0.1:8765/v1/audio/speech",
            "confirmed-origin",
        ),
        (
            "api_key",
            "http://voice.example.test:8765/v1/audio/speech",
            "malformed",
        ),
    ),
)
def test_rejected_persisted_confirmation_is_deleted_on_next_explicit_save(
    authentication_mode: str,
    base_url: str,
    stored_confirmation: str,
) -> None:
    confirmed = normalize_openai_compatible_endpoint(
        "http://voice.example.test:8765/v1/audio/speech"
    )
    persisted_value = (
        openai_destination_fingerprint("openai", confirmed)
        if stored_confirmation == "confirmed-origin"
        else stored_confirmation
    )
    original = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": base_url,
                    "OPENAI_AUTH_MODE": authentication_mode,
                    "OPENAI_NONE_HTTP_CONFIRMATION": persisted_value,
                }
            }
        },
        environment={},
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        deepcopy(original),
        configure_provider="openai",
    )

    assert original.openai_plaintext_confirmation is None
    assert proposal.delete_setting_keys == ("OPENAI_NONE_HTTP_CONFIRMATION",)


def test_absent_confirmation_key_does_not_request_needless_delete() -> None:
    original = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": "https://voice.example.test/v1/audio/speech",
                    "OPENAI_AUTH_MODE": "none",
                }
            }
        },
        environment={},
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        deepcopy(original),
        configure_provider="openai",
    )

    assert original.openai_plaintext_confirmation_cleanup_needed is False
    assert "OPENAI_NONE_HTTP_CONFIRMATION" not in proposal.delete_setting_keys


def test_normalized_confirmation_projection_does_not_claim_persisted_cleanup() -> None:
    original = load_global_speech_tts_state(
        {
            "APP_TTS_CONFIG": {
                "OPENAI_BASE_URL": "https://voice.example.test/v1/audio/speech",
                "OPENAI_AUTH_MODE": "none",
                "OPENAI_NONE_HTTP_CONFIRMATION": "normalized-only",
            }
        },
        environment={},
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        deepcopy(original),
        configure_provider="openai",
    )

    assert original.openai_plaintext_confirmation_cleanup_needed is False
    assert "OPENAI_NONE_HTTP_CONFIRMATION" not in proposal.delete_setting_keys


def test_changing_confirmed_origin_to_https_deletes_old_confirmation() -> None:
    endpoint = normalize_openai_compatible_endpoint(
        "http://voice.example.test:8765/v1/audio/speech"
    )
    fingerprint = openai_destination_fingerprint("openai", endpoint)
    original = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": endpoint.speech_url,
                    "OPENAI_AUTH_MODE": "none",
                    "OPENAI_NONE_HTTP_CONFIRMATION": fingerprint,
                }
            }
        },
        environment={},
    )
    draft = deepcopy(original)
    draft.providers["openai"]["base_url"] = "https://other.example.test/v1/audio/speech"

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="openai",
    )

    assert proposal.delete_setting_keys == ("OPENAI_NONE_HTTP_CONFIRMATION",)


def test_correcting_invalid_endpoint_deletes_rejected_persisted_confirmation() -> None:
    original = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": "not-an-endpoint",
                    "OPENAI_AUTH_MODE": "none",
                    "OPENAI_NONE_HTTP_CONFIRMATION": "malformed",
                }
            }
        },
        environment={},
    )
    draft = deepcopy(original)
    draft.providers["openai"].update(
        {
            "base_url": "https://api.openai.com/v1/audio/speech",
            "authentication_mode": "api_key",
        }
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="openai",
    )

    assert proposal.settings["OPENAI_AUTH_MODE"] == "api_key"
    assert proposal.delete_setting_keys == ("OPENAI_NONE_HTTP_CONFIRMATION",)


def test_cross_provider_save_owns_only_selected_fields_and_global_cleanup() -> None:
    confirmed = normalize_openai_compatible_endpoint(
        "http://voice.example.test:8765/v1/audio/speech"
    )
    fingerprint = openai_destination_fingerprint("openai", confirmed)
    original = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": "https://api.openai.com/v1/audio/speech",
                    "OPENAI_AUTH_MODE": "api_key",
                    "OPENAI_NONE_HTTP_CONFIRMATION": fingerprint,
                }
            }
        },
        environment={},
    )
    draft = deepcopy(original)
    draft.providers["elevenlabs"]["stability"] = 0.7

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="elevenlabs",
    )

    assert proposal.changed_provider_ids == ("elevenlabs",)
    assert proposal.settings["ELEVENLABS_VOICE_STABILITY"] == 0.7
    assert all(not key.startswith("OPENAI_") for key in proposal.settings)
    assert "openai_api_key" not in proposal.settings
    assert proposal.delete_setting_keys == ("OPENAI_NONE_HTTP_CONFIRMATION",)


def test_cross_provider_cleanup_only_does_not_claim_provider_fields() -> None:
    original = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_AUTH_MODE": "api_key",
                    "OPENAI_NONE_HTTP_CONFIRMATION": "malformed",
                }
            }
        },
        environment={},
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        deepcopy(original),
        configure_provider="elevenlabs",
    )

    assert proposal.settings == {}
    assert proposal.changed_provider_ids == ()
    assert proposal.delete_setting_keys == ("OPENAI_NONE_HTTP_CONFIRMATION",)


def test_stale_confirmation_cannot_reactivate_after_cross_provider_cleanup() -> None:
    prior_endpoint = normalize_openai_compatible_endpoint(
        "http://prior.example.test:8765/v1/audio/speech"
    )
    fingerprint = openai_destination_fingerprint("openai", prior_endpoint)
    original = load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "OPENAI_BASE_URL": (
                        "http://current.example.test:8765/v1/audio/speech"
                    ),
                    "OPENAI_AUTH_MODE": "none",
                    "OPENAI_NONE_HTTP_CONFIRMATION": fingerprint,
                }
            }
        },
        environment={},
    )
    cleaned = deepcopy(original)
    original.openai_plaintext_confirmation_cleanup_needed = False
    cleaned.openai_plaintext_confirmation_cleanup_needed = False
    cleaned.providers["openai"]["base_url"] = prior_endpoint.speech_url

    with pytest.raises(GlobalSpeechTTSValidationError) as error:
        build_global_speech_tts_save_proposal(
            original,
            cleaned,
            configure_provider="openai",
        )

    assert error.value.field_id == "authentication_mode"
    assert cleaned.openai_plaintext_confirmation is None


@pytest.mark.parametrize(
    "base_url",
    (
        "http://127.0.0.1:8765/v1/audio/speech",
        "http://[::1]:8765/v1/audio/speech",
        "https://voice.example.test/v1/audio/speech",
    ),
)
def test_none_auth_needs_no_plaintext_confirmation_for_safe_transports(
    base_url: str,
) -> None:
    original = load_global_speech_tts_state({}, environment={})
    draft = deepcopy(original)
    draft.providers["openai"].update(
        {"base_url": base_url, "authentication_mode": "none"}
    )

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="openai",
    )

    assert proposal.settings["OPENAI_AUTH_MODE"] == "none"
    assert "OPENAI_NONE_HTTP_CONFIRMATION" not in proposal.settings


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
            **AudioCppSettingsConfig().to_mapping(),
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


def test_restore_non_secret_defaults_preserves_a_set_default_profile_id() -> None:
    settings = _settings()
    settings["COMPREHENSIVE_CONFIG_RAW"]["app_tts"][  # type: ignore[index]
        "default_profile_id"
    ] = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    state = load_global_speech_tts_state(settings, environment={})

    restored = restore_non_secret_defaults(state, configure_provider="openai")

    # default_profile_id is a distinct precedence rung above the raw defaults
    # axes: restoring the axes must not silently drop the saved default voice.
    assert restored.defaults.default_profile_id == (
        "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    )
    # ...while the axes this function is actually meant to reset still reset
    # (original fixture is provider "audio_cpp" / voice_mode "server_default";
    # TTSPreferencesSnapshot.from_settings({}) defaults to "openai" / "exact").
    assert restored.defaults.provider_id == "openai"
    assert restored.defaults.voice_mode == "exact"


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


def test_default_profile_id_round_trips_from_settings() -> None:
    settings = _settings()
    settings["COMPREHENSIVE_CONFIG_RAW"]["app_tts"][  # type: ignore[index]
        "default_profile_id"
    ] = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

    state = load_global_speech_tts_state(settings, environment={})

    assert state.defaults.default_profile_id == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


def test_absent_default_profile_id_loads_as_none() -> None:
    state = load_global_speech_tts_state(_settings(), environment={})

    assert state.defaults.default_profile_id is None


def test_setting_default_profile_id_lands_in_save_settings() -> None:
    original = load_global_speech_tts_state(_settings(), environment={})
    draft = deepcopy(original)
    draft.defaults.default_profile_id = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

    proposal = build_global_speech_tts_save_proposal(
        original, draft, configure_provider="audio_cpp"
    )

    assert proposal.settings["default_profile_id"] == (
        "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    )


def test_clearing_default_profile_id_deletes_the_key() -> None:
    settings = _settings()
    settings["COMPREHENSIVE_CONFIG_RAW"]["app_tts"][  # type: ignore[index]
        "default_profile_id"
    ] = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    original = load_global_speech_tts_state(settings, environment={})
    draft = deepcopy(original)
    draft.defaults.default_profile_id = None

    proposal = build_global_speech_tts_save_proposal(
        original, draft, configure_provider="audio_cpp"
    )

    assert "default_profile_id" in proposal.delete_setting_keys
    assert "default_profile_id" not in proposal.settings


def test_unchanged_default_profile_id_is_not_written() -> None:
    settings = _settings()
    settings["COMPREHENSIVE_CONFIG_RAW"]["app_tts"][  # type: ignore[index]
        "default_profile_id"
    ] = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    original = load_global_speech_tts_state(settings, environment={})
    draft = deepcopy(original)

    proposal = build_global_speech_tts_save_proposal(
        original, draft, configure_provider="audio_cpp"
    )

    assert "default_profile_id" not in proposal.settings
    assert "default_profile_id" not in proposal.delete_setting_keys


@pytest.mark.parametrize("blank_value", ("", "   ", "\t\n"))
def test_blank_default_profile_id_loads_as_none(blank_value: str) -> None:
    settings = _settings()
    settings["COMPREHENSIVE_CONFIG_RAW"]["app_tts"][  # type: ignore[index]
        "default_profile_id"
    ] = blank_value

    state = load_global_speech_tts_state(settings, environment={})

    assert state.defaults.default_profile_id is None


def test_malformed_default_profile_id_still_loads_unchanged() -> None:
    settings = _settings()
    settings["COMPREHENSIVE_CONFIG_RAW"]["app_tts"][  # type: ignore[index]
        "default_profile_id"
    ] = "not-a-uuid"

    state = load_global_speech_tts_state(settings, environment={})

    # A malformed value is a defined dangling state, not a load-time error:
    # it must survive the loader unchanged so a later task can surface it
    # honestly and refuse it at speak time, rather than the loader silently
    # discarding it here.
    assert state.defaults.default_profile_id == "not-a-uuid"
