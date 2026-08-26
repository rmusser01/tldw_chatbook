from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID
import tomllib

from loguru import logger
import pytest
import toml

from tldw_chatbook import config as config_module
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler
from tldw_chatbook.TTS import (
    ProviderHealth,
    STTSGeneratedAudio,
    STTSPlaygroundRequest,
    TTSModelInfo,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.adapter_types import TTSNativeCapabilityObservation
from tldw_chatbook.TTS.effective_settings import (
    TTSCharacterProfileSelection,
    TTSEffectiveSettingsResolver,
    TTSSelectionOverrides,
)
from tldw_chatbook.TTS.legacy_catalogs import legacy_catalog
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSLoadState,
    StudioTTSPreferenceStore,
    StudioTTSSelectionOverrides,
    StudioTTSWriteStatus,
)
from tldw_chatbook.TTS.TTS_Generation import TTSService
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    SpeechLocalDependencyAvailability,
    project_speech_tts_status,
    speech_tts_navigation_context,
    speech_tts_navigation_target_from_context,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConfigurationState,
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
)
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    build_global_speech_tts_save_proposal,
    load_global_speech_tts_state,
)


@pytest.fixture(autouse=True)
def _isolated_config_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_CACHE", None)
    monkeypatch.setattr(config_module, "_CONFIG_CACHE_SOURCE", None)
    monkeypatch.setattr(config_module, "_SETTINGS_CACHE", None)
    monkeypatch.setattr(config_module, "_SETTINGS_CACHE_SOURCE", None)
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 0)


def _write_config(path: Path, values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(dict(values)), encoding="utf-8")


def _read_config(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _studio_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    values: Mapping[str, object],
) -> tuple[StudioTTSPreferenceStore, Path]:
    path = tmp_path / "config.toml"
    _write_config(path, values)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    return StudioTTSPreferenceStore(), path


def _global_preferences(*, voice_id: str) -> TTSPreferencesSnapshot:
    return TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="tts-1-hd",
        voice_mode="exact",
        voice_id=voice_id,
        response_format="mp3",
        speed=1.0,
    )


def test_authentication_mode_save_never_projects_or_mutates_credentials() -> None:
    settings = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "app_tts": {
                "OPENAI_BASE_URL": "http://127.0.0.1:8765/v1/audio/speech",
                "OPENAI_AUTH_MODE": "api_key",
            },
            "api_settings": {"openai": {"api_key": "OWNERSHIP_SECRET_SENTINEL"}},
        }
    }
    original = load_global_speech_tts_state(settings, environment={})
    draft = replace(original, providers={**original.providers})
    draft.providers = {key: dict(value) for key, value in original.providers.items()}
    draft.providers["openai"]["authentication_mode"] = "none"

    proposal = build_global_speech_tts_save_proposal(
        original,
        draft,
        configure_provider="openai",
    )

    assert proposal.settings["OPENAI_AUTH_MODE"] == "none"
    assert "openai_api_key" not in proposal.settings
    assert "OWNERSHIP_SECRET_SENTINEL" not in repr(proposal)
    assert original.credentials == draft.credentials


async def _no_catalog(_provider_id: str) -> TTSProviderCatalog:
    raise AssertionError("complete exact legacy selections must not read a catalog")


@pytest.mark.asyncio
async def test_studio_save_reset_and_preview_preserve_other_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character_owner = {
        "assignment": "character-42/profile-7",
        "profile_revision": 3,
    }
    global_owner = {
        "default_provider": "openai",
        "default_model_mode": "exact",
        "default_model": "tts-1-hd",
        "default_voice_mode": "exact",
        "default_voice": "global-before",
        "default_format": "mp3",
        "default_speed": 1.0,
    }
    store, config_path = _studio_store(
        tmp_path,
        monkeypatch,
        {
            "app_tts": global_owner,
            "character_tts": character_owner,
            "API": {"openai_api_key": "global-owner-only"},
        },
    )
    reconfigure_calls: list[tuple[object, ...]] = []

    async def forbidden_reconfigure(*args: object, **kwargs: object) -> None:
        reconfigure_calls.append((*args, kwargs))
        raise AssertionError("Studio persistence must not reconfigure an adapter")

    monkeypatch.setattr(TTSService, "reconfigure_provider", forbidden_reconfigure)
    initial = store.load().snapshot
    candidate = replace(
        initial,
        selection=StudioTTSSelectionOverrides(
            voice_mode="exact",
            voice_id="studio-preview",
        ),
    )

    saved = store.save(candidate)

    assert saved.status is StudioTTSWriteStatus.SAVED
    assert saved.snapshot is not None
    persisted = _read_config(config_path)
    assert persisted["app_tts"] == global_owner
    assert persisted["character_tts"] == character_owner
    assert persisted["API"] == {"openai_api_key": "global-owner-only"}
    assert reconfigure_calls == []

    studio_effective = await TTSEffectiveSettingsResolver().resolve_studio(
        studio_preferences=saved.snapshot,
        global_preferences=_global_preferences(voice_id="global-before"),
        global_preferences_revision=4,
        provider_revision_reader=lambda _provider_id: 8,
        catalog_reader=_no_catalog,
    )
    assert studio_effective.voice_id == "studio-preview"

    reset = store.reset_to_global(saved.snapshot)

    assert reset.status is StudioTTSWriteStatus.SAVED
    assert reset.snapshot is not None
    after_reset = _read_config(config_path)
    assert after_reset["speech_studio"] == {
        "schema_version": 1,
        "revision": 2,
    }
    assert after_reset["app_tts"] == global_owner
    assert after_reset["character_tts"] == character_owner
    assert reconfigure_calls == []

    later_global = _global_preferences(voice_id="global-after")
    reset_effective = await TTSEffectiveSettingsResolver().resolve_studio(
        studio_preferences=reset.snapshot,
        global_preferences=later_global,
        global_preferences_revision=5,
        provider_revision_reader=lambda _provider_id: 8,
        catalog_reader=_no_catalog,
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="openai",
            model_mode="exact",
            model_id="tts-1-hd",
            voice_mode="exact",
            voice_id="character-voice",
            response_format="mp3",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=11,
        profile_revision=3,
        profile_id=UUID("11111111-1111-4111-8111-111111111111"),
    )
    roleplay_effective = await TTSEffectiveSettingsResolver().resolve_non_studio(
        character_profile=character,
        global_preferences=later_global,
        global_preferences_revision=5,
        provider_revision_reader=lambda _provider_id: 8,
        catalog_reader=_no_catalog,
    )

    assert reset_effective.voice_id == "global-after"
    assert roleplay_effective.voice_id == "character-voice"
    assert _read_config(config_path)["character_tts"] == character_owner


def test_migration_and_disabled_studio_reader_are_additive_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "app_tts": {
            "default_provider": "alltalk",
            "default_model_mode": "exact",
            "default_model": "alltalk-default",
            "default_voice_mode": "exact",
            "default_voice": "global-reader-voice.wav",
            "default_format": "wav",
            "default_speed": 1.0,
            "ALLTALK_TTS_VOICE_DEFAULT": "studio-migrated-voice.wav",
            "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT": "flac",
            "CHATTERBOX_EXAGGERATION": "malformed-independent-value",
            "CHATTERBOX_CFG_WEIGHT": 0.25,
        },
        "tts_settings": {"default_tts_provider": "alltalk"},
        "character_tts": {"assignment": "must-survive"},
    }
    older_reader_before = TTSPreferencesSnapshot.from_settings(raw)
    store, config_path = _studio_store(tmp_path, monkeypatch, raw)
    real_write = config_module.atomic_private_write_text
    writes = 0

    def counted_write(*args: object, **kwargs: object):
        nonlocal writes
        writes += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(config_module, "atomic_private_write_text", counted_write)

    first = store.load()
    second = store.load()

    assert first.state is StudioTTSLoadState.MIGRATED
    assert second.state is StudioTTSLoadState.LOADED
    assert first.snapshot == second.snapshot
    assert first.issues == ("app_tts.CHATTERBOX_EXAGGERATION",)
    assert first.snapshot.selection.voice_id == "studio-migrated-voice.wav"
    assert first.snapshot.selection.response_format == "flac"
    assert first.snapshot.provider_options == {"chatterbox": {"cfg_weight": 0.25}}
    assert writes == 1

    migrated = _read_config(config_path)
    older_reader_after = TTSPreferencesSnapshot.from_settings(migrated)
    assert older_reader_after == older_reader_before
    assert migrated["app_tts"] == raw["app_tts"]
    assert migrated["tts_settings"] == raw["tts_settings"]
    assert migrated["character_tts"] == raw["character_tts"]

    before_disabled_read = config_path.read_bytes()
    disabled_studio_effective = asyncio.run(
        TTSEffectiveSettingsResolver().resolve_non_studio(
            global_preferences=older_reader_after,
            global_preferences_revision=0,
            provider_revision_reader=lambda _provider_id: 0,
            catalog_reader=_no_catalog,
        )
    )
    assert disabled_studio_effective.provider_id == older_reader_before.provider_id
    assert disabled_studio_effective.voice_id == older_reader_before.voice_id
    assert config_path.read_bytes() == before_disabled_read


def test_privacy_sentinels_do_not_cross_owned_output_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = {
        "credential": "CREDENTIAL_SENTINEL_7be67a",
        "masked": "•••••••••••••••••••",
        "environment": "ENVIRONMENT_SENTINEL_a9f31c",
        "text": "SYNTHESIS_TEXT_SENTINEL_5d36e8",
        "body": "PROVIDER_BODY_SENTINEL_b412ac",
        "url": "https://user:pass@URL_SENTINEL.invalid/path?token=private",
        "exception": "EXCEPTION_SENTINEL_f8a39d",
    }
    monkeypatch.setenv("ELEVENLABS_API_KEY", sentinels["environment"])
    store, config_path = _studio_store(
        tmp_path,
        monkeypatch,
        {
            "app_tts": {
                "default_provider": "alltalk",
                "ALLTALK_TTS_VOICE_DEFAULT": sentinels["masked"],
                "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT": "flac",
                "ALLTALK_TTS_URL_DEFAULT": sentinels["url"],
            },
            "API": {"elevenlabs_api_key": sentinels["credential"]},
            "character_tts": {"assignment": "safe-character-assignment"},
        },
    )
    log_messages: list[str] = []
    sink_id = logger.add(log_messages.append, level="DEBUG", format="{message}")
    try:
        loaded = store.load()
    finally:
        logger.remove(sink_id)

    target = SpeechTTSNavigationTarget(
        "audio_cpp",
        SpeechTTSNavigationIntent.TEST,
    )
    tainted_catalog = TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=3,
        health=ProviderHealth(
            state="unavailable",
            fresh=True,
            diagnostic=sentinels["body"],
            recovery_action=sentinels["url"],
        ),
        models=(
            TTSModelInfo(
                model_id="safe-model",
                display_name=sentinels["exception"],
                family="native",
                upstream_mode="audio.cpp",
                formats=("wav",),
                voices=(),
                supports_speed=False,
            ),
        ),
    )
    tainted_observation = TTSNativeCapabilityObservation(
        snapshot=TTSNativeCapabilitySnapshot(
            provider_id="audio_cpp",
            configuration_revision=9,
            state="unverified",
            catalog=tainted_catalog,
            voice_results={},
        ),
        observed_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    assert sentinels["body"] in repr(tainted_observation)
    assert sentinels["exception"] in repr(tainted_observation)

    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=9,
        model_id="safe-model",
        observation=tainted_observation,
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
        current_runtime_revision=9,
        applied_configuration_revision=9,
    )
    tainted_context = {
        "provider": "audio_cpp",
        "intent": "test",
        "credential": sentinels["credential"],
        "text": sentinels["text"],
        "url": sentinels["url"],
    }
    assert speech_tts_navigation_target_from_context(tainted_context) is None
    requested_selection = TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id="safe-model",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
        configuration_revision=9,
    )
    artifact = STTSGeneratedAudio(
        path=tmp_path / "safe-artifact.wav",
        provider_id="audio_cpp",
        model_id="safe-model",
        voice_id=None,
        source_text=sentinels["text"],
        operation_id="safe-operation",
        audio_format="wav",
        content_type="audio/wav",
        metadata={"sample_rate": 8_000},
        requested_selection=requested_selection,
    )
    arbitrary_failure = RuntimeError(
        " ".join(
            (
                sentinels["exception"],
                sentinels["body"],
                sentinels["text"],
                sentinels["url"],
            )
        )
    )
    safe_outputs = (
        toml.dumps({"speech_studio": _read_config(config_path)["speech_studio"]}),
        repr(loaded.issues),
        repr(speech_tts_navigation_context(target)),
        repr(projection),
        repr(projection.rows()),
        repr(speech_tts_navigation_target_from_context(tainted_context)),
        STTSEventHandler._generation_error_copy(arbitrary_failure),
        repr(artifact.metadata),
        repr(artifact.requested_selection),
        "".join(log_messages),
    )
    rendered = "\n".join(safe_outputs)

    assert loaded.state is StudioTTSLoadState.MIGRATED
    assert loaded.issues == ("app_tts.ALLTALK_TTS_VOICE_DEFAULT",)
    assert artifact.source_text == sentinels["text"]  # approved transient payload
    for sentinel in sentinels.values():
        assert sentinel not in rendered


class _LegacyStreamService:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str, object]] = []
        self.native_calls = 0

    async def synthesize(self, *_args: object, **_kwargs: object) -> None:
        self.native_calls += 1
        raise AssertionError("legacy providers must remain behind the bridge")

    async def generate_audio_stream(
        self,
        request: object,
        internal_model_id: str,
        progress_sink: object = None,
    ) -> AsyncIterator[bytes]:
        self.calls.append((request, internal_model_id, progress_sink))
        yield b"RIFF"
        yield b"legacy-complete-response"


@pytest.mark.parametrize(
    ("provider_id", "model_id", "options", "expected_internal", "extra_params"),
    (
        ("openai", "tts-1-hd", {}, "openai_official_tts1hd", None),
        (
            "elevenlabs",
            "eleven_turbo_v2",
            {},
            "elevenlabs_eleven_turbo_v2",
            None,
        ),
        (
            "kokoro",
            "kokoro",
            {"use_onnx": False},
            "local_kokoro_default_pytorch",
            None,
        ),
        (
            "chatterbox",
            "chatterbox",
            {"exaggeration": 0.7},
            "local_chatterbox_default",
            {"exaggeration": 0.7},
        ),
        (
            "higgs",
            "higgs-v2",
            {"temperature": 0.8},
            "local_higgs_v2",
            {"temperature": 0.8},
        ),
        ("alltalk", "default", {}, "alltalk_default", None),
    ),
)
@pytest.mark.asyncio
async def test_every_legacy_provider_retains_its_accepted_request_shape(
    provider_id: str,
    model_id: str,
    options: Mapping[str, object],
    expected_internal: str,
    extra_params: Mapping[str, object] | None,
) -> None:
    service = _LegacyStreamService()
    handler = STTSEventHandler(
        app=SimpleNamespace(notify=lambda *_args, **_kwargs: None)
    )
    handler._stts_service = service
    request_snapshot = STTSPlaygroundRequest(
        operation_id=f"legacy-{provider_id}",
        provider_id=provider_id,
        model_id=model_id,
        text="synthetic legacy request",
        voice_id="alloy",
        response_format="wav",
        speed=1.0,
        options=options,
    )

    artifact = await handler._generate_legacy(request_snapshot, None)

    try:
        assert len(service.calls) == 1
        request, internal_model_id, progress_sink = service.calls[0]
        assert request.model == model_id
        assert request.input == "synthetic legacy request"
        assert request.voice == "alloy"
        assert request.response_format == "wav"
        assert request.speed == 1.0
        assert request.extra_params == extra_params
        assert internal_model_id == expected_internal
        assert progress_sink is None
        assert service.native_calls == 0
        assert artifact.provider_id == provider_id
        assert artifact.model_id == model_id
        assert artifact.audio_format == "wav"
        assert artifact.path.read_bytes() == b"RIFFlegacy-complete-response"
    finally:
        artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_approximate_legacy_catalog_cannot_invalidate_omitted_exact_model() -> (
    None
):
    catalog = legacy_catalog("openai")
    assert catalog.approximate is True
    assert all(model.model_id != "gpt-4o-mini-tts" for model in catalog.models)
    catalog_reads: list[str] = []

    async def read_catalog(provider_id: str) -> TTSProviderCatalog:
        catalog_reads.append(provider_id)
        return catalog

    exact = TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="gpt-4o-mini-tts",
        voice_mode="exact",
        voice_id="custom-exact-voice",
        response_format="mp3",
        speed=1.0,
    )

    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        global_preferences=exact,
        global_preferences_revision=6,
        provider_revision_reader=lambda _provider_id: 9,
        catalog_reader=read_catalog,
    )

    assert resolved.provider_id == "openai"
    assert resolved.model_id == "gpt-4o-mini-tts"
    assert resolved.voice_id == "custom-exact-voice"
    assert catalog_reads == []
