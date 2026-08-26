from __future__ import annotations

from dataclasses import fields
from uuid import UUID

import pytest

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.effective_settings import (
    TTSCharacterProfileSelection,
    TTSDefaultProfileSelection,
    TTSEffectiveResolutionError,
    TTSEffectiveSelectionRevisions,
    TTSEffectiveSelectionSnapshot,
    TTSEffectiveSettingsResolver,
    TTSSelectionOverrides,
    TTSSelectionSource,
    TTSStudioDraftSelection,
    tts_configuration_is_active,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSPreferencesSnapshot,
    StudioTTSSelectionOverrides,
)

_CHARACTER_PROFILE_ID = UUID("11111111-1111-4111-8111-111111111111")
_DEFAULT_PROFILE_ID = UUID("22222222-2222-4222-8222-222222222222")


@pytest.mark.asyncio
async def test_provider_only_projection_matches_full_resolution_precedence() -> None:
    resolver = TTSEffectiveSettingsResolver()
    runtime = _ResolutionRuntime()
    global_preferences = _global_preferences(provider_id="openai")
    explicit = TTSSelectionOverrides(provider_id="chatterbox")

    projected = resolver.project_provider(
        global_preferences=global_preferences,
        explicit=explicit,
    )
    resolved = await resolver.resolve_non_studio(
        global_preferences=global_preferences,
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
        explicit=explicit,
    )

    assert projected == resolved.provider_id == "chatterbox"


def _global_preferences(
    *,
    provider_id: str = "openai",
    model_mode: str = "exact",
    model_id: str | None = "tts-1-hd",
    voice_mode: str = "exact",
    voice_id: str | None = "shimmer",
    response_format: str = "mp3",
    speed: float = 1.25,
) -> TTSPreferencesSnapshot:
    return TTSPreferencesSnapshot(
        provider_id=provider_id,
        model_mode=model_mode,  # type: ignore[arg-type]
        model_id=model_id,
        voice_mode=voice_mode,  # type: ignore[arg-type]
        voice_id=voice_id,
        response_format=response_format,
        speed=speed,
    )


def _audio_cpp_catalog(
    *model_ids: str,
    revision: int = 12,
    omit_voice_uses_server_default: bool = True,
) -> TTSProviderCatalog:
    return TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=revision,
        health=ProviderHealth(state="available", fresh=True),
        models=tuple(
            TTSModelInfo(
                model_id=model_id,
                display_name=model_id,
                family="audio_cpp",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                supports_options=(),
                omit_voice_uses_server_default=omit_voice_uses_server_default,
            )
            for model_id in model_ids
        ),
    )


def _audio_cpp_capability(
    *model_ids: str,
    voices: tuple[str, ...] | None = None,
    omit_voice_uses_server_default: bool = True,
    catalog_revision: int = 12,
    configuration_revision: int = 4,
) -> TTSNativeCapabilitySnapshot:
    catalog = TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=catalog_revision,
        health=ProviderHealth(state="available", fresh=True),
        models=tuple(
            TTSModelInfo(
                model_id=model_id,
                display_name=model_id,
                family="audio_cpp",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                supports_options=(),
                omit_voice_uses_server_default=omit_voice_uses_server_default,
            )
            for model_id in model_ids
        ),
    )
    voice_results = (
        {}
        if voices is None or not model_ids
        else {
            model_ids[0]: TTSVoiceDiscoveryResult(
                provider_id="audio_cpp",
                model_id=model_ids[0],
                catalog_revision=catalog_revision,
                voices=voices,
                state="complete",
            )
        }
    )
    return TTSNativeCapabilitySnapshot(
        provider_id="audio_cpp",
        configuration_revision=configuration_revision,
        state="complete",
        catalog=catalog,
        voice_results=voice_results,
    )


class _ResolutionRuntime:
    def __init__(
        self,
        catalog: TTSProviderCatalog | None = None,
        capability: TTSNativeCapabilitySnapshot | None = None,
    ) -> None:
        self.catalog = catalog
        self.capability = capability
        self.catalog_calls: list[str] = []
        self.capability_calls: list[tuple[str, str, str | None]] = []
        self.revision_calls: list[str] = []

    def provider_revision(self, provider_id: str) -> int:
        self.revision_calls.append(provider_id)
        return 4

    async def read_catalog(self, provider_id: str) -> TTSProviderCatalog:
        self.catalog_calls.append(provider_id)
        if self.catalog is None:
            raise AssertionError("an exact selection must not read a catalog")
        return self.catalog

    async def read_native_capability(
        self,
        provider_id: str,
        model_id: str,
        voice_id: str | None,
    ) -> TTSNativeCapabilitySnapshot:
        self.capability_calls.append((provider_id, model_id, voice_id))
        if self.capability is None:
            raise AssertionError("native capability evidence was not configured")
        return self.capability


def test_provider_revision_accessors_keep_publication_and_runtime_ids_distinct() -> (
    None
):
    revisions = TTSEffectiveSelectionRevisions(
        global_preferences=8,
        studio_preferences=None,
        character_repository=None,
        character_profile=None,
        default_profile_repository=None,
        default_profile_revision=None,
        provider_configuration=41,
        provider_catalog=None,
        provider_saved=7,
        provider_applied=7,
    )

    assert revisions.provider_saved == 7
    assert revisions.provider_applied == 7
    assert revisions.provider_active == 41
    assert revisions.provider_configuration == 41


class _ActiveConfigurationService:
    def __init__(self, *, saved: int, applied: int, active: int) -> None:
        self.saved = saved
        self.applied = applied
        self.active = active

    def saved_configuration_revision(self, provider_id: str) -> int:
        assert provider_id == "openai"
        return self.saved

    def applied_configuration_revision(self, provider_id: str) -> int:
        assert provider_id == "openai"
        return self.applied

    def configuration_revision(self, provider_id: str) -> int:
        assert provider_id == "openai"
        return self.active


def test_active_configuration_compares_publication_generations_not_runtime_revision() -> (
    None
):
    service = _ActiveConfigurationService(saved=7, applied=7, active=41)

    assert tts_configuration_is_active(service, "openai", 7) is True

    service.applied = 6
    assert tts_configuration_is_active(service, "openai", 7) is False


def test_active_configuration_rejects_missing_active_runtime_identity() -> None:
    service = _ActiveConfigurationService(saved=7, applied=7, active=-1)

    assert tts_configuration_is_active(service, "openai", 7) is False


def test_active_configuration_rejects_unsaved_bootstrap_generation() -> None:
    service = _ActiveConfigurationService(saved=0, applied=0, active=1)

    assert tts_configuration_is_active(service, "openai", 0) is False


@pytest.mark.asyncio
async def test_global_only_resolution_preserves_legacy_selection_without_catalog() -> (
    None
):
    runtime = _ResolutionRuntime()

    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert resolved.provider_id == "openai"
    assert resolved.model_mode == "exact"
    assert resolved.model_id == "tts-1-hd"
    assert resolved.voice_mode == "exact"
    assert resolved.voice_id == "shimmer"
    assert resolved.response_format == "mp3"
    assert resolved.speed == 1.25
    assert dict(resolved.provider_options) == {}
    for axis in (
        "provider_id",
        "model_mode",
        "model_id",
        "voice_mode",
        "voice_id",
        "response_format",
        "speed",
    ):
        assert resolved.sources[axis] is TTSSelectionSource.GLOBAL
    assert resolved.sources["provider_options"] is TTSSelectionSource.PROVIDER_FALLBACK
    assert resolved.revisions == TTSEffectiveSelectionRevisions(
        global_preferences=7,
        studio_preferences=None,
        character_repository=None,
        character_profile=None,
        default_profile_repository=None,
        default_profile_revision=None,
        provider_configuration=4,
        provider_catalog=None,
    )
    assert runtime.catalog_calls == []
    assert runtime.revision_calls == ["openai"]


@pytest.mark.asyncio
async def test_normal_resolution_applies_explicit_then_character_then_global() -> None:
    runtime = _ResolutionRuntime(
        capability=_audio_cpp_capability(
            "character-model",
            voices=("one-message-voice",),
        )
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="character-model",
            voice_mode="exact",
            voice_id="character-voice",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=9,
        profile_revision=6,
        profile_id=_CHARACTER_PROFILE_ID,
    )

    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        explicit=TTSSelectionOverrides(
            voice_mode="exact",
            voice_id="one-message-voice",
        ),
        character_profile=character,
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
        native_capability_reader=runtime.read_native_capability,
    )

    assert resolved.provider_id == "audio_cpp"
    assert resolved.model_id == "character-model"
    assert resolved.voice_id == "one-message-voice"
    assert resolved.sources["provider_id"] is TTSSelectionSource.CHARACTER_PROFILE
    assert resolved.sources["model_id"] is TTSSelectionSource.CHARACTER_PROFILE
    assert resolved.sources["voice_mode"] is TTSSelectionSource.EXPLICIT
    assert resolved.sources["voice_id"] is TTSSelectionSource.EXPLICIT
    assert resolved.revisions.character_repository == 9
    assert resolved.revisions.character_profile == 6


@pytest.mark.asyncio
async def test_missing_character_selection_uses_global_without_character_revision() -> (
    None
):
    runtime = _ResolutionRuntime()

    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        character_profile=None,
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert resolved.provider_id == "openai"
    assert resolved.sources["provider_id"] is TTSSelectionSource.GLOBAL
    assert resolved.revisions.character_repository is None
    assert resolved.revisions.character_profile is None


@pytest.mark.asyncio
async def test_explicit_provider_does_not_inherit_other_provider_values() -> None:
    runtime = _ResolutionRuntime(_audio_cpp_catalog("first-model", "second-model"))

    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        explicit=TTSSelectionOverrides(provider_id="audio_cpp"),
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert resolved.provider_id == "audio_cpp"
    assert resolved.model_mode == "first_available"
    assert resolved.model_id == "first-model"
    assert resolved.voice_mode == "server_default"
    assert resolved.voice_id is None
    assert resolved.response_format == "wav"
    assert resolved.speed == 1.0
    assert resolved.sources["provider_id"] is TTSSelectionSource.EXPLICIT
    for axis in (
        "model_mode",
        "model_id",
        "voice_mode",
        "voice_id",
        "response_format",
        "speed",
        "provider_options",
    ):
        assert resolved.sources[axis] is TTSSelectionSource.PROVIDER_FALLBACK
    assert runtime.catalog_calls == ["audio_cpp"]


@pytest.mark.asyncio
async def test_first_available_reads_one_catalog_and_freezes_its_revision() -> None:
    runtime = _ResolutionRuntime(_audio_cpp_catalog("model-a", "model-b", revision=31))

    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        global_preferences=_global_preferences(
            provider_id="audio_cpp",
            model_mode="first_available",
            model_id=None,
            voice_mode="server_default",
            voice_id=None,
            response_format="wav",
            speed=1.0,
        ),
        global_preferences_revision=8,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert resolved.model_mode == "first_available"
    assert resolved.model_id == "model-a"
    assert resolved.voice_mode == "server_default"
    assert resolved.voice_id is None
    assert resolved.revisions.provider_catalog == 31
    assert runtime.catalog_calls == ["audio_cpp"]


@pytest.mark.asyncio
async def test_studio_resolution_uses_draft_saved_global_then_fallback() -> None:
    runtime = _ResolutionRuntime()
    saved = StudioTTSPreferencesSnapshot(
        revision=5,
        selection=StudioTTSSelectionOverrides(
            provider_id="chatterbox",
            model_mode="exact",
            model_id="chatterbox",
            voice_mode="exact",
            voice_id="saved-voice",
            response_format="wav",
            speed=1.0,
        ),
        provider_options={"chatterbox": {"exaggeration": 0.7, "cfg_weight": 0.4}},
    )
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            voice_mode="exact",
            voice_id="preview-voice",
        ),
        base_revision=5,
        preview=True,
    )

    resolved = await TTSEffectiveSettingsResolver().resolve_studio(
        studio_draft=draft,
        studio_preferences=saved,
        global_preferences=_global_preferences(),
        global_preferences_revision=11,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert resolved.provider_id == "chatterbox"
    assert resolved.model_id == "chatterbox"
    assert resolved.voice_id == "preview-voice"
    assert resolved.response_format == "wav"
    assert dict(resolved.provider_options) == {
        "exaggeration": 0.7,
        "cfg_weight": 0.4,
    }
    assert resolved.sources["provider_id"] is TTSSelectionSource.STUDIO_SAVED
    assert resolved.sources["voice_id"] is TTSSelectionSource.STUDIO_DRAFT
    assert resolved.sources["provider_options"] is TTSSelectionSource.STUDIO_SAVED
    assert dict(resolved.provider_option_sources) == {
        "exaggeration": TTSSelectionSource.STUDIO_SAVED,
        "cfg_weight": TTSSelectionSource.STUDIO_SAVED,
    }
    assert resolved.studio_preview is True
    assert resolved.revisions.studio_preferences == 5


@pytest.mark.asyncio
async def test_studio_never_receives_an_implicit_character_layer() -> None:
    runtime = _ResolutionRuntime()
    resolved = await TTSEffectiveSettingsResolver().resolve_studio(
        studio_preferences=StudioTTSPreferencesSnapshot(),
        global_preferences=_global_preferences(),
        global_preferences_revision=3,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert TTSSelectionSource.CHARACTER_PROFILE not in resolved.sources.values()
    assert resolved.revisions.character_repository is None
    assert resolved.revisions.character_profile is None


@pytest.mark.asyncio
async def test_sparse_saved_studio_values_inherit_same_provider_global_axes() -> None:
    runtime = _ResolutionRuntime()
    saved = StudioTTSPreferencesSnapshot(
        revision=2,
        selection=StudioTTSSelectionOverrides(
            voice_mode="exact",
            voice_id="studio-voice",
        ),
    )

    resolved = await TTSEffectiveSettingsResolver().resolve_studio(
        studio_preferences=saved,
        global_preferences=_global_preferences(),
        global_preferences_revision=4,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert resolved.provider_id == "openai"
    assert resolved.model_id == "tts-1-hd"
    assert resolved.voice_id == "studio-voice"
    assert resolved.sources["provider_id"] is TTSSelectionSource.GLOBAL
    assert resolved.sources["model_id"] is TTSSelectionSource.GLOBAL
    assert resolved.sources["voice_id"] is TTSSelectionSource.STUDIO_SAVED
    assert resolved.studio_preview is False
    assert resolved.revisions.studio_preferences == 2


@pytest.mark.asyncio
async def test_saved_studio_selection_is_distinct_from_unsaved_preview() -> None:
    runtime = _ResolutionRuntime()
    saved = StudioTTSPreferencesSnapshot(
        revision=6,
        selection=StudioTTSSelectionOverrides(
            provider_id="alltalk",
            model_mode="exact",
            model_id="alltalk",
            voice_mode="exact",
            voice_id="saved.wav",
            response_format="wav",
            speed=1.0,
        ),
    )

    resolved = await TTSEffectiveSettingsResolver().resolve_studio(
        studio_preferences=saved,
        global_preferences=_global_preferences(),
        global_preferences_revision=4,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert resolved.studio_preview is False
    assert resolved.sources["provider_id"] is TTSSelectionSource.STUDIO_SAVED
    assert resolved.sources["voice_id"] is TTSSelectionSource.STUDIO_SAVED


@pytest.mark.asyncio
async def test_explicit_empty_options_clear_lower_studio_options() -> None:
    runtime = _ResolutionRuntime()
    saved = StudioTTSPreferencesSnapshot(
        revision=3,
        selection=StudioTTSSelectionOverrides(
            provider_id="chatterbox",
            model_mode="exact",
            model_id="chatterbox",
            voice_mode="exact",
            voice_id="default",
            response_format="wav",
            speed=1.0,
        ),
        provider_options={"chatterbox": {"exaggeration": 0.8}},
    )
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(provider_options={}),
        base_revision=3,
    )

    resolved = await TTSEffectiveSettingsResolver().resolve_studio(
        studio_draft=draft,
        studio_preferences=saved,
        global_preferences=_global_preferences(),
        global_preferences_revision=4,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert dict(resolved.provider_options) == {}
    assert dict(resolved.provider_option_sources) == {}
    assert resolved.sources["provider_options"] is TTSSelectionSource.STUDIO_DRAFT


@pytest.mark.asyncio
async def test_sparse_draft_options_inherit_independent_saved_options() -> None:
    runtime = _ResolutionRuntime()
    saved = StudioTTSPreferencesSnapshot(
        revision=3,
        selection=StudioTTSSelectionOverrides(
            provider_id="chatterbox",
            model_mode="exact",
            model_id="chatterbox",
            voice_mode="exact",
            voice_id="default",
            response_format="wav",
            speed=1.0,
        ),
        provider_options={"chatterbox": {"exaggeration": 0.5, "cfg_weight": 0.4}},
    )
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(provider_options={"exaggeration": 0.9}),
        base_revision=3,
    )

    resolved = await TTSEffectiveSettingsResolver().resolve_studio(
        studio_draft=draft,
        studio_preferences=saved,
        global_preferences=_global_preferences(),
        global_preferences_revision=4,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert dict(resolved.provider_options) == {
        "exaggeration": 0.9,
        "cfg_weight": 0.4,
    }
    assert dict(resolved.provider_option_sources) == {
        "exaggeration": TTSSelectionSource.STUDIO_DRAFT,
        "cfg_weight": TTSSelectionSource.STUDIO_SAVED,
    }
    assert resolved.sources["provider_options"] is TTSSelectionSource.STUDIO_DRAFT


@pytest.mark.asyncio
async def test_saved_options_follow_their_provider_when_draft_switches_provider() -> (
    None
):
    runtime = _ResolutionRuntime()
    saved = StudioTTSPreferencesSnapshot(
        revision=3,
        selection=StudioTTSSelectionOverrides(provider_id="openai"),
        provider_options={"chatterbox": {"exaggeration": 0.7}},
    )
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(provider_id="chatterbox"),
        base_revision=3,
    )

    resolved = await TTSEffectiveSettingsResolver().resolve_studio(
        studio_draft=draft,
        studio_preferences=saved,
        global_preferences=_global_preferences(),
        global_preferences_revision=4,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert resolved.provider_id == "chatterbox"
    assert dict(resolved.provider_options) == {"exaggeration": 0.7}
    assert resolved.sources["provider_options"] is TTSSelectionSource.STUDIO_SAVED
    assert dict(resolved.provider_option_sources) == {
        "exaggeration": TTSSelectionSource.STUDIO_SAVED
    }


@pytest.mark.asyncio
async def test_stale_studio_draft_blocks_without_falling_through() -> None:
    runtime = _ResolutionRuntime()
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(voice_id="do-not-fall-through"),
        base_revision=4,
    )

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_studio(
            studio_draft=draft,
            studio_preferences=StudioTTSPreferencesSnapshot(revision=5),
            global_preferences=_global_preferences(),
            global_preferences_revision=3,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "revision_incoherent"
    assert caught.value.axis == "studio_preferences"
    assert caught.value.source is TTSSelectionSource.STUDIO_DRAFT
    assert runtime.revision_calls == []
    assert runtime.catalog_calls == []


@pytest.mark.asyncio
async def test_missing_exact_model_blocks_instead_of_using_dynamic_fallback() -> None:
    runtime = _ResolutionRuntime(_audio_cpp_catalog("must-not-be-used"))

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            explicit=TTSSelectionOverrides(
                provider_id="audio_cpp",
                model_mode="exact",
            ),
            global_preferences=_global_preferences(),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "missing_exact"
    assert caught.value.axis == "model_id"
    assert caught.value.source is TTSSelectionSource.EXPLICIT
    assert runtime.catalog_calls == []


@pytest.mark.asyncio
async def test_authoritative_catalog_blocks_removed_exact_audio_cpp_model() -> None:
    runtime = _ResolutionRuntime(capability=_audio_cpp_capability("available-model"))

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            global_preferences=_global_preferences(
                provider_id="audio_cpp",
                model_mode="exact",
                model_id="removed-model",
                voice_mode="server_default",
                voice_id=None,
                response_format="wav",
                speed=1.0,
            ),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
            native_capability_reader=runtime.read_native_capability,
        )

    assert caught.value.code == "missing_exact"
    assert caught.value.axis == "model_id"
    assert caught.value.source is TTSSelectionSource.GLOBAL
    assert runtime.capability_calls == [("audio_cpp", "removed-model", None)]


@pytest.mark.asyncio
async def test_authoritative_voices_block_removed_exact_audio_cpp_voice() -> None:
    runtime = _ResolutionRuntime(
        capability=_audio_cpp_capability(
            "model",
            voices=("available-voice",),
        )
    )

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            global_preferences=_global_preferences(
                provider_id="audio_cpp",
                model_mode="exact",
                model_id="model",
                voice_mode="exact",
                voice_id="removed-voice",
                response_format="wav",
                speed=1.0,
            ),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
            native_capability_reader=runtime.read_native_capability,
        )

    assert caught.value.code == "missing_exact"
    assert caught.value.axis == "voice_id"
    assert caught.value.source is TTSSelectionSource.GLOBAL
    assert runtime.capability_calls == [("audio_cpp", "model", "removed-voice")]


@pytest.mark.asyncio
async def test_server_default_blocks_when_selected_model_requires_a_voice() -> None:
    runtime = _ResolutionRuntime(
        _audio_cpp_catalog(
            "model",
            omit_voice_uses_server_default=False,
        )
    )

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            global_preferences=_global_preferences(
                provider_id="audio_cpp",
                model_mode="first_available",
                model_id=None,
                voice_mode="server_default",
                voice_id=None,
                response_format="wav",
                speed=1.0,
            ),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "unsupported_selection"
    assert caught.value.axis == "voice_mode"
    assert caught.value.source is TTSSelectionSource.GLOBAL


@pytest.mark.asyncio
async def test_incomplete_character_profile_blocks_instead_of_using_global_values() -> (
    None
):
    runtime = _ResolutionRuntime()
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id=None,
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=2,
        profile_revision=3,
        profile_id=_CHARACTER_PROFILE_ID,
    )

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            character_profile=character,
            global_preferences=_global_preferences(
                provider_id="audio_cpp",
                model_mode="exact",
                model_id="global-model",
                voice_mode="server_default",
                voice_id=None,
                response_format="wav",
                speed=1.0,
            ),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "missing_exact"
    assert caught.value.axis == "model_id"
    assert caught.value.source is TTSSelectionSource.CHARACTER_PROFILE


@pytest.mark.asyncio
async def test_unknown_explicit_provider_blocks_without_reading_runtime() -> None:
    runtime = _ResolutionRuntime()

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            explicit=TTSSelectionOverrides(provider_id="future_provider"),
            global_preferences=_global_preferences(),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "provider_unknown"
    assert caught.value.axis == "provider_id"
    assert caught.value.source is TTSSelectionSource.EXPLICIT
    assert runtime.revision_calls == []
    assert runtime.catalog_calls == []


@pytest.mark.asyncio
async def test_dynamic_mode_ignores_only_lower_exact_identifier() -> None:
    runtime = _ResolutionRuntime(_audio_cpp_catalog("catalog-model"))

    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        explicit=TTSSelectionOverrides(model_mode="first_available"),
        global_preferences=_global_preferences(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="lower-exact-model",
            voice_mode="server_default",
            voice_id=None,
            response_format="wav",
            speed=1.0,
        ),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert resolved.model_id == "catalog-model"
    assert resolved.sources["model_mode"] is TTSSelectionSource.EXPLICIT
    assert resolved.sources["model_id"] is TTSSelectionSource.EXPLICIT
    assert runtime.catalog_calls == ["audio_cpp"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "catalog",
    (
        TTSProviderCatalog(
            provider_id="audio_cpp",
            revision=2,
            health=ProviderHealth(state="unavailable", fresh=False),
            models=(_audio_cpp_catalog("stale").models[0],),
        ),
        TTSProviderCatalog(
            provider_id="openai",
            revision=2,
            health=ProviderHealth(state="available", fresh=True),
            models=(_audio_cpp_catalog("wrong-provider").models[0],),
        ),
        TTSProviderCatalog(
            provider_id="audio_cpp",
            revision=2,
            health=ProviderHealth(state="unavailable", fresh=True),
            models=(_audio_cpp_catalog("retained-but-unavailable").models[0],),
        ),
    ),
)
async def test_dynamic_selection_rejects_unaccepted_catalog(
    catalog: TTSProviderCatalog,
) -> None:
    runtime = _ResolutionRuntime(catalog)

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            global_preferences=_global_preferences(
                provider_id="audio_cpp",
                model_mode="first_available",
                model_id=None,
                voice_mode="server_default",
                voice_id=None,
                response_format="wav",
                speed=1.0,
            ),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code in {"catalog_unavailable", "revision_incoherent"}
    assert caught.value.axis == "provider_catalog"
    assert runtime.catalog_calls == ["audio_cpp"]


@pytest.mark.asyncio
async def test_invalid_provider_revision_blocks_after_selection() -> None:
    runtime = _ResolutionRuntime()

    def invalid_revision(provider_id: str) -> int:
        runtime.revision_calls.append(provider_id)
        return -1

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            global_preferences=_global_preferences(),
            global_preferences_revision=7,
            provider_revision_reader=invalid_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "revision_incoherent"
    assert caught.value.axis == "provider_configuration"
    assert runtime.revision_calls == ["openai"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection",
    (
        TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="model",
            voice_mode="server_default",
            response_format="mp3",
            speed=1.0,
            provider_options={},
        ),
        TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.25,
            provider_options={},
        ),
        TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={"arbitrary": True},
        ),
    ),
)
async def test_audio_cpp_constraints_fail_closed_at_every_layer(
    selection: TTSSelectionOverrides,
) -> None:
    runtime = _ResolutionRuntime()

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            explicit=selection,
            global_preferences=_global_preferences(),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "unsupported_selection"
    assert caught.value.source is TTSSelectionSource.EXPLICIT


@pytest.mark.asyncio
async def test_unknown_provider_option_blocks_without_using_saved_or_global() -> None:
    runtime = _ResolutionRuntime()

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            explicit=TTSSelectionOverrides(
                provider_id="chatterbox",
                model_mode="exact",
                model_id="chatterbox",
                voice_mode="exact",
                voice_id="default",
                response_format="wav",
                speed=1.0,
                provider_options={"credential": "do-not-use"},
            ),
            global_preferences=_global_preferences(),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "unsupported_selection"
    assert caught.value.axis == "provider_options"
    assert caught.value.source is TTSSelectionSource.EXPLICIT
    assert "do-not-use" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_options",
    (
        {"temperature": 2.1},
        {"num_candidates": True},
        {"num_candidates": 0},
        {"validate_with_whisper": "yes"},
    ),
)
async def test_invalid_request_option_value_blocks_before_admission(
    provider_options: dict[str, object],
) -> None:
    runtime = _ResolutionRuntime()

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            explicit=TTSSelectionOverrides(
                provider_id="chatterbox",
                model_mode="exact",
                model_id="chatterbox",
                voice_mode="exact",
                voice_id="default",
                response_format="wav",
                speed=1.0,
                provider_options=provider_options,
            ),
            global_preferences=_global_preferences(),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "invalid_selection"
    assert caught.value.axis == "provider_options"
    assert caught.value.source is TTSSelectionSource.EXPLICIT
    assert runtime.revision_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "private_value",
    (
        "https://user:credential@example.test/private?token=secret",
        "//user:credential@example.test/private?token=secret",
    ),
)
async def test_endpoint_with_embedded_credentials_cannot_become_an_exact_id(
    private_value: str,
) -> None:
    runtime = _ResolutionRuntime()

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            explicit=TTSSelectionOverrides(
                model_mode="exact",
                model_id=private_value,
            ),
            global_preferences=_global_preferences(),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "invalid_selection"
    assert caught.value.axis == "model_id"
    assert private_value not in str(caught.value)
    assert private_value not in repr(caught.value)
    assert runtime.revision_calls == []


@pytest.mark.asyncio
async def test_effective_snapshot_is_immutable_and_contains_no_sensitive_payload() -> (
    None
):
    runtime = _ResolutionRuntime()
    original_options = {"exaggeration": 0.6, "cfg_weight": 0.3}
    explicit = TTSSelectionOverrides(
        provider_id="chatterbox",
        model_mode="exact",
        model_id="chatterbox",
        voice_mode="exact",
        voice_id="default",
        response_format="wav",
        speed=1.0,
        provider_options=original_options,
    )
    original_options["exaggeration"] = 0.1

    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        explicit=explicit,
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
    )

    assert dict(resolved.provider_options) == {
        "exaggeration": 0.6,
        "cfg_weight": 0.3,
    }
    with pytest.raises(TypeError):
        resolved.provider_options["exaggeration"] = 0.2  # type: ignore[index]
    with pytest.raises(TypeError):
        resolved.sources["provider_id"] = TTSSelectionSource.GLOBAL  # type: ignore[index]

    field_names = {item.name for item in fields(TTSEffectiveSelectionSnapshot)}
    forbidden_fragments = {
        "text",
        "credential",
        "endpoint",
        "url",
        "widget",
        "character",
        "adapter",
    }
    assert not any(
        fragment in field_name
        for field_name in field_names
        for fragment in forbidden_fragments
    )


def test_resolution_error_copy_is_bounded_and_value_free() -> None:
    error = TTSEffectiveResolutionError(
        code="unsupported_selection",
        axis="provider_options",
        source=TTSSelectionSource.EXPLICIT,
    )

    assert error.code == "unsupported_selection"
    assert error.axis == "provider_options"
    assert error.source is TTSSelectionSource.EXPLICIT
    assert "provider_options" in str(error)
    assert "credential" not in str(error)


@pytest.mark.asyncio
async def test_default_profile_wins_over_global_and_loses_to_character() -> None:
    runtime = _ResolutionRuntime(
        capability=_audio_cpp_capability("default-model", voices=("default-voice",))
    )
    default_profile = TTSDefaultProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="default-model",
            voice_mode="exact",
            voice_id="default-voice",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=4,
        profile_revision=2,
        profile_id=_DEFAULT_PROFILE_ID,
    )
    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        default_profile=default_profile,
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
        native_capability_reader=runtime.read_native_capability,
    )

    assert resolved.voice_id == "default-voice"
    assert resolved.sources["voice_id"] is TTSSelectionSource.DEFAULT_PROFILE
    assert resolved.sources["model_id"] is TTSSelectionSource.DEFAULT_PROFILE
    assert resolved.revisions.default_profile_repository == 4
    assert resolved.revisions.default_profile_revision == 2


@pytest.mark.asyncio
async def test_character_profile_outranks_default_profile() -> None:
    runtime = _ResolutionRuntime(
        capability=_audio_cpp_capability("character-model", voices=("character-voice",))
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="character-model",
            voice_mode="exact",
            voice_id="character-voice",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=9,
        profile_revision=6,
        profile_id=_CHARACTER_PROFILE_ID,
    )
    default_profile = TTSDefaultProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="default-model",
            voice_mode="exact",
            voice_id="default-voice",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=4,
        profile_revision=2,
        profile_id=_DEFAULT_PROFILE_ID,
    )
    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        character_profile=character,
        default_profile=default_profile,
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
        native_capability_reader=runtime.read_native_capability,
    )

    assert resolved.voice_id == "character-voice"
    assert resolved.sources["voice_id"] is TTSSelectionSource.CHARACTER_PROFILE


@pytest.mark.asyncio
async def test_no_default_profile_still_falls_through_to_global() -> None:
    runtime = _ResolutionRuntime(
        capability=_audio_cpp_capability("character-model", voices=("character-voice",))
    )
    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
        native_capability_reader=runtime.read_native_capability,
    )

    assert resolved.sources["voice_id"] is TTSSelectionSource.GLOBAL
    assert resolved.revisions.default_profile_repository is None
    assert resolved.revisions.default_profile_revision is None


@pytest.mark.asyncio
async def test_incomplete_default_profile_blocks_instead_of_using_global_values() -> (
    None
):
    runtime = _ResolutionRuntime()
    default_profile = TTSDefaultProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id=None,
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=2,
        profile_revision=3,
        profile_id=_DEFAULT_PROFILE_ID,
    )

    with pytest.raises(TTSEffectiveResolutionError) as caught:
        await TTSEffectiveSettingsResolver().resolve_non_studio(
            default_profile=default_profile,
            global_preferences=_global_preferences(
                provider_id="audio_cpp",
                model_mode="exact",
                model_id="global-model",
                voice_mode="server_default",
                voice_id=None,
                response_format="wav",
                speed=1.0,
            ),
            global_preferences_revision=7,
            provider_revision_reader=runtime.provider_revision,
            catalog_reader=runtime.read_catalog,
        )

    assert caught.value.code == "missing_exact"
    assert caught.value.axis == "model_id"
    assert caught.value.source is TTSSelectionSource.DEFAULT_PROFILE


def test_default_profile_revisions_must_travel_together() -> None:
    with pytest.raises(ValueError):
        TTSEffectiveSelectionRevisions(
            global_preferences=1,
            studio_preferences=None,
            character_repository=None,
            character_profile=None,
            default_profile_repository=4,
            default_profile_revision=None,
            provider_configuration=1,
            provider_catalog=1,
        )
