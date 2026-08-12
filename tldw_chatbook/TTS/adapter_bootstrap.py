from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, cast

from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import (
    TTSAdapter,
    TTSProviderDescriptor,
    TTSProviderSpec,
)
from tldw_chatbook.TTS.audio_cpp_config import (
    AudioCppConfig,
)
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppManagedSetupSource,
    AudioCppSettingsConfig,
    project_audio_cpp_settings_config,
)
from tldw_chatbook.TTS.audio_cpp_managed_config import (
    collect_provider_credential_environment_names,
)
from tldw_chatbook.TTS.audio_cpp_supervisor import AudioCppSupervisor
from tldw_chatbook.TTS.legacy_bridge import (
    _legacy_config_snapshot as _legacy_config_snapshot,
    legacy_provider_specs,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_reference_materialization import (
    TTSCloneReferenceMaterializer,
)
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferenceStore
from tldw_chatbook.TTS.TTS_Generation import TTSService
from tldw_chatbook.config import get_user_data_dir


def _create_audio_cpp_adapter(
    config: Mapping[str, Any],
    supervisor: AudioCppSupervisor | None,
) -> TTSAdapter:
    validated_config = AudioCppConfig.from_mapping(config)
    settings_config = AudioCppSettingsConfig.from_mapping(config)
    adapter_module = import_module("tldw_chatbook.TTS.adapters.audio_cpp")
    adapter_factory = cast(
        Callable[..., TTSAdapter],
        adapter_module.AudioCppAdapter,
    )
    kwargs: dict[str, object] = {}
    if (
        settings_config.mode == "managed"
        and settings_config.managed_setup_source is AudioCppManagedSetupSource.GUIDED
    ):
        kwargs["guided_settings"] = settings_config
    if supervisor is not None:
        kwargs["supervisor"] = supervisor
    return adapter_factory(validated_config, **kwargs)


def audio_cpp_provider_spec(
    app_config: Mapping[str, Any],
    *,
    supervisor: AudioCppSupervisor | None = None,
) -> TTSProviderSpec:
    """Build the lazy native audio.cpp provider specification.

    Args:
        app_config: Normalized application settings with optional raw config.

    Returns:
        A native exclusive provider spec with an independent config snapshot.
    """
    config = project_audio_cpp_settings_config(app_config)
    return TTSProviderSpec(
        descriptor=TTSProviderDescriptor(
            provider_id="audio_cpp",
            display_name="audio.cpp",
            native=True,
        ),
        factory=lambda replacement: _create_audio_cpp_adapter(
            replacement,
            supervisor,
        ),
        initial_config=config.to_mapping(),
        exclusive_reconfigure=True,
    )


def build_default_tts_service(
    app_config: Mapping[str, Any],
) -> TTSService:
    """Build the lazy application-owned service from a configuration snapshot.

    Args:
        app_config: Normalized application settings with optional raw config.

    Returns:
        A service whose adapters remain unmaterialized until first use.
    """
    preferences_snapshot = TTSPreferencesSnapshot.from_settings(app_config)
    supervisor = AudioCppSupervisor(
        provider_credential_names=(
            collect_provider_credential_environment_names(app_config)
        )
    )
    # The bridge snapshots and normalizes OpenAI-compatible endpoint/auth fields
    # before any lazy legacy adapter is materialized.
    legacy_specs = legacy_provider_specs(app_config)
    registry = TTSAdapterRegistry(
        specs=(
            audio_cpp_provider_spec(app_config, supervisor=supervisor),
            *legacy_specs,
        ),
        aliases={},
    )
    studio_preferences = StudioTTSPreferenceStore()
    clone_materializer = TTSCloneReferenceMaterializer(
        get_user_data_dir() / "tts_clone_materializations"
    )
    return TTSService(
        registry,
        max_concurrent_operations=4,
        preferences_snapshot=preferences_snapshot,
        studio_preferences_loader=lambda: (
            studio_preferences.load(migrate=False).snapshot
        ),
        audio_cpp_supervisor=supervisor,
        clone_materializer=clone_materializer,
    )
