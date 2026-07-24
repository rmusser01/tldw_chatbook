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
    project_audio_cpp_config,
)
from tldw_chatbook.TTS.legacy_bridge import (
    _legacy_config_snapshot as _legacy_config_snapshot,
    legacy_provider_specs,
)
from tldw_chatbook.TTS.TTS_Generation import TTSService


def _create_audio_cpp_adapter(config: Mapping[str, Any]) -> TTSAdapter:
    validated_config = AudioCppConfig.from_mapping(config)
    adapter_module = import_module("tldw_chatbook.TTS.adapters.audio_cpp")
    adapter_factory = cast(
        Callable[[AudioCppConfig], TTSAdapter],
        adapter_module.AudioCppAdapter,
    )
    return adapter_factory(validated_config)


def audio_cpp_provider_spec(
    app_config: Mapping[str, Any],
) -> TTSProviderSpec:
    """Build the lazy native audio.cpp provider specification.

    Args:
        app_config: Normalized application settings with optional raw config.

    Returns:
        A native exclusive provider spec with an independent config snapshot.
    """
    config = project_audio_cpp_config(app_config)
    return TTSProviderSpec(
        descriptor=TTSProviderDescriptor(
            provider_id="audio_cpp",
            display_name="audio.cpp",
            native=True,
        ),
        factory=_create_audio_cpp_adapter,
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
    registry = TTSAdapterRegistry(
        specs=(
            audio_cpp_provider_spec(app_config),
            *legacy_provider_specs(app_config),
        ),
        aliases={},
    )
    return TTSService(registry, max_concurrent_operations=4)
