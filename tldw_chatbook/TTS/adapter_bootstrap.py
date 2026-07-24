from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.legacy_bridge import (
    _legacy_config_snapshot as _legacy_config_snapshot,
    legacy_provider_specs,
)
from tldw_chatbook.TTS.TTS_Generation import TTSService


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
        specs=legacy_provider_specs(app_config),
        aliases={},
    )
    return TTSService(registry, max_concurrent_operations=4)
