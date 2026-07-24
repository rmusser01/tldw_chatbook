from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.legacy_bridge import legacy_provider_specs
from tldw_chatbook.TTS.TTS_Generation import TTSService


def _legacy_config_snapshot(
    app_config: Mapping[str, Any],
) -> dict[str, Any]:
    nested_raw = app_config.get("COMPREHENSIVE_CONFIG_RAW")
    source = nested_raw if isinstance(nested_raw, Mapping) else app_config
    snapshot = deepcopy(dict(source))
    if "app_tts" not in snapshot:
        normalized_tts = app_config.get("APP_TTS_CONFIG", {})
        snapshot["app_tts"] = (
            deepcopy(dict(normalized_tts))
            if isinstance(normalized_tts, Mapping)
            else {}
        )
    return snapshot


def build_default_tts_service(
    app_config: Mapping[str, Any],
) -> TTSService:
    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(_legacy_config_snapshot(app_config)),
        aliases={},
    )
    return TTSService(registry, max_concurrent_operations=4)
