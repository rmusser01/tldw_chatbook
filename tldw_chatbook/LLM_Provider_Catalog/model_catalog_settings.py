"""Settings contract for automatic model catalog refresh (ADR-014)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat.provider_readiness import provider_config_key

AUTO_REFRESH_PROVIDER_LIST_KEYS: tuple[str, ...] = (
    "OpenAI",
    "Anthropic",
    "MistralAI",
    "Moonshot",
    "OpenRouter",
    "ZAI",
)

SELECTOR_MERGE_CAP = 50
DEFAULT_STALE_AFTER_HOURS = 24.0


@dataclass(frozen=True)
class ModelCatalogSettings:
    """Parsed [model_catalog] config; provider sets hold normalized keys."""

    auto_refresh_enabled: bool = True
    stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS
    auto_refresh_disabled: frozenset[str] = frozenset()
    write_to_config: frozenset[str] = frozenset()


def _normalized_key_set(value: object) -> frozenset[str]:
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(
        provider_config_key(str(entry))
        for entry in value
        if isinstance(entry, str) and entry.strip()
    )


def load_model_catalog_settings(settings: Mapping[str, Any] | None) -> ModelCatalogSettings:
    section = (settings or {}).get("model_catalog", {})
    if not isinstance(section, Mapping):
        return ModelCatalogSettings()
    enabled = section.get("auto_refresh_enabled", True)
    try:
        stale_after_hours = max(0.0, float(section.get("stale_after_hours", DEFAULT_STALE_AFTER_HOURS)))
    except (TypeError, ValueError):
        stale_after_hours = DEFAULT_STALE_AFTER_HOURS
    return ModelCatalogSettings(
        auto_refresh_enabled=enabled if isinstance(enabled, bool) else True,
        stale_after_hours=stale_after_hours,
        auto_refresh_disabled=_normalized_key_set(section.get("auto_refresh_disabled")),
        write_to_config=_normalized_key_set(section.get("write_to_config")),
    )
