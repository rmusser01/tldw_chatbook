"""Settings contract for automatic model catalog refresh (ADR-020)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, field_validator

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


class _ModelCatalogSection(BaseModel):
    """Raw [model_catalog] TOML section; garbage values fall back per field."""

    auto_refresh_enabled: bool = True
    stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS
    auto_refresh_disabled: frozenset[str] = frozenset()
    write_to_config: frozenset[str] = frozenset()

    @field_validator("auto_refresh_enabled", mode="before")
    @classmethod
    def _fallback_enabled(cls, value: object) -> bool:
        # Non-bool garbage keeps the safe default (refresh enabled).
        return value if isinstance(value, bool) else True

    @field_validator("stale_after_hours", mode="before")
    @classmethod
    def _fallback_stale_after_hours(cls, value: object) -> float:
        # Unparseable keeps the default; negatives clamp to 0 (always-stale).
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return DEFAULT_STALE_AFTER_HOURS

    @field_validator("auto_refresh_disabled", "write_to_config", mode="before")
    @classmethod
    def _fallback_key_set(cls, value: object) -> frozenset[str]:
        # Non-list garbage (and non-string entries) normalize away.
        return _normalized_key_set(value)


def load_model_catalog_settings(settings: Mapping[str, Any] | None) -> ModelCatalogSettings:
    """Parse the ``[model_catalog]`` section of the loaded settings.

    Args:
        settings: Loaded application settings mapping, or None.

    Returns:
        ModelCatalogSettings: Validated settings. Missing section or garbage
        values fall back to safe defaults field by field (refresh enabled,
        24h staleness, empty provider sets); provider sets hold
        provider_config_key-normalized names.
    """
    section = (settings or {}).get("model_catalog", {})
    if not isinstance(section, Mapping):
        return ModelCatalogSettings()
    parsed = _ModelCatalogSection.model_validate(dict(section))
    return ModelCatalogSettings(
        auto_refresh_enabled=parsed.auto_refresh_enabled,
        stale_after_hours=parsed.stale_after_hours,
        auto_refresh_disabled=parsed.auto_refresh_disabled,
        write_to_config=parsed.write_to_config,
    )
