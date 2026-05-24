"""Shared provider/model resolution for Console and Settings surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EffectiveProviderModel:
    """Resolved provider/model values and the source each value came from."""

    provider: Any
    model: Any
    provider_source: str
    model_source: str


def _selected_text(value: Any) -> bool:
    """Return whether a provider/model-like value is meaningfully selected."""
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text != "None" and not text.startswith("Select.")


def _chat_default(app_instance: Any, key: str) -> Any:
    config = getattr(app_instance, "app_config", {}) or {}
    defaults = config.get("chat_defaults", {})
    return defaults.get(key) if isinstance(defaults, dict) else None


def resolve_effective_provider_model(
    app_instance: Any,
    *,
    console_provider: Any = None,
    console_model: Any = None,
    settings_provider: Any = None,
    settings_model: Any = None,
) -> EffectiveProviderModel:
    """Resolve the canonical provider/model pair for Console-adjacent UI.

    Settings drafts win because they are what the user is evaluating before
    save. Console controls win next because they are the active run surface.
    The default OpenAI reactive value is ignored when config already names a
    non-OpenAI provider, matching the existing Console readiness behavior.
    """
    configured_provider = _chat_default(app_instance, "provider")
    reactive_provider = getattr(app_instance, "chat_api_provider_value", None)

    if settings_provider is not None:
        provider = settings_provider
        provider_source = "settings_draft"
    elif console_provider is not None:
        provider = console_provider
        provider_source = "console_control"
    elif (
        _selected_text(configured_provider)
        and str(reactive_provider or "").strip() == "OpenAI"
        and str(configured_provider).strip() != "OpenAI"
    ):
        provider = configured_provider
        provider_source = "chat_defaults"
    elif reactive_provider is not None:
        provider = reactive_provider
        provider_source = "app_reactive"
    else:
        provider = configured_provider
        provider_source = "chat_defaults"

    reactive_model = (
        getattr(app_instance, "chat_api_model_value", None)
        or getattr(app_instance, "chat_model_value", None)
    )
    configured_model = _chat_default(app_instance, "model")

    if settings_model is not None:
        model = settings_model
        model_source = "settings_draft"
    elif console_model is not None:
        model = console_model
        model_source = "console_control"
    elif reactive_model is not None:
        model = reactive_model
        model_source = "app_reactive"
    else:
        model = configured_model
        model_source = "chat_defaults"

    return EffectiveProviderModel(
        provider=provider,
        model=model,
        provider_source=provider_source,
        model_source=model_source,
    )
