"""Shared provider/model resolution for Console and Settings surfaces."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...Chat.provider_readiness import provider_config_key
from ...LLM_Provider_Catalog.model_discovery_contracts import MergedModelEntry


MODEL_CAPABILITY_UNKNOWN_WARNING = (
    "Capabilities unknown until saved or verified; text chat is assumed."
)


@dataclass(frozen=True)
class EffectiveProviderModel:
    """Resolved provider/model values and the source each value came from."""

    provider: Any
    model: Any
    provider_source: str
    model_source: str


@dataclass(frozen=True)
class ResolvedProviderModelOption:
    """Console model selector option with runtime discovery metadata."""

    label: str
    model_id: str
    source: str
    capability_status: str
    persisted: bool
    warning: str = ""


def _selected_text(value: Any) -> bool:
    """Return whether a provider/model-like value is meaningfully selected."""
    if value is None or value is False:
        return False
    text = str(value).strip()
    return bool(text) and text != "None" and not text.startswith("Select.")


def _chat_default(app_instance: Any, key: str) -> Any:
    config = getattr(app_instance, "app_config", {}) or {}
    defaults = config.get("chat_defaults", {}) if isinstance(config, dict) else {}
    return defaults.get(key) if isinstance(defaults, dict) else None


def _providers_models(app_instance: Any) -> Mapping[str, Sequence[str]]:
    providers_models = getattr(app_instance, "providers_models", None)
    return providers_models if isinstance(providers_models, Mapping) else {}


def _saved_models_for_provider(
    providers_models: Mapping[str, Sequence[str]],
    provider: str,
) -> list[str]:
    provider_key = provider_config_key(provider)
    model_ids: list[str] = []
    for configured_provider, configured_models in providers_models.items():
        if provider_config_key(str(configured_provider)) != provider_key:
            continue
        if not isinstance(configured_models, Sequence) or isinstance(configured_models, (str, bytes)):
            continue
        for configured_model in configured_models:
            model_id = str(configured_model or "").strip()
            if model_id and model_id not in model_ids:
                model_ids.append(model_id)
    return model_ids


def _warning_for_model(source: str, capability_status: str) -> str:
    if source in {"runtime_discovered", "persisted_discovered"} and capability_status == "unknown":
        return MODEL_CAPABILITY_UNKNOWN_WARNING
    return ""


def _option_from_entry(entry: MergedModelEntry) -> ResolvedProviderModelOption:
    model_id = str(entry.model_id).strip()
    source = str(entry.source)
    capability_status = str(entry.capability_status)
    label = model_id
    if source == "runtime_discovered":
        label = f"{model_id} | runtime discovered"
    if capability_status == "unknown" and source != "saved":
        label = f"{label} | capability unknown"
    return ResolvedProviderModelOption(
        label=label,
        model_id=model_id,
        source=source,
        capability_status=capability_status,
        persisted=bool(entry.persisted),
        warning=_warning_for_model(source, capability_status),
    )


def _option_from_saved_model(model_id: str) -> ResolvedProviderModelOption:
    return ResolvedProviderModelOption(
        label=model_id,
        model_id=model_id,
        source="saved",
        capability_status="known",
        persisted=True,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _merged_model_entries_from_scope(
    app_instance: Any,
    *,
    provider: str,
) -> tuple[MergedModelEntry, ...]:
    scope_service = getattr(app_instance, "llm_provider_catalog_scope_service", None)
    merge_models = getattr(scope_service, "merge_saved_and_discovered_models", None)
    if not callable(merge_models):
        return ()
    result = await _maybe_await(
        merge_models(
            mode="local",
            provider=provider,
        )
    )
    return tuple(entry for entry in result if isinstance(entry, MergedModelEntry))


async def resolve_provider_model_options(
    app_instance: Any,
    *,
    provider: str,
    current_model: str | None = None,
) -> list[ResolvedProviderModelOption]:
    """Return saved and runtime-discovered model selector options for a provider."""
    provider_key = provider_config_key(provider)
    saved_models = _saved_models_for_provider(_providers_models(app_instance), provider_key)
    options: list[ResolvedProviderModelOption] = []
    seen_model_ids: set[str] = set()

    for model_id in saved_models:
        options.append(_option_from_saved_model(model_id))
        seen_model_ids.add(model_id)

    merged_entries = await _merged_model_entries_from_scope(app_instance, provider=provider_key)
    for entry in merged_entries:
        option = _option_from_entry(entry)
        if option.model_id and option.model_id not in seen_model_ids:
            options.append(option)
            seen_model_ids.add(option.model_id)

    current_model_id = str(current_model or "").strip()
    if current_model_id and current_model_id not in seen_model_ids:
        options.insert(0, _option_from_saved_model(current_model_id))
    return options


def resolve_effective_provider_model(
    app_instance: Any,
    *,
    console_provider: Any = None,
    console_model: Any = None,
    settings_provider: Any = None,
    settings_model: Any = None,
) -> EffectiveProviderModel:
    """Resolve the canonical provider/model pair for Console-adjacent UI.

    Args:
        app_instance: Application object that may expose config and reactive provider/model values.
        console_provider: Provider selected by the Console control surface.
        console_model: Model selected by the Console control surface.
        settings_provider: Provider staged in Settings before save.
        settings_model: Model staged in Settings before save.

    Returns:
        Resolved provider/model values plus labels naming each selected source.

    Settings drafts win because they are what the user is evaluating before save.
    Console controls win next because they are the active run surface. The default
    OpenAI reactive value is ignored when config already names a non-OpenAI
    provider, matching the existing Console readiness behavior.
    """
    configured_provider = _chat_default(app_instance, "provider")
    reactive_provider = getattr(app_instance, "chat_api_provider_value", None)

    if _selected_text(settings_provider):
        provider = settings_provider
        provider_source = "settings_draft"
    elif _selected_text(console_provider):
        provider = console_provider
        provider_source = "console_control"
    elif (
        _selected_text(configured_provider)
        and str(reactive_provider or "").strip() == "OpenAI"
        and str(configured_provider).strip() != "OpenAI"
    ):
        provider = configured_provider
        provider_source = "chat_defaults"
    elif _selected_text(reactive_provider):
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

    if _selected_text(settings_model):
        model = settings_model
        model_source = "settings_draft"
    elif _selected_text(console_model):
        model = console_model
        model_source = "console_control"
    elif _selected_text(reactive_model):
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
