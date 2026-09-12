"""Pure spawn-target routing for sub-agents (ADR-147, TASK-32477).

Resolution order (each level fills only blanks left above): ad-hoc spawn
args (gated) -> preset fields -> [agents] sub-agent default -> inherit
parent. Params are NEVER inherited from the parent; they resolve through
the six-layer stack in resolve_child_params.
"""
from __future__ import annotations

import fnmatch
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Agents.agent_models import AgentDefinition
# Module-level only because ADR-147 defines Chat/sampling_params.py as the
# pure, dependency-free leaf shared by presets, registry entries, and this
# resolver; every other Chat import below stays lazy (inside functions) so
# agent_service.py remains the only impure Agents module.
from tldw_chatbook.Chat.sampling_params import (
    KNOWN_SAMPLING_PARAM_KEYS, params_to_dict, params_to_tuple,
)

logger = logging.getLogger(__name__)

DEFAULT_SUBAGENT_DEFAULT_PROVIDER = ""
DEFAULT_SUBAGENT_DEFAULT_MODEL = ""
DEFAULT_SPAWN_OVERRIDE_ENABLED = False
DEFAULT_SPAWN_OVERRIDE_ALLOWLIST: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentsRoutingConfig:
    subagent_default_provider: str = DEFAULT_SUBAGENT_DEFAULT_PROVIDER
    subagent_default_model: str = DEFAULT_SUBAGENT_DEFAULT_MODEL
    spawn_override_enabled: bool = DEFAULT_SPAWN_OVERRIDE_ENABLED
    spawn_override_allowlist: tuple[str, ...] = DEFAULT_SPAWN_OVERRIDE_ALLOWLIST


def load_agents_routing_config() -> AgentsRoutingConfig:
    """Read the [agents] routing keys via the same _setting accessor the
    other agent keys use (Agents/run_log.py)."""
    from tldw_chatbook.Agents.run_log import _setting

    raw = _setting("spawn_override_allowlist", DEFAULT_SPAWN_OVERRIDE_ALLOWLIST) or ()
    if isinstance(raw, str):
        raw = (raw,)
    return AgentsRoutingConfig(
        subagent_default_provider=str(
            _setting("subagent_default_provider", "") or "").strip(),
        subagent_default_model=str(
            _setting("subagent_default_model", "") or "").strip(),
        spawn_override_enabled=bool(_setting("spawn_override_enabled", False)),
        spawn_override_allowlist=tuple(
            str(entry).strip() for entry in raw if str(entry).strip()),
    )


class RoutingError(Exception):
    """A refused spawn routing. ``code`` is machine-readable; ``level``
    names the failing resolution level (override / preset / default)."""

    def __init__(self, code: str, message: str, *, level: str) -> None:
        super().__init__(message)
        self.code = code
        self.level = level


@dataclass(frozen=True)
class SpawnTarget:
    provider: str
    model: str
    base_url: str | None
    params: tuple[tuple[str, object], ...]
    source: str  # "override" | "preset" | "default" | "inherit"


def allowlist_matches(allowlist: tuple[str, ...], provider: str, model: str) -> bool:
    """Bare 'provider' entries match any model; 'provider/glob' entries
    additionally fnmatch the model (case-insensitive)."""
    for entry in allowlist:
        entry_provider, sep, glob = entry.partition("/")
        if entry_provider != provider:
            continue
        if not sep:
            return True
        if model and fnmatch.fnmatchcase(model.lower(), glob.lower()):
            return True
    return False


def _allowlist_covers_provider(allowlist: tuple[str, ...], provider: str) -> bool:
    return any(entry.partition("/")[0] == provider for entry in allowlist)


def _configured_model_for(app_config: Mapping[str, Any], provider: str) -> str:
    """The provider's configured/default model (mirrors the console
    selection builder's configured_model lookup); '' when none. Registry
    entries carry no default model by design (ADR-146)."""
    from tldw_chatbook.Chat.custom_endpoint_registry import split_custom_endpoint_id
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    if split_custom_endpoint_id(provider):
        return ""
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return ""
    section = api_settings.get(provider_config_key(provider))
    if not isinstance(section, Mapping):
        return ""
    for key in ("model", "api_model", "default_model"):
        value = str(section.get(key) or "").strip()
        if value:
            return value
    return ""


def _default_readiness(app_config: Mapping[str, Any], provider: str) -> str | None:
    """None when the provider may be sent to, else the human-readable block.

    Wraps get_provider_readiness (Chat/provider_readiness.py:476) — the same
    readiness the Console send path projects, custom-ep aware per ADR-146: a
    ``custom-ep:<slug>`` id checks its registry entry's FAMILY readiness key
    (a raw custom-ep id is not an api_settings section and would always read
    as "Unknown provider"), and an entry whose declared credential does not
    resolve blocks even when the family is keyless — both mirroring
    console_provider_gateway's resolution seam.
    """
    from tldw_chatbook.Chat.console_provider_support import (
        resolve_console_provider_identity,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        _custom_endpoint_missing_key_readiness,
    )
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        entry_for, family_execution_key,
    )
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    entry = entry_for(app_config, provider)
    identity = resolve_console_provider_identity(
        family_execution_key(entry.family) if entry is not None else provider
    )
    readiness = get_provider_readiness(identity.readiness_key, app_config)
    if entry is not None and readiness.ready:
        readiness = (
            _custom_endpoint_missing_key_readiness(
                entry, identity.readiness_key, None
            )
            or readiness
        )
    return None if readiness.ready else readiness.user_message


def resolve_child_params(
    app_config: Mapping[str, Any],
    provider: str,
    model: str,
    *,
    preset_params: tuple[tuple[str, object], ...] = (),
) -> tuple[tuple[str, object], ...]:
    """Six-layer stack, highest first: preset params -> per-model profile ->
    console.provider_defaults -> registry entry params -> chat_defaults ->
    api_settings scalars -> function fallbacks. Layers 2-6 are merged by
    build_default_console_session_settings; entry params ride its
    extra_sources seam; preset params overlay last."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_default_console_session_settings,
    )
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        entry_for, split_custom_endpoint_id,
    )

    entry_params: dict[str, object] = {}
    if split_custom_endpoint_id(provider):
        entry = entry_for(app_config, provider)
        if entry is not None:
            entry_params = params_to_dict(entry.params)
    extra = (entry_params,) if entry_params else ()
    settings = build_default_console_session_settings(
        app_config, provider, model or None, extra_sources=extra)
    merged = {
        key: getattr(settings, key)
        for key in KNOWN_SAMPLING_PARAM_KEYS
        if getattr(settings, key, None) is not None
    }
    merged.update(params_to_dict(preset_params))
    return params_to_tuple(merged)


def resolve_spawn_target(
    app_config: Mapping[str, Any],
    *,
    parent_provider: str,
    parent_model: str,
    preset: AgentDefinition | None = None,
    override_provider: str = "",
    override_model: str = "",
    routing: AgentsRoutingConfig,
    readiness: Callable[[Mapping[str, Any], str], str | None] | None = None,
) -> SpawnTarget:
    """Resolve where a spawned child runs. Raises RoutingError on refusal."""
    from tldw_chatbook.Chat.console_provider_support import (
        supported_console_provider_readiness_keys,
    )
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        entry_for, split_custom_endpoint_id,
    )
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    override_provider = (override_provider or "").strip()
    override_model = (override_model or "").strip()
    if (override_provider or override_model) and not routing.spawn_override_enabled:
        raise RoutingError(
            "override_disabled",
            "ad-hoc provider/model args are disabled "
            "([agents] spawn_override_enabled = false)",
            level="override")
    if override_provider and not _allowlist_covers_provider(
            routing.spawn_override_allowlist, override_provider):
        raise RoutingError(
            "provider_not_allowlisted",
            f"provider '{override_provider}' is not in spawn_override_allowlist",
            level="override")

    if override_provider:
        provider, source = override_provider, "override"
    elif preset is not None and preset.provider:
        provider, source = preset.provider, "preset"
    elif routing.subagent_default_provider:
        provider, source = routing.subagent_default_provider, "default"
    else:
        provider, source = parent_provider, "inherit"

    # Provider validity is checked before model fill so a routed-to phantom
    # provider/endpoint reports itself (unknown_provider/unknown_endpoint_slug)
    # instead of the secondary no_model_resolved.
    base_url: str | None = None
    if split_custom_endpoint_id(provider):
        entry = entry_for(app_config, provider)
        if entry is None:
            raise RoutingError(
                "unknown_endpoint_slug",
                f"registry has no endpoint '{provider}'",
                level=source)
        base_url = entry.base_url
    elif provider_config_key(provider) not in set(
            supported_console_provider_readiness_keys()):
        raise RoutingError(
            "unknown_provider",
            f"'{provider}' is not a known provider id",
            level=source)

    model = (
        override_model
        or (preset.model if preset is not None else "")
        or routing.subagent_default_model
    )
    if not model and provider == parent_provider:
        model = parent_model
    if not model and source != "inherit":
        model = _configured_model_for(app_config, provider)
        if not model:
            raise RoutingError(
                "no_model_resolved",
                f"target provider '{provider}' has no configured/default "
                "model; set one or name a model",
                level=source)

    if (override_provider or override_model) and provider != parent_provider \
            and not allowlist_matches(
                routing.spawn_override_allowlist, provider, model):
        raise RoutingError(
            "provider_not_allowlisted",
            f"final target '{provider}/{model}' matches no allowlist entry",
            level="override")

    check = readiness if readiness is not None else _default_readiness
    blocked = check(app_config, provider)
    if blocked is not None:
        raise RoutingError("provider_not_ready", blocked, level=source)

    params = resolve_child_params(
        app_config, provider, model,
        preset_params=preset.params if preset is not None else ())
    return SpawnTarget(
        provider=provider, model=model, base_url=base_url,
        params=params, source=source)
