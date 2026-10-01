"""Managed presets use existing AgentDefinition and narrowed parent tools."""

import hashlib
import json

from tldw_chatbook.Agents.agent_models import AgentDefinition, validate_agent_definition

from .admission import PluginUnavailable, RunPluginSnapshot
from .components import component_body
from .context import instruction_block


def resolve_agent_tools(
    value: str | tuple[str, ...], eligible: frozenset[str]
) -> frozenset[str]:
    """Keep explicit EMPTY distinct from inheritance and reject unavailable names."""
    if value == "inherit":
        return eligible
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError("plugin_agent_tools_invalid")
    requested = frozenset(value)
    if not requested <= eligible:
        raise ValueError("plugin_agent_tools_unavailable")
    return requested


def agent_definition(
    snapshot: RunPluginSnapshot,
    component_id: str,
    eligible: frozenset[str],
    *,
    parent_provider: str = "",
) -> AgentDefinition:
    """Build an ephemeral preset; declared model references require host mapping."""
    component = snapshot.inspection.inventory[component_id]
    definition = json.loads(component.definition_json)
    if component.kind != "agent":
        raise PluginUnavailable("plugin_agent_unavailable")
    from .host_references import mapped_model, mapped_tools

    mappings = json.loads(snapshot.mappings_json)
    declared = mapped_tools(snapshot.installation_id, component, mappings)
    tools = eligible if declared is None else frozenset(declared) & eligible
    provider, model = mapped_model(component, mappings)
    if model:
        from tldw_chatbook.Chat.provider_readiness import provider_config_key

        if not parent_provider or provider_config_key(provider) != provider_config_key(
            parent_provider
        ):
            raise PluginUnavailable("plugin_model_parent_mismatch")
        provider = parent_provider  # Retain the actual parent endpoint identity.
    preset = AgentDefinition(
        name="plugin-agent-"
        + hashlib.sha256(
            f"{snapshot.installation_id}:{component_id}".encode()
        ).hexdigest()[:48],
        description=definition.get("description", ""),
        instructions=instruction_block(
            snapshot.installation_id,
            component_id,
            snapshot.revision_digest,
            component_body(snapshot, component_id),
        ),
        tool_allowlist=tuple(sorted(tools)),
        provider=provider,
        model=model,
    )
    if validate_agent_definition(preset):
        raise PluginUnavailable("plugin_agent_definition_invalid")
    return preset
