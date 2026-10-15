"""Native agent presets narrow the existing agent runtime's authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    from .components import NativeComponent
from tldw_chatbook.Agents.agent_models import AgentDefinition, PluginContextText


@dataclass(frozen=True)
class NativeAgentDefinition(AgentDefinition):
    """Live-only native fields; standalone serialization remains unchanged."""

    plugin_context: PluginContextText | None = None
    inherit_tools: bool = False


def resolve_agent_tools(
    value: str | tuple[str, ...], eligible: frozenset[str]
) -> frozenset[str]:
    """Resolve an explicit allowlist without converting empty into inherit."""
    if value == "inherit":
        return eligible
    if not isinstance(value, tuple) or any(not isinstance(v, str) for v in value):
        raise ValueError("plugin_agent_tools_unavailable")
    requested = frozenset(value)
    if not requested <= eligible:
        raise ValueError("plugin_agent_tools_unavailable")
    return requested


def resolve_component_tools(
    component: NativeComponent, registry: ToolCatalogRegistry, eligible: frozenset[str]
) -> frozenset[str]:
    """Resolve immutable canonical restrictions against the actual parent registry."""
    import json

    definition = json.loads(component.definition_json)
    requested = definition.get("tools", definition.get("allowed-tools", "inherit"))
    if (
        component.kind == "skill"
        and isinstance(requested, str)
        and requested != "inherit"
    ):
        requested = tuple(requested.split())
    if requested == "inherit":
        return resolve_agent_tools("inherit", eligible)
    if not requested:
        return frozenset()
    ids = {
        entry.id: entry.name
        for entry in registry.list_catalog()
        if entry.name in eligible
    }
    mappings = {
        row["mapping_id"]: row for row in json.loads(component.snapshot.mappings_json)
    }
    names = []
    for declared in requested:
        if declared.startswith(("builtin:", "local:")) and declared in ids:
            names.append(ids[declared])
            continue
        mapping = mappings.get(tool_constraint_id(component.component_id, declared))
        if mapping is None:
            raise ValueError("plugin_agent_tools_unavailable")
        # M4's runtime names encode installation/component/raw-tool identity.
        tool = next(
            (
                row
                for row in mappings.values()
                if row["kind"] == "tool"
                and row["mapping_id"] == mapping["target_reference"]
            ),
            None,
        )
        if tool is None:
            raise ValueError("plugin_agent_tools_unavailable")
        import hashlib

        raw_name = tool["target_reference"].partition("::")[2]
        name = (
            "plugin_mcp_"
            + hashlib.sha256(
                (
                    component.snapshot.installation_id
                    + ":"
                    + tool["component_id"]
                    + ":"
                    + raw_name
                ).encode()
            ).hexdigest()[:48]
        )
        names.append(name)
    return resolve_agent_tools(tuple(names), eligible)


def tool_constraint_id(component_id: str, declared: str) -> str:
    import hashlib

    return component_id + ":tool:" + hashlib.sha256(declared.encode()).hexdigest()[:24]


def project_agent(
    component: NativeComponent,
    registry: ToolCatalogRegistry,
    eligible: frozenset[str],
    provider: str,
) -> NativeAgentDefinition:
    """Adapt a native preset without putting package prose into system authority."""
    import hashlib
    import json

    from .admission import PluginUnavailable
    from .components import NativeComponent, component_body
    from .context import instruction_block

    if not isinstance(component, NativeComponent) or component.kind != "agent":
        raise PluginUnavailable("plugin_agent_not_admitted")
    snapshot = component.snapshot
    definition = json.loads(component.definition_json)
    model = resolve_component_model(component, provider)
    tools = resolve_component_tools(component, registry, eligible)
    return NativeAgentDefinition(
        name="plugin-"
        + hashlib.sha256(
            (snapshot.installation_id + ":" + component.component_id).encode()
        ).hexdigest()[:48],
        description=(
            snapshot.alias
            + ":"
            + component.component_id.partition(":")[2]
            + " — "
            + definition["description"]
        )[:200],
        instructions="",
        plugin_context=instruction_block(
            snapshot.installation_id,
            component.component_id,
            snapshot.revision_digest,
            component_body(component),
        ),
        inherit_tools=False,
        tool_allowlist=tuple(sorted(tools)),
        model=model,
    )


def resolve_component_model(component: NativeComponent, provider: str) -> str:
    """A model mapping only selects a supported model on the parent's provider."""
    import json

    from .admission import PluginUnavailable

    definition = json.loads(component.definition_json)
    requested = definition.get("model", definition.get("metadata", {}).get("model"))
    if requested is None:
        return ""
    mappings = [
        row
        for row in json.loads(component.snapshot.mappings_json)
        if row["component_id"] == component.component_id and row["kind"] == "model"
    ]
    if len(mappings) != 1:
        raise PluginUnavailable("plugin_model_mapping_required")
    mapped_provider, _, model = mappings[0]["target_reference"].partition("/")
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    if not model or provider_config_key(mapped_provider) != provider_config_key(
        provider
    ):
        raise PluginUnavailable("plugin_model_provider_mismatch")
    return model
