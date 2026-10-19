"""Reviewed references adapt existing catalogs and routing, never grant tools."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .admission import PluginUnavailable
from .models import ComponentRecord, PackageInspection
from .package_files import canonical_json


def declared_tools(component: ComponentRecord) -> tuple[str, ...] | None:
    """Distinguish declared EMPTY from inheritance without rewriting labels."""
    definition = json.loads(component.definition_json)
    value = (
        definition.get("tools")
        if component.kind == "agent"
        else definition.get("allowed-tools")
    )
    if value is None or component.kind == "agent" and value == "inherit":
        return None
    return tuple(value) if isinstance(value, list) else tuple(value.split())


def mapping_id(component_id: str, label: str) -> str:
    return (
        "native-tool-"
        + hashlib.sha256((component_id + "\0" + label).encode()).hexdigest()
    )


def owned_mcp_tool_name(installation_id: str, component_id: str, tool: str) -> str:
    return (
        "plugin_mcp_"
        + hashlib.sha256(
            (installation_id + ":" + component_id + ":" + tool).encode()
        ).hexdigest()[:48]
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def capture_reference(
    installation_id: str,
    inspection: PackageInspection,
    component_id: str,
    kind: str,
    target: str,
    *,
    label: str = "",
    mappings: tuple[dict, ...] | list[dict] = (),
) -> dict:
    """Capture only current host catalog references for selected native material."""
    component = inspection.inventory[component_id]
    definition = json.loads(component.definition_json)
    credentials = []
    if kind == "model":
        from tldw_chatbook.Chat.custom_endpoint_registry import (
            entry_for,
            split_custom_endpoint_id,
        )
        from tldw_chatbook.Chat.provider_readiness import provider_config_key
        from tldw_chatbook.config import get_cli_providers_and_models, load_settings

        if component.kind != "agent" or "model" not in definition:
            raise ValueError("plugin_model_not_declared")
        provider, separator, model = target.partition("::")
        config = load_settings()
        if not separator or model not in get_cli_providers_and_models().get(
            provider, ()
        ):
            raise PermissionError("plugin_model_reference_unavailable")
        endpoint = (
            entry_for(config, provider) if split_custom_endpoint_id(provider) else None
        )
        if split_custom_endpoint_id(provider) and endpoint is None:
            raise PermissionError("plugin_model_reference_unavailable")
        settings = config.get("api_settings", {}).get(provider_config_key(provider), {})
        configuration = {
            "provider": provider,
            "model": model,
            "base_url": endpoint.base_url
            if endpoint
            else settings.get("api_url", settings.get("base_url")),
        }
        identity = "native-model-" + hashlib.sha256(component_id.encode()).hexdigest()
    elif kind == "tool":
        if label not in (declared_tools(component) or ()):
            raise ValueError("plugin_tool_not_declared")
        identity = mapping_id(component_id, label)
        if target.startswith("builtin:"):
            from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

            owner = BuiltinToolProvider()
            if target not in {entry.id for entry in owner.list_catalog()}:
                raise PermissionError("plugin_tool_reference_unavailable")
            configuration = asdict(owner.load_schema(target))
        else:
            sources = [
                row
                for row in mappings
                if row["kind"] == "tool"
                and row["installation_id"] == installation_id
                and row["revision_digest"] == inspection.effective_digest
                and row["target_reference"] == target
                and row["component_id"] in inspection.inventory
                and inspection.inventory[row["component_id"]].kind == "mcp"
            ]
            if len(sources) != 1:
                raise PermissionError("plugin_tool_reference_unavailable")
            source = sources[0]
            pending, closure = list(component.dependencies), set()
            while pending:
                key = pending.pop()
                if key in closure:
                    continue
                closure.add(key)
                pending.extend(inspection.inventory[key].dependencies)
            if source["component_id"] not in closure:
                raise PermissionError("plugin_tool_dependency_required")
            configuration = source
            credentials = source["credential_bindings"]
    else:
        raise ValueError("plugin_reference_kind_invalid")
    return {
        "installation_id": installation_id,
        "mapping_id": identity,
        "component_id": component_id,
        "revision_digest": inspection.effective_digest,
        "kind": kind,
        "target_reference": target,
        "definition_digest": _digest(definition),
        "configuration_digest": _digest(configuration),
        "credential_bindings": credentials,
    }


def validate_reference(
    mapping: dict,
    inspection: PackageInspection,
    mappings: tuple[dict, ...] | list[dict],
) -> None:
    component = inspection.inventory[mapping["component_id"]]
    label = next(
        (
            name
            for name in declared_tools(component) or ()
            if mapping_id(component.component_id, name) == mapping["mapping_id"]
        ),
        "",
    )
    current = capture_reference(
        mapping["installation_id"],
        inspection,
        mapping["component_id"],
        mapping["kind"],
        mapping["target_reference"],
        label=label,
        mappings=mappings,
    )
    if current != mapping:
        raise PermissionError("plugin_host_reference_changed")


def capture_model_mapping(
    owner: Any,
    installation_id: str,
    inspection: PackageInspection,
    component_id: str,
    target: str,
) -> dict[str, Any]:
    """Capture a canonical current reference from the actual local model owner."""
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )

    from .package_files import canonical_json

    if not isinstance(owner, LocalLLMProviderCatalogService):
        raise PluginUnavailable("plugin_model_owner_unavailable")
    component = inspection.inventory[component_id]
    definition = json.loads(component.definition_json)
    if component.kind not in {"skill", "agent"} or not definition.get(
        "model", definition.get("metadata", {}).get("model")
    ):
        raise ValueError("plugin_model_mapping_invalid")
    record = owner.get_model_metadata(target)
    if record["id"] != target:
        raise ValueError("plugin_model_reference_not_canonical")
    endpoint = owner._current_endpoint_fingerprint(provider_key=record["provider"])
    if endpoint is None:
        raise PluginUnavailable("plugin_model_configuration_unavailable")
    return {
        "installation_id": installation_id,
        "mapping_id": component_id + ":model",
        "component_id": component_id,
        "revision_digest": inspection.effective_digest,
        "kind": "model",
        "target_reference": target,
        "definition_digest": hashlib.sha256(
            component.definition_json.encode()
        ).hexdigest(),
        "configuration_digest": hashlib.sha256(
            canonical_json({"model": record, "endpoint": endpoint}).encode()
        ).hexdigest(),
        "credential_bindings": [],
    }


def capture_tool_constraint(
    coordinator,
    installation_id: str,
    inspection: PackageInspection,
    component_id: str,
    declared: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    """Review a restriction against an already reviewed exact M4 tool reference."""
    from .package_files import canonical_json

    component = inspection.inventory[component_id]
    definition = json.loads(component.definition_json)
    tokens = definition.get("tools", definition.get("allowed-tools", ()))
    if isinstance(tokens, str):
        tokens = tokens.split()
    pending, dependencies = list(component.dependencies), set()
    while pending:
        key = pending.pop()
        if key not in dependencies:
            dependencies.add(key)
            pending.extend(inspection.inventory[key].dependencies)
    if (
        component.kind not in {"agent", "skill"}
        or declared not in tokens
        or declared.startswith(("builtin:", "local:"))
        or declared == "inherit"
        or target["kind"] != "tool"
        or target["installation_id"] != installation_id
        or target["revision_digest"] != inspection.effective_digest
        or target["component_id"] not in dependencies
    ):
        raise PluginUnavailable("plugin_tool_mapping_invalid")
    coordinator.validate_mapping(target, inspection, (target,))
    return {
        "installation_id": installation_id,
        "mapping_id": tool_constraint_id(component_id, declared),
        "component_id": component_id,
        "revision_digest": inspection.effective_digest,
        "kind": "configuration",
        "target_reference": target["mapping_id"],
        "definition_digest": hashlib.sha256(
            component.definition_json.encode()
        ).hexdigest(),
        "configuration_digest": hashlib.sha256(
            canonical_json(target).encode()
        ).hexdigest(),
        "credential_bindings": [],
    }


def tool_constraint_id(component_id: str, declared: str) -> str:
    return component_id + ":tool:" + hashlib.sha256(declared.encode()).hexdigest()[:24]


def _reference_form(mapping: dict, inspection: PackageInspection) -> str:
    component = inspection.inventory.get(mapping["component_id"])
    if component is None or mapping["revision_digest"] != inspection.effective_digest:
        raise PermissionError("plugin_reference_identity_invalid")
    identity, kind = mapping["mapping_id"], mapping["kind"]
    if component.kind in {"agent", "skill"}:
        if kind == "model":
            if identity == component.component_id + ":model":
                return "canonical_model"
            if (
                component.kind == "agent"
                and identity
                == "native-model-"
                + hashlib.sha256(component.component_id.encode()).hexdigest()
            ):
                return "dev_model"
        for label in declared_tools(component) or ():
            if kind == "tool" and identity == mapping_id(component.component_id, label):
                return "dev_tool"
            if kind == "configuration" and identity == tool_constraint_id(
                component.component_id, label
            ):
                return "canonical_tool"
    elif (
        component.kind == "mcp"
        and kind in {"connection", "tool"}
        and not identity.startswith("native-")
    ):
        return "mcp"
    raise PermissionError("plugin_reference_identity_invalid")


def validate_canonical_reference(coordinator, mapping, inspection, mappings) -> None:
    form = _reference_form(mapping, inspection)
    if form == "canonical_model":
        current = capture_model_mapping(
            coordinator.model_mapping_owner,
            mapping["installation_id"],
            inspection,
            mapping["component_id"],
            mapping["target_reference"],
        )
    elif form == "canonical_tool":
        component = inspection.inventory[mapping["component_id"]]
        labels = [
            label
            for label in declared_tools(component) or ()
            if tool_constraint_id(component.component_id, label)
            == mapping["mapping_id"]
        ]
        targets = [
            row
            for row in mappings
            if row["installation_id"] == mapping["installation_id"]
            and row["revision_digest"] == mapping["revision_digest"]
            and row["mapping_id"] == mapping["target_reference"]
            and row["kind"] == "tool"
        ]
        if len(labels) != 1 or len(targets) != 1:
            raise PermissionError("plugin_tool_mapping_invalid")
        current = capture_tool_constraint(
            coordinator,
            mapping["installation_id"],
            inspection,
            mapping["component_id"],
            labels[0],
            targets[0],
        )
    else:
        raise PermissionError("plugin_reference_identity_invalid")
    if current != mapping:
        raise PermissionError("plugin_host_reference_changed")


def mapped_tools(
    installation_id: str,
    component: ComponentRecord,
    mappings: tuple[dict, ...] | list[dict],
) -> tuple[str, ...] | None:
    """Resolve protected references; direct canonical IDs still need the actual registry."""
    declared = declared_tools(component)
    if declared is None:
        return None
    resolved = []
    for label in declared:
        choices = [
            row
            for row in mappings
            if row["installation_id"] == installation_id
            and row["component_id"] == component.component_id
            and row["mapping_id"]
            in {
                mapping_id(component.component_id, label),
                tool_constraint_id(component.component_id, label),
            }
        ]
        if not choices and label.startswith(("builtin:", "local:")):
            resolved.append(label)
            continue
        if len(choices) != 1:
            raise PermissionError("plugin_tool_mapping_required")
        mapping = choices[0]
        target = mapping["target_reference"]
        if mapping["mapping_id"] == mapping_id(component.component_id, label):
            if mapping["kind"] != "tool":
                raise PermissionError("plugin_tool_mapping_invalid")
            if target.startswith("builtin:"):
                from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

                resolved.append(BuiltinToolProvider().load_schema(target).name)
                continue
            sources = [row for row in mappings if row["target_reference"] == target]
        else:
            if mapping["kind"] != "configuration":
                raise PermissionError("plugin_tool_mapping_invalid")
            sources = [row for row in mappings if row["mapping_id"] == target]
        sources = [
            row
            for row in sources
            if row["installation_id"] == installation_id
            and row["revision_digest"] == mapping["revision_digest"]
            and row["kind"] == "tool"
            and row["component_id"].startswith("mcp:")
        ]
        if len(sources) != 1:
            raise PermissionError("plugin_tool_mapping_invalid")
        source = sources[0]
        resolved.append(
            owned_mcp_tool_name(
                installation_id,
                source["component_id"],
                source["target_reference"].partition("::")[2],
            )
        )
    return tuple(resolved)


def mapped_model(
    component: ComponentRecord, mappings: tuple[dict, ...] | list[dict]
) -> tuple[str, str]:
    definition = json.loads(component.definition_json)
    if not definition.get("model", definition.get("metadata", {}).get("model")):
        return "", ""
    rows = [
        row
        for row in mappings
        if row["component_id"] == component.component_id and row["kind"] == "model"
    ]
    if len(rows) != 1:
        raise PermissionError("plugin_model_mapping_required")
    row = rows[0]
    if row["mapping_id"] == component.component_id + ":model":
        provider, separator, model = row["target_reference"].partition("/")
    elif (
        row["mapping_id"]
        == "native-model-" + hashlib.sha256(component.component_id.encode()).hexdigest()
    ):
        provider, separator, model = row["target_reference"].partition("::")
    else:
        raise PermissionError("plugin_model_mapping_invalid")
    if not separator or not provider or not model:
        raise PermissionError("plugin_model_mapping_invalid")
    return provider, model


def resolve_tool_ceiling(values, registry, eligible: frozenset[str]) -> tuple[str, ...]:
    """Resolve canonical IDs against the actual parent's catalog; never grant."""
    if values is None:
        return tuple(sorted(eligible))
    ids = (
        dict(registry)
        if isinstance(registry, dict)
        else (
            {row.id: row.name for row in registry.list_catalog()}
            if registry is not None
            else {}
        )
    )
    resolved = []
    for value in values:
        if value.startswith(("builtin:", "local:")):
            if value not in ids:
                raise PermissionError("plugin_tool_unavailable")
            value = ids[value]
        if value not in eligible:
            raise PermissionError("plugin_tool_unavailable")
        resolved.append(value)
    return tuple(sorted(set(resolved)))
