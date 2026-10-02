"""Reviewed references adapt existing catalogs and routing, never grant tools."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

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
            source = next(
                (
                    row
                    for row in mappings
                    if row["kind"] == "tool"
                    and row["target_reference"] == target
                    and inspection.inventory[row["component_id"]].kind == "mcp"
                ),
                None,
            )
            if source is None:
                raise PermissionError("plugin_tool_reference_unavailable")
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


def mapped_tools(
    installation_id: str,
    component: ComponentRecord,
    mappings: tuple[dict, ...] | list[dict],
) -> tuple[str, ...] | None:
    """Resolve reviewed names only; callers intersect with the actual parent."""
    declared = declared_tools(component)
    if declared is None:
        return None
    resolved = []
    for label in declared:
        mapping = next(
            (
                row
                for row in mappings
                if row["mapping_id"] == mapping_id(component.component_id, label)
            ),
            None,
        )
        if mapping is None:
            raise PermissionError("plugin_tool_mapping_required")
        target = mapping["target_reference"]
        if target.startswith("builtin:"):
            resolved.append(target.removeprefix("builtin:"))
        else:
            source = next(
                row
                for row in mappings
                if row["kind"] == "tool"
                and row["target_reference"] == target
                and row["component_id"].startswith("mcp:")
            )
            resolved.append(
                owned_mcp_tool_name(
                    installation_id, source["component_id"], target.partition("::")[2]
                )
            )
    return tuple(resolved)


def mapped_model(
    component: ComponentRecord, mappings: tuple[dict, ...] | list[dict]
) -> tuple[str, str]:
    if "model" not in json.loads(component.definition_json):
        return "", ""
    rows = [
        row
        for row in mappings
        if row["component_id"] == component.component_id and row["kind"] == "model"
    ]
    if len(rows) != 1:
        raise PermissionError("plugin_model_mapping_required")
    return tuple(rows[0]["target_reference"].split("::", 1))
