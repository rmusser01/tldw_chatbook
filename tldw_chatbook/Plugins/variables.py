"""Pure declared-reference projection through the existing protected owner."""

from __future__ import annotations

import hashlib
import json
import re

from .package_files import canonical_json

ROOT_NAMES = frozenset({"PLUGIN_ROOT", "PLUGIN_DATA", "CHATBOOK_WORKSPACE_DATA"})
REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
HANDLE = re.compile(r"var1\.([0-9a-f]{32})\.([1-9][0-9]{0,18})")


def consumed(component) -> frozenset[str]:
    """Inspect only qualified command fields; never vendor arbitrary metadata."""
    definition = json.loads(component.definition_json)
    if component.kind != "hook" or definition.get("type") != "command":
        return frozenset()
    result = set()
    for value in [
        *definition.get("argv", ()),
        definition.get("cwd", ""),
        *definition.get("env", {}).values(),
    ]:
        if isinstance(value, str):
            result.update(REFERENCE.findall(value))
        elif isinstance(value, dict) and set(value) == {"variable"}:
            result.add(value["variable"])
    return frozenset(result)


def mapping_id(component_id: str, name: str) -> str:
    return (
        "variable:" + hashlib.sha256((component_id + "\0" + name).encode()).hexdigest()
    )


def handle(metadata: dict) -> str:
    return f"var1.{metadata['reference_id']}.{metadata['authority_generation']}"


def reference(value: str) -> tuple[str, int]:
    matched = HANDLE.fullmatch(value) if isinstance(value, str) else None
    if matched is None or int(matched[2]) >= 2**63:
        raise ValueError("plugin_variable_reference_invalid")
    return matched[1], int(matched[2])


def credentials(coordinator):
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService

    if not isinstance(coordinator.mcp_mapping_owner, LocalMCPControlService):
        raise PermissionError("plugin_variable_owner_unavailable")
    return coordinator.mcp_mapping_owner._credentials()


def arguments(installation_id, name, declaration):
    return {
        "installation_id": installation_id,
        "name": name,
        "value_type": declaration["type"],
        "secret": declaration["secret"],
    }


def _retained_mapping(installation_id, inspection, component_id, name, target):
    """Rebuild the reviewed row without consulting mutable credential storage."""
    component = inspection.inventory[component_id]
    declarations = json.loads(inspection.variables_json)
    if (
        name not in consumed(component)
        or name not in declarations
        or name in ROOT_NAMES
    ):
        raise ValueError("plugin_variable_mapping_invalid")
    declaration = declarations[name]
    reference_id, generation = reference(target)
    metadata = {
        "schema": 2,
        "kind": "plugin_variable",
        "reference_id": reference_id,
        "authority_generation": generation,
        **arguments(installation_id, name, declaration),
        "revoked": False,
    }
    return {
        "installation_id": installation_id,
        "revision_digest": inspection.effective_digest,
        "component_id": component_id,
        "mapping_id": mapping_id(component_id, name),
        "kind": "configuration",
        "target_reference": target,
        "definition_digest": hashlib.sha256(
            canonical_json(
                {
                    "component": json.loads(component.definition_json),
                    "declaration": declaration,
                }
            ).encode()
        ).hexdigest(),
        "configuration_digest": hashlib.sha256(
            canonical_json(metadata).encode()
        ).hexdigest(),
        "credential_bindings": [],
    }


def capture_mapping(
    coordinator, installation_id, inspection, component_id, name, target
):
    mapping = _retained_mapping(installation_id, inspection, component_id, name, target)
    credentials(coordinator).validate_variable(
        *reference(target),
        **arguments(installation_id, name, json.loads(inspection.variables_json)[name]),
    )
    return mapping


def _validate_retained_mapping(mapping, inspection):
    """Check authenticated binding, not readiness; return its declared name."""
    names = [
        name
        for name in consumed(inspection.inventory[mapping["component_id"]])
        if mapping_id(mapping["component_id"], name) == mapping["mapping_id"]
    ]
    if (
        len(names) != 1
        or _retained_mapping(
            mapping["installation_id"],
            inspection,
            mapping["component_id"],
            names[0],
            mapping["target_reference"],
        )
        != mapping
    ):
        raise ValueError("plugin_variable_mapping_changed")
    return names[0]


def validate_mapping(coordinator, mapping, inspection):
    name = _validate_retained_mapping(mapping, inspection)
    credentials(coordinator).validate_variable(
        *reference(mapping["target_reference"]),
        **arguments(
            mapping["installation_id"],
            name,
            json.loads(inspection.variables_json)[name],
        ),
    )


def values(
    coordinator, installation_id, inspection, component_id, mappings, *, resolve=False
):
    """Check readiness without values; return typed values only for actual launch."""
    from tldw_chatbook.MCP.credential_bindings import CredentialBindingService

    declarations = json.loads(inspection.variables_json)
    result = {}
    for name in consumed(inspection.inventory[component_id]) - ROOT_NAMES:
        declaration = declarations.get(name)
        if declaration is None:
            raise ValueError("plugin_variable_undeclared")
        selected = [
            row
            for row in mappings
            if row["component_id"] == component_id
            and row["mapping_id"] == mapping_id(component_id, name)
        ]
        if selected:
            if len(selected) != 1:
                raise ValueError("plugin_variable_mapping_invalid")
            validate_mapping(coordinator, selected[0], inspection)
            if resolve:
                result[name] = credentials(coordinator).resolve_variable(
                    *reference(selected[0]["target_reference"]),
                    **arguments(installation_id, name, declaration),
                )
        else:
            if declaration["secret"] or "default" not in declaration:
                raise ValueError("plugin_variable_unset")
            CredentialBindingService._variable_value(
                declaration["type"], declaration["default"]
            )
            if resolve:
                result[name] = declaration["default"]
    return result


def text(value: str | int | bool) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    return str(value)
