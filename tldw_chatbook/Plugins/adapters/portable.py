"""Portable 1.0.0 inventory plus the closed Chatbook v1 extension."""

import hashlib
import json
import re
from pathlib import PurePosixPath

from ..models import ComponentRecord, Diagnostic
from ..package_files import PackageFileError, canonical_json
from ..schemas import (
    MCP_SCHEMA,
    NAMESPACE,
    NON_REQUIRED_EVENTS,
    closed,
    frontmatter,
    require,
    validate_extension,
    validate_hook,
    validate_mcp,
)


class ComponentLimitError(Exception):
    """Exhaustion must bypass component-local parse failure isolation."""


def inventory_package(
    capture,
    manifest: dict,
    *,
    max_components: int,
    skill_paths=None,
    skill_definitions=None,
):
    """Return inventory and constraints; missing declarations never imply permission."""
    inventory = {}
    diagnostics = []
    edges = {}
    variables = {}
    global_blockers = []

    def add(kind, local_id, path, definition, blockers=(), support="supported"):
        key = f"{kind}:{local_id}"
        if key in inventory:
            existing = inventory[key]
            inventory[key] = existing.model_copy(
                update={
                    "support": "invalid",
                    "activation_blockers": tuple(
                        sorted(set(existing.activation_blockers) | {"duplicate_id"})
                    ),
                }
            )
            diagnostics.append(
                Diagnostic(code="duplicate_id", path=path, component_id=key)
            )
            return
        if len(inventory) >= max_components:
            raise ComponentLimitError("component_count_limit")
        inventory[key] = ComponentRecord(
            component_id=key,
            kind=kind,
            local_id=local_id,
            path=path,
            definition_json=canonical_json(definition),
            activation_blockers=tuple(blockers),
            support=support,
        )

    if "skills" in capture.files:
        diagnostics.append(Diagnostic(code="skills_location_invalid", path="skills"))
    if skill_paths is None:
        skill_paths = [
            path
            for path in capture.files
            if len(PurePosixPath(path).parts) == 3
            and PurePosixPath(path).parts[0] == "skills"
            and PurePosixPath(path).name == "SKILL.md"
        ]
    for path in sorted(skill_paths):
        try:
            definition = (skill_definitions or {}).get(path)
            if definition is None:
                definition = frontmatter(
                    capture.read(path),
                    "skill",
                    directory_name=PurePosixPath(path).parent.name,
                )
            blockers = (
                ("skill_tools_mapping_required",)
                if definition.get("allowed-tools")
                else ()
            )
            add("skill", definition["name"], path, definition, blockers)
        except PackageFileError as exc:
            if str(exc) == "component_count_limit":
                raise
            diagnostics.append(Diagnostic(code=str(exc), path=path))

    if "mcp.json" in capture.files:
        try:
            mcp = closed(
                capture.document("mcp.json"),
                {"$schema", "mcpServers"},
                {"$schema", "mcpServers"},
            )
            require(
                mcp["$schema"] == MCP_SCHEMA and isinstance(mcp["mcpServers"], dict)
            )
            for local_id, raw in mcp["mcpServers"].items():
                try:
                    require(
                        bool(local_id)
                        and len(local_id) <= 128
                        and ":" not in local_id
                        and not any(c.isspace() for c in local_id)
                    )
                    definition = validate_mcp(raw, capture)
                    unsupported = definition["type"] == "sse"
                    add(
                        "mcp",
                        local_id,
                        "mcp.json",
                        definition,
                        ("transport_unsupported",) if unsupported else (),
                        "unsupported" if unsupported else "supported",
                    )
                except (PackageFileError, ValueError):
                    add(
                        "mcp",
                        local_id,
                        "mcp.json",
                        raw,
                        ("definition_invalid",),
                        "invalid",
                    )
                    diagnostics.append(
                        Diagnostic(
                            code="mcp_entry_invalid",
                            path="mcp.json",
                            component_id=f"mcp:{local_id}",
                        )
                    )
        except PackageFileError as exc:
            diagnostics.append(Diagnostic(code=str(exc), path="mcp.json"))
    elif "mcp.json" in capture.directories:
        diagnostics.append(Diagnostic(code="mcp_location_invalid", path="mcp.json"))

    extensions = manifest.get("extensions", {})
    if not isinstance(extensions, dict):
        diagnostics.append(Diagnostic(code="extensions_ignored", path="plugin.json"))
        global_blockers.append("constraints_unknown")
        extensions = {}
    for namespace in extensions:
        if namespace != NAMESPACE:
            diagnostics.append(Diagnostic(code="extension_ignored", path="plugin.json"))
    if NAMESPACE in extensions:
        try:
            extension = validate_extension(extensions[NAMESPACE])
        except (PackageFileError, TypeError, ValueError):
            global_blockers.append("constraints_unknown")
            diagnostics.append(Diagnostic(code="extension_invalid", path="plugin.json"))
            extension = {}
        edges = {
            key: tuple(deps) for key, deps in extension.get("requires", {}).items()
        }
        variables = extension.get("variables", {})
        for kind, field in (
            ("command", "commands"),
            ("rule", "rules"),
            ("agent", "agents"),
        ):
            for path in extension.get(field, []):
                try:
                    definition = frontmatter(capture.read(path), kind)
                    blockers = (
                        ("model_mapping_required",)
                        if kind == "agent" and "model" in definition
                        else ()
                    )
                    if (
                        kind == "agent"
                        and isinstance(definition["tools"], list)
                        and definition["tools"]
                    ):
                        blockers += ("tool_mapping_required",)
                    add(kind, definition["name"], path, definition, blockers)
                except PackageFileError as exc:
                    add(
                        kind,
                        PurePosixPath(path).stem,
                        path,
                        {},
                        ("definition_invalid",),
                        "invalid",
                    )
                    diagnostics.append(Diagnostic(code=str(exc), path=path))
        if "hooks" in extension:
            path = extension["hooks"]
            try:
                hooks = closed(
                    capture.document(path), {"version", "hooks"}, {"version", "hooks"}
                )
                require(
                    type(hooks["version"]) is int
                    and hooks["version"] == 2
                    and isinstance(hooks["hooks"], list)
                )
                require(len(hooks["hooks"]) <= 64, "hook_count_limit")
                for index, raw in enumerate(hooks["hooks"]):
                    try:
                        definition = validate_hook(raw, capture)
                        local_id = definition["id"]
                        add("hook", local_id, path, definition)
                        if definition["type"] == "mcp_tool":
                            server = definition["server"]
                            if not server.startswith("mcp:"):
                                server = f"mcp:{server}"
                            edges[f"hook:{local_id}"] = tuple(
                                dict.fromkeys(
                                    (*edges.get(f"hook:{local_id}", ()), server)
                                )
                            )
                    except (PackageFileError, TypeError, ValueError):
                        local_id = raw.get("id") if isinstance(raw, dict) else None
                        if not isinstance(local_id, str) or not local_id:
                            local_id = f"invalid-{index}"
                        add(
                            "hook",
                            local_id,
                            path,
                            raw,
                            ("definition_invalid",),
                            "invalid",
                        )
                        if (
                            not isinstance(raw, dict)
                            or type(raw.get("required", False)) is not bool
                            or raw.get("required")
                        ):
                            global_blockers.append("constraints_unknown")
                        diagnostics.append(
                            Diagnostic(
                                code="hook_invalid",
                                path=path,
                                component_id=f"hook:{local_id}",
                            )
                        )
            except PackageFileError as exc:
                diagnostics.append(Diagnostic(code=str(exc), path=path))
                global_blockers.append("constraints_unknown")

    for name, declaration in variables.items():
        if declaration["required"] and "default" not in declaration:
            global_blockers.append(f"variable_unset:{name}")
    if edges.keys() - inventory.keys():
        global_blockers.append("dependency_scope_unknown")
    for key, component in list(inventory.items()):
        blockers = set(component.activation_blockers) | set(global_blockers)
        if component.kind == "hook" and component.support != "invalid":
            definition = json.loads(component.definition_json)
            if isinstance(definition, dict):
                values = [
                    *definition.get("argv", []),
                    *definition.get("env", {}).values(),
                ]
                for value in values:
                    if isinstance(value, str):
                        for variable in re.findall(
                            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value
                        ):
                            if (
                                variable not in {"PLUGIN_ROOT", "PLUGIN_DATA"}
                                and variable not in variables
                            ):
                                blockers.add(f"variable_undeclared:{variable}")
        for dep in edges.get(key, ()):
            if dep not in inventory:
                blockers.add(f"dependency_missing:{dep}")
            elif dep.startswith("hook:"):
                definition = json.loads(inventory[dep].definition_json)
                if (
                    isinstance(definition, dict)
                    and isinstance(definition.get("event"), str)
                    and definition["event"] in NON_REQUIRED_EVENTS
                ):
                    blockers.add(f"dependency_event_invalid:{dep}")
        inventory[key] = component.model_copy(
            update={
                "dependencies": edges.get(key, ()),
                "activation_blockers": tuple(sorted(blockers)),
            }
        )

    # A cycle taints only its cycle and downstream dependents.
    visiting = []
    visited = set()
    cyclic = set()

    def visit(key):
        if key in visiting:
            cyclic.update(visiting[visiting.index(key) :])
            return
        if key in visited or key not in inventory:
            return
        visiting.append(key)
        for dep in edges.get(key, ()):
            visit(dep)
        visiting.pop()
        visited.add(key)

    for key in inventory:
        visit(key)
    for key in cyclic:
        record = inventory[key]
        inventory[key] = record.model_copy(
            update={
                "activation_blockers": tuple(
                    sorted(set(record.activation_blockers) | {"dependency_cycle"})
                )
            }
        )
    changed = True
    while changed:
        changed = False
        for key, record in list(inventory.items()):
            blockers = set(record.activation_blockers)
            for dep in record.dependencies:
                if dep in inventory and inventory[dep].activation_blockers:
                    blockers.add(f"dependency_blocked:{dep}")
            if blockers != set(record.activation_blockers):
                inventory[key] = record.model_copy(
                    update={"activation_blockers": tuple(sorted(blockers))}
                )
                changed = True
    # Body changes affect interpretation; ignored manifest metadata does not.
    component_digests = {
        key: hashlib.sha256(capture.read(record.path)).hexdigest()
        for key, record in inventory.items()
        if record.kind in {"skill", "command", "rule", "agent"}
        and record.path.removeprefix("./") in capture.files
    }
    return (
        inventory,
        edges,
        variables,
        diagnostics,
        tuple(sorted(set(global_blockers))),
        component_digests,
    )
