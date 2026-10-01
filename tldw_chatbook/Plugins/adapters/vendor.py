"""Bounded vendor inventory reuses native records, metadata and authority contracts."""

import hashlib
from pathlib import PurePosixPath

from ..models import ComponentRecord, Diagnostic
from ..package_files import (
    MAX_DOCUMENT_BYTES,
    PackageCapture,
    PackageFileError,
    canonical_json,
    parse_document,
    validate_relative_member,
)
from ..schemas import (
    PLUGIN_SCHEMA,
    frontmatter,
    require,
    validate_manifest,
    validate_mcp,
)
from .portable import ComponentLimitError, inventory_package
from .vendor_hooks import inventory_vendor_hooks

CATALOG_PATH = ".chatbook-plugin/catalog.json"
VERSIONS = {
    "openai": "chatbook-openai/2026-10-01.1",
    "cursor": "chatbook-cursor/2026-10-01.1",
}
PRESENTATION = {
    "$schema",
    "name",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
    "logo",
    "displayName",
    "interface",
    "category",
    "tags",
    "source",
}
COMPONENTS = {
    "skills",
    "commands",
    "rules",
    "agents",
    "hooks",
    "mcpServers",
    "apps",
    "variables",
}


def catalog_input(capture: PackageCapture, external: dict | None) -> tuple[dict, bool]:
    """Executable catalog inputs must survive normal materialization/recovery."""
    retained = capture.document(CATALOG_PATH) if CATALOG_PATH in capture.files else None
    if external is None:
        return retained or {}, True
    external = parse_document(canonical_json(external).encode())
    return external, not external or retained == external


def _paths(capture, declaration: dict, field: str) -> tuple[str, ...]:
    suffixes = {
        "skills": {"SKILL.md"},
        "commands": {".md", ".mdc", ".markdown", ".txt"},
        "rules": {".md", ".mdc", ".markdown"},
        "agents": {".md", ".mdc", ".markdown"},
    }[field]
    explicit = field in declaration
    paths = declaration[field] if explicit else [field]
    paths = [paths] if isinstance(paths, str) else paths
    require(
        isinstance(paths, list)
        and all(isinstance(path, str) and path for path in paths),
        "vendor_component_paths_invalid",
    )
    found = set()
    for value in paths:
        path = validate_relative_member(value).as_posix()
        if path in capture.files:
            if field == "skills":
                require(
                    PurePosixPath(path).name == "SKILL.md", "vendor_skill_path_invalid"
                )
            else:
                require(
                    PurePosixPath(path).suffix in suffixes,
                    "vendor_component_path_invalid",
                )
            found.add(path)
        elif path in capture.directories:
            for member in capture.files:
                if not member.startswith(path + "/"):
                    continue
                rest = PurePosixPath(member[len(path) + 1 :])
                if field == "skills":
                    if rest.name == "SKILL.md" and len(rest.parts) <= 2:
                        found.add(member)
                elif rest.suffix in suffixes:
                    found.add(member)
        elif explicit:
            raise PackageFileError("vendor_component_path_unavailable")
    if (
        field == "skills"
        and not explicit
        and "skills" not in capture.directories
        and "SKILL.md" in capture.files
    ):
        found.add("SKILL.md")
    return tuple(sorted(found))


def _document(
    capture, path: str, kind: str, dialect: str
) -> tuple[dict, tuple[str, ...]]:
    data = capture.read(path)
    require(len(data) <= MAX_DOCUMENT_BYTES, "document_bytes_limit")
    local_id = (
        PurePosixPath(path).parent.name if kind == "skill" else PurePosixPath(path).stem
    )
    blockers = []

    def transform(raw):
        raw = dict(raw)
        if kind == "skill":
            metadata = dict(raw.get("metadata", {}))
            runtime_keys = {
                "context",
                "user_invocable",
                "disable_model_invocation",
                "model",
            }
            if runtime_keys & metadata.keys():
                raise PackageFileError("vendor_skill_metadata_unsupported")
            for source, target in (
                ("disable-model-invocation", "disable_model_invocation"),
                ("user-invocable", "user_invocable"),
            ):
                if source in raw:
                    if dialect == "openai":
                        blockers.append("vendor_codex_skill_flags_unqualified")
                    require(type(raw[source]) is bool, "vendor_manual_flag_invalid")
                    metadata[target] = str(raw.pop(source)).lower()
            raw["metadata"] = metadata
            policy_path = str(PurePosixPath(path).parent / "agents/openai.yaml")
            if policy_path in capture.files:
                import yaml

                from ..schemas import _FrontmatterLoader

                policy_data = capture.read(policy_path)
                require(len(policy_data) <= MAX_DOCUMENT_BYTES, "document_bytes_limit")
                try:
                    policy = yaml.load(
                        policy_data.decode("utf-8"), Loader=_FrontmatterLoader
                    )
                    policy = parse_document(canonical_json(policy).encode())
                except (ValueError, TypeError, UnicodeError, yaml.YAMLError):
                    raise PackageFileError("vendor_skill_policy_invalid") from None
                if policy.keys() - {
                    "interface",
                    "policy",
                    "dependencies",
                } or policy.get("dependencies"):
                    blockers.append("vendor_skill_dependencies_unavailable")
                values = policy.get("policy", {})
                require(isinstance(values, dict), "vendor_skill_policy_invalid")
                if values.keys() - {"allow_implicit_invocation"}:
                    blockers.append("vendor_skill_policy_unsupported")
                if "allow_implicit_invocation" in values:
                    require(
                        type(values["allow_implicit_invocation"]) is bool,
                        "vendor_skill_policy_invalid",
                    )
                    manual = not values["allow_implicit_invocation"]
                    if (
                        "disable_model_invocation" in metadata
                        and metadata["disable_model_invocation"] != str(manual).lower()
                    ):
                        blockers.append("vendor_manual_flags_conflict")
                    metadata["disable_model_invocation"] = str(manual).lower()
                metadata["vendor_policy_digest"] = hashlib.sha256(
                    policy_data
                ).hexdigest()
            return raw
        if kind == "rule":
            if raw.keys() - {"name", "description", "alwaysApply", "globs"}:
                blockers.append("vendor_rule_constraints_unsupported")
            if raw.get("alwaysApply") is not True or raw.get("globs"):
                blockers.append("vendor_rule_activation_unsupported")
            return {
                "name": raw.get("name", local_id),
                "mode": "always" if not blockers else "manual",
            }
        if kind == "agent":
            if (
                raw.get("readonly", False) is not False
                or raw.get("is_background", False) is not False
            ):
                blockers.append("vendor_agent_limits_unsupported")
            if raw.keys() - {
                "name",
                "description",
                "tools",
                "model",
                "readonly",
                "is_background",
                "color",
            }:
                blockers.append("vendor_agent_constraints_unsupported")
            result = {
                key: value
                for key, value in raw.items()
                if key in {"name", "description", "tools", "model"}
            }
            result.setdefault("tools", "inherit")
            if result.get("model") == "inherit":
                result.pop("model")
            return result
        if raw.keys() - {"name", "description"}:
            blockers.append("vendor_command_constraints_unsupported")
        return {
            "name": raw.get("name", local_id),
            "description": raw.get("description", local_id),
        }

    if data.startswith((b"---\n", b"---\r\n")):
        # Root SKILL.md is Cursor's explicit single-skill exception.
        result = frontmatter(
            data,
            kind,
            directory_name=None if path == "SKILL.md" else local_id,
            transform=transform,
        )
    else:
        require(kind == "command", "frontmatter_required")
        raw = {"name": local_id, "description": local_id}
        result = frontmatter(("---\n" + canonical_json(raw) + "\n---\n").encode(), kind)
    if "$ARGUMENTS" in data.decode("utf-8") or "${" in data.decode("utf-8"):
        blockers.append("vendor_instruction_expansion_unsupported")
    return result, tuple(blockers)


def inventory_vendor(
    capture: PackageCapture,
    declaration: dict,
    dialect: str,
    *,
    max_components: int,
    portable_manifest: dict | None = None,
) -> tuple[
    dict[str, ComponentRecord],
    dict,
    dict,
    list[Diagnostic],
    tuple[str, ...],
    dict,
    dict,
]:
    """Normalize tested material, preserving unsupported constraints and exclusions."""
    diagnostics, extra, skill_defs, blocked = [], {}, {}, {}
    global_blockers = []
    require(isinstance(declaration, dict), "vendor_manifest_invalid")
    if declaration.keys() - PRESENTATION - COMPONENTS:
        global_blockers.append("vendor_manifest_constraints_unknown")
    if declaration.get("variables"):
        global_blockers.append("vendor_variables_unconfigured")
    if declaration.get("apps"):
        global_blockers.append("vendor_apps_unavailable")
    identity = validate_manifest(
        {
            "$schema": PLUGIN_SCHEMA,
            **{
                key: value
                for key, value in declaration.items()
                if key
                in PRESENTATION
                - {
                    "$schema",
                    "source",
                    "interface",
                    "logo",
                    "displayName",
                    "category",
                    "tags",
                }
            },
        }
    )

    def add(record):
        if record.component_id in extra:
            record = record.model_copy(
                update={"support": "invalid", "activation_blockers": ("duplicate_id",)}
            )
        extra[record.component_id] = record
        if len(extra) + len(skill_defs) > max_components:
            raise ComponentLimitError("component_count_limit")

    if portable_manifest is None:
        for kind, field in (
            ("skill", "skills"),
            ("command", "commands"),
            ("rule", "rules"),
            ("agent", "agents"),
        ):
            if (
                dialect == "openai"
                and kind in {"command", "rule", "agent"}
                and field not in declaration
            ):
                continue  # OpenAI agents/openai.yaml is presentation, not a preset.
            try:
                paths = _paths(capture, declaration, field)
            except PackageFileError as error:
                diagnostics.append(Diagnostic(code=str(error), path=field))
                if kind == "rule":
                    global_blockers.append("vendor_rule_scope_unknown")
                continue
            for path in paths:
                local = (
                    PurePosixPath(path).parent.name
                    if kind == "skill"
                    else PurePosixPath(path).stem
                )
                try:
                    definition, blockers = _document(capture, path, kind, dialect)
                    local = definition["name"]
                    if kind == "skill":
                        skill_defs[path] = definition
                        if blockers:
                            blocked["skill:" + local] = blockers
                        continue
                    if dialect == "openai":
                        blockers += ("vendor_codex_component_contract_unqualified",)
                    if kind == "agent":
                        if definition["tools"] not in ("inherit", []):
                            blockers += ("tool_mapping_required",)
                        if "model" in definition:
                            blockers += ("model_mapping_required",)
                    status = (
                        "unsupported"
                        if any(b.startswith("vendor_") for b in blockers)
                        else "adapted"
                    )
                except PackageFileError as error:
                    definition, blockers = {}, (str(error),)
                    status = "unsupported" if "unsupported" in str(error) else "invalid"
                add(
                    ComponentRecord(
                        component_id=kind + ":" + local,
                        kind=kind,
                        local_id=local,
                        path=path,
                        definition_json=canonical_json(definition),
                        support=status,
                        activation_blockers=blockers,
                    )
                )
        inventory, edges, variables, native_diags, native_blockers, bodies = (
            inventory_package(
                capture,
                {"extensions": {}},
                max_components=max_components,
                skill_paths=tuple(skill_defs),
                skill_definitions=skill_defs,
            )
        )
    else:
        if declaration.get("mcpServers"):
            global_blockers.append("vendor_openai_mcp_mapping_unqualified")
        inventory, edges, variables, native_diags, native_blockers, bodies = (
            inventory_package(capture, portable_manifest, max_components=max_components)
        )
        if any(
            key in declaration
            for key in ("skills", "commands", "rules", "agents", "mcpServers")
        ):
            diagnostics.append(
                Diagnostic(
                    code="portable_component_locations_canonical", path="plugin.json"
                )
            )
    diagnostics.extend(native_diags)
    global_blockers.extend(native_blockers)

    if portable_manifest is None:
        value = declaration.get(
            "mcpServers", "mcp.json" if dialect == "cursor" else ".mcp.json"
        )
        if isinstance(value, str):
            path = validate_relative_member(value).as_posix()
            require(
                path in capture.files or "mcpServers" not in declaration,
                "vendor_mcp_path_unavailable",
            )
            value = capture.document(path) if path in capture.files else {}
        require(isinstance(value, dict), "vendor_mcp_locations_unsupported")
        servers = value.get("mcpServers", value)
        require(
            isinstance(servers, dict) and len(servers) <= max_components,
            "vendor_mcp_invalid",
        )
        for local, raw in servers.items():
            try:
                require(
                    isinstance(local, str)
                    and len(local) <= 128
                    and ":" not in local
                    and local
                    and not any(c.isspace() for c in local),
                    "vendor_mcp_name_invalid",
                )
                require(isinstance(raw, dict), "vendor_mcp_invalid")
                native = dict(raw)
                native.setdefault("type", "stdio" if "command" in raw else "http")
                definition = validate_mcp(native, capture)
                blockers = (
                    ("transport_unsupported",) if definition["type"] == "sse" else ()
                )
                if "${" in canonical_json(raw):
                    blockers += ("vendor_mcp_variables_unqualified",)
                status = "unsupported" if blockers else "adapted"
            except PackageFileError as error:
                definition, blockers, status = raw, (str(error),), "unsupported"
            add(
                ComponentRecord(
                    component_id="mcp:" + local,
                    kind="mcp",
                    local_id=local,
                    path="mcp.json",
                    definition_json=canonical_json(definition),
                    support=status,
                    activation_blockers=blockers,
                )
            )

    hook_value = declaration.get(
        "hooks", None if portable_manifest is not None else "hooks/hooks.json"
    )
    if isinstance(hook_value, str):
        hook_path = validate_relative_member(hook_value).as_posix()
        if hook_path in capture.files:
            hook_value = capture.document(hook_path)
        elif "hooks" in declaration:
            global_blockers.append("vendor_guard_scope_unknown")
            hook_value = None
        else:
            hook_value = None
    else:
        hook_path = (
            ".codex-plugin/plugin.json"
            if dialect == "openai"
            else ".cursor-plugin/plugin.json"
        )
    if hook_value is not None:
        try:
            hooks, guard = inventory_vendor_hooks(hook_value, hook_path, dialect)
            for row in hooks.values():
                add(row)
            if guard:
                global_blockers.append("vendor_guard_scope_unknown")
        except PackageFileError as error:
            global_blockers.append("vendor_guard_scope_unknown")
            diagnostics.append(Diagnostic(code=str(error), path=hook_path))
    for key, row in extra.items():
        if key in inventory:
            row = row.model_copy(
                update={"support": "invalid", "activation_blockers": ("duplicate_id",)}
            )
        inventory[key] = row
    if len(inventory) > max_components:
        raise ComponentLimitError("component_count_limit")
    for key, row in list(inventory.items()):
        blockers = tuple(
            sorted({*row.activation_blockers, *blocked.get(key, ()), *global_blockers})
        )
        status = "unsupported" if blocked.get(key) else row.support
        if row.kind == "skill" and status == "supported":
            status = "adapted"
        inventory[key] = row.model_copy(
            update={"support": status, "activation_blockers": blockers}
        )
    return (
        inventory,
        edges,
        variables,
        diagnostics,
        tuple(sorted(set(global_blockers))),
        bodies,
        identity,
    )
