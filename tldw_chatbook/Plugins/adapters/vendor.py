"""Bounded vendor inventory reuses native records, metadata and authority contracts."""

import hashlib

import yaml
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
    _FrontmatterLoader,
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
            require(
                isinstance(raw.get("metadata", {}), dict),
                "vendor_skill_metadata_invalid",
            )
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
    _recipe: str = "integrated_2",
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
    if _recipe != "dev_20261001":
        records, variables, diagnostics, blockers = _foreign_records(
            capture, declaration, dialect=dialect,
            portable_root=portable_manifest is not None, _recipe=_recipe,
        )
        manifest = portable_manifest or {"name": declaration["name"]}
        inventory, edges, variables, native_diags, blockers, bodies = inventory_package(
            capture, manifest, max_components=max_components,
            vendor_records=records, vendor_variables=variables,
            vendor_blockers=blockers, discover_portable=portable_manifest is not None,
            _recipe=_recipe,
        )
        diagnostics.extend(native_diags)
        return inventory, edges, variables, diagnostics, blockers, bodies, manifest
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
                _recipe="dev_20261001",
            )
        )
    else:
        if declaration.get("mcpServers"):
            global_blockers.append("vendor_openai_mcp_mapping_unqualified")
        inventory, edges, variables, native_diags, native_blockers, bodies = (
            inventory_package(capture, portable_manifest, max_components=max_components, _recipe="dev_20261001")
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


VENDOR_FIELDS = frozenset(
    {
        "name",
        "displayName",
        "version",
        "description",
        "author",
        "homepage",
        "repository",
        "license",
        "keywords",
        "logo",
        "category",
        "tags",
        "skills",
        "rules",
        "agents",
        "commands",
        "hooks",
        "mcpServers",
        "variables",
        "apps",
        "interface",
        "source",
        "policy",
        "products",
    }
)


def _branch_paths(capture, value, *, suffix: str, default: str, explicit=False):
    """Resolve only captured, contained paths; an explicit empty list stays empty."""
    paths = value if explicit else default
    paths = [paths] if isinstance(paths, str) else paths
    require(isinstance(paths, list) and all(isinstance(p, str) for p in paths))
    found = set()
    for path in paths:
        member = validate_relative_member(path).as_posix()
        if member in capture.files:
            require(member.endswith(suffix), "component_path_type_unsupported")
            found.add(member)
        elif member in capture.directories:
            found.update(
                p
                for p in capture.files
                if p.startswith(member + "/") and p.endswith(suffix)
            )
        elif explicit:
            raise PackageFileError("component_path_unavailable")
    return sorted(found)


def _metadata(capture, path):
    lines = capture.read(path).decode("utf-8").splitlines()
    require(bool(lines) and lines[0] == "---", "frontmatter_required")
    try:
        end = lines.index("---", 1)
        value = yaml.load("\n".join(lines[1:end]), Loader=_FrontmatterLoader)
        require(isinstance(value, dict))
        return parse_document(canonical_json(value).encode())
    except (ValueError, TypeError, yaml.YAMLError):
        raise PackageFileError("frontmatter_invalid") from None


def _branch_instruction(capture, path, kind):
    """Map supported metadata to native constraints without changing body bytes."""
    raw = _metadata(capture, path)
    value = dict(raw)
    if kind == "skill":
        metadata = dict(value.get("metadata", {}))
        for foreign, native in (
            ("disable-model-invocation", "disable_model_invocation"),
            ("user-invocable", "user_invocable"),
        ):
            if foreign in value:
                flag = value.pop(foreign)
                require(type(flag) is bool, "manual_declaration_invalid")
                mapped = str(flag).lower()
                require(
                    native not in metadata or metadata[native] == mapped,
                    "manual_declaration_conflict",
                )
                metadata[native] = mapped
        for key in ("context", "model"):
            if key in value:
                mapped = value.pop(key)
                require(
                    key not in metadata or metadata[key] == mapped, "metadata_conflict"
                )
                metadata[key] = mapped
        require(
            metadata.get("context", "inline") in {"inline", "fork"},
            "context_unsupported",
        )
        require(
            not (
                metadata.get("disable_model_invocation") == "true"
                and metadata.get("user_invocable") == "false"
            ),
            "manual_declaration_conflict",
        )
        if metadata:
            value["metadata"] = metadata
    elif kind == "rule":
        require(
            set(value) <= {"name", "description", "globs", "alwaysApply", "mode"},
            "rule_constraints_unsupported",
        )
        require(type(value.get("alwaysApply", False)) is bool)
        require(
            not (value.get("globs") and value.get("alwaysApply")), "rule_scope_conflict"
        )
        mode = "always" if value.get("alwaysApply") else "manual"
        require("mode" not in value or value["mode"] == mode, "rule_scope_conflict")
        value = {"name": value.get("name", PurePosixPath(path).stem), "mode": mode}
    elif kind == "agent":
        # An omitted foreign tool contract is not proof of inheritance.
        require("tools" in value, "agent_tools_unspecified")
    normalized = frontmatter(
        ("---\n" + yaml.safe_dump(value) + "---\n").encode(),
        kind,
        directory_name=value.get("name") if kind == "skill" else None,
    )
    blockers = []
    if normalized.get("allowed-tools"):
        blockers.append("skill_tools_mapping_required")
    if (
        kind == "agent"
        and isinstance(normalized.get("tools"), list)
        and normalized["tools"]
    ):
        blockers.append("tool_mapping_required")
    if "model" in normalized or normalized.get("metadata", {}).get("model"):
        blockers.append("model_mapping_required")
    return normalized, tuple(blockers)


def _foreign_variables(raw, *, _recipe="integrated_2"):
    """Accept the scalar JSON Schema subset represented by native declarations."""
    if raw is None:
        return {}
    require(
        isinstance(raw, dict) and set(raw) <= {"type", "properties", "required"},
        "variable_schema_unsupported",
    )
    require(
        raw.get("type") == "object" and isinstance(raw.get("properties"), dict),
        "variable_schema_unsupported",
    )
    required = raw.get("required", [])
    require(
        isinstance(required, list) and all(isinstance(name, str) for name in required)
    )
    require(set(required) <= raw["properties"].keys())
    result = {}
    for name, declaration in raw["properties"].items():
        require(
            isinstance(declaration, dict)
            and set(declaration) <= {"type", "default", "description", "title"},
            "variable_schema_unsupported",
        )
        result[name] = {
            "type": declaration.get("type"),
            "required": name in required,
            "secret": False,
        }
        if "default" in declaration:
            result[name]["default"] = declaration["default"]
    from ..schemas import validate_extension

    return validate_extension({"version": 1, "variables": result}, _recipe=_recipe)["variables"]


def _foreign_records(capture, overlay, *, dialect, portable_root=False, _recipe="integrated_2"):
    """Normalize vendor records for the existing native dependency inventory."""
    from .vendor_hooks import _inventory_qualified_hooks

    records, diagnostics, blockers = [], [], []
    if overlay.keys() - (VENDOR_FIELDS | (PRESENTATION if _recipe == "integrated_2" else set())):
        blockers.append("vendor_fields_unsupported")
        diagnostics.append(Diagnostic(code="vendor_fields_unsupported"))
    if "policy" in overlay:
        diagnostics.append(Diagnostic(code="foreign_policy_no_grant"))
        if _recipe == "integrated_2" and overlay["policy"]:
            blockers.append("vendor_manifest_constraints_unknown")
    if _recipe == "integrated_2" and portable_root and overlay.get("mcpServers"):
        blockers.append("vendor_openai_mcp_mapping_unqualified")

    def add(kind, path, definition, local_id=None, reasons=(), support="adapted"):
        if local_id is None:
            name = definition.get("name") if isinstance(definition, dict) else None
            local_id = name or PurePosixPath(path).stem
        records.append(
            ComponentRecord(
                component_id=f"{kind}:{local_id}",
                kind=kind,
                local_id=local_id,
                path=path,
                definition_json=canonical_json(definition),
                support=support,
                activation_blockers=tuple(reasons),
            )
        )

    for kind, field, suffix in (
        ("skill", "skills", "SKILL.md"),
        ("rule", "rules", ".mdc"),
        ("agent", "agents", ".md"),
        ("command", "commands", ".md"),
    ):
        if portable_root and kind == "skill":
            continue
        if dialect == "openai" and kind != "skill" and field not in overlay:
            continue
        try:
            if _recipe == "integrated_2":
                paths = _paths(capture, overlay, field)
            else:
                paths = _branch_paths(
                    capture,
                    overlay.get(field),
                    suffix=suffix,
                    default=field,
                    explicit=field in overlay,
                )
                if (
                    kind == "skill"
                    and dialect == "cursor"
                    and "skills" not in overlay
                    and "skills" not in capture.directories
                    and "SKILL.md" in capture.files
                ):
                    paths = ["SKILL.md"]
            for path in paths:
                try:
                    definition, reasons = (
                        _integrated_instruction(capture, path, kind, dialect)
                        if _recipe == "integrated_2"
                        else _branch_instruction(capture, path, kind)
                    )
                    add(kind, path, definition, reasons=reasons)
                except (PackageFileError, UnicodeError, TypeError, ValueError) as exc:
                    local_id = (
                        PurePosixPath(path).parent.name
                        if kind == "skill"
                        else PurePosixPath(path).stem
                    )
                    add(
                        kind,
                        path,
                        {},
                        local_id=local_id,
                        reasons=(
                            (
                                str(exc)
                                if isinstance(exc, PackageFileError)
                                else "definition_invalid"
                            ),
                        ),
                        support=("invalid" if _recipe == "integrated_2" and str(exc) == "vendor_skill_metadata_invalid" else "unsupported"),
                    )
        except PackageFileError as exc:
            blockers.append(str(exc))
            diagnostics.append(Diagnostic(code=str(exc), path=field))

    if not portable_root:
        value = overlay.get(
            "mcpServers", "./mcp.json" if dialect == "cursor" else "./.mcp.json"
        )
        values = value if isinstance(value, list) else [value]
        for index, source in enumerate(values):
            path = (
                source
                if isinstance(source, str)
                else (
                    ".cursor-plugin/plugin.json"
                    if dialect == "cursor"
                    else ".codex-plugin/plugin.json"
                )
            )
            try:
                if isinstance(source, str):
                    member = validate_relative_member(source).as_posix()
                    if member not in capture.files and "mcpServers" not in overlay:
                        continue
                    document = capture.document(source)
                else:
                    document = source
                require(isinstance(document, dict))
                servers = document.get("mcpServers", document)
                require(isinstance(servers, dict))
                for entry_index, name in enumerate(sorted(servers)):
                    raw = servers[name]
                    valid_name = (
                        bool(name)
                        and len(name) <= 128
                        and ":" not in name
                        and not any(c.isspace() or ord(c) < 32 for c in name)
                    )
                    # Colons are reserved from valid source IDs. Sorted positions
                    # also survive canonical catalog replay without collisions.
                    local_id = name if valid_name else f"invalid:{index}:{entry_index}"
                    try:
                        require(valid_name)
                        require(isinstance(raw, dict))
                        definition = dict(raw)
                        if "oauth_resource" in definition:
                            definition.pop("oauth_resource")
                            diagnostics.append(
                                Diagnostic(
                                    code="foreign_oauth_metadata_no_grant",
                                    path=path,
                                    component_id=f"mcp:{name}",
                                )
                            )
                        if "type" not in definition:
                            definition["type"] = (
                                "http" if "url" in definition else "stdio"
                            )
                        if dialect == "cursor" and definition.get("type") == "stdio":

                            def variable_text(text):
                                require(isinstance(text, str))
                                require(
                                    "${PLUGIN_ROOT}" not in text
                                    and "${PLUGIN_DATA}" not in text,
                                    "cursor_portable_variable_unsupported",
                                )
                                return text.replace(
                                    "${CURSOR_PLUGIN_ROOT}", "${PLUGIN_ROOT}"
                                ).replace("${CLAUDE_PLUGIN_ROOT}", "${PLUGIN_ROOT}")

                            require("command" in definition)
                            command = variable_text(definition["command"])
                            if command.startswith("${PLUGIN_ROOT}/"):
                                command = "./" + command[len("${PLUGIN_ROOT}/") :]
                            definition["command"] = command
                            if "args" in definition:
                                require(isinstance(definition["args"], list))
                                definition["args"] = [
                                    variable_text(arg) for arg in definition["args"]
                                ]
                            if "env" in definition:
                                require(isinstance(definition["env"], dict))
                                definition["env"] = {
                                    key: variable_text(item)
                                    for key, item in definition["env"].items()
                                }
                            if "cwd" in definition:
                                definition["cwd"] = variable_text(definition["cwd"])
                        if definition.get("type") == "http":
                            definition["type"] = "streamable-http"
                        definition = validate_mcp(definition, capture)
                        require(definition["type"] != "sse", "transport_unsupported")
                        add("mcp", path, definition, name)
                    except (PackageFileError, TypeError, ValueError):
                        add(
                            "mcp",
                            path,
                            raw,
                            local_id,
                            ("definition_unsupported",),
                            "unsupported",
                        )
                        if not valid_name:
                            diagnostics.append(
                                Diagnostic(
                                    code="mcp_name_invalid",
                                    path=path,
                                    component_id=f"mcp:{local_id}",
                                )
                            )
            except (PackageFileError, TypeError, ValueError):
                blockers.append("mcp_definition_unsupported")
                diagnostics.append(
                    Diagnostic(code="mcp_definition_unsupported", path=path)
                )
    if "apps" in overlay:
        diagnostics.append(Diagnostic(code="foreign_apps_unavailable"))
    try:
        variables = _foreign_variables(overlay.get("variables"), _recipe=_recipe)
    except (PackageFileError, TypeError, ValueError):
        variables = {}
        blockers.append("variable_schema_unsupported")
    hook_records, hook_diagnostics, hook_blockers = _inventory_qualified_hooks(
        capture, overlay, dialect, _recipe=_recipe
    )
    records.extend(hook_records)
    diagnostics.extend(hook_diagnostics)
    blockers.extend(hook_blockers)
    require(len(records) <= 512, "component_count_limit")
    return records, variables, diagnostics, tuple(sorted(set(blockers)))


def _integrated_instruction(capture, path, kind, dialect):
    """Map represented constraints, retaining current policy and scope refusals."""
    if kind == "command" and not capture.read(path).startswith((b"---\n", b"---\r\n")):
        return _document(capture, path, kind, dialect)
    raw = _metadata(capture, path)
    text = capture.read(path).decode("utf-8")
    require("$ARGUMENTS" not in text and "${" not in text, "vendor_instruction_expansion_unsupported")
    if kind == "skill":
        require(isinstance(raw.get("metadata", {}), dict), "vendor_skill_metadata_invalid")
    if kind == "agent":
        require(raw.get("readonly", False) is False and raw.get("is_background", False) is False,
                "vendor_agent_limits_unsupported")
        require("tools" in raw, "agent_tools_unspecified")
        require(not (raw.keys() - {"name", "description", "tools", "model", "readonly", "is_background", "color"}),
                "vendor_agent_constraints_unsupported")
        value = {k: v for k, v in raw.items() if k in {"name", "description", "tools", "model"}}
        if value.get("model") == "inherit":
            value.pop("model")
        definition = frontmatter(("---\n" + yaml.safe_dump(value) + "---\n").encode(), kind)
        blockers = []
        if isinstance(definition["tools"], list) and definition["tools"]:
            blockers.append("tool_mapping_required")
        if "model" in definition:
            blockers.append("model_mapping_required")
        return definition, tuple(blockers)
    if kind == "rule":
        require(not raw.get("globs"), "vendor_rule_activation_unsupported")
    definition, reasons = _branch_instruction(capture, path, kind)
    if kind == "skill":
        policy_path = str(PurePosixPath(path).parent / "agents/openai.yaml")
        if policy_path in capture.files:
            policy_data = capture.read(policy_path)
            require(len(policy_data) <= MAX_DOCUMENT_BYTES, "document_bytes_limit")
            try:
                policy = yaml.load(policy_data.decode("utf-8"), Loader=_FrontmatterLoader)
                policy = parse_document(canonical_json(policy).encode())
            except (ValueError, TypeError, UnicodeError, yaml.YAMLError):
                raise PackageFileError("vendor_skill_policy_invalid") from None
            require(isinstance(policy, dict), "vendor_skill_policy_invalid")
            blockers = list(reasons)
            if policy.keys() - {"interface", "policy", "dependencies"} or policy.get("dependencies"):
                blockers.append("vendor_skill_dependencies_unavailable")
            values = policy.get("policy", {})
            require(isinstance(values, dict), "vendor_skill_policy_invalid")
            if values.keys() - {"allow_implicit_invocation"}:
                blockers.append("vendor_skill_policy_unsupported")
            metadata = dict(definition.get("metadata", {}))
            if "allow_implicit_invocation" in values:
                require(type(values["allow_implicit_invocation"]) is bool, "vendor_skill_policy_invalid")
                manual = str(not values["allow_implicit_invocation"]).lower()
                require("disable_model_invocation" not in metadata or metadata["disable_model_invocation"] == manual,
                        "vendor_manual_flags_conflict")
                metadata["disable_model_invocation"] = manual
            metadata["vendor_policy_digest"] = hashlib.sha256(policy_data).hexdigest()
            definition["metadata"] = metadata
            reasons = tuple(blockers)
    return definition, reasons
