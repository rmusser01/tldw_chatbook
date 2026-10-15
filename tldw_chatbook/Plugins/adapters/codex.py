"""Version-pinned Codex interpretation, without vendor permission grants."""


def overlay_for_openai(inline: dict | None, compatibility: dict | None) -> dict:
    """Select the complete inline overlay when present, including an empty one."""
    return dict(inline if inline is not None else (compatibility or {}))


from pathlib import Path, PurePosixPath

import yaml

from ..models import ComponentRecord, Diagnostic, PackageInspection
from ..package_files import (
    PackageFileError,
    canonical_json,
    parse_document,
    validate_relative_member,
)
from ..schemas import _FrontmatterLoader, frontmatter, require, validate_mcp

ADAPTER_VERSION = "chatbook-codex/1"
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


def inspect_codex(root: Path, catalog_overlay: dict | None = None) -> PackageInspection:
    """Inspect captured Codex data through the common package owner."""
    from ..inspection import inspect_package

    return inspect_package(root, dialect="openai", catalog_overlay=catalog_overlay)


def component_paths(capture, value, *, suffix: str, default: str, explicit=False):
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


def normalize_instruction(capture, path, kind):
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


def normalize_variables(raw):
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

    return validate_extension({"version": 1, "variables": result})["variables"]


def vendor_inventory(capture, overlay, *, dialect, portable_root=False):
    """Normalize vendor records for the existing native dependency inventory."""
    from .vendor_hooks import inventory_hooks

    records, diagnostics, blockers = [], [], []
    if overlay.keys() - VENDOR_FIELDS:
        blockers.append("vendor_fields_unsupported")
        diagnostics.append(Diagnostic(code="vendor_fields_unsupported"))
    if "policy" in overlay:
        diagnostics.append(Diagnostic(code="foreign_policy_no_grant"))

    def add(kind, path, definition, local_id=None, reasons=(), support="adapted"):
        local_id = local_id or definition.get("name") or PurePosixPath(path).stem
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
            paths = component_paths(
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
                    definition, reasons = normalize_instruction(capture, path, kind)
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
                        support="unsupported",
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
                for name, raw in servers.items():
                    try:
                        require(isinstance(raw, dict))
                        require(
                            isinstance(name, str)
                            and bool(name)
                            and len(name) <= 128
                            and ":" not in name
                            and not any(c.isspace() for c in name)
                        )
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
                            name,
                            ("definition_unsupported",),
                            "unsupported",
                        )
            except (PackageFileError, TypeError, ValueError):
                blockers.append("mcp_definition_unsupported")
                diagnostics.append(
                    Diagnostic(code="mcp_definition_unsupported", path=path)
                )
    if "apps" in overlay:
        diagnostics.append(Diagnostic(code="foreign_apps_unavailable"))
    try:
        variables = normalize_variables(overlay.get("variables"))
    except (PackageFileError, TypeError, ValueError):
        variables = {}
        blockers.append("variable_schema_unsupported")
    hook_records, hook_diagnostics, hook_blockers = inventory_hooks(
        capture, overlay, dialect
    )
    records.extend(hook_records)
    diagnostics.extend(hook_diagnostics)
    blockers.extend(hook_blockers)
    require(len(records) <= 512, "component_count_limit")
    return records, variables, diagnostics, tuple(sorted(set(blockers)))
