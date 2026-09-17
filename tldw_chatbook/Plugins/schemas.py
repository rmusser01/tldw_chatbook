"""Locally pinned native validation rules (no schema downloads or execution).

Agent Plugins 1.0.0 and Agent Skills references are recorded with test fixtures.
Native extension and hook envelopes follow ADR-162/163. Validation errors expose
codes only; package bodies and configuration values are never diagnostics.
"""

import math
import re
from urllib.parse import urlsplit

import yaml

from tldw_chatbook.Utils.input_validation import validate_env_var_reference

from . import package_files
from .package_files import PackageFileError, canonical_json, validate_relative_member

NAMESPACE = "io.github.rmusser01.chatbook"
PLUGIN_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,128}\Z")
COMPONENT_PATTERN = re.compile(r"(?:skill|mcp|command|rule|agent|hook):[^:\s]{1,128}\Z")
RESERVED_VARIABLES = {"PLUGIN_ROOT", "PLUGIN_DATA"}
EFFECTS = {
    "SessionStart": {"context", "deny"},
    "UserPromptSubmit": {"deny", "context"},
    "PreToolUse": {"updated_input", "deny", "context"},
    "ApprovalRequested": set(),
    "PostToolUse": {"context"},
    "PostToolUseFailure": {"context"},
    "SubagentStart": {"deny", "child_limits", "context"},
    "SubagentStop": {"context"},
    "PreCompact": {"context"},
    "PostCompact": {"context"},
    "Stop": {"continuation", "stop_continuations"},
    "Interrupt": set(),
    "SessionEnd": set(),
}
NON_REQUIRED_EVENTS = {"Stop", "Interrupt", "SessionEnd", "ApprovalRequested"}


def require(condition: bool, code: str = "definition_invalid") -> None:
    if not condition:
        raise PackageFileError(code)


def closed(value: object, allowed: set[str], required: set[str] = frozenset()) -> dict:
    require(isinstance(value, dict))
    require(not (value.keys() - allowed) and required <= value.keys())
    return value


def strings(value: object, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(
            isinstance(item, str) and (bool(item) or not nonempty) for item in value
        )
    )


def validate_manifest(value: dict) -> dict:
    require(value.get("$schema") == PLUGIN_SCHEMA, "manifest_schema_unsupported")
    name = value.get("name")
    require(
        isinstance(name, str)
        and len(name) <= 64
        and re.fullmatch(r"(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", name)
        is not None,
        "manifest_name_invalid",
    )
    for field in ("version", "description", "homepage", "repository", "license"):
        require(
            field not in value or isinstance(value[field], str),
            "manifest_field_invalid",
        )
    if "author" in value:
        author = closed(value["author"], {"name", "email", "url"})
        require(
            all(isinstance(item, str) for item in author.values()),
            "manifest_author_invalid",
        )
    require(
        "keywords" not in value or strings(value["keywords"]),
        "manifest_keywords_invalid",
    )
    return value


def validate_extension(value: object) -> dict:
    value = closed(
        value,
        {"version", "commands", "rules", "agents", "hooks", "requires", "variables"},
        {"version"},
    )
    require(
        type(value["version"]) is int and value["version"] == 1,
        "extension_version_unsupported",
    )
    for field in ("commands", "rules", "agents"):
        require(strings(value.get(field, [])))
        for path in value.get(field, []):
            validate_relative_member(path)
    if "hooks" in value:
        validate_relative_member(value["hooks"])
    requires = value.get("requires", {})
    require(isinstance(requires, dict))
    for key, dependencies in requires.items():
        require(COMPONENT_PATTERN.fullmatch(key) is not None and strings(dependencies))
        require(
            all(COMPONENT_PATTERN.fullmatch(dep) is not None for dep in dependencies)
        )
    variables = value.get("variables", {})
    require(isinstance(variables, dict))
    for name, declaration in variables.items():
        require(
            validate_env_var_reference(name)
            and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", name) is not None
            and name not in RESERVED_VARIABLES
        )
        declaration = closed(
            declaration,
            {"type", "required", "secret", "default"},
            {"type", "required", "secret"},
        )
        require(declaration["type"] in ("string", "integer", "boolean"))
        require(
            type(declaration["required"]) is bool
            and type(declaration["secret"]) is bool
        )
        if "default" in declaration:
            require(not declaration["secret"])
            require(
                type(declaration["default"])
                is {"string": str, "integer": int, "boolean": bool}[declaration["type"]]
            )
    return value


class _FrontmatterLoader(yaml.SafeLoader):
    """Reject aliases and duplicate fields instead of silently losing constraints."""

    def compose_node(self, parent, index):
        require(not self.check_event(yaml.AliasEvent), "frontmatter_alias_unsupported")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            require(
                isinstance(key, str) and key not in result,
                "frontmatter_duplicate_or_invalid_key",
            )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def frontmatter(data: bytes, kind: str, *, directory_name: str | None = None) -> dict:
    require(len(data) <= package_files.MAX_DOCUMENT_BYTES, "document_bytes_limit")
    try:
        text = data.decode("utf-8")
        lines = text.splitlines()
        require(bool(lines) and lines[0] == "---", "frontmatter_required")
        end = next(i for i, line in enumerate(lines[1:], 1) if line == "---")
        value = yaml.load("\n".join(lines[1:end]), Loader=_FrontmatterLoader)
        require(isinstance(value, dict))
        # Also enforces native JSON types/depth and forbids YAML objects/nonfinite values.
        package_files.parse_document(canonical_json(value).encode())
    except (
        yaml.YAMLError,
        UnicodeError,
        StopIteration,
        RecursionError,
        TypeError,
        ValueError,
    ):
        raise PackageFileError("frontmatter_invalid") from None
    common = {"name", "description"}
    allowed = {
        "skill": common | {"license", "compatibility", "metadata", "allowed-tools"},
        "command": common | {"arguments"},
        "rule": {"name", "mode"},
        "agent": common | {"tools", "model"},
    }
    required = {"name", "mode"} if kind == "rule" else common
    if kind == "agent":
        required |= {"tools"}
    closed(value, allowed[kind], required)
    name = value["name"]
    require(isinstance(name, str) and bool(name) and len(name) <= 64)
    if kind == "skill":
        require(
            name == directory_name
            and name == name.lower()
            and not name.startswith("-")
            and not name.endswith("-")
            and "--" not in name
            and all(c.isalnum() or c == "-" for c in name)
        )
        for field, limit in (("description", 1024), ("compatibility", 500)):
            require(
                field not in value
                or isinstance(value[field], str)
                and 0 < len(value[field]) <= limit
            )
        for field in ("license", "allowed-tools"):
            require(field not in value or isinstance(value[field], str))
        if "metadata" in value:
            require(
                isinstance(value["metadata"], dict)
                and all(
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in value["metadata"].items()
                )
            )
    else:
        require(ID_PATTERN.fullmatch(name) is not None)
        require(
            "description" not in value
            or isinstance(value["description"], str)
            and bool(value["description"])
        )
    if kind == "rule":
        require(value["mode"] in ("always", "manual"))
    elif kind == "command":
        args = value.get("arguments", [])
        require(
            strings(args)
            and len(set(args)) == len(args)
            and all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", arg) for arg in args)
        )
    elif kind == "agent":
        tools = value["tools"]
        require(
            tools == "inherit"
            or strings(tools, nonempty=False)
            and all(re.fullmatch(r"[\w.:-]+", item) for item in tools)
        )
        require(
            "model" not in value
            or isinstance(value["model"], str)
            and bool(value["model"])
        )
    return value


def validate_mcp(value: object, capture) -> dict:
    require(isinstance(value, dict))
    transport = value.get("type")
    if transport == "stdio":
        closed(value, {"type", "command", "args", "env", "cwd"}, {"type", "command"})
        command = value["command"]
        require(isinstance(command, str) and bool(command) and "\x00" not in command)
        if command.startswith("./"):
            capture.read(command)
        else:
            require(
                not any(c in command for c in "/\\$:")
                and not any(c.isspace() for c in command)
            )
        require(strings(value.get("args", [])))
        env = value.get("env", {})
        require(
            isinstance(env, dict)
            and not (env.keys() & RESERVED_VARIABLES)
            and all(
                validate_env_var_reference(k) and isinstance(v, str) and "\x00" not in v
                for k, v in env.items()
            )
        )
        if "cwd" in value:
            cwd = value["cwd"]
            require(isinstance(cwd, str))
            if cwd.startswith("${PLUGIN_DATA}"):
                suffix = cwd[len("${PLUGIN_DATA}") :]
                require(not suffix or suffix.startswith("/"))
                if suffix:
                    validate_relative_member(suffix[1:])
            else:
                require(
                    cwd.startswith(("./", "${PLUGIN_ROOT}/")) or cwd == "${PLUGIN_ROOT}"
                )
                relative = (
                    cwd.removeprefix("${PLUGIN_ROOT}").removeprefix("/")
                    if cwd.startswith("${PLUGIN_ROOT}")
                    else cwd
                )
                if relative not in ("", "./"):
                    relative = validate_relative_member(relative).as_posix()
                    require(relative in capture.directories)
    else:
        closed(value, {"type", "url", "headers"}, {"type", "url"})
        require(transport in ("streamable-http", "sse"))
        require(isinstance(value["url"], str))
        url = urlsplit(value["url"])
        require(
            url.scheme in ("http", "https")
            and bool(url.hostname)
            and "@" not in url.netloc
            and "#" not in value["url"]
        )
        require(
            url.scheme == "https" or url.hostname in ("localhost", "127.0.0.1", "::1")
        )
        headers = value.get("headers", {})
        require(
            isinstance(headers, dict)
            and all(isinstance(k, str) for k in headers)
            and len({k.lower() for k in headers}) == len(headers)
            and all(
                isinstance(v, str)
                and re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", k)
                and (not v or v[0] not in " \t" and v[-1] not in " \t")
                and all(
                    char == "\t" or (ord(char) >= 32 and ord(char) != 127) for char in v
                )
                for k, v in headers.items()
            )
        )
    return value


def validate_hook(value: object, capture) -> dict:
    common = {
        "id",
        "event",
        "type",
        "effects",
        "required",
        "require_context",
        "match",
        "timeout_seconds",
    }
    require(isinstance(value, dict))
    command = value.get("type") == "command"
    specific = {"argv", "env", "cwd"} if command else {"server", "tool", "input"}
    value = closed(value, common | specific, {"id", "event", "type", "effects"})
    require(
        isinstance(value["id"], str) and ID_PATTERN.fullmatch(value["id"]) is not None
    )
    require(value["type"] in ("command", "mcp_tool"))
    event = value["event"]
    require(isinstance(event, str) and event in EFFECTS)
    effects = value["effects"]
    require(
        strings(effects)
        and len(set(effects)) == len(effects)
        and set(effects) <= EFFECTS[event]
    )
    for field in ("required", "require_context"):
        require(type(value.get(field, False)) is bool)
    require(not value.get("required") or event not in NON_REQUIRED_EVENTS)
    require(not value.get("require_context") or "context" in effects)
    timeout = value.get("timeout_seconds", 10)
    require(
        type(timeout) in (int, float) and 0 < timeout <= 60 and math.isfinite(timeout)
    )
    if "match" in value:
        match = closed(value["match"], {"tool_id", "provider", "operation", "reason"})
        require(
            bool(match)
            and all(strings(patterns, nonempty=True) for patterns in match.values())
        )
        require(
            sum(len(patterns) for patterns in match.values()) <= 32
            and all(len(p) <= 256 for patterns in match.values() for p in patterns)
        )
        # Only tool events have qualified tool fields; reason applies to lifecycle events.
        tool_events = {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "ApprovalRequested",
        }
        require(
            event in tool_events
            or not (match.keys() & {"tool_id", "provider", "operation"})
        )
    if command:
        require(strings(value.get("argv"), nonempty=True))
        require(all("\x00" not in arg for arg in value["argv"]))
        executable = value["argv"][0]
        if executable.startswith("./"):
            capture.read(executable)
        elif not executable.startswith(("/", "${PLUGIN_ROOT}/")):
            require("/" not in executable and "\\" not in executable)
        for arg in value["argv"]:
            if arg.startswith("${PLUGIN_ROOT}/"):
                capture.read(arg[len("${PLUGIN_ROOT}/") :])
        env = value.get("env", {})
        require(
            isinstance(env, dict)
            and not (env.keys() & RESERVED_VARIABLES)
            and all(
                validate_env_var_reference(k) and isinstance(v, str)
                for k, v in env.items()
            )
        )
        if "cwd" in value and value["cwd"] not in (".", "./"):
            require(
                validate_relative_member(value["cwd"]).as_posix() in capture.directories
            )
    else:
        require(
            all(
                isinstance(value.get(k), str) and bool(value[k])
                for k in ("server", "tool")
            )
        )
        require(isinstance(value.get("input", {}), dict))
    return value
