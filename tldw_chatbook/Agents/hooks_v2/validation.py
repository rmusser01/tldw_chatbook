"""Closed validation for explicit v2 hooks.

Raw command/text results must enter through :func:`decode_result`. Validation
of an already-decoded object cannot recover duplicate JSON keys or transport
framing provenance.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from tldw_chatbook.Utils.path_validation import validate_existing_absolute_directory

from .models import ContextBlock, HookEvent, HookHandler, HookResult

EFFECTS = {
    "SessionStart": frozenset({"context", "deny"}),
    "UserPromptSubmit": frozenset({"deny", "context"}),
    "PreToolUse": frozenset({"updated_input", "deny", "context"}),
    "ApprovalRequested": frozenset(),
    "PostToolUse": frozenset({"context"}),
    "PostToolUseFailure": frozenset({"context"}),
    "SubagentStart": frozenset({"deny", "child_limits", "context"}),
    "SubagentStop": frozenset({"context"}),
    "PreCompact": frozenset({"context"}),
    "PostCompact": frozenset({"context"}),
    "Stop": frozenset({"continuation", "stop_continuations"}),
    "Interrupt": frozenset(),
    "SessionEnd": frozenset(),
}
NON_REQUIRED_EVENTS = frozenset(
    {"Stop", "Interrupt", "SessionEnd", "ApprovalRequested"}
)
MATCH_KEYS = {
    "SessionStart": frozenset({"reason"}),
    "PreToolUse": frozenset({"tool_id", "provider", "operation"}),
    "PostToolUse": frozenset({"tool_id", "provider", "operation"}),
    "PostToolUseFailure": frozenset({"tool_id", "provider", "operation", "reason"}),
}
ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
DECLARED_RE = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
TEMPLATE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.]*)\}")
RESULT_BYTES = 16 * 1024
INPUT_BYTES = 1024 * 1024
MAX_NODES = 16_384
MAX_DEPTH = 32
MAX_HANDLERS = 256
CAPS = frozenset(
    {
        "max_steps",
        "max_model_turns",
        "max_wall_seconds",
        "max_subagents",
        "max_subagent_result_chars",
        "max_tool_result_chars",
        "max_total_tokens",
        "max_tool_call_seconds",
        "max_model_retries",
    }
)
POSITIVE_CAPS = frozenset(
    {
        "max_steps",
        "max_model_turns",
        "max_wall_seconds",
        "max_subagent_result_chars",
    }
)
DURATION_CAPS = frozenset({"max_wall_seconds", "max_tool_call_seconds"})


def _fail(message: str) -> None:
    raise ValueError(message)


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _object(
    value: object,
    *,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str] = frozenset(),
) -> dict:
    if not isinstance(value, dict) or any(type(k) is not str for k in value):
        _fail("expected JSON object")
    if not required <= value.keys() or value.keys() - allowed:
        _fail("missing or unsupported field")
    return value


def _json_tree(
    value: object,
    *,
    max_bytes: int = INPUT_BYTES,
    string_transform: Callable[[str], object] | None = None,
    allow_frozen: bool = False,
) -> Any:
    """Copy JSON with incremental byte/node/depth checks before allocation.

    The optional string transform is used only for typed template values. Host
    projections it returns are copied without re-expansion, preserving one pass.
    """
    count = 0
    byte_count = 0
    seen: set[int] = set()

    def charge(size: int) -> None:
        nonlocal byte_count
        byte_count += size
        if byte_count > max_bytes:
            _fail("JSON byte limit exceeded")

    def charge_string(value: str) -> None:
        if len(value) > max_bytes:
            _fail("JSON byte limit exceeded")
        charge(2)  # Quotes.
        for offset in range(0, len(value), 1024):
            chunk = value[offset : offset + 1024]
            try:
                escaped = json.dumps(chunk, ensure_ascii=False)[1:-1]
                charge(len(escaped.encode("utf-8")))
            except (UnicodeError, ValueError) as exc:
                raise ValueError("invalid UTF-8 JSON") from exc

    def copy(node: object, depth: int, expand_strings: bool) -> Any:
        nonlocal count
        if depth > MAX_DEPTH:
            _fail("JSON depth limit exceeded")
        if type(node) is str and expand_strings and string_transform is not None:
            return copy(string_transform(node), depth, False)
        count += 1
        if count > MAX_NODES:
            _fail("JSON node limit exceeded")
        if type(node) is str:
            charge_string(node)
            return node
        if node is None or type(node) in (bool, int, float):
            if type(node) is float and not math.isfinite(node):
                _fail("non-finite JSON number")
            try:
                charge(len(json.dumps(node, allow_nan=False).encode("utf-8")))
            except (OverflowError, ValueError) as exc:
                raise ValueError("invalid JSON number") from exc
            return node
        is_object = isinstance(node, dict) or (
            allow_frozen and isinstance(node, Mapping)
        )
        is_array = isinstance(node, list) or (allow_frozen and isinstance(node, tuple))
        if is_object or is_array:
            identity = id(node)
            if identity in seen:
                _fail("cyclic JSON value")
            seen.add(identity)
            try:
                if is_array:
                    charge(1)
                    result = []
                    for index, item in enumerate(node):
                        if index:
                            charge(1)
                        result.append(copy(item, depth + 1, expand_strings))
                    charge(1)
                    return result
                charge(1)
                result = {}
                for index, (key, item) in enumerate(node.items()):
                    if type(key) is not str:
                        _fail("JSON object keys must be strings")
                    count += 1
                    if count > MAX_NODES:
                        _fail("JSON node limit exceeded")
                    if index:
                        charge(1)
                    charge_string(key)
                    charge(1)
                    result[key] = copy(item, depth + 1, expand_strings)
                charge(1)
                return result
            finally:
                seen.remove(identity)
        _fail("non-JSON value")

    return copy(value, 0, True)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _strings(value: object, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        _fail("expected string array")
    if any(type(item) is not str or (nonempty and not item) for item in value):
        _fail("invalid string array")
    return tuple(value)


def _template_paths(value: object) -> set[str]:
    paths: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, str):
            for match in TEMPLATE_RE.finditer(node):
                paths.add(match.group(1))
            if "${" in TEMPLATE_RE.sub("", node):
                _fail("invalid template expression")
        elif isinstance(node, dict):
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    return paths


EVENT_DATA_KEYS = {
    "SessionStart": frozenset({"reason"}),
    "UserPromptSubmit": frozenset({"prompt"}),
    "PreToolUse": frozenset(
        {
            "tool_name",
            "tool_args",
            "tool_id",
            "provider",
            "operation",
            "definition_hash",
            "original_arguments",
            "candidate_arguments",
        }
    ),
    "PostToolUse": frozenset(
        {
            "tool_name",
            "tool_args",
            "tool_id",
            "provider",
            "operation",
            "definition_hash",
            "status",
            "result",
            "is_error",
        }
    ),
    "PostToolUseFailure": frozenset(
        {
            "tool_name",
            "tool_args",
            "tool_id",
            "provider",
            "operation",
            "definition_hash",
            "status",
            "reason",
            "result",
            "is_error",
        }
    ),
    "ApprovalRequested": frozenset({"calls", "session_active"}),
    "SubagentStart": frozenset(
        {"child_task", "tool_ids", "budget_caps", "model", "provider"}
    ),
    "PreCompact": frozenset({"candidate", "reason"}),
    "PostCompact": frozenset({"reason", "memory_id", "summarized_prefix_digest"}),
    "SubagentStop": frozenset({"child_run_id", "status"}),
    "Stop": frozenset({"status"}),
}


def _valid_template_path(event: str, path: str) -> bool:
    parts = path.split(".")
    if parts[0] in HookEvent.model_fields and parts[0] != "data":
        return len(parts) == 1
    if len(parts) < 2 or parts[0] != "data":
        return False
    # An untrusted argument/result subtree can be sent only as a whole JSON
    # value. Arbitrary nested keys never become documented privileged paths.
    return len(parts) == 2 and parts[1] in EVENT_DATA_KEYS.get(event, frozenset())


def parse_handlers(
    value: object, *, declared_variable_names: frozenset[str] | set[str] | None = None
) -> tuple[HookHandler, ...]:
    """Validate a complete ordered declaration batch; never omit invalid entries."""
    value = _json_tree(value)
    if not isinstance(value, list) or len(value) > MAX_HANDLERS:
        _fail("handlers must be a bounded array")
    declared = frozenset(declared_variable_names or ())
    seen: set[str] = set()
    handlers = []
    for raw in value:
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
        if (
            not isinstance(raw, dict)
            or type(raw.get("type")) is not str
            or raw["type"] not in {"command", "mcp_tool"}
        ):
            _fail("unknown handler type")
        specific = (
            {"argv", "env", "cwd"}
            if raw["type"] == "command"
            else {"server", "tool", "input"}
        )
        raw = _object(
            raw, allowed=common | specific, required={"id", "event", "type", "effects"}
        )
        if (
            type(raw["id"]) is not str
            or not ID_RE.fullmatch(raw["id"])
            or raw["id"] in seen
        ):
            _fail("invalid or duplicate handler ID")
        seen.add(raw["id"])
        event = raw["event"]
        if type(event) is not str or event not in EFFECTS:
            _fail("unsupported hook event")
        effects = _strings(raw["effects"])
        if len(set(effects)) != len(effects) or not set(effects) <= EFFECTS[event]:
            _fail("unsupported or duplicate effect")
        for key in ("required", "require_context"):
            if type(raw.get(key, False)) is not bool:
                _fail("requirement flag must be boolean")
        if (
            raw.get("required") or raw.get("require_context")
        ) and event in NON_REQUIRED_EVENTS:
            _fail("requirements forbidden on observation/Stop event")
        if raw.get("require_context") and "context" not in effects:
            _fail("require_context needs context effect")
        timeout = raw.get("timeout_seconds", 10)
        if not _finite_number(timeout) or not 0 < timeout <= 60:
            _fail("invalid handler timeout")
        match = raw.get("match")
        if match is not None:
            match = _object(match, allowed=MATCH_KEYS.get(event, frozenset()))
            if not match:
                _fail("empty matcher")
            patterns = {
                key: _strings(items, nonempty=True) for key, items in match.items()
            }
            if sum(map(len, patterns.values())) > 32 or any(
                len(p) > 256 for ps in patterns.values() for p in ps
            ):
                _fail("matcher limit exceeded")
            match = _freeze(patterns)
        kwargs: dict[str, Any] = {}
        if raw["type"] == "command":
            argv = _strings(raw.get("argv"))
            if not argv or not argv[0] or any("\x00" in arg for arg in argv):
                _fail("NUL in argv")
            env = _object(
                raw.get("env", {}),
                allowed=(
                    set(raw.get("env", {}))
                    if isinstance(raw.get("env", {}), dict)
                    else set()
                ),
            )
            clean_env = {}
            for key, item in env.items():
                if not ENV_RE.fullmatch(key) or key in {"PLUGIN_ROOT", "PLUGIN_DATA"}:
                    _fail("invalid or reserved environment name")
                if type(item) is str and "\x00" not in item:
                    clean_env[key] = item
                elif (
                    isinstance(item, dict)
                    and set(item) == {"variable"}
                    and type(item["variable"]) is str
                    and DECLARED_RE.fullmatch(item["variable"])
                    and item["variable"] in declared
                ):
                    clean_env[key] = _freeze(item)
                else:
                    _fail("invalid environment value or undeclared reference")
            cwd = raw.get("cwd")
            if cwd is not None:
                if type(cwd) is not str:
                    _fail("invalid cwd")
                try:
                    cwd = str(validate_existing_absolute_directory(cwd))
                except ValueError:
                    _fail("invalid cwd")
            kwargs.update(argv=argv, env=_freeze(clean_env), cwd=cwd)
        else:
            for key in ("server", "tool"):
                if type(raw.get(key)) is not str or not raw[key]:
                    _fail("missing MCP identity")
            input_value = _object(
                raw.get("input", {}),
                allowed=(
                    set(raw.get("input", {}))
                    if isinstance(raw.get("input", {}), dict)
                    else set()
                ),
            )
            if any(
                not _valid_template_path(event, path)
                for path in _template_paths(input_value)
            ):
                _fail("undocumented MCP template path")
            kwargs.update(
                server=raw["server"], tool=raw["tool"], input=_freeze(input_value)
            )
        handlers.append(
            HookHandler.model_construct(
                id=raw["id"],
                event=event,
                type=raw["type"],
                effects=frozenset(effects),
                required=raw.get("required", False),
                require_context=raw.get("require_context", False),
                match=match,
                timeout_seconds=float(timeout),
                **kwargs,
            )
        )
    return tuple(handlers)


def parse_native_handlers(
    value: object, *, declared_variable_names: frozenset[str] | set[str] | None = None
) -> tuple[HookHandler, ...]:
    """Validate the closed native v2 file envelope."""
    value = _json_tree(value)
    raw = _object(value, allowed={"version", "hooks"}, required={"version", "hooks"})
    if type(raw["version"]) is not int or raw["version"] != 2:
        _fail("unsupported hook definition version")
    if not isinstance(raw["hooks"], list) or len(raw["hooks"]) > 64:
        _fail("native installation hook limit exceeded")
    return parse_handlers(raw["hooks"], declared_variable_names=declared_variable_names)


def handler_phase(handler: HookHandler, *, dependency_required: bool = False) -> str:
    """Classify a reviewed declaration with the host's active dependency state."""
    if dependency_required and handler.event in NON_REQUIRED_EVENTS:
        _fail("incoming requirement forbidden for this event")
    if "updated_input" in handler.effects:
        return "transform"
    if "deny" in handler.effects or handler.required or dependency_required:
        return "validate"
    return "context" if handler.effects else "observe"


def _child_limits(value: object) -> Mapping[str, Any]:
    raw = _object(value, allowed={"tool_ids", "budget_caps"})
    if not raw:
        _fail("empty child limits")
    result: dict[str, Any] = {}
    if "tool_ids" in raw:
        ids = _strings(raw["tool_ids"])
        if any(
            not item.strip() or len(item) > 256 or "\x00" in item for item in ids
        ) or len(ids) != len(set(ids)):
            _fail("invalid child tool IDs")
        result["tool_ids"] = ids
    if "budget_caps" in raw:
        caps = _object(raw["budget_caps"], allowed=CAPS)
        if not caps:
            _fail("empty child budget caps")
        for key, amount in caps.items():
            if key in DURATION_CAPS:
                valid = _finite_number(amount)
            else:
                valid = type(amount) is int
            if not valid or amount < 0 or (key in POSITIVE_CAPS and amount == 0):
                _fail("invalid child budget cap")
            if key == "max_steps":
                from tldw_chatbook.Agents.agent_models import MAX_RUN_CONTROL_STEPS

                if amount > MAX_RUN_CONTROL_STEPS:
                    _fail("child step cap exceeds RunBudget ceiling")
        result["budget_caps"] = _freeze(caps)
    return _freeze(result)


def parse_result(value: object, handler: HookHandler) -> HookResult:
    """Validate an already decoded result object's shape and declared effects."""
    value = _json_tree(value, max_bytes=RESULT_BYTES)
    raw = _object(
        value,
        allowed={
            "version",
            "decision",
            "context",
            "updated_input",
            "child_limits",
            "continuation",
            "stop_continuations",
            "reason",
        },
        required={"version", "decision"},
    )
    if (
        type(raw["version"]) is not int
        or raw["version"] != 2
        or raw["decision"] not in ("pass", "deny")
    ):
        _fail("invalid v2 result header")
    if raw["decision"] == "deny" and "deny" not in handler.effects:
        _fail("undeclared deny effect")
    effect_fields = {
        "context",
        "updated_input",
        "child_limits",
        "continuation",
        "stop_continuations",
    }
    if any(key in raw and key not in handler.effects for key in effect_fields):
        _fail("undeclared result effect")
    reason = raw.get("reason")
    if reason is not None and (
        type(reason) is not str or len(reason.encode("utf-8")) > 4096
    ):
        _fail("invalid result reason")
    context = []
    total_context = 0
    if "context" in raw:
        if not isinstance(raw["context"], list):
            _fail("context must be an array")
        for item in raw["context"]:
            item = _object(
                item, allowed={"text", "lifetime"}, required={"text", "lifetime"}
            )
            if (
                type(item["text"]) is not str
                or type(item["lifetime"]) is not str
                or item["lifetime"] not in {"turn", "runtime"}
            ):
                _fail("invalid context block")
            size = len(item["text"].encode("utf-8"))
            total_context += size
            if (
                size > 4096
                or total_context > 16384
                or (item["lifetime"] == "runtime" and handler.event != "SessionStart")
                or (handler.event == "SessionStart" and item["lifetime"] != "runtime")
            ):
                _fail("invalid context size or lifetime")
            context.append(ContextBlock(text=item["text"], lifetime=item["lifetime"]))
    if handler.require_context and not any(block.text.strip() for block in context):
        _fail("required context is empty")
    updated = raw.get("updated_input")
    if "updated_input" in raw:
        updated = _freeze(
            _object(
                updated, allowed=set(updated) if isinstance(updated, dict) else set()
            )
        )
    limits = _child_limits(raw["child_limits"]) if "child_limits" in raw else None
    continuation = None
    if "continuation" in raw:
        item = _object(raw["continuation"], allowed={"message"}, required={"message"})
        message = item["message"]
        if (
            type(message) is not str
            or not message.strip()
            or len(message.encode("utf-8")) > 4096
        ):
            _fail("invalid continuation message")
        continuation = _freeze(item)
    stop = raw.get("stop_continuations")
    if "stop_continuations" in raw and stop is not True:
        _fail("stop_continuations must be true")
    return HookResult.model_construct(
        version=2,
        decision=raw["decision"],
        context=tuple(context),
        updated_input=updated,
        child_limits=limits,
        continuation=continuation,
        stop_continuations=stop,
        reason=reason,
    )


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate JSON key")
        result[key] = value
    return result


def _raw_text(raw: bytes | str, *, limit: int) -> str:
    if type(raw) is bytes:
        if len(raw) > limit:
            _fail("raw JSON byte limit exceeded")
        return raw.decode("utf-8", errors="strict")
    elif type(raw) is str:
        if len(raw.encode("utf-8")) > limit:
            _fail("raw JSON byte limit exceeded")
        return raw
    _fail("raw JSON must be bytes or text")


def decode_native_handlers(
    raw: bytes | str,
    *,
    declared_variable_names: frozenset[str] | set[str] | None = None,
) -> tuple[HookHandler, ...]:
    """Decode a bounded native hook file without losing duplicate-key evidence."""
    text = _raw_text(raw, limit=INPUT_BYTES)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda _: _fail("non-JSON constant"),
        )
    except RecursionError as exc:
        raise ValueError("JSON depth exceeded") from exc
    return parse_native_handlers(value, declared_variable_names=declared_variable_names)


def decode_result(raw: bytes | str, handler: HookHandler) -> HookResult:
    """Decode bounded command stdout or MCP text with strict JSON provenance."""
    text = _raw_text(raw, limit=RESULT_BYTES)
    if text == "":
        return parse_result({"version": 2, "decision": "pass"}, handler)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda _: _fail("non-JSON constant"),
        )
    except RecursionError as exc:
        raise ValueError("JSON depth exceeded") from exc
    return parse_result(value, handler)


def parse_event(value: object) -> HookEvent:
    """Validate a host-produced event envelope and documented data projection."""
    value = _json_tree(value)
    fields = set(HookEvent.model_fields)
    required = {
        "protocol_version",
        "event_id",
        "event",
        "timestamp",
        "runtime_session_id",
        "initiator",
        "origin",
        "causal_chain_id",
        "causal_depth",
        "data",
    }
    raw = _object(value, allowed=fields, required=required)
    if type(raw["protocol_version"]) is not int or raw["protocol_version"] != 2:
        _fail("unsupported event version")
    if type(raw["event"]) is not str or raw["event"] not in EFFECTS:
        _fail("unsupported event")
    for key in required - {"protocol_version", "event", "causal_depth", "data"}:
        if type(raw[key]) is not str or not raw[key]:
            _fail("missing host event identity")
    for key in fields - required:
        if key in raw and (type(raw[key]) is not str or not raw[key]):
            _fail("invalid optional host identity")
    if raw["initiator"] not in {
        "manual",
        "scheduled",
        "continuation",
        "child",
        "host_cleanup",
    }:
        _fail("invalid initiator")
    if type(raw["causal_depth"]) is not int or raw["causal_depth"] < 0:
        _fail("invalid causal depth")
    data = _object(raw["data"], allowed=EVENT_DATA_KEYS.get(raw["event"], frozenset()))
    if raw["event"] == "SessionStart" and (
        type(data.get("reason")) is not str
        or data["reason"] not in {"startup", "resume", "configuration_changed"}
    ):
        _fail("invalid SessionStart reason")
    string_fields = {
        "prompt",
        "tool_name",
        "tool_id",
        "provider",
        "operation",
        "definition_hash",
        "status",
        "reason",
        "child_run_id",
        "child_task",
        "model",
        "memory_id",
        "summarized_prefix_digest",
    }
    if any(type(item) is not str for key, item in data.items() if key in string_fields):
        _fail("invalid documented event string")
    if raw["event"] in {"PreCompact", "PostCompact"} and data.get("reason") not in {
        "manual",
        "automatic",
    }:
        _fail("invalid compaction reason")
    if "candidate" in data and (
        not isinstance(data["candidate"], list)
        or any(not isinstance(row, dict) for row in data["candidate"])
    ):
        _fail("compaction candidate must be a message array")
    if "tool_ids" in data or "budget_caps" in data:
        _child_limits(
            {key: data[key] for key in ("tool_ids", "budget_caps") if key in data}
        )
    if "memory_id" in data and (
        not data["memory_id"].strip() or len(data["memory_id"]) > 200
    ):
        _fail("invalid committed memory identity")
    if "summarized_prefix_digest" in data and (
        not data["summarized_prefix_digest"].strip()
        or len(data["summarized_prefix_digest"]) > 256
    ):
        _fail("invalid committed memory digest")
    for key in ("tool_args", "original_arguments", "candidate_arguments", "result"):
        if key in data and not isinstance(data[key], dict):
            _fail("tool arguments/results must be JSON objects")
    for key in ("session_active", "is_error"):
        if key in data and type(data[key]) is not bool:
            _fail("invalid documented event boolean")
    if "calls" in data:
        if not isinstance(data["calls"], list):
            _fail("approval calls must be an array")
        for call in data["calls"]:
            call = _object(
                call,
                allowed={"name", "args_summary"},
                required={"name", "args_summary"},
            )
            if any(type(item) is not str for item in call.values()):
                _fail("invalid approval summary")
    if raw["event"] in {"PreToolUse", "PostToolUse", "PostToolUseFailure"} and any(
        type(data.get(key)) is not str or not data[key]
        for key in ("tool_id", "provider")
    ):
        _fail("missing resolved tool identity")
    if raw["event"] == "PostToolUseFailure" and (
        type(data.get("reason")) is not str or not data["reason"]
    ):
        _fail("missing host failure reason")
    return HookEvent.model_construct(**{**raw, "data": _freeze(data)})
