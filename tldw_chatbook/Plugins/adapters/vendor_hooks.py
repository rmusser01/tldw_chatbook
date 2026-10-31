"""Inventory foreign contracts without borrowing native runtime qualification."""

import hashlib
import json
import re
import shlex
import sys

from tldw_chatbook.Agents.hooks_v2.models import HookHandler

from ..models import ComponentRecord, Diagnostic
from ..package_files import PackageFileError, canonical_json, parse_document
from ..schemas import require, validate_hook

# These are proposed correspondences, not supported runtime handlers. Native
# events cannot currently supply the vendor cwd/model/permission/transcript fields.
EVENTS = {
    "codex-hooks/2026-10-01": {
        "PreToolUse": "PreToolUse",
        "PermissionRequest": "ApprovalRequested",
        "PostToolUse": "PostToolUse",
        "UserPromptSubmit": "UserPromptSubmit",
        "SubagentStart": "SubagentStart",
        "SubagentStop": "SubagentStop",
        "SessionStart": "SessionStart",
        "SessionEnd": "SessionEnd",
        "PreCompact": "PreCompact",
        "PostCompact": "PostCompact",
        "Stop": "Stop",
        "Interrupt": "Interrupt",
    },
    "cursor-hooks/1@2026-10-01": {
        "preToolUse": "PreToolUse",
        "postToolUse": "PostToolUse",
        "postToolUseFailure": "PostToolUseFailure",
        "beforeSubmitPrompt": "UserPromptSubmit",
        "subagentStart": "SubagentStart",
        "subagentStop": "SubagentStop",
        "sessionStart": "SessionStart",
        "sessionEnd": "SessionEnd",
        "preCompact": "PreCompact",
        "stop": "Stop",
    },
}
OBSERVERS = {
    "PostToolUse",
    "PostToolUseFailure",
    "SubagentStop",
    "SessionEnd",
    "Interrupt",
    "Stop",
}


def normalize_vendor_hook(value: dict, dialect_version: str) -> dict:
    """Record every qualification axis; incomplete source contracts never execute."""
    if dialect_version not in EVENTS or not isinstance(value, dict):
        raise PackageFileError("vendor_hook_dialect_unsupported")
    value = parse_document(canonical_json(value).encode())
    event = value.get("event")
    target = EVENTS[dialect_version].get(event) if isinstance(event, str) else None
    reason, argv = "vendor_hook_payload_cwd_output_unqualified", None
    command = value.get("command")
    if value.get("type", "command") != "command":
        reason = "vendor_hook_execution_type_unsupported"
    elif not isinstance(command, str) or not command or sys.platform == "win32":
        reason = "vendor_hook_command_platform_unqualified"
    elif any(character in command for character in ";&|<>`$\n\r"):
        reason = "vendor_hook_shell_syntax_unsupported"
    else:
        try:
            argv = shlex.split(command, posix=True)
        except ValueError:
            reason = "vendor_hook_quoting_unsupported"
    if target is None:
        reason = "vendor_hook_event_unsupported"
    elif value.get("matcher") not in (None, "", "*"):
        reason = "vendor_hook_regex_unqualified"
    # Even simple argv is only parser evidence. Timeout/exit/output/timing,
    # payload and cwd must be exercised together before returning a HookHandler.
    return {
        "source": value,
        "dialect_version": dialect_version,
        "target_event": target,
        "proposed_argv": argv,
        "qualification": {
            "timing": False,
            "payload": False,
            "matcher": value.get("matcher") in (None, "", "*"),
            "input_output": False,
            "cwd": False,
            "argv": bool(argv),
            "timeout": False,
        },
        "reason": reason,
        "guard_scope_unknown": bool(
            value.keys()
            - {
                "event",
                "command",
                "type",
                "matcher",
                "timeout",
                "required",
                "failClosed",
            }
        )
        or target not in OBSERVERS
        or value.get("required") is not False
        and "required" in value
        or value.get("failClosed") is not False
        and "failClosed" in value,
    }


def inventory_vendor_hooks(
    value: dict, path: str, dialect: str
) -> tuple[dict[str, ComponentRecord], bool]:
    """Retain unsupported hooks and conservatively preserve unknown guards."""
    version = (
        "codex-hooks/2026-10-01" if dialect == "openai" else "cursor-hooks/1@2026-10-01"
    )
    if not isinstance(value, dict) or not isinstance(value.get("hooks"), dict):
        raise PackageFileError("vendor_hook_configuration_invalid")
    if dialect == "cursor" and (
        type(value.get("version", 1)) is not int or value.get("version", 1) != 1
    ):
        raise PackageFileError("vendor_hook_version_unsupported")
    root_scope = {key: item for key, item in value.items() if key != "hooks"}
    result, guard = (
        {},
        bool(root_scope.keys() - ({"version"} if dialect == "cursor" else set())),
    )
    count = 0
    for event, groups in value["hooks"].items():
        if not isinstance(groups, list):
            raise PackageFileError("vendor_hook_configuration_invalid")
        for group in groups:
            if not isinstance(group, dict):
                raise PackageFileError("vendor_hook_configuration_invalid")
            handlers = group.get("hooks", []) if dialect == "openai" else [group]
            if not isinstance(handlers, list) or not handlers:
                raise PackageFileError("vendor_hook_configuration_invalid")
            for raw in handlers:
                count += 1
                if count > 64 or not isinstance(raw, dict):
                    raise PackageFileError("vendor_hook_count_or_definition_invalid")
                merged = {**raw, "event": event}
                if dialect == "openai" and "matcher" in group:
                    merged["matcher"] = group["matcher"]
                proposal = normalize_vendor_hook(merged, version)
                group_scope = (
                    {key: item for key, item in group.items() if key != "hooks"}
                    if dialect == "openai"
                    else {}
                )
                proposal["source_scope"] = {"root": root_scope, "group": group_scope}
                proposal["guard_scope_unknown"] |= bool(
                    group_scope.keys() - {"matcher"}
                )
                guard |= proposal["guard_scope_unknown"]
                local_id = (
                    "vendor-"
                    + hashlib.sha256(f"{path}:{event}:{count}".encode()).hexdigest()[
                        :24
                    ]
                )
                key = "hook:" + local_id
                result[key] = ComponentRecord(
                    component_id=key,
                    kind="hook",
                    local_id=local_id,
                    path=path,
                    definition_json=canonical_json(proposal),
                    support="unsupported",
                    activation_blockers=(proposal["reason"],),
                )
    return result, guard


# Source event, host event and maximum effects are one closed protocol contract.
PROTOCOLS = {
    "codex/1:UserPromptSubmit": ("UserPromptSubmit", frozenset({"deny", "context"})),
    "codex/1:PreToolUse": (
        "PreToolUse",
        frozenset({"deny", "updated_input", "context"}),
    ),
    "codex/1:PostToolUse": ("PostToolUse", frozenset({"context"})),
    "cursor/1:beforeSubmitPrompt": ("UserPromptSubmit", frozenset({"deny"})),
    "cursor/1:preToolUse": ("PreToolUse", frozenset({"deny", "updated_input"})),
    "cursor/1:beforeMCPExecution": ("PreToolUse", frozenset({"deny"})),
    "cursor/1:postToolUse": ("PostToolUse", frozenset({"context"})),
    "cursor/1:postToolUseFailure": ("PostToolUseFailure", frozenset({"context"})),
}


def validate_protocol(protocol, event, kind, effects):
    """Refuse forged or mismatched codec declarations before execution."""
    require(
        isinstance(protocol, str) and protocol in PROTOCOLS,
        "vendor_protocol_unsupported",
    )
    expected, allowed = PROTOCOLS[protocol]
    require(
        event == expected and kind == "command" and set(effects) == allowed,
        "vendor_protocol_mismatch",
    )


def _qualified_vendor_hook(value: dict, dialect_version: str) -> HookHandler:
    """Normalize only simple POSIX argv and qualified events into native handlers.

    Guards use Chatbook's fail-closed policy and retained package cwd. These are
    disclosed adaptations, not original-host failure/cwd equivalence.
    """
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    require(isinstance(value, dict))
    require(
        set(value)
        <= {"id", "event", "type", "command", "matcher", "timeout", "failClosed"},
        "vendor_hook_fields_unsupported",
    )
    require(value.get("type", "command") == "command", "vendor_hook_type_unsupported")
    require(value.get("matcher", "") in ("", ".*"), "vendor_matcher_unsupported")
    require(type(value.get("failClosed", False)) is bool)
    protocol = dialect_version + ":" + str(value.get("event", ""))
    require(protocol in PROTOCOLS, "vendor_event_unsupported")
    require(sys.platform == "darwin", "vendor_argv_platform_unqualified")
    command = value.get("command")
    require(
        isinstance(command, str) and bool(command) and len(command.encode()) <= 16384
    )
    # Expansion is one host pass; reject every other shell interpolation/operator.
    command = command.replace("${CURSOR_PLUGIN_ROOT}", "${PLUGIN_ROOT}").replace(
        "${CLAUDE_PLUGIN_ROOT}", "${PLUGIN_ROOT}"
    )
    literal = re.sub(r"\$\{[A-Z][A-Z0-9_]{0,63}\}", "", command)
    require(
        not any(c in literal for c in "|&;<>`$*?[]~{}()#\n\r\x00"),
        "shell_form_unsupported",
    )
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        raise PackageFileError("shell_form_unsupported") from None
    require(bool(argv) and "=" not in argv[0], "shell_form_unsupported")
    event, effects = PROTOCOLS[protocol]
    return parse_handlers(
        [
            {
                "id": value.get("id", "vendor"),
                "event": event,
                "type": "command",
                "effects": sorted(effects),
                "argv": argv,
                "timeout_seconds": value.get("timeout", 10),
                "vendor_protocol": protocol,
            }
        ]
    )[0]


def matches_vendor(handler, event):
    """Preserve source MCP-only and success-only timing at native match points."""
    protocol = handler.vendor_protocol
    if protocol == "cursor/1:beforeMCPExecution":
        return event.data.get("operation") == "mcp.call"
    if protocol == "cursor/1:postToolUse":
        return (
            event.data.get("status") == "success"
            and event.data.get("is_error") is False
        )
    return True


def encode_vendor_payload(handler, event):
    """Project the event-specific source input without adding host authority."""
    from tldw_chatbook.Agents.hooks_v2.validation import INPUT_BYTES, _json_tree

    protocol = handler.vendor_protocol
    payload = {
        "session_id": event.runtime_session_id,
        "cwd": handler.cwd,
        "hook_event_name": protocol.partition(":")[2],
    }
    if event.event == "UserPromptSubmit":
        require("prompt" in event.data, "vendor_payload_unavailable")
        payload["prompt"] = event.data["prompt"]
    else:
        data = event.data
        require(
            "tool_name" in data and "tool_args" in data, "vendor_payload_unavailable"
        )
        arguments = data.get("candidate_arguments", data["tool_args"])
        payload.update(tool_name=data["tool_name"], tool_input=arguments)
        if protocol == "cursor/1:beforeMCPExecution":
            payload["tool_input"] = canonical_json(
                _json_tree(arguments, allow_frozen=True)
            )
        if event.event in {"PostToolUse", "PostToolUseFailure"}:
            require("result" in data, "vendor_payload_unavailable")
            payload["tool_response"] = data["result"]
    return canonical_json(
        _json_tree(payload, allow_frozen=True, max_bytes=INPUT_BYTES)
    ).encode()


def decode_vendor_result(raw: bytes, handler, returncode: int):
    """Decode bounded source output into the existing effect validator."""
    from tldw_chatbook.Agents.hooks_v2.validation import (
        RESULT_BYTES,
        _no_duplicate_keys,
        _raw_text,
        parse_result,
    )

    text = _raw_text(raw, limit=RESULT_BYTES)
    if returncode == 2 and "deny" in handler.effects:
        return parse_result({"version": 2, "decision": "deny"}, handler)
    require(returncode == 0, "vendor_nonzero_exit")
    value = (
        json.loads(text, object_pairs_hook=_no_duplicate_keys) if text.strip() else {}
    )
    require(isinstance(value, dict), "vendor_output_unsupported")
    output = {"version": 2, "decision": "pass"}
    protocol = handler.vendor_protocol
    if protocol.startswith("cursor/"):
        allowed = {"permission", "updated_input", "user_message", "additional_context"}
        if handler.event == "UserPromptSubmit":
            allowed = {"continue", "user_message"}
        require(set(value) <= allowed, "vendor_output_unsupported")
        permission = value.get("permission", "allow")
        require(permission in {"allow", "deny"}, "vendor_permission_unsupported")
        if "continue" in value:
            require(type(value["continue"]) is bool)
        if permission == "deny" or value.get("continue") is False:
            output["decision"] = "deny"
        for source, target in (
            ("updated_input", "updated_input"),
            ("user_message", "reason"),
        ):
            if source in value:
                output[target] = value[source]
        context = value.get("additional_context")
    else:
        require(
            set(value) <= {"decision", "reason", "hookSpecificOutput"},
            "vendor_output_unsupported",
        )
        require(
            value.get("decision", "approve") in {"approve", "block"},
            "vendor_decision_unsupported",
        )
        specific = value.get("hookSpecificOutput", {})
        require(
            isinstance(specific, dict)
            and set(specific)
            <= {
                "hookEventName",
                "permissionDecision",
                "permissionDecisionReason",
                "updatedInput",
                "additionalContext",
            },
            "vendor_output_unsupported",
        )
        require(
            specific.get("hookEventName", protocol.partition(":")[2])
            == protocol.partition(":")[2]
        )
        permission = specific.get("permissionDecision", "allow")
        require(permission in {"allow", "deny"}, "vendor_permission_unsupported")
        if permission == "deny" or value.get("decision") == "block":
            output["decision"] = "deny"
        if "updatedInput" in specific:
            output["updated_input"] = specific["updatedInput"]
        if "reason" in value or "permissionDecisionReason" in specific:
            output["reason"] = specific.get(
                "permissionDecisionReason", value.get("reason")
            )
        context = specific.get("additionalContext")
    if context is not None:
        output["context"] = [{"text": context, "lifetime": "turn"}]
    return parse_result(output, handler)


def _inventory_qualified_hooks(capture, overlay, dialect, *, _recipe="integrated_2"):
    """Retain unsupported handlers and unknown guard scope as explicit blockers."""
    records, diagnostics, blockers = [], [], []
    source = overlay.get("hooks", "./hooks/hooks.json")
    sources = source if isinstance(source, list) else [source]
    for source_index, value in enumerate(sources):
        path = (
            value
            if isinstance(value, str)
            else (
                ".cursor-plugin/plugin.json"
                if dialect == "cursor"
                else ".codex-plugin/plugin.json"
            )
        )
        try:
            if isinstance(value, str):
                if (
                    value.removeprefix("./") not in capture.files
                    and "hooks" not in overlay
                ):
                    continue
                document = capture.document(value)
            else:
                document = value
            require(
                isinstance(document, dict)
                and set(document) <= {"version", "hooks"}
                and isinstance(document.get("hooks"), dict)
            )
            if _recipe == "integrated_2" and dialect == "openai":
                require("version" not in document, "vendor_hook_version_unsupported")
            if dialect == "cursor":
                require(
                    type(document.get("version")) is int and document["version"] == 1,
                    "vendor_hook_version_unsupported",
                )
            for event, groups in document["hooks"].items():
                require(isinstance(groups, list))
                for group_index, group in enumerate(groups):
                    require(isinstance(group, dict))
                    if dialect == "openai":
                        require(set(group) <= {"matcher", "hooks"} and "hooks" in group)
                    handlers = (
                        group.get("hooks", []) if dialect == "openai" else [group]
                    )
                    require(isinstance(handlers, list))
                    for index, raw in enumerate(handlers):
                        local_id = (
                            f"{dialect}-{source_index}-{group_index}-{index}-{event}"
                        )
                        require(len(records) < 64, "hook_count_limit")
                        key = f"hook:{local_id}"
                        try:
                            require(isinstance(raw, dict))
                            if (
                                _recipe == "integrated_2"
                                and raw.get("failClosed") is True
                            ):
                                protocol = (
                                    ("codex/1" if dialect == "openai" else "cursor/1")
                                    + ":"
                                    + event
                                )
                                require(
                                    "deny" in PROTOCOLS.get(protocol, (None, ()))[1],
                                    "guard_scope_unknown",
                                )
                            candidate = {**raw, "id": local_id, "event": event}
                            if dialect == "openai" and "matcher" in group:
                                candidate["matcher"] = group["matcher"]
                            handler = _qualified_vendor_hook(
                                candidate,
                                "codex/1" if dialect == "openai" else "cursor/1",
                            )
                            definition = {
                                k: v
                                for k, v in handler.model_dump().items()
                                if v is not None
                            }
                            definition = validate_hook(
                                definition, capture, _recipe=_recipe
                            )
                            records.append(
                                ComponentRecord(
                                    component_id=key,
                                    kind="hook",
                                    local_id=local_id,
                                    path=path,
                                    support="adapted",
                                    definition_json=canonical_json(definition),
                                )
                            )
                        except (PackageFileError, TypeError, ValueError):
                            records.append(
                                ComponentRecord(
                                    component_id=key,
                                    kind="hook",
                                    local_id=local_id,
                                    path=path,
                                    support="unsupported",
                                    definition_json=canonical_json(raw),
                                    activation_blockers=("vendor_hook_unsupported",),
                                )
                            )
                            diagnostics.append(
                                Diagnostic(
                                    code="vendor_hook_unsupported",
                                    path=path,
                                    component_id=key,
                                )
                            )
                            # Unknown source semantics can hide a required guard.
                            blockers.append(f"guard_scope_unknown:{key}")
        except (PackageFileError, TypeError, ValueError):
            blockers.append("guard_scope_unknown")
            diagnostics.append(Diagnostic(code="vendor_hooks_invalid", path=path))
    return records, diagnostics, tuple(sorted(set(blockers)))
