"""Inventory foreign contracts without borrowing native runtime qualification."""

import hashlib
import shlex
import sys

from ..models import ComponentRecord
from ..package_files import PackageFileError, canonical_json, parse_document

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
