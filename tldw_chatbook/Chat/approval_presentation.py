"""Ephemeral, owner-captured approval facts; never a permission input."""

from __future__ import annotations

import copy
import json
import shlex
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from unicodedata import category

from tldw_chatbook.MCP.redaction import redact_args, redact_mapping

if TYPE_CHECKING:
    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall


@dataclass(frozen=True)
class ApprovalAuthority:
    """Display facts supplied by the permission owner."""

    provider_kind: Literal[
        "mcp", "builtin", "local", "virtual_cli", "raw_shell", "runtime"
    ]
    profile_id: str | None
    profile_label: str
    location_label: str
    grant_domain: Literal["profile", "console_chat", "none"]
    stamp_domain: Literal["call", "tool_name", "shared_group"]
    revocation_label: str
    inherits_default: bool = False


@dataclass(frozen=True)
class ApprovalRowView:
    """One addressable verdict, including every captured call it controls."""

    verdict_key: str
    call_count: int
    action_label: str
    targets: tuple[str, ...]
    authority: ApprovalAuthority
    legal_decisions: tuple[str, ...]
    requires_review: bool
    withheld_scope_copy: str
    argument_sets: tuple[Mapping[str, object], ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class ApprovalBatchView:
    """Worker snapshot identified by its owning round and semantic revision."""

    round_id: str
    session_id: str
    run_id: str
    revision: int
    rows: tuple[ApprovalRowView, ...]
    call_count: int
    bulk_once: bool
    bulk_deny: bool


_DECISIONS = (
    "approve_once",
    "approve_session",
    "allow_matching",
    "always_allow",
    "deny",
)
_ACTIONS = {
    "read_file": "Read file",
    "fs_read": "Read file",
    "write_file": "Write file",
    "fs_write": "Write file",
    "fs_edit": "Edit file",
    "list_directory": "List directory",
    "fs_list": "List directory",
    "fs_glob": "Find files",
    "fs_grep": "Search files",
}


def display_literal(value: str) -> str:
    """Expose line, terminal and Unicode formatting controls in display facts."""
    return "".join(
        char.encode("unicode_escape").decode("ascii")
        if category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        else char
        for char in value
    )


def profile_authority(
    provider_kind: Literal["mcp", "builtin", "local", "virtual_cli"],
    profile_id: str | None,
    location_label: str,
    stamp_domain: Literal["call", "tool_name", "shared_group"],
) -> ApprovalAuthority:
    """Capture the exact permission-profile identity (the store's name)."""
    if profile_id is None:
        return ApprovalAuthority(
            provider_kind,
            None,
            "Unknown permission scope",
            location_label,
            "none",
            stamp_domain,
            "Unknown revocation scope",
        )
    return ApprovalAuthority(
        provider_kind,
        profile_id,
        "Default" if profile_id == "default" else profile_id,
        location_label,
        "profile",
        stamp_domain,
        "MCP Permissions",
        profile_id == "default",
    )


_PARAMETER_PREVIEW_BYTES = 256


def _display_target(arguments: Mapping[str, object], provider_kind: str) -> str:
    """Keep identifying targets complete and label bounded parameter excerpts."""
    safe = redact_mapping(dict(arguments or {}))
    if provider_kind == "virtual_cli" and isinstance(safe.get("argv"), (list, tuple)):
        tokens = [str(safe.get("command", "")), *(str(value) for value in safe["argv"])]
        return shlex.join(redact_args(tokens))
    for key in ("path", "command", "url", "uri", "target", "destination", "filename"):
        if key in safe:
            return str(safe[key])
    text = display_literal(json.dumps(safe, ensure_ascii=False, default=str))
    encoded = text.encode("utf-8")
    if len(encoded) > _PARAMETER_PREVIEW_BYTES:
        text = (
            encoded[:_PARAMETER_PREVIEW_BYTES].decode("utf-8", errors="ignore")
            + "… (parameters omitted)"
        )
    return "Parameters preview: " + text


def capture_approval_view(
    pending: Sequence[MCPPendingCall],
    *,
    round_id: str,
    session_id: str,
    run_id: str,
    revision: int,
) -> ApprovalBatchView:
    """Capture presentation on the owning worker without changing gate data.

    Args:
        pending: Producer-owned pending requests, in existing verdict order.
        round_id: Exact approval-round identity.
        session_id: Owning Console chat identity.
        run_id: Owning run identity.
        revision: Owner-supplied semantic version; change when replacing a snapshot.

    Returns:
        A display-only snapshot. Argument bodies are excluded from hot equality.
    """
    grouped: dict[str, list[MCPPendingCall]] = {}
    for call in pending:
        grouped.setdefault(call.call_id or call.llm_name, []).append(call)
    stamps = Counter(
        (call.server_key, call.tool_name, call.presentation_authority.profile_id)
        for calls in grouped.values()
        for call in calls[:1]
        if call.presentation_authority is not None
        and call.presentation_authority.provider_kind == "mcp"
        and call.presentation_authority.stamp_domain == "tool_name"
    )
    rows = []
    for key, calls in grouped.items():
        first = calls[0]
        owner = first.presentation_authority
        if owner is None or any(call.presentation_authority != owner for call in calls):
            owner = ApprovalAuthority(
                "runtime",
                None,
                "Unknown permission scope",
                "Unknown location",
                "none",
                "shared_group",
                "Unknown revocation scope",
            )
        choices = set(_DECISIONS)
        for call in calls:
            choices.intersection_update(call.options or _DECISIONS)
        if owner.grant_domain == "none":
            choices.intersection_update(("approve_once", "deny"))
        withheld = ""
        if (
            owner.provider_kind == "mcp"
            and owner.stamp_domain == "tool_name"
            and stamps[(first.server_key, first.tool_name, owner.profile_id)] > 1
            and "allow_matching" in choices
        ):
            choices.remove("allow_matching")
            withheld = (
                "Remember these inputs is unavailable: independent calls to this tool "
                "share a permission stamp. Allow once remains available."
            )
        owner = replace(
            owner,
            profile_label=display_literal(owner.profile_label),
            location_label=display_literal(owner.location_label),
            revocation_label=display_literal(owner.revocation_label),
        )
        action = first.tool_name
        if owner.provider_kind in ("local", "builtin"):
            action = _ACTIONS.get(first.tool_name, first.tool_name)
        elif owner.provider_kind == "virtual_cli":
            action = "Run " + first.tool_name
        elif owner.provider_kind == "raw_shell":
            action = "Run shell command"
        arguments = tuple(
            MappingProxyType(
                copy.deepcopy(
                    dict(
                        call.captured_arguments
                        if call.captured_arguments is not None
                        else call.arguments
                    )
                )
            )
            for call in calls
        )
        # Display redaction must precede target derivation. Matching keeps originals above.
        targets = tuple(
            display_literal(_display_target(call.arguments, owner.provider_kind))
            for call in calls
        )
        rows.append(
            ApprovalRowView(
                key,
                len(calls),
                display_literal(action),
                targets,
                owner,
                tuple(decision for decision in _DECISIONS if decision in choices),
                any(call.requires_individual_review for call in calls),
                withheld,
                arguments,
            )
        )
    captured = tuple(rows)
    return ApprovalBatchView(
        round_id,
        session_id,
        run_id,
        revision,
        captured,
        len(pending),
        bool(captured)
        and all(
            "approve_once" in row.legal_decisions and not row.requires_review
            for row in captured
        ),
        bool(captured) and all("deny" in row.legal_decisions for row in captured),
    )


def scope_copy(row: ApprovalRowView, decision: str) -> str:
    """Explain a supported choice using captured owner facts and full counts."""
    if decision not in row.legal_decisions:
        return "This scope is unavailable for the captured request."
    calls = (
        "this call" if row.call_count == 1 else f"all {row.call_count} captured calls"
    )
    if decision == "approve_once":
        return f"Allow {calls} once."
    if decision == "deny":
        return f"Deny {calls}; the model is told not to retry."
    owner = row.authority
    if decision == "approve_session":
        if owner.grant_domain == "console_chat":
            return (
                "Until Chatbook exits: future shell commands in this Console chat. "
                "Clears on Disarm or when Chatbook exits."
            )
        return (
            f"Until Chatbook exits: every call to this tool in profile {owner.profile_label}, "
            "across chats using that exact profile. Revoke in MCP Tools: selected tool → Revoke."
        )
    scope = (
        f"Remember exactly the inputs of {calls}"
        if decision == "allow_matching"
        else "Remember this tool"
    )
    route = (
        "MCP Tools: selected tool → Remove the matching input rule"
        if decision == "allow_matching"
        else "MCP Permissions: change this tool’s policy"
    )
    copy_text = f"{scope} in profile {owner.profile_label}. Revoke in {route}."
    if owner.inherits_default:
        copy_text += " Named profiles inherit Default wherever they leave policy unset."
    return copy_text


def _build_approval_payload(
    round_id: str,
    session_id: str,
    run_id: str,
    pending: list[MCPPendingCall],
    timeout_seconds: float,
    deadline: float | None,
    *,
    read_global_ToolExecutionPolicy: Callable[[], Any],
    revision: int = 1,
) -> dict[str, Any]:
    """Marshal one approval round's card payload.

    ADR-090: rows carry ``rationale`` (the model's advisory context) and
    ``description`` (the tool definition's own text, for the external
    summarizer); the payload carries a ``summary`` slot that starts ``None``
    and is filled by the advisory summarizer -- payload-carried so any
    remount re-renders it rather than depending on a live patch surviving.
    ``revision`` changes only when the owner replaces the captured pending
    snapshot; advisory summary patches retain both the snapshot and revision.
    """
    view = capture_approval_view(
        pending,
        round_id=round_id,
        session_id=session_id,
        run_id=run_id,
        revision=revision,
    )
    return {
        "view": view,
        "presentation_state": {},
        "presentation_revision": revision,
        "round_id": round_id,
        "session_id": session_id,
        "run_id": run_id,
        "calls": [
            {
                "llm_name": call.llm_name,
                "server_key": call.server_key,
                "tool_name": call.tool_name,
                "server_label": call.server_label,
                "arguments": dict(call.arguments or {}),
                "reason": call.reason,
                "options": list(call.options),
                "effects": list(call.effects),
                "execution_policy": (
                    call.execution_policy.value
                    if isinstance(
                        call.execution_policy, read_global_ToolExecutionPolicy()
                    )
                    else read_global_ToolExecutionPolicy().BOUNDED_ABANDONABLE.value
                ),
                "path_precheck_failed": call.path_precheck_failed,
                "call_id": call.call_id,
                "full_command": call.full_command,
                "warning": call.warning,
                "scope_notice": call.scope_notice,
                "rationale": str(getattr(call, "rationale", "") or ""),
                "description": str(getattr(call, "description", "") or ""),
            }
            for call in pending
        ],
        "timeout_seconds": timeout_seconds,
        "deadline_monotonic": deadline,
        "summary": None,
    }
