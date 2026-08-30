"""Effective workspace assistant default resolution (server reason-code contract)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable, Mapping

from .models import WorkspaceAssistantDefaults

DEGRADED_REASONS = (
    "persona_deleted",
    "persona_unavailable",
    "persona_feature_disabled",
    "permission_denied",
    "invalid_default",
    "unsupported_assistant_kind",
)


@dataclass(frozen=True)
class WorkspaceEffectiveAssistantDefault:
    status: str  # "available" | "unavailable" | "none"
    source: str  # "workspace" | "none"
    assistant_kind: str | None = None
    assistant_id: str | None = None
    label: str | None = None
    persona_memory_mode: str | None = None
    degraded_reason: str | None = None


_NONE = WorkspaceEffectiveAssistantDefault(status="none", source="none")


def resolve_effective_assistant_default(
    defaults: WorkspaceAssistantDefaults | None,
    persona_lookup: Callable[[str], Mapping | None],
) -> WorkspaceEffectiveAssistantDefault:
    if defaults is None:
        return _NONE
    if defaults.assistant_kind != "persona":
        return WorkspaceEffectiveAssistantDefault(
            "unavailable", "workspace", degraded_reason="unsupported_assistant_kind"
        )
    record = persona_lookup(defaults.assistant_id)
    if record is None or record.get("deleted"):
        return WorkspaceEffectiveAssistantDefault(
            "unavailable", "workspace", degraded_reason="persona_deleted"
        )
    if not isinstance(record, Mapping) or not str(record.get("id") or ""):
        return WorkspaceEffectiveAssistantDefault(
            "unavailable", "workspace", degraded_reason="persona_unavailable"
        )
    return WorkspaceEffectiveAssistantDefault(
        status="available",
        source="workspace",
        assistant_kind="persona",
        assistant_id=defaults.assistant_id,
        label=str(record.get("name") or "") or None,
        persona_memory_mode=defaults.persona_memory_mode,
    )


#: Server key the (display-only) posture preview resolves permissions
#: against — the built-in tool server, which is hash-free so an explicit
#: ``allow`` resolves at full fidelity (see
#: ``MCP.permission_store.resolve_effective_state_by_key``).
PREVIEW_SERVER_KEY = "builtin:tldw_chatbook"


def compose_posture_preview(
    persona_rules: Iterable[Mapping] | None,
    store_payload: Mapping | None,
    profile_id: str | None,
    tool_names: Sequence[str],
) -> list[str]:
    """Compose one read-only posture line per tool (Task 10, display-only).

    Per tool, one line of the form ``"<name>: <state> — <deciding layer>"``
    where ``<state>`` is ``available | ask | denied | capped (<n>)``.
    Deciding-layer precedence mirrors the runtime gates:

    1. Global kill switch (``store_payload["kill_switch"]``) denies all.
    2. Persona policy (``Agents.persona_policy.evaluate_tool_policy`` with
       ``rule_kind="mcp_tool"`` — the preview is display-only, so every
       name is previewed as an MCP tool): denied, ask
       (``require_confirmation``), or capped (``max_calls_per_turn``).
    3. Permission store resolution
       (``MCP.permission_store.resolve_effective_state_by_key`` against
       ``PREVIEW_SERVER_KEY`` with ``profile_id``): allow → ``available``,
       ask → ``ask``, deny → ``denied``.

    Args:
        persona_rules: The persona's raw ``policy_rules`` entries (may be
            ``None``; malformed entries are dropped by the parser).
        store_payload: A permission-store payload dict (``store.load()``
            shape) or ``None`` for an empty one.
        profile_id: Named permission profile to resolve against;
            ``None``/empty means the default profile.
        tool_names: Tool names to preview. Empty input degrades to a
            single ``"Tool catalog unavailable"`` line — never raises.

    Returns:
        The preview lines, one per tool name (or the single degrade line).
    """
    from tldw_chatbook.Agents.persona_policy import (
        evaluate_tool_policy,
        parse_persona_policy_from_rules,
    )
    from tldw_chatbook.MCP.permission_store import resolve_effective_state_by_key

    names = [str(name) for name in tool_names if str(name)]
    if not names:
        return ["Tool catalog unavailable"]
    payload = dict(store_payload) if isinstance(store_payload, Mapping) else {}
    profile = str(profile_id or "") or "default"
    lines: list[str] = []
    if payload.get("kill_switch"):
        return [f"{name}: denied — kill switch" for name in names]
    policy = parse_persona_policy_from_rules(persona_rules)
    for name in names:
        verdict = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name=name)
        if not verdict.advertised:
            lines.append(f"{name}: denied — persona policy")
            continue
        if verdict.requires_confirmation:
            lines.append(f"{name}: ask — persona policy")
            continue
        if verdict.max_calls_per_turn is not None:
            lines.append(
                f"{name}: capped ({verdict.max_calls_per_turn}) — persona policy"
            )
            continue
        try:
            state = resolve_effective_state_by_key(
                payload, PREVIEW_SERVER_KEY, name, profile_id=profile
            ).state
        except Exception:  # noqa: BLE001 -- display-only preview degrades
            state = "ask"
        label = {"allow": "available", "ask": "ask"}.get(state, "denied")
        lines.append(f"{name}: {label} — permissions")
    return lines
