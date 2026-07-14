"""Readiness model for the MCP Hub.

Single source of truth mapping control-plane state onto display states,
priority-ordered reason codes, and the allowed-action set the Hub UI renders.
Adapted from the tldw_server webui readiness model (mcpHubReadiness.ts) to
chatbook's control-plane data shapes. Pure logic: no Textual, no I/O beyond
an injectable environment mapping (added in the derivation half of this
module, Task 2).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReadinessState(str, Enum):
    NEEDS_SETUP = "needs_setup"
    CHECKING = "checking"
    READY = "ready"
    NEEDS_ATTENTION = "needs_attention"
    NO_TOOLS = "no_tools"
    STALE = "stale"


class ReasonCode(str, Enum):
    NOT_CONFIGURED = "not_configured"
    AUTH_MISSING = "auth_missing"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    PREFLIGHT_FAILED = "preflight_failed"
    UNREACHABLE = "unreachable"
    DISCOVERY_FAILED = "discovery_failed"
    CONFIG_CHANGED = "config_changed"
    DISCOVERY_NOT_RUN = "discovery_not_run"
    NO_TOOLS_RETURNED = "no_tools_returned"
    CATALOG_EXPIRED = "catalog_expired"
    PARTIAL_CAPABILITY = "partial_capability"


class HubAction(str, Enum):
    ADD_SERVER = "add_server"
    EDIT_CONFIG = "edit_config"
    OPEN_CREDENTIALS = "open_credentials"
    CONNECT = "connect"
    REFRESH_DISCOVERY = "refresh_discovery"
    VALIDATE = "validate"
    VIEW_DETAILS = "view_details"
    OPEN_TOOL_CATALOG = "open_tool_catalog"
    OPEN_AUDIT = "open_audit"


# First matching code wins the display state, primary message, and action set.
REASON_PRIORITY: tuple[ReasonCode, ...] = (
    ReasonCode.NOT_CONFIGURED,
    ReasonCode.AUTH_MISSING,
    ReasonCode.RUNTIME_UNAVAILABLE,
    ReasonCode.PREFLIGHT_FAILED,
    ReasonCode.UNREACHABLE,
    ReasonCode.DISCOVERY_FAILED,
    ReasonCode.CONFIG_CHANGED,
    ReasonCode.DISCOVERY_NOT_RUN,
    ReasonCode.NO_TOOLS_RETURNED,
    ReasonCode.CATALOG_EXPIRED,
    ReasonCode.PARTIAL_CAPABILITY,
)

REASON_TO_STATE: dict[ReasonCode, ReadinessState] = {
    ReasonCode.NOT_CONFIGURED: ReadinessState.NEEDS_SETUP,
    ReasonCode.AUTH_MISSING: ReadinessState.NEEDS_SETUP,
    # Discovered but not currently connected: usable knowledge, inactive
    # runtime — stale, not broken (disabled != broken contract).
    ReasonCode.RUNTIME_UNAVAILABLE: ReadinessState.STALE,
    ReasonCode.PREFLIGHT_FAILED: ReadinessState.NEEDS_ATTENTION,
    ReasonCode.UNREACHABLE: ReadinessState.NEEDS_ATTENTION,
    ReasonCode.DISCOVERY_FAILED: ReadinessState.NEEDS_ATTENTION,
    ReasonCode.CONFIG_CHANGED: ReadinessState.NEEDS_ATTENTION,
    ReasonCode.DISCOVERY_NOT_RUN: ReadinessState.NEEDS_SETUP,
    ReasonCode.NO_TOOLS_RETURNED: ReadinessState.NO_TOOLS,
    ReasonCode.CATALOG_EXPIRED: ReadinessState.STALE,
    ReasonCode.PARTIAL_CAPABILITY: ReadinessState.STALE,
}

REASON_TO_ACTIONS: dict[ReasonCode, tuple[HubAction, ...]] = {
    ReasonCode.NOT_CONFIGURED: (HubAction.ADD_SERVER,),
    ReasonCode.AUTH_MISSING: (HubAction.OPEN_CREDENTIALS, HubAction.EDIT_CONFIG, HubAction.VIEW_DETAILS),
    ReasonCode.RUNTIME_UNAVAILABLE: (HubAction.CONNECT, HubAction.VIEW_DETAILS),
    ReasonCode.PREFLIGHT_FAILED: (HubAction.EDIT_CONFIG, HubAction.VIEW_DETAILS),
    ReasonCode.UNREACHABLE: (HubAction.VALIDATE, HubAction.EDIT_CONFIG, HubAction.VIEW_DETAILS),
    ReasonCode.DISCOVERY_FAILED: (HubAction.REFRESH_DISCOVERY, HubAction.VIEW_DETAILS),
    ReasonCode.CONFIG_CHANGED: (HubAction.REFRESH_DISCOVERY, HubAction.VIEW_DETAILS),
    ReasonCode.DISCOVERY_NOT_RUN: (HubAction.CONNECT, HubAction.VALIDATE, HubAction.VIEW_DETAILS),
    ReasonCode.NO_TOOLS_RETURNED: (HubAction.REFRESH_DISCOVERY, HubAction.EDIT_CONFIG, HubAction.VIEW_DETAILS),
    ReasonCode.CATALOG_EXPIRED: (HubAction.REFRESH_DISCOVERY, HubAction.VIEW_DETAILS),
    ReasonCode.PARTIAL_CAPABILITY: (HubAction.VIEW_DETAILS, HubAction.OPEN_AUDIT),
}

READY_ACTIONS: tuple[HubAction, ...] = (
    HubAction.OPEN_TOOL_CATALOG,
    HubAction.REFRESH_DISCOVERY,
    HubAction.VIEW_DETAILS,
)

STATE_GLYPHS: dict[ReadinessState, str] = {
    ReadinessState.READY: "●",
    ReadinessState.CHECKING: "◐",
    ReadinessState.NEEDS_SETUP: "○",
    ReadinessState.NEEDS_ATTENTION: "!",
    ReadinessState.NO_TOOLS: "∅",
    ReadinessState.STALE: "◌",
}

STATE_LABELS: dict[ReadinessState, str] = {
    ReadinessState.READY: "Ready",
    ReadinessState.CHECKING: "Checking",
    ReadinessState.NEEDS_SETUP: "Needs setup",
    ReadinessState.NEEDS_ATTENTION: "Needs attention",
    ReadinessState.NO_TOOLS: "No tools",
    ReadinessState.STALE: "Stale",
}

# Short, human-facing phrase for each reason code (A3b). The inspector leads
# its readiness explanation with this instead of the internal `ReasonCode`
# value (e.g. "runtime_unavailable") -- that vocabulary is control-plane
# jargon, not something a user should ever have to read to understand what's
# wrong with a server.
REASON_LABELS: dict[ReasonCode, str] = {
    ReasonCode.NOT_CONFIGURED: "Not configured",
    ReasonCode.AUTH_MISSING: "Missing credentials",
    ReasonCode.RUNTIME_UNAVAILABLE: "Not connected",
    ReasonCode.PREFLIGHT_FAILED: "Failed pre-flight checks",
    ReasonCode.UNREACHABLE: "Server unreachable",
    ReasonCode.DISCOVERY_FAILED: "Tool discovery failed",
    ReasonCode.CONFIG_CHANGED: "Configuration changed since last check",
    ReasonCode.DISCOVERY_NOT_RUN: "Not validated yet",
    ReasonCode.NO_TOOLS_RETURNED: "No tools returned",
    ReasonCode.CATALOG_EXPIRED: "Catalog out of date",
    ReasonCode.PARTIAL_CAPABILITY: "Partially available",
}


def resolve_state(reasons: tuple[ReasonCode, ...]) -> ReadinessState:
    """Return the display state for a reason set via the priority order."""
    for code in REASON_PRIORITY:
        if code in reasons:
            return REASON_TO_STATE[code]
    return ReadinessState.READY


@dataclass(frozen=True)
class ReadinessSnapshot:
    """One server's readiness as rendered by rail, table, and inspector."""

    server_key: str  # "<source>:<stable_id>", e.g. "local:docs", "server:main"
    label: str
    source: str  # "local" | "server" | "builtin"
    state: ReadinessState
    reasons: tuple[ReasonCode, ...]
    message: str
    tool_count: int | None = None
    resource_count: int | None = None
    prompt_count: int | None = None
    transport: str = "stdio"
    auth_display: str = "none"
    scope_display: str = "—"
    is_connected: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_reason(self) -> ReasonCode | None:
        for code in REASON_PRIORITY:
            if code in self.reasons:
                return code
        return None

    @property
    def allowed_actions(self) -> tuple[HubAction, ...]:
        primary = self.primary_reason
        if primary is None:
            return READY_ACTIONS
        return REASON_TO_ACTIONS[primary]

    def badge_text(self) -> str:
        return f"{STATE_GLYPHS[self.state]} {STATE_LABELS[self.state]}"


def aggregate_summary(snapshots: list[ReadinessSnapshot]) -> str:
    """Build a one-line hub summary across all server readiness snapshots.

    Args:
        snapshots: Readiness snapshots for every server currently shown in
            the hub (any source: local, server, or builtin).

    Returns:
        A human-readable summary, e.g. "2 of 4 servers ready — 1 needs
        setup, 1 stale.", or "No MCP servers configured yet." when empty.
    """
    if not snapshots:
        return "No MCP servers configured yet."
    total = len(snapshots)
    counts = Counter(snap.state for snap in snapshots)
    ready = counts.get(ReadinessState.READY, 0)
    problems = [
        f"{count} {STATE_LABELS[state].lower()}"
        for state, count in counts.items()
        if state is not ReadinessState.READY and count
    ]
    suffix = f" — {', '.join(problems)}" if problems else ""
    return f"{ready} of {total} servers ready{suffix}."


BUILTIN_SERVER_KEY = "builtin:tldw_chatbook"
BUILTIN_CLIENT_SNIPPET = (
    '{\n'
    '  "mcpServers": {\n'
    '    "tldw_chatbook": {\n'
    '      "command": "python3",\n'
    '      "args": ["-m", "tldw_chatbook.MCP"]\n'
    '    }\n'
    '  }\n'
    '}'
)

_PLACEHOLDER_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")


def env_placeholder_names(env_placeholders: Mapping[str, str]) -> list[str]:
    """Extract the environment-variable names referenced by $NAME/${NAME} values."""
    names: list[str] = []
    for raw in env_placeholders.values():
        match = _PLACEHOLDER_RE.match(str(raw).strip())
        if match:
            names.append(match.group(1))
    return names


def _snapshot_counts(snapshot: Mapping[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    if not snapshot:
        return None, None, None
    return (
        len(snapshot.get("tools") or []),
        len(snapshot.get("resources") or []),
        len(snapshot.get("prompts") or []),
    )


def local_profile_readiness(
    record: dict[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> ReadinessSnapshot:
    """Derive readiness for one LocalMCPControlService.get_external_servers() item.

    Durable signals: discovery_snapshot presence and env placeholders vs the
    environment. Ephemeral: is_connected. needs_attention/checking/stale-by-age
    require persisted attempt tracking that does not exist yet (Phase 2+).
    """
    env = os.environ if environ is None else environ
    profile_id = str(record.get("profile_id") or "unknown")
    placeholders = dict(record.get("env_placeholders") or {})
    snapshot = record.get("discovery_snapshot")
    is_connected = bool(record.get("is_connected"))

    reasons: list[ReasonCode] = []
    missing = [name for name in env_placeholder_names(placeholders) if name not in env]
    if missing:
        reasons.append(ReasonCode.AUTH_MISSING)
    tool_count, resource_count, prompt_count = _snapshot_counts(snapshot)
    if snapshot is None:
        reasons.append(ReasonCode.DISCOVERY_NOT_RUN)
    else:
        if (tool_count or 0) == 0 and (resource_count or 0) == 0 and (prompt_count or 0) == 0:
            reasons.append(ReasonCode.NO_TOOLS_RETURNED)
        if not is_connected:
            reasons.append(ReasonCode.RUNTIME_UNAVAILABLE)

    reason_tuple = tuple(reasons)
    state = resolve_state(reason_tuple)
    if missing:
        message = f"Missing environment variables: {', '.join(missing)}."
    elif snapshot is None:
        message = "Not validated yet — connect or test to discover tools."
    elif not is_connected:
        message = f"{tool_count or 0} tools discovered; not currently connected."
    else:
        message = f"Connected — {tool_count or 0} tools available."

    return ReadinessSnapshot(
        server_key=f"local:{profile_id}",
        label=profile_id,
        source="local",
        state=state,
        reasons=reason_tuple,
        message=message,
        tool_count=tool_count,
        resource_count=resource_count,
        prompt_count=prompt_count,
        transport="stdio",
        auth_display=f"env ({len(placeholders)})" if placeholders else "none",
        scope_display="Personal",
        is_connected=is_connected,
        detail={
            "command": record.get("command"),
            "args": list(record.get("args") or []),
            "env_placeholders": placeholders,
            "missing_env": missing,
            "discovery_snapshot": snapshot,
        },
    )


def server_target_readiness(target: Any) -> ReadinessSnapshot:
    """Derive readiness for a configured tldw_server target (connection level)."""
    server_id = str(getattr(target, "server_id", "unknown"))
    label = str(getattr(target, "label", server_id))
    reachability = getattr(target, "last_known_reachability", None)
    auth_state = getattr(target, "last_known_auth_state", None)

    reasons: list[ReasonCode] = []
    if reachability == "unreachable":
        reasons.append(ReasonCode.UNREACHABLE)
    if auth_state in ("auth_required", "session_invalid"):
        reasons.append(ReasonCode.AUTH_MISSING)
    if reachability in (None, "unknown") and not reasons:
        reasons.append(ReasonCode.DISCOVERY_NOT_RUN)

    reason_tuple = tuple(reasons)
    state = resolve_state(reason_tuple)
    if state is ReadinessState.READY:
        message = "Reachable and authenticated."
    elif ReasonCode.UNREACHABLE in reason_tuple:
        message = "Server unreachable at last check."
    elif ReasonCode.AUTH_MISSING in reason_tuple:
        message = "Authentication required — sign in to this server."
    else:
        message = "Not checked yet — open the server to probe it."

    return ReadinessSnapshot(
        server_key=f"server:{server_id}",
        label=label,
        source="server",
        state=state,
        reasons=reason_tuple,
        message=message,
        transport="http",
        auth_display=str(getattr(target, "auth_mode", "api_key")),
        scope_display="—",
        detail={"base_url": getattr(target, "base_url", None)},
    )


_VALID_REASON_VALUES = {code.value for code in ReasonCode}
_VALID_STATE_VALUES = {state.value for state in ReadinessState}


def server_external_record_readiness(record: dict[str, Any], *, server_id: str) -> ReadinessSnapshot:
    """Normalize a raw /mcp/hub external-server record (pass-through dict).

    Readiness fields are backend-owned and not guaranteed; unknown or missing
    vocabulary degrades to discovery_not_run with an honest message rather
    than inventing a state. When the backend reports a `display_state` but no
    (or no recognized) `reason_codes`, that `display_state` is trusted
    directly instead of being fed through `resolve_state(())` -- an empty
    reason tuple always resolves to READY, which would silently contradict a
    non-ready state the backend explicitly reported.

    Args:
        record: One raw external-server record from `/mcp/hub`, as returned
            pass-through by the control-plane client.
        server_id: The connection-level server id this record belongs to,
            used to build the composite `server_key`.

    Returns:
        A `ReadinessSnapshot` normalizing the record's readiness fields.
    """
    external_id = str(record.get("server_id") or record.get("id") or record.get("name") or "unknown")
    label = str(record.get("name") or external_id)
    raw_reasons = record.get("reason_codes")
    reasons: tuple[ReasonCode, ...] = ()
    if isinstance(raw_reasons, (list, tuple)):
        reasons = tuple(
            ReasonCode(value) for value in raw_reasons if isinstance(value, str) and value in _VALID_REASON_VALUES
        )

    display_state = record.get("display_state")
    status_message = record.get("status_message")

    if reasons:
        state = resolve_state(reasons)
        message = str(status_message or "Reported by server.")
    elif isinstance(display_state, str) and display_state in _VALID_STATE_VALUES:
        # Reported, but without (recognized) reason codes to back it up --
        # trust the backend's explicit state rather than inventing READY.
        state = ReadinessState(display_state)
        message = str(status_message or "Reported by server without reason codes.")
    elif isinstance(display_state, str) and display_state:
        # A non-empty display_state that isn't valid ReadinessState
        # vocabulary: something is wrong, but we don't know what -- surface
        # it rather than pretending the server is ready.
        state = ReadinessState.NEEDS_ATTENTION
        message = "Server reported an unrecognized state."
    else:
        reasons = (ReasonCode.DISCOVERY_NOT_RUN,)
        state = resolve_state(reasons)
        message = "Readiness not reported by the server — validate to check."

    tool_count = record.get("tool_count")
    if tool_count is None and isinstance(record.get("tools"), list):
        tool_count = len(record["tools"])

    return ReadinessSnapshot(
        server_key=f"server:{server_id}/{external_id}",
        label=label,
        source="server",
        state=state,
        reasons=reasons,
        message=message,
        tool_count=tool_count if isinstance(tool_count, int) else None,
        transport=str(record.get("transport") or "stdio"),
        auth_display=str(record.get("credential_state") or "—"),
        scope_display=str(record.get("owner_scope_type") or "—"),
        detail={"raw": record},
    )


def builtin_readiness(
    *,
    enabled: bool,
    expose_tools: bool = True,
    expose_resources: bool = True,
    expose_prompts: bool = True,
) -> ReadinessSnapshot:
    """Readiness for chatbook's own MCP server (stdio-only; started by clients
    via `python -m tldw_chatbook.MCP`, never in-process)."""
    if enabled:
        reasons: tuple[ReasonCode, ...] = ()
        message = "Served over stdio when an MCP client launches chatbook."
    else:
        reasons = (ReasonCode.NOT_CONFIGURED,)
        message = "Disabled in config ([mcp].enabled = false)."
    return ReadinessSnapshot(
        server_key=BUILTIN_SERVER_KEY,
        label="tldw_chatbook (built-in)",
        source="builtin",
        state=resolve_state(reasons),
        reasons=reasons,
        message=message,
        transport="stdio",
        auth_display="none",
        scope_display="—",
        detail={
            "expose_tools": expose_tools,
            "expose_resources": expose_resources,
            "expose_prompts": expose_prompts,
            "client_snippet": BUILTIN_CLIENT_SNIPPET,
        },
    )
