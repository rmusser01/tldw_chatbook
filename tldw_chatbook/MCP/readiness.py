"""Readiness model for the MCP Hub.

Single source of truth mapping control-plane state onto display states,
priority-ordered reason codes, and the allowed-action set the Hub UI renders.
Adapted from the tldw_server webui readiness model (mcpHubReadiness.ts) to
chatbook's control-plane data shapes. Pure logic: no Textual, no I/O beyond
an injectable environment mapping (added in the derivation half of this
module, Task 2).
"""

from __future__ import annotations

from collections import Counter
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
    """One-line hub summary, e.g. '2 of 4 servers ready — 1 needs setup, 1 stale.'"""
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
