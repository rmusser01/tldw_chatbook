"""ToolProvider for workspace-local fs_/web_/todo_ tools.

Spec: Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md.
ADR: backlog/decisions/032. Mirrors MCPToolProvider's approval discipline:
clear-first per-turn stamps, fail-closed invoke with pinned refusal
strings, stamp_scope() isolation around nested sub-agent runs. All Protocol
methods are sync and worker-thread safe; no Textual/event-loop imports.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState

from .agent_models import ToolCatalogEntry, ToolResult, ToolSchema
from .mcp_tool_provider import MCPPendingCall

SOURCE = "local"
LOCAL_SERVER_KEY = "local:__local__"
LOCAL_SERVER_LABEL = "Local workspace"

# Pinned refusal strings (spec §3.3) — tests assert on these verbatim.
LOCAL_DENY_REFUSAL = "blocked by local tool permissions (set to Off)"
LOCAL_TIMEOUT_REFUSAL = "user did not approve within the time limit; do not retry"
LOCAL_KILL_SWITCH_REFUSAL = "blocked — local tools are switched off"

_MAX_RESULT_BYTES = 32 * 1024
_MAX_ERROR_CHARS = 300


@dataclass(frozen=True)
class LocalToolSpec:
    """One local tool: schema plus its sync handler (args dict -> text)."""

    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], str]
    tags: tuple[str, ...] = ()


def _fit_result(text: str) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= _MAX_RESULT_BYTES:
        return text
    return raw[:_MAX_RESULT_BYTES].decode("utf-8", errors="ignore") + "\n… [truncated]"


class LocalToolProvider:
    """Exposes LocalToolSpecs behind the ToolProvider protocol, gated per call.

    Args:
        workspace_root: Confinement root for all path-taking tools.
        specs: Tool specs; defaults to the built-in set (fs_list pilot).
        resolve_state: (HubTool) -> EffectiveToolState, injected by the
            controller (owns permission-store access).
        kill_switch: () -> bool master off-switch.
        approval_callback: invoke()'s single-call fallback gate for an
            "ask"-state tool with no batch stamp; None fails closed.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        specs: list[LocalToolSpec] | None = None,
        resolve_state: Callable[[HubTool], EffectiveToolState] | None = None,
        kill_switch: Callable[[], bool] = lambda: False,
        approval_callback: Callable[[list[MCPPendingCall]], dict[str, str]] | None = None,
    ) -> None:
        self._root = workspace_root
        self._specs = {s.name: s for s in (specs if specs is not None else _default_specs(workspace_root))}
        self._resolve_state = resolve_state or (lambda hub: EffectiveToolState(state="ask", origin="global_default"))
        self._kill_switch = kill_switch
        self._approval_callback = approval_callback
        self._stamps: dict[str, str] = {}

    # -- catalog ------------------------------------------------------

    def _tool_id(self, name: str) -> str:
        return f"{SOURCE}:{name}"

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                id=self._tool_id(s.name),
                name=s.name,
                one_line_description=s.description.splitlines()[0],
                source=SOURCE,
            )
            for s in self._specs.values()
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        spec = self._specs[tool_id.split(":", 1)[1]]
        return ToolSchema(
            id=tool_id, name=spec.name,
            description=spec.description, parameters=spec.parameters,
        )

    def hub_tool_for(self, name: str) -> HubTool:
        """The HubTool view used for permission resolution (carries risk tags)."""
        spec = self._specs[name]
        return HubTool(
            server_key=LOCAL_SERVER_KEY,
            server_label=LOCAL_SERVER_LABEL,
            source="local",
            name=spec.name,
            description=spec.description,
            input_schema=spec.parameters,
            tags=spec.tags,
            stale=False,
            executable=True,
        )

    # -- approval stamps (mirror MCPToolProvider) ----------------------

    def apply_batch_decisions(self, decisions: dict[str, str]) -> None:
        """REPLACE this turn's stamps (never merge) — clear-first discipline."""
        self._stamps = dict(decisions)

    @contextmanager
    def stamp_scope(self) -> Iterator[None]:
        """Snapshot/restore stamps around a nested sub-agent run."""
        saved = self._stamps
        self._stamps = {}
        try:
            yield
        finally:
            self._stamps = saved

    def pending_gate_for(self, name: str, args: dict) -> MCPPendingCall | None:
        """The approval payload when this call needs human gating, else None."""
        spec = self._specs.get(name)
        if spec is None:
            return None
        state = self._resolve_state(self.hub_tool_for(name))
        if state.state != "ask":
            return None
        reason = (
            "config_changed" if state.config_changed
            else "risk_floored" if state.risk_floored
            else "ask"
        )
        return MCPPendingCall(
            llm_name=name,
            server_key=LOCAL_SERVER_KEY,
            tool_name=name,
            server_label=LOCAL_SERVER_LABEL,
            arguments=args,
            reason=reason,
        )

    # -- invocation -----------------------------------------------------

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[1] if ":" in tool_id else tool_id
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(ok=False, error=f"Unknown local tool: {name}")
        if self._kill_switch():
            return ToolResult(ok=False, error=LOCAL_KILL_SWITCH_REFUSAL)
        verdict = self._verdict_for(name)
        if verdict in ("deny",):
            return ToolResult(ok=False, error=LOCAL_DENY_REFUSAL)
        if verdict in ("timeout", "no_callback"):
            return ToolResult(ok=False, error=LOCAL_TIMEOUT_REFUSAL)
        try:
            return ToolResult(ok=True, content=_fit_result(spec.handler(args)))
        except Exception as exc:  # noqa: BLE001 — never raises across the boundary
            return ToolResult(ok=False, error=str(exc)[:_MAX_ERROR_CHARS])

    def _verdict_for(self, name: str) -> str:
        """Resolve this call's gate decision: allow executes; anything else refuses."""
        state = self._resolve_state(self.hub_tool_for(name))
        if state.state == "allow":
            return "allow"
        if state.state == "deny":
            return "deny"
        # ask: per-turn stamp wins; then single-call fallback; then fail closed.
        stamp = self._stamps.get(name)
        if stamp in ("approve_once", "approve_session", "always_allow"):
            return "allow"
        if stamp == "deny":
            return "deny"
        if stamp == "timeout":
            return "timeout"
        if self._approval_callback is not None:
            decision = self._approval_callback([self.pending_gate_for(name, {})]).get(name, "timeout")
            return "allow" if decision in ("approve_once", "approve_session", "always_allow") else decision
        return "no_callback"


def _default_specs(workspace_root: Path) -> list[LocalToolSpec]:
    from tldw_chatbook.Tools.local_tool_impls import list_directory

    return [
        LocalToolSpec(
            name="fs_list",
            description="List a directory's entries (dirs first, then files), relative to the workspace root.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path, relative to the workspace root (use \".\" for the root)."},
                },
                "required": ["path"],
            },
            handler=lambda args: list_directory(args["path"], workspace_root=workspace_root),
            tags=(),
        ),
    ]
