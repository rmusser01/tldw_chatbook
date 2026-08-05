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

from loguru import logger

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
        specs: Tool specs; defaults to the built-in set (fs_list, fs_read, fs_write, fs_edit).
        resolve_state: (HubTool) -> EffectiveToolState, injected by the
            controller (owns permission-store access).
        kill_switch: () -> bool master off-switch.
        approval_callback: invoke()'s single-call fallback gate for an
            "ask"-state tool with no batch stamp; None fails closed.
        is_session_approved: (HubTool) -> bool session-grant check (MCP
            Finding I1 parity); None means "no session store" (never
            short-circuits).
        persist_approval: (HubTool, decision) -> None side-effect hook for
            "approve_session"/"always_allow" verdicts (session grant write /
            permission-store "allow" with definition_hash); None means the
            decision executes this turn but is not persisted.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        specs: list[LocalToolSpec] | None = None,
        resolve_state: Callable[[HubTool], EffectiveToolState] | None = None,
        kill_switch: Callable[[], bool] = lambda: False,
        approval_callback: Callable[[list[MCPPendingCall]], dict[str, str]] | None = None,
        is_session_approved: Callable[[HubTool], bool] | None = None,
        persist_approval: Callable[[HubTool, str], None] | None = None,
    ) -> None:
        self._root = workspace_root
        self._specs = {s.name: s for s in (specs if specs is not None else _default_specs(workspace_root))}
        self._resolve_state = resolve_state or (lambda hub: EffectiveToolState(state="ask", origin="global_default"))
        self._kill_switch = kill_switch
        self._approval_callback = approval_callback
        self._is_session_approved = is_session_approved
        self._persist_approval = persist_approval
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
        name = tool_id.split(":", 1)[1] if ":" in tool_id else tool_id
        spec = self._specs[name]
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
        """Snapshot/restore stamps around a nested sub-agent run.

        Clears on entry -- a deliberate divergence from a pure snapshot:
        the child run starts stamp-less and re-checks permissions itself,
        so a parent's verdict can never leak into nested invocations.
        The parent's stamps are restored on exit, even on exception.
        """
        saved = self._stamps
        self._stamps = {}
        try:
            yield
        finally:
            self._stamps = saved

    def pending_gate_for(self, name: str, args: dict) -> MCPPendingCall | None:
        """The approval payload when this call needs human gating, else None."""
        # Same `local:`-prefix tolerance as invoke()/load_schema(): the
        # registry invokes by catalog id ("local:fs_list") while the review
        # hook resolves by LLM-facing name ("fs_list").
        name = name.split(":", 1)[1] if ":" in name else name
        spec = self._specs.get(name)
        if spec is None:
            return None
        hub = self.hub_tool_for(name)
        try:
            state = self._resolve_state(hub)
        except Exception as exc:  # noqa: BLE001 — fail closed to "let invoke handle it"
            logger.warning(
                f"LocalToolProvider: resolve_state failed for {name}: {exc}"
            )
            return None
        if state.state != "ask":
            return None
        # Finding I1 parity: a live session approval makes invoke() execute
        # without a stamp, so asking again here would be a pure re-prompt.
        if self._is_session_approved_safe(hub):
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
        """Execute one tool call. Never raises across the boundary.

        Fail-closed: only an explicit "allow" verdict executes; "deny" and
        any unrecognized verdict refuse with LOCAL_DENY_REFUSAL (mirrors
        MCPToolProvider._apply_verdict's fallthrough), "timeout"/
        "no_callback" with LOCAL_TIMEOUT_REFUSAL.
        """
        name = tool_id.split(":", 1)[1] if ":" in tool_id else tool_id
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(ok=False, error=f"Unknown local tool: {name}")
        if self._kill_switch_engaged():
            return ToolResult(ok=False, error=LOCAL_KILL_SWITCH_REFUSAL)
        verdict = self._verdict_for(name, args)
        if verdict == "allow":
            try:
                return ToolResult(ok=True, content=_fit_result(spec.handler(args)))
            except Exception as exc:  # noqa: BLE001 — never raises across the boundary
                return ToolResult(ok=False, error=(str(exc) or repr(exc))[:_MAX_ERROR_CHARS])
        if verdict in ("timeout", "no_callback"):
            return ToolResult(ok=False, error=LOCAL_TIMEOUT_REFUSAL)
        # "deny" and any unrecognized verdict fail closed the same way.
        return ToolResult(ok=False, error=LOCAL_DENY_REFUSAL)

    def _kill_switch_engaged(self) -> bool:
        """Never-raise kill-switch read.

        The fail-open/fail-closed POLICY for store read errors lives in the
        injected callable, not here: the controller's composition closure
        swallows `get_kill_switch` errors and returns False (fail open --
        deliberate MCPToolProvider._kill_switch_engaged parity), so in
        production this guard never sees a raise. What remains here is
        only the protocol boundary: if the injected callable ITSELF
        propagates (a test double, or a future composition without its own
        guard), invoke() still cannot raise, and that case fails closed
        (treated as engaged).
        """
        try:
            return bool(self._kill_switch())
        except Exception as exc:  # noqa: BLE001 — invoke() must never raise
            logger.warning(f"LocalToolProvider: kill_switch read failed: {exc}")
            return True

    def _verdict_for(self, name: str, args: dict) -> str:
        """Resolve this call's gate decision: only "allow" executes.

        Never raises: every injected callable is guarded, and a guard trip
        resolves to a refusing verdict.
        """
        hub = self.hub_tool_for(name)
        try:
            state = self._resolve_state(hub)
        except Exception as exc:  # noqa: BLE001 — fail closed on a resolution failure
            logger.warning(f"LocalToolProvider: resolve_state failed for {name}: {exc}")
            return "deny"
        if state.state == "allow":
            return "allow"
        if state.state == "deny":
            return "deny"
        # ask: per-turn stamp wins; then a live session approval; then the
        # single-call fallback; then fail closed.
        stamp = self._stamps.get(name)
        if stamp in ("approve_once", "approve_session", "always_allow"):
            if stamp != "approve_once":
                self._persist_approval_safe(hub, stamp)
            return "allow"
        if stamp == "deny":
            return "deny"
        if stamp == "timeout":
            return "timeout"
        if self._is_session_approved_safe(hub):
            return "allow"
        if self._approval_callback is not None:
            gate = self.pending_gate_for(name, args)
            if gate is None:
                # state re-resolution failed or flipped mid-call; fail closed.
                return "timeout"
            try:
                decisions = self._approval_callback([gate])
            except Exception as exc:  # noqa: BLE001 — fail closed on a callback failure
                logger.warning(f"LocalToolProvider: approval_callback failed for {name}: {exc}")
                return "timeout"
            decision = (decisions or {}).get(name, "timeout")
            if decision in ("approve_session", "always_allow"):
                self._persist_approval_safe(hub, decision)
            return "allow" if decision in ("approve_once", "approve_session", "always_allow") else decision
        return "no_callback"

    def _is_session_approved_safe(self, hub: HubTool) -> bool:
        """Never-raise session-grant read; absent/failed read means not approved."""
        if self._is_session_approved is None:
            return False
        try:
            return bool(self._is_session_approved(hub))
        except Exception as exc:  # noqa: BLE001 — a read failure must not deny silently-wrongly
            logger.warning(
                f"LocalToolProvider: is_session_approved failed for {hub.name}: {exc}"
            )
            return False

    def _persist_approval_safe(self, hub: HubTool, decision: str) -> None:
        """Never-raise persistence side effect; a failure must not block execution."""
        if self._persist_approval is None:
            return
        try:
            self._persist_approval(hub, decision)
        except Exception as exc:  # noqa: BLE001 — persistence failure must not block execution
            logger.warning(
                f"LocalToolProvider: persist_approval ({decision}) failed for {hub.name}: {exc}"
            )


def _default_specs(workspace_root: Path) -> list[LocalToolSpec]:
    from tldw_chatbook.Tools.local_tool_impls import (
        MAX_GLOB_RESULTS,
        MAX_GREP_RESULTS,
        edit_file,
        glob_files,
        grep_files,
        list_directory,
        read_file,
        write_file,
    )

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
        LocalToolSpec(
            name="fs_read",
            description="Read a text file with 1-based line numbers; pages via offset/limit. Refuses binary files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root."},
                    "offset": {"type": "integer", "default": 1, "description": "1-based first line to return."},
                    "limit": {"type": "integer", "description": "Maximum number of lines to return (default: all)."},
                },
                "required": ["path"],
            },
            handler=lambda args: read_file(
                args["path"],
                workspace_root=workspace_root,
                offset=args.get("offset", 1),
                limit=args.get("limit"),
            ),
            tags=(),
        ),
        LocalToolSpec(
            name="fs_write",
            description="Create or overwrite a file with the given content (full-file write), relative to the workspace root.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root. Parent directory must already exist."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
            handler=lambda args: write_file(args["path"], args["content"], workspace_root=workspace_root),
            tags=("mutates",),
        ),
        LocalToolSpec(
            name="fs_edit",
            description="Replace an exact string in a file. Fails unless the match is unique, unless replace_all is true.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root."},
                    "old_string": {"type": "string", "description": "Exact string to replace; must occur exactly once unless replace_all is true."},
                    "new_string": {"type": "string", "description": "Replacement string."},
                    "replace_all": {"type": "boolean", "default": False, "description": "Replace every occurrence of old_string."},
                },
                "required": ["path", "old_string", "new_string"],
            },
            handler=lambda args: edit_file(
                args["path"],
                args["old_string"],
                args["new_string"],
                workspace_root=workspace_root,
                replace_all=args.get("replace_all", False),
            ),
            tags=("mutates",),
        ),
        LocalToolSpec(
            name="fs_glob",
            description="Match files under the workspace with a glob pattern, newest-mtime first, workspace-relative paths.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern relative to the workspace root (e.g. \"**/*.py\"). Hidden dirs under the root are searched."},
                    "max_results": {"type": "integer", "description": "Maximum number of paths to return (default 100)."},
                },
                "required": ["pattern"],
            },
            handler=lambda args: glob_files(
                args["pattern"],
                workspace_root=workspace_root,
                max_results=args.get("max_results", MAX_GLOB_RESULTS),
            ),
            tags=(),
        ),
        LocalToolSpec(
            name="fs_grep",
            description="Regex search under the workspace: matching lines (default), file names, or per-file match counts.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "mode": {"type": "string", "enum": ["content", "files", "count"], "default": "content", "description": "\"content\": relpath:lineno:line; \"files\": matching paths only; \"count\": relpath:match_count."},
                    "max_results": {"type": "integer", "description": "Maximum number of result lines to return (default 100)."},
                },
                "required": ["pattern"],
            },
            handler=lambda args: grep_files(
                args["pattern"],
                workspace_root=workspace_root,
                mode=args.get("mode", "content"),
                max_results=args.get("max_results", MAX_GREP_RESULTS),
            ),
            tags=(),
        ),
    ]
