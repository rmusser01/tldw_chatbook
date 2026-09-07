"""One model tool backed by independently gated read-only virtual commands."""

from __future__ import annotations

import copy
import re
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, ContextManager, Iterator, Mapping, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.virtual_cli_impls import (
    MAX_ARGV_ITEMS,
    VIRTUAL_CLI_COMMANDS,
    VirtualCliCommand,
    VirtualCliArgumentError,
    VirtualCliRegistry,
    parse_request,
)
from tldw_chatbook.Tools.workspace_tool_executor import (
    WorkspaceToolExecutionError,
    WorkspaceToolExecutor,
)

from .agent_models import ToolCall, ToolCatalogEntry, ToolResult, ToolSchema
from .local_tool_provider import (
    LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL,
    LOCAL_DENY_REFUSAL,
    LOCAL_GATE_ERROR_REFUSAL,
    LOCAL_KILL_SWITCH_REFUSAL,
    LOCAL_ROOT_CHANGED_REFUSAL,
    LOCAL_TIMEOUT_REFUSAL,
    RunAdmittedWorkspaceRoot,
)
from .mcp_tool_provider import MCPPendingCall
from .run_context import current_run_id, current_tool_call_id
from .tool_catalog import redact_root_locator

VIRTUAL_CLI_TOOL_NAME = "virtual_cli"
VIRTUAL_CLI_SERVER_KEY = "local:__virtual_cli__"
VIRTUAL_CLI_SERVER_LABEL = "Virtual CLI (read-only)"
SOURCE = "virtual_cli"

_MAX_RESULT_BYTES = 32 * 1024
_MAX_ERROR_CHARS = 300
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")

_USAGE = {
    "ls": "ls [PATH]",
    "cat": "cat PATH [--offset N] [--limit N]",
    "grep": "grep PATTERN [--mode content|files|count]",
    "find": "find GLOB",
    "stat": "stat PATH",
    "git_status": "git_status [PATH]",
    "git_diff": "git_diff [--staged] [--range REF] [--path PATH] [--stat]",
    "git_log": "git_log [--count N] [--path PATH]",
    "git_blame": "git_blame PATH [--start N] [--end N]",
    "git_branches": "git_branches",
}

_MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "enum": list(VIRTUAL_CLI_COMMANDS),
            "description": "One fixed read-only virtual command.",
        },
        "argv": {
            "type": "array",
            "items": {"type": "string", "maxLength": 4096},
            "maxItems": MAX_ARGV_ITEMS,
            "description": "Arguments for the selected command; no shell syntax.",
        },
    },
    "required": ["command", "argv"],
    "additionalProperties": False,
}


class _VirtualCliInput(BaseModel):
    """Validated model-facing Virtual CLI arguments."""

    model_config = ConfigDict(extra="forbid")

    command: VirtualCliCommand
    argv: list[str] = Field(max_length=MAX_ARGV_ITEMS)


class _AdmittedVirtualCliInput(_VirtualCliInput):
    """Validated Virtual CLI arguments with run-root selection."""

    root_alias: str | None = None


def _sanitize_result(text: str) -> str:
    text = _ANSI_ESCAPE.sub("", text)
    text = "".join(
        char
        for char in text
        if char in "\n\t" or (ord(char) >= 32 and not 127 <= ord(char) <= 159)
    )
    raw = text.encode("utf-8")
    if len(raw) <= _MAX_RESULT_BYTES:
        return text
    return raw[:_MAX_RESULT_BYTES].decode("utf-8", errors="ignore") + "\n… [truncated]"


class VirtualCliProvider:
    """Expose one schema while resolving command and admitted-root authority."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        resolve_state: Callable[[HubTool], EffectiveToolState] | None = None,
        local_tools_enabled: Callable[[], bool] = lambda: True,
        kill_switch: Callable[[], bool] = lambda: False,
        approval_callback: Callable[[list[MCPPendingCall]], dict[str, str]]
        | None = None,
        is_session_approved: Callable[[HubTool], bool] | None = None,
        persist_approval: Callable[[HubTool, str], None] | None = None,
        record_decision: Callable[[HubTool, str], None] | None = None,
        root_guard: Callable[[], bool] | None = None,
        authority_scope: Callable[[], ContextManager[Path]] | None = None,
        result_redaction_root: Path | None = None,
        workspace_executor: WorkspaceToolExecutor | None = None,
        admitted_roots: Sequence[RunAdmittedWorkspaceRoot] | None = None,
    ) -> None:
        selected_executor = (
            WorkspaceToolExecutor(workspace_root)
            if workspace_executor is None
            else workspace_executor
        )
        self._registry = VirtualCliRegistry(
            workspace_root,
            workspace_executor=selected_executor,
        )
        ordered_roots = (
            None
            if admitted_roots is None
            else tuple(sorted(admitted_roots, key=lambda authority: authority.alias))
        )
        self._admitted_roots = (
            None
            if ordered_roots is None
            else {authority.alias: authority for authority in ordered_roots}
        )
        if admitted_roots is not None and len(self._admitted_roots) != len(
            admitted_roots
        ):
            raise ValueError("admitted root aliases must be unique")
        self._registries_by_alias: dict[str, VirtualCliRegistry] = {}
        self._model_schema = copy.deepcopy(_MODEL_SCHEMA)
        if self._admitted_roots:
            usable_roots: dict[str, RunAdmittedWorkspaceRoot] = {}
            for alias, authority in self._admitted_roots.items():
                try:
                    executor = authority.workspace_executor or WorkspaceToolExecutor(
                        authority.root
                    )
                    registry = VirtualCliRegistry(
                        authority.root,
                        workspace_executor=executor,
                    )
                except Exception:  # noqa: BLE001 - a raced root is revoked
                    continue
                self._registries_by_alias[alias] = registry
                usable_roots[alias] = authority
            self._admitted_roots = usable_roots
            if usable_roots:
                aliases = list(usable_roots)
                self._model_schema["properties"]["root_alias"] = {
                    "type": "string",
                    "enum": aliases,
                    "description": (
                        "Stable workspace-folder binding alias for this run."
                    ),
                }
                if len(aliases) > 1:
                    self._model_schema["required"].append("root_alias")
        self._resolve_state = resolve_state or (
            lambda _hub: EffectiveToolState(state="ask", origin="global_default")
        )
        self._local_tools_enabled = local_tools_enabled
        self._kill_switch = kill_switch
        self._approval_callback = approval_callback
        self._is_session_approved = is_session_approved
        self._persist_approval = persist_approval
        self._record_decision = record_decision
        self._root_guard = root_guard
        self._authority_scope = authority_scope
        self._result_redaction_root = (
            Path(result_redaction_root).resolve()
            if result_redaction_root is not None
            else None
        )
        self._stamps: dict[tuple[str, str], str] = {}
        self._stamps_lock = threading.Lock()

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                id=f"{SOURCE}:{VIRTUAL_CLI_TOOL_NAME}",
                name=VIRTUAL_CLI_TOOL_NAME,
                one_line_description="Run a fixed read-only virtual command without a host shell.",
                source=SOURCE,
            )
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = tool_id.split(":", 1)[-1]
        if name != VIRTUAL_CLI_TOOL_NAME:
            raise KeyError(f"Unknown virtual CLI tool: {tool_id}")
        return ToolSchema(
            id=tool_id,
            name=VIRTUAL_CLI_TOOL_NAME,
            description=(
                "Run one allowlisted read-only workspace or Git command. "
                "argv is structured and is never parsed by a host shell."
            ),
            parameters=self._model_schema,
        )

    def hub_tool_for(self, command: str) -> HubTool:
        if command not in VIRTUAL_CLI_COMMANDS:
            raise KeyError(command)
        usage = _USAGE[command]
        properties = {
            "argv": {
                "type": "array",
                "items": {"type": "string"},
                "description": usage,
            }
        }
        required = ["argv"]
        root_alias_schema = self._model_schema["properties"].get("root_alias")
        if root_alias_schema is not None:
            properties["root_alias"] = copy.deepcopy(root_alias_schema)
            if "root_alias" in self._model_schema["required"]:
                required.append("root_alias")
        return HubTool(
            server_key=VIRTUAL_CLI_SERVER_KEY,
            server_label=VIRTUAL_CLI_SERVER_LABEL,
            source="local",
            name=command,
            description=(
                f"Read-only virtual command: {usage}. No host shell is invoked. "
                "Permission is independent from equivalent filesystem and Git tools."
            ),
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            tags=(),
            stale=False,
            executable=True,
        )

    def hub_tools(self) -> list[HubTool]:
        return [self.hub_tool_for(command) for command in VIRTUAL_CLI_COMMANDS]

    def _validated_args(
        self, args: Mapping[str, object]
    ) -> tuple[str, Sequence[str], RunAdmittedWorkspaceRoot | None]:
        if not isinstance(args, Mapping):
            raise VirtualCliArgumentError("virtual_cli arguments must be an object")
        model = _AdmittedVirtualCliInput if self._admitted_roots else _VirtualCliInput
        try:
            validated = model.model_validate(dict(args))
        except PydanticValidationError as exc:
            raise VirtualCliArgumentError(
                "virtual_cli arguments do not match the tool schema"
            ) from exc
        alias = (
            validated.root_alias
            if isinstance(validated, _AdmittedVirtualCliInput)
            else None
        )
        authority = self._select_admitted_root(alias)
        request, _parsed = parse_request(validated.command, validated.argv)
        return request.command, request.argv, authority

    def _select_admitted_root(self, alias: object) -> RunAdmittedWorkspaceRoot | None:
        if self._admitted_roots is None:
            return None
        if not self._admitted_roots:
            raise VirtualCliArgumentError("no workspace root was admitted for this run")
        if alias is None:
            if len(self._admitted_roots) != 1:
                raise VirtualCliArgumentError(
                    "root_alias is required when multiple roots are admitted"
                )
            return next(iter(self._admitted_roots.values()))
        if not isinstance(alias, str) or alias not in self._admitted_roots:
            raise VirtualCliArgumentError(
                "root_alias does not name a root admitted for this run"
            )
        return self._admitted_roots[alias]

    def _authority_is_valid(self, authority: RunAdmittedWorkspaceRoot | None) -> bool:
        if authority is None:
            return self._root_is_valid()
        try:
            return bool(authority.guard(False))
        except Exception:  # noqa: BLE001 - invocation must fail closed
            return False

    def pending_gate_for(self, call: ToolCall) -> MCPPendingCall | None:
        if call.name != VIRTUAL_CLI_TOOL_NAME:
            return None
        try:
            command, _argv, authority = self._validated_args(call.args)
            if not self._authority_is_valid(authority):
                return None
            state = self._resolve_state(self.hub_tool_for(command))
        except Exception:
            return None
        hub = self.hub_tool_for(command)
        if state.state != "ask" or self._session_approved(hub):
            return None
        return MCPPendingCall(
            llm_name=VIRTUAL_CLI_TOOL_NAME,
            server_key=VIRTUAL_CLI_SERVER_KEY,
            tool_name=command,
            server_label=VIRTUAL_CLI_SERVER_LABEL,
            arguments=dict(call.args),
            reason=(
                "config_changed"
                if state.config_changed
                else "risk_floored"
                if state.risk_floored
                else "ask"
            ),
            call_id=call.call_id or command,
        )

    def apply_batch_decisions(
        self,
        run_id: str,
        decisions: dict[str, str],
        pending: Sequence[MCPPendingCall] = (),
    ) -> None:
        with self._stamps_lock:
            self._stamps = {
                key: value for key, value in self._stamps.items() if key[0] != run_id
            }
            for row in pending:
                key = row.call_id or row.tool_name
                decision = decisions.get(key)
                if decision is not None:
                    self._stamps[(run_id, key)] = decision

    def _pop_stamp(self, run_id: str, command: str) -> str | None:
        key = current_tool_call_id() or command
        with self._stamps_lock:
            return self._stamps.pop((run_id, key), None)

    @contextmanager
    def stamp_scope(self, run_id: str) -> Iterator[None]:
        """Hide and restore this run's pending verdicts around a child run."""
        with self._stamps_lock:
            saved = {
                key: value for key, value in self._stamps.items() if key[0] == run_id
            }
            self._stamps = {
                key: value for key, value in self._stamps.items() if key[0] != run_id
            }
        try:
            yield
        finally:
            with self._stamps_lock:
                self._stamps = {
                    key: value
                    for key, value in self._stamps.items()
                    if key[0] != run_id
                }
                self._stamps.update(saved)

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[-1]
        if name != VIRTUAL_CLI_TOOL_NAME:
            return ToolResult(ok=False, error=f"Unknown virtual CLI tool: {name}")
        try:
            command, argv, authority = self._validated_args(args)
        except VirtualCliArgumentError as exc:
            return ToolResult(ok=False, error=f"invalid virtual_cli request: {exc}")
        hub = self.hub_tool_for(command)
        if not self._authority_is_valid(authority):
            self._record(hub, "denied")
            return ToolResult.blocked(LOCAL_ROOT_CHANGED_REFUSAL)
        if not self._local_tools_are_enabled():
            self._record(hub, "denied")
            return ToolResult.blocked(LOCAL_KILL_SWITCH_REFUSAL)
        if self._kill_switch_engaged():
            self._record(hub, "denied")
            return ToolResult.blocked(LOCAL_KILL_SWITCH_REFUSAL)
        try:
            state = self._resolve_state(hub)
        except Exception:
            self._record(hub, "denied")
            return ToolResult.blocked(LOCAL_GATE_ERROR_REFUSAL)
        if state.state == "deny":
            self._record(hub, "denied")
            return ToolResult.blocked(LOCAL_DENY_REFUSAL)
        if state.state == "allow":
            verdict = "allow"
        else:
            verdict = self._ask_verdict(hub, command, args)
        if verdict != "allow":
            self._record(hub, "denied-timeout" if verdict == "timeout" else "denied")
            refusal = (
                LOCAL_TIMEOUT_REFUSAL if verdict == "timeout" else LOCAL_DENY_REFUSAL
            )
            return ToolResult.blocked(refusal)

        def execute() -> ToolResult:
            if not self._authority_is_valid(authority):
                return ToolResult.blocked(LOCAL_ROOT_CHANGED_REFUSAL)
            if not self._local_tools_are_enabled() or self._kill_switch_engaged():
                return ToolResult.blocked(LOCAL_KILL_SWITCH_REFUSAL)
            try:
                registry = (
                    self._registry
                    if authority is None
                    else self._registries_by_alias[authority.alias]
                )
                content = registry.execute(command, argv)
                redaction_root = (
                    self._result_redaction_root if authority is None else authority.root
                )
                content = redact_root_locator(content, redaction_root)
                return ToolResult(ok=True, content=_sanitize_result(content))
            except WorkspaceToolExecutionError as exc:
                if exc.code == "root_pin_failed":
                    return ToolResult.blocked(LOCAL_ROOT_CHANGED_REFUSAL)
                if exc.code not in {"invalid_request", "tool_failure"}:
                    return ToolResult.blocked(LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL)
                error = redact_root_locator(
                    str(exc),
                    self._result_redaction_root
                    if authority is None
                    else authority.root,
                )
                return ToolResult(ok=False, error=error[:_MAX_ERROR_CHARS])
            except Exception as exc:  # noqa: BLE001 - provider boundary
                error = redact_root_locator(
                    str(exc) or repr(exc),
                    self._result_redaction_root
                    if authority is None
                    else authority.root,
                )
                return ToolResult(ok=False, error=error[:_MAX_ERROR_CHARS])

        scope_factory = (
            self._authority_scope if authority is None else authority.authority_scope
        )
        scope = scope_factory() if scope_factory else nullcontext()
        try:
            with scope:
                return execute()
        except Exception:
            return ToolResult.blocked(LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL)

    def _ask_verdict(self, hub: HubTool, command: str, args: dict) -> str:
        stamp = self._pop_stamp(current_run_id(), command)
        if stamp in {"approve_once", "approve_session", "always_allow"}:
            if stamp != "approve_once":
                self._persist(hub, stamp)
            return "allow"
        if stamp in {"deny", "timeout"}:
            return stamp
        if self._session_approved(hub):
            return "allow"
        if self._approval_callback is None:
            return "timeout"
        pending = self.pending_gate_for(ToolCall(VIRTUAL_CLI_TOOL_NAME, args))
        if pending is None:
            return "timeout"
        try:
            decisions = self._approval_callback([pending]) or {}
        except Exception:
            return "timeout"
        decision = decisions.get(pending.call_id or pending.llm_name, "timeout")
        if decision in {"approve_session", "always_allow"}:
            self._persist(hub, decision)
        return (
            "allow"
            if decision in {"approve_once", "approve_session", "always_allow"}
            else decision
        )

    def _root_is_valid(self) -> bool:
        if self._root_guard is None:
            return True
        try:
            return bool(self._root_guard())
        except Exception:
            return False

    def _local_tools_are_enabled(self) -> bool:
        try:
            return bool(self._local_tools_enabled())
        except Exception:
            return False

    def _kill_switch_engaged(self) -> bool:
        try:
            return bool(self._kill_switch())
        except Exception:
            return True

    def _session_approved(self, hub: HubTool) -> bool:
        if self._is_session_approved is None:
            return False
        try:
            return bool(self._is_session_approved(hub))
        except Exception:
            return False

    def _persist(self, hub: HubTool, decision: str) -> None:
        if self._persist_approval is None:
            return
        try:
            self._persist_approval(hub, decision)
        except Exception as exc:
            logger.warning(
                "Virtual CLI approval persistence failed (exception_type={})",
                type(exc).__name__,
            )

    def _record(self, hub: HubTool, decision: str) -> None:
        if self._record_decision is None:
            return
        try:
            self._record_decision(hub, decision)
        except Exception as exc:
            logger.warning(
                "Virtual CLI decision audit failed (exception_type={})",
                type(exc).__name__,
            )
