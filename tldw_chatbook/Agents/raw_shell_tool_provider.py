"""Ask-only model adapter for Chatbook's explicitly armed raw shell runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
import threading
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import ValidationError as PydanticValidationError

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.raw_cli_executor import (
    RawCliHardlineViolation,
    MAX_RAW_COMMAND_BYTES,
    MAX_RAW_TIMEOUT_SECONDS,
    RawCliRequest,
    RawCliResult,
    RawCliStreamEvent,
    validate_raw_cli_request,
)
from tldw_chatbook.Utils.input_validation import RawShellExecInput

from .agent_models import ToolCall, ToolCatalogEntry, ToolResult, ToolSchema
from .mcp_tool_provider import MCPPendingCall
from .run_context import current_run_id, current_tool_call_id

RAW_SHELL_TOOL_NAME = "shell_exec"
RAW_SHELL_SERVER_KEY = "local:__local__"
RAW_SHELL_SERVER_LABEL = "Raw CLI (unsafe host shell)"
SOURCE = "raw_shell"

RAW_SHELL_DENY_REFUSAL = "blocked by raw shell permissions (set to Off)"
RAW_SHELL_APPROVAL_REFUSAL = (
    "raw shell execution requires command-visible user approval; do not retry"
)
RAW_SHELL_APPROVAL_WARNING = (
    "This command runs with the full authority of the OS user and is not "
    "workspace confined. The command and output may persist in a local log."
)
RAW_SHELL_SESSION_SCOPE_NOTICE = (
    "Allow all raw shell commands for this Console session covers future raw "
    "shell commands, not only this displayed command. It clears on Disarm or "
    "when Chatbook exits."
)
RAW_SHELL_APPROVAL_OPTIONS = ("approve_once", "approve_session", "deny")
_MAX_MODEL_RESULT_CHARS = 4000
RawShellProgressSink = Callable[
    [str, str, RawCliStreamEvent | RawCliResult], None
]

_SHELLS = ("auto", "bash", "powershell", "cmd")

_MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_RAW_COMMAND_BYTES,
            "description": (
                "Exact command for the selected host shell. This is not workspace "
                "confined and runs with the full authority of the OS user."
            ),
        },
        "shell": {
            "type": "string",
            "enum": list(_SHELLS),
            "default": "auto",
            "description": "Host shell selector.",
        },
        "initial_directory": {
            "type": "string",
            "description": "Optional absolute existing host directory.",
        },
        "timeout_seconds": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": MAX_RAW_TIMEOUT_SECONDS,
            "default": MAX_RAW_TIMEOUT_SECONDS,
        },
    },
    "required": ["command"],
    "additionalProperties": False,
}


class _RawCliRuntime(Protocol):
    @property
    def permitted(self) -> bool: ...

    @property
    def armed(self) -> bool: ...

    def execute(
        self,
        request: RawCliRequest,
        on_event: Callable[[RawCliStreamEvent], None],
        **kwargs: Any,
    ) -> RawCliResult: ...

    def grant_model_session(self, console_session_id: str) -> None: ...

    def model_session_granted(self, console_session_id: str) -> bool: ...

    def set_model_authority_revoker(
        self, callback: Callable[[], None] | None
    ) -> None: ...


def resolve_raw_shell_state(
    effective: EffectiveToolState,
) -> Literal["ask", "deny"]:
    """Project every non-Off stored state to Ask.

    The permission store uses ``deny`` internally; canonical UI renderers label
    that state ``Off``. In particular, a stored or hand-edited ``allow`` never
    becomes silent model authority for a host shell.

    Args:
        effective: Resolved permission-store state for the raw-shell tool.

    Returns:
        ``"deny"`` only for Off; otherwise ``"ask"``.
    """

    return "deny" if effective.state == "deny" else "ask"


class RawShellToolProvider:
    """Expose one conditional model schema and one always-projectable policy row."""

    def __init__(
        self,
        *,
        runtime: _RawCliRuntime,
        console_session_id: str,
        initial_directory: Callable[[], Path],
        resolve_state: Callable[[HubTool], EffectiveToolState] | None = None,
        local_tools_enabled: Callable[[], bool] = lambda: True,
        kill_switch: Callable[[], bool] = lambda: False,
        progress_sink: RawShellProgressSink | None = None,
    ) -> None:
        """Initialize one Console-session raw-shell provider.

        Args:
            runtime: Existing app-owned raw CLI runtime.
            console_session_id: Nonblank Console session identity.
            initial_directory: Resolver for the default execution directory.
            resolve_state: Resolver for the effective Ask/Off policy.
            local_tools_enabled: Live local-tools gate probe.
            kill_switch: Live global tool kill-switch probe.
            progress_sink: Optional bounded progress observer.

        Raises:
            ValueError: If ``console_session_id`` is blank.
            TypeError: If ``initial_directory`` is not callable.
        """
        if not isinstance(console_session_id, str) or not console_session_id.strip():
            raise ValueError("console_session_id must be a nonblank string")
        if not callable(initial_directory):
            raise TypeError("initial_directory must be callable")
        self._runtime = runtime
        self.console_session_id = console_session_id
        self._initial_directory = initial_directory
        self._resolve_state = resolve_state or (
            lambda _hub: EffectiveToolState(state="ask", origin="global_default")
        )
        self._local_tools_enabled = local_tools_enabled
        self._kill_switch = kill_switch
        self._progress_sink = progress_sink
        self._stamps: dict[tuple[str, str], str] = {}
        self._stamps_lock = threading.Lock()
        self._authority_generation = 0

    def catalog_enabled(self) -> bool:
        """Return whether all live gates currently permit schema discovery.

        Returns:
            ``True`` only while every raw-shell catalog gate is open.
        """

        try:
            return bool(
                self._runtime.permitted is True
                and self._runtime.armed is True
                and self._local_tools_enabled()
                and not self._kill_switch()
            )
        except Exception:
            return False

    def list_catalog(self) -> list[ToolCatalogEntry]:
        """List the raw-shell catalog row when every live gate is open.

        Returns:
            A one-entry catalog while available, otherwise an empty list.
        """
        if not self.catalog_enabled():
            return []
        return [
            ToolCatalogEntry(
                id=f"{SOURCE}:{RAW_SHELL_TOOL_NAME}",
                name=RAW_SHELL_TOOL_NAME,
                one_line_description=(
                    "Run one explicitly approved command in a real host shell."
                ),
                source=SOURCE,
            )
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        """Load the conditional raw-shell tool schema.

        Args:
            tool_id: Catalog id or tool name to load.

        Returns:
            The structured ``shell_exec`` schema.

        Raises:
            KeyError: If the id is unknown or a live gate has closed.
        """
        name = tool_id.split(":", 1)[-1]
        if name != RAW_SHELL_TOOL_NAME or not self.catalog_enabled():
            raise KeyError(f"Raw shell tool is unavailable: {tool_id}")
        return ToolSchema(
            id=tool_id,
            name=RAW_SHELL_TOOL_NAME,
            description=(
                "Run a non-interactive command in a real host shell with the full "
                "authority of the OS user. Every command requires visible approval "
                "unless the user grants this Console session temporary authority."
            ),
            parameters=_MODEL_SCHEMA,
        )

    @staticmethod
    def hub_tool() -> HubTool:
        """Return the stable Ask/Off policy identity shown in Tools.

        Returns:
            The always-projectable raw-shell Hub tool row.
        """

        return HubTool(
            server_key=RAW_SHELL_SERVER_KEY,
            server_label=RAW_SHELL_SERVER_LABEL,
            source="local",
            name=RAW_SHELL_TOOL_NAME,
            description=(
                "Unsafe real host-shell execution with full OS-user authority. "
                "Only Ask or Off is honored; Allow is always coerced to Ask."
            ),
            input_schema=_MODEL_SCHEMA,
            tags=("process",),
            stale=False,
            executable=True,
        )

    def _validated_request(
        self,
        args: Mapping[str, object],
        *,
        invocation_id: str = "validation",
    ) -> RawCliRequest:
        if not isinstance(args, Mapping):
            raise ValueError("arguments must be an object")
        try:
            validated = RawShellExecInput.model_validate(dict(args))
        except PydanticValidationError as exc:
            raise ValueError("arguments do not match the shell_exec schema") from exc
        directory = (
            Path(validated.initial_directory)
            if validated.initial_directory is not None
            else Path(self._initial_directory())
        )
        request = RawCliRequest(
            invocation_id=invocation_id,
            caller="model",
            command=validated.command,
            shell=validated.shell,
            initial_directory=directory,
            timeout_seconds=validated.timeout_seconds,
            console_session_id=self.console_session_id,
        )
        return validate_raw_cli_request(request)

    def pending_gate_for(self, call: ToolCall) -> MCPPendingCall | None:
        """Build one complete, independently addressable raw approval row.

        Args:
            call: Proposed model tool call.

        Returns:
            A command-visible approval row, or ``None`` when no review is
            required or the call is invalid or unavailable.
        """
        if call.name != RAW_SHELL_TOOL_NAME or not self.catalog_enabled():
            return None
        try:
            request = self._validated_request(
                call.args,
                invocation_id=call.call_id or "raw-shell-approval",
            )
            state = resolve_raw_shell_state(self._resolve_state(self.hub_tool()))
            session_granted = self._runtime.model_session_granted(
                self.console_session_id
            )
        except Exception:
            return None
        if state != "ask" or session_granted:
            return None
        return MCPPendingCall(
            llm_name=RAW_SHELL_TOOL_NAME,
            server_key=RAW_SHELL_SERVER_KEY,
            tool_name=RAW_SHELL_TOOL_NAME,
            server_label=RAW_SHELL_SERVER_LABEL,
            arguments={
                "command": request.command,
                "shell": request.shell,
                "initial_directory": str(request.initial_directory),
                "timeout_seconds": request.timeout_seconds,
            },
            reason="ask",
            options=RAW_SHELL_APPROVAL_OPTIONS,
            call_id=call.call_id,
            full_command=request.command,
            warning=RAW_SHELL_APPROVAL_WARNING,
            scope_notice=RAW_SHELL_SESSION_SCOPE_NOTICE,
        )

    def apply_batch_decisions(
        self,
        run_id: str,
        decisions: dict[str, str],
        pending: Sequence[MCPPendingCall] = (),
        *,
        authority_generation: int | None = None,
    ) -> None:
        """Replace this run's stamps and apply any session-wide grant.

        Args:
            run_id: Agent run that will consume the decisions.
            decisions: Approval decision keyed by normalized call id.
            pending: Approval rows shown to the user.
            authority_generation: Optional generation captured before review.
        """
        with self._stamps_lock:
            if (
                authority_generation is not None
                and authority_generation != self._authority_generation
            ):
                return
            self._stamps = {
                key: value for key, value in self._stamps.items() if key[0] != run_id
            }
            grant_session = False
            for row in pending:
                key = row.call_id or row.llm_name
                decision = decisions.get(key)
                if decision not in RAW_SHELL_APPROVAL_OPTIONS:
                    continue
                self._stamps[(run_id, key)] = decision
                if decision == "approve_session":
                    grant_session = True
            if not grant_session:
                return
            try:
                state = resolve_raw_shell_state(self._resolve_state(self.hub_tool()))
                if self.catalog_enabled() and state == "ask":
                    # Keep the provider lock through the runtime write. Disarm
                    # closes the runtime first and then waits on this same lock
                    # to advance the generation, so an old approval can land
                    # either wholly before disarm or wholly before revocation,
                    # never after a completed disarm/re-arm cycle.
                    self._runtime.grant_model_session(self.console_session_id)
            except Exception:
                return

    @property
    def authority_generation(self) -> int:
        """Return the generation that approval round trips must still match.

        Returns:
            Current raw-shell authority generation.
        """
        with self._stamps_lock:
            return self._authority_generation

    def revoke_approval_stamps(self) -> int:
        """Invalidate every pending stamp, including hidden scope snapshots.

        Returns:
            Number of approval stamps removed.
        """
        with self._stamps_lock:
            revoked = len(self._stamps)
            self._stamps.clear()
            self._authority_generation += 1
            return revoked

    def _pop_stamp(self, run_id: str, fallback: str) -> str | None:
        key = current_tool_call_id() or fallback
        with self._stamps_lock:
            stamp = self._stamps.pop((run_id, key), None)
            if stamp is None and key != fallback:
                stamp = self._stamps.pop((run_id, fallback), None)
            return stamp

    @contextmanager
    def stamp_scope(self, run_id: str) -> Iterator[None]:
        """Hide and restore this run's approvals around a nested child run.

        Args:
            run_id: Parent run whose stamps must be isolated.

        Yields:
            Control while the parent run's stamps are hidden.
        """
        with self._stamps_lock:
            authority_generation = self._authority_generation
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
                if authority_generation == self._authority_generation:
                    self._stamps.update(saved)

    def invoke(self, tool_id: str, args: Mapping[str, object]) -> ToolResult:
        """Validate, authorize, and execute one model raw-shell call.

        Args:
            tool_id: Catalog id or tool name being invoked.
            args: Untrusted model arguments for ``shell_exec``.

        Returns:
            A bounded ordinary tool result or a fail-closed refusal.
        """
        name = tool_id.split(":", 1)[-1]
        if name != RAW_SHELL_TOOL_NAME:
            return ToolResult(ok=False, error=f"Unknown raw shell tool: {name}")
        run_id = current_run_id()
        call_id = current_tool_call_id() or RAW_SHELL_TOOL_NAME
        try:
            request = self._validated_request(
                args,
                invocation_id=current_tool_call_id() or f"raw-shell-{uuid4()}",
            )
        except RawCliHardlineViolation as exc:
            # TASK-25905 AC#4: the floor's refusal is ITS OWN thing -- it
            # names the rule, states it is not a user denial, and no
            # approval option (session grants included) can clear it.
            return ToolResult.blocked(
                f"{exc} — this is the built-in safety floor, "
                "not a user denial; it cannot be approved or overridden"
            )
        except (TypeError, ValueError, OSError) as exc:
            return ToolResult(ok=False, error=f"invalid shell_exec request: {exc}")
        stamp = self._pop_stamp(run_id, RAW_SHELL_TOOL_NAME)
        if not self.catalog_enabled():
            return ToolResult.blocked(RAW_SHELL_DENY_REFUSAL)
        try:
            state = resolve_raw_shell_state(self._resolve_state(self.hub_tool()))
        except Exception:
            return ToolResult.blocked(RAW_SHELL_DENY_REFUSAL)
        if state == "deny":
            return ToolResult.blocked(RAW_SHELL_DENY_REFUSAL)
        if stamp in {"deny", "timeout"}:
            return ToolResult.blocked(RAW_SHELL_APPROVAL_REFUSAL)
        approved = stamp in {"approve_once", "approve_session"}
        if not approved:
            try:
                approved = self._runtime.model_session_granted(
                    self.console_session_id
                )
            except Exception:
                approved = False
        if not approved:
            return ToolResult.blocked(RAW_SHELL_APPROVAL_REFUSAL)

        # Final provider-side recheck immediately before handing the validated
        # request to RawCliRuntime. The runtime performs its own lock-protected
        # permission/arm recheck again at worker admission.
        if not self.catalog_enabled():
            return ToolResult.blocked(RAW_SHELL_DENY_REFUSAL)
        try:
            if resolve_raw_shell_state(self._resolve_state(self.hub_tool())) == "deny":
                return ToolResult.blocked(RAW_SHELL_DENY_REFUSAL)
            result = self._runtime.execute(
                request,
                lambda event: self._emit_progress(run_id, call_id, event),
            )
            self._emit_progress(run_id, call_id, result)
        except Exception as exc:
            detail = (str(exc) or type(exc).__name__)[:300]
            return ToolResult(
                ok=False,
                error=f"raw shell execution failed: {detail}",
                outcome="failed",
            )
        return self._tool_result(result)

    def _emit_progress(
        self,
        run_id: str,
        call_id: str,
        event: RawCliStreamEvent | RawCliResult,
    ) -> None:
        """Forward bounded session-only progress without affecting execution."""
        sink = self._progress_sink
        if sink is None:
            return
        try:
            sink(run_id, call_id, event)
        except Exception:
            return

    @staticmethod
    def _tool_result(result: RawCliResult) -> ToolResult:
        """Map one executor settlement to the ordinary bounded tool contract."""
        exit_code = "none" if result.exit_code is None else str(result.exit_code)
        detail = (
            f"terminal_state: {result.terminal_state}\n"
            f"resolved_shell: {result.resolved_shell}\n"
            f"initial_directory: {result.initial_directory}\n"
            f"elapsed_seconds: {result.elapsed_seconds:.3f}\n"
            f"exit_code: {exit_code}\n"
            f"truncated: {str(result.truncated).lower()}\n"
            f"cleanup_proven: {str(result.cleanup_proven).lower()}\n"
            f"stdout:\n{result.stdout_preview or '(no output)'}\n"
            f"stderr:\n{result.stderr_preview or '(no output)'}"
        )[:_MAX_MODEL_RESULT_CHARS]
        if result.terminal_state == "exited" and result.exit_code == 0:
            return ToolResult(ok=True, content=detail)
        if result.terminal_state == "timed_out":
            outcome = "timeout"
        elif result.terminal_state == "cancelled":
            outcome = "cancelled"
        elif result.terminal_state == "refused":
            outcome = "blocked"
        else:
            outcome = "failed"
        return ToolResult(ok=False, error=detail, outcome=outcome)


__all__ = [
    "RAW_SHELL_SERVER_KEY",
    "RAW_SHELL_SERVER_LABEL",
    "RAW_SHELL_TOOL_NAME",
    "RawShellProgressSink",
    "RawShellToolProvider",
    "resolve_raw_shell_state",
]
