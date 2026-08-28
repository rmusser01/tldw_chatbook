"""Ask-only model adapter for Chatbook's explicitly armed raw shell runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.raw_cli_executor import (
    MAX_RAW_COMMAND_BYTES,
    MAX_RAW_TIMEOUT_SECONDS,
    RawCliRequest,
    RawCliResult,
    RawCliStreamEvent,
    validate_raw_cli_request,
)

from .agent_models import ToolCatalogEntry, ToolResult, ToolSchema

RAW_SHELL_TOOL_NAME = "shell_exec"
RAW_SHELL_SERVER_KEY = "local:__local__"
RAW_SHELL_SERVER_LABEL = "Raw CLI (unsafe host shell)"
SOURCE = "raw_shell"

RAW_SHELL_DENY_REFUSAL = "blocked by raw shell permissions (set to Off)"
RAW_SHELL_APPROVAL_REFUSAL = (
    "raw shell execution requires command-visible user approval; do not retry"
)

_ALLOWED_ARGUMENTS = frozenset(
    {"command", "shell", "initial_directory", "timeout_seconds"}
)
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


def resolve_raw_shell_state(
    effective: EffectiveToolState,
) -> Literal["ask", "deny"]:
    """Project every non-Off stored state to Ask.

    The permission store uses ``deny`` internally; canonical UI renderers label
    that state ``Off``. In particular, a stored or hand-edited ``allow`` never
    becomes silent model authority for a host shell.
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
    ) -> None:
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

    def catalog_enabled(self) -> bool:
        """Return whether all live gates currently permit schema discovery."""

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

    def hub_tool(self) -> HubTool:
        """Return the stable Ask/Off policy identity shown in Tools."""

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
        unexpected = set(args) - _ALLOWED_ARGUMENTS
        if unexpected:
            raise ValueError("unexpected shell_exec arguments")
        command = args.get("command")
        if not isinstance(command, str):
            raise ValueError("command must be a string")
        shell = args.get("shell", "auto")
        if not isinstance(shell, str) or shell not in _SHELLS:
            raise ValueError("shell must be auto, bash, powershell, or cmd")
        timeout = args.get("timeout_seconds", MAX_RAW_TIMEOUT_SECONDS)
        if "initial_directory" in args:
            raw_directory = args["initial_directory"]
            if not isinstance(raw_directory, str):
                raise ValueError("initial_directory must be a string")
            directory = Path(raw_directory)
        else:
            directory = Path(self._initial_directory())
        request = RawCliRequest(
            invocation_id=invocation_id,
            caller="model",
            command=command,
            shell=cast(Any, shell),
            initial_directory=directory,
            timeout_seconds=cast(Any, timeout),
            console_session_id=self.console_session_id,
        )
        return validate_raw_cli_request(request)

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[-1]
        if name != RAW_SHELL_TOOL_NAME:
            return ToolResult(ok=False, error=f"Unknown raw shell tool: {name}")
        try:
            self._validated_request(args)
        except (TypeError, ValueError, OSError) as exc:
            return ToolResult(ok=False, error=f"invalid shell_exec request: {exc}")
        try:
            state = resolve_raw_shell_state(self._resolve_state(self.hub_tool()))
        except Exception:
            return ToolResult.blocked(RAW_SHELL_DENY_REFUSAL)
        if state == "deny":
            return ToolResult.blocked(RAW_SHELL_DENY_REFUSAL)
        return ToolResult.blocked(RAW_SHELL_APPROVAL_REFUSAL)


__all__ = [
    "RAW_SHELL_SERVER_KEY",
    "RAW_SHELL_SERVER_LABEL",
    "RAW_SHELL_TOOL_NAME",
    "RawShellToolProvider",
    "resolve_raw_shell_state",
]
