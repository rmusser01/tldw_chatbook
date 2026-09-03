from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from tldw_chatbook.Agents.raw_shell_tool_provider import (
    RAW_SHELL_SERVER_KEY,
    RAW_SHELL_TOOL_NAME,
    RawShellToolProvider,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.raw_cli_executor import MAX_RAW_COMMAND_BYTES
from tldw_chatbook.Utils.input_validation import RawShellExecInput


class _RuntimeProbe:
    def __init__(self, *, permitted: bool = True, armed: bool = True) -> None:
        self.permitted = permitted
        self.armed = armed
        self.execute_calls: list[object] = []

    def execute(self, request, on_event):
        self.execute_calls.append((request, on_event))
        raise AssertionError("Slice 1 must not launch the raw executor")


def _provider(
    tmp_path: Path,
    *,
    runtime: _RuntimeProbe | None = None,
    state: EffectiveToolState | None = None,
    **kwargs,
) -> RawShellToolProvider:
    runtime = runtime or _RuntimeProbe()
    state = state or EffectiveToolState(state="ask", origin="global_default")
    return RawShellToolProvider(
        runtime=runtime,
        console_session_id="console-session",
        initial_directory=lambda: tmp_path,
        resolve_state=lambda _hub: state,
        local_tools_enabled=lambda: True,
        kill_switch=lambda: False,
        **kwargs,
    )


def test_provider_exposes_one_conditional_structured_schema(tmp_path: Path) -> None:
    provider = _provider(tmp_path)

    assert [(row.id, row.name) for row in provider.list_catalog()] == [
        ("raw_shell:shell_exec", RAW_SHELL_TOOL_NAME)
    ]
    schema = provider.load_schema("raw_shell:shell_exec")
    assert schema.name == "shell_exec"
    assert schema.parameters["required"] == ["command"]
    assert schema.parameters["additionalProperties"] is False
    properties = schema.parameters["properties"]
    assert properties["command"]["type"] == "string"
    assert properties["command"]["maxLength"] == MAX_RAW_COMMAND_BYTES
    assert properties["shell"]["enum"] == ["auto", "bash", "powershell", "cmd"]
    assert properties["shell"]["default"] == "auto"
    assert properties["initial_directory"]["type"] == "string"
    assert properties["timeout_seconds"]["maximum"] == 300


def test_shared_raw_shell_input_model_is_strict_and_bounded() -> None:
    validated = RawShellExecInput.model_validate(
        {
            "command": "printf hello",
            "shell": "bash",
            "initial_directory": "/tmp",
            "timeout_seconds": 17,
        }
    )
    assert validated.command == "printf hello"
    assert validated.shell == "bash"
    assert validated.initial_directory == "/tmp"
    assert validated.timeout_seconds == 17.0

    invalid_payloads = (
        {"command": 7},
        {"command": "pwd", "shell": "zsh"},
        {"command": "pwd", "initial_directory": None},
        {"command": "pwd", "timeout_seconds": True},
        {"command": "pwd", "unexpected": "value"},
    )
    for payload in invalid_payloads:
        with pytest.raises(PydanticValidationError):
            RawShellExecInput.model_validate(payload)


@pytest.mark.parametrize(
    ("permitted", "armed", "local_enabled", "blocked"),
    (
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ),
)
def test_catalog_is_absent_unless_every_runtime_gate_permits_it(
    tmp_path: Path,
    permitted: bool,
    armed: bool,
    local_enabled: bool,
    blocked: bool,
) -> None:
    provider = RawShellToolProvider(
        runtime=_RuntimeProbe(permitted=permitted, armed=armed),
        console_session_id="console-session",
        initial_directory=lambda: tmp_path,
        resolve_state=lambda _hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
        local_tools_enabled=lambda: local_enabled,
        kill_switch=lambda: blocked,
    )

    assert bool(provider.list_catalog()) is (
        permitted and armed and local_enabled and not blocked
    )


def test_raw_shell_hub_tool_is_always_projectable_and_process_tagged(
    tmp_path: Path,
) -> None:
    provider = _provider(
        tmp_path,
        runtime=_RuntimeProbe(permitted=False, armed=False),
    )

    hub = provider.hub_tool()

    assert hub.server_key == RAW_SHELL_SERVER_KEY
    assert hub.name == RAW_SHELL_TOOL_NAME
    assert hub.tags == ("process",)
    assert hub.executable is True


@pytest.mark.parametrize(
    "args",
    (
        {},
        {"command": 7},
        {"command": ""},
        {"command": "hello\x00world"},
        {"command": "x" * (MAX_RAW_COMMAND_BYTES + 1)},
        {"command": "pwd", "shell": "zsh"},
        {"command": "pwd", "shell": 1},
        {"command": "pwd", "initial_directory": "relative/path"},
        {"command": "pwd", "timeout_seconds": True},
        {"command": "pwd", "timeout_seconds": 0},
        {"command": "pwd", "timeout_seconds": 301},
        {"command": "pwd", "unexpected": "value"},
    ),
)
def test_invalid_arguments_fail_before_permission_or_launch(
    tmp_path: Path,
    args: dict,
) -> None:
    runtime = _RuntimeProbe()
    permission_reads: list[str] = []
    provider = RawShellToolProvider(
        runtime=runtime,
        console_session_id="console-session",
        initial_directory=lambda: tmp_path,
        resolve_state=lambda hub: permission_reads.append(hub.name)
        or EffectiveToolState(state="allow", origin="tool_override"),
        local_tools_enabled=lambda: True,
        kill_switch=lambda: False,
    )

    result = provider.invoke("raw_shell:shell_exec", args)

    assert result.ok is False
    assert "invalid shell_exec request" in result.error
    assert permission_reads == []
    assert runtime.execute_calls == []


def test_missing_initial_directory_fails_before_permission_or_launch(
    tmp_path: Path,
) -> None:
    runtime = _RuntimeProbe()
    permission_reads: list[str] = []
    provider = RawShellToolProvider(
        runtime=runtime,
        console_session_id="console-session",
        initial_directory=lambda: tmp_path,
        resolve_state=lambda hub: permission_reads.append(hub.name)
        or EffectiveToolState(state="ask", origin="global_default"),
        local_tools_enabled=lambda: True,
        kill_switch=lambda: False,
    )

    result = provider.invoke(
        "raw_shell:shell_exec",
        {"command": "pwd", "initial_directory": str(tmp_path / "missing")},
    )

    assert result.ok is False
    assert "invalid shell_exec request" in result.error
    assert permission_reads == []
    assert runtime.execute_calls == []


def test_schema_load_fails_closed_after_catalog_gate_changes(tmp_path: Path) -> None:
    runtime = _RuntimeProbe()
    provider = _provider(tmp_path, runtime=runtime)
    assert provider.load_schema("raw_shell:shell_exec").name == "shell_exec"

    runtime.armed = False

    with pytest.raises(KeyError):
        provider.load_schema("raw_shell:shell_exec")


def test_console_raw_shell_resolution_captures_the_exact_named_profile(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    class RecordingService:
        def __init__(self):
            self.calls = []

        def get_kill_switch(self):
            return False

        def gate_tool_test_for_profile(self, hub, profile_id):
            self.calls.append((hub.server_key, hub.name, profile_id))
            return EffectiveToolState(state="allow", origin="tool_override")

        def gate_tool_test(self, hub):
            self.calls.append((hub.server_key, hub.name, "default"))
            return EffectiveToolState(state="allow", origin="tool_override")

    class Runtime(_RuntimeProbe):
        def set_model_authority_revoker(self, callback):
            self.revoker = callback

    service = RecordingService()
    controller = object.__new__(ConsoleChatController)
    controller.app = SimpleNamespace(
        unified_mcp_service=service,
        raw_cli_runtime=Runtime(),
    )
    controller._agent_bridge = None
    context = SimpleNamespace(
        tool_configuration={"local_tools_enabled": True},
        tool_policy_profile_id="research",
    )

    provider, _review = controller._compose_raw_shell_provider(
        session_id="session-1", turn_context=context, project_root=tmp_path
    )
    state = provider._resolve_state(provider.hub_tool())

    assert state.state == "allow"
    assert service.calls == [
        (RAW_SHELL_SERVER_KEY, RAW_SHELL_TOOL_NAME, "research")
    ]
