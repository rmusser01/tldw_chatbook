import os
from io import BytesIO
from pathlib import Path

import pytest
from loguru import logger

import tldw_chatbook.Agents.virtual_cli_provider as virtual_cli_provider
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL,
    LOCAL_ROOT_CHANGED_REFUSAL,
)
from tldw_chatbook.Agents.run_context import (
    current_tool_call_id,
    use_run_id,
    use_tool_call_id,
)
from tldw_chatbook.Agents.virtual_cli_provider import (
    VIRTUAL_CLI_SERVER_KEY,
    VIRTUAL_CLI_TOOL_NAME,
    VirtualCliProvider,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState, definition_hash
from tldw_chatbook.Tools.virtual_cli_impls import VIRTUAL_CLI_COMMANDS
from tldw_chatbook.Tools.workspace_tool_executor import (
    WorkspaceToolExecutionError,
    WorkspaceToolExecutor,
)
from tldw_chatbook.Tools.workspace_tool_protocol import WorkspaceToolResponse
from tldw_chatbook.Tools.workspace_tool_worker import run_workspace_worker

ALLOW = EffectiveToolState(state="allow", origin="tool_override")
ASK = EffectiveToolState(state="ask", origin="global_default")
DENY = EffectiveToolState(state="deny", origin="tool_override")


class RecordingWorkspaceExecutor:
    def __init__(
        self,
        result: str = "1\thello",
        error: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.error_message = error_message
        self.calls: list[tuple[str, dict, str]] = []

    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
        self.calls.append((operation, dict(arguments), intent))
        if self.error is not None:
            raise WorkspaceToolExecutionError(self.error, self.error_message)
        return self.result


class InProcessWorkspaceExecutor:
    def __init__(self, workspace_root: Path) -> None:
        self._executor = WorkspaceToolExecutor(workspace_root)

    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
        request = self._executor._build_request(operation, arguments, intent=intent)
        stdout = BytesIO()
        run_workspace_worker(BytesIO(request.to_bytes()), stdout, BytesIO())
        response = WorkspaceToolResponse.from_bytes(
            stdout.getvalue().splitlines()[-1],
            expected_operation_id=request.operation_id,
        )
        if response.outcome != "success":
            raise WorkspaceToolExecutionError(response.code, response.error)
        return response.result or ""


def make_provider(tmp_path: Path, state=ASK, **kwargs) -> VirtualCliProvider:
    use_default_executor = kwargs.pop("use_default_executor", False)
    kwargs.setdefault("resolve_state", lambda _hub: state)
    kwargs.setdefault("local_tools_enabled", lambda: True)
    kwargs.setdefault("kill_switch", lambda: False)
    if not use_default_executor:
        kwargs.setdefault("workspace_executor", RecordingWorkspaceExecutor())
    return VirtualCliProvider(workspace_root=tmp_path, **kwargs)


def test_provider_exposes_one_structured_model_tool(tmp_path):
    provider = make_provider(tmp_path)

    assert [(item.id, item.name) for item in provider.list_catalog()] == [
        ("virtual_cli:virtual_cli", VIRTUAL_CLI_TOOL_NAME)
    ]
    schema = provider.load_schema("virtual_cli:virtual_cli")
    assert schema.name == "virtual_cli"
    assert schema.parameters["required"] == ["command", "argv"]
    assert schema.parameters["additionalProperties"] is False
    assert schema.parameters["properties"]["command"]["enum"] == list(
        VIRTUAL_CLI_COMMANDS
    )
    assert schema.parameters["properties"]["argv"]["type"] == "array"
    assert "shell" not in schema.parameters["properties"]
    assert "command_line" not in schema.parameters["properties"]


def test_provider_constructs_and_injects_real_executor_by_default(
    tmp_path, monkeypatch
):
    constructed: list[Path] = []

    class RecordingFactory(RecordingWorkspaceExecutor):
        def __init__(self, workspace_root: Path) -> None:
            constructed.append(workspace_root)
            super().__init__(result="leased-result")

    monkeypatch.setattr(
        virtual_cli_provider,
        "WorkspaceToolExecutor",
        RecordingFactory,
        raising=False,
    )

    result = make_provider(tmp_path, state=ALLOW, use_default_executor=True).invoke(
        "virtual_cli",
        {"command": "ls", "argv": ["."]},
    )

    assert result.ok and result.content == "leased-result"
    assert constructed == [tmp_path]


@pytest.mark.parametrize(
    ("code", "expected"),
    (
        ("root_pin_failed", LOCAL_ROOT_CHANGED_REFUSAL),
        ("containment_unavailable", LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL),
        ("protocol_failure", LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL),
        ("spawn_failed", LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL),
    ),
)
def test_virtual_cli_executor_boundary_failures_map_to_pinned_refusals(
    tmp_path, code, expected
):
    executor = RecordingWorkspaceExecutor(error=code)
    provider = make_provider(
        tmp_path,
        state=ALLOW,
        workspace_executor=executor,
    )

    result = provider.invoke(
        "virtual_cli",
        {"command": "ls", "argv": ["."]},
    )

    assert not result.ok and result.outcome == "blocked"
    assert result.error == expected
    assert executor.calls == [("fs_list", {"path": "."}, "read")]


def test_virtual_cli_domain_failure_text_is_redacted_and_bounded(tmp_path):
    executor = RecordingWorkspaceExecutor(
        error="invalid_request",
        error_message=f"bounded domain failure: {tmp_path}/marker " + ("x" * 400),
    )
    provider = make_provider(
        tmp_path,
        state=ALLOW,
        workspace_executor=executor,
        result_redaction_root=tmp_path,
    )

    result = provider.invoke(
        "virtual_cli",
        {"command": "ls", "argv": ["."]},
    )

    assert not result.ok and result.outcome is None
    assert str(tmp_path) not in result.error
    assert result.error.startswith("bounded domain failure: marker ")
    assert len(result.error) == 300


def test_virtual_cli_root_guard_refuses_as_root_drift_before_executor(tmp_path):
    executor = RecordingWorkspaceExecutor()
    provider = make_provider(
        tmp_path,
        state=ALLOW,
        workspace_executor=executor,
        root_guard=lambda: False,
    )

    result = provider.invoke(
        "virtual_cli",
        {"command": "ls", "argv": ["."]},
    )

    assert not result.ok and result.outcome == "blocked"
    assert result.error == LOCAL_ROOT_CHANGED_REFUSAL
    assert executor.calls == []


def test_virtual_cli_refuses_root_replaced_after_second_guard(tmp_path):
    locator = tmp_path / "workspace"
    locator.mkdir()
    (locator / "sentinel.txt").write_bytes(b"A_ONLY")
    retained = tmp_path / "retained-a"
    calls = 0

    def replace_after_second_guard() -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            os.replace(locator, retained)
            locator.mkdir()
            (locator / "sentinel.txt").write_bytes(b"B_BYTE_EXACT\x00\xff")
        return True

    provider = make_provider(
        locator,
        state=ALLOW,
        root_guard=replace_after_second_guard,
        use_default_executor=True,
    )

    result = provider.invoke(
        "virtual_cli", {"command": "cat", "argv": ["sentinel.txt"]}
    )

    assert calls == 2
    assert not result.ok and result.outcome == "blocked"
    assert result.error == LOCAL_ROOT_CHANGED_REFUSAL
    assert (locator / "sentinel.txt").read_bytes() == b"B_BYTE_EXACT\x00\xff"


def test_virtual_cli_preserves_bounded_domain_failure_text(tmp_path):
    provider = make_provider(
        tmp_path,
        state=ALLOW,
        workspace_executor=InProcessWorkspaceExecutor(tmp_path),
    )

    result = provider.invoke(
        "virtual_cli", {"command": "cat", "argv": ["missing.txt"]}
    )

    assert not result.ok and result.outcome is None
    assert result.error == "file not found: missing.txt"


def test_tool_call_id_context_is_nested_and_restored():
    assert current_tool_call_id() == ""
    with use_tool_call_id("outer"):
        assert current_tool_call_id() == "outer"
        with use_tool_call_id("inner"):
            assert current_tool_call_id() == "inner"
        assert current_tool_call_id() == "outer"
    assert current_tool_call_id() == ""


def test_provider_projects_ten_distinct_permission_tools(tmp_path):
    tools = make_provider(tmp_path).hub_tools()

    assert [tool.name for tool in tools] == list(VIRTUAL_CLI_COMMANDS)
    assert {tool.server_key for tool in tools} == {VIRTUAL_CLI_SERVER_KEY}
    hashes = {definition_hash(tool.description, tool.input_schema) for tool in tools}
    assert len(hashes) == len(VIRTUAL_CLI_COMMANDS)


def test_pending_gate_uses_selected_command_and_call_identity(tmp_path):
    provider = make_provider(tmp_path)
    call = ToolCall(
        "virtual_cli", {"command": "cat", "argv": ["README.md"]}, call_id="c2"
    )

    pending = provider.pending_gate_for(call)

    assert pending is not None
    assert pending.llm_name == "virtual_cli"
    assert pending.server_key == VIRTUAL_CLI_SERVER_KEY
    assert pending.tool_name == "cat"
    assert pending.call_id == "c2"
    assert pending.arguments == call.args


def test_missing_permission_is_ask_and_discoverability_does_not_execute(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("secret", encoding="utf-8")
    provider = make_provider(tmp_path)

    result = provider.invoke("virtual_cli", {"command": "cat", "argv": ["note.txt"]})

    assert not result.ok
    assert result.outcome == "blocked"
    assert "approve" in result.error.lower()


def test_one_command_permission_does_not_authorize_another(tmp_path):
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    states = {"cat": ALLOW, "ls": DENY}
    provider = make_provider(
        tmp_path, state=None, resolve_state=lambda hub: states[hub.name]
    )

    cat = provider.invoke("virtual_cli", {"command": "cat", "argv": ["note.txt"]})
    listing = provider.invoke("virtual_cli", {"command": "ls", "argv": ["."]})

    assert cat.ok and "hello" in cat.content
    assert not listing.ok and listing.outcome == "blocked"


def test_approve_once_stamp_is_call_scoped(tmp_path):
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    provider = make_provider(tmp_path)
    first = provider.pending_gate_for(
        ToolCall("virtual_cli", {"command": "cat", "argv": ["note.txt"]}, "c1")
    )
    second = provider.pending_gate_for(
        ToolCall("virtual_cli", {"command": "cat", "argv": ["note.txt"]}, "c2")
    )
    assert first is not None and second is not None
    provider.apply_batch_decisions("run", {"c1": "approve_once"}, [first, second])

    with use_run_id("run"), use_tool_call_id("c1"):
        allowed = provider.invoke(
            "virtual_cli", {"command": "cat", "argv": ["note.txt"]}
        )
    with use_run_id("run"), use_tool_call_id("c2"):
        blocked = provider.invoke(
            "virtual_cli", {"command": "cat", "argv": ["note.txt"]}
        )

    assert allowed.ok
    assert not blocked.ok and blocked.outcome == "blocked"


@pytest.mark.parametrize(
    "kwargs",
    (
        {"kill_switch": lambda: True},
        {"local_tools_enabled": lambda: False},
    ),
)
def test_invocation_rechecks_global_gates(tmp_path, kwargs):
    provider = make_provider(tmp_path, state=ALLOW, **kwargs)
    provider._registry.execute = lambda *_args: pytest.fail("must not dispatch")

    result = provider.invoke("virtual_cli", {"command": "ls", "argv": ["."]})

    assert not result.ok and result.outcome == "blocked"


def test_kill_switch_flip_before_core_dispatch_fails_closed(tmp_path):
    reads = iter((False, True))
    provider = make_provider(
        tmp_path,
        state=ALLOW,
        kill_switch=lambda: next(reads),
    )
    provider._registry.execute = lambda *_args: pytest.fail("must not dispatch")

    result = provider.invoke("virtual_cli", {"command": "ls", "argv": ["."]})

    assert not result.ok and result.outcome == "blocked"


def test_invalid_argv_fails_before_permission_resolution(tmp_path):
    resolves = []
    provider = make_provider(
        tmp_path,
        state=None,
        resolve_state=lambda hub: resolves.append(hub.name) or ALLOW,
    )

    result = provider.invoke(
        "virtual_cli", {"command": "git_diff", "argv": ["--not-a-flag"]}
    )

    assert not result.ok
    assert "invalid" in result.error.lower()
    assert resolves == []


def test_result_controls_are_sanitized_and_capped(tmp_path):
    provider = make_provider(tmp_path, state=ALLOW)
    provider._registry.execute = lambda *_args: "ok\x1b[31m\x07\n" + ("x" * 40000)

    result = provider.invoke("virtual_cli", {"command": "ls", "argv": ["."]})

    assert result.ok
    assert "\x1b" not in result.content and "\x07" not in result.content
    assert "\n" in result.content
    assert len(result.content.encode("utf-8")) <= 32 * 1024 + 20


def test_permission_is_rechecked_after_review(tmp_path):
    state = ASK
    provider = make_provider(tmp_path, state=None, resolve_state=lambda _hub: state)
    call = ToolCall("virtual_cli", {"command": "ls", "argv": ["."]}, "c1")
    pending = provider.pending_gate_for(call)
    assert pending is not None
    provider.apply_batch_decisions("run", {"c1": "approve_once"}, [pending])
    state = DENY

    with use_run_id("run"), use_tool_call_id("c1"):
        result = provider.invoke("virtual_cli", call.args)

    assert not result.ok and result.outcome == "blocked"


@pytest.mark.parametrize(
    ("callback_name", "callback_kwarg", "event_label"),
    [
        pytest.param(
            "_persist",
            "persist_approval",
            "Virtual CLI approval persistence failed",
            id="persist",
        ),
        pytest.param(
            "_record",
            "record_decision",
            "Virtual CLI decision audit failed",
            id="record",
        ),
    ],
)
def test_callback_failure_logs_only_safe_fixed_metadata(
    tmp_path, callback_name, callback_kwarg, event_label
):
    exception_sentinel = "/private/customer/secret-project/token=exception-sentinel"
    callback_input_sentinel = "/private/customer/secret-project/input-sentinel"
    traceback_local_sentinel = "/private/customer/secret-project/traceback-local"

    class DistinctiveCallbackFailure(RuntimeError):
        pass

    def fail_callback(*_args):
        traceback_local = traceback_local_sentinel
        assert traceback_local
        raise DistinctiveCallbackFailure(exception_sentinel)

    provider = make_provider(tmp_path, **{callback_kwarg: fail_callback})
    records = []
    sink_id = logger.add(records.append, level="WARNING", format="{message}")
    try:
        getattr(provider, callback_name)(
            provider.hub_tool_for("ls"), callback_input_sentinel
        )
    finally:
        logger.remove(sink_id)

    assert len(records) == 1
    record = records[0].record
    assert record["message"] == (
        f"{event_label} (exception_type=DistinctiveCallbackFailure)"
    )
    assert record["exception"] is None
    rendered = str(records[0])
    assert exception_sentinel not in rendered
    assert callback_input_sentinel not in rendered
    assert traceback_local_sentinel not in rendered
