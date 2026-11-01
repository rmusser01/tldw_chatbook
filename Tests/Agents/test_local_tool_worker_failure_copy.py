"""TASK-33940.2: a failed file-tool worker is not a missing scratch space.

Every executor failure code other than root_pin_failed, invalid_request and
tool_failure used to reach the model as "Private scratch space is
unavailable; the tool was not run." -- for an admitted WORKSPACE folder too,
pointing the agent at the wrong thing. Found live: a stale editable install
made every fs_* call fail with ``worker_crashed`` that way.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Agents.test_agent_runtime import fence, make_deps
from tldw_chatbook.Agents import local_tool_provider as ltp
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL,
    LOCAL_ROOT_CHANGED_REFUSAL,
    LocalToolProvider,
    RunAdmittedWorkspaceRoot,
)
from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutionError

_WORKER_CODES = (
    "worker_crashed",
    "worker_timed_out",
    "worker_failure",
    "spawn_failed",
    "protocol_failure",
    "containment_unavailable",
    "cleanup_unproven",
)

_ALLOW = EffectiveToolState(state="allow", origin="global_default")


class _FailingExecutor:
    def __init__(self, code: str) -> None:
        self.code = code
        self.calls: list[dict] = []

    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
        self.calls.append(dict(arguments))
        raise WorkspaceToolExecutionError(self.code)


@pytest.fixture(autouse=True)
def _no_config_reads(monkeypatch):
    # Same precedent as test_local_tool_provider.py: provider construction
    # reads the web-deep-search gate, which trips the sandboxed-config failure.
    monkeypatch.setattr(
        ltp, "get_cli_setting", lambda _section, _key=None, default=None: default
    )


def _admitted(root: Path, executor) -> RunAdmittedWorkspaceRoot:
    return RunAdmittedWorkspaceRoot(
        workspace_id="workspace-1",
        binding_id="folder-1",
        alias="folder-1",
        root=root,
        locator_fingerprint="fingerprint-folder-1",
        root_identity=((str(root), 1, 2, 0o40755),),
        allow_write=False,
        guard=lambda _write: True,
        workspace_executor=executor,
    )


@pytest.mark.parametrize("code", _WORKER_CODES)
def test_fs_worker_failure_on_a_workspace_folder_never_blames_scratch(
    tmp_path, code
) -> None:
    """An fs_* executor failure on an admitted workspace folder names the worker.

    Args:
        tmp_path: Directory standing in for the admitted workspace folder.
        code: The ``WorkspaceToolExecutionError`` code the worker raises.
    """
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: _ALLOW,
        admitted_roots=(_admitted(tmp_path, _FailingExecutor(code)),),
    )

    result = provider.invoke("fs_list", {"path": "."})

    assert not result.ok
    assert result.outcome is None
    assert result.dispatch_state == "uncertain"
    assert result.approval_decision is None
    assert "scratch" not in result.error.lower()
    assert "worker failed" in result.error


@pytest.mark.parametrize("code", _WORKER_CODES)
def test_virtual_cli_worker_failure_never_blames_scratch(tmp_path, code) -> None:
    """A virtual_cli executor failure names the worker, never scratch.

    Args:
        tmp_path: Directory standing in for the admitted workspace folder.
        code: The ``WorkspaceToolExecutionError`` code the worker raises.
    """
    provider = VirtualCliProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: _ALLOW,
        admitted_roots=(_admitted(tmp_path, _FailingExecutor(code)),),
    )

    result = provider.invoke("virtual_cli", {"command": "ls", "argv": ["."]})

    assert not result.ok
    assert result.outcome is None
    assert result.dispatch_state == "uncertain"
    assert result.approval_decision is None
    assert "scratch" not in result.error.lower()
    assert "worker failed" in result.error


@pytest.mark.parametrize("code", _WORKER_CODES)
@pytest.mark.parametrize("tool_name", ["fs_list", "fs_read", "virtual_cli"])
@pytest.mark.parametrize("native", [False, True])
def test_worker_failures_with_changing_paths_stop_before_another_model_call(
    tmp_path, code, tool_name, native
) -> None:
    """Worker failures count toward the existing limit independently of denials."""
    executor = _FailingExecutor(code)
    provider_class = (
        VirtualCliProvider if tool_name == "virtual_cli" else LocalToolProvider
    )
    provider = provider_class(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: _ALLOW,
        admitted_roots=(_admitted(tmp_path, executor),),
    )
    calls = [
        ToolCall(
            tool_name,
            {"command": "ls", "argv": [f"attempt-{index}"]}
            if tool_name == "virtual_cli"
            else {"path": f"attempt-{index}"},
            f"call-{index}",
        )
        for index in range(4)
    ]
    turns = [
        ModelTurn(
            tool_calls=(call,),
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.args),
                        },
                    }
                ],
            },
        )
        if native
        else ModelTurn(text=fence(call.name, call.args))
        for call in calls
    ] + [ModelTurn(text="must not request another model turn")]
    records = []
    deps = make_deps(turns, invoke=lambda call: provider.invoke(call.name, call.args))
    deps.on_record = lambda kind, payload: records.append((kind, payload))
    outcome = run_agent_loop(
        AgentConfig(
            model="fixture",
            system_prompt="",
            allowed_tools=(tool_name,),
            budget=RunBudget(max_steps=50, denial_circuit_breaker_limit=0),
        ),
        [],
        [provider.load_schema(tool_name)],
        deps,
    )

    assert outcome.status == "stuck"
    assert [arguments["path"] for arguments in executor.calls] == [
        "attempt-0",
        "attempt-1",
        "attempt-2",
    ]
    assert sum(step.kind == "model" for step in outcome.steps) == 3
    results = [step for step in outcome.steps if step.kind == "tool_result"]
    assert len(results) == 3
    assert all(step.tool_outcome == "failed" for step in results)
    assert "worker failed" in results[-1].result
    assert outcome.steps[-2] is results[-1]
    assert len([row for kind, row in records if kind == "tool_result"]) == 3
    assert "failed 3 times in a row" in outcome.steps[-1].summary
    assert not outcome.denial_count


@pytest.mark.parametrize("provider_class", [LocalToolProvider, VirtualCliProvider])
@pytest.mark.parametrize(
    "denied", [False, True], ids=["root-changed", "permission-off"]
)
def test_authority_and_permission_refusals_remain_blocked(
    tmp_path, provider_class, denied
) -> None:
    """Worker failure classification must not reclassify real policy refusals."""
    executor = _FailingExecutor("worker_crashed" if denied else "root_pin_failed")
    provider = provider_class(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: (
            EffectiveToolState(state="deny", origin="tool_override")
            if denied
            else _ALLOW
        ),
        admitted_roots=(_admitted(tmp_path, executor),),
    )
    virtual = provider_class is VirtualCliProvider
    result = provider.invoke(
        "virtual_cli" if virtual else "fs_list",
        {"command": "ls", "argv": ["."]} if virtual else {"path": "."},
    )

    assert not result.ok
    assert result.outcome == "blocked"
    assert result.approval_decision == ("denied" if denied else None)
    assert len(executor.calls) == (0 if denied else 1)
    if not denied:
        assert result.error == LOCAL_ROOT_CHANGED_REFUSAL


@pytest.mark.parametrize("code", _WORKER_CODES)
@pytest.mark.parametrize("virtual", [False, True], ids=["fs-write", "virtual-cli"])
def test_goal_stops_after_a_worker_failure_without_confirmed_effects(
    tmp_path, monkeypatch, code, virtual
) -> None:
    """Classifying execution failure must not allow unattended goal retries."""
    from tldw_chatbook.Agents import automatic_work_runtime

    automatic = SimpleNamespace(
        check=lambda: None,
        goal=SimpleNamespace(
            permits_call=lambda name: True,
            observe_tool_result=lambda name, args, content: content,
        ),
    )
    monkeypatch.setattr(
        automatic_work_runtime, "current_automatic_work", lambda: automatic
    )
    executor = _FailingExecutor(code)
    provider_class = VirtualCliProvider if virtual else LocalToolProvider
    provider = provider_class(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: _ALLOW,
        admitted_roots=(replace(_admitted(tmp_path, executor), allow_write=True),),
    )
    tool_name = "virtual_cli" if virtual else "fs_write"
    turns = [
        ModelTurn(
            text=fence(
                tool_name,
                {"command": "ls", "argv": [f"attempt-{index}"]}
                if virtual
                else {"path": f"attempt-{index}", "content": "write once"},
            )
        )
        for index in range(2)
    ] + [ModelTurn(text="must not continue after uncertain execution")]
    deps = make_deps(turns, invoke=lambda call: provider.invoke(call.name, call.args))
    outcome = run_agent_loop(
        AgentConfig(
            model="fixture",
            system_prompt="",
            allowed_tools=(tool_name,),
            budget=RunBudget(max_steps=50),
        ),
        [],
        [provider.load_schema(tool_name)],
        deps,
    )

    assert outcome.status == "stuck"
    assert outcome.termination_reason == "unknown_effect"
    assert len(executor.calls) == 1
    assert sum(step.kind == "model" for step in outcome.steps) == 1
    assert outcome.steps[-1].kind == "tool_result"
    assert outcome.steps[-1].tool_outcome == "failed"


def test_a_failed_scratch_lease_still_says_scratch_is_unavailable(tmp_path) -> None:
    """A genuine private-scratch lease failure keeps the scratch refusal.

    Args:
        tmp_path: Directory standing in for the chat's private scratch root.
    """

    @contextlib.contextmanager
    def failing_lease():
        raise RuntimeError("scratch generation retired")
        yield tmp_path  # pragma: no cover

    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: _ALLOW,
        authority_scope=failing_lease,
        admitted_roots=None,
    )

    result = provider.invoke("fs_list", {"path": "."})

    assert result.error == LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL
    assert result.outcome == "blocked"
    assert result.approval_decision is None
