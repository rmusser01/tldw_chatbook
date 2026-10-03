"""TASK-33940.2: a failed file-tool worker is not a missing scratch space.

Every executor failure code other than root_pin_failed, invalid_request and
tool_failure used to reach the model as "Private scratch space is
unavailable; the tool was not run." -- for an admitted WORKSPACE folder too,
pointing the agent at the wrong thing. Found live: a stale editable install
made every fs_* call fail with ``worker_crashed`` that way.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from tldw_chatbook.Agents import local_tool_provider as ltp
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL,
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

    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
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
    assert "scratch" not in result.error.lower()
    assert "worker failed" in result.error


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
