from __future__ import annotations

import io
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Tools.workspace_tool_executor import (
    DIAGNOSTIC_STDERR_MAX_BYTES,
    WorkspaceToolExecutionError,
    WorkspaceToolExecutor,
    workspace_worker_environment,
)
from tldw_chatbook.Tools.workspace_tool_protocol import (
    MAX_RESPONSE_BYTES,
    WorkspaceToolRequest,
    WorkspaceToolResponse,
)
from tldw_chatbook.Tools.workspace_tool_worker import run_workspace_worker
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain


_OPERATION_ID = "fixed-operation-id"


class _RecordingInput(io.BytesIO):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events
        self.was_closed = False

    def write(self, value: bytes) -> int:
        self._events.append("stdin-write")
        return super().write(value)

    def close(self) -> None:
        self.was_closed = True
        self._events.append("stdin-close")


class _FakePopen:
    def __init__(
        self,
        response: bytes,
        stderr: bytes,
        events: list[str],
        *,
        returncode: int = 0,
        timeout: bool = False,
    ) -> None:
        self.pid = 7321
        self.stdin = _RecordingInput(events)
        self.stdout = io.BytesIO(response)
        self.stderr = io.BytesIO(stderr)
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._timeout = timeout
        self._events = events

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._events.append("wait")
        if self._timeout:
            raise subprocess.TimeoutExpired("fixed-worker", timeout)
        self.returncode = self._final_returncode
        return self._final_returncode

    def terminate(self) -> None:
        self._events.append("terminate")
        self.returncode = -15

    def kill(self) -> None:
        self._events.append("kill")
        self.returncode = -9


class _FakeTree:
    cleanup_proven = True
    instances: list[_FakeTree] = []

    def __init__(self, process: Any, admission: Any, identity: Any) -> None:
        self.process = process
        self.admission = admission
        self.identity = identity
        self.events = process._process._events
        type(self).instances.append(self)

    def admit(self) -> None:
        self.events.append("admit")

    def close(self) -> bool:
        self.events.append("close-tree")
        return type(self).cleanup_proven

    def terminate_tree(self, **_kwargs: Any) -> bool:
        self.events.append("terminate-tree")
        self.process._process.returncode = -9
        return type(self).cleanup_proven


def _response(
    *,
    outcome: str = "success",
    code: str = "ok",
    result: str | None = "RESULT",
    error: str | None = None,
) -> bytes:
    admitted = WorkspaceToolResponse(
        operation_id=_OPERATION_ID,
        outcome="admitted",
        code="root_pinned",
        result=None,
        error=None,
        elapsed_ms=0,
        truncated=False,
        cleanup_proven=True,
    ).to_bytes()
    terminal = WorkspaceToolResponse(
        operation_id=_OPERATION_ID,
        outcome=outcome,  # type: ignore[arg-type]
        code=code,
        result=result,
        error=error,
        elapsed_ms=1,
        truncated=False,
        cleanup_proven=True,
    ).to_bytes()
    return admitted + b"\n" + terminal + b"\n"


def _install_fake_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    response: bytes | None = None,
    stderr: bytes = b"",
    returncode: int = 0,
    timeout: bool = False,
) -> tuple[WorkspaceToolExecutor, dict[str, Any], list[str], _FakePopen]:
    root = tmp_path / "private-root-marker"
    root.mkdir()
    (root / "private-path-marker.txt").write_text("payload", encoding="utf-8")
    events: list[str] = []
    process = _FakePopen(
        response if response is not None else _response(),
        stderr,
        events,
        returncode=returncode,
        timeout=timeout,
    )
    captured: dict[str, Any] = {}

    def fake_popen(argv: list[str], **kwargs: Any) -> _FakePopen:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.ExecutorProcessTree", _FakeTree
    )
    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.uuid.uuid4",
        lambda: SimpleNamespace(hex=_OPERATION_ID),
    )
    _FakeTree.cleanup_proven = True
    _FakeTree.instances.clear()
    return WorkspaceToolExecutor(root), captured, events, process


def test_executor_uses_fixed_private_launch_and_admits_before_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PRIVATE_ENV_MARKER", "private-environment-value")
    executor, captured, events, process = _install_fake_launch(monkeypatch, tmp_path)

    with caplog.at_level(logging.DEBUG):
        result = executor.execute(
            "stat_path",
            {"path": "private-path-marker.txt"},
            intent="read",
        )

    assert result == "RESULT"
    assert captured["argv"] == [
        sys.executable,
        "-I",
        "-m",
        "tldw_chatbook.Tools.workspace_tool_worker",
    ]
    kwargs = captured["kwargs"]
    assert kwargs["shell"] is False
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["start_new_session"] is (sys.platform != "win32")
    assert "PRIVATE_ENV_MARKER" not in kwargs["env"]
    assert "private-root-marker" not in repr(captured)
    assert "private-path-marker" not in repr(captured)
    assert events.index("admit") < events.index("stdin-write")
    request = WorkspaceToolRequest.from_bytes(process.stdin.getvalue())
    assert request.arguments == {"path": "private-path-marker.txt"}
    assert process.stdin.was_closed
    diagnostic_text = caplog.text
    assert "private-root-marker" not in diagnostic_text
    assert "private-path-marker" not in diagnostic_text
    assert "private-environment-value" not in diagnostic_text


def test_worker_environment_is_a_small_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "private-key-marker")
    monkeypatch.setenv("HTTP_PROXY", "private-proxy-marker")
    monkeypatch.setenv("PYTHONPATH", "private-python-marker")

    environment = workspace_worker_environment()

    assert set(environment) <= {
        "PATH",
        "LANG",
        "LC_ALL",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
    }
    assert "private" not in repr(environment)


def test_spawn_exception_details_are_not_retained_in_the_public_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "private-root-marker"
    root.mkdir()

    def refuse_spawn(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("private-exception-marker")

    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.subprocess.Popen", refuse_spawn
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(WorkspaceToolExecutionError) as caught:
            WorkspaceToolExecutor(root).execute(
                "stat_path", {"path": "."}, intent="read"
            )

    assert caught.value.code == "spawn_failed"
    assert caught.value.__cause__ is None
    assert "private-exception-marker" not in repr(caught.value)
    assert "private-exception-marker" not in caplog.text


def test_worker_pins_and_dispatches_one_real_stat_request(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "note.txt").write_text("hello", encoding="utf-8")
    chain = capture_directory_chain(root)
    request = WorkspaceToolRequest(
        operation_id="worker-stat",
        operation="stat_path",
        intent="read",
        root_locator=chain.canonical_root,
        root_identity=chain.identities[0],
        ancestor_identities=chain.identities,
        arguments={"path": "note.txt"},
        timeout_seconds=300,
        output_max_bytes=MAX_RESPONSE_BYTES,
    )
    stdout = io.BytesIO()

    exit_code = run_workspace_worker(io.BytesIO(request.to_bytes()), stdout, io.BytesIO())

    frames = [WorkspaceToolResponse.from_bytes(line) for line in stdout.getvalue().splitlines()]
    assert exit_code == 0
    assert [frame.outcome for frame in frames] == ["admitted", "success"]
    assert frames[1].code == "ok"
    assert frames[1].result is not None
    assert "path: note.txt" in frames[1].result
    assert "size: 5" in frames[1].result


@pytest.mark.parametrize(
    ("response", "code"),
    [
        pytest.param(b"not-json\n", "protocol_failure", id="malformed"),
        pytest.param(b"x" * (MAX_RESPONSE_BYTES + 1), "protocol_failure", id="oversized"),
        pytest.param(
            _response() + _response().splitlines()[-1] + b"\n",
            "protocol_failure",
            id="duplicate-terminal",
        ),
    ],
)
def test_malformed_oversized_or_duplicate_worker_output_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: bytes,
    code: str,
) -> None:
    executor, _captured, _events, _process = _install_fake_launch(
        monkeypatch,
        tmp_path,
        response=response,
    )

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        executor.execute("stat_path", {"path": "private-path-marker.txt"}, intent="read")

    assert caught.value.code == code


def test_timeout_terminates_the_tree_and_returns_no_in_process_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _captured, events, _process = _install_fake_launch(
        monkeypatch,
        tmp_path,
        timeout=True,
    )

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        executor.execute("stat_path", {"path": "private-path-marker.txt"}, intent="read")

    assert caught.value.code == "worker_timed_out"
    assert "terminate-tree" in events


def test_crash_and_bounded_stderr_return_only_fixed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stderr_marker = b"private-stderr-marker"
    executor, _captured, _events, _process = _install_fake_launch(
        monkeypatch,
        tmp_path,
        response=b"",
        stderr=stderr_marker * (DIAGNOSTIC_STDERR_MAX_BYTES + 10),
        returncode=23,
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(WorkspaceToolExecutionError) as caught:
            executor.execute(
                "stat_path", {"path": "private-path-marker.txt"}, intent="read"
            )

    assert caught.value.code == "worker_crashed"
    assert "private-stderr-marker" not in str(caught.value)
    assert "private-stderr-marker" not in caplog.text


def test_unproven_cleanup_refuses_an_otherwise_successful_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _captured, _events, _process = _install_fake_launch(monkeypatch, tmp_path)
    _FakeTree.cleanup_proven = False

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        executor.execute("stat_path", {"path": "private-path-marker.txt"}, intent="read")

    assert caught.value.code == "cleanup_unproven"


def test_unsupported_closed_operation_is_a_stable_worker_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _captured, _events, _process = _install_fake_launch(
        monkeypatch,
        tmp_path,
        response=_response(
            outcome="failure",
            code="unsupported_operation",
            result=None,
            error="workspace operation is not implemented",
        ),
    )

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        executor.execute("fs_list", {"path": "."}, intent="read")

    assert caught.value.code == "unsupported_operation"
