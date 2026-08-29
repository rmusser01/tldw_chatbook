from __future__ import annotations

import io
import json
import logging
import multiprocessing
import os
import site
import subprocess
import sys
import threading
import time
import venv
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
from tldw_chatbook.Tools.workspace_root_pin import (
    WorkspaceRootPinError,
    pin_workspace_root,
)
from tldw_chatbook.Tools.workspace_tool_dispatch import (
    WorkspaceToolDispatchError,
    execute_pinned_operation,
)
from tldw_chatbook.Tools.workspace_tool_protocol import (
    MAX_RESPONSE_BYTES,
    WorkspaceToolRequest,
    WorkspaceToolResponse,
)
from tldw_chatbook.Tools.workspace_tool_worker import run_workspace_worker
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain


_OPERATION_ID = "fixed-operation-id"


READ_CASES = (
    ("fs_list", {"path": "."}, "A_ONLY"),
    ("fs_read", {"path": "sentinel.txt", "offset": 1}, "A_ONLY"),
    ("fs_glob", {"pattern": "**/*.txt", "max_results": 100}, "sentinel.txt"),
    (
        "fs_grep",
        {"pattern": "A_ONLY", "mode": "content", "max_results": 100},
        "A_ONLY",
    ),
)


def _post_pin_read_operation_child(
    locator: str,
    chain: Any,
    operation: str,
    arguments: dict[str, Any],
    ready: Any,
    resume: Any,
    output: Any,
) -> None:
    """Execute one read only after the parent attempts a root replacement."""
    try:
        with pin_workspace_root(Path(locator), chain) as root:
            ready.set()
            if not resume.wait(5):
                raise RuntimeError("test barrier timed out")
            request = WorkspaceToolRequest(
                operation_id="post-pin-read",
                operation=operation,  # type: ignore[arg-type]
                intent="read",
                root_locator=chain.canonical_root,
                root_identity=chain.identities[0],
                ancestor_identities=chain.identities,
                arguments=arguments,
                timeout_seconds=300,
                output_max_bytes=MAX_RESPONSE_BYTES,
            )
            try:
                output.put(("result", execute_pinned_operation(request, root)))
            except WorkspaceToolDispatchError as error:
                output.put(("refused", error.code))
    except WorkspaceRootPinError:
        output.put(("refused", "root_pin_failed"))
    except BaseException as error:
        output.put(("error", type(error).__name__))


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
        wait_error: BaseException | None = None,
    ) -> None:
        self.pid = 7321
        self.stdin = _RecordingInput(events)
        self.stdout = io.BytesIO(response)
        self.stderr = io.BytesIO(stderr)
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._timeout = timeout
        self._wait_error = wait_error
        self._events = events

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._events.append("wait")
        if self._wait_error is not None:
            raise self._wait_error
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
        if self.process._process.returncode is None:
            self.process._process.returncode = -9
        return type(self).cleanup_proven


def _response(
    *,
    outcome: str = "success",
    code: str = "ok",
    result: str | None = "RESULT",
    error: str | None = None,
    admitted_overrides: dict[str, Any] | None = None,
) -> bytes:
    admitted_values: dict[str, Any] = {
        "operation_id": _OPERATION_ID,
        "outcome": "admitted",
        "code": "root_pinned",
        "result": None,
        "error": None,
        "elapsed_ms": 0,
        "truncated": False,
        "cleanup_proven": True,
    }
    admitted_values.update(admitted_overrides or {})
    admitted = WorkspaceToolResponse(
        **admitted_values,
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
    wait_error: BaseException | None = None,
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
        wait_error=wait_error,
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


@pytest.mark.parametrize("path", [r"\outside.txt", "C:outside.txt"])
def test_worker_dispatch_refuses_cross_platform_rooted_stat_paths(
    tmp_path: Path,
    path: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    chain = capture_directory_chain(root)
    request = WorkspaceToolRequest(
        operation_id="worker-rooted-stat",
        operation="stat_path",
        intent="read",
        root_locator=chain.canonical_root,
        root_identity=chain.identities[0],
        ancestor_identities=chain.identities,
        arguments={"path": path},
        timeout_seconds=300,
        output_max_bytes=MAX_RESPONSE_BYTES,
    )
    stdout = io.BytesIO()

    exit_code = run_workspace_worker(io.BytesIO(request.to_bytes()), stdout, io.BytesIO())

    frames = [WorkspaceToolResponse.from_bytes(line) for line in stdout.getvalue().splitlines()]
    assert exit_code == 2
    assert [frame.outcome for frame in frames] == ["admitted", "failure"]
    assert frames[1].code == "invalid_request"
    assert path not in (frames[1].error or "")


def test_real_isolated_subprocess_executes_this_worktree_vertical_slice(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment_root = tmp_path / "isolated-runtime"
    # Symlinking preserves the signed macOS interpreter; copying it can make the
    # temporary launcher abort before Python starts. Windows ignores this flag.
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment_root)
    runtime_python = environment_root / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    try:
        site_query = subprocess.run(
            [
                str(runtime_python),
                "-I",
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("isolated runtime site query timed out", pytrace=False)
    except subprocess.CalledProcessError:
        pytest.fail("isolated runtime site query failed", pytrace=False)
    isolated_site_packages = Path(site_query.stdout.strip())
    dependency_paths = [
        Path(value).resolve()
        for value in site.getsitepackages()
        if Path(value).is_dir()
    ]
    (isolated_site_packages / "task2-worktree.pth").write_text(
        "\n".join(str(path) for path in (repository_root, *dependency_paths)) + "\n",
        encoding="utf-8",
    )

    workspace = tmp_path / "real-workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello", encoding="utf-8")
    harness = """
import json
import sys
from pathlib import Path
from tldw_chatbook.Tools import workspace_tool_worker
from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor

expected_root = Path(sys.argv[1]).resolve()
worker_source = Path(workspace_tool_worker.__file__).resolve()
if not worker_source.is_relative_to(expected_root):
    raise RuntimeError("wrong checkout imported")
result = WorkspaceToolExecutor(Path(sys.argv[2])).execute(
    "stat_path", {"path": "note.txt"}, intent="read"
)
print(json.dumps({"worker_source": str(worker_source), "result": result}))
"""
    completed = subprocess.run(
        [
            str(runtime_python),
            "-I",
            "-c",
            harness,
            str(repository_root),
            str(workspace),
        ],
        env={"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert Path(payload["worker_source"]).is_relative_to(repository_root)
    assert "path: note.txt" in payload["result"]
    assert "size: 5" in payload["result"]


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


@pytest.mark.parametrize(
    "admitted_overrides",
    [
        pytest.param({"operation_id": "wrong-operation-id"}, id="wrong-operation-id"),
        pytest.param({"outcome": "success"}, id="wrong-outcome"),
        pytest.param({"code": "not_pinned"}, id="wrong-code"),
        pytest.param({"result": "private-result"}, id="result-present"),
        pytest.param({"error": "private-error"}, id="error-present"),
        pytest.param({"truncated": True}, id="truncated"),
        pytest.param({"cleanup_proven": False}, id="cleanup-unproven"),
        pytest.param({"elapsed_ms": 2}, id="elapsed-after-terminal"),
    ],
)
def test_admitted_frame_requires_exact_content_free_root_pinned_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admitted_overrides: dict[str, Any],
) -> None:
    executor, _captured, _events, _process = _install_fake_launch(
        monkeypatch,
        tmp_path,
        response=_response(admitted_overrides=admitted_overrides),
    )

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        executor.execute("stat_path", {"path": "private-path-marker.txt"}, intent="read")

    assert caught.value.code == "protocol_failure"


def test_nonzero_worker_exit_rejects_a_valid_success_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _captured, _events, _process = _install_fake_launch(
        monkeypatch,
        tmp_path,
        response=_response(),
        returncode=23,
    )

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        executor.execute("stat_path", {"path": "private-path-marker.txt"}, intent="read")

    assert caught.value.code == "worker_crashed"
    assert caught.value.__cause__ is None


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


class _BlockingInput(_RecordingInput):
    def __init__(
        self,
        events: list[str],
        write_started: threading.Event,
        release_write: threading.Event,
    ) -> None:
        super().__init__(events)
        self._write_started = write_started
        self._release_write = release_write

    def write(self, value: bytes) -> int:
        self._events.append("stdin-write")
        self._write_started.set()
        if not self._release_write.wait(5):
            raise RuntimeError("test write barrier timed out")
        raise BrokenPipeError("simulated worker stopped consuming stdin")


def test_outer_deadline_covers_a_stalled_stdin_write_and_proves_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private-root-marker"
    root.mkdir()
    (root / "private-path-marker.txt").write_text("payload", encoding="utf-8")
    events: list[str] = []
    write_started = threading.Event()
    wait_started = threading.Event()
    release_write = threading.Event()
    process = _FakePopen(b"", b"", events)
    process.stdin = _BlockingInput(events, write_started, release_write)

    def wait_for_deadline(timeout: float | None = None) -> int:
        events.append("wait")
        wait_started.set()
        raise subprocess.TimeoutExpired("fixed-worker", timeout)

    process.wait = wait_for_deadline  # type: ignore[method-assign]

    def fake_popen(_argv: list[str], **_kwargs: Any) -> _FakePopen:
        return process

    class ReleasingTree(_FakeTree):
        def terminate_tree(self, **kwargs: Any) -> bool:
            events.append("terminate-tree")
            events.append(f"cleanup-budget:{kwargs}")
            release_write.set()
            self.process._process.returncode = -9
            return True

    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.ExecutorProcessTree", ReleasingTree
    )
    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.WORKSPACE_HELPER_TIMEOUT_SECONDS",
        1,
    )
    outcome: list[BaseException] = []

    def execute() -> None:
        try:
            WorkspaceToolExecutor(root).execute(
                "stat_path", {"path": "private-path-marker.txt"}, intent="read"
            )
        except BaseException as error:
            outcome.append(error)

    caller = threading.Thread(target=execute)
    caller.start()
    try:
        assert write_started.wait(2), "executor never attempted the bounded stdin write"
        assert wait_started.wait(2), "blocking stdin write prevented the outer deadline"
    finally:
        release_write.set()
        caller.join(5)

    assert not caller.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], WorkspaceToolExecutionError)
    assert outcome[0].code == "worker_timed_out"  # type: ignore[union-attr]
    assert "terminate-tree" in events


class _BlockingCloseInput(_RecordingInput):
    def __init__(
        self,
        events: list[str],
        close_started: threading.Event,
        close_finished: threading.Event,
        release_close: threading.Event,
    ) -> None:
        super().__init__(events)
        self._close_started = close_started
        self._close_finished = close_finished
        self._release_close = release_close

    def close(self) -> None:
        self._events.append("stdin-close-started")
        self._close_started.set()
        if not self._release_close.wait(5):
            raise RuntimeError("test close barrier timed out")
        super().close()
        self._close_finished.set()


def test_outer_deadline_bounds_multiphase_tree_settlement_and_pipe_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private-root-marker"
    root.mkdir()
    (root / "private-path-marker.txt").write_text("payload", encoding="utf-8")
    events: list[str] = []
    operation_wait_started = threading.Event()
    cleanup_started = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    release_blockers = threading.Event()
    finished = threading.Event()
    process = _FakePopen(b"", b"", events)
    process.stdin = _BlockingCloseInput(
        events,
        close_started,
        close_finished,
        release_blockers,
    )

    def wait_for_real_budget(timeout: float | None = None) -> int:
        events.append("wait")
        operation_wait_started.set()
        assert timeout is not None
        if not release_blockers.wait(timeout):
            raise subprocess.TimeoutExpired("fixed-worker", timeout)
        process.returncode = -9
        return -9

    process.wait = wait_for_real_budget  # type: ignore[method-assign]

    def fake_popen(_argv: list[str], **_kwargs: Any) -> _FakePopen:
        return process

    class BlockingTree(_FakeTree):
        def terminate_tree(self, **_kwargs: Any) -> bool:
            events.append("terminate-tree-started")
            cleanup_started.set()
            if not release_blockers.wait(5):
                raise RuntimeError("test cleanup barrier timed out")
            self.process._process.returncode = -9
            return True

    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.ExecutorProcessTree", BlockingTree
    )
    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.WORKSPACE_HELPER_TIMEOUT_SECONDS",
        1,
    )
    outcome: list[BaseException] = []
    started_at = time.monotonic()

    def execute() -> None:
        try:
            WorkspaceToolExecutor(root).execute(
                "stat_path", {"path": "private-path-marker.txt"}, intent="read"
            )
        except BaseException as error:
            outcome.append(error)
        finally:
            finished.set()

    caller = threading.Thread(target=execute)
    caller.start()
    try:
        assert operation_wait_started.wait(1), "operation wait did not start"
        assert close_started.wait(1), "writer did not reach the blocking pipe close"
        assert cleanup_started.wait(2), "tree settlement did not start"
        assert finished.wait(1.4), "caller exceeded the one-second outer deadline"
        elapsed = time.monotonic() - started_at
        assert elapsed >= 0.75
        assert elapsed < 1.4
    finally:
        release_blockers.set()
        caller.join(5)
        assert close_finished.wait(1), "pipe close did not finish after release"

    assert not caller.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], WorkspaceToolExecutionError)
    assert outcome[0].code == "cleanup_unproven"  # type: ignore[union-attr]
    assert events.count("stdin-close-started") >= 2
    assert process.stdin.was_closed


def test_pre_tree_terminate_timeout_falls_back_to_kill_and_bounded_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private-root-marker"
    root.mkdir()
    events: list[str] = []
    process = _FakePopen(b"", b"", events)

    def wait_for_cleanup(timeout: float | None = None) -> int:
        events.append("cleanup-wait")
        if process.returncode is None:
            raise subprocess.TimeoutExpired("fixed-worker", timeout)
        return process.returncode

    def terminate_without_exit() -> None:
        events.append("terminate")

    process.wait = wait_for_cleanup  # type: ignore[method-assign]
    process.terminate = terminate_without_exit  # type: ignore[method-assign]

    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.subprocess.Popen",
        lambda _argv, **_kwargs: process,
    )

    def fail_tree_construction(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("private-tree-construction-marker")

    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.ExecutorProcessTree",
        fail_tree_construction,
    )

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        WorkspaceToolExecutor(root).execute("stat_path", {"path": "."}, intent="read")

    assert caught.value.code == "worker_failure"
    assert events.index("terminate") < events.index("kill")
    assert events.count("cleanup-wait") == 2
    assert process.stdout.closed
    assert process.stderr.closed


def test_cancellation_identity_survives_cleanup_exception_and_closes_pipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = KeyboardInterrupt("private-cancellation-marker")
    executor, _captured, events, process = _install_fake_launch(
        monkeypatch,
        tmp_path,
        wait_error=cancellation,
    )

    class RaisingCleanupTree(_FakeTree):
        def terminate_tree(self, **_kwargs: Any) -> bool:
            events.append("terminate-tree")
            raise RuntimeError("private-cleanup-marker")

    monkeypatch.setattr(
        "tldw_chatbook.Tools.workspace_tool_executor.ExecutorProcessTree",
        RaisingCleanupTree,
    )

    with pytest.raises(BaseException) as caught:
        executor.execute("stat_path", {"path": "private-path-marker.txt"}, intent="read")

    assert caught.value is cancellation
    assert "terminate-tree" in events
    assert process.stdin.was_closed
    assert process.stdout.closed
    assert process.stderr.closed


def test_cleanup_supervisor_join_cancellation_preserves_identity_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cancellation = KeyboardInterrupt("private-supervisor-join-marker")
    executor, _captured, events, process = _install_fake_launch(monkeypatch, tmp_path)
    original_join = threading.Thread.join
    join_interrupted = threading.Event()
    finished = threading.Event()
    results: list[str] = []
    outcome: list[BaseException] = []

    def interrupt_completed_cleanup_join(
        thread: threading.Thread,
        timeout: float | None = None,
    ) -> None:
        original_join(thread, timeout)
        if thread.name == "workspace-worker-cleanup" and not join_interrupted.is_set():
            join_interrupted.set()
            raise cancellation

    def execute() -> None:
        try:
            results.append(
                executor.execute(
                    "stat_path",
                    {"path": "private-path-marker.txt"},
                    intent="read",
                )
            )
        except BaseException as error:
            outcome.append(error)
        finally:
            finished.set()

    monkeypatch.setattr(threading.Thread, "join", interrupt_completed_cleanup_join)
    with caplog.at_level(logging.DEBUG):
        caller = threading.Thread(target=execute, name="workspace-executor-test-caller")
        caller.start()
        assert finished.wait(1), "supervisor join cancellation did not return boundedly"
        caller.join(1)

    assert not caller.is_alive()
    assert join_interrupted.is_set()
    assert results == []
    assert len(outcome) == 1
    assert outcome[0] is cancellation
    assert "terminate-tree" in events
    assert process.stdin.was_closed
    assert process.stdout.closed
    assert process.stderr.closed
    assert "private-supervisor-join-marker" not in caplog.text
    assert "private-root-marker" not in caplog.text
    assert "private-path-marker" not in caplog.text
    assert "RESULT" not in caplog.text


@pytest.mark.parametrize(
    ("start_error", "expected_code"),
    [
        pytest.param(
            RuntimeError("private-supervisor-start-marker"),
            "worker_failure",
            id="ordinary-start-failure",
        ),
        pytest.param(
            KeyboardInterrupt("private-supervisor-cancellation-marker"),
            None,
            id="cancellation-start-failure",
        ),
    ],
)
def test_cleanup_supervisor_start_failure_precedes_authority_and_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start_error: BaseException,
    expected_code: str | None,
) -> None:
    executor, _captured, events, process = _install_fake_launch(monkeypatch, tmp_path)
    original_start = threading.Thread.start

    def fail_cleanup_supervisor(thread: threading.Thread) -> None:
        if thread.name == "workspace-worker-cleanup":
            raise start_error
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_cleanup_supervisor)

    with pytest.raises(BaseException) as caught:
        executor.execute("stat_path", {"path": "private-path-marker.txt"}, intent="read")

    if expected_code is None:
        assert caught.value is start_error
    else:
        assert isinstance(caught.value, WorkspaceToolExecutionError)
        assert caught.value.code == expected_code
        assert "private-supervisor-start-marker" not in repr(caught.value)
    assert "stdin-write" not in events
    assert "terminate-tree" in events
    assert process.stdin.was_closed
    assert process.stdout.closed
    assert process.stderr.closed


@pytest.mark.parametrize("cancel_during_wait", [False, True])
def test_pipe_closer_start_failure_is_bounded_and_preserves_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_during_wait: bool,
) -> None:
    cancellation = KeyboardInterrupt("private-inflight-cancellation-marker")
    executor, _captured, events, process = _install_fake_launch(
        monkeypatch,
        tmp_path,
        wait_error=cancellation if cancel_during_wait else None,
    )
    original_start = threading.Thread.start

    def fail_first_pipe_closer(thread: threading.Thread) -> None:
        if thread.name == "workspace-worker-pipe-close-0":
            raise RuntimeError("private-pipe-close-start-marker")
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_first_pipe_closer)
    started_at = time.monotonic()

    with pytest.raises(BaseException) as caught:
        executor.execute("stat_path", {"path": "private-path-marker.txt"}, intent="read")

    assert time.monotonic() - started_at < 1
    if cancel_during_wait:
        assert caught.value is cancellation
    else:
        assert isinstance(caught.value, WorkspaceToolExecutionError)
        assert caught.value.code == "cleanup_unproven"
        assert "private-pipe-close-start-marker" not in repr(caught.value)
    assert events.index("admit") < events.index("stdin-write")
    assert "terminate-tree" in events
    assert process.stdout.closed
    assert process.stderr.closed


@pytest.mark.parametrize(
    ("lifecycle_error", "expected_code", "propagates"),
    [
        pytest.param(
            RuntimeError("private-runtime-marker"),
            "worker_failure",
            False,
            id="unexpected-exception",
        ),
        pytest.param(
            KeyboardInterrupt("private-cancellation-marker"),
            None,
            True,
            id="cancellation-base-exception",
        ),
    ],
)
def test_post_spawn_lifecycle_exceptions_always_cleanup_and_close_pipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_error: BaseException,
    expected_code: str | None,
    propagates: bool,
) -> None:
    executor, _captured, events, process = _install_fake_launch(
        monkeypatch,
        tmp_path,
        wait_error=lifecycle_error,
    )

    with pytest.raises(BaseException) as caught:
        executor.execute("stat_path", {"path": "private-path-marker.txt"}, intent="read")

    if propagates:
        assert caught.value is lifecycle_error
    else:
        assert isinstance(caught.value, WorkspaceToolExecutionError)
        assert caught.value.code == expected_code
        assert caught.value.__cause__ is None
        assert "private-runtime-marker" not in repr(caught.value)
    assert "terminate-tree" in events
    assert process.stdin.was_closed
    assert process.stdout.closed
    assert process.stderr.closed


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


@pytest.mark.parametrize(("operation", "arguments", "expected"), READ_CASES)
def test_pre_pin_read_operations_refuse_a_replaced_root(
    tmp_path: Path,
    operation: str,
    arguments: dict[str, Any],
    expected: str,
) -> None:
    """Every read rejects a replacement before it can observe B's contents."""
    locator = tmp_path / "workspace"
    locator.mkdir()
    (locator / "sentinel.txt").write_text("A_ONLY", encoding="utf-8")
    chain = capture_directory_chain(locator)
    replacement = tmp_path / "replacement-b"
    replacement.mkdir()
    (replacement / "sentinel.txt").write_text("B_ONLY", encoding="utf-8")
    retained = tmp_path / "retained-a"
    os.replace(locator, retained)
    os.replace(replacement, locator)
    request = WorkspaceToolRequest(
        operation_id="pre-pin-read",
        operation=operation,  # type: ignore[arg-type]
        intent="read",
        root_locator=chain.canonical_root,
        root_identity=chain.identities[0],
        ancestor_identities=chain.identities,
        arguments=arguments,
        timeout_seconds=300,
        output_max_bytes=MAX_RESPONSE_BYTES,
    )
    stdout = io.BytesIO()

    exit_code = run_workspace_worker(
        io.BytesIO(request.to_bytes()), stdout, io.BytesIO()
    )

    frames = [
        WorkspaceToolResponse.from_bytes(line) for line in stdout.getvalue().splitlines()
    ]
    assert exit_code == 2
    assert [frame.outcome for frame in frames] == ["failure"]
    assert frames[0].code == "root_pin_failed"
    assert expected not in (frames[0].error or "")
    assert "B_ONLY" not in (frames[0].error or "")


def test_pinned_read_refuses_a_sensitive_relative_name(tmp_path: Path) -> None:
    """Pinned reads apply the bounded exclusion names before opening a file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "credentials").write_text("SECRET", encoding="utf-8")
    chain = capture_directory_chain(workspace)
    request = WorkspaceToolRequest(
        operation_id="sensitive-read",
        operation="fs_read",
        intent="read",
        root_locator=chain.canonical_root,
        root_identity=chain.identities[0],
        ancestor_identities=chain.identities,
        arguments={"path": "credentials", "offset": 1},
        timeout_seconds=300,
        output_max_bytes=MAX_RESPONSE_BYTES,
    )

    with pin_workspace_root(workspace, chain) as root:
        with pytest.raises(WorkspaceToolDispatchError) as caught:
            execute_pinned_operation(request, root)

    assert caught.value.code == "invalid_request"
    assert "SECRET" not in str(caught.value)


@pytest.mark.parametrize(("operation", "arguments", "expected"), READ_CASES)
def test_post_pin_read_operations_never_redirect_to_replaced_root(
    tmp_path: Path,
    operation: str,
    arguments: dict[str, Any],
    expected: str,
) -> None:
    """Pinned read bodies return A or refuse, never the replacement's B data."""
    locator = tmp_path / "workspace"
    locator.mkdir()
    replacement = tmp_path / "replacement-b"
    replacement.mkdir()
    if operation == "fs_list":
        (locator / "A_ONLY").write_text("a", encoding="utf-8")
        (replacement / "B_ONLY").write_text("b", encoding="utf-8")
    elif operation == "fs_glob":
        (locator / "sentinel.txt").write_text("A_ONLY", encoding="utf-8")
        (replacement / "B_ONLY.txt").write_text("B_ONLY", encoding="utf-8")
    else:
        (locator / "sentinel.txt").write_text("A_ONLY", encoding="utf-8")
        (replacement / "sentinel.txt").write_text("B_ONLY", encoding="utf-8")
    chain = capture_directory_chain(locator)

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_post_pin_read_operation_child,
        args=(str(locator), chain, operation, arguments, ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "read worker did not pin its root"

    retained = tmp_path / "retained-a"
    replacement_refused = False
    try:
        os.replace(locator, retained)
        os.replace(replacement, locator)
    except OSError:
        replacement_refused = True
        if retained.exists() and not locator.exists():
            os.replace(retained, locator)
    finally:
        resume.set()
    process.join(10)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("post-pin read worker did not exit")
    assert process.exitcode == 0

    outcome, value = output.get(timeout=2)
    assert outcome == "result"
    assert "B_ONLY" not in value
    assert expected in value
    if os.name == "nt":
        assert replacement_refused, "Windows should lock the retained current directory"
