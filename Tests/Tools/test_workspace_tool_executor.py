from __future__ import annotations

import io
import json
import logging
import multiprocessing
import os
import shutil
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
    ("fs_list", {"path": ".", "sensitive_exclusions": []}, "A_ONLY"),
    ("fs_read", {"path": "sentinel.txt", "offset": 1, "sensitive_exclusions": []}, "A_ONLY"),
    ("fs_glob", {"pattern": "**/*.txt", "max_results": 100, "sensitive_exclusions": []}, "sentinel.txt"),
    (
        "fs_grep",
        {"pattern": "A_ONLY", "mode": "content", "max_results": 100, "sensitive_exclusions": [], "content_exclusions": []},
        "A_ONLY",
    ),
)

MUTATION_CASES = (
    ("fs_write", {"path": "note.txt", "content": "changed"}),
    (
        "fs_edit",
        {
            "path": "note.txt",
            "old_string": "before",
            "new_string": "after",
            "replace_all": False,
        },
    ),
)

GIT_RACE_CASES = (
    ("git_status", {}, "A_STATUS.txt", "B_STATUS.txt"),
    ("git_diff", {}, "A_DIFF", "B_DIFF"),
    ("git_log", {"count": 20}, "A_LOG", "B_LOG"),
    ("git_blame", {"path": "blame.txt"}, "A_BLAME", "B_BLAME"),
    ("git_branches", {}, "A_BRANCH", "B_BRANCH"),
)

TWO_FILE_PATCH = """\
--- a/note.txt
+++ b/note.txt
@@ -1 +1 @@
-before
+after
--- a/other.txt
+++ b/other.txt
@@ -1 +1 @@
-first
+second
"""


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


def _post_pin_request_child(
    locator: str,
    chain: Any,
    request_bytes: bytes,
    ready: Any,
    resume: Any,
    output: Any,
) -> None:
    """Dispatch a parent-built request after its pinned root is replaced."""
    try:
        request = WorkspaceToolRequest.from_bytes(request_bytes)
        with pin_workspace_root(Path(locator), chain) as root:
            ready.set()
            if not resume.wait(5):
                raise RuntimeError("test barrier timed out")
            os.environ.pop("TLDW_CONFIG_PATH", None)
            output.put(("result", execute_pinned_operation(request, root)))
    except WorkspaceToolDispatchError as error:
        output.put(("refused", error.code))
    except WorkspaceRootPinError:
        output.put(("refused", "root_pin_failed"))
    except BaseException as error:
        output.put(("error", type(error).__name__))


def _pre_pin_request_child(
    request_bytes: bytes,
    ready: Any,
    resume: Any,
    output: Any,
) -> None:
    """Pause a real worker request immediately before root pinning begins."""
    try:
        ready.set()
        if not resume.wait(5):
            raise RuntimeError("test barrier timed out")
        stdout = io.BytesIO()
        exit_code = run_workspace_worker(
            io.BytesIO(request_bytes), stdout, io.BytesIO()
        )
        output.put((exit_code, stdout.getvalue()))
    except BaseException as error:
        output.put(("error", type(error).__name__))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_git_race_repository(repo: Path, marker: str) -> None:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    (repo / "blame.txt").write_text(f"{marker}_BLAME\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", f"{marker}_LOG")
    _git(repo, "branch", f"{marker}_BRANCH")
    (repo / "tracked.txt").write_text(f"base\n{marker}_DIFF\n", encoding="utf-8")
    (repo / f"{marker}_STATUS.txt").write_text(marker, encoding="utf-8")


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


@pytest.fixture(scope="module")
def isolated_runtime_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create one isolated interpreter that imports this worktree under ``-I``."""
    repository_root = Path(__file__).resolve().parents[2]
    environment_root = tmp_path_factory.mktemp("isolated-runtime") / "venv"
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
    return runtime_python


def test_real_isolated_subprocess_executes_this_worktree_vertical_slice(
    tmp_path: Path, isolated_runtime_python: Path
) -> None:
    repository_root = Path(__file__).resolve().parents[2]

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
            str(isolated_runtime_python),
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


def test_real_executor_surfaces_an_ordinary_edit_failure_frame(
    tmp_path: Path, isolated_runtime_python: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("before", encoding="utf-8")
    harness = """
import json
import sys
from pathlib import Path
from tldw_chatbook.Tools.workspace_tool_executor import (
    WorkspaceToolExecutionError,
    WorkspaceToolExecutor,
)

try:
    WorkspaceToolExecutor(Path(sys.argv[1])).execute(
        "fs_edit",
        {"path": "note.txt", "old_string": "missing", "new_string": "after"},
        intent="write",
    )
except WorkspaceToolExecutionError as error:
    print(json.dumps({"code": error.code, "cause": error.__cause__ is None}))
"""

    completed = subprocess.run(
        [str(isolated_runtime_python), "-I", "-c", harness, str(workspace)],
        env={"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload == {"code": "tool_failure", "cause": True}


def test_real_executor_surfaces_a_patch_target_mismatch_refusal(
    tmp_path: Path, isolated_runtime_python: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    note = workspace / "note.txt"
    note.write_bytes(b"before\n")

    harness = '''
import json
import sys
from pathlib import Path
from tldw_chatbook.Tools.workspace_tool_executor import (
    WorkspaceToolExecutionError,
    WorkspaceToolExecutor,
)

class TargetMismatchExecutor(WorkspaceToolExecutor):
    def _build_request(self, operation, arguments, *, intent):
        request = super()._build_request(operation, arguments, intent=intent)
        request.arguments["targets"] = ["different.txt"]
        return request

try:
    TargetMismatchExecutor(Path(sys.argv[1])).execute(
        "fs_patch",
        {"diff": "--- a/note.txt\\n+++ b/note.txt\\n@@ -1 +1 @@\\n-before\\n+after\\n"},
        intent="write",
    )
except WorkspaceToolExecutionError as error:
    print(json.dumps({"code": error.code, "cause": error.__cause__ is None}))
'''

    completed = subprocess.run(
        [str(isolated_runtime_python), "-I", "-c", harness, str(workspace)],
        env={"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload == {"code": "invalid_request", "cause": True}
    assert note.read_bytes() == b"before\n"


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


@pytest.mark.parametrize("pattern", ("../outside/*.txt", "safe\x00name/*.txt"))
def test_direct_worker_rejects_a_bypassed_unsafe_glob_pattern(pattern: str) -> None:
    chain = capture_directory_chain(Path.cwd())
    request = WorkspaceToolRequest(
        operation_id="unsafe-direct-glob",
        operation="fs_glob",
        intent="read",
        root_locator=chain.canonical_root,
        root_identity=chain.identities[0],
        ancestor_identities=chain.identities,
        arguments={"pattern": pattern, "sensitive_exclusions": []},
        timeout_seconds=30,
        output_max_bytes=MAX_RESPONSE_BYTES,
    )

    with pytest.raises(WorkspaceToolDispatchError) as caught:
        execute_pinned_operation(request, SimpleNamespace())

    assert caught.value.code == "invalid_request"


def _run_worker_request(request: WorkspaceToolRequest) -> WorkspaceToolResponse:
    """Run a hand-built pinned request and return its sole terminal frame."""
    stdout = io.BytesIO()
    exit_code = run_workspace_worker(io.BytesIO(request.to_bytes()), stdout, io.BytesIO())
    frames = [
        WorkspaceToolResponse.from_bytes(line) for line in stdout.getvalue().splitlines()
    ]
    assert exit_code == (0 if frames[-1].outcome == "success" else 2), frames
    return frames[-1]


def test_pinned_worker_applies_exclusions_to_both_lexical_and_resolved_aliases(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "public.txt").write_text("PUBLIC", encoding="utf-8")
    os.symlink(workspace / "public.txt", workspace / "credentials")
    chain = capture_directory_chain(workspace)
    request = WorkspaceToolRequest(
        operation_id="lexical-alias",
        operation="fs_read",
        intent="read",
        root_locator=chain.canonical_root,
        root_identity=chain.identities[0],
        ancestor_identities=chain.identities,
        arguments={
            "path": "credentials",
            "offset": 1,
            "sensitive_exclusions": [{"kind": "name", "value": "credentials"}],
        },
        timeout_seconds=30,
        output_max_bytes=MAX_RESPONSE_BYTES,
    )

    denied = _run_worker_request(request)
    assert denied.outcome == "failure"
    assert "PUBLIC" not in (denied.error or "")

    os.unlink(workspace / "credentials")
    os.symlink(workspace / "public.txt", workspace / "safe-alias")
    allowed = WorkspaceToolRequest(
        operation_id="safe-alias",
        operation="fs_read",
        intent="read",
        root_locator=chain.canonical_root,
        root_identity=chain.identities[0],
        ancestor_identities=chain.identities,
        arguments={
            "path": "safe-alias",
            "offset": 1,
            "sensitive_exclusions": [{"kind": "name", "value": "credentials"}],
        },
        timeout_seconds=30,
        output_max_bytes=MAX_RESPONSE_BYTES,
    )
    accepted = _run_worker_request(allowed)
    assert accepted.outcome == "success"
    assert accepted.result is not None and "PUBLIC" in accepted.result


@pytest.mark.parametrize(
    ("operation", "arguments", "expected"),
    (
        ("fs_read", {"path": "protected/child.txt", "offset": 1}, "failure"),
        ("fs_glob", {"pattern": "**/*", "max_results": 100}, "success"),
        (
            "fs_grep",
            {"pattern": "DIRECT_SECRET", "mode": "content", "max_results": 100},
            "success",
        ),
    ),
)
def test_pinned_worker_honors_direct_children_exclusions(
    tmp_path: Path, operation: str, arguments: dict[str, Any], expected: str
) -> None:
    workspace = tmp_path / "workspace"
    protected = workspace / "protected"
    protected.mkdir(parents=True)
    (protected / "child.txt").write_text("DIRECT_SECRET", encoding="utf-8")
    chain = capture_directory_chain(workspace)
    request = WorkspaceToolRequest(
        operation_id=f"direct-children-{operation}",
        operation=operation,  # type: ignore[arg-type]
        intent="read",
        root_locator=chain.canonical_root,
        root_identity=chain.identities[0],
        ancestor_identities=chain.identities,
        arguments={
            **arguments,
            "sensitive_exclusions": [
                {"kind": "direct_children", "value": "protected"}
            ],
            **(
                {"content_exclusions": [{"kind": "direct_children", "value": "protected"}]}
                if operation == "fs_grep"
                else {}
            ),
        },
        timeout_seconds=30,
        output_max_bytes=MAX_RESPONSE_BYTES,
    )

    response = _run_worker_request(request)
    assert response.outcome == expected
    assert response.result != "protected/child.txt:1:DIRECT_SECRET"
    assert "child.txt" not in (response.result or "")


@pytest.mark.parametrize(
    ("operation", "arguments", "target_kind"),
    (
        ("fs_read", {"path": "link", "offset": 1}, "escaping"),
        ("fs_read", {"path": "link", "offset": 1}, "sensitive"),
        ("fs_glob", {"pattern": "**/*", "max_results": 100}, "escaping"),
        (
            "fs_grep",
            {"pattern": "RETARGET_SECRET", "mode": "content", "max_results": 100},
            "sensitive",
        ),
    ),
)
def test_pinned_worker_refuses_stably_retargeted_symlinks_before_operation(
    tmp_path: Path, operation: str, arguments: dict[str, Any], target_kind: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "public.txt").write_text("PUBLIC", encoding="utf-8")
    (workspace / "credentials").write_text("RETARGET_SECRET", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("RETARGET_SECRET", encoding="utf-8")
    link = workspace / "link"
    os.symlink(workspace / "public.txt", link)
    request = WorkspaceToolExecutor(workspace)._build_request(
        operation, arguments, intent="read"
    )
    os.unlink(link)
    os.symlink(outside if target_kind == "escaping" else workspace / "credentials", link)

    response = _run_worker_request(request)
    if operation == "fs_read":
        assert response.outcome == "failure"
    else:
        assert response.outcome == "success"
    assert "link" not in (response.result or "")
    if operation == "fs_grep":
        assert response.result == "(no matches for 'RETARGET_SECRET')"
    else:
        assert "RETARGET_SECRET" not in (response.result or "")


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
    """Pinned reads reject a sensitive name while the parent still has policy."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "credentials").write_text("SECRET", encoding="utf-8")
    with pytest.raises(WorkspaceToolExecutionError) as caught:
        WorkspaceToolExecutor(workspace)._build_request(
            "fs_read", {"path": "credentials", "offset": 1}, intent="read"
        )

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


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
@pytest.mark.parametrize(
    ("operation", "arguments", "expected_a", "forbidden_b"), GIT_RACE_CASES
)
def test_post_pin_git_operations_never_redirect_to_replaced_root(
    tmp_path: Path,
    operation: str,
    arguments: dict[str, Any],
    expected_a: str,
    forbidden_b: str,
) -> None:
    """All read-only Git operations stay bound to retained repository A."""
    locator = tmp_path / "workspace"
    replacement = tmp_path / "replacement-b"
    _init_git_race_repository(locator, "A")
    _init_git_race_repository(replacement, "B")
    request = WorkspaceToolExecutor(locator)._build_request(
        operation, arguments, intent="read"
    )
    chain = capture_directory_chain(locator)

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_post_pin_request_child,
        args=(str(locator), chain, request.to_bytes(), ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "Git worker did not pin its root"

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
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("post-pin Git worker did not exit")
    assert process.exitcode == 0

    outcome, value = output.get(timeout=2)
    assert outcome == "result"
    assert expected_a in value
    assert forbidden_b not in value
    if os.name == "nt":
        assert replacement_refused, "Windows should lock the retained current directory"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_pinned_git_supports_linked_worktree_without_granting_metadata_fs_access(
    tmp_path: Path,
) -> None:
    """Git may follow a linked-worktree pointer that ordinary fs tools cannot."""
    primary = tmp_path / "primary"
    primary.mkdir()
    _git(primary, "init")
    _git(primary, "config", "user.email", "test@example.invalid")
    _git(primary, "config", "user.name", "Test User")
    _git(primary, "config", "commit.gpgsign", "false")
    (primary / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(primary, "add", ".")
    _git(primary, "commit", "-m", "linked initial")
    worktree = tmp_path / "workspace"
    _git(primary, "worktree", "add", "-b", "linked-branch", str(worktree))

    git_pointer = (worktree / ".git").read_text(encoding="utf-8").strip()
    assert git_pointer.startswith("gitdir: ")
    admin = Path(git_pointer.removeprefix("gitdir: ")).resolve()
    assert worktree.resolve() not in admin.parents

    (worktree / "tracked.txt").write_text("base\nlinked diff\n", encoding="utf-8")
    cases = (
        ("git_status", {}, "tracked.txt"),
        ("git_diff", {}, "linked diff"),
        ("git_log", {"count": 20}, "linked initial"),
        ("git_blame", {"path": "tracked.txt"}, "base"),
        ("git_branches", {}, "linked-branch"),
    )
    executor = WorkspaceToolExecutor(worktree)
    for operation, arguments, expected in cases:
        response = _run_worker_request(
            executor._build_request(operation, arguments, intent="read")
        )
        assert response.outcome == "success", (operation, response.code)
        assert expected in (response.result or ""), operation

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        executor._build_request(
            "fs_read", {"path": str(admin / "HEAD"), "offset": 1}, intent="read"
        )
    assert caught.value.code == "invalid_request"


@pytest.mark.parametrize(("operation", "arguments"), MUTATION_CASES)
def test_pre_pin_mutations_refuse_a_replaced_root_without_touching_b_or_external_bytes(
    tmp_path: Path,
    operation: str,
    arguments: dict[str, Any],
) -> None:
    """A mutation admitted for A must not follow its locator to replacement B."""
    locator = tmp_path / "workspace"
    locator.mkdir()
    (locator / "note.txt").write_bytes(b"before")
    request = WorkspaceToolExecutor(locator)._build_request(
        operation, arguments, intent="write"
    )
    replacement = tmp_path / "replacement-b"
    replacement.mkdir()
    b_note = b"B-note-byte-exact\r\n"
    (replacement / "note.txt").write_bytes(b_note)
    external = tmp_path / "external-sentinel.bin"
    external_bytes = b"external-byte-exact\x00\xff"
    external.write_bytes(external_bytes)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_pre_pin_request_child,
        args=(request.to_bytes(), ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "mutation worker did not reach the pre-pin barrier"
    retained = tmp_path / "retained-a"
    try:
        os.replace(locator, retained)
        os.replace(replacement, locator)
    finally:
        resume.set()
    process.join(10)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("pre-pin mutation worker did not exit")
    assert process.exitcode == 0
    exit_code, raw_frames = output.get(timeout=2)

    frames = [
        WorkspaceToolResponse.from_bytes(line) for line in raw_frames.splitlines()
    ]
    assert exit_code == 2
    assert [frame.outcome for frame in frames] == ["failure"]
    assert frames[0].code == "root_pin_failed"
    assert (locator / "note.txt").read_bytes() == b_note
    assert external.read_bytes() == external_bytes


def test_pre_pin_two_file_patch_refuses_without_touching_any_b_or_external_bytes(
    tmp_path: Path,
) -> None:
    locator = tmp_path / "workspace"
    locator.mkdir()
    (locator / "note.txt").write_bytes(b"before\n")
    (locator / "other.txt").write_bytes(b"first\n")
    request = WorkspaceToolExecutor(locator)._build_request(
        "fs_patch", {"diff": TWO_FILE_PATCH}, intent="write"
    )
    replacement = tmp_path / "replacement-b"
    replacement.mkdir()
    b_note = b"B-note-byte-exact\r\n"
    b_other = b"B-other-byte-exact\x00\xff"
    (replacement / "note.txt").write_bytes(b_note)
    (replacement / "other.txt").write_bytes(b_other)
    external = tmp_path / "external-sentinel.bin"
    external_bytes = b"external-byte-exact\x00\xff"
    external.write_bytes(external_bytes)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_pre_pin_request_child,
        args=(request.to_bytes(), ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "patch worker did not reach the pre-pin barrier"
    retained = tmp_path / "retained-a"
    try:
        os.replace(locator, retained)
        os.replace(replacement, locator)
    finally:
        resume.set()
    process.join(10)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("pre-pin patch worker did not exit")
    assert process.exitcode == 0
    exit_code, raw_frames = output.get(timeout=2)

    frames = [
        WorkspaceToolResponse.from_bytes(line) for line in raw_frames.splitlines()
    ]
    assert exit_code == 2
    assert [frame.outcome for frame in frames] == ["failure"]
    assert frames[0].code == "root_pin_failed"
    assert (locator / "note.txt").read_bytes() == b_note
    assert (locator / "other.txt").read_bytes() == b_other
    assert external.read_bytes() == external_bytes


@pytest.mark.parametrize(
    ("operation", "arguments", "expected_a"),
    (
        ("fs_write", {"path": "note.txt", "content": "changed"}, b"changed"),
        (
            "fs_edit",
            {
                "path": "note.txt",
                "old_string": "before",
                "new_string": "after",
                "replace_all": False,
            },
            b"after",
        ),
        ("fs_patch", {"diff": TWO_FILE_PATCH}, b"after\n"),
    ),
)
def test_post_pin_mutations_never_redirect_to_replaced_root(
    tmp_path: Path,
    operation: str,
    arguments: dict[str, Any],
    expected_a: bytes,
) -> None:
    """Post-pin mutations land only in retained A, or root replacement is refused."""
    locator = tmp_path / "workspace"
    locator.mkdir()
    (locator / "note.txt").write_bytes(b"before\n" if operation == "fs_patch" else b"before")
    if operation == "fs_patch":
        (locator / "other.txt").write_bytes(b"first\n")
    request = WorkspaceToolExecutor(locator)._build_request(
        operation, arguments, intent="write"
    )
    chain = capture_directory_chain(locator)
    replacement = tmp_path / "replacement-b"
    replacement.mkdir()
    b_note = b"B-note-byte-exact\r\n"
    b_other = b"B-other-byte-exact\x00\xff"
    (replacement / "note.txt").write_bytes(b_note)
    if operation == "fs_patch":
        (replacement / "other.txt").write_bytes(b_other)
    external = tmp_path / "external-sentinel.bin"
    external_bytes = b"external-byte-exact\x00\xff"
    external.write_bytes(external_bytes)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_post_pin_request_child,
        args=(str(locator), chain, request.to_bytes(), ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "mutation worker did not pin its root"

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
        pytest.fail("post-pin mutation worker did not exit")
    assert process.exitcode == 0

    outcome, value = output.get(timeout=2)
    assert outcome == "result", value
    a_root = locator if replacement_refused else retained
    assert (a_root / "note.txt").read_bytes() == expected_a
    if operation == "fs_patch":
        assert (a_root / "other.txt").read_bytes() == b"second\n"
    if replacement_refused:
        assert not locator.samefile(replacement)
    else:
        assert (locator / "note.txt").read_bytes() == b_note
        if operation == "fs_patch":
            assert (locator / "other.txt").read_bytes() == b_other
    assert external.read_bytes() == external_bytes


def test_two_file_patch_uses_one_admitted_frame_and_one_requested_root_identity(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_bytes(b"before\n")
    (workspace / "other.txt").write_bytes(b"first\n")
    chain = capture_directory_chain(workspace)
    request = WorkspaceToolExecutor(workspace)._build_request(
        "fs_patch", {"diff": TWO_FILE_PATCH}, intent="write"
    )
    stdout = io.BytesIO()

    exit_code = run_workspace_worker(
        io.BytesIO(request.to_bytes()), stdout, io.BytesIO()
    )

    frames = [
        WorkspaceToolResponse.from_bytes(line) for line in stdout.getvalue().splitlines()
    ]
    assert exit_code == 0
    assert [frame.outcome for frame in frames] == ["admitted", "success"]
    assert sum(frame.outcome == "admitted" for frame in frames) == 1
    assert request.root_identity == chain.identities[0]
    assert request.arguments["targets"] == ["note.txt", "other.txt"]
    assert (workspace / "note.txt").read_bytes() == b"after\n"
    assert (workspace / "other.txt").read_bytes() == b"second\n"


def test_worker_rejects_patch_target_set_mismatch_before_writing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_bytes(b"before\n")
    (workspace / "other.txt").write_bytes(b"first\n")
    request = WorkspaceToolExecutor(workspace)._build_request(
        "fs_patch", {"diff": TWO_FILE_PATCH}, intent="write"
    )
    request.arguments["targets"] = ["note.txt", "different.txt"]

    response = _run_worker_request(request)

    assert response.outcome == "failure"
    assert response.code == "invalid_request"
    assert (workspace / "note.txt").read_bytes() == b"before\n"
    assert (workspace / "other.txt").read_bytes() == b"first\n"


@pytest.mark.parametrize("operation", ("fs_write", "fs_edit", "fs_patch"))
@pytest.mark.parametrize("retarget", ("external", "sensitive"))
def test_worker_refuses_mutation_target_retargeted_after_parent_admission(
    tmp_path: Path,
    operation: str,
    retarget: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    public = workspace / "public.txt"
    public_bytes = b"before\n"
    public.write_bytes(public_bytes)
    sensitive = workspace / "credentials"
    sensitive_bytes = b"before\nsensitive-byte-exact\x00\xff"
    sensitive.write_bytes(sensitive_bytes)
    external = tmp_path / "external-sentinel.bin"
    external_bytes = b"before\nexternal-byte-exact\x00\xff"
    external.write_bytes(external_bytes)
    alias = workspace / "alias.txt"
    os.symlink(public, alias)

    if operation == "fs_write":
        arguments = {"path": "alias.txt", "content": "changed"}
    elif operation == "fs_edit":
        arguments = {
            "path": "alias.txt",
            "old_string": "before",
            "new_string": "after",
        }
    else:
        arguments = {
            "diff": """\
--- a/alias.txt
+++ b/alias.txt
@@ -1 +1 @@
-before
+after
"""
        }
    request = WorkspaceToolExecutor(workspace)._build_request(
        operation, arguments, intent="write"
    )
    alias.unlink()
    os.symlink(external if retarget == "external" else sensitive, alias)

    response = _run_worker_request(request)

    assert response.outcome == "failure"
    assert response.code == "invalid_request"
    assert public.read_bytes() == public_bytes
    assert external.read_bytes() == external_bytes
    assert sensitive.read_bytes() == sensitive_bytes


def test_parent_validates_every_patch_target_before_worker_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_bytes(b"before\n")
    runtime_config = workspace / "runtime-config.toml"
    runtime_config.write_bytes(b"SECRET\n")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(runtime_config))
    diff = TWO_FILE_PATCH.replace("other.txt", "runtime-config.toml").replace(
        "first", "SECRET"
    ).replace("second", "CHANGED")

    def unexpected_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid patch targets must be refused before spawn")

    monkeypatch.setattr(subprocess, "Popen", unexpected_spawn)
    with pytest.raises(WorkspaceToolExecutionError) as caught:
        WorkspaceToolExecutor(workspace).execute(
            "fs_patch", {"diff": diff}, intent="write"
        )

    assert caught.value.code == "invalid_request"
    assert (workspace / "note.txt").read_bytes() == b"before\n"
    assert runtime_config.read_bytes() == b"SECRET\n"


@pytest.mark.parametrize(
    ("operation", "arguments"),
    (
        ("fs_list", {"path": "."}),
        ("fs_glob", {"pattern": "**/*", "max_results": 100}),
        ("fs_grep", {"pattern": "needle", "mode": "content", "max_results": 100}),
    ),
)
def test_parent_serializes_runtime_sensitive_exclusions_for_every_read_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    arguments: dict[str, Any],
) -> None:
    """The parent captures runtime exclusions before the worker environment is stripped."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("needle\n", encoding="utf-8")
    runtime_config = workspace / "runtime-config.toml"
    runtime_config.write_text("needle = 'SECRET'\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(runtime_config))

    request = WorkspaceToolExecutor(workspace)._build_request(
        operation, arguments, intent="read"
    )

    exclusions = request.arguments["sensitive_exclusions"]
    assert {"kind": "file", "value": "runtime-config.toml"} in exclusions


def test_parent_read_exclusions_do_not_recursively_enumerate_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sensitive exclusion preparation is bounded by policy inputs, not tree size."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "ordinary.txt").write_text("ordinary", encoding="utf-8")

    def _unexpected_rglob(self: Path, pattern: str) -> Any:
        raise AssertionError("parent exclusion preparation must not walk the workspace")

    monkeypatch.setattr(Path, "rglob", _unexpected_rglob)
    request = WorkspaceToolExecutor(workspace)._build_request(
        "fs_list", {"path": "."}, intent="read"
    )

    assert request.arguments["sensitive_exclusions"]


def test_parent_refuses_runtime_config_read_before_worker_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime config override is adjudicated while the parent still has it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime_config = workspace / "runtime-config.toml"
    runtime_config.write_text("SECRET", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(runtime_config))

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        WorkspaceToolExecutor(workspace)._build_request(
            "fs_read", {"path": "runtime-config.toml", "offset": 1}, intent="read"
        )

    assert caught.value.code == "invalid_request"


@pytest.mark.parametrize(
    "pattern",
    ("../outside/*.txt", "/outside/*.txt", r"\outside\*.txt", r"C:\outside\*.txt"),
)
def test_parent_refuses_parent_and_cross_platform_rooted_glob_patterns(
    tmp_path: Path, pattern: str
) -> None:
    """Pinned glob admission never grants a pattern a route outside the root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        WorkspaceToolExecutor(workspace)._build_request(
            "fs_glob", {"pattern": pattern, "max_results": 100}, intent="read"
        )

    assert caught.value.code == "invalid_request"


@pytest.mark.parametrize(
    ("operation", "arguments"),
    (
        ("fs_list", {"path": "."}),
        ("fs_read", {"path": "safe-link", "offset": 1}),
        ("fs_glob", {"pattern": "**/*", "max_results": 100}),
        ("fs_grep", {"pattern": "ALIAS_SECRET", "mode": "content", "max_results": 100}),
    ),
)
def test_pinned_operations_do_not_disclose_in_root_sensitive_symlink_aliases(
    tmp_path: Path,
    operation: str,
    arguments: dict[str, Any],
) -> None:
    """Parent-derived aliases hide sensitive in-root targets from every read."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "credentials").write_text("ALIAS_SECRET", encoding="utf-8")
    os.symlink(workspace / "credentials", workspace / "safe-link")
    executor = WorkspaceToolExecutor(workspace)

    if operation == "fs_read":
        with pytest.raises(WorkspaceToolExecutionError) as caught:
            executor._build_request(operation, arguments, intent="read")
        assert caught.value.code == "invalid_request"
        return

    request = executor._build_request(operation, arguments, intent="read")
    stdout = io.BytesIO()
    exit_code = run_workspace_worker(
        io.BytesIO(request.to_bytes()), stdout, io.BytesIO()
    )
    frames = [
        WorkspaceToolResponse.from_bytes(line) for line in stdout.getvalue().splitlines()
    ]

    assert exit_code == 0
    assert frames[-1].result is not None
    assert "safe-link" not in frames[-1].result


def test_parent_exclusions_survive_root_replacement_and_worker_env_stripping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent-built location exclusion remains effective after the root rename."""
    locator = tmp_path / "workspace"
    locator.mkdir()
    runtime_config = locator / "runtime-config.toml"
    runtime_config.write_text("A_CONFIG_SECRET", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(runtime_config))
    request = WorkspaceToolExecutor(locator)._build_request(
        "fs_grep",
        {"pattern": "CONFIG_SECRET", "mode": "content", "max_results": 100},
        intent="read",
    )
    chain = capture_directory_chain(locator)
    replacement = tmp_path / "replacement-b"
    replacement.mkdir()
    (replacement / "runtime-config.toml").write_text("B_CONFIG_SECRET", encoding="utf-8")

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_post_pin_request_child,
        args=(str(locator), chain, request.to_bytes(), ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "read worker did not pin its root"

    retained = tmp_path / "retained-a"
    try:
        os.replace(locator, retained)
        os.replace(replacement, locator)
    except OSError as error:
        if os.name == "nt":
            pytest.skip(f"Windows current-directory sharing refused root replacement: {error}")
        pytest.fail(f"POSIX root replacement unexpectedly failed: {error}")
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
    assert "A_CONFIG_SECRET" not in value
    assert "B_CONFIG_SECRET" not in value
