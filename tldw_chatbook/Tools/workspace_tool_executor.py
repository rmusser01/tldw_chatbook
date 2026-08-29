"""Parent-side one-shot executor for pinned workspace tool requests."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from tldw_chatbook.STT.executor_process_tree import (
    ExecutorProcessTree,
    ProcessContainmentError,
    WorkerContainmentIdentity,
)
from tldw_chatbook.Tools.local_tool_impls import resolve_workspace_path
from tldw_chatbook.Tools.workspace_tool_protocol import (
    MAX_RESPONSE_BYTES,
    WorkspaceProtocolError,
    WorkspaceToolRequest,
    WorkspaceToolResponse,
)
from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryIdentityError,
    capture_directory_chain,
)

WORKSPACE_HELPER_TIMEOUT_SECONDS = 300
DIAGNOSTIC_STDERR_MAX_BYTES = 8 * 1024
_FIXED_WORKER_MODULE = "tldw_chatbook.Tools.workspace_tool_worker"
_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class WorkspaceToolExecutionError(RuntimeError):
    """A stable, content-free one-shot workspace execution refusal."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"workspace operation failed ({code})")


class _PopenProcessAdapter:
    """Expose only the process lifecycle expected by ExecutorProcessTree."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self.pid = process.pid

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def join(self, timeout: float | None = None) -> None:
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()


def workspace_worker_environment() -> dict[str, str]:
    """Return the fixed runtime allowlist inherited by the isolated helper."""
    environment = {"PATH": os.defpath}
    if os.name == "posix":
        environment.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    else:
        for key in ("SYSTEMROOT", "WINDIR"):
            value = os.environ.get(key)
            if value:
                environment[key] = value
    return environment


class WorkspaceToolExecutor:
    """Launch one contained helper for one workspace operation."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = Path(workspace_root)

    def execute(
        self,
        operation: str,
        arguments: dict[str, Any],
        *,
        intent: str,
    ) -> str:
        """Validate, execute once, prove cleanup, and return bounded text."""
        request = self._build_request(operation, arguments, intent=intent)
        process: subprocess.Popen[bytes] | None = None
        tree: ExecutorProcessTree | None = None
        stdout_capture: list[bytes] = []
        stderr_capture: list[bytes] = []
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        try:
            argv = [sys.executable, "-I", "-m", _FIXED_WORKER_MODULE]
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=workspace_worker_environment(),
                shell=False,
                start_new_session=(os.name == "posix"),
            )
            adapter = _PopenProcessAdapter(process)
            identity = WorkerContainmentIdentity(
                pid=process.pid,
                process_group_id=process.pid if os.name == "posix" else None,
            )
            admission = threading.Event()
            tree = ExecutorProcessTree(adapter, admission, identity)
            tree.admit()

            if process.stdout is None or process.stderr is None or process.stdin is None:
                raise WorkspaceToolExecutionError("spawn_failed")
            stdout_thread = _start_bounded_reader(
                process.stdout,
                MAX_RESPONSE_BYTES,
                stdout_capture,
                name="workspace-worker-stdout",
            )
            stderr_thread = _start_bounded_reader(
                process.stderr,
                DIAGNOSTIC_STDERR_MAX_BYTES,
                stderr_capture,
                name="workspace-worker-stderr",
            )
            process.stdin.write(request.to_bytes())
            process.stdin.close()
            try:
                process.wait(timeout=request.timeout_seconds)
            except subprocess.TimeoutExpired:
                cleanup = tree.terminate_tree()
                _join_readers(stdout_thread, stderr_thread)
                if not cleanup:
                    raise WorkspaceToolExecutionError("cleanup_unproven") from None
                raise WorkspaceToolExecutionError("worker_timed_out") from None

            _join_readers(stdout_thread, stderr_thread)
            cleanup = tree.close()
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven")
            if process.returncode != 0 and not stdout_capture:
                raise WorkspaceToolExecutionError("worker_crashed")
            response = _parse_worker_output(
                b"".join(stdout_capture),
                expected_operation_id=request.operation_id,
            )
            if not response.cleanup_proven:
                raise WorkspaceToolExecutionError("cleanup_unproven")
            if response.outcome == "failure":
                raise WorkspaceToolExecutionError(response.code)
            if response.result is None:
                raise WorkspaceToolExecutionError("protocol_failure")
            return response.result
        except WorkspaceToolExecutionError:
            if tree is not None and process is not None and process.poll() is None:
                cleanup = _terminate_tree(tree)
                if not cleanup:
                    raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise
        except ProcessContainmentError:
            cleanup = _terminate_tree(tree) if tree is not None else _stop_process(process)
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise WorkspaceToolExecutionError("containment_unavailable") from None
        except (OSError, ValueError):
            cleanup = _terminate_tree(tree) if tree is not None else _stop_process(process)
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise WorkspaceToolExecutionError("spawn_failed") from None
        finally:
            del stderr_capture

    def _build_request(
        self,
        operation: str,
        arguments: dict[str, Any],
        *,
        intent: str,
    ) -> WorkspaceToolRequest:
        try:
            chain = capture_directory_chain(self._workspace_root)
            normalized = dict(arguments)
            if operation == "stat_path" and type(arguments) is dict:
                raw_path = arguments.get("path")
                if type(raw_path) is str:
                    target = resolve_workspace_path(
                        raw_path,
                        chain.canonical_root,
                        intent="read",
                    )
                    normalized["path"] = str(target.relative_to(chain.canonical_root))
            request = WorkspaceToolRequest(
                operation_id=uuid.uuid4().hex,
                operation=operation,  # type: ignore[arg-type]
                intent=intent,  # type: ignore[arg-type]
                root_locator=chain.canonical_root,
                root_identity=chain.identities[0],
                ancestor_identities=chain.identities,
                arguments=normalized,
                timeout_seconds=WORKSPACE_HELPER_TIMEOUT_SECONDS,
                output_max_bytes=MAX_RESPONSE_BYTES,
            )
            return WorkspaceToolRequest.from_bytes(request.to_bytes())
        except (DirectoryIdentityError, WorkspaceProtocolError, OSError, ValueError):
            raise WorkspaceToolExecutionError("invalid_request") from None


def _start_bounded_reader(
    stream: BinaryIO,
    limit: int,
    capture: list[bytes],
    *,
    name: str,
) -> threading.Thread:
    def read() -> None:
        retained = 0
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            if retained <= limit:
                piece = chunk[: limit + 1 - retained]
                capture.append(piece)
                retained += len(piece)

    thread = threading.Thread(target=read, name=name, daemon=True)
    thread.start()
    return thread


def _join_readers(*threads: threading.Thread | None) -> None:
    for thread in threads:
        if thread is not None:
            thread.join(2.0)


def _parse_worker_output(
    raw: bytes,
    *,
    expected_operation_id: str,
) -> WorkspaceToolResponse:
    if len(raw) > MAX_RESPONSE_BYTES:
        raise WorkspaceToolExecutionError("protocol_failure")
    lines = raw.splitlines()
    if not lines or len(lines) > 2:
        raise WorkspaceToolExecutionError("protocol_failure")
    try:
        frames = [
            WorkspaceToolResponse.from_bytes(
                line,
                expected_operation_id=expected_operation_id,
            )
            for line in lines
        ]
    except WorkspaceProtocolError:
        raise WorkspaceToolExecutionError("protocol_failure") from None
    terminal = frames[-1]
    if len(frames) == 2 and frames[0].outcome != "admitted":
        raise WorkspaceToolExecutionError("protocol_failure")
    if terminal.outcome not in {"success", "failure"}:
        raise WorkspaceToolExecutionError("protocol_failure")
    if not _SAFE_CODE.fullmatch(terminal.code):
        raise WorkspaceToolExecutionError("protocol_failure")
    return terminal


def _terminate_tree(tree: ExecutorProcessTree | None) -> bool:
    if tree is None:
        return True
    try:
        return tree.terminate_tree()
    except Exception:
        return False


def _stop_process(process: subprocess.Popen[bytes] | None) -> bool:
    if process is None or process.poll() is not None:
        return True
    try:
        process.terminate()
        process.wait(timeout=2.0)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)
        return process.poll() is not None
    except (OSError, subprocess.TimeoutExpired):
        return False


__all__ = [
    "DIAGNOSTIC_STDERR_MAX_BYTES",
    "WorkspaceToolExecutionError",
    "WorkspaceToolExecutor",
    "workspace_worker_environment",
]
