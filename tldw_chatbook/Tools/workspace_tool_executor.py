"""Parent-side one-shot executor for pinned workspace tool requests."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
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
_CLEANUP_RESERVE_SECONDS = 4.0


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
        deadline = time.monotonic() + request.timeout_seconds
        process: subprocess.Popen[bytes] | None = None
        tree: ExecutorProcessTree | None = None
        stdout_capture: list[bytes] = []
        stderr_capture: list[bytes] = []
        writer_errors: list[BaseException] = []
        writer_thread: threading.Thread | None = None
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        cleanup_attempted = False
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
            writer_thread = _start_request_writer(
                process.stdin,
                request.to_bytes(),
                writer_errors,
            )
            try:
                process.wait(timeout=_operation_wait_budget(deadline))
            except subprocess.TimeoutExpired:
                cleanup_attempted = True
                cleanup = _settle_process(tree, process, deadline)
                _join_threads_until(deadline, writer_thread, stdout_thread, stderr_thread)
                if not cleanup:
                    raise WorkspaceToolExecutionError("cleanup_unproven") from None
                raise WorkspaceToolExecutionError("worker_timed_out") from None

            if not _join_threads_until(
                deadline,
                writer_thread,
                stdout_thread,
                stderr_thread,
            ):
                raise WorkspaceToolExecutionError("worker_timed_out")
            cleanup_attempted = True
            cleanup = _settle_process(tree, process, deadline)
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven")
            if process.returncode != 0:
                raise WorkspaceToolExecutionError("worker_crashed")
            if writer_errors:
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
            if process is not None and not cleanup_attempted:
                cleanup_attempted = True
                cleanup = _settle_process(tree, process, deadline)
                if not cleanup:
                    raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise
        except ProcessContainmentError:
            cleanup_attempted = True
            cleanup = _settle_process(tree, process, deadline)
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise WorkspaceToolExecutionError("containment_unavailable") from None
        except (OSError, ValueError):
            cleanup_attempted = True
            cleanup = _settle_process(tree, process, deadline)
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise WorkspaceToolExecutionError("spawn_failed") from None
        except Exception:
            cleanup_attempted = True
            cleanup = _settle_process(tree, process, deadline)
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise WorkspaceToolExecutionError("worker_failure") from None
        except BaseException:
            cleanup_attempted = True
            _settle_process(tree, process, deadline)
            raise
        finally:
            _close_process_pipes(process)
            _join_threads_until(deadline, writer_thread, stdout_thread, stderr_thread)
            writer_errors.clear()
            stderr_capture.clear()

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


def _start_request_writer(
    stream: BinaryIO,
    request: bytes,
    errors: list[BaseException],
) -> threading.Thread:
    def write() -> None:
        try:
            stream.write(request)
        except BaseException as error:
            errors.append(error)
        finally:
            try:
                stream.close()
            except BaseException as error:
                errors.append(error)

    thread = threading.Thread(target=write, name="workspace-worker-stdin", daemon=True)
    thread.start()
    return thread


def _join_threads_until(
    deadline: float,
    *threads: threading.Thread | None,
) -> bool:
    for thread in threads:
        if thread is not None:
            thread.join(_remaining_seconds(deadline))
    return all(thread is None or not thread.is_alive() for thread in threads)


def _operation_wait_budget(deadline: float) -> float:
    remaining = _remaining_seconds(deadline)
    reserve = min(_CLEANUP_RESERVE_SECONDS, remaining / 5.0)
    return max(0.0, remaining - reserve)


def _remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


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
    if len(frames) == 2:
        admitted = frames[0]
        if (
            admitted.outcome != "admitted"
            or admitted.code != "root_pinned"
            or admitted.result is not None
            or admitted.error is not None
            or admitted.truncated
            or not admitted.cleanup_proven
            or admitted.elapsed_ms > terminal.elapsed_ms
        ):
            raise WorkspaceToolExecutionError("protocol_failure")
    if terminal.outcome not in {"success", "failure"}:
        raise WorkspaceToolExecutionError("protocol_failure")
    if not _SAFE_CODE.fullmatch(terminal.code):
        raise WorkspaceToolExecutionError("protocol_failure")
    return terminal


def _settle_process(
    tree: ExecutorProcessTree | None,
    process: subprocess.Popen[bytes] | None,
    deadline: float,
) -> bool:
    if process is None:
        return True
    remaining = _remaining_seconds(deadline)
    term_timeout = remaining / 2.0
    kill_timeout = remaining - term_timeout
    try:
        if tree is not None:
            return tree.terminate_tree(
                term_timeout=term_timeout,
                kill_timeout=kill_timeout,
            )
        if process.poll() is not None:
            return True
        process.terminate()
        process.wait(timeout=term_timeout)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=kill_timeout)
        return process.poll() is not None
    except (OSError, subprocess.TimeoutExpired):
        return False


def _close_process_pipes(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass


__all__ = [
    "DIAGNOSTIC_STDERR_MAX_BYTES",
    "WorkspaceToolExecutionError",
    "WorkspaceToolExecutor",
    "workspace_worker_environment",
]
