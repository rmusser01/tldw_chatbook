"""Parent-side one-shot executor for pinned workspace tool requests."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Callable

from tldw_chatbook.STT.executor_process_tree import (
    ExecutorProcessTree,
    ProcessContainmentError,
    WorkerContainmentIdentity,
)
from tldw_chatbook.Tools.local_tool_impls import resolve_workspace_path
from tldw_chatbook.Tools.patch_tool_impls import parse_patch_targets
from tldw_chatbook.Tools.remote_root_types import (
    AdmittedRoot,
    LocalRoot,
    RemoteRoot,
)
from tldw_chatbook.Tools.workspace_tool_protocol import (
    MAX_RESPONSE_BYTES,
    WorkspaceProtocolError,
    WorkspaceToolRequest,
    WorkspaceToolResponse,
)
from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryChain,
    DirectoryIdentity,
    DirectoryIdentityError,
    capture_directory_chain,
)
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Utils.sensitive_paths import (
    SensitiveExclusion,
    SensitivePathContext,
    merge_sensitive_context,
    resolve_sensitive_context,
    sensitive_exclusions_under,
)

WORKSPACE_HELPER_TIMEOUT_SECONDS = 300
DIAGNOSTIC_STDERR_MAX_BYTES = 8 * 1024
_FIXED_WORKER_MODULE = "tldw_chatbook.Tools.workspace_tool_worker"
_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_CLEANUP_RESERVE_SECONDS = 4.0
_READ_OPERATIONS = frozenset({"fs_list", "fs_read", "fs_glob", "fs_grep"})
_GIT_OPERATIONS = frozenset(
    {"git_status", "git_diff", "git_log", "git_blame", "git_branches"}
)


class WorkspaceToolExecutionError(RuntimeError):
    """A stable code plus optional protocol-validated worker failure text."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(
            message if message is not None else f"workspace operation failed ({code})"
        )


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


def workspace_worker_environment(workspace_root: Path) -> dict[str, str]:
    """Return the fixed runtime allowlist inherited by the isolated helper.

    Args:
        workspace_root: Admitted root whose contents must not supply executables.

    Returns:
        A small environment with only absolute, external PATH entries and the
        minimum platform variables needed by the child runtime.
    """
    root = workspace_root.resolve()
    path_entries: list[str] = []
    seen: set[str] = set()
    for raw_entry in os.environ.get("PATH", os.defpath).split(os.pathsep):
        entry = Path(raw_entry)
        if not raw_entry or not entry.is_absolute():
            continue
        resolved = entry.resolve()
        if resolved == root or root in resolved.parents:
            continue
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            path_entries.append(str(resolved))
    environment = {
        "PATH": os.pathsep.join(path_entries),
        "NoDefaultCurrentDirectoryInExePath": "1",
    }
    if os.name == "posix":
        environment.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    else:
        for key in ("SYSTEMROOT", "WINDIR"):
            value = os.environ.get(key)
            if value:
                environment[key] = value
    return environment


class WorkspaceToolExecutor:
    """Launch one contained helper for one workspace operation.

    Phase 3a (task 15) added a remote request-building mode: constructed
    with a :class:`~tldw_chatbook.Tools.remote_root_types.RemoteRoot`
    descriptor the executor validates and normalizes requests LEXICALLY
    and carries the binding-cache identity -- it never touches the
    laptop's disk and :meth:`execute` refuses to spawn a local worker
    (spawning is the ssh transport executor's job; Task 14 built it).
    A ``LocalRoot`` (or plain ``Path``) keeps the historical fd-based
    behavior byte-identically.
    """

    def __init__(
        self,
        workspace_root: "Path | AdmittedRoot",
        *,
        user_exclusion_paths: Callable[[], tuple[Path, ...]] | None = None,
        remote_identity_source: Callable[[], list[list[object]] | None] | None = None,
        remote_sensitive_exclusions: (
            Callable[[], tuple[SensitiveExclusion, ...]] | None
        ) = None,
    ) -> None:
        if isinstance(workspace_root, RemoteRoot):
            if remote_identity_source is None:
                raise ValueError(
                    "remote roots require remote_identity_source (the binding "
                    "status cache's captured identity chain)"
                )
            if not callable(remote_identity_source):
                raise ValueError("remote_identity_source must be callable")
            # No validate_path, no capture_directory_chain, no laptop stat:
            # the descriptor is pure data and identity is cache-owned.
            self._remote_root: RemoteRoot | None = workspace_root
            self._workspace_root: Path | None = None
            self._authority_chain: DirectoryChain | None = None
            self._remote_identity_source = remote_identity_source
            self._remote_sensitive_exclusions = remote_sensitive_exclusions
            self._user_exclusion_paths = user_exclusion_paths
            return
        self._remote_root = None
        if isinstance(workspace_root, LocalRoot):
            workspace_root = workspace_root.path
        candidate = Path(os.path.abspath(workspace_root))
        try:
            self._workspace_root = validate_path(
                candidate,
                candidate,
                redact_paths=True,
                allow_hidden=True,
            )
        except ValueError:
            raise WorkspaceToolExecutionError("invalid_request") from None
        self._authority_chain: DirectoryChain = capture_directory_chain(
            self._workspace_root
        )
        self._user_exclusion_paths = user_exclusion_paths

    def _call_context(self) -> SensitivePathContext:
        """Per-call deny context: base denylist merged with user exclusions."""
        base = resolve_sensitive_context()
        if self._user_exclusion_paths is None:
            return base
        try:
            extra = tuple(self._user_exclusion_paths())
        except Exception:  # provider failure falls back to the base denylist
            return base
        if not extra:
            return base
        return merge_sensitive_context(
            base,
            extra_files=tuple(path for path in extra if path.is_file()),
            extra_dirs=tuple(path for path in extra if not path.is_file()),
        )

    def execute(
        self,
        operation: str,
        arguments: dict[str, Any],
        *,
        intent: str,
    ) -> str:
        """Validate, execute once, prove cleanup, and return bounded text.

        Args:
            operation: Closed workspace protocol operation name.
            arguments: Operation-specific arguments to validate and serialize.
            intent: Required access intent for the requested operation.

        Returns:
            Bounded text returned by the successful one-shot worker.

        Raises:
            WorkspaceToolExecutionError: If admission, validation, execution,
                protocol handling, containment, or cleanup fails.
        """
        if self._remote_root is not None:
            # A local helper spawn would read the LAPTOP's filesystem
            # against a remote root's POSIX path -- exactly the silent
            # laptop IO Phase 3a exists to make impossible. Spawning for
            # remote roots is the ssh transport executor's job.
            raise NotImplementedError(
                "remote root reached laptop-disk path: local worker spawn "
                "(use the ssh transport executor)"
            )
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
        cleanup_supervisor: _CleanupSupervisor | None = None
        cleanup_attempted = False
        try:
            argv = [sys.executable, "-I", "-m", _FIXED_WORKER_MODULE]
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=workspace_worker_environment(self._workspace_root),
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
            candidate_supervisor = _CleanupSupervisor(tree, process, deadline)
            candidate_supervisor.start()
            cleanup_supervisor = candidate_supervisor
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
            operation_deadline = _operation_phase_deadline(deadline)
            try:
                process.wait(timeout=_remaining_seconds(operation_deadline))
            except subprocess.TimeoutExpired:
                cleanup_attempted = True
                cleanup = _settle_after_spawn(
                    cleanup_supervisor,
                    tree,
                    process,
                    deadline,
                )
                _join_threads_until(deadline, writer_thread, stdout_thread, stderr_thread)
                if not cleanup:
                    raise WorkspaceToolExecutionError("cleanup_unproven") from None
                raise WorkspaceToolExecutionError("worker_timed_out") from None

            if not _join_threads_until(
                operation_deadline,
                writer_thread,
                stdout_thread,
                stderr_thread,
            ):
                raise WorkspaceToolExecutionError("worker_timed_out")
            cleanup_attempted = True
            cleanup = _settle_after_spawn(
                cleanup_supervisor,
                tree,
                process,
                deadline,
            )
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven")
            if writer_errors:
                raise WorkspaceToolExecutionError("worker_crashed")
            try:
                response = _parse_worker_output(
                    b"".join(stdout_capture),
                    expected_operation_id=request.operation_id,
                )
            except WorkspaceToolExecutionError:
                if process.returncode != 0:
                    raise WorkspaceToolExecutionError("worker_crashed") from None
                raise
            if not response.cleanup_proven:
                raise WorkspaceToolExecutionError("cleanup_unproven")
            if response.outcome == "failure":
                raise WorkspaceToolExecutionError(response.code, response.error)
            if process.returncode != 0:
                raise WorkspaceToolExecutionError("worker_crashed")
            if response.result is None:
                raise WorkspaceToolExecutionError("protocol_failure")
            return response.result
        except WorkspaceToolExecutionError:
            if process is not None and not cleanup_attempted:
                cleanup_attempted = True
                cleanup = _settle_after_spawn(
                    cleanup_supervisor,
                    tree,
                    process,
                    deadline,
                )
                if not cleanup:
                    raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise
        except ProcessContainmentError:
            cleanup_attempted = True
            cleanup = _settle_after_spawn(
                cleanup_supervisor,
                tree,
                process,
                deadline,
            )
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise WorkspaceToolExecutionError("containment_unavailable") from None
        except (OSError, ValueError):
            cleanup_attempted = True
            cleanup = _settle_after_spawn(
                cleanup_supervisor,
                tree,
                process,
                deadline,
            )
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise WorkspaceToolExecutionError("spawn_failed") from None
        except Exception:
            cleanup_attempted = True
            cleanup = _settle_after_spawn(
                cleanup_supervisor,
                tree,
                process,
                deadline,
            )
            if not cleanup:
                raise WorkspaceToolExecutionError("cleanup_unproven") from None
            raise WorkspaceToolExecutionError("worker_failure") from None
        except BaseException:
            cleanup_attempted = True
            _settle_after_spawn(
                cleanup_supervisor,
                tree,
                process,
                deadline,
            )
            raise
        finally:
            active_error = sys.exception()
            pipes_closed, cleanup_cancellation = _close_process_pipes_until(
                process,
                deadline,
            )
            threads_joined = False
            try:
                threads_joined = _join_threads_until(
                    deadline,
                    writer_thread,
                    stdout_thread,
                    stderr_thread,
                )
            except BaseException as error:
                if cleanup_cancellation is None and not isinstance(error, Exception):
                    cleanup_cancellation = error
            writer_errors.clear()
            stdout_capture.clear()
            stderr_capture.clear()
            active_cancellation = (
                active_error is not None and not isinstance(active_error, Exception)
            )
            if not active_cancellation:
                if cleanup_cancellation is not None:
                    raise cleanup_cancellation
                if not pipes_closed or not threads_joined:
                    raise WorkspaceToolExecutionError("cleanup_unproven") from None

    def _build_request(
        self,
        operation: str,
        arguments: dict[str, Any],
        *,
        intent: str,
    ) -> WorkspaceToolRequest:
        if self._remote_root is not None:
            return self._build_remote_request(operation, arguments, intent=intent)
        try:
            current_chain = capture_directory_chain(self._workspace_root)
            if current_chain != self._authority_chain:
                raise WorkspaceToolExecutionError("root_pin_failed")
            chain = self._authority_chain
        except (DirectoryIdentityError, OSError, ValueError):
            raise WorkspaceToolExecutionError("root_pin_failed") from None

        try:
            normalized = dict(arguments)
            if operation == "stat_path" and type(arguments) is dict:
                raw_path = arguments.get("path")
                if type(raw_path) is str:
                    target = resolve_workspace_path(
                        raw_path,
                        chain.canonical_root,
                        intent="read",
                        context=self._call_context(),
                    )
                    normalized["path"] = str(target.relative_to(chain.canonical_root))
            if operation in _GIT_OPERATIONS and type(arguments) is dict:
                context = self._call_context()
                root = resolve_workspace_path(
                    ".", chain.canonical_root, intent="list", context=context
                )
                exclusions, _ = _parent_read_exclusions(root, context)
                normalized["sensitive_exclusions"] = _serialize_exclusions(
                    exclusions
                )
                raw_path = arguments.get("path")
                if raw_path is not None:
                    if type(raw_path) is not str:
                        raise ValueError("invalid git path")
                    target = resolve_workspace_path(
                        raw_path, root, intent="read", context=context
                    )
                    normalized["path"] = target.relative_to(root).as_posix() or "."
            if operation in _READ_OPERATIONS and type(arguments) is dict:
                context = self._call_context()
                root = resolve_workspace_path(
                    ".", chain.canonical_root, intent="list", context=context
                )
                exclusions, content_exclusions = _parent_read_exclusions(root, context)
                normalized["sensitive_exclusions"] = _serialize_exclusions(exclusions)
                if operation == "fs_grep":
                    normalized["content_exclusions"] = _serialize_exclusions(
                        content_exclusions
                    )
                if operation in {"fs_list", "fs_read"}:
                    raw_path = arguments.get("path")
                    if type(raw_path) is not str:
                        raise ValueError("invalid read path")
                    resolve_workspace_path(
                        raw_path,
                        root,
                        intent="list" if operation == "fs_list" else "read",
                        context=context,
                    )
                    normalized["path"] = _normalize_relative_path(raw_path)
                if operation == "fs_glob":
                    raw_pattern = arguments.get("pattern")
                    if type(raw_pattern) is not str:
                        raise ValueError("invalid glob pattern")
                    normalized["pattern"] = _normalize_glob_pattern(raw_pattern)
            if operation in {"fs_write", "fs_edit"} and type(arguments) is dict:
                raw_path = arguments.get("path")
                if type(raw_path) is not str:
                    raise ValueError("invalid mutation path")
                context = self._call_context()
                resolve_workspace_path(
                    raw_path,
                    chain.canonical_root,
                    intent="write",
                    context=context,
                )
                normalized["path"] = _normalize_relative_path(raw_path)
                exclusions, _ = _parent_read_exclusions(
                    chain.canonical_root, context
                )
                normalized["sensitive_exclusions"] = _serialize_exclusions(
                    exclusions
                )
            if operation == "fs_patch" and type(arguments) is dict:
                raw_diff = arguments.get("diff")
                if type(raw_diff) is not str:
                    raise ValueError("invalid patch")
                context = self._call_context()
                targets: list[str] = []
                for patch_file in parse_patch_targets(raw_diff):
                    rel_path = patch_file.new_path
                    if rel_path is None:
                        raise ValueError("invalid patch target")
                    resolve_workspace_path(
                        rel_path,
                        chain.canonical_root,
                        intent="write",
                        context=context,
                    )
                    targets.append(_normalize_relative_path(rel_path))
                normalized["targets"] = targets
                exclusions, _ = _parent_read_exclusions(
                    chain.canonical_root, context
                )
                normalized["sensitive_exclusions"] = _serialize_exclusions(
                    exclusions
                )
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
        except WorkspaceToolExecutionError:
            raise
        except (DirectoryIdentityError, WorkspaceProtocolError, OSError, ValueError):
            raise WorkspaceToolExecutionError("invalid_request") from None

    def _remote_serialized_exclusions(self) -> list[dict[str, str]]:
        """Serialize caller-supplied relative exclusions verbatim.

        The remote builder NEVER resolves exclusions against the laptop's
        filesystem (``_parent_read_exclusions``' laptop walk is
        local-only): the caller hands in already-relative
        :class:`SensitiveExclusion` entries -- Task 17 routes the binding's
        stored relative paths here -- and they ride the request as-is for
        worker-side matching.
        """
        if self._remote_sensitive_exclusions is None:
            return []
        try:
            return _serialize_exclusions(tuple(self._remote_sensitive_exclusions()))
        except Exception:  # noqa: BLE001 - a failed provider degrades to none
            return []

    def _build_remote_request(
        self,
        operation: str,
        arguments: dict[str, Any],
        *,
        intent: str,
    ) -> WorkspaceToolRequest:
        """Build one request for a remote root with ZERO laptop filesystem access.

        Mirrors the LOCAL builder's per-operation argument admission using
        lexical rules only (see :func:`_normalize_remote_relative_path`);
        identity comes from the binding status cache (the ping-captured
        chain), never a laptop ``capture_directory_chain``; exclusions are
        serialized from caller-supplied relative entries. The emitted
        ``WorkspaceToolRequest`` wire dict is shape-identical to a local
        one, so the remote worker (same bundle) parses it unchanged.
        """
        try:
            chain = self._remote_identity_source()
            if not chain:
                # The cached identity IS the root pin (the ping captured
                # it); no chain means the pin was never proven here.
                raise WorkspaceToolExecutionError("root_pin_failed")
            identities = tuple(
                DirectoryIdentity(
                    device=int(entry[1]),
                    inode=int(entry[2]),
                    mode=int(entry[3]),
                    reparse=False,
                )
                for entry in chain
            )
            locator = str(chain[0][0])
            exclusions = self._remote_serialized_exclusions()
            normalized = dict(arguments)
            if operation == "stat_path" and type(arguments) is dict:
                raw_path = arguments.get("path")
                if type(raw_path) is str:
                    normalized["path"] = _normalize_remote_relative_path(raw_path)
            if operation in _GIT_OPERATIONS and type(arguments) is dict:
                normalized["sensitive_exclusions"] = exclusions
                raw_path = arguments.get("path")
                if raw_path is not None:
                    if type(raw_path) is not str:
                        raise ValueError("invalid git path")
                    normalized["path"] = (
                        _normalize_remote_relative_path(raw_path) or "."
                    )
            if operation in _READ_OPERATIONS and type(arguments) is dict:
                normalized["sensitive_exclusions"] = exclusions
                if operation == "fs_grep":
                    normalized["content_exclusions"] = exclusions
                if operation in {"fs_list", "fs_read"}:
                    raw_path = arguments.get("path")
                    if type(raw_path) is not str:
                        raise ValueError("invalid read path")
                    normalized["path"] = _normalize_remote_relative_path(raw_path)
                if operation == "fs_glob":
                    raw_pattern = arguments.get("pattern")
                    if type(raw_pattern) is not str:
                        raise ValueError("invalid glob pattern")
                    normalized["pattern"] = _normalize_glob_pattern(raw_pattern)
            if operation in {"fs_write", "fs_edit"} and type(arguments) is dict:
                raw_path = arguments.get("path")
                if type(raw_path) is not str:
                    raise ValueError("invalid mutation path")
                normalized["path"] = _normalize_remote_relative_path(
                    raw_path, intent="write"
                )
                normalized["sensitive_exclusions"] = exclusions
            if operation == "fs_patch" and type(arguments) is dict:
                raw_diff = arguments.get("diff")
                if type(raw_diff) is not str:
                    raise ValueError("invalid patch")
                targets: list[str] = []
                for patch_file in parse_patch_targets(raw_diff):
                    rel_path = patch_file.new_path
                    if rel_path is None:
                        raise ValueError("invalid patch target")
                    targets.append(
                        _normalize_remote_relative_path(rel_path, intent="write")
                    )
                normalized["targets"] = targets
                normalized["sensitive_exclusions"] = exclusions
            request = WorkspaceToolRequest(
                operation_id=uuid.uuid4().hex,
                operation=operation,  # type: ignore[arg-type]
                intent=intent,  # type: ignore[arg-type]
                root_locator=Path(locator),
                root_identity=identities[0],
                ancestor_identities=identities,
                arguments=normalized,
                timeout_seconds=WORKSPACE_HELPER_TIMEOUT_SECONDS,
                output_max_bytes=MAX_RESPONSE_BYTES,
            )
            return WorkspaceToolRequest.from_bytes(request.to_bytes())
        except WorkspaceToolExecutionError:
            raise
        except (
            WorkspaceProtocolError,
            ValueError,
            TypeError,
            IndexError,
            KeyError,
        ):
            raise WorkspaceToolExecutionError("invalid_request") from None


def _normalize_remote_relative_path(path: str, *, intent: str = "read") -> str:
    """Lexically confine one root-relative remote path; no filesystem access.

    The remote mirror of the checks the LOCAL builder gets from
    ``resolve_workspace_path`` + ``_normalize_relative_path``, reduced to
    what is decidable without the laptop's disk:

    - absolute, drive, or anchored forms are rejected (checked through
      BOTH ``PurePosixPath`` and ``PureWindowsPath`` so a Windows-rooted
      form is refused on any parent host);
    - ``..`` components are rejected under BOTH separator conventions --
      with no resolved ancestry available to the parent, the component
      check IS the confinement;
    - NUL is rejected;
    - hidden components are ALLOWED under the root (ADR-032 parity with
      the family's ``allow_hidden`` policy: a coding agent that cannot
      read ``.github/`` is useless);
    - write intents refuse an exact ``.git`` path component (case-insensitive,
      matching :func:`tldw_chatbook.Utils.sensitive_paths.is_git_metadata_write`)
      -- the one hidden-path rule that is purely lexical, so remote write
      admission is never WEAKER than the local one. The denylist proper
      rides the request as serialized exclusions and is matched by the
      worker (Task 17).
    """
    if "\x00" in path:
        raise ValueError("invalid workspace path")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if (
        posix.is_absolute()
        or posix.anchor
        or windows.is_absolute()
        or windows.root
        or windows.drive
        or windows.anchor
    ):
        raise ValueError("invalid workspace path")
    for part in (*posix.parts, *windows.parts):
        if part == "..":
            raise ValueError("invalid workspace path")
    if intent == "write" and any(part.lower() == ".git" for part in posix.parts):
        raise ValueError("invalid workspace path")
    return posix.as_posix()


def _normalize_glob_pattern(pattern: str) -> str:
    """Reject lexical glob routes that could traverse outside a pinned root."""
    windows = PureWindowsPath(pattern)
    if (
        Path(pattern).is_absolute()
        or windows.is_absolute()
        or windows.root
        or windows.drive
        or any(part == ".." for part in pattern.replace("\\", "/").split("/"))
    ):
        raise ValueError("invalid glob pattern")
    return pattern


def _normalize_relative_path(path: str) -> str:
    """Retain a lexical relative target after parent-side policy admission."""
    for lexical in (PurePosixPath(path), PureWindowsPath(path)):
        if lexical.drive or lexical.root or lexical.anchor:
            raise ValueError("invalid workspace path")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or "\x00" in path:
        raise ValueError("invalid workspace path")
    return relative.as_posix()


def _parent_read_exclusions(
    root: Path, context: Any
) -> tuple[tuple[SensitiveExclusion, ...], tuple[SensitiveExclusion, ...]]:
    """Capture sensitive and unsafe symlink aliases before worker launch."""
    exclusions = list(sensitive_exclusions_under(root, context))
    content_exclusions = list(exclusions)
    return tuple(dict.fromkeys(exclusions)), tuple(dict.fromkeys(content_exclusions))


def _serialize_exclusions(exclusions: tuple[SensitiveExclusion, ...]) -> list[dict[str, str]]:
    """Serialize bounded parent exclusions into the closed worker request."""
    return [{"kind": exclusion.kind, "value": exclusion.value} for exclusion in exclusions]


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


def _operation_phase_deadline(deadline: float) -> float:
    remaining = _remaining_seconds(deadline)
    reserve = min(_CLEANUP_RESERVE_SECONDS, remaining / 5.0)
    return deadline - reserve


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


class _CleanupSupervisor:
    """Pre-started deadline guard for every post-authority settlement path."""

    def __init__(
        self,
        tree: ExecutorProcessTree,
        process: subprocess.Popen[bytes],
        deadline: float,
    ) -> None:
        self._tree = tree
        self._process = process
        self._deadline = deadline
        self._requested = threading.Event()
        self._outcome: list[bool] = []
        self._thread = threading.Thread(
            target=self._run,
            name="workspace-worker-cleanup",
            daemon=True,
        )

    def start(self) -> None:
        """Establish the supervisor before any request bytes are written."""
        self._thread.start()

    def settle(self) -> bool:
        """Request settlement and wait only through the caller's deadline."""
        self._requested.set()
        try:
            self._thread.join(_remaining_seconds(self._deadline))
        except Exception:
            return False
        return bool(self._outcome and self._outcome[0])

    def _run(self) -> None:
        self._requested.wait()
        try:
            self._outcome.append(
                _settle_process_blocking(
                    self._tree,
                    self._process,
                    self._deadline,
                )
            )
        except BaseException:
            self._outcome.append(False)


def _settle_after_spawn(
    supervisor: _CleanupSupervisor | None,
    tree: ExecutorProcessTree | None,
    process: subprocess.Popen[bytes] | None,
    deadline: float,
) -> bool:
    if process is None:
        return True
    if supervisor is not None:
        return supervisor.settle()
    return _settle_without_supervisor(tree, process)


def _settle_without_supervisor(
    tree: ExecutorProcessTree | None,
    process: subprocess.Popen[bytes],
) -> bool:
    """Attempt immediate nonblocking containment when supervision could not start."""
    if tree is not None:
        try:
            return tree.terminate_tree(term_timeout=0.0, kill_timeout=0.0)
        except BaseException:
            return False
    if process.poll() is not None:
        return True
    try:
        process.terminate()
    except BaseException:
        pass
    try:
        process.wait(timeout=0.0)
    except BaseException:
        pass
    if process.poll() is None:
        try:
            process.kill()
        except BaseException:
            pass
        try:
            process.wait(timeout=0.0)
        except BaseException:
            pass
    return process.poll() is not None


def _settle_process_blocking(
    tree: ExecutorProcessTree | None,
    process: subprocess.Popen[bytes],
    deadline: float,
) -> bool:
    remaining = _remaining_seconds(deadline)
    term_timeout = remaining / 2.0
    kill_timeout = remaining - term_timeout
    if tree is not None:
        # ExecutorProcessTree may use each phase timeout in two sequential waits
        # (leader/job and group). Quarter budgets keep both phases inside the
        # remaining outer window; the supervising thread is the final guard.
        term_timeout = remaining / 4.0
        kill_timeout = remaining / 4.0
        return tree.terminate_tree(
            term_timeout=term_timeout,
            kill_timeout=kill_timeout,
        )
    if process.poll() is not None:
        return True
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=min(term_timeout, _remaining_seconds(deadline)))
    except (OSError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=min(kill_timeout, _remaining_seconds(deadline)))
        except (OSError, subprocess.TimeoutExpired):
            pass
    return process.poll() is not None


def _close_process_pipes_until(
    process: subprocess.Popen[bytes] | None,
    deadline: float,
) -> tuple[bool, BaseException | None]:
    if process is None:
        return True, None

    def close(stream: BinaryIO) -> None:
        try:
            stream.close()
        except BaseException:
            pass

    threads: list[threading.Thread] = []
    cleanup_proven = True
    cancellation: BaseException | None = None
    for index, stream in enumerate((process.stdin, process.stdout, process.stderr)):
        if stream is not None:
            try:
                thread = threading.Thread(
                    target=close,
                    args=(stream,),
                    name=f"workspace-worker-pipe-close-{index}",
                    daemon=True,
                )
                thread.start()
                threads.append(thread)
            except BaseException as error:
                cleanup_proven = False
                if cancellation is None and not isinstance(error, Exception):
                    cancellation = error
    for thread in threads:
        try:
            thread.join(_remaining_seconds(deadline))
            if thread.is_alive():
                cleanup_proven = False
        except BaseException as error:
            cleanup_proven = False
            if cancellation is None and not isinstance(error, Exception):
                cancellation = error
    return cleanup_proven, cancellation


__all__ = [
    "DIAGNOSTIC_STDERR_MAX_BYTES",
    "WorkspaceToolExecutionError",
    "WorkspaceToolExecutor",
    "workspace_worker_environment",
]
