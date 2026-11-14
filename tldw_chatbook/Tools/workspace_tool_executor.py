"""Parent-side execution and custody for local and SSH workspace tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Event
from typing import Any, BinaryIO, Callable

from tldw_chatbook.STT.executor_process_tree import (
    ExecutorProcessTree,
    ProcessContainmentError,
    WorkerContainmentIdentity,
)
from tldw_chatbook.Tools.local_tool_impls import (
    _is_relative_sensitive_path,
    resolve_workspace_path,
)
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
    refuses_new_directory_chain,
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


_current = ContextVar("native_model_ssh", default=None)


def current_ssh_call():
    """Return only the attachment carried by the original model tool worker."""
    return _current.get()


def bind_model_ssh(provider, authority, spec):
    call = _current.get()
    if call is not None:
        call.bind(provider, authority, spec)


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
            if user_exclusion_paths is not None:
                # Task 17: ``user_exclusion_paths`` classifies entries with
                # LAPTOP stats (``_call_context``'s ``path.is_file()``) --
                # exactly the silent laptop IO Phase 3 forbids for remote
                # roots. Remote exclusions arrive pre-classified as
                # ``SensitiveExclusion`` entries through
                # ``remote_sensitive_exclusions`` (file-vs-dir is decided by
                # the stored binding kind; a worker stat is available
                # through the existing ``stat_path`` wire op where a caller
                # genuinely needs one -- no new op was added).
                raise ValueError(
                    "remote roots classify exclusions worker-side: pass "
                    "remote_sensitive_exclusions, not user_exclusion_paths"
                )
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

    @property
    def local_authority_chain(self) -> DirectoryChain | None:
        """Return the immutable captured local pin; SSH has no local chain."""
        return self._authority_chain

    def _call_context(self) -> SensitivePathContext:
        """Per-call deny context: base denylist merged with user exclusions.

        LOCAL roots only. A remote root must never reach this method: the
        file-vs-dir classification of user exclusions is a ``path.is_file()``
        stat, and for a remote binding that stat would hit the LAPTOP's
        same-named path -- the exact wrong-file hazard Phase 3 closes.
        Task 17's answer to "how does a remote root answer is_file": it
        doesn't need to (binding exclusions arrive pre-classified as
        ``SensitiveExclusion`` entries), and a caller that genuinely needs
        a remote stat uses the EXISTING ``stat_path`` wire op (schema,
        both decoders, and corpus already shipped in Tasks 8-13) through
        the ssh/loopback transport executor -- no new op was added.
        """
        if self._remote_root is not None:
            raise RuntimeError(
                "remote root reached laptop-disk path: _call_context "
                "exclusion classification (remote exclusions ride the "
                "request as serialized SensitiveExclusion entries)"
            )
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

            if (
                process.stdout is None
                or process.stderr is None
                or process.stdin is None
            ):
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
                _join_threads_until(
                    deadline, writer_thread, stdout_thread, stderr_thread
                )
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
            active_cancellation = active_error is not None and not isinstance(
                active_error, Exception
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
                normalized["sensitive_exclusions"] = _serialize_exclusions(exclusions)
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
            if (
                operation in {"fs_write", "fs_edit", "fs_mkdir"}
                and type(arguments) is dict
            ):
                raw_path = arguments.get("path")
                if type(raw_path) is not str:
                    raise ValueError("invalid mutation path")
                context = self._call_context()
                resolved = resolve_workspace_path(
                    raw_path,
                    chain.canonical_root,
                    intent="write",
                    context=context,
                )
                if operation == "fs_mkdir" and refuses_new_directory_chain(
                    resolved, context=context
                ):
                    # The write-intent choke point guards the ancestors of
                    # a FILE target; mkdir's leaf IS a new directory, so
                    # the shadow-directory walk must start at the target
                    # itself (TASK-849 guard, applied per fs_mkdir).
                    raise ValueError("invalid mutation path")
                normalized["path"] = _normalize_relative_path(raw_path)
                exclusions, _ = _parent_read_exclusions(chain.canonical_root, context)
                normalized["sensitive_exclusions"] = _serialize_exclusions(exclusions)
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
                exclusions, _ = _parent_read_exclusions(chain.canonical_root, context)
                normalized["sensitive_exclusions"] = _serialize_exclusions(exclusions)
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

    def _remote_exclusion_entries(self) -> tuple[SensitiveExclusion, ...]:
        """The caller-supplied relative exclusion entries, fail-closed.

        Shared by the wire serializer and the lexical ``stat_path``
        admission: a provider that raises means the exclusion state is
        INDETERMINATE, and the only closed outcome is refusing the
        operation (never an empty list -- see
        :meth:`_remote_serialized_exclusions`).
        """
        if self._remote_sensitive_exclusions is None:
            return ()
        try:
            return tuple(self._remote_sensitive_exclusions())
        except Exception:  # noqa: BLE001 - indeterminate exclusions refuse the call
            raise WorkspaceToolExecutionError("invalid_request") from None

    def _remote_serialized_exclusions(self) -> list[dict[str, str]]:
        """Serialize caller-supplied relative exclusions verbatim, fail-closed.

        The remote builder NEVER resolves exclusions against the laptop's
        filesystem (``_parent_read_exclusions``' laptop walk is
        local-only): the caller hands in already-relative
        :class:`SensitiveExclusion` entries -- the binding's stored
        relative paths (Task 17) -- and they ride the request as-is for
        worker-side matching.

        Task 17 fail-closed choice: an exclusions provider that raises
        REFUSES the operation (``invalid_request``). Degrading to ``[]``
        -- the pre-Task-17 behavior -- silently un-excluded every
        sensitive subtree for that one request; deny-all-write was
        considered and declined because reads would still disclose
        excluded content, and ADR-174 makes exclusions fully invisible.
        """
        entries = self._remote_exclusion_entries()
        return _serialize_exclusions(entries) if entries else []

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

        Hidden-path policy (controller-confirmed reading): remote READS
        allow hidden components under the root -- ADR-300 parity with the
        local builder's ``allow_hidden=True`` -- while remote WRITES
        reject a ``.git`` path component, the one lexically-checkable
        hidden-path rule the local side enforces.
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
                    stat_path_value = _normalize_remote_relative_path(raw_path)
                    normalized["path"] = stat_path_value
                    # ADR-174 (fully invisible): the stat_path wire schema
                    # carries no exclusions field, so the serialized binding
                    # exclusions refuse LEXICALLY here -- the parent-side
                    # mirror of the local builder's resolve-with-context
                    # admission for stat. The worker-side denylist covers
                    # the remote-home floor in dispatch.
                    entries = self._remote_exclusion_entries()
                    if entries and _is_relative_sensitive_path(
                        PurePosixPath(stat_path_value),
                        entries,
                        is_directory=True,
                    ):
                        raise ValueError("invalid workspace path")
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
            if (
                operation in {"fs_write", "fs_edit", "fs_mkdir"}
                and type(arguments) is dict
            ):
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
    - hidden components are ALLOWED under the root (ADR-300 parity with
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


def _serialize_exclusions(
    exclusions: tuple[SensitiveExclusion, ...],
) -> list[dict[str, str]]:
    """Serialize bounded parent exclusions into the closed worker request."""
    return [
        {"kind": exclusion.kind, "value": exclusion.value} for exclusion in exclusions
    ]


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


# Concrete native local executor sources for ordinary callable containment.
_NATIVE_ORDINARY_EXECUTOR_CLASS = WorkspaceToolExecutor
_NATIVE_ORDINARY_EXECUTOR_METHODS = tuple(
    (name, value)
    for name, value in vars(WorkspaceToolExecutor).items()
    if callable(value) or isinstance(value, (staticmethod, classmethod, property))
)
_NATIVE_ORDINARY_EXECUTOR_HELPERS = tuple(
    (name, value)
    for name, value in tuple(globals().items())
    if type(value).__name__ == "function" and value.__module__ == __name__
)
_NATIVE_ORDINARY_EXECUTOR_FUNCTIONS = tuple(
    (
        fn,
        fn.__code__,
        fn.__defaults__,
        fn.__kwdefaults__,
        tuple((fn.__kwdefaults__ or {}).items()),
        fn.__closure__,
        tuple((cell, cell.cell_contents) for cell in fn.__closure__ or ()),
    )
    for fn in dict.fromkeys(
        value
        for _name, value in (
            *_NATIVE_ORDINARY_EXECUTOR_METHODS,
            *_NATIVE_ORDINARY_EXECUTOR_HELPERS,
        )
        if type(value).__name__ == "function"
    )
)


def capture_model_ssh(registry, name):
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.Agents.virtual_cli_provider import NATIVE_VIRTUAL_CLASS
    from tldw_chatbook.Tools.remote_root_types import RemoteRoot

    from tldw_chatbook.Utils.platform_files import supports_native_process_groups

    selection = registry.resolve_owner_for_name(name)
    if (
        not supports_native_process_groups()
        or selection is None
        or not (name.startswith("fs_") or name == "virtual_cli")
    ):
        return None
    provider = selection[1]
    virtual = name == "virtual_cli" and type(provider) is NATIVE_VIRTUAL_CLASS
    authorities = inspect.getattr_static(provider, "_admitted_roots", None)
    if authorities is not None and type(authorities) is not dict:
        raise RuntimeError("original SSH authority source changed")
    if not (virtual or type(provider) is LocalToolProvider) or not any(
        isinstance(a.root, RemoteRoot) for a in (authorities or {}).values()
    ):
        return None
    if virtual:
        require_virtual_sources(provider)
    return ModelSshCall(registry, name, selection, virtual=virtual)


def virtual_source_records(provider=None):
    """Read only the original finite virtual provider/parser source records."""
    from tldw_chatbook.Agents import virtual_cli_provider as virtual
    from tldw_chatbook.Tools import virtual_cli_impls as commands
    from tldw_chatbook.Tools import workspace_tool_executor as executor

    from tldw_chatbook.Agents.activation import callable_source_current as _function_unchanged

    values, functions = [], []
    native = provider is None or type(provider) is virtual.NATIVE_VIRTUAL_CLASS
    if not native:
        return (), (), False
    for module, target in ((virtual, provider), (commands, None)):
        cls = module.NATIVE_VIRTUAL_CLASS
        values.append(
            (module, cls.__name__, inspect.getattr_static(module, cls.__name__), None)
        )
        native &= values[-1][2] is cls
        for name, original in module._NATIVE_VIRTUAL_METHODS:
            receiver = cls if target is None else target
            actual = inspect.getattr_static(receiver, name, None)
            values.append((receiver, name, actual, None))
            native &= actual is original
        functions.extend(module._NATIVE_VIRTUAL_FUNCTIONS)
    for module, sources in (
        (virtual, virtual._NATIVE_VIRTUAL_IMPORTS),
        (commands, commands._NATIVE_VIRTUAL_HELPERS),
    ):
        for name, original in sources:
            actual = inspect.getattr_static(module, name, None)
            values.append((module, name, actual, None))
            native &= actual is original
    cls = executor._NATIVE_ORDINARY_EXECUTOR_CLASS
    for obj, sources in (
        (
            executor,
            (
                ("WorkspaceToolExecutor", cls),
                *executor._NATIVE_ORDINARY_EXECUTOR_HELPERS,
            ),
        ),
        (cls, executor._NATIVE_ORDINARY_EXECUTOR_METHODS),
    ):
        for name, original in sources:
            actual = inspect.getattr_static(obj, name, None)
            values.append((obj, name, actual, None))
            native &= actual is original
    functions.extend(executor._NATIVE_ORDINARY_EXECUTOR_FUNCTIONS)
    parsers, entries = commands._NATIVE_VIRTUAL_PARSERS
    native &= commands._PARSERS is parsers and tuple(parsers.items()) == entries
    for parser, actions in commands._NATIVE_VIRTUAL_ACTIONS:
        actual = inspect.getattr_static(parser, "_actions", None)
        native &= type(actual) is list and tuple(actual) == actions
    for obj, name, original in commands._NATIVE_VIRTUAL_PARSER_SOURCES:
        actual = inspect.getattr_static(obj, name, None)
        values.append((obj, name, actual, None))
        native &= actual is original
    native &= all(_function_unchanged(record) for record in functions)
    if provider is not None:
        registries = inspect.getattr_static(provider, "_registries_by_alias", None)
        authorities = inspect.getattr_static(provider, "_admitted_roots", None)
        if (
            type(registries) is not dict
            or authorities is not None
            and type(authorities) is not dict
        ):
            return tuple(values), tuple(functions), False
        for registry in registries.values():
            native &= type(registry) is commands.NATIVE_VIRTUAL_CLASS
            for name, original in commands._NATIVE_VIRTUAL_METHODS:
                native &= inspect.getattr_static(registry, name, None) is original
    return tuple(values), tuple(functions), native


def require_virtual_sources(provider):
    if not virtual_source_records(provider)[2]:
        raise RuntimeError("original virtual provider/parser source changed")


def _original_methods(target, methods):
    return all(
        getattr(current, "__func__", current) is method
        for name, method in methods.items()
        for current in (getattr(target, name, None),)
    )


@dataclass
class _Request:
    executor: object
    wire: bytes
    operation_id: str
    issued: bool = False
    local_returned: bool = False
    terminal: object = None

    @property
    def proven(self):
        return not self.issued or (
            self.local_returned
            and self.terminal is not None
            and self.terminal.cleanup_proven
        )


class RemoteSshSource:
    """The actual pre-enqueue authority, adapter, executor and transport inputs."""

    def __init__(self, authority, *, executor=None, local_specs=True):
        self.authority = authority
        self.local_specs = local_specs
        self.adapter = authority.workspace_executor if authority is not None else None
        self.executor = (
            getattr(self.adapter, "_inner", None) if authority is not None else executor
        )
        self.cfg = getattr(self.executor, "_ssh", None)
        self.cfg_values = dict(vars(self.cfg)) if self.cfg is not None else {}
        self.registry = None
        if self.cfg is not None and self.cfg.session_key:
            from tldw_chatbook.Tools.remote_session_registry import get_session_registry

            self.registry = get_session_registry()
        self.control_dir = (
            self.cfg.masters._control_dir if self.cfg is not None else None
        )
        self.manager_configuration = (
            _master_configuration(self.cfg.masters) if self.cfg is not None else None
        )
        self.transport_grace = (
            self.cfg.transport.grace_seconds if self.cfg is not None else None
        )
        self.transport_manager = (
            self.cfg.transport._manager if self.cfg is not None else None
        )
        self.executor_values = tuple(
            getattr(self.executor, k, None)
            for k in (
                "_root",
                "_root_locator",
                "_identity_chain_source",
                "_budget_seconds",
                "_sensitive_exclusions",
            )
        )

    def check(self):
        from tldw_chatbook.Agents import local_tool_provider as local
        from tldw_chatbook.Tools import remote_workspace_executor as executor
        from tldw_chatbook.Tools import remote_workspace_transport as transport

        if self.cfg is not None and self.control_dir is None:
            # Another original call can finish the manager's lazy resolution
            # after this call was captured. Accept only its actual stock receipt.
            receipt = self.cfg.masters._native_control_directory
            if (
                type(receipt) is transport._ControlDirectorySource
                and receipt.configuration == self.manager_configuration
                and receipt.directory is self.cfg.masters._control_dir
            ):
                self.control_dir = receipt.directory

        if (
            self.authority is not None
            and (
                type(self.adapter) is not executor.RemoteExecutorDispatchAdapter
                or self.authority.workspace_executor is not self.adapter
                or self.adapter._inner is not self.executor
                or self.adapter._alias != self.authority.alias
                or not _original_methods(self.adapter, executor._NATIVE_SSH_ADAPTER_METHODS)
            )
            or type(self.executor) is not executor.RemoteWorkspaceToolExecutor
            or self.executor._ssh is not self.cfg
            or self.cfg is None
            or vars(self.cfg) != self.cfg_values
            or any(
                getattr(self.cfg, key) is not self.cfg_values[key]
                for key in ("loc", "cache", "masters", "transport", "session_spawn")
            )
            or _master_configuration(self.cfg.masters) != self.manager_configuration
            or self.cfg.masters._control_dir is not self.control_dir
            or self.cfg.transport.grace_seconds != self.transport_grace
            or self.cfg.session_spawn is not None
            or tuple(
                getattr(self.executor, k, None)
                for k in (
                    "_root",
                    "_root_locator",
                    "_identity_chain_source",
                    "_budget_seconds",
                    "_sensitive_exclusions",
                )
            )
            != self.executor_values
            or not _original_methods(self.executor, executor._NATIVE_SSH_METHODS)
            or type(self.cfg.transport) is not transport.RemoteWorkspaceTransport
            or self.cfg.transport._manager is not self.transport_manager
            or self.transport_manager is not self.cfg.masters
            or type(self.cfg.transport).__dict__.get("master_manager")
            is not transport._NATIVE_TRANSPORT_MANAGER
            or not _original_methods(
                self.cfg.transport, transport._NATIVE_TRANSPORT_METHODS
            )
            or type(self.cfg.masters) is not transport.SshMasterManager
            or not _original_methods(self.cfg.masters, transport._NATIVE_MASTER_METHODS)
            or self.authority is not None
            and self.local_specs
            and local._default_specs is not local._NATIVE_DEFAULT_SPECS
        ):
            raise RuntimeError("original SSH executor source changed")
        from tldw_chatbook.Tools import remote_binding_locator as locator
        from tldw_chatbook.Tools import remote_session_registry as sessions

        if locator.canonicalize_locator is not locator._NATIVE_CANONICALIZE:
            raise RuntimeError("original SSH destination helper changed")
        if self.registry is not None and (
            sessions._REGISTRY is not self.registry
            or type(self.registry) is not sessions.RemoteSessionRegistry
            or not _original_methods(self.registry, sessions._NATIVE_REGISTRY_METHODS)
            or sessions.get_session_registry is not sessions._NATIVE_REGISTRY_GET
        ):
            raise RuntimeError("original SSH session registry changed")
        if self.registry is not None:
            from tldw_chatbook.Tools import remote_session_worker as session

            for name, original in session._NATIVE_SESSION_FUNCTIONS.items():
                if getattr(session, name) is not original:
                    raise RuntimeError("original SSH session request callback changed")
        for name, original in executor._NATIVE_SSH_FUNCTIONS.items():
            if getattr(executor, name) is not original:
                raise RuntimeError("original SSH request callback changed")
        for name, original in transport._NATIVE_TRANSPORT_FUNCTIONS.items():
            if getattr(transport, name) is not original:
                raise RuntimeError("original SSH transport callback changed")


class ModelSshCall:
    def __init__(self, registry, name, selection, *, virtual=False):
        from tldw_chatbook.Tools.remote_root_types import RemoteRoot

        self.registry, self.name, self.selection = registry, name, selection
        self.provider = selection[1]
        self.virtual = virtual
        self.virtual_callbacks = ()
        self.virtual_functions = ()
        self.virtual_partials = ()
        if virtual:
            require_virtual_sources(self.provider)
            from functools import partial
            from types import FunctionType, MethodType

            from tldw_chatbook.Agents.virtual_cli_provider import (
                observe_virtual_callbacks,
            )

            from tldw_chatbook.Agents.activation import capture_callable_source as _function_record

            self.virtual_callbacks = observe_virtual_callbacks(self.provider)
            for _name, value in self.virtual_callbacks:
                function = value
                if type(value) is partial:
                    self.virtual_partials += (
                        (value, value.func, value.args, tuple(value.keywords.items())),
                    )
                    function = value.func
                if type(function) is MethodType:
                    function = function.__func__
                if type(function) is FunctionType:
                    self.virtual_functions += (_function_record(function),)
        self.generation, self.catalog = (
            registry._catalog_generation,
            registry._catalog_snapshot,
        )
        self.authorities = dict(self.provider._admitted_roots)
        self.specs = (
            {}
            if virtual
            else {
                alias: dict(items)
                for alias, items in self.provider._native_ssh_specs.items()
            }
        )
        self.virtual_registries = (
            dict(self.provider._registries_by_alias) if virtual else {}
        )
        self.sources = {
            alias: RemoteSshSource(a, local_specs=not virtual)
            for alias, a in self.authorities.items()
            if isinstance(a.root, RemoteRoot)
        }
        self.source = self.operation = self.request = None
        self.requests = []
        self.master_launches = []
        self.session_starts = []
        self.finished = False
        self.uncertain = False

    def check(self):
        from tldw_chatbook.Agents import local_tool_provider as local
        from tldw_chatbook.Agents.tool_catalog import (
            _NATIVE_SSH_CATALOG_METHODS,
            ToolCatalogRegistry,
        )

        if self.virtual:
            require_virtual_sources(self.provider)
            from tldw_chatbook.Agents.activation import callable_source_current as _function_unchanged

            if (
                any(
                    inspect.getattr_static(self.provider, name, None) is not value
                    for name, value in self.virtual_callbacks
                )
                or any(
                    not _function_unchanged(record) for record in self.virtual_functions
                )
                or any(
                    callback.func is not function
                    or callback.args is not arguments
                    or len(callback.keywords) != len(keywords)
                    or any(
                        callback.keywords.get(key) is not value
                        for key, value in keywords
                    )
                    for callback, function, arguments, keywords in self.virtual_partials
                )
            ):
                raise RuntimeError("original virtual callback source changed")
        if (
            (not self.virtual and type(self.provider) is not local.LocalToolProvider)
            or type(self.registry) is not ToolCatalogRegistry
            or not _original_methods(self.registry, _NATIVE_SSH_CATALOG_METHODS)
            or self.registry._catalog_generation != self.generation
            or self.registry._catalog_snapshot is not self.catalog
            or self.registry.resolve_owner_for_name(self.name) != self.selection
            or self.provider._admitted_roots != self.authorities
            or any(
                self.provider._admitted_roots.get(alias) is not authority
                for alias, authority in self.authorities.items()
            )
            or not self.virtual
            and not _original_methods(self.provider, local._NATIVE_SSH_PROVIDER_METHODS)
        ):
            raise RuntimeError("original SSH tool authority changed")
        if self.virtual and (
            self.provider._registries_by_alias != self.virtual_registries
            or any(
                self.provider._registries_by_alias.get(alias) is not original
                for alias, original in self.virtual_registries.items()
            )
            or any(
                inspect.getattr_static(
                    self.virtual_registries[alias], "_workspace_executor", None
                )
                is not authority.workspace_executor
                for alias, authority in self.authorities.items()
            )
        ):
            raise RuntimeError("original virtual SSH registry changed")
        for source in self.sources.values():
            source.check()

    def control_directory(self, manager, directory):
        for source in self.sources.values():
            if source.cfg.masters is manager:
                if manager._control_dir is not directory or source.control_dir not in (
                    None,
                    directory,
                ):
                    raise RuntimeError("original SSH control directory changed")
                source.control_dir = directory

    def bind(self, provider, authority, spec):
        self.check()
        if provider is not self.provider:
            raise RuntimeError("original SSH provider changed")
        if authority is None or authority.alias not in self.sources:
            return
        if self.authorities.get(authority.alias) is not authority or (
            self.virtual_registries.get(authority.alias) is not spec
            if self.virtual
            else self.specs[authority.alias].get(self.name) is not spec
        ):
            raise RuntimeError("original SSH selected handler changed")
        self.source = self.sources[authority.alias]

    @contextmanager
    def activate(self, operation):
        if operation is None:
            yield
            return
        self.operation = operation
        token = _current.set(self)
        try:
            self.check()
            yield
        finally:
            if self.uncertain or any(
                not r.proven
                for r in (*self.requests, *self.master_launches, *self.session_starts)
            ):
                operation.mark_stopping()
                operation.owner.mark_cleanup_unproven()
                Event().wait()  # Unknown physical state retains the original worker/storage.
            try:
                operation.finish()
                self.finished = True
            finally:
                _current.reset(token)

    @contextmanager
    def exchange(self, executor, wire):
        self.check()
        if self.source is None or executor is not self.source.executor:
            raise RuntimeError("original SSH request executor changed")
        request = _Request(executor, wire, json.loads(wire)["operation_id"])
        self.requests.append(request)
        previous, self.request = self.request, request
        try:
            yield request
        finally:
            self.request = previous

    def response(self, request, result):
        from tldw_chatbook.Tools.remote_workspace_executor import _parse_worker_output
        from tldw_chatbook.Tools.workspace_tool_executor import (
            WorkspaceToolExecutionError,
        )

        self.check()
        if result.response is not None:
            try:
                request.terminal = _parse_worker_output(
                    result.response, expected_operation_id=request.operation_id
                )
            except WorkspaceToolExecutionError:
                pass  # Preserve the executor's existing protocol-error rendering.

    def retire_process(self, proc, threads):
        """Join actual pipes and the original new-session group on this worker."""
        from tldw_chatbook.Notes.git_process_containment import (
            OwnedProcessTree,
            ProcessTreeController,
        )

        try:
            for thread in threads:
                thread.join()
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            # Popen was created by the pinned stock source with start_new_session.
            tree, controller = OwnedProcessTree(proc, proc.pid), ProcessTreeController()
            with asyncio.Runner() as runner:
                if not runner.run(controller.wait(tree, timeout=float("inf"))):
                    raise OSError("SSH local group proof unavailable")
            controller.close(tree)
        except BaseException:
            self.uncertain = True
            raise

    def finite_helper(self, argv, *, timeout, text=False, errors=None):
        """Preserve ssh helper argv/config/stdin while owning its finite tree."""

        async def run():
            from tldw_chatbook.Notes.git_process_containment import (
                ProcessTreeAdmissionError,
                ProcessTreeController,
            )

            controller, tree, io = ProcessTreeController(), None, None
            try:
                try:
                    tree = await controller.spawn(
                        *argv,
                        cwd=os.getcwd(),
                        environment=os.environ.copy(),
                        stdin=None,
                    )
                except ProcessTreeAdmissionError as error:
                    tree = error.tree
                    raise
                io = asyncio.create_task(tree.process.communicate())
                deadline = asyncio.get_running_loop().time() + timeout
                try:
                    stdout, stderr = await asyncio.wait_for(asyncio.shield(io), timeout)
                except TimeoutError:
                    raise subprocess.TimeoutExpired(argv, timeout) from None
                if not await controller.wait(
                    tree, timeout=max(0, deadline - asyncio.get_running_loop().time())
                ):
                    raise subprocess.TimeoutExpired(argv, timeout)
                controller.close(tree)
                if text:
                    import locale

                    encoding = locale.getencoding()
                    stdout, stderr = (
                        part.decode(encoding, errors or "strict")
                        .replace("\r\n", "\n")
                        .replace("\r", "\n")
                        for part in (stdout, stderr)
                    )
                return subprocess.CompletedProcess(
                    argv, tree.process.returncode, stdout, stderr
                )
            except BaseException:
                if tree is not None:
                    self.operation.mark_stopping()
                    if io is None:
                        io = asyncio.create_task(tree.process.communicate())
                    try:
                        for stop in (controller.terminate, controller.kill):
                            try:
                                stop(tree)
                            except OSError:
                                pass
                            if await controller.wait(tree, timeout=0.25):
                                break
                        else:
                            if not await controller.wait(tree, timeout=float("inf")):
                                raise OSError("SSH helper cleanup unavailable")
                        await asyncio.shield(asyncio.gather(io, return_exceptions=True))
                        controller.close(tree)
                    except BaseException:  # noqa: BLE001 -- failed native proof cannot retire
                        self.uncertain = True
                raise

        self.check()
        with asyncio.Runner() as runner:
            return runner.run(run())


@dataclass(frozen=True)
class _MasterAcceptance:
    manager: object
    locator: object
    control_dir: object
    configuration: tuple
    check_argv: tuple
    launcher: object


def _master_configuration(manager):
    return (
        manager._ssh_bin,
        manager._control_persist,
        manager._enabled,
        manager._connect_timeout_s,
        manager._state_dir,
    )


@dataclass
class _MasterLaunch:
    manager: object
    locator: object
    argv: tuple
    process: object = None
    returned: bool = False
    accepted: bool = False
    never_started: bool = False

    @property
    def proven(self):
        return self.never_started or (self.returned and self.accepted)


def run_master_launcher(manager, loc, argv, timeout):
    """Observe the original launcher; successful adoption is a separate fact."""
    native = _current.get()
    if native is None:
        return subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout,
            check=False,
            start_new_session=True,
        )
    native.check()
    attempt = _MasterLaunch(manager, loc, tuple(argv))
    native.master_launches.append(attempt)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
    except OSError:
        attempt.never_started = True
        raise
    attempt.process = proc
    with proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        result = subprocess.CompletedProcess(argv, proc.wait(), stdout, stderr)
    attempt.returned = True
    return result


def accept_master(manager, loc, control_dir):
    """Consume exact finite verification into the existing manager's custody."""
    native = _current.get()
    if native is None:
        return
    from tldw_chatbook.Tools.remote_workspace_transport import _host_key

    native.check()
    key = _host_key(loc)
    config = _master_configuration(manager)
    attempt = next(
        (
            a
            for a in reversed(native.master_launches)
            if a.manager is manager and a.locator is loc
        ),
        None,
    )
    with manager._registry_lock:
        previous = manager._native_acceptances.get(key)
        reusable = (
            type(previous) is _MasterAcceptance
            and previous.manager is manager
            and previous.control_dir == control_dir
            and previous.configuration == config
            and not manager._closed
            and manager._registrations.get(key) == previous.locator
            and (attempt is None or attempt.accepted)
        )
    if reusable:
        with manager._registry_lock:
            manager._native_checks.pop(key, None)
        return
    if attempt is not None and attempt.never_started:
        return
    # An existing path, a successful -MNf or one directory entry cannot adopt.
    # _master_alive uses the original finite native helper and records this call.
    if not manager._master_alive(loc):
        native.uncertain = True
        return
    with manager._registry_lock:
        checked = manager._native_checks.pop(key, None)
        if (
            checked is None
            or checked[0] is not native
            or checked[1] is not loc
            or manager._closed
        ):
            native.uncertain = True
            return
        attempt = next(
            (
                a
                for a in reversed(native.master_launches)
                if a.manager is manager and a.locator is loc
            ),
            None,
        )
        if attempt is not None and not attempt.returned:
            native.uncertain = True
            return
        acceptance = _MasterAcceptance(
            manager, loc, control_dir, config, checked[2], attempt
        )
        manager._native_acceptances[key] = acceptance
        if attempt is not None:
            attempt.accepted = True


@dataclass(frozen=True)
class _SessionReady:
    worker: object
    process: object
    reader: object
    stderr: object
    spawn: object
    stamp: str


@dataclass(frozen=True)
class _SessionAcceptance:
    registry: object
    key: tuple
    ready: _SessionReady


@dataclass
class _SessionStart:
    registry: object
    key: tuple
    worker: object
    proven: bool = False


def record_session_ready(worker, stamp):
    worker._native_ready = _SessionReady(
        worker,
        worker._proc,
        worker._reader_thread,
        worker._stderr_thread,
        worker._spawn,
        stamp,
    )


def session_publication(registry, key, worker):
    """Pure observation at the existing registry's actual publication boundary."""
    from tldw_chatbook.Tools.remote_session_worker import (
        _NATIVE_SESSION_METHODS,
        RemoteSessionWorker,
    )

    ready = getattr(worker, "_native_ready", None)
    if (
        type(worker) is RemoteSessionWorker
        and _original_methods(worker, _NATIVE_SESSION_METHODS)
        and type(ready) is _SessionReady
        and ready.worker is worker
        and ready.process is worker._proc
        and ready.reader is worker._reader_thread
        and ready.stderr is worker._stderr_thread
        and ready.spawn is worker._spawn
        and worker._spawn is worker._native_spawn
    ):
        return _SessionAcceptance(registry, key, ready)
    return None


def starting_session(registry, key, worker):
    native = _current.get()
    if native is None:
        return
    from tldw_chatbook.Tools.remote_session_registry import _NATIVE_REGISTRY_METHODS
    from tldw_chatbook.Tools.remote_session_worker import (
        _NATIVE_SESSION_METHODS,
        RemoteSessionWorker,
    )

    native.check()
    cfg = native.source.cfg
    if (
        type(worker) is not RemoteSessionWorker
        or not _original_methods(worker, _NATIVE_SESSION_METHODS)
        or not _original_methods(registry, _NATIVE_REGISTRY_METHODS)
        or worker._transport is not cfg.transport
        or worker._loc is not cfg.loc
        or worker._python != cfg.python
        or worker._spawn is not worker._native_spawn
        or key != (cfg.session_key, cfg.binding_id)
    ):
        raise RuntimeError("original SSH session start changed")
    native.session_starts.append(_SessionStart(registry, key, worker))


def unpublished_session(registry, key, worker):
    native = _current.get()
    if native is None:
        return
    attempt = next(
        a
        for a in reversed(native.session_starts)
        if a.registry is registry and a.key == key and a.worker is worker
    )
    if worker._proc is not None:
        native.retire_process(
            worker._proc,
            tuple(
                t
                for t in (worker._reader_thread, worker._stderr_thread)
                if t is not None
            ),
        )
    attempt.proven = True


def accept_session(registry, key, worker):
    native = _current.get()
    if native is None:
        return
    from tldw_chatbook.Tools.remote_session_worker import _NATIVE_SESSION_METHODS

    native.check()
    if registry is not native.source.registry:
        raise RuntimeError("original SSH session registry changed")
    acceptance = worker._native_acceptance
    if (
        type(acceptance) is not _SessionAcceptance
        or acceptance.registry is not registry
        or acceptance.key != key
        or acceptance.ready.worker is not worker
        or acceptance.ready.process is not worker._proc
        or acceptance.ready.reader is not worker._reader_thread
        or acceptance.ready.stderr is not worker._stderr_thread
        or acceptance.ready.spawn is not worker._spawn
        or not _original_methods(worker, _NATIVE_SESSION_METHODS)
    ):
        raise RuntimeError("original SSH session acceptance unavailable")
    for attempt in native.session_starts:
        if (
            attempt.registry is registry
            and attempt.key == key
            and attempt.worker is worker
        ):
            attempt.proven = True
