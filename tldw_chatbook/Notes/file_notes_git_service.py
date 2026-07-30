"""Pure session grouping and Git status policy for File Notes."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
from collections.abc import Awaitable, Collection, Generator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    FileSystemIdentity,
    GitMutationLease,
    GitStatusLease,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    RepositoryIdentity,
    SequencedSessionChange,
    SessionBinding,
    SessionChangeGroup,
    SessionChangeTopology,
    SessionGitStatus,
    SessionGitRow,
    StagingOwnership,
    coalesce_session_changes,
)

PorcelainKind = Literal[
    "ordinary",
    "rename",
    "unmerged",
    "untracked",
    "ignored",
    "nested_repository",
    "unavailable",
    "error",
]
GitArg = str | bytes
_SettlementValue = TypeVar("_SettlementValue")
DiscoveryState = Literal[
    "ready",
    "not_repository",
    "unavailable",
    "unsupported",
    "unsafe_root",
]
HeadReadFailureKind = Literal["unavailable", "error"]
GitActionState = Literal["success", "blocked", "stale", "error", "uncertain"]
RetainedGitChildState = Literal[
    "alive",
    "natural",
    "stop_requested",
    "forced_stop",
    "uncertain",
]

_REDIRECTING_GIT_ENVIRONMENT = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_SHALLOW_FILE",
        "GIT_GRAFT_FILE",
        "GIT_REPLACE_REF_BASE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_EXEC_PATH",
        "GIT_PREFIX",
        "GIT_CONFIG",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_GLOB_PATHSPECS",
        "GIT_NOGLOB_PATHSPECS",
        "GIT_LITERAL_PATHSPECS",
        "GIT_ICASE_PATHSPECS",
    }
)
_DYNAMIC_CONFIG_ENVIRONMENT_PREFIXES = (
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
)
DEFAULT_GIT_STDERR_LIMIT_BYTES = 4096


_RETAINED_CHILD_TOKEN_SECRET = object()


class RetainedGitChildToken:
    """Opaque identity for exactly one still-running uncertain Git child."""

    __slots__ = ()

    def __new__(
        cls,
        secret: object | None = None,
    ) -> RetainedGitChildToken:
        if secret is not _RETAINED_CHILD_TOKEN_SECRET:
            raise TypeError("Retained Git child tokens are created by the runner")
        return super().__new__(cls)


@dataclass(frozen=True, slots=True)
class RetainedGitChildSettlement:
    """Non-sealing observation of one exact retained Git child."""

    state: RetainedGitChildState
    returncode: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    stop_requested: bool = False
    force_stopped: bool = False


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Byte-preserving result from one direct Git child process."""

    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    termination_uncertain: bool = False
    retained_child: RetainedGitChildToken | None = None
    stop_requested: bool = False
    force_stopped: bool = False


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Machine-safe repository discovery without granting process trust."""

    state: DiscoveryState
    repository: RepositoryIdentity | None = None
    head: HeadIdentity | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class GitActionResult:
    """Checked result of one retained Git index action."""

    action: Literal["stage", "unstage"]
    state: GitActionState
    requested_group_ids: tuple[int, ...]
    staged_group_ids: tuple[int, ...] = ()
    unstaged_group_ids: tuple[int, ...] = ()
    clean_group_ids: tuple[int, ...] = ()
    blocked_group_ids: tuple[int, ...] = ()
    message: str | None = None


@dataclass(frozen=True, slots=True)
class _StageInspection:
    """Fresh repository facts used by one exact Stage pre/postflight."""

    repository: RepositoryIdentity
    head: HeadIdentity
    groups: tuple[SessionChangeGroup, ...]
    repository_groups: Mapping[int, SessionChangeGroup]
    rows: Mapping[int, SessionGitRow]
    index_sequence: tuple[IndexEntry, ...]
    index_entries: Mapping[str, IndexEntry]
    status_records: tuple[PorcelainRecord, ...]


@dataclass(frozen=True, slots=True)
class _RawGitInspection:
    """Shared fresh HEAD/index/status facts for status and Stage."""

    head: HeadIdentity
    index_entries: tuple[IndexEntry, ...]
    status_records: tuple[PorcelainRecord, ...]


@dataclass(frozen=True, slots=True)
class _RawGitInspectionFailure:
    """Typed shared inspection refusal without UI/action policy."""

    state: Literal["stale", "unavailable", "error", "uncertain"]
    message: str
    head: HeadIdentity | None = None
    revoke_ownership: bool = False


@dataclass(frozen=True, slots=True)
class _HeadReadFailure:
    """Typed failure that cannot be confused with a semantic HEAD state."""

    kind: HeadReadFailureKind
    message: str


GitStatusAdmissionReason = Literal[
    "untrusted",
    "mutation_active",
    "stale_binding",
    "shutdown",
    "status_active",
]
GitMutationAdmissionReason = Literal[
    "untrusted",
    "mutation_active",
    "recovery_required",
    "transition_active",
    "stale_binding",
    "shutdown",
]


class GitStatusAdmissionError(RuntimeError):
    """Typed synchronous refusal before any worktree-aware child starts."""

    def __init__(
        self,
        reason: GitStatusAdmissionReason,
        message: str,
    ) -> None:
        super().__init__(message)
        self.reason = reason


class GitMutationAdmissionError(RuntimeError):
    """Typed synchronous refusal before a retained mutation task exists."""

    def __init__(
        self,
        reason: GitMutationAdmissionReason,
        message: str,
    ) -> None:
        super().__init__(message)
        self.reason = reason


class GitShutdownAffinityError(RuntimeError):
    """Retryable refusal when active shutdown starts off its owning loop."""

    retryable_shutdown = True


class GitRunCancelled(asyncio.CancelledError):
    """Caller cancellation with either a retained child or terminal result."""

    def __init__(
        self,
        *,
        retained_child: RetainedGitChildToken | None = None,
        result: GitCommandResult | None = None,
    ) -> None:
        if (retained_child is None) == (result is None):
            raise ValueError(
                "Cancellation requires exactly one retained child or result"
            )
        super().__init__(
            (
                "Git command waiter cancelled; child retained"
                if retained_child is not None
                else "Git command waiter cancelled after child settlement"
            )
        )
        self.retained_child = retained_child
        self.result = result


class GitProcessRunner(Protocol):
    """Injectable direct-argv child-process boundary."""

    async def run(
        self,
        argv: Sequence[GitArg],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        """Run one command without accepting a shell option."""

    def shutdown(self) -> Awaitable[bool] | None:
        """Seal admission and return retained finite child settlement."""

    def read_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> RetainedGitChildSettlement:
        """Read one exact retained child without changing its lifecycle."""

    async def settle_retained_child(
        self,
        token: RetainedGitChildToken,
        *,
        timeout: float = 0.0,
    ) -> RetainedGitChildSettlement:
        """Wait boundedly for one exact retained child without stopping it."""

    def release_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> bool:
        """Release terminal evidence; refuse live or uncertain children."""


@dataclass(frozen=True, slots=True)
class _ImmediateSettlement(Generic[_SettlementValue]):
    """Reusable awaitable for shutdown paths with no asynchronous work."""

    value: _SettlementValue

    def __await__(self) -> Generator[None, None, _SettlementValue]:
        if False:
            yield None
        return self.value


@dataclass(frozen=True, slots=True)
class _RetainedSettlement(Generic[_SettlementValue]):
    """Cancellation-safe reusable view of an internally retained task."""

    task: asyncio.Task[_SettlementValue]

    def __await__(self) -> Generator[Any, None, _SettlementValue]:
        return asyncio.shield(self.task).__await__()


@dataclass(slots=True)
class _RetainedChildRecord:
    """One operation record spanning spawn, child, and terminal settlement."""

    token: RetainedGitChildToken
    ready: asyncio.Event
    process: asyncio.subprocess.Process | None = None
    communication: asyncio.Task[tuple[bytes, bytes]] | None = None
    stop_requested: bool = False
    force_stopped: bool = False
    timed_out: bool = False
    exposed: bool = False
    released: bool = False
    owned_task: asyncio.Task[GitCommandResult] | None = None
    settlement: RetainedGitChildSettlement | None = None


def build_git_environment(
    ambient: Mapping[str, str] | None = None,
    *,
    for_status: bool = False,
    stable_locale: bool = False,
) -> dict[str, str]:
    """Build a sanitized environment for a direct Git child process.

    Args:
        ambient: Source environment. Defaults to the current process
            environment.
        for_status: Whether to disable optional Git index locks for a
            read-only status command.
        stable_locale: Whether to force the stable ``C`` locale.

    Returns:
        A copied environment without Git redirection or dynamic-config
        variables and with interactive prompting disabled.
    """
    source = os.environ if ambient is None else ambient
    environment = {
        key: value
        for key, value in source.items()
        if (
            key not in _REDIRECTING_GIT_ENVIRONMENT
            and key != "GIT_CONFIG_COUNT"
            and not key.startswith(_DYNAMIC_CONFIG_ENVIRONMENT_PREFIXES)
        )
    }
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if stable_locale:
        environment["LC_ALL"] = "C"
    if for_status:
        environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def build_status_argv(
    git_executable: str,
    repository_paths: Sequence[bytes],
) -> tuple[GitArg, ...]:
    """Build the literal, NUL-safe, no-renames session status command."""
    return (
        git_executable,
        "--literal-pathspecs",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "status.renames=false",
        "-c",
        "diff.renames=false",
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--no-renames",
        "--",
        *repository_paths,
    )


def build_index_argv(git_executable: str) -> tuple[GitArg, ...]:
    """Build one complete, NUL-safe semantic index read."""
    return (
        git_executable,
        "--literal-pathspecs",
        "-c",
        "core.fsmonitor=false",
        "ls-files",
        "-z",
        "--stage",
        "-v",
        "--",
    )


def build_stage_argv(
    git_executable: str,
    repository_paths: Sequence[bytes],
) -> tuple[GitArg, ...]:
    """Build one exact fail-fast literal path-scoped Stage command."""
    return (
        git_executable,
        "--literal-pathspecs",
        "-c",
        "add.ignoreErrors=false",
        "add",
        "--all",
        "--",
        *repository_paths,
    )


def build_unstage_argv(git_executable: str) -> tuple[GitArg, ...]:
    """Build one exact index-only saved-baseline Unstage command."""
    return (
        git_executable,
        "update-index",
        "-z",
        "--index-info",
    )


def build_update_index_payload(
    ownership: StagingOwnership,
    current_index_entries: Mapping[str, IndexEntry],
) -> bytes:
    """Build exact conflict removals followed by saved baseline records."""
    conflict_records, baseline_records = _update_index_records(
        ownership,
        current_index_entries,
    )
    return b"".join((*conflict_records, *baseline_records))


def _update_index_records(
    ownership: StagingOwnership,
    current_index_entries: Mapping[str, IndexEntry],
) -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    object_id_lengths = {
        len(entry.object_id)
        for entry in (
            *(
                baseline.entry
                for baseline in ownership.original_baselines.values()
                if baseline.entry is not None
            ),
            *(
                entry
                for entry in ownership.post_stage_entries.values()
                if entry is not None
            ),
        )
    }
    if len(object_id_lengths) != 1 or not object_id_lengths.issubset({40, 64}):
        raise ValueError("Unstage ownership has an invalid object ID width")
    zero_oid = b"0" * object_id_lengths.pop()

    conflict_paths = sorted(
        path
        for path in compute_unstage_closure(
            ownership.original_baselines,
            current_index_entries,
        )
        if path not in ownership.original_baselines
    )
    conflict_records = tuple(
        b"0 " + zero_oid + b"\t" + _index_info_path_bytes(path) + b"\0"
        for path in conflict_paths
    )
    baseline_records: list[bytes] = []
    for path, baseline in sorted(ownership.original_baselines.items()):
        raw_path = _index_info_path_bytes(path)
        entry = baseline.entry
        if entry is None:
            baseline_records.append(
                b"0 " + zero_oid + b"\t" + raw_path + b"\0"
            )
            continue
        if (
            entry.path != path
            or entry.stage != 0
            or entry.semantic_flags
            or len(entry.mode) != 6
            or any(character not in "01234567" for character in entry.mode)
            or len(entry.object_id) != len(zero_oid)
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in entry.object_id
            )
        ):
            raise ValueError("Unstage baseline is not an exact stage-0 entry")
        baseline_records.append(
            entry.mode.encode("ascii")
            + b" "
            + entry.object_id.lower().encode("ascii")
            + b" 0\t"
            + raw_path
            + b"\0"
        )
    return conflict_records, tuple(baseline_records)


def _index_info_path_bytes(path: str) -> bytes:
    if (
        not path
        or path.startswith("/")
        or "\0" in path
        or any(component in {"", ".", ".."} for component in path.split("/"))
    ):
        raise ValueError("Unstage path is unsafe")
    return os.fsencode(path)


def _build_combined_update_index_payload(
    ownership: Sequence[StagingOwnership],
    current_index_entries: Mapping[str, IndexEntry],
) -> bytes:
    conflict_records: list[bytes] = []
    baseline_records: list[bytes] = []
    seen_conflicts: set[bytes] = set()
    seen_baselines: set[str] = set()
    for item in ownership:
        conflicts, baselines = _update_index_records(item, current_index_entries)
        for record in conflicts:
            if record not in seen_conflicts:
                seen_conflicts.add(record)
                conflict_records.append(record)
        for path in item.original_baselines:
            if path in seen_baselines:
                raise ValueError("Unstage groups overlap in the Git index")
            seen_baselines.add(path)
        baseline_records.extend(baselines)
    return b"".join((*sorted(conflict_records), *sorted(baseline_records)))


def _baseline_matches_index(
    ownership: StagingOwnership,
    current_index_entries: Mapping[str, IndexEntry],
) -> bool:
    if any(
        current_index_entries.get(path) != baseline.entry
        for path, baseline in ownership.original_baselines.items()
    ):
        return False
    inserted_baseline_paths = tuple(
        path
        for path, baseline in ownership.original_baselines.items()
        if baseline.entry is not None
    )
    return all(
        current_index_entries.get(path) is None
        for path in ownership.post_stage_entries
        if (
            path not in ownership.original_baselines
            and any(
                _paths_overlap(path, baseline_path)
                for baseline_path in inserted_baseline_paths
            )
        )
    )


def sanitize_git_stderr(
    payload: bytes,
    *,
    limit: int = DEFAULT_GIT_STDERR_LIMIT_BYTES,
) -> str:
    """Bound diagnostics and make terminal control bytes visible."""
    if limit <= 0:
        return ""
    decoded = os.fsdecode(payload)
    parts: list[str] = []
    size = 0
    replacements = {"\n": r"\n", "\r": r"\r", "\t": r"\t"}
    for character in decoded:
        codepoint = ord(character)
        if character in replacements:
            piece = replacements[character]
        elif codepoint < 32 or 127 <= codepoint <= 159:
            piece = f"\\x{codepoint:02x}"
        elif 0xDC80 <= codepoint <= 0xDCFF:
            piece = f"\\x{codepoint - 0xDC00:02x}"
        else:
            piece = character
        encoded = piece.encode("utf-8", "surrogateescape")
        if size + len(encoded) > limit:
            break
        parts.append(piece)
        size += len(encoded)
    return "".join(parts)


class AsyncGitProcessRunner:
    """Own direct-argv asyncio Git children without a shell boundary."""

    def __init__(
        self,
        *,
        terminate_timeout: float = 0.25,
        kill_timeout: float = 0.25,
        stderr_limit: int = DEFAULT_GIT_STDERR_LIMIT_BYTES,
    ) -> None:
        self._sealed = False
        self._terminate_timeout = terminate_timeout
        self._kill_timeout = kill_timeout
        self._stderr_limit = stderr_limit
        self._processes: set[asyncio.subprocess.Process] = set()
        self._run_tasks: set[asyncio.Task[object]] = set()
        self._retained_children: dict[
            RetainedGitChildToken,
            _RetainedChildRecord,
        ] = {}
        self._shutdown_event: asyncio.Event | None = None
        self._shutdown_settlement: Awaitable[bool] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(
        self,
        argv: Sequence[GitArg],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        """Execute one direct child and preserve all standard streams as bytes."""
        if self._sealed:
            return GitCommandResult(127, b"", b"Git runner is shut down")
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            self._shutdown_event = asyncio.Event()
        elif self._loop is not loop:
            raise RuntimeError("Git runner cannot span multiple event loops")
        assert self._shutdown_event is not None
        token = RetainedGitChildToken(_RETAINED_CHILD_TOKEN_SECRET)
        record = _RetainedChildRecord(token, asyncio.Event())
        self._retained_children[token] = record
        run_task = loop.create_task(
            self._run_owned_command(
                record,
                argv,
                cwd=cwd,
                environment=environment,
                stdin=stdin,
                timeout=timeout,
            )
        )
        record.owned_task = run_task
        self._run_tasks.add(run_task)
        run_task.add_done_callback(
            lambda task: self._run_task_completed(task, record)
        )
        try:
            result = await asyncio.shield(run_task)
            if result.retained_child is None:
                self._release_unexposed_record(record)
            else:
                record.exposed = True
            return result
        except asyncio.CancelledError:
            if record.communication is not None:
                settlement = self._read_record(record)
                if settlement.state not in {"alive", "uncertain"}:
                    result = self._result_from_settlement(
                        settlement,
                        timed_out=record.timed_out,
                        stop_requested=record.stop_requested,
                        force_stopped=record.force_stopped,
                    )
                    self._discard_record(record)
                    raise GitRunCancelled(result=result) from None
            record.exposed = True
            raise GitRunCancelled(retained_child=token) from None
        except BaseException:
            self._retained_children.pop(token, None)
            raise

    async def _run_owned_command(
        self,
        record: _RetainedChildRecord,
        argv: Sequence[GitArg],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None,
        timeout: float | None,
    ) -> GitCommandResult:
        """Run one child in a runner-owned task immune to caller cancellation."""
        assert self._shutdown_event is not None
        shutdown_waiter: asyncio.Task[bool] | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=dict(environment),
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._processes.add(process)
            communication = asyncio.create_task(process.communicate(stdin))
            record.process = process
            record.communication = communication
            record.ready.set()
            shutdown_waiter = asyncio.create_task(
                self._shutdown_event.wait()
            )
            done, _ = await asyncio.wait(
                {communication, shutdown_waiter},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if communication in done:
                settlement = self._read_record(record)
                if settlement.state == "natural":
                    return self._result_from_settlement(settlement)
                return self._uncertain_result(record)

            shutdown_requested = shutdown_waiter in done
            record.timed_out = not shutdown_requested
            terminated = await self._stop_record(record)
            if terminated:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(communication),
                        timeout=self._kill_timeout,
                    )
                except TimeoutError:
                    terminated = False
            settlement = self._read_record(record)
            if terminated and settlement.state not in {"alive", "uncertain"}:
                return self._result_from_settlement(
                    settlement,
                    timed_out=not shutdown_requested,
                    stop_requested=record.stop_requested,
                    force_stopped=record.force_stopped,
                )
            return self._uncertain_result(
                record,
                timed_out=not shutdown_requested,
                fallback_stderr=(
                    b"Git command stopped during shutdown"
                    if shutdown_requested
                    else b"Git command timed out"
                ),
            )
        except asyncio.CancelledError:
            if record.process is not None:
                await self._stop_record(record)
                record.exposed = True
            raise
        finally:
            if record.process is None:
                record.ready.set()
            if shutdown_waiter is not None and not shutdown_waiter.done():
                shutdown_waiter.cancel()
                await asyncio.gather(
                    shutdown_waiter,
                    return_exceptions=True,
                )

    def _run_task_completed(
        self,
        task: asyncio.Task[GitCommandResult],
        record: _RetainedChildRecord,
    ) -> None:
        """Release one owned task and retrieve any otherwise-orphaned error."""
        self._run_tasks.discard(task)
        if record.owned_task is task:
            record.owned_task = None
        if not task.cancelled():
            task.exception()
        if record.released:
            self._clear_record(record)

    def shutdown(self) -> Awaitable[bool]:
        """Seal admissions and return retained finite cleanup settlement."""
        if self._shutdown_settlement is not None:
            return self._shutdown_settlement
        running_loop: asyncio.AbstractEventLoop | None = None
        if self._run_tasks or self._processes:
            assert self._loop is not None
            assert self._shutdown_event is not None
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError as error:
                raise GitShutdownAffinityError(
                    "Active Git shutdown must be initiated from its event loop"
                ) from error
            if running_loop is not self._loop:
                raise GitShutdownAffinityError(
                    "Active Git shutdown must be initiated from its event loop"
                )
        self._sealed = True
        if not self._run_tasks and not self._processes:
            self._release_terminal_records()
            self._shutdown_settlement = _ImmediateSettlement(True)
            return self._shutdown_settlement
        assert running_loop is not None
        assert self._shutdown_event is not None
        retained_before_shutdown = tuple(
            record.token
            for record in self._retained_children.values()
            if (
                record.exposed
                and record.process is not None
                and (
                    record.owned_task is None
                    or record.owned_task.done()
                )
            )
        )
        self._shutdown_event.set()
        settlement = _RetainedSettlement(
            running_loop.create_task(
                self._settle_shutdown(retained_before_shutdown)
            )
        )
        self._shutdown_settlement = settlement
        return settlement

    async def _bounded_process_wait(
        self,
        process: asyncio.subprocess.Process,
        timeout: float,
    ) -> bool:
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    async def _stop_record(
        self,
        record: _RetainedChildRecord,
    ) -> bool:
        """Request bounded termination and then force-stop one child."""
        process = record.process
        assert process is not None
        if self._read_record(record).state not in {"alive", "uncertain"}:
            return True
        record.stop_requested = True
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        except OSError:
            pass
        terminated = await self._bounded_process_wait(
            process,
            self._terminate_timeout,
        )
        if terminated:
            return True
        record.force_stopped = True
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError:
            pass
        return await self._bounded_process_wait(
            process,
            self._kill_timeout,
        )

    async def _settle_shutdown(
        self,
        retained_before_shutdown: tuple[RetainedGitChildToken, ...],
    ) -> bool:
        current = asyncio.current_task()
        run_tasks = tuple(
            task
            for task in self._run_tasks
            if task is not current
        )
        run_failed = False
        pending_tasks: set[asyncio.Task[object]] = set()
        if run_tasks:
            done_tasks, pending_tasks = await asyncio.wait(
                run_tasks,
                timeout=(
                    self._terminate_timeout
                    + (2 * self._kill_timeout)
                ),
            )
            for task in done_tasks:
                self._run_tasks.discard(task)
                if task.cancelled():
                    run_failed = True
                    continue
                try:
                    task.result()
                except BaseException:
                    run_failed = True
        all_children_settled = True
        for token in retained_before_shutdown:
            record = self._retained_children[token]
            settlement = self._read_record(record)
            if settlement.state not in {"alive", "uncertain"}:
                continue
            terminated = await self._stop_record(record)
            communication = record.communication
            assert communication is not None
            if terminated and not communication.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(communication),
                        timeout=self._kill_timeout,
                    )
                except TimeoutError:
                    terminated = False
            settlement = self._read_record(record)
            if terminated and settlement.state not in {"alive", "uncertain"}:
                if not record.exposed:
                    self._release_unexposed_record(record)
            else:
                record.exposed = True
                all_children_settled = False
        self._release_terminal_records()
        if self._processes:
            all_children_settled = False
        return (
            all_children_settled
            and not pending_tasks
            and not self._processes
            and not self._run_tasks
            and not run_failed
        )

    def read_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> RetainedGitChildSettlement:
        """Read one exact retained child without sealing or stopping it."""
        record = self._record_for_token(token)
        self._require_retained_child_loop(record)
        if record.communication is None:
            return RetainedGitChildSettlement("uncertain")
        return self._read_record(record)

    async def settle_retained_child(
        self,
        token: RetainedGitChildToken,
        *,
        timeout: float = 0.0,
    ) -> RetainedGitChildSettlement:
        """Wait boundedly for one exact retained child, without stopping it."""
        record = self._record_for_token(token)
        self._require_retained_child_loop(record)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        if record.communication is None and not record.ready.is_set() and timeout > 0:
            try:
                await asyncio.wait_for(
                    record.ready.wait(),
                    timeout=timeout,
                )
            except TimeoutError:
                return RetainedGitChildSettlement("uncertain")
        if record.communication is None:
            return RetainedGitChildSettlement("uncertain")
        settlement = self._read_record(record)
        remaining = deadline - loop.time()
        if (
            settlement.state not in {"alive", "uncertain"}
            or remaining <= 0
        ):
            return settlement
        try:
            await asyncio.wait_for(
                asyncio.shield(record.communication),
                timeout=remaining,
            )
        except TimeoutError:
            pass
        return self._read_record(record)

    def release_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> bool:
        """Release terminal evidence; refuse live or uncertain children."""
        record = self._record_for_token(token)
        self._require_retained_child_loop(record)
        if record.communication is None:
            return False
        settlement = self._read_record(record)
        if settlement.state in {"alive", "uncertain"}:
            return False
        self._discard_record(record)
        return True

    def _record_for_token(
        self,
        token: RetainedGitChildToken,
    ) -> _RetainedChildRecord:
        if not isinstance(token, RetainedGitChildToken):
            raise ValueError("Unknown retained Git child token")
        try:
            return self._retained_children[token]
        except KeyError:
            raise ValueError("Unknown retained Git child token") from None

    def _require_retained_child_loop(
        self,
        record: _RetainedChildRecord,
    ) -> None:
        if record.settlement is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise GitShutdownAffinityError(
                "A live retained Git child must be read from its event loop"
            ) from error
        if loop is not self._loop:
            raise GitShutdownAffinityError(
                "A live retained Git child must be read from its event loop"
            )

    def _read_record(
        self,
        record: _RetainedChildRecord,
    ) -> RetainedGitChildSettlement:
        if record.settlement is not None:
            return record.settlement
        communication = record.communication
        process = record.process
        assert communication is not None
        assert process is not None
        if not communication.done():
            return RetainedGitChildSettlement(
                "alive",
                stop_requested=record.stop_requested,
                force_stopped=record.force_stopped,
            )
        try:
            stdout, stderr = communication.result()
        except BaseException:
            return RetainedGitChildSettlement(
                "uncertain",
                stop_requested=record.stop_requested,
                force_stopped=record.force_stopped,
            )
        returncode = process.returncode
        if returncode is None:
            return RetainedGitChildSettlement(
                "uncertain" if record.stop_requested else "alive",
                stop_requested=record.stop_requested,
                force_stopped=record.force_stopped,
            )
        if returncode >= 0 or not record.stop_requested:
            state: RetainedGitChildState = "natural"
        elif record.force_stopped:
            state = "forced_stop"
        else:
            state = "stop_requested"
        settlement = RetainedGitChildSettlement(
            state,
            returncode,
            stdout,
            self._bounded_stderr(stderr),
            record.stop_requested,
            record.force_stopped,
        )
        record.settlement = settlement
        self._processes.discard(process)
        return settlement

    def _uncertain_result(
        self,
        record: _RetainedChildRecord,
        *,
        timed_out: bool = False,
        fallback_stderr: bytes = b"",
    ) -> GitCommandResult:
        settlement = self._read_record(record)
        if settlement.state not in {"alive", "uncertain"}:
            return self._result_from_settlement(
                settlement,
                timed_out=timed_out,
                stop_requested=record.stop_requested,
                force_stopped=record.force_stopped,
            )
        return GitCommandResult(
            settlement.returncode,
            settlement.stdout,
            settlement.stderr or self._bounded_stderr(fallback_stderr),
            timed_out=timed_out,
            termination_uncertain=True,
            retained_child=record.token,
            stop_requested=record.stop_requested,
            force_stopped=record.force_stopped,
        )

    def _result_from_settlement(
        self,
        settlement: RetainedGitChildSettlement,
        *,
        timed_out: bool = False,
        stop_requested: bool = False,
        force_stopped: bool = False,
    ) -> GitCommandResult:
        stopped = stop_requested or settlement.state in {
            "stop_requested",
            "forced_stop",
        }
        return GitCommandResult(
            settlement.returncode,
            settlement.stdout,
            settlement.stderr,
            timed_out=timed_out,
            termination_uncertain=stopped,
            stop_requested=stopped,
            force_stopped=(
                force_stopped or settlement.state == "forced_stop"
            ),
        )

    def _release_unexposed_record(
        self,
        record: _RetainedChildRecord,
    ) -> None:
        if record.exposed:
            return
        self._discard_record(record)

    def _release_terminal_records(self) -> None:
        """Discard only exact terminal records, retaining uncertainty."""
        for record in tuple(self._retained_children.values()):
            if record.communication is None:
                continue
            settlement = self._read_record(record)
            if settlement.state not in {"alive", "uncertain"}:
                self._discard_record(record)

    def _discard_record(
        self,
        record: _RetainedChildRecord,
    ) -> None:
        """Remove one terminal/unexposed record and its heavy references."""
        if record.process is not None:
            self._processes.discard(record.process)
        self._retained_children.pop(record.token, None)
        record.released = True
        if record.owned_task is None or record.owned_task.done():
            self._clear_record(record)

    @staticmethod
    def _clear_record(record: _RetainedChildRecord) -> None:
        """Drop heavy terminal references after the owned task is finished."""
        record.process = None
        record.communication = None
        record.owned_task = None
        record.settlement = None

    def _bounded_stderr(self, stderr: bytes) -> bytes:
        return sanitize_git_stderr(
            stderr,
            limit=self._stderr_limit,
        ).encode("utf-8", "surrogateescape")


class FileNotesGitService:
    """Trusted, process-owned Git projection for one File Notes owner."""

    def __init__(
        self,
        owner: FileNotesSessionOwner,
        *,
        runner: GitProcessRunner | None = None,
        git_executable: str | None = None,
        environment: Mapping[str, str] | None = None,
        discovery_timeout: float = 3.0,
        status_timeout: float = 5.0,
    ) -> None:
        self._owner = owner
        self._runner = runner or AsyncGitProcessRunner()
        self._environment = dict(
            os.environ if environment is None else environment
        )
        self._git_executable = (
            git_executable
            if git_executable is not None
            else shutil.which("git", path=self._environment.get("PATH"))
        )
        self._discovery_timeout = discovery_timeout
        self._status_timeout = status_timeout
        self._sealed = False
        self._status_cycle: asyncio.Task[SessionGitStatus] | None = None
        self._status_cycle_binding: SessionBinding | None = None
        self._status_waiter: asyncio.Task[SessionGitStatus] | None = None
        self._pending_status: tuple[
            SessionBinding,
            tuple[SequencedSessionChange, ...],
            RepositoryIdentity,
            int,
            int,
        ] | None = None
        self._rerun_available = False
        self._status_dirty = False
        self._status_request_generation = 0
        self._action_cycle: asyncio.Task[GitActionResult] | None = None
        self._action_waiter: asyncio.Task[GitActionResult] | None = None
        self._shutdown_settlement: Awaitable[None] | None = None

    async def discover(
        self,
        binding: SessionBinding,
    ) -> DiscoveryResult:
        """Discover repository/HEAD identity without inspecting worktree state."""
        if self._sealed:
            return DiscoveryResult(
                "unavailable",
                message="File Notes Git service is shut down",
            )
        if binding != self._owner.current_binding():
            return DiscoveryResult(
                "unavailable",
                message="File Notes root binding is stale",
            )
        root = self._safe_root(binding)
        if root is None:
            return DiscoveryResult(
                "unsafe_root",
                message="Selected File Notes root is not a safe directory",
            )
        if self._git_executable is None:
            return DiscoveryResult(
                "unavailable",
                message="Git is not installed",
            )

        inside = await self._run_discovery(
            root,
            ("rev-parse", "--is-inside-work-tree"),
        )
        if inside is None:
            return DiscoveryResult(
                "unavailable",
                message="Git repository discovery failed",
            )
        if inside.returncode != 0:
            diagnostic = sanitize_git_stderr(inside.stderr)
            normalized_diagnostic = diagnostic.lower()
            safety_markers = (
                "dubious ownership",
                "safe.directory",
                "permission denied",
                "operation not permitted",
                "access is denied",
            )
            if (
                "not a git repository" in normalized_diagnostic
                and not any(
                    marker in normalized_diagnostic
                    for marker in safety_markers
                )
            ):
                return DiscoveryResult(
                    "not_repository",
                    message=(
                        "Selected File Notes root is not in a Git worktree"
                    ),
                )
            message = "Git refused repository discovery"
            if diagnostic:
                message = f"{message}: {diagnostic}"
            return DiscoveryResult(
                "unsafe_root",
                message=message,
            )
        if inside.stdout != b"true\n":
            return DiscoveryResult(
                "not_repository",
                message="Selected File Notes root is not in a Git worktree",
            )

        resolved_paths = await self._read_repository_paths(root)
        if resolved_paths is None:
            return DiscoveryResult(
                "unsupported",
                message="Git returned an unsupported repository mapping",
            )
        worktree_root, git_dir, git_common_dir = resolved_paths
        try:
            root.relative_to(worktree_root)
        except ValueError:
            return DiscoveryResult(
                "unsafe_root",
                message="Selected File Notes root is outside the Git worktree",
            )

        repository = RepositoryIdentity(
            worktree_root=str(worktree_root),
            git_dir=str(git_dir),
            git_common_dir=str(git_common_dir),
            worktree_identity=_filesystem_identity(worktree_root),
            git_dir_identity=_filesystem_identity(git_dir),
            git_common_dir_identity=_filesystem_identity(git_common_dir),
        )
        head_result = await self._read_head(root)
        if isinstance(head_result, _HeadReadFailure):
            return DiscoveryResult(
                (
                    "unavailable"
                    if head_result.kind == "unavailable"
                    else "unsupported"
                ),
                message=head_result.message,
            )
        return DiscoveryResult(
            "ready",
            repository=repository,
            head=head_result,
        )

    async def revalidate_repository(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> bool:
        """Rediscover repository mapping, then restat its trusted identities."""
        valid = False
        root = self._safe_root(binding)
        if (
            not self._sealed
            and root is not None
            and self._git_executable is not None
            and binding == self._owner.current_binding()
        ):
            resolved_paths = await self._read_repository_paths(root)
            valid = (
                not self._sealed
                and binding == self._owner.current_binding()
                and resolved_paths is not None
                and tuple(str(path) for path in resolved_paths)
                == (
                    repository.worktree_root,
                    repository.git_dir,
                    repository.git_common_dir,
                )
                and self._repository_identity_matches(binding, repository)
            )
        if not valid:
            self._owner.clear_trust_if_matches(binding, repository)
        return valid

    def retained_status(
        self,
        binding: SessionBinding,
    ) -> asyncio.Task[SessionGitStatus] | None:
        """Return matching active status work without requesting a rerun."""
        cycle = self._status_cycle
        waiter = self._status_waiter
        if (
            binding != self._owner.current_binding()
            or cycle is None
            or cycle.done()
            or waiter is None
            or waiter.done()
        ):
            return None
        pending = self._pending_status
        if self._status_cycle_binding == binding:
            return waiter
        if pending is not None and pending[0] == binding:
            return waiter
        return None

    def start_status(
        self,
        binding: SessionBinding,
        changes: tuple[SequencedSessionChange, ...],
    ) -> asyncio.Task[SessionGitStatus]:
        """Synchronously admit one retained trusted status query."""
        if self._sealed:
            raise GitStatusAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        snapshot = self._owner.snapshot(binding)
        repository = snapshot.trusted_repository
        if binding != self._owner.current_binding():
            raise GitStatusAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        if repository is None:
            raise GitStatusAdmissionError(
                "untrusted",
                "Repository trust is required before Git status",
            )
        if self._status_cycle is not None and not self._status_cycle.done():
            admission = self._owner.admit_status(binding)
            if admission.reason == "mutation_active":
                raise GitStatusAdmissionError(
                    "mutation_active",
                    "A File Notes Git mutation is active",
                )
            if admission.reason != "status_active":
                if admission.lease is not None:
                    admission.lease.release()
                reason = admission.reason or "status_active"
                raise GitStatusAdmissionError(
                    reason,
                    "File Notes Git status cannot be coalesced",
                )
            if self._rerun_available:
                invalidation_generation = (
                    admission.invalidation_generation
                )
                if invalidation_generation is None:
                    raise RuntimeError(
                        "Active status admission omitted its generation"
                    )
                self._status_request_generation += 1
                self._pending_status = (
                    binding,
                    tuple(changes),
                    repository,
                    self._status_request_generation,
                    invalidation_generation,
                )
            else:
                self._status_request_generation += 1
                self._status_dirty = True
            waiter = self._status_waiter
            if waiter is None or waiter.done():
                waiter = self._create_task(
                    self._shield_status_cycle(self._status_cycle)
                )
                self._status_waiter = waiter
            return waiter

        admission = self._owner.admit_status(binding)
        lease = admission.lease
        if lease is None:
            reason = admission.reason or "status_active"
            raise GitStatusAdmissionError(
                reason,
                "File Notes Git status admission was refused",
            )
        self._status_request_generation += 1
        request_generation = self._status_request_generation
        self._pending_status = None
        self._rerun_available = True
        self._status_dirty = False
        cycle: asyncio.Task[SessionGitStatus] | None = None
        try:
            cycle = self._create_task(
                self._run_status_cycle(
                    binding,
                    tuple(changes),
                    repository,
                    lease,
                    request_generation,
                )
            )
            waiter = self._create_task(self._shield_status_cycle(cycle))
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            lease.release()
            self._pending_status = None
            self._rerun_available = False
            self._status_dirty = False
            raise
        self._status_cycle = cycle
        self._status_cycle_binding = binding
        self._status_waiter = waiter
        cycle.add_done_callback(self._status_cycle_completed)
        return waiter

    def start_stage(
        self,
        binding: SessionBinding,
        group_ids: Collection[int],
    ) -> asyncio.Task[GitActionResult]:
        """Synchronously admit and retain one exact Stage operation."""
        requested = tuple(dict.fromkeys(group_ids))
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        snapshot = self._owner.snapshot(binding)
        if binding != self._owner.current_binding():
            raise GitMutationAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        repository = snapshot.trusted_repository
        if repository is None:
            raise GitMutationAdmissionError(
                "untrusted",
                "Repository trust is required before staging",
            )
        admission = self._owner.admit_mutation(binding)
        lease = admission.lease
        if lease is None:
            reason = admission.reason or "mutation_active"
            raise GitMutationAdmissionError(
                reason,
                "File Notes Git mutation admission was refused",
            )
        status_cycle = self._status_cycle
        cycle: asyncio.Task[GitActionResult] | None = None
        try:
            cycle = self._create_action_task(
                self._run_stage_cycle(
                    binding,
                    requested,
                    repository,
                    lease,
                    status_cycle,
                )
            )
            waiter = self._create_action_task(
                self._shield_action_cycle(cycle)
            )
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            lease.release()
            raise
        self._action_cycle = cycle
        self._action_waiter = waiter
        cycle.add_done_callback(self._action_cycle_completed)
        return waiter

    def start_unstage(
        self,
        binding: SessionBinding,
        group_ids: Collection[int],
    ) -> asyncio.Task[GitActionResult]:
        """Synchronously admit and retain one exact Unstage operation."""
        requested = tuple(dict.fromkeys(group_ids))
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        snapshot = self._owner.snapshot(binding)
        if binding != self._owner.current_binding():
            raise GitMutationAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        repository = snapshot.trusted_repository
        if repository is None:
            raise GitMutationAdmissionError(
                "untrusted",
                "Repository trust is required before unstaging",
            )
        admission = self._owner.admit_mutation(binding)
        lease = admission.lease
        if lease is None:
            reason = admission.reason or "mutation_active"
            raise GitMutationAdmissionError(
                reason,
                "File Notes Git mutation admission was refused",
            )
        status_cycle = self._status_cycle
        cycle: asyncio.Task[GitActionResult] | None = None
        try:
            cycle = self._create_action_task(
                self._run_unstage_cycle(
                    binding,
                    requested,
                    repository,
                    lease,
                    status_cycle,
                )
            )
            waiter = self._create_action_task(
                self._shield_action_cycle(cycle)
            )
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            lease.release()
            raise
        self._action_cycle = cycle
        self._action_waiter = waiter
        cycle.add_done_callback(self._action_cycle_completed)
        return waiter

    async def _run_unstage_cycle(
        self,
        binding: SessionBinding,
        requested: tuple[int, ...],
        repository: RepositoryIdentity,
        lease: GitMutationLease,
        status_cycle: asyncio.Task[SessionGitStatus] | None,
    ) -> GitActionResult:
        try:
            if status_cycle is not None and not status_cycle.done():
                await asyncio.shield(status_cycle)
            if self._sealed:
                return GitActionResult(
                    "unstage",
                    "uncertain",
                    requested,
                    message="File Notes Git service shut down during Unstage",
                )
            inspection = await self._inspect_unstage(binding, repository)
            if isinstance(inspection, GitActionResult):
                return replace(
                    inspection,
                    action="unstage",
                    requested_group_ids=requested,
                    blocked_group_ids=(
                        requested
                        if inspection.state == "blocked"
                        else inspection.blocked_group_ids
                    ),
                )
            return await self._apply_unstage(binding, requested, inspection)
        finally:
            lease.release()

    async def _run_stage_cycle(
        self,
        binding: SessionBinding,
        requested: tuple[int, ...],
        repository: RepositoryIdentity,
        lease: GitMutationLease,
        status_cycle: asyncio.Task[SessionGitStatus] | None,
    ) -> GitActionResult:
        try:
            if status_cycle is not None and not status_cycle.done():
                await asyncio.shield(status_cycle)
            if self._sealed:
                return GitActionResult(
                    "stage",
                    "uncertain",
                    requested,
                    message="File Notes Git service shut down during Stage",
                )
            inspection = await self._inspect_stage(
                binding,
                repository,
            )
            if isinstance(inspection, GitActionResult):
                return replace(
                    inspection,
                    requested_group_ids=requested,
                    blocked_group_ids=(
                        requested
                        if inspection.state == "blocked"
                        else inspection.blocked_group_ids
                    ),
                )
            return await self._apply_stage(
                binding,
                requested,
                inspection,
            )
        finally:
            lease.release()

    async def _shield_action_cycle(
        self,
        cycle: asyncio.Task[GitActionResult],
    ) -> GitActionResult:
        return await asyncio.shield(cycle)

    def _action_cycle_completed(
        self,
        cycle: asyncio.Task[GitActionResult],
    ) -> None:
        if self._action_cycle is cycle:
            self._action_cycle = None
            self._action_waiter = None

    def _create_action_task(
        self,
        coroutine: object,
    ) -> asyncio.Task[GitActionResult]:
        try:
            return asyncio.get_running_loop().create_task(coroutine)  # type: ignore[arg-type]
        except BaseException:
            close = getattr(coroutine, "close", None)
            if close is not None:
                close()
            raise

    def shutdown(self) -> Awaitable[None]:
        """Seal admission and return retained finite service settlement."""
        if self._shutdown_settlement is not None:
            return self._shutdown_settlement
        cycle = self._status_cycle
        waiter = self._status_waiter
        action_cycle = self._action_cycle
        action_waiter = self._action_waiter
        active_task = next(
            (
                task
                for task in (cycle, waiter, action_cycle, action_waiter)
                if task is not None and not task.done()
            ),
            None,
        )
        if active_task is not None:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError as error:
                raise GitShutdownAffinityError(
                    "Active Git shutdown must be initiated from its event loop"
                ) from error
            if running_loop is not active_task.get_loop():
                raise GitShutdownAffinityError(
                    "Active Git shutdown must be initiated from its event loop"
                )
        runner_settlement = self._runner.shutdown()
        self._sealed = True
        binding = self._owner.current_binding()
        self._pending_status = None
        self._rerun_available = False
        self._status_dirty = False
        if binding is not None:
            self._owner.clear_ownership(binding)
            self._owner.clear_status(binding)
        if (
            (cycle is None or cycle.done())
            and (waiter is None or waiter.done())
            and (action_cycle is None or action_cycle.done())
            and (action_waiter is None or action_waiter.done())
            and (
                runner_settlement is None
                or isinstance(runner_settlement, _ImmediateSettlement)
            )
        ):
            self._status_cycle = None
            self._status_cycle_binding = None
            self._status_waiter = None
            self._action_cycle = None
            self._action_waiter = None
            self._shutdown_settlement = _ImmediateSettlement(None)
            return self._shutdown_settlement
        settlement = _RetainedSettlement(
            asyncio.get_running_loop().create_task(
                self._settle_shutdown(
                    binding,
                    cycle,
                    waiter,
                    action_cycle,
                    action_waiter,
                    runner_settlement,
                )
            )
        )
        self._shutdown_settlement = settlement
        return settlement

    async def _settle_shutdown(
        self,
        binding: SessionBinding | None,
        cycle: asyncio.Task[SessionGitStatus] | None,
        waiter: asyncio.Task[SessionGitStatus] | None,
        action_cycle: asyncio.Task[GitActionResult] | None,
        action_waiter: asyncio.Task[GitActionResult] | None,
        runner_settlement: Awaitable[bool] | None,
    ) -> None:
        """Join every retained task and preserve fail-closed shutdown state."""
        owned_tasks = tuple(
            task
            for task in (cycle, waiter, action_cycle, action_waiter)
            if task is not None and not task.done()
        )
        results: list[object] = []
        if owned_tasks:
            results.extend(
                await asyncio.gather(
                    *(asyncio.shield(task) for task in owned_tasks),
                    return_exceptions=True,
                )
            )
        runner_confirmed = True
        if runner_settlement is not None:
            try:
                runner_confirmed = bool(
                    await asyncio.shield(runner_settlement)
                    if isinstance(runner_settlement, asyncio.Future)
                    else await runner_settlement
                )
            except BaseException as error:
                results.append(error)
                runner_confirmed = False
        self._status_cycle = None
        self._status_cycle_binding = None
        self._status_waiter = None
        self._action_cycle = None
        self._action_waiter = None
        self._pending_status = None
        self._rerun_available = False
        self._status_dirty = False
        if binding is not None and (
            not runner_confirmed
            or any(isinstance(result, BaseException) for result in results)
        ):
            self._owner.clear_ownership(binding)
            self._owner.clear_status(binding)

    async def _shield_status_cycle(
        self,
        cycle: asyncio.Task[SessionGitStatus],
    ) -> SessionGitStatus:
        return await asyncio.shield(cycle)

    def _status_cycle_completed(
        self,
        cycle: asyncio.Task[SessionGitStatus],
    ) -> None:
        if self._status_cycle is cycle:
            self._status_cycle = None
            self._status_cycle_binding = None
            self._pending_status = None
            self._rerun_available = False
            self._status_dirty = False

    async def _run_status_cycle(
        self,
        binding: SessionBinding,
        changes: tuple[SequencedSessionChange, ...],
        repository: RepositoryIdentity,
        lease: GitStatusLease,
        request_generation: int,
    ) -> SessionGitStatus:
        invalidation_generation = lease.invalidation_generation
        try:
            result = await self._query_status(
                binding,
                changes,
                repository,
            )
            lease.release()
            pending = self._pending_status
            self._pending_status = None
            if pending is None:
                return self._publish_cycle_result(
                    binding,
                    result,
                    request_generation,
                    invalidation_generation,
                )

            (
                pending_binding,
                pending_changes,
                pending_repository,
                pending_generation,
                pending_invalidation_generation,
            ) = pending
            self._status_cycle_binding = pending_binding
            admission = self._owner.admit_status(pending_binding)
            next_lease = admission.lease
            if next_lease is None:
                self._rerun_available = False
                message = (
                    "Git status rerun was suppressed because a mutation "
                    "was admitted"
                )
                stale = self._local_status(
                    pending_binding,
                    "stale",
                    rows=self._disabled_rows(result.rows, message),
                    repository=pending_repository,
                    head=result.head,
                    message=message,
                )
                return self._publish_cycle_result(
                    pending_binding,
                    stale,
                    pending_generation,
                    pending_invalidation_generation,
                )

            lease = next_lease
            invalidation_generation = pending_invalidation_generation
            self._rerun_available = False
            result = await self._query_status(
                pending_binding,
                pending_changes,
                pending_repository,
            )
            if self._status_dirty:
                self._status_dirty = False
                dirty_generation = self._status_request_generation
                message = (
                    "Newer File Notes changes are known; refresh Git status"
                )
                result = self._local_status(
                    pending_binding,
                    "stale",
                    rows=self._disabled_rows(result.rows, message),
                    repository=result.repository,
                    head=result.head,
                    message=message,
                )
                pending_generation = dirty_generation
            return self._publish_cycle_result(
                pending_binding,
                result,
                pending_generation,
                invalidation_generation,
            )
        finally:
            lease.release()

    def _create_task(
        self,
        coroutine: object,
    ) -> asyncio.Task[SessionGitStatus]:
        try:
            return asyncio.get_running_loop().create_task(coroutine)  # type: ignore[arg-type]
        except BaseException:
            close = getattr(coroutine, "close", None)
            if close is not None:
                close()
            raise

    def _publish_cycle_result(
        self,
        binding: SessionBinding,
        status: SessionGitStatus,
        request_generation: int,
        invalidation_generation: int,
    ) -> SessionGitStatus:
        if (
            not self._sealed
            and request_generation == self._status_request_generation
            and binding == self._owner.current_binding()
        ):
            self._owner.publish_status(
                binding,
                status,
                invalidation_generation=invalidation_generation,
            )
        return status

    async def _query_status(
        self,
        binding: SessionBinding,
        changes: tuple[SequencedSessionChange, ...],
        repository: RepositoryIdentity,
    ) -> SessionGitStatus:
        if (
            self._owner.snapshot(binding).trusted_repository != repository
            or not await self.revalidate_repository(binding, repository)
        ):
            return self._local_status(
                binding,
                "stale",
                message="Repository identity changed; trust was cleared",
            )

        root = self._safe_root(binding)
        if root is None:
            self._owner.clear_trust(binding)
            return self._local_status(
                binding,
                "stale",
                message="Selected File Notes root is no longer safe",
            )
        sparse = await self._sparse_checkout_state(repository)
        if sparse is None:
            return self._failed_status(
                binding,
                "error",
                repository=repository,
                message="Unable to verify sparse-checkout state",
            )
        if sparse:
            return self._failed_status(
                binding,
                "unavailable",
                repository=repository,
                message="Sparse checkout or sparse index is unsupported",
            )

        groups = coalesce_session_changes(changes)
        repository_groups: list[SessionChangeGroup] = []
        original_groups: dict[int, SessionChangeGroup] = {}
        invalid_rows: dict[int, SessionGitRow] = {}
        for group in groups:
            mapped_group, invalid_row = self._map_group(
                root,
                repository,
                group,
            )
            if invalid_row is not None:
                invalid_rows[group.group_id] = invalid_row
                continue
            assert mapped_group is not None
            repository_groups.append(mapped_group)
            original_groups[group.group_id] = group

        raw = await self._read_raw_git_inspection(
            binding,
            root,
            repository,
            repository_groups,
        )
        if isinstance(raw, _RawGitInspectionFailure):
            if raw.revoke_ownership:
                self._owner.clear_ownership(binding)
            return self._failed_status(
                binding,
                (
                    "stale"
                    if raw.state == "uncertain"
                    else raw.state
                ),
                repository=repository,
                head=raw.head,
                message=raw.message,
            )
        head = raw.head
        index_sequence = raw.index_entries
        index_entries = _stage_zero_index(index_sequence)
        conflicted_paths = {
            entry.path for entry in index_sequence if entry.stage != 0
        }
        status_records = raw.status_records

        ownership_by_id: dict[int, StagingOwnership] = {}
        current_ownership = self._owner.snapshot(binding).staging_ownership
        retained_ownership = dict(current_ownership)
        for repository_group in repository_groups:
            owned = current_ownership.get(repository_group.group_id)
            if owned is None:
                continue
            original_group = original_groups[repository_group.group_id]
            mapped_ownership = _map_ownership_topology(
                owned,
                original_group,
                repository_group,
            )
            if (
                mapped_ownership is None
                or mapped_ownership.repository != repository
                or mapped_ownership.head != head
                or any(
                    path in conflicted_paths
                    or index_entries.get(path) != expected
                    for path, expected in mapped_ownership.post_stage_entries.items()
                )
            ):
                retained_ownership.pop(repository_group.group_id, None)
                continue
            ownership_by_id[repository_group.group_id] = mapped_ownership
        if len(retained_ownership) != len(current_ownership):
            self._owner.publish_ownership(binding, retained_ownership)

        classified = classify_session_rows(
            repository_groups,
            status_records,
            index_sequence,
            ownership_by_id,
        )
        classified_by_id = {
            row.group_id: replace(
                row,
                group=original_groups[row.group_id],
            )
            for row in classified
        }
        rows = tuple(
            invalid_rows.get(group.group_id)
            or classified_by_id[group.group_id]
            for group in groups
        )
        return self._local_status(
            binding,
            "ready",
            rows=rows,
            repository=repository,
            head=head,
        )

    async def _read_raw_git_inspection(
        self,
        binding: SessionBinding,
        root: Path,
        repository: RepositoryIdentity,
        groups: Sequence[SessionChangeGroup],
    ) -> _RawGitInspection | _RawGitInspectionFailure:
        """Read the shared fresh HEAD, complete index, and scoped status."""
        head_result = await self._read_head(root)
        if isinstance(head_result, _HeadReadFailure):
            return _RawGitInspectionFailure(
                (
                    "unavailable"
                    if head_result.kind == "unavailable"
                    else "error"
                ),
                head_result.message,
                revoke_ownership=True,
            )
        current_ownership = self._owner.snapshot(binding).staging_ownership
        retained_ownership = {
            group_id: ownership
            for group_id, ownership in current_ownership.items()
            if (
                ownership.repository == repository
                and ownership.head == head_result
            )
        }
        if len(retained_ownership) != len(current_ownership):
            self._owner.publish_ownership(binding, retained_ownership)

        index_result = await self._run_status_command(
            repository,
            build_index_argv(self._git_executable_or_raise()),
        )
        if not _command_succeeded(index_result):
            return _RawGitInspectionFailure(
                (
                    "uncertain"
                    if index_result.termination_uncertain
                    else "stale"
                ),
                _command_failure_message(index_result, "Git index read failed"),
                head=head_result,
                revoke_ownership=index_result.termination_uncertain,
            )
        try:
            index_entries = parse_index_entries_z(index_result.stdout)
        except GitIndexParseError as error:
            return _RawGitInspectionFailure(
                "error",
                str(error),
                head=head_result,
                revoke_ownership=True,
            )

        index_by_path = _stage_zero_index(index_entries)
        conflicted_paths = {
            entry.path for entry in index_entries if entry.stage != 0
        }
        current_ownership = self._owner.snapshot(binding).staging_ownership
        retained_ownership = {
            group_id: ownership
            for group_id, ownership in current_ownership.items()
            if all(
                path not in conflicted_paths
                and index_by_path.get(path) == expected
                for path, expected in ownership.post_stage_entries.items()
            )
        }
        if len(retained_ownership) != len(current_ownership):
            self._owner.publish_ownership(binding, retained_ownership)

        status_records: tuple[PorcelainRecord, ...] = ()
        repository_paths = tuple(
            os.fsencode(path)
            for group in groups
            for path in group.endpoints
        )
        if repository_paths:
            allowed_paths = frozenset(
                {
                    *(path for group in groups for path in group.endpoints),
                    *(
                        entry.path
                        for entry in index_entries
                        if any(
                            _paths_overlap(entry.path, endpoint)
                            for group in groups
                            for endpoint in group.endpoints
                        )
                    ),
                }
            )
            status_result = await self._run_status_command(
                repository,
                build_status_argv(
                    self._git_executable_or_raise(),
                    repository_paths,
                ),
            )
            if not _command_succeeded(status_result):
                return _RawGitInspectionFailure(
                    (
                        "uncertain"
                        if status_result.termination_uncertain
                        else "stale"
                    ),
                    _command_failure_message(status_result, "Git status failed"),
                    head=head_result,
                )
            try:
                status_records = parse_porcelain_v2_z(
                    status_result.stdout,
                    allowed_paths=allowed_paths,
                )
            except PorcelainV2ParseError as error:
                return _RawGitInspectionFailure(
                    "error",
                    str(error),
                    head=head_result,
                )
        return _RawGitInspection(
            head_result,
            index_entries,
            status_records,
        )

    async def _inspect_stage(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> _StageInspection | GitActionResult:
        """Read one fresh identity/HEAD/index/status Stage preflight."""
        failure = lambda state, message: GitActionResult(  # noqa: E731
            "stage",
            state,
            (),
            message=message,
        )
        snapshot = self._owner.snapshot(binding)
        if (
            snapshot.trusted_repository != repository
            or not await self.revalidate_repository(binding, repository)
        ):
            return failure("stale", "Repository identity changed; trust was cleared")
        root = self._safe_root(binding)
        if root is None:
            self._owner.clear_trust_if_matches(binding, repository)
            return failure("stale", "Selected File Notes root is no longer safe")
        sparse = await self._sparse_checkout_state(repository)
        if sparse is None:
            return failure("error", "Unable to verify sparse-checkout state")
        if sparse:
            return failure(
                "blocked",
                "Sparse checkout or sparse index is unsupported",
            )

        groups = coalesce_session_changes(snapshot.changes)
        repository_groups: dict[int, SessionChangeGroup] = {}
        invalid_rows: dict[int, SessionGitRow] = {}
        for group in groups:
            mapped, invalid = self._map_group(root, repository, group)
            if invalid is not None:
                invalid_rows[group.group_id] = invalid
            else:
                assert mapped is not None
                repository_groups[group.group_id] = mapped

        raw = await self._read_raw_git_inspection(
            binding,
            root,
            repository,
            tuple(repository_groups.values()),
        )
        if isinstance(raw, _RawGitInspectionFailure):
            if raw.revoke_ownership:
                self._owner.clear_ownership(binding)
            return failure(
                (
                    "uncertain"
                    if raw.state in {"stale", "unavailable", "uncertain"}
                    else "error"
                ),
                raw.message,
            )
        head_result = raw.head
        index_sequence = raw.index_entries
        index_entries = _stage_zero_index(index_sequence)
        status_records = raw.status_records

        ownership_by_id: dict[int, StagingOwnership] = {}
        groups_by_id = {group.group_id: group for group in groups}
        for group_id, owned in snapshot.staging_ownership.items():
            original_group = groups_by_id.get(group_id)
            repository_group = repository_groups.get(group_id)
            if original_group is None or repository_group is None:
                continue
            mapped = _map_ownership_topology(
                owned,
                original_group,
                repository_group,
            )
            if mapped is not None:
                ownership_by_id[group_id] = mapped

        classified = classify_session_rows(
            tuple(repository_groups.values()),
            status_records,
            index_sequence,
            ownership_by_id,
        )
        rows = {row.group_id: row for row in classified}
        rows.update(invalid_rows)
        return _StageInspection(
            repository=repository,
            head=head_result,
            groups=groups,
            repository_groups=repository_groups,
            rows=rows,
            index_sequence=index_sequence,
            index_entries=index_entries,
            status_records=status_records,
        )

    async def _inspect_unstage(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> _StageInspection | GitActionResult:
        """Read exact Unstage facts without imposing Stage worktree types."""
        failure = lambda state, message: GitActionResult(  # noqa: E731
            "unstage",
            state,
            (),
            message=message,
        )
        snapshot = self._owner.snapshot(binding)
        if (
            snapshot.trusted_repository != repository
            or not await self.revalidate_repository(binding, repository)
        ):
            return failure("stale", "Repository identity changed; trust was cleared")
        root = self._safe_root(binding)
        if root is None:
            self._owner.clear_trust_if_matches(binding, repository)
            return failure("stale", "Selected File Notes root is no longer safe")
        sparse = await self._sparse_checkout_state(repository)
        if sparse is None:
            return failure("error", "Unable to verify sparse-checkout state")
        if sparse:
            return failure(
                "blocked",
                "Sparse checkout or sparse index is unsupported",
            )

        groups = coalesce_session_changes(snapshot.changes)
        repository_groups: dict[int, SessionChangeGroup] = {}
        invalid_rows: dict[int, SessionGitRow] = {}
        for group in groups:
            mapped, invalid = self._map_group_for_unstage(
                root,
                repository,
                group,
            )
            if invalid is not None:
                invalid_rows[group.group_id] = invalid
            else:
                assert mapped is not None
                repository_groups[group.group_id] = mapped

        raw = await self._read_raw_git_inspection(
            binding,
            root,
            repository,
            tuple(repository_groups.values()),
        )
        if isinstance(raw, _RawGitInspectionFailure):
            if raw.revoke_ownership:
                self._owner.clear_ownership(binding)
            return failure(
                (
                    "uncertain"
                    if raw.state in {"stale", "unavailable", "uncertain"}
                    else "error"
                ),
                raw.message,
            )
        index_entries = _stage_zero_index(raw.index_entries)
        ownership_by_id: dict[int, StagingOwnership] = {}
        groups_by_id = {group.group_id: group for group in groups}
        for group_id, owned in snapshot.staging_ownership.items():
            original_group = groups_by_id.get(group_id)
            repository_group = repository_groups.get(group_id)
            if original_group is None or repository_group is None:
                continue
            mapped = _map_ownership_topology(
                owned,
                original_group,
                repository_group,
            )
            if mapped is not None:
                ownership_by_id[group_id] = mapped
        rows = {
            row.group_id: row
            for row in classify_session_rows(
                tuple(repository_groups.values()),
                raw.status_records,
                raw.index_entries,
                ownership_by_id,
            )
        }
        rows.update(invalid_rows)
        return _StageInspection(
            repository=repository,
            head=raw.head,
            groups=groups,
            repository_groups=repository_groups,
            rows=rows,
            index_sequence=raw.index_entries,
            index_entries=index_entries,
            status_records=raw.status_records,
        )

    async def _apply_stage(
        self,
        binding: SessionBinding,
        requested: tuple[int, ...],
        inspection: _StageInspection,
    ) -> GitActionResult:
        """Apply one checked Stage command and publish exact ownership."""
        snapshot = self._owner.snapshot(binding)
        ownership = dict(snapshot.staging_ownership)
        staged: list[int] = []
        clean: list[int] = []
        blocked: list[int] = []
        pathspecs: list[bytes] = []
        baselines: dict[int, dict[str, IndexBaseline]] = {}
        affected_paths: dict[int, tuple[str, ...]] = {}
        groups_by_id = {group.group_id: group for group in inspection.groups}
        ownership_revoked = False

        for group_id in requested:
            group = groups_by_id.get(group_id)
            repository_group = inspection.repository_groups.get(group_id)
            row = inspection.rows.get(group_id)
            if group is None or repository_group is None or row is None:
                blocked.append(group_id)
                continue
            current_owned = ownership.get(group_id)
            owned_clean = False
            if current_owned is None:
                eligible = row.state == "unstaged" and row.stage_action == "stage"
            else:
                owned_matches = self._owned_stage_preflight_matches(
                    current_owned,
                    inspection,
                    repository_group,
                )
                eligible = (
                    owned_matches
                    and row.state
                    in {"owned_newer_edits", "owned_topology_changed"}
                    and row.stage_action == "stage_update"
                )
                owned_clean = (
                    owned_matches
                    and row.state == "owned"
                    and row.stage_action is None
                )
            effective = stage_pathspecs(
                repository_group,
                inspection.status_records,
                groups=tuple(inspection.repository_groups.values()),
            )
            if not eligible:
                if owned_clean:
                    clean.append(group_id)
                    continue
                if current_owned is not None:
                    ownership.pop(group_id, None)
                    ownership_revoked = True
                if row.state == "clean":
                    clean.append(group_id)
                else:
                    blocked.append(group_id)
                continue
            if not effective:
                clean.append(group_id)
                continue
            decoded = tuple(os.fsdecode(path) for path in effective)
            staged.append(group_id)
            pathspecs.extend(effective)
            affected_paths[group_id] = decoded
            original = (
                dict(current_owned.original_baselines)
                if current_owned is not None
                else {}
            )
            for path in decoded:
                original.setdefault(
                    path,
                    IndexBaseline(inspection.index_entries.get(path)),
                )
            baselines[group_id] = original

        if not staged:
            if ownership_revoked:
                self._owner.publish_stage_result(
                    binding,
                    inspection.repository,
                    ownership,
                )
            return GitActionResult(
                "stage",
                "blocked",
                requested,
                clean_group_ids=tuple(clean),
                blocked_group_ids=tuple(blocked),
                message="No requested session group is eligible for Stage",
            )
        if not await self.revalidate_repository(binding, inspection.repository):
            return GitActionResult(
                "stage",
                "stale",
                requested,
                blocked_group_ids=tuple(requested),
                message="Repository identity changed before Stage",
            )
        root = self._safe_root(binding)
        endpoint_safety_changed = root is None
        if root is not None:
            for group_id in staged:
                remapped, invalid = self._map_group(
                    root,
                    inspection.repository,
                    groups_by_id[group_id],
                )
                if (
                    invalid is not None
                    or remapped is None
                    or remapped != inspection.repository_groups[group_id]
                    or not stage_group_is_closed(
                        remapped,
                        inspection.index_entries,
                    )
                ):
                    endpoint_safety_changed = True
                    break
        if endpoint_safety_changed:
            return GitActionResult(
                "stage",
                "blocked",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message="File Notes endpoint safety changed before Stage",
            )

        result = await self._run_stage_command(
            inspection.repository,
            build_stage_argv(
                self._git_executable_or_raise(),
                tuple(dict.fromkeys(pathspecs)),
            ),
        )
        if not _command_succeeded(result):
            for group_id in staged:
                ownership.pop(group_id, None)
            self._owner.publish_stage_result(
                binding,
                inspection.repository,
                ownership,
            )
            return GitActionResult(
                "stage",
                "uncertain" if result.termination_uncertain else "error",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message=_index_mutation_failure_message(
                    result,
                    "Git Stage failed",
                    inspection.repository,
                ),
            )

        postflight = await self._read_stage_postflight(
            binding,
            inspection.repository,
        )
        if isinstance(postflight, GitActionResult):
            self._owner.clear_ownership(binding)
            self._owner.clear_status(binding)
            return replace(postflight, requested_group_ids=requested)
        post_head, post_index = postflight
        latest_groups = {
            group.group_id: group
            for group in coalesce_session_changes(
                self._owner.snapshot(binding).changes
            )
        }
        if post_head != inspection.head or any(
            latest_groups.get(group_id) is None
            or latest_groups[group_id].topology_signature
            != groups_by_id[group_id].topology_signature
            for group_id in staged
        ):
            self._owner.clear_ownership(binding)
            self._owner.clear_status(binding)
            return GitActionResult(
                "stage",
                "uncertain",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message="Git HEAD or session topology changed during Stage",
            )
        if any(
            not stage_group_is_closed(
                inspection.repository_groups[group_id],
                post_index,
            )
            for group_id in staged
        ):
            self._owner.clear_ownership(binding)
            self._owner.clear_status(binding)
            return GitActionResult(
                "stage",
                "uncertain",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message="Git index topology changed during Stage",
            )

        for group_id in staged:
            previous = ownership.get(group_id)
            paths = set(affected_paths[group_id])
            if previous is not None:
                paths.update(previous.post_stage_entries)
            post_entries = {path: post_index.get(path) for path in paths}
            if any(
                entry is not None
                and (entry.stage != 0 or entry.semantic_flags)
                for entry in post_entries.values()
            ):
                self._owner.clear_ownership(binding)
                self._owner.clear_status(binding)
                return GitActionResult(
                    "stage",
                    "uncertain",
                    requested,
                    blocked_group_ids=tuple(staged + blocked),
                    clean_group_ids=tuple(clean),
                    message="Git index semantics changed during Stage",
                )
            if any(
                inspection.index_entries.get(path) == post_index.get(path)
                for path in affected_paths[group_id]
            ):
                self._owner.clear_ownership(binding)
                self._owner.clear_status(binding)
                return GitActionResult(
                    "stage",
                    "uncertain",
                    requested,
                    blocked_group_ids=tuple(staged + blocked),
                    clean_group_ids=tuple(clean),
                    message="Git Stage did not change every effective path",
                )
            ownership[group_id] = StagingOwnership(
                repository=inspection.repository,
                head=post_head,
                approved_endpoint_topology=groups_by_id[group_id].endpoints,
                approved_move_edges=groups_by_id[group_id].move_edges,
                approved_current_path=groups_by_id[group_id].current_path,
                original_baselines=baselines[group_id],
                post_stage_entries=post_entries,
            )

        if not self._owner.publish_stage_result(
            binding,
            inspection.repository,
            ownership,
        ):
            return GitActionResult(
                "stage",
                "uncertain",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message="Stage result no longer matches the selected session",
            )
        return GitActionResult(
            "stage",
            "success",
            requested,
            staged_group_ids=tuple(staged),
            clean_group_ids=tuple(clean),
            blocked_group_ids=tuple(blocked),
        )

    @staticmethod
    def _owned_stage_preflight_matches(
        ownership: StagingOwnership,
        inspection: _StageInspection,
        group: SessionChangeGroup,
    ) -> bool:
        if (
            ownership.repository != inspection.repository
            or ownership.head != inspection.head
            or any(
                inspection.index_entries.get(path) != expected
                for path, expected in ownership.post_stage_entries.items()
            )
            or not stage_group_is_closed(group, inspection.index_entries)
        ):
            return False
        return not any(
            path not in ownership.post_stage_entries
            for record in inspection.status_records
            for path in _staged_record_paths(record)
            if path in group.endpoints
        )

    async def _read_stage_postflight(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> tuple[HeadIdentity, Mapping[str, IndexEntry]] | GitActionResult:
        if not await self.revalidate_repository(binding, repository):
            return GitActionResult(
                "stage",
                "uncertain",
                (),
                message="Repository identity changed after Stage",
            )
        root = self._safe_root(binding)
        if root is None:
            return GitActionResult(
                "stage",
                "uncertain",
                (),
                message="Selected File Notes root changed after Stage",
            )
        head = await self._read_head(root)
        if isinstance(head, _HeadReadFailure):
            return GitActionResult(
                "stage",
                "uncertain",
                (),
                message=head.message,
            )
        index_result = await self._run_stage_command(
            repository,
            build_index_argv(self._git_executable_or_raise()),
        )
        if not _command_succeeded(index_result):
            return GitActionResult(
                "stage",
                "uncertain",
                (),
                message=_command_failure_message(
                    index_result,
                    "Git post-Stage index read failed",
                ),
            )
        try:
            entries = parse_index_entries_z(index_result.stdout)
        except GitIndexParseError as error:
            return GitActionResult("stage", "uncertain", (), message=str(error))
        return head, _stage_zero_index(entries)

    async def _run_stage_command(
        self,
        repository: RepositoryIdentity,
        argv: Sequence[GitArg],
    ) -> GitCommandResult:
        try:
            return await self._runner.run(
                argv,
                cwd=repository.worktree_root,
                environment=build_git_environment(self._environment),
            )
        except OSError as error:
            return GitCommandResult(
                127,
                b"",
                str(error).encode("utf-8", "replace"),
            )

    async def _apply_unstage(
        self,
        binding: SessionBinding,
        requested: tuple[int, ...],
        inspection: _StageInspection,
    ) -> GitActionResult:
        """Restore only exact saved baselines for valid selected ownership."""
        snapshot = self._owner.snapshot(binding)
        groups_by_id = {group.group_id: group for group in inspection.groups}
        valid: list[int] = []
        blocked: list[int] = []
        revoked: list[int] = []
        selected_ownership: dict[int, StagingOwnership] = {}
        mapped_ownership: dict[int, StagingOwnership] = {}

        for group_id in requested:
            owned = snapshot.staging_ownership.get(group_id)
            group = groups_by_id.get(group_id)
            repository_group = inspection.repository_groups.get(group_id)
            row = inspection.rows.get(group_id)
            if (
                owned is None
                or group is None
                or repository_group is None
                or row is None
            ):
                blocked.append(group_id)
                continue
            mapped = _map_ownership_topology(
                owned,
                group,
                repository_group,
            )
            if (
                mapped is None
                or owned.topology_signature != group.topology_signature
            ):
                blocked.append(group_id)
                continue
            if any(
                entry.stage != 0
                and entry.path in repository_group.endpoints
                for entry in inspection.index_sequence
            ):
                blocked.append(group_id)
                revoked.append(group_id)
                continue
            if not ownership_signature_matches(
                mapped,
                repository=inspection.repository,
                head=inspection.head,
                topology=repository_group.topology_signature,
                current_index_entries=inspection.index_entries,
            ):
                blocked.append(group_id)
                revoked.append(group_id)
                continue
            if (
                not row.unstage_eligible
                or not unstage_group_is_closed(
                    repository_group,
                    mapped.original_baselines,
                    inspection.index_entries,
                    mapped,
                )
            ):
                blocked.append(group_id)
                continue
            valid.append(group_id)
            selected_ownership[group_id] = owned
            mapped_ownership[group_id] = mapped

        if revoked and not self._owner.publish_unstage_result(
            binding,
            inspection.repository,
            {
                group_id: snapshot.staging_ownership[group_id]
                for group_id in revoked
            },
            revoked,
        ):
            return GitActionResult(
                "unstage",
                "uncertain",
                requested,
                blocked_group_ids=requested,
                message="Unstage ownership changed during preflight",
            )
        if not valid:
            return GitActionResult(
                "unstage",
                "blocked",
                requested,
                blocked_group_ids=tuple(blocked),
                message="No requested session group is eligible for Unstage",
            )
        if not await self.revalidate_repository(binding, inspection.repository):
            self._owner.clear_ownership(binding)
            return GitActionResult(
                "unstage",
                "stale",
                requested,
                blocked_group_ids=tuple(valid + blocked),
                message="Repository identity changed before Unstage",
            )

        try:
            payload = _build_combined_update_index_payload(
                tuple(mapped_ownership[group_id] for group_id in valid),
                inspection.index_entries,
            )
        except ValueError as error:
            self._owner.publish_unstage_result(
                binding,
                inspection.repository,
                selected_ownership,
                valid,
            )
            return GitActionResult(
                "unstage",
                "blocked",
                requested,
                blocked_group_ids=tuple(valid + blocked),
                message=str(error),
            )
        result = await self._run_unstage_command(
            inspection.repository,
            build_unstage_argv(self._git_executable_or_raise()),
            payload,
        )
        postflight = await self._read_unstage_postflight(
            binding,
            inspection.repository,
        )
        if isinstance(postflight, GitActionResult):
            if self._owner.snapshot(binding).trusted_repository == inspection.repository:
                self._owner.publish_unstage_result(
                    binding,
                    inspection.repository,
                    selected_ownership,
                    valid,
                )
            self._owner.clear_status(binding)
            return replace(
                postflight,
                requested_group_ids=requested,
                blocked_group_ids=tuple(valid + blocked),
            )
        post_head, post_index = postflight
        latest_groups = {
            group.group_id: group
            for group in coalesce_session_changes(
                self._owner.snapshot(binding).changes
            )
        }
        verified = (
            _command_succeeded(result)
            and post_head == inspection.head
            and all(
                latest_groups.get(group_id) is not None
                and latest_groups[group_id].topology_signature
                == groups_by_id[group_id].topology_signature
                and _baseline_matches_index(
                    mapped_ownership[group_id],
                    post_index,
                )
                for group_id in valid
            )
        )
        if not verified:
            if not self._owner.publish_unstage_result(
                binding,
                inspection.repository,
                selected_ownership,
                valid,
            ):
                self._owner.clear_ownership(binding)
            return GitActionResult(
                "unstage",
                (
                    "uncertain"
                    if result.termination_uncertain or _command_succeeded(result)
                    else "error"
                ),
                requested,
                blocked_group_ids=tuple(valid + blocked),
                message=(
                    "Unstage postflight did not verify the saved baseline"
                    if _command_succeeded(result)
                    else _index_mutation_failure_message(
                        result,
                        "Git Unstage failed",
                        inspection.repository,
                    )
                ),
            )
        if not self._owner.publish_unstage_result(
            binding,
            inspection.repository,
            selected_ownership,
            valid,
        ):
            return GitActionResult(
                "unstage",
                "uncertain",
                requested,
                blocked_group_ids=tuple(valid + blocked),
                message="Unstage result no longer matches the selected session",
            )
        return GitActionResult(
            "unstage",
            "success",
            requested,
            unstaged_group_ids=tuple(valid),
            blocked_group_ids=tuple(blocked),
        )

    async def _read_unstage_postflight(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> tuple[HeadIdentity, Mapping[str, IndexEntry]] | GitActionResult:
        if not await self.revalidate_repository(binding, repository):
            return GitActionResult(
                "unstage",
                "uncertain",
                (),
                message="Repository identity changed after Unstage",
            )
        root = self._safe_root(binding)
        if root is None:
            return GitActionResult(
                "unstage",
                "uncertain",
                (),
                message="Selected File Notes root changed after Unstage",
            )
        head = await self._read_head(root)
        if isinstance(head, _HeadReadFailure):
            return GitActionResult(
                "unstage",
                "uncertain",
                (),
                message=head.message,
            )
        index_result = await self._run_unstage_command(
            repository,
            build_index_argv(self._git_executable_or_raise()),
            None,
        )
        if not _command_succeeded(index_result):
            return GitActionResult(
                "unstage",
                "uncertain",
                (),
                message=_command_failure_message(
                    index_result,
                    "Git post-Unstage index read failed",
                ),
            )
        try:
            entries = parse_index_entries_z(index_result.stdout)
        except GitIndexParseError as error:
            return GitActionResult("unstage", "uncertain", (), message=str(error))
        return head, _stage_zero_index(entries)

    async def _run_unstage_command(
        self,
        repository: RepositoryIdentity,
        argv: Sequence[GitArg],
        stdin: bytes | None,
    ) -> GitCommandResult:
        try:
            return await self._runner.run(
                argv,
                cwd=repository.worktree_root,
                environment=build_git_environment(self._environment),
                stdin=stdin,
            )
        except OSError as error:
            return GitCommandResult(
                127,
                b"",
                str(error).encode("utf-8", "replace"),
            )

    async def _sparse_checkout_state(
        self,
        repository: RepositoryIdentity,
    ) -> bool | None:
        for key in ("core.sparseCheckout", "index.sparse"):
            result = await self._run_status_command(
                repository,
                (
                    self._git_executable_or_raise(),
                    "config",
                    "--bool",
                    "--get",
                    key,
                ),
            )
            if result.timed_out or result.termination_uncertain:
                return None
            if result.returncode == 1:
                continue
            if result.returncode != 0:
                return None
            if result.stdout == b"true\n":
                return True
            if result.stdout != b"false\n":
                return None
        return False

    async def _run_status_command(
        self,
        repository: RepositoryIdentity,
        argv: Sequence[GitArg],
    ) -> GitCommandResult:
        try:
            result = await self._runner.run(
                argv,
                cwd=repository.worktree_root,
                environment=build_git_environment(
                    self._environment,
                    for_status=True,
                ),
                timeout=self._status_timeout,
            )
        except OSError as error:
            return GitCommandResult(
                127,
                b"",
                str(error).encode("utf-8", "replace"),
            )
        if result.termination_uncertain:
            binding = self._owner.current_binding()
            if (
                binding is not None
                and self._owner.snapshot(binding).trusted_repository
                == repository
            ):
                self._owner.clear_ownership(binding)
        return result

    def _map_group(
        self,
        notes_root: Path,
        repository: RepositoryIdentity,
        group: SessionChangeGroup,
    ) -> tuple[SessionChangeGroup | None, SessionGitRow | None]:
        mapping: dict[str, str] = {}
        for endpoint in group.endpoints:
            mapped, problem = _map_session_endpoint(
                notes_root,
                Path(repository.worktree_root),
                endpoint,
            )
            if problem == "nested_repository":
                return None, SessionGitRow(
                    group,
                    "nested_repository",
                    disabled_reason="Nested repository unsupported",
                )
            if problem is not None:
                return None, SessionGitRow(
                    group,
                    "unsupported",
                    disabled_reason=problem,
                )
            assert mapped is not None
            mapping[endpoint] = mapped
        return (
            replace(
                group,
                endpoints=tuple(mapping[path] for path in group.endpoints),
                source_path=mapping[group.source_path],
                destination_path=(
                    None
                    if group.destination_path is None
                    else mapping[group.destination_path]
                ),
                current_path=mapping[group.current_path],
                move_edges=tuple(
                    (mapping[source], mapping[destination])
                    for source, destination in group.move_edges
                ),
            ),
            None,
        )

    def _map_group_for_unstage(
        self,
        notes_root: Path,
        repository: RepositoryIdentity,
        group: SessionChangeGroup,
    ) -> tuple[SessionChangeGroup | None, SessionGitRow | None]:
        """Map safe endpoint text without rejecting owned replacements."""
        worktree_root = Path(repository.worktree_root)
        try:
            root_prefix = notes_root.relative_to(worktree_root)
        except ValueError:
            return None, SessionGitRow(
                group,
                "unsupported",
                disabled_reason="Unsafe File Notes path",
            )
        mapping: dict[str, str] = {}
        for endpoint in group.endpoints:
            components = endpoint.split("/")
            if (
                not endpoint
                or endpoint.startswith("/")
                or "\0" in endpoint
                or ".git" in components
                or any(component in {"", ".", ".."} for component in components)
            ):
                return None, SessionGitRow(
                    group,
                    "unsupported",
                    disabled_reason="Unsafe File Notes path",
                )
            for depth in range(1, len(components) + 1):
                boundary = notes_root.joinpath(*components[:depth])
                try:
                    boundary_stat = boundary.stat(follow_symlinks=False)
                except (FileNotFoundError, NotADirectoryError):
                    continue
                except OSError:
                    return None, SessionGitRow(
                        group,
                        "unsupported",
                        disabled_reason="Unsafe File Notes path",
                    )
                if stat.S_ISLNK(boundary_stat.st_mode):
                    return None, SessionGitRow(
                        group,
                        "unsupported",
                        disabled_reason="Unsafe File Notes path",
                    )
                if stat.S_ISDIR(boundary_stat.st_mode):
                    try:
                        (boundary / ".git").stat(follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        return None, SessionGitRow(
                            group,
                            "unsupported",
                            disabled_reason="Unsafe nested repository boundary",
                        )
                    else:
                        return None, SessionGitRow(
                            group,
                            "nested_repository",
                            disabled_reason="Nested repository unsupported",
                        )
            mapped = root_prefix.joinpath(*components).as_posix()
            if not mapped or mapped == ".":
                return None, SessionGitRow(
                    group,
                    "unsupported",
                    disabled_reason="Unsafe File Notes path",
                )
            mapping[endpoint] = mapped
        return (
            replace(
                group,
                endpoints=tuple(mapping[path] for path in group.endpoints),
                source_path=mapping[group.source_path],
                destination_path=(
                    None
                    if group.destination_path is None
                    else mapping[group.destination_path]
                ),
                current_path=mapping[group.current_path],
                move_edges=tuple(
                    (mapping[source], mapping[destination])
                    for source, destination in group.move_edges
                ),
            ),
            None,
        )

    def _failed_status(
        self,
        binding: SessionBinding,
        state: Literal["stale", "unavailable", "error"],
        *,
        repository: RepositoryIdentity,
        head: HeadIdentity | None = None,
        message: str | None = None,
    ) -> SessionGitStatus:
        rows: tuple[SessionGitRow, ...] = ()
        snapshot = self._owner.snapshot(binding)
        previous = snapshot.git_status
        if (
            binding == self._owner.current_binding()
            and snapshot.trusted_repository == repository
            and previous is not None
            and previous.binding_generation == binding.generation
            and previous.repository == repository
        ):
            rows = self._disabled_rows(previous.rows, message)
        return self._local_status(
            binding,
            state,
            rows=rows,
            repository=repository,
            head=head,
            message=message,
        )

    @staticmethod
    def _disabled_rows(
        rows: tuple[SessionGitRow, ...],
        message: str | None,
    ) -> tuple[SessionGitRow, ...]:
        reason = message or "Git status refresh is required"
        return tuple(
            replace(
                row,
                stage_action=None,
                unstage_eligible=False,
                disabled_reason=row.disabled_reason or reason,
            )
            for row in rows
        )

    def _local_status(
        self,
        binding: SessionBinding,
        state: Literal["ready", "stale", "unavailable", "error"],
        *,
        rows: tuple[SessionGitRow, ...] = (),
        repository: RepositoryIdentity | None = None,
        head: HeadIdentity | None = None,
        message: str | None = None,
    ) -> SessionGitStatus:
        generation = self._owner.next_status_generation(binding)
        return SessionGitStatus(
            binding_generation=binding.generation,
            status_generation=0 if generation is None else generation,
            state=state,
            rows=rows,
            repository=repository,
            head=head,
            message=message,
        )

    def _git_executable_or_raise(self) -> str:
        if self._git_executable is None:
            raise RuntimeError("Git is not installed")
        return self._git_executable

    def _safe_root(self, binding: SessionBinding) -> Path | None:
        root = Path(binding.root_key)
        try:
            root_stat = root.stat(follow_symlinks=False)
            canonical = root.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or canonical != root
        ):
            return None
        return canonical

    async def _run_discovery(
        self,
        root: Path,
        arguments: Sequence[str],
    ) -> GitCommandResult | None:
        assert self._git_executable is not None
        try:
            result = await self._runner.run(
                (self._git_executable, *arguments),
                cwd=str(root),
                environment=build_git_environment(
                    self._environment,
                    stable_locale=True,
                ),
                timeout=self._discovery_timeout,
            )
        except OSError:
            return None
        if result.timed_out or result.termination_uncertain:
            return None
        return result

    async def _read_repository_paths(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path] | None:
        """Read the canonical worktree, Git-dir, and common-dir mapping."""
        path_commands = (
            ("rev-parse", "--path-format=absolute", "--show-toplevel"),
            ("rev-parse", "--absolute-git-dir"),
            ("rev-parse", "--path-format=absolute", "--git-common-dir"),
        )
        resolved_paths: list[Path] = []
        for arguments in path_commands:
            result = await self._run_discovery(root, arguments)
            if result is None or result.returncode != 0:
                return None
            resolved = _canonical_directory_from_git(result.stdout)
            if resolved is None:
                return None
            resolved_paths.append(resolved)
        return resolved_paths[0], resolved_paths[1], resolved_paths[2]

    async def _read_head(
        self,
        root: Path,
    ) -> HeadIdentity | _HeadReadFailure:
        symbolic = await self._run_head_probe(
            root,
            ("symbolic-ref", "--quiet", "HEAD"),
        )
        if isinstance(symbolic, _HeadReadFailure):
            return symbolic
        if symbolic.timed_out or symbolic.termination_uncertain:
            return self._head_command_failure(
                symbolic,
                "Git symbolic HEAD read failed",
            )

        branch: str | None
        if symbolic.returncode == 0:
            branch = _single_git_value(symbolic.stdout)
            if branch is None or not branch.startswith("refs/"):
                return _HeadReadFailure(
                    "error",
                    "Git symbolic HEAD output is malformed",
                )
        elif (
            symbolic.returncode == 1
            and symbolic.stdout == b""
            and symbolic.stderr == b""
        ):
            branch = None
        else:
            return self._head_command_failure(
                symbolic,
                "Git symbolic HEAD read failed",
            )

        revision = await self._run_head_probe(
            root,
            ("rev-parse", "--verify", "--quiet", "HEAD^{commit}"),
        )
        if isinstance(revision, _HeadReadFailure):
            return revision
        if revision.timed_out or revision.termination_uncertain:
            return self._head_command_failure(
                revision,
                "Git HEAD commit read failed",
            )
        if revision.returncode == 0:
            object_id = _ascii_object_id(revision.stdout)
            if object_id is None:
                return _HeadReadFailure(
                    "error",
                    "Git HEAD commit output is malformed",
                )
            if branch is None:
                return HeadIdentity.detached(object_id)
            return HeadIdentity.attached(branch, object_id)

        if branch is None:
            return self._head_command_failure(
                revision,
                "Detached Git HEAD does not resolve to a commit",
            )
        if not (
            revision.returncode == 1
            and revision.stdout == b""
            and revision.stderr == b""
        ):
            return self._head_command_failure(
                revision,
                "Git HEAD commit read failed",
            )

        reference = await self._run_head_probe(
            root,
            ("show-ref", "--exists", branch),
        )
        if isinstance(reference, _HeadReadFailure):
            return reference
        if reference.timed_out or reference.termination_uncertain:
            return self._head_command_failure(
                reference,
                "Git HEAD reference lookup failed",
            )
        if (
            reference.returncode == 2
            and reference.stdout == b""
            and reference.stderr == b""
        ):
            return HeadIdentity.unborn(branch)
        if (
            reference.returncode == 0
            and reference.stdout == b""
            and reference.stderr == b""
        ):
            return _HeadReadFailure(
                "error",
                "Git HEAD reference exists but does not resolve to a commit",
            )
        if reference.returncode != 129:
            return self._head_command_failure(
                reference,
                "Git HEAD reference lookup failed",
            )

        fallback = await self._run_head_probe(
            root,
            ("show-ref", "--verify", "--quiet", branch),
        )
        if isinstance(fallback, _HeadReadFailure):
            return fallback
        if fallback.timed_out or fallback.termination_uncertain:
            return self._head_command_failure(
                fallback,
                "Git HEAD reference lookup failed",
            )
        if (
            fallback.returncode == 1
            and fallback.stdout == b""
            and fallback.stderr == b""
        ):
            return HeadIdentity.unborn(branch)
        if (
            fallback.returncode == 0
            and fallback.stdout == b""
            and fallback.stderr == b""
        ):
            return _HeadReadFailure(
                "error",
                "Git HEAD reference exists but does not resolve to a commit",
            )
        return self._head_command_failure(
            fallback,
            "Git HEAD reference lookup failed",
        )

    async def _run_head_probe(
        self,
        root: Path,
        arguments: Sequence[str],
    ) -> GitCommandResult | _HeadReadFailure:
        assert self._git_executable is not None
        try:
            return await self._runner.run(
                (self._git_executable, *arguments),
                cwd=str(root),
                environment=build_git_environment(
                    self._environment,
                    stable_locale=True,
                ),
                timeout=self._discovery_timeout,
            )
        except OSError as error:
            diagnostic = sanitize_git_stderr(
                str(error).encode("utf-8", "replace")
            )
            message = "Git HEAD process could not start"
            if diagnostic:
                message = f"{message}: {diagnostic}"
            return _HeadReadFailure("unavailable", message)

    @staticmethod
    def _head_command_failure(
        result: GitCommandResult,
        fallback: str,
    ) -> _HeadReadFailure:
        kind: HeadReadFailureKind = (
            "unavailable"
            if result.timed_out or result.termination_uncertain
            else "error"
        )
        return _HeadReadFailure(
            kind,
            _command_failure_message(result, fallback),
        )

    def _repository_identity_matches(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> bool:
        root = self._safe_root(binding)
        if root is None:
            return False
        expected = (
            (
                repository.worktree_root,
                repository.worktree_identity,
            ),
            (repository.git_dir, repository.git_dir_identity),
            (
                repository.git_common_dir,
                repository.git_common_dir_identity,
            ),
        )
        resolved: list[Path] = []
        for path_text, identity in expected:
            path = Path(path_text)
            try:
                canonical = path.resolve(strict=True)
            except (OSError, RuntimeError):
                return False
            if (
                canonical != path
                or not canonical.is_dir()
                or _filesystem_identity(canonical) != identity
            ):
                return False
            resolved.append(canonical)
        worktree_root = resolved[0]
        try:
            root.relative_to(worktree_root)
        except ValueError:
            return False
        return True


def build_file_notes_session_owner() -> FileNotesSessionOwner:
    """Build one process owner with exactly one attached Git service."""
    owner = FileNotesSessionOwner()
    owner.attach_git_service(FileNotesGitService(owner))
    return owner


def _canonical_directory_from_git(payload: bytes) -> Path | None:
    value = _single_git_value(payload)
    if value is None or "\0" in value:
        return None
    try:
        path = Path(value).resolve(strict=True)
        path_stat = path.stat(follow_symlinks=False)
    except (OSError, RuntimeError):
        return None
    if not stat.S_ISDIR(path_stat.st_mode):
        return None
    return path


def _single_git_value(payload: bytes) -> str | None:
    if not payload.endswith(b"\n") or b"\0" in payload:
        return None
    value = payload[:-1]
    if not value:
        return None
    return os.fsdecode(value)


def _ascii_object_id(payload: bytes) -> str | None:
    if not payload.endswith(b"\n"):
        return None
    object_id = payload[:-1]
    if len(object_id) not in {40, 64}:
        return None
    try:
        value = object_id.decode("ascii")
    except UnicodeDecodeError:
        return None
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        return None
    return value.lower()


def _filesystem_identity(path: Path) -> FileSystemIdentity:
    path_stat = path.stat(follow_symlinks=False)
    return FileSystemIdentity(
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
    )


def _map_session_endpoint(
    notes_root: Path,
    worktree_root: Path,
    relative_path: str,
) -> tuple[str | None, str | None]:
    if (
        not relative_path
        or relative_path.startswith("/")
        or "\0" in relative_path
    ):
        return None, "Unsafe File Notes path"
    components = relative_path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        return None, "Unsafe File Notes path"
    if ".git" in components:
        return None, "nested_repository"

    candidate = notes_root.joinpath(*components)
    for depth in range(1, len(components)):
        parent = notes_root.joinpath(*components[:depth])
        try:
            parent_stat = parent.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError:
            return None, "Unsafe File Notes path"
        if (
            stat.S_ISLNK(parent_stat.st_mode)
            or not stat.S_ISDIR(parent_stat.st_mode)
        ):
            return None, "Unsafe File Notes path"
        git_boundary = parent / ".git"
        try:
            git_boundary.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError:
            return None, "Unsafe nested repository boundary"
        else:
            return None, "nested_repository"

    try:
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(notes_root)
        candidate.relative_to(worktree_root)
    except (OSError, RuntimeError, ValueError):
        return None, "File Notes path leaves the trusted worktree"

    try:
        endpoint_stat = candidate.stat(follow_symlinks=False)
    except FileNotFoundError:
        endpoint_stat = None
    except OSError:
        return None, "Unable to inspect File Notes path"
    if endpoint_stat is not None:
        if stat.S_ISLNK(endpoint_stat.st_mode):
            return None, "Symlink endpoints are unsupported"
        if stat.S_ISDIR(endpoint_stat.st_mode):
            return None, "Directory endpoints are unsupported"
        if not stat.S_ISREG(endpoint_stat.st_mode):
            return None, "Non-regular endpoints are unsupported"

    return candidate.relative_to(worktree_root).as_posix(), None


def _command_succeeded(result: GitCommandResult) -> bool:
    return (
        result.returncode == 0
        and not result.timed_out
        and not result.termination_uncertain
    )


def _stage_zero_index(
    entries: Sequence[IndexEntry],
) -> dict[str, IndexEntry]:
    """Return an exact path map only when each entry is unconflicted stage 0."""
    mapped: dict[str, IndexEntry] = {}
    for entry in entries:
        if entry.stage == 0:
            mapped[entry.path] = entry
    return mapped


def _command_failure_message(
    result: GitCommandResult,
    fallback: str,
) -> str:
    if result.termination_uncertain:
        return f"{fallback}: child termination is uncertain"
    if result.timed_out:
        return f"{fallback}: timed out"
    diagnostic = sanitize_git_stderr(result.stderr)
    if not diagnostic:
        return fallback
    return f"{fallback}: {diagnostic}"


def _index_mutation_failure_message(
    result: GitCommandResult,
    fallback: str,
    repository: RepositoryIdentity,
) -> str:
    """Report an existing Git index lock without deleting or retrying it."""
    if (
        result.returncode != 0
        and not result.timed_out
        and not result.termination_uncertain
    ):
        try:
            (Path(repository.git_dir) / "index.lock").lstat()
        except OSError:
            pass
        else:
            return "Git index busy; retry"
    return _command_failure_message(result, fallback)


class PorcelainV2ParseError(ValueError):
    """Raised when porcelain-v2 bytes are incomplete or malformed."""


class GitIndexParseError(ValueError):
    """Raised when a complete NUL-safe index read is ambiguous or malformed."""


class PorcelainPathOutsideSessionError(PorcelainV2ParseError):
    """Raised when Git reports a path outside the complete session whitelist."""


@dataclass(frozen=True, slots=True)
class PorcelainRecord:
    """One byte-safely decoded porcelain-v2 status record."""

    kind: PorcelainKind
    path: str | None
    index_status: str = "."
    worktree_status: str = "."
    submodule: str | None = None
    modes: tuple[str, ...] = ()
    object_ids: tuple[str, ...] = ()
    original_path: str | None = None
    score: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "object_ids", tuple(self.object_ids))


def parse_index_entries_z(payload: bytes) -> tuple[IndexEntry, ...]:
    """Parse `ls-files --stage -v -z` without losing filename bytes."""
    if not payload:
        return ()
    if not payload.endswith(b"\0"):
        raise GitIndexParseError("Git index payload is not NUL terminated")
    entries: list[IndexEntry] = []
    identities: set[tuple[str, int]] = set()
    for raw_entry in payload[:-1].split(b"\0"):
        if not raw_entry:
            raise GitIndexParseError("Git index payload contains an empty entry")
        if len(raw_entry) < 3 or raw_entry[1:2] != b" ":
            raise GitIndexParseError("Git index entry lacks a semantic tag")
        raw_tag = raw_entry[:1]
        if not (
            raw_tag.isalpha()
            or raw_tag == b"?"
        ):
            raise GitIndexParseError("Git index semantic tag is unsupported")
        try:
            metadata, raw_path = raw_entry[2:].split(b"\t", 1)
        except ValueError as error:
            raise GitIndexParseError(
                "Git index entry lacks a path boundary"
            ) from error
        fields = metadata.split(b" ")
        if len(fields) != 3:
            raise GitIndexParseError("Git index entry metadata is malformed")
        raw_mode, raw_object_id, raw_stage = fields
        if (
            len(raw_mode) != 6
            or any(byte not in b"01234567" for byte in raw_mode)
        ):
            raise GitIndexParseError("Git index mode is malformed")
        if (
            len(raw_object_id) not in {40, 64}
            or any(
                byte not in b"0123456789abcdefABCDEF"
                for byte in raw_object_id
            )
        ):
            raise GitIndexParseError("Git index object ID is malformed")
        if raw_stage not in {b"0", b"1", b"2", b"3"}:
            raise GitIndexParseError("Git index stage is malformed")
        if not raw_path:
            raise GitIndexParseError("Git index path is empty")
        path = os.fsdecode(raw_path)
        if (
            path.startswith("/")
            or "\0" in path
            or any(component in {"", ".", ".."} for component in path.split("/"))
        ):
            raise GitIndexParseError("Git index path is unsafe")
        stage = int(raw_stage)
        identity = path, stage
        if identity in identities:
            raise GitIndexParseError(
                "Git index contains a duplicate path and stage"
            )
        identities.add(identity)
        tag = raw_tag.decode("ascii")
        flags: list[str] = []
        if tag.upper() == "S":
            flags.append("skip-worktree")
        if tag.islower():
            flags.append("assume-unchanged")
        entries.append(
            IndexEntry(
                path=path,
                mode=raw_mode.decode("ascii"),
                object_id=raw_object_id.decode("ascii").lower(),
                stage=stage,
                semantic_flags=tuple(flags),
            )
        )
    return tuple(entries)


def parse_porcelain_v2_z(
    payload: bytes,
    *,
    allowed_paths: frozenset[str],
) -> tuple[PorcelainRecord, ...]:
    """Parse NUL-delimited porcelain-v2 bytes and fail closed on every path."""
    if not payload:
        return ()
    if not payload.endswith(b"\0"):
        raise PorcelainV2ParseError("Porcelain-v2 payload is not NUL terminated")

    fields = payload[:-1].split(b"\0")
    records: list[PorcelainRecord] = []
    position = 0
    while position < len(fields):
        raw_record = fields[position]
        position += 1
        if not raw_record:
            raise PorcelainV2ParseError("Porcelain-v2 payload contains an empty record")
        if raw_record.startswith(b"# "):
            continue
        marker = raw_record[:1]
        if marker == b"1":
            records.append(_parse_ordinary(raw_record, allowed_paths))
            continue
        if marker == b"2":
            if position >= len(fields):
                raise PorcelainV2ParseError(
                    "Porcelain-v2 rename record lacks its original path"
                )
            original_path = _decode_allowed_path(
                fields[position],
                allowed_paths,
            )
            position += 1
            records.append(
                _parse_rename(
                    raw_record,
                    original_path,
                    allowed_paths,
                )
            )
            continue
        if marker == b"u":
            records.append(_parse_unmerged(raw_record, allowed_paths))
            continue
        if marker == b"?":
            records.append(
                _parse_simple_record(
                    raw_record,
                    kind="untracked",
                    allowed_paths=allowed_paths,
                )
            )
            continue
        if marker == b"!":
            records.append(
                _parse_simple_record(
                    raw_record,
                    kind="ignored",
                    allowed_paths=allowed_paths,
                )
            )
            continue
        raise PorcelainV2ParseError("Unsupported porcelain-v2 record type")
    return tuple(records)


def compute_stage_closure(
    endpoints: Collection[str],
    index_entries: Mapping[str, IndexEntry],
) -> frozenset[str]:
    """Return literal endpoints plus tracked index ancestors/descendants."""
    closure = set(endpoints)
    for endpoint in endpoints:
        closure.update(
            path
            for path in index_entries
            if _paths_overlap(endpoint, path)
        )
    return frozenset(closure)


def compute_unstage_closure(
    baselines: Mapping[str, IndexBaseline],
    current_index_entries: Mapping[str, IndexEntry],
) -> frozenset[str]:
    """Return paths an exact baseline restoration may replace in the index."""
    closure = set(baselines)
    for path, baseline in baselines.items():
        if baseline.entry is None:
            continue
        closure.update(
            current_path
            for current_path in current_index_entries
            if _paths_overlap(path, current_path)
        )
    return frozenset(closure)


def stage_group_is_closed(
    group: SessionChangeGroup,
    index_entries: Mapping[str, IndexEntry],
) -> bool:
    """Return whether the Stage closure remains inside one session lineage."""
    return compute_stage_closure(
        group.endpoints,
        index_entries,
    ).issubset(group.endpoints)


def unstage_group_is_closed(
    group: SessionChangeGroup,
    baselines: Mapping[str, IndexBaseline],
    current_index_entries: Mapping[str, IndexEntry],
    ownership: StagingOwnership,
) -> bool:
    """Return whether replacement conflicts are exact same-group ownership."""
    closure = compute_unstage_closure(
        baselines,
        current_index_entries,
    )
    if (
        not closure.issubset(group.endpoints)
        or ownership.topology_signature != group.topology_signature
    ):
        return False
    return all(
        ownership.post_stage_entries.get(path) == entry
        and path in ownership.post_stage_entries
        for path, entry in current_index_entries.items()
        if path in closure
    )


def stage_pathspecs(
    group: SessionChangeGroup,
    status_records: Sequence[PorcelainRecord],
    *,
    groups: Sequence[SessionChangeGroup],
) -> tuple[bytes, ...]:
    """Encode only effective endpoints, omitting absent transient lineage."""
    if group.group_id in _ambiguous_group_ids(groups, status_records):
        return ()
    changed_paths = _effective_mutation_paths(group, status_records)
    return tuple(
        os.fsencode(path)
        for path in group.endpoints
        if path in changed_paths
    )


def ownership_signature_matches(
    ownership: StagingOwnership,
    *,
    repository: RepositoryIdentity,
    head: HeadIdentity,
    topology: SessionChangeTopology,
    current_index_entries: Mapping[str, IndexEntry],
) -> bool:
    """Compare exact repository, HEAD, topology, entry, and flag evidence."""
    if (
        ownership.repository != repository
        or ownership.head != head
        or ownership.topology_signature != topology
    ):
        return False
    return all(
        current_index_entries.get(path) == expected
        for path, expected in ownership.post_stage_entries.items()
    )


def _map_ownership_topology(
    ownership: StagingOwnership,
    original_group: SessionChangeGroup,
    repository_group: SessionChangeGroup,
) -> StagingOwnership | None:
    """Project owner topology into repository coordinates for row policy."""
    mapping = dict(zip(original_group.endpoints, repository_group.endpoints))
    try:
        endpoints = tuple(
            mapping[path] for path in ownership.approved_endpoint_topology
        )
        move_edges = tuple(
            (mapping[source], mapping[destination])
            for source, destination in ownership.approved_move_edges
        )
        current_path = mapping[ownership.approved_current_path]
    except KeyError:
        return None
    return replace(
        ownership,
        approved_endpoint_topology=endpoints,
        approved_move_edges=move_edges,
        approved_current_path=current_path,
    )


def classify_session_rows(
    groups: Sequence[SessionChangeGroup],
    status_records: Sequence[PorcelainRecord],
    index_entries: Mapping[str, IndexEntry] | Sequence[IndexEntry],
    ownership: Mapping[int, StagingOwnership],
) -> tuple[SessionGitRow, ...]:
    """Apply the frozen row/action policy to every coalesced session group."""
    flattened_entries = (
        tuple(index_entries.values())
        if isinstance(index_entries, Mapping)
        else tuple(index_entries)
    )
    entries_by_path: dict[str, IndexEntry] = {}
    for entry in flattened_entries:
        if entry.path not in entries_by_path or entry.stage == 0:
            entries_by_path[entry.path] = entry
    global_records = tuple(
        record for record in status_records if record.path is None
    )
    ambiguous_groups = _ambiguous_group_ids(groups, status_records)
    rows: list[SessionGitRow] = []
    for group in groups:
        records = global_records + tuple(
            record
            for record in status_records
            if _record_touches_group(record, group)
        )
        entries = tuple(
            entry
            for entry in flattened_entries
            if entry.path in group.endpoints
        )
        rows.append(
            _classify_group(
                group,
                records,
                entries,
                entries_by_path,
                ownership.get(group.group_id),
                group.group_id in ambiguous_groups,
            )
        )
    return tuple(rows)


def _parse_ordinary(
    raw_record: bytes,
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    fields = raw_record.split(b" ", 8)
    if len(fields) != 9 or fields[0] != b"1":
        raise PorcelainV2ParseError("Malformed ordinary porcelain-v2 record")
    index_status, worktree_status = _decode_xy(fields[1])
    return PorcelainRecord(
        kind="ordinary",
        path=_decode_allowed_path(fields[8], allowed_paths),
        index_status=index_status,
        worktree_status=worktree_status,
        submodule=_decode_ascii(fields[2], "submodule state"),
        modes=tuple(
            _decode_ascii(field, "file mode")
            for field in fields[3:6]
        ),
        object_ids=tuple(
            _decode_ascii(field, "object ID")
            for field in fields[6:8]
        ),
    )


def _parse_rename(
    raw_record: bytes,
    original_path: str,
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    fields = raw_record.split(b" ", 9)
    if len(fields) != 10 or fields[0] != b"2":
        raise PorcelainV2ParseError("Malformed rename porcelain-v2 record")
    index_status, worktree_status = _decode_xy(fields[1])
    return PorcelainRecord(
        kind="rename",
        path=_decode_allowed_path(fields[9], allowed_paths),
        index_status=index_status,
        worktree_status=worktree_status,
        submodule=_decode_ascii(fields[2], "submodule state"),
        modes=tuple(
            _decode_ascii(field, "file mode")
            for field in fields[3:6]
        ),
        object_ids=tuple(
            _decode_ascii(field, "object ID")
            for field in fields[6:8]
        ),
        original_path=original_path,
        score=_decode_ascii(fields[8], "rename score"),
    )


def _parse_unmerged(
    raw_record: bytes,
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    fields = raw_record.split(b" ", 10)
    if len(fields) != 11 or fields[0] != b"u":
        raise PorcelainV2ParseError("Malformed unmerged porcelain-v2 record")
    index_status, worktree_status = _decode_xy(fields[1])
    return PorcelainRecord(
        kind="unmerged",
        path=_decode_allowed_path(fields[10], allowed_paths),
        index_status=index_status,
        worktree_status=worktree_status,
        submodule=_decode_ascii(fields[2], "submodule state"),
        modes=tuple(
            _decode_ascii(field, "file mode")
            for field in fields[3:7]
        ),
        object_ids=tuple(
            _decode_ascii(field, "object ID")
            for field in fields[7:10]
        ),
    )


def _parse_simple_record(
    raw_record: bytes,
    *,
    kind: Literal["untracked", "ignored"],
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    if len(raw_record) < 3 or raw_record[1:2] != b" ":
        raise PorcelainV2ParseError(f"Malformed {kind} porcelain-v2 record")
    return PorcelainRecord(
        kind=kind,
        path=_decode_allowed_path(raw_record[2:], allowed_paths),
    )


def _decode_allowed_path(
    raw_path: bytes,
    allowed_paths: frozenset[str],
) -> str:
    if not raw_path:
        raise PorcelainV2ParseError("Porcelain-v2 record contains an empty path")
    path = os.fsdecode(raw_path)
    if path not in allowed_paths:
        raise PorcelainPathOutsideSessionError(
            f"Git reported a path outside the session whitelist: {path!r}"
        )
    return path


def _decode_xy(raw_xy: bytes) -> tuple[str, str]:
    if len(raw_xy) != 2:
        raise PorcelainV2ParseError("Porcelain-v2 XY status must contain two bytes")
    xy = _decode_ascii(raw_xy, "XY status")
    return xy[0], xy[1]


def _decode_ascii(value: bytes, label: str) -> str:
    try:
        return value.decode("ascii")
    except UnicodeDecodeError as error:
        raise PorcelainV2ParseError(
            f"Porcelain-v2 {label} is not ASCII"
        ) from error


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(f"{second}/")
        or second.startswith(f"{first}/")
    )


def _record_touches_group(
    record: PorcelainRecord,
    group: SessionChangeGroup,
) -> bool:
    return (
        record.path in group.endpoints
        or record.original_path in group.endpoints
    )


def _classify_group(
    group: SessionChangeGroup,
    records: Sequence[PorcelainRecord],
    entries: Sequence[IndexEntry],
    all_index_entries: Mapping[str, IndexEntry],
    owned: StagingOwnership | None,
    ambiguous_lineage: bool,
) -> SessionGitRow:
    if ambiguous_lineage:
        return SessionGitRow(
            group,
            "ambiguous_lineage",
            disabled_reason=(
                "Ambiguous session lineage: effective path belongs to multiple groups"
            ),
        )
    error = next((record for record in records if record.kind == "error"), None)
    if error is not None:
        return SessionGitRow(
            group,
            "error",
            disabled_reason=error.message or "Git status failed",
        )
    unavailable = next(
        (record for record in records if record.kind == "unavailable"),
        None,
    )
    if unavailable is not None:
        return SessionGitRow(
            group,
            "unavailable",
            disabled_reason=unavailable.message or "Git is unavailable",
        )
    if any(record.kind == "nested_repository" for record in records):
        return SessionGitRow(
            group,
            "nested_repository",
            disabled_reason="Nested repository unsupported",
        )
    if (
        any(record.kind == "unmerged" for record in records)
        or any(entry.stage != 0 for entry in entries)
    ):
        return SessionGitRow(
            group,
            "conflict",
            disabled_reason="Git conflict",
        )
    if any(record.kind == "ignored" for record in records):
        return SessionGitRow(
            group,
            "ignored",
            disabled_reason="Ignored by Git",
        )
    unsupported_reason = _unsupported_semantic_reason(entries)
    if unsupported_reason is not None:
        return SessionGitRow(
            group,
            "unsupported",
            disabled_reason=unsupported_reason,
        )
    if not stage_group_is_closed(group, all_index_entries):
        return SessionGitRow(
            group,
            "unsafe_closure",
            disabled_reason="Git mutation closure leaves this session lineage",
        )

    unstaged = any(
        record.kind == "untracked" or record.worktree_status != "."
        for record in records
    )
    staged = any(
        record.kind in {"ordinary", "rename"}
        and record.index_status != "."
        for record in records
    )

    if owned is not None:
        owned_entries_match = all(
            all_index_entries.get(path) == expected
            for path, expected in owned.post_stage_entries.items()
        )
        unowned_staged = any(
            path not in owned.post_stage_entries
            for record in records
            for path in _staged_record_paths(record)
        )
        if (
            owned_entries_match
            and not unowned_staged
            and owned.topology_signature != group.topology_signature
        ):
            return SessionGitRow(
                group,
                "owned_topology_changed",
                stage_action="stage_update",
                disabled_reason="Path lineage changed; Stage update required",
            )
        if (
            owned_entries_match
            and not unowned_staged
            and owned.topology_signature == group.topology_signature
        ):
            if unstaged:
                return SessionGitRow(
                    group,
                    "owned_newer_edits",
                    stage_action="stage_update",
                    unstage_eligible=True,
                )
            return SessionGitRow(
                group,
                "owned",
                unstage_eligible=True,
            )

    if staged and unstaged:
        return SessionGitRow(
            group,
            "external_partial",
            disabled_reason="External index state",
        )
    if staged:
        return SessionGitRow(
            group,
            "external_staged",
            disabled_reason="External index state",
        )
    if unstaged:
        return SessionGitRow(group, "unstaged", stage_action="stage")
    return SessionGitRow(group, "clean")


def _staged_record_paths(record: PorcelainRecord) -> tuple[str, ...]:
    if (
        record.kind not in {"ordinary", "rename"}
        or record.index_status == "."
    ):
        return ()
    return tuple(
        path
        for path in (record.path, record.original_path)
        if path is not None
    )


def _effective_mutation_paths(
    group: SessionChangeGroup,
    status_records: Sequence[PorcelainRecord],
) -> frozenset[str]:
    paths: set[str] = set()
    for record in status_records:
        if (
            not _record_touches_group(record, group)
            or (
                record.kind != "untracked"
                and record.worktree_status == "."
            )
        ):
            continue
        for path in (record.path, record.original_path):
            if path in group.endpoints:
                paths.add(path)
    return frozenset(paths)


def _ambiguous_group_ids(
    groups: Sequence[SessionChangeGroup],
    status_records: Sequence[PorcelainRecord],
) -> frozenset[int]:
    path_groups: dict[str, set[int]] = {}
    for group in groups:
        for path in _effective_mutation_paths(group, status_records):
            path_groups.setdefault(path, set()).add(group.group_id)
    return frozenset(
        group_id
        for group_ids in path_groups.values()
        if len(group_ids) > 1
        for group_id in group_ids
    )


def _unsupported_semantic_reason(
    entries: Sequence[IndexEntry],
) -> str | None:
    reasons = {
        flag
        for entry in entries
        for flag in entry.semantic_flags
    }
    if any(
        entry.object_id and set(entry.object_id) == {"0"}
        for entry in entries
    ):
        reasons.add("intent-to-add")
    if not reasons:
        return None
    return f"Unsupported Git index state: {', '.join(sorted(reasons))}"
