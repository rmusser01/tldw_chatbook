"""Generation-fenced protocol and parent controller for local batch STT.

The protocol portion of this module deliberately imports no provider, artifact,
ingestion, or UI implementation.  Spawned workers can import these frozen data
objects without loading a native speech runtime.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import shutil
import stat
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from tldw_chatbook.Utils.fd_protection import protect_file_descriptors
from .contracts import (
    BufferAudioSource,
    DeviceFailureOrigin,
    ExecutionDevice,
    FileAudioSource,
    TranscriptionFailureCode,
)
from .executor_process_tree import (
    ExecutorProcessTree,
    ProcessContainmentError,
    WorkerContainmentIdentity,
)

_MAX_RECOVERY_ACTIONS = 8
_MAX_RECOVERY_ACTION_LENGTH = 80
_CPU_FALLBACK_REQUESTED_DEVICE_OPTION = "_local_stt_cpu_fallback_requested_device"


def _require_generation_and_attempt(generation: int, attempt_id: str) -> None:
    if type(generation) is not int or generation <= 0:
        raise ValueError("generation must be a positive integer")
    if type(attempt_id) is not str or not attempt_id.strip():
        raise ValueError("attempt_id must be a non-empty string")


def _require_nonempty_text(field_name: str, value: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _canonical_dependency_refs(
    references: tuple[tuple[str, str, str], ...],
    *,
    field_name: str = "managed_dependency_refs",
) -> tuple[tuple[str, str, str], ...]:
    if type(references) is not tuple or any(
        type(reference) is not tuple
        or len(reference) != 3
        or any(
            type(component) is not str or not component.strip()
            for component in reference
        )
        for reference in references
    ):
        raise ValueError(f"{field_name} must contain three-string tuples")
    return tuple(sorted(set(references)))


class WorkerPhase(str, Enum):
    """Stable progress phases owned by the heavy worker."""

    PREPARING = "preparing"
    LOADING = "loading"
    TRANSCRIBING = "transcribing"
    POST_PROCESSING = "post-processing"


class ExecutorBusyError(RuntimeError):
    """Raised when the single active-request slot is occupied."""


class ExecutorUnavailableError(RuntimeError):
    """Raised when another worker generation cannot be started safely."""


class LocalSourceChangedError(RuntimeError):
    """Raised when an unmanaged local model no longer matches its snapshot."""


@dataclass(frozen=True, slots=True)
class LocalSourceSnapshot:
    """Private transient identity for unmanaged local model files."""

    token: str = field(repr=False)
    paths: tuple[Path, ...] = field(repr=False)
    identities: tuple[tuple[int, int, int, int], ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require_nonempty_text("token", self.token)
        if not self.paths or len(self.paths) != len(self.identities):
            raise ValueError(
                "snapshot paths and identities must be non-empty and aligned"
            )
        if any(not isinstance(path, Path) for path in self.paths):
            raise TypeError("snapshot paths must contain only Path values")
        if any(
            type(identity) is not tuple
            or len(identity) != 4
            or any(type(component) is not int for component in identity)
            for identity in self.identities
        ):
            raise TypeError("snapshot identities must be four-integer tuples")


def _local_file_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        metadata = path.lstat()
    except OSError:
        raise LocalSourceChangedError("Local STT model files changed") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise LocalSourceChangedError("Local STT model files changed")
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def snapshot_local_source(paths: tuple[Path, ...]) -> LocalSourceSnapshot:
    """Capture a path-private metadata identity for local model files.

    Args:
        paths: Required regular model files to snapshot.

    Returns:
        A path-private snapshot containing file identities and a digest token.

    Raises:
        ValueError: If ``paths`` is not a non-empty tuple.
        LocalSourceChangedError: If a path is missing, not a regular file, or
            cannot be inspected safely.
    """

    if type(paths) is not tuple or not paths:
        raise ValueError("paths must be a non-empty tuple")
    absolute_paths = tuple(path.absolute() for path in paths)
    identities = tuple(_local_file_identity(path) for path in absolute_paths)
    digest = hashlib.sha256()
    for index, identity in enumerate(identities):
        digest.update(f"{index}:{identity!r};".encode("ascii"))
    return LocalSourceSnapshot(
        token=digest.hexdigest(),
        paths=absolute_paths,
        identities=identities,
    )


def validate_local_source_snapshot(snapshot: LocalSourceSnapshot) -> None:
    """Verify that every required local model file is unchanged.

    Args:
        snapshot: Previously captured path-private local source snapshot.

    Returns:
        None.

    Raises:
        TypeError: If ``snapshot`` is not a ``LocalSourceSnapshot``.
        LocalSourceChangedError: If any snapshotted file changed or cannot be
            inspected safely.
    """

    if type(snapshot) is not LocalSourceSnapshot:
        raise TypeError("snapshot must be a LocalSourceSnapshot")
    current = tuple(_local_file_identity(path) for path in snapshot.paths)
    if current != snapshot.identities:
        raise LocalSourceChangedError("Local STT model files changed")


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Complete identity of the one model allowed to reside in a worker."""

    provider_id: str
    model_id: str
    root_revision: str | None
    closure_fingerprint: str | None
    precision: str
    device: ExecutionDevice
    local_snapshot_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_nonempty_text("provider_id", self.provider_id)
        _require_nonempty_text("model_id", self.model_id)
        _require_nonempty_text("precision", self.precision)
        if self.root_revision is not None:
            _require_nonempty_text("root_revision", self.root_revision)
        if self.closure_fingerprint is not None:
            _require_nonempty_text("closure_fingerprint", self.closure_fingerprint)
        if type(self.device) is not ExecutionDevice:
            raise TypeError("device must be an ExecutionDevice")
        if self.local_snapshot_token is not None:
            _require_nonempty_text("local_snapshot_token", self.local_snapshot_token)


@dataclass(frozen=True, slots=True)
class ExecutorRequest:
    """One heavy batch request sent to a specific executor generation."""

    generation: int
    attempt_id: str
    job_id: str | None
    source: FileAudioSource | BufferAudioSource = field(repr=False)
    identity: ModelIdentity
    options: dict[str, Any] = field(repr=False)
    segment_end_frames: tuple[int, ...] = ()
    local_source: LocalSourceSnapshot | None = field(default=None, repr=False)
    managed_store_root: Path | None = field(default=None, repr=False)
    managed_artifact_ref: tuple[str, str, str] | None = None
    managed_dependency_refs: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_generation_and_attempt(self.generation, self.attempt_id)
        if self.job_id is not None:
            _require_nonempty_text("job_id", self.job_id)
        if type(self.source) not in (FileAudioSource, BufferAudioSource):
            raise TypeError("source must be a FileAudioSource or BufferAudioSource")
        if self.segment_end_frames:
            if type(self.source) is not BufferAudioSource:
                raise ValueError("segment_end_frames require a buffer source")
            frame_bytes = self.source.channels * self.source.sample_width
            total_frames = len(self.source.audio) // frame_bytes
            if (
                any(type(end) is not int or end <= 0 for end in self.segment_end_frames)
                or any(
                    a >= b
                    for a, b in zip(
                        self.segment_end_frames,
                        self.segment_end_frames[1:],
                    )
                )
                or self.segment_end_frames[-1] != total_frames
            ):
                raise ValueError(
                    "segment_end_frames must increase to the final PCM frame"
                )
        if type(self.identity) is not ModelIdentity:
            raise TypeError("identity must be a ModelIdentity")
        if type(self.options) is not dict:
            raise TypeError("options must be a dict")
        if (
            self.local_source is not None
            and type(self.local_source) is not LocalSourceSnapshot
        ):
            raise TypeError("local_source must be a LocalSourceSnapshot")
        if self.managed_store_root is not None and not isinstance(
            self.managed_store_root, Path
        ):
            raise TypeError("managed_store_root must be a Path")
        if self.managed_artifact_ref is not None:
            if (
                type(self.managed_artifact_ref) is not tuple
                or len(self.managed_artifact_ref) != 3
                or any(
                    type(component) is not str or not component.strip()
                    for component in self.managed_artifact_ref
                )
            ):
                raise ValueError(
                    "managed_artifact_ref must contain three non-empty strings"
                )
        object.__setattr__(
            self,
            "managed_dependency_refs",
            _canonical_dependency_refs(self.managed_dependency_refs),
        )
        if self.managed_artifact_ref is not None and self.managed_dependency_refs:
            raise ValueError(
                "managed_artifact_ref and managed_dependency_refs are mutually exclusive"
            )
        has_managed_reference = self.managed_artifact_ref is not None or bool(
            self.managed_dependency_refs
        )
        if (self.managed_store_root is None) == has_managed_reference:
            raise ValueError(
                "managed_store_root requires a managed artifact or dependency reference"
            )
        if self.local_source is not None and self.managed_artifact_ref is not None:
            raise ValueError(
                "local_source and managed_artifact_ref are mutually exclusive"
            )


@dataclass(frozen=True, slots=True)
class ExecutorEvent:
    """One bounded worker-owned phase transition."""

    generation: int
    attempt_id: str
    phase: WorkerPhase

    def __post_init__(self) -> None:
        _require_generation_and_attempt(self.generation, self.attempt_id)
        if type(self.phase) is not WorkerPhase:
            raise TypeError("phase must be a WorkerPhase")


@dataclass(frozen=True, slots=True)
class ExecutorResident:
    """Worker confirmation that one exact model identity is resident."""

    generation: int
    attempt_id: str
    identity: ModelIdentity
    managed_lease_refs: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_generation_and_attempt(self.generation, self.attempt_id)
        if type(self.identity) is not ModelIdentity:
            raise TypeError("identity must be a ModelIdentity")
        object.__setattr__(
            self,
            "managed_lease_refs",
            _canonical_dependency_refs(
                self.managed_lease_refs,
                field_name="managed_lease_refs",
            ),
        )


@dataclass(frozen=True, slots=True)
class ExecutorResult:
    """Successful parsed-media payload from one worker attempt."""

    generation: int
    attempt_id: str
    payload: dict[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        _require_generation_and_attempt(self.generation, self.attempt_id)
        if type(self.payload) is not dict:
            raise TypeError("payload must be a dict")


@dataclass(frozen=True, slots=True)
class ExecutorFailure:
    """Bounded path-private failure from one worker attempt."""

    generation: int
    attempt_id: str
    code: TranscriptionFailureCode
    recovery_actions: tuple[str, ...] = ()
    failed_attempt: dict[str, Any] | None = field(default=None, repr=False)
    device_failure_origin: DeviceFailureOrigin | None = None
    failed_device: ExecutionDevice | None = None

    def __post_init__(self) -> None:
        _require_generation_and_attempt(self.generation, self.attempt_id)
        if type(self.code) is not TranscriptionFailureCode:
            raise TypeError("code must be a TranscriptionFailureCode")
        if type(self.recovery_actions) is not tuple:
            raise TypeError("recovery_actions must be a tuple")
        if len(self.recovery_actions) > _MAX_RECOVERY_ACTIONS:
            raise ValueError("too many recovery actions")
        if any(
            type(action) is not str
            or not action.strip()
            or len(action) > _MAX_RECOVERY_ACTION_LENGTH
            for action in self.recovery_actions
        ):
            raise ValueError("recovery actions must be bounded non-empty strings")
        if self.failed_attempt is not None and type(self.failed_attempt) is not dict:
            raise TypeError("failed_attempt must be a dict")
        if (
            self.device_failure_origin is not None
            and type(self.device_failure_origin) is not DeviceFailureOrigin
        ):
            raise TypeError("device_failure_origin must be a DeviceFailureOrigin")
        if (
            self.failed_device is not None
            and type(self.failed_device) is not ExecutionDevice
        ):
            raise TypeError("failed_device must be an ExecutionDevice")


class _AttemptTerminalGuard:
    """Accept exactly one matching terminal envelope for one active attempt."""

    __slots__ = ("_attempt_id", "_consumed", "_generation")

    def __init__(self, *, generation: int, attempt_id: str) -> None:
        _require_generation_and_attempt(generation, attempt_id)
        self._generation = generation
        self._attempt_id = attempt_id
        self._consumed = False

    def accept(self, envelope: ExecutorResult | ExecutorFailure) -> bool:
        """Consume and accept a matching terminal envelope once."""

        if type(envelope) not in {ExecutorResult, ExecutorFailure}:
            return False
        if (
            self._consumed
            or envelope.generation != self._generation
            or envelope.attempt_id != self._attempt_id
        ):
            return False
        self._consumed = True
        return True


@dataclass(slots=True)
class _ActiveCallbacks:
    on_event: Callable[[ExecutorEvent], None]
    on_result: Callable[[ExecutorResult], None]
    on_failure: Callable[[ExecutorFailure], None]


@dataclass(slots=True)
class _DetachedWorker:
    generation: int
    process: Any
    connection: Connection
    tree: ExecutorProcessTree
    scratch_path: Path


def _ignore_event(_event: ExecutorEvent) -> None:
    return


def _ignore_result(_result: ExecutorResult) -> None:
    return


def _ignore_failure(_failure: ExecutorFailure) -> None:
    return


class LocalSTTExecutor:
    """Own one generation-fenced spawn worker and no private request queue."""

    def __init__(
        self,
        *,
        worker_target: Callable[..., None] | None = None,
        completed_job_limit: int = 20,
        startup_timeout: float = 10.0,
        graceful_shutdown_timeout: float = 1.0,
        force_stop_timeout: float = 2.0,
    ) -> None:
        if type(completed_job_limit) is not int or completed_job_limit <= 0:
            raise ValueError("completed_job_limit must be a positive integer")
        for name, value in (
            ("startup_timeout", startup_timeout),
            ("graceful_shutdown_timeout", graceful_shutdown_timeout),
            ("force_stop_timeout", force_stop_timeout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")
        self._context = multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._completed_job_limit = completed_job_limit
        self._startup_timeout = float(startup_timeout)
        self._graceful_shutdown_timeout = float(graceful_shutdown_timeout)
        self._force_stop_timeout = float(force_stop_timeout)
        self._lock = threading.RLock()
        self._generation_counter = 0
        self._worker_generation: int | None = None
        self._process: Any | None = None
        self._connection: Connection | None = None
        self._cancellation_event: Any | None = None
        self._tree: ExecutorProcessTree | None = None
        self._scratch_path: Path | None = None
        self._reader_thread: threading.Thread | None = None
        self._retirement_thread: threading.Thread | None = None
        self._retirement_complete = threading.Event()
        self._retirement_complete.set()
        self._active_request: ExecutorRequest | None = None
        self._active_callbacks: _ActiveCallbacks | None = None
        self._terminal_guard: _AttemptTerminalGuard | None = None
        self._resident_identity: ModelIdentity | None = None
        self._resident_dependency_refs: tuple[tuple[str, str, str], ...] = ()
        self._resident_lease_refs: tuple[tuple[str, str, str], ...] = ()
        self._unhealthy_identity: ModelIdentity | None = None
        self._latest_phase: WorkerPhase | None = None
        self._completed_jobs = 0
        self._cpu_retry_used = False
        self._busy = False
        self._retiring = False
        self._unavailable = False
        self._closed = False

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation_counter

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def retiring(self) -> bool:
        with self._lock:
            return self._retiring

    @property
    def unavailable(self) -> bool:
        with self._lock:
            return self._unavailable or self._closed

    @property
    def resident_identity(self) -> ModelIdentity | None:
        with self._lock:
            return self._resident_identity

    @property
    def unhealthy_identity(self) -> ModelIdentity | None:
        with self._lock:
            return self._unhealthy_identity

    def submit(
        self,
        *,
        attempt_id: str,
        job_id: str | None,
        source: FileAudioSource | BufferAudioSource,
        identity: ModelIdentity,
        options: dict[str, Any],
        segment_end_frames: tuple[int, ...] = (),
        local_source: LocalSourceSnapshot | None = None,
        managed_store_root: Path | None = None,
        managed_artifact_ref: tuple[str, str, str] | None = None,
        managed_dependency_refs: tuple[tuple[str, str, str], ...] = (),
        on_event: Callable[[ExecutorEvent], None] = _ignore_event,
        on_result: Callable[[ExecutorResult], None] = _ignore_result,
        on_failure: Callable[[ExecutorFailure], None] = _ignore_failure,
        explicit_retry: bool = False,
    ) -> int:
        """Dispatch one request immediately or fail without queueing it."""

        canonical_dependency_refs = _canonical_dependency_refs(managed_dependency_refs)
        with self._lock:
            self._assert_dispatch_available(identity, explicit_retry=explicit_retry)
            if self._busy:
                raise ExecutorBusyError("Local STT executor already has active work")
            if self._retiring:
                raise ExecutorUnavailableError("Local STT executor is still stopping")
            identity_changed = self._resident_identity is not None and (
                self._resident_identity != identity
                or self._resident_dependency_refs != canonical_dependency_refs
            )
            lifetime_exhausted = self._completed_jobs >= self._completed_job_limit
            if self._process is not None and (identity_changed or lifetime_exhausted):
                if not self._retire_idle_worker_locked():
                    raise ExecutorUnavailableError(
                        "Previous local STT worker did not stop safely"
                    )
            if self._process is None:
                self._start_worker_locked()
            assert self._worker_generation is not None
            assert self._connection is not None
            assert self._cancellation_event is not None
            self._cancellation_event.clear()
            request_options = dict(options)
            request_options.pop(_CPU_FALLBACK_REQUESTED_DEVICE_OPTION, None)
            request = ExecutorRequest(
                generation=self._worker_generation,
                attempt_id=attempt_id,
                job_id=job_id,
                source=source,
                identity=identity,
                options=request_options,
                segment_end_frames=segment_end_frames,
                local_source=local_source,
                managed_store_root=managed_store_root,
                managed_artifact_ref=managed_artifact_ref,
                managed_dependency_refs=canonical_dependency_refs,
            )
            self._active_request = request
            self._active_callbacks = _ActiveCallbacks(
                on_event=on_event,
                on_result=on_result,
                on_failure=on_failure,
            )
            self._terminal_guard = _AttemptTerminalGuard(
                generation=request.generation,
                attempt_id=request.attempt_id,
            )
            self._latest_phase = None
            self._cpu_retry_used = False
            self._busy = True
            try:
                self._connection.send(request)
            except (BrokenPipeError, EOFError, OSError) as error:
                self._clear_active_locked()
                raise ExecutorUnavailableError(
                    "Local STT worker rejected the request"
                ) from error
            return request.generation

    def cancel(self, attempt_id: str) -> bool:
        """Request cooperative cancellation for the exact active attempt."""

        with self._lock:
            request = self._active_request
            if (
                not self._busy
                or request is None
                or request.attempt_id != attempt_id
                or self._cancellation_event is None
            ):
                return False
            self._cancellation_event.set()
            return True

    def force_stop(self, attempt_id: str) -> bool:
        """Detach one attempt, emit cancellation once, then kill off-thread."""

        callback: Callable[[ExecutorFailure], None] | None = None
        failure: ExecutorFailure | None = None
        with self._lock:
            request = self._active_request
            guard = self._terminal_guard
            callbacks = self._active_callbacks
            if (
                not self._busy
                or request is None
                or request.attempt_id != attempt_id
                or guard is None
                or callbacks is None
            ):
                return False
            failure = ExecutorFailure(
                request.generation,
                request.attempt_id,
                TranscriptionFailureCode.CANCELLED,
            )
            if not guard.accept(failure):
                return False
            callback = callbacks.on_failure
            self._clear_active_locked()
            detached = self._detach_worker_locked()
            if detached is None:
                return False
            self._retiring = True
            self._retirement_complete.clear()
            thread = threading.Thread(
                target=self._terminate_detached,
                args=(detached,),
                name=f"local-stt-retire-{request.generation}",
                daemon=True,
            )
            self._retirement_thread = thread
            thread.start()
        assert callback is not None and failure is not None
        self._deliver(callback, failure)
        return True

    def wait_for_retirement(self, timeout: float | None = None) -> bool:
        """Wait for a force-stopped generation to be proven dead and cleaned."""

        return self._retirement_complete.wait(timeout)

    def recycle_idle_managed_reference(
        self,
        reference: tuple[str, str, str],
    ) -> bool:
        """Retire an idle resident that leases one exact managed artifact.

        Args:
            reference: Canonical artifact ID, revision, and variant.

        Returns:
            True only when a matching idle generation is proven retired.
        """

        canonical = _canonical_dependency_refs(
            (reference,),
            field_name="reference",
        )[0]
        with self._lock:
            if (
                self._closed
                or self._unavailable
                or self._busy
                or self._retiring
                or self._resident_identity is None
                or canonical not in self._resident_lease_refs
            ):
                return False
            return self._retire_idle_worker_locked()

    def clear_unhealthy_identity(self, identity: ModelIdentity) -> bool:
        """Clear the one session-local unhealthy identity for explicit retry."""

        with self._lock:
            if self._unhealthy_identity != identity:
                return False
            self._unhealthy_identity = None
            return True

    def close(self) -> None:
        """Idempotently detach and stop the current generation."""

        detached: _DetachedWorker | None = None
        with self._lock:
            if not self._closed:
                self._closed = True
                self._clear_active_locked()
                detached = self._detach_worker_locked()
                self._resident_dependency_refs = ()
                self._resident_lease_refs = ()
            retirement = self._retirement_thread
        if detached is not None:
            self._terminate_detached(detached, update_state=False)
        if retirement is not None and retirement is not threading.current_thread():
            retirement.join(self._force_stop_timeout * 2 + 1.0)

    def _assert_dispatch_available(
        self, identity: ModelIdentity, *, explicit_retry: bool
    ) -> None:
        if self._closed or self._unavailable:
            raise ExecutorUnavailableError("Local STT executor is unavailable")
        if self._unhealthy_identity == identity:
            if not explicit_retry:
                raise ExecutorUnavailableError(
                    "Local STT model requires an explicit retry"
                )
            self._unhealthy_identity = None

    def _start_worker_locked(self) -> None:
        if self._unavailable or self._closed:
            raise ExecutorUnavailableError("Local STT executor is unavailable")
        target = self._worker_target
        if target is None:
            from .executor_worker import run_executor_worker

            target = run_executor_worker
        self._generation_counter += 1
        generation = self._generation_counter
        scratch_path = Path(tempfile.mkdtemp(prefix=f"tldw_stt_g{generation}_"))
        scratch_path.chmod(0o700)
        with protect_file_descriptors():
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            admission_event = self._context.Event()
            cancellation_event = self._context.Event()
            process = self._context.Process(
                target=target,
                args=(
                    child_connection,
                    admission_event,
                    cancellation_event,
                    generation,
                    str(scratch_path),
                ),
                name=f"local-stt-worker-{generation}",
            )
        tree: ExecutorProcessTree | None = None
        try:
            with protect_file_descriptors():
                process.start()
            child_connection.close()
            if not parent_connection.poll(self._startup_timeout):
                raise ProcessContainmentError("worker bootstrap timed out")
            bootstrap = parent_connection.recv()
            if (
                type(bootstrap) is not tuple
                or len(bootstrap) != 2
                or bootstrap[0] != "bootstrap"
                or type(bootstrap[1]) is not WorkerContainmentIdentity
            ):
                raise ProcessContainmentError("worker bootstrap was invalid")
            tree = ExecutorProcessTree(process, admission_event, bootstrap[1])
            tree.admit()
        except BaseException as error:
            child_connection.close()
            if tree is not None:
                stopped = tree.terminate_tree(
                    term_timeout=self._force_stop_timeout,
                    kill_timeout=self._force_stop_timeout,
                )
            else:
                if process.is_alive():
                    process.terminate()
                    process.join(self._force_stop_timeout)
                stopped = not process.is_alive()
            parent_connection.close()
            if stopped:
                shutil.rmtree(scratch_path, ignore_errors=True)
            else:
                self._unavailable = True
            raise ExecutorUnavailableError(
                "Local STT worker could not start safely"
            ) from error
        self._worker_generation = generation
        self._process = process
        self._connection = parent_connection
        self._cancellation_event = cancellation_event
        self._tree = tree
        self._scratch_path = scratch_path
        self._resident_identity = None
        self._resident_dependency_refs = ()
        self._resident_lease_refs = ()
        self._completed_jobs = 0
        reader = threading.Thread(
            target=self._reader_loop,
            args=(parent_connection, generation, process),
            name=f"local-stt-reader-{generation}",
            daemon=True,
        )
        self._reader_thread = reader
        reader.start()

    def _reader_loop(
        self,
        connection: Connection,
        generation: int,
        process: Any,
    ) -> None:
        while True:
            try:
                envelope = connection.recv()
            except (EOFError, OSError):
                self._handle_worker_exit(generation, process)
                return
            if type(envelope) is tuple:
                continue
            self._handle_worker_envelope(envelope)

    def _handle_worker_envelope(self, envelope: object) -> None:
        callback: Callable[[Any], None] | None = None
        delivered: object | None = None
        retry: tuple[ExecutorRequest, _ActiveCallbacks] | None = None
        with self._lock:
            request = self._active_request
            callbacks = self._active_callbacks
            if request is None or callbacks is None:
                return
            if type(envelope) is ExecutorEvent:
                if not self._matches_active(envelope):
                    return
                self._latest_phase = envelope.phase
                callback = callbacks.on_event
                delivered = envelope
            elif type(envelope) is ExecutorResident:
                if (
                    not self._matches_active(envelope)
                    or envelope.identity != request.identity
                ):
                    return
                self._resident_identity = envelope.identity
                self._resident_dependency_refs = request.managed_dependency_refs
                self._resident_lease_refs = envelope.managed_lease_refs
            elif type(envelope) in {ExecutorResult, ExecutorFailure}:
                if not self._matches_active(envelope):
                    return
                if type(envelope) is ExecutorFailure and self._should_retry_on_cpu(
                    request, envelope
                ):
                    retry = (request, callbacks)
                    self._cpu_retry_used = True
                else:
                    guard = self._terminal_guard
                    if guard is None or not guard.accept(envelope):
                        return
                    if type(envelope) is ExecutorResult:
                        self._completed_jobs += 1
                        callback = callbacks.on_result
                    else:
                        callback = callbacks.on_failure
                        if (
                            envelope.code is TranscriptionFailureCode.ENGINE_CRASHED
                            and self._latest_phase
                            in {WorkerPhase.LOADING, WorkerPhase.TRANSCRIBING}
                        ):
                            self._unhealthy_identity = request.identity
                    delivered = envelope
                    self._clear_active_locked()
            else:
                return
        if retry is not None:
            self._retry_on_cpu(*retry)
        elif callback is not None and delivered is not None:
            self._deliver(callback, delivered)

    def _retry_on_cpu(
        self, request: ExecutorRequest, callbacks: _ActiveCallbacks
    ) -> None:
        failure: ExecutorFailure | None = None
        deliver: Callable[[ExecutorFailure], None] | None = None
        with self._lock:
            if self._active_request != request:
                return
            self._clear_active_locked()
            if not self._retire_idle_worker_locked():
                failure = ExecutorFailure(
                    request.generation,
                    request.attempt_id,
                    TranscriptionFailureCode.ENGINE_CRASHED,
                    recovery_actions=("retry_faster_whisper",),
                )
                deliver = callbacks.on_failure
            else:
                retry_identity = replace(request.identity, device=ExecutionDevice.CPU)
                try:
                    self._start_worker_locked()
                    assert self._worker_generation is not None
                    assert self._connection is not None
                    assert self._cancellation_event is not None
                    self._cancellation_event.clear()
                    retry_request = replace(
                        request,
                        generation=self._worker_generation,
                        identity=retry_identity,
                        options={
                            **request.options,
                            _CPU_FALLBACK_REQUESTED_DEVICE_OPTION: (
                                request.identity.device.value
                            ),
                        },
                    )
                    self._active_request = retry_request
                    self._active_callbacks = callbacks
                    self._terminal_guard = _AttemptTerminalGuard(
                        generation=retry_request.generation,
                        attempt_id=retry_request.attempt_id,
                    )
                    self._latest_phase = None
                    self._cpu_retry_used = True
                    self._busy = True
                    self._connection.send(retry_request)
                except (
                    BrokenPipeError,
                    EOFError,
                    ExecutorUnavailableError,
                    OSError,
                ):
                    self._clear_active_locked()
                    if self._process is not None:
                        self._retire_idle_worker_locked()
                    failure = ExecutorFailure(
                        request.generation,
                        request.attempt_id,
                        TranscriptionFailureCode.ENGINE_CRASHED,
                        recovery_actions=("retry_faster_whisper",),
                    )
                    deliver = callbacks.on_failure
                else:
                    return
        if deliver is not None and failure is not None:
            self._deliver(deliver, failure)

    def _should_retry_on_cpu(
        self,
        request: ExecutorRequest,
        failure: ExecutorFailure,
    ) -> bool:
        if (
            self._cpu_retry_used
            or failure.device_failure_origin is None
            or failure.failed_device is None
        ):
            return False
        from .coordinator import device_retry_policy_for_failure

        policy = device_retry_policy_for_failure(
            requested_device=request.identity.device,
            failed_device=failure.failed_device,
            origin=failure.device_failure_origin,
            retry_device=ExecutionDevice.CPU,
            worker_will_recycle=True,
        )
        return policy.max_retries == 1

    def _matches_active(self, envelope: Any) -> bool:
        request = self._active_request
        return bool(
            request is not None
            and envelope.generation == request.generation
            and envelope.attempt_id == request.attempt_id
            and envelope.generation == self._worker_generation
        )

    def _handle_worker_exit(self, generation: int, process: Any) -> None:
        callback: Callable[[ExecutorFailure], None] | None = None
        failure: ExecutorFailure | None = None
        with self._lock:
            if generation != self._worker_generation or process is not self._process:
                return
            process.join(0.1)
            request = self._active_request
            callbacks = self._active_callbacks
            guard = self._terminal_guard
            if request is not None and callbacks is not None and guard is not None:
                failure = ExecutorFailure(
                    generation,
                    request.attempt_id,
                    TranscriptionFailureCode.ENGINE_CRASHED,
                    recovery_actions=("retry_faster_whisper",),
                )
                if guard.accept(failure):
                    callback = callbacks.on_failure
                    if self._latest_phase in {
                        WorkerPhase.LOADING,
                        WorkerPhase.TRANSCRIBING,
                    }:
                        self._unhealthy_identity = request.identity
            detached = self._detach_worker_locked()
            self._clear_active_locked()
        if detached is not None:
            proven = detached.tree.close()
            detached.connection.close()
            if proven:
                shutil.rmtree(detached.scratch_path, ignore_errors=True)
            else:
                with self._lock:
                    self._unavailable = True
        if callback is not None and failure is not None:
            self._deliver(callback, failure)

    def _retire_idle_worker_locked(self) -> bool:
        detached = self._detach_worker_locked()
        if detached is None:
            return True
        if detached.process.is_alive():
            try:
                detached.connection.send(("close", detached.generation))
            except (BrokenPipeError, EOFError, OSError):
                pass
            detached.process.join(self._graceful_shutdown_timeout)
        if detached.process.is_alive():
            proven = detached.tree.terminate_tree(
                term_timeout=self._force_stop_timeout,
                kill_timeout=self._force_stop_timeout,
            )
        else:
            proven = detached.tree.close()
        detached.connection.close()
        if proven:
            shutil.rmtree(detached.scratch_path, ignore_errors=True)
        else:
            self._unavailable = True
        return proven

    def _detach_worker_locked(self) -> _DetachedWorker | None:
        if (
            self._worker_generation is None
            or self._process is None
            or self._connection is None
            or self._tree is None
            or self._scratch_path is None
        ):
            return None
        detached = _DetachedWorker(
            generation=self._worker_generation,
            process=self._process,
            connection=self._connection,
            tree=self._tree,
            scratch_path=self._scratch_path,
        )
        self._worker_generation = None
        self._process = None
        self._connection = None
        self._cancellation_event = None
        self._tree = None
        self._scratch_path = None
        self._reader_thread = None
        self._resident_identity = None
        self._resident_dependency_refs = ()
        self._resident_lease_refs = ()
        self._completed_jobs = 0
        return detached

    def _terminate_detached(
        self, detached: _DetachedWorker, *, update_state: bool = True
    ) -> None:
        proven = False
        try:
            proven = detached.tree.terminate_tree(
                term_timeout=self._force_stop_timeout,
                kill_timeout=self._force_stop_timeout,
            )
            if proven:
                shutil.rmtree(detached.scratch_path, ignore_errors=True)
        except Exception:
            proven = False
        finally:
            detached.connection.close()
            with self._lock:
                if not proven:
                    self._unavailable = True
                if update_state:
                    self._retiring = False
                    self._retirement_complete.set()

    def _clear_active_locked(self) -> None:
        self._active_request = None
        self._active_callbacks = None
        self._terminal_guard = None
        self._latest_phase = None
        self._busy = False

    @staticmethod
    def _deliver(callback: Callable[[Any], None], envelope: object) -> None:
        try:
            callback(envelope)
        except Exception:
            return


__all__ = [
    "ExecutorBusyError",
    "ExecutorEvent",
    "ExecutorFailure",
    "ExecutorRequest",
    "ExecutorResident",
    "ExecutorResult",
    "ExecutorUnavailableError",
    "LocalSourceChangedError",
    "LocalSourceSnapshot",
    "LocalSTTExecutor",
    "ModelIdentity",
    "WorkerPhase",
    "snapshot_local_source",
    "validate_local_source_snapshot",
]
