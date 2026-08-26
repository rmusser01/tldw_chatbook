from __future__ import annotations

import asyncio
import codecs
import logging
import os
import re
import time
import unicodedata
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Set as AbstractSet
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlsplit

from rich.markup import escape as escape_rich_markup

from tldw_chatbook.TTS._async_lifecycle import (
    current_shutdown_deadline,
    join_retained_task,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSOperationCode,
    TTSOperationError,
    TTSProviderReconfiguringError,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_managed_config import (
    AudioCppManagedLaunchConfig,
    build_audio_cpp_child_environment,
    collect_provider_credential_environment_names,
    validate_audio_cpp_managed_launch,
)

_DiagnosticStream = Literal["stdout", "stderr"]
_InternalDiagnosticPhase = Literal[
    "launch_revalidation",
    "generation_cleanup",
    "artifact_cleanup",
]
_MAX_DIAGNOSTIC_LINES = 200
_MAX_DIAGNOSTIC_BYTES = 65_536
_MAX_DIAGNOSTIC_LINE_BYTES = 4_096
_ANSI_ESCAPE_RE = re.compile(
    r"(?:\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x9b[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-_])"
)
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_ -]?key|token|secret|password|credential|authorization|auth)"
    r"\b\s*[:=]\s*)"
    r'(?:"[^"]*"|\'[^\']*\'|bearer\s+[^\s,;]+|[^\s,;]+)'
)
_BEARER_SECRET_RE = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
_REDACTION = "<redacted>"
_AUDIO_CPP_SUPERVISOR_OWNER_TOKEN = object()
_ASYNCIO_SPAWN_LOG_SUPPRESSION_ACTIVE: ContextVar[bool] = ContextVar(
    "audio_cpp_asyncio_spawn_log_suppression_active",
    default=False,
)


class _AsyncioSpawnPrivacyFilter(logging.Filter):
    """Suppress asyncio records only while it renders the private launch argv."""

    def filter(self, record: logging.LogRecord) -> bool:
        del record
        return not _ASYNCIO_SPAWN_LOG_SUPPRESSION_ACTIVE.get()


_ASYNCIO_SPAWN_PRIVACY_FILTER = _AsyncioSpawnPrivacyFilter()


@dataclass(frozen=True, slots=True)
class AudioCppDiagnosticLine:
    """One sanitized display line captured from an owned audio.cpp child."""

    stream: _DiagnosticStream
    text: str


@dataclass(slots=True)
class _DiagnosticStreamState:
    decoder: Any
    pending: str = ""
    emitted_at_boundary: bool = False


def _new_stream_state() -> _DiagnosticStreamState:
    return _DiagnosticStreamState(
        decoder=codecs.getincrementaldecoder("utf-8")(errors="replace")
    )


def _utf8_prefix(value: str, byte_limit: int) -> str:
    if not value or byte_limit <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def _remove_unsafe_controls(value: str) -> str:
    return "".join(
        character for character in value if unicodedata.category(character)[0] != "C"
    )


class _AudioCppDiagnosticRing:
    """Incrementally sanitize and retain a bounded child-output snapshot."""

    def __init__(self, *, home_directory: Path | None = None) -> None:
        if home_directory is None:
            try:
                self._home_directory = str(Path.home())
            except (OSError, RuntimeError, ValueError):
                self._home_directory = ""
        else:
            self._home_directory = str(home_directory)
        self._entries: deque[tuple[AudioCppDiagnosticLine, int]] = deque()
        self._retained_bytes = 0
        self._dropped_lines = 0
        self._streams = self._new_streams()
        self._content_suppressed = False

    @staticmethod
    def _new_streams() -> dict[_DiagnosticStream, _DiagnosticStreamState]:
        return {"stdout": _new_stream_state(), "stderr": _new_stream_state()}

    def feed(self, stream: _DiagnosticStream, chunk: bytes) -> None:
        """Consume one raw output chunk without retaining it.

        Args:
            stream: Child pipe that produced the chunk.
            chunk: Raw bytes read from that pipe.
        """
        if self._content_suppressed:
            return
        state = self._streams[stream]
        decoded = state.decoder.decode(chunk, final=False)
        self._consume_decoded(stream, state, decoded)

    def finish(self, stream: _DiagnosticStream) -> None:
        """Flush a child pipe's decoder and its final unterminated line."""
        if self._content_suppressed:
            return
        state = self._streams[stream]
        decoded = state.decoder.decode(b"", final=True)
        self._consume_decoded(stream, state, decoded)
        if state.pending:
            self._retain(stream, state.pending)
            state.pending = ""
        state.emitted_at_boundary = False
        state.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def snapshot(self) -> tuple[tuple[AudioCppDiagnosticLine, ...], int]:
        """Return immutable retained lines and the eviction count."""
        return tuple(line for line, _size in self._entries), self._dropped_lines

    def clear(self) -> None:
        """Clear all output and decoder state at a generation boundary."""
        self._entries.clear()
        self._retained_bytes = 0
        self._dropped_lines = 0
        self._streams = self._new_streams()
        self._content_suppressed = False

    def suppress_content(self) -> None:
        """Discard all child content for the remainder of this generation."""
        self._entries.clear()
        self._retained_bytes = 0
        self._dropped_lines = 0
        self._streams = self._new_streams()
        self._content_suppressed = True

    def _consume_decoded(
        self,
        stream: _DiagnosticStream,
        state: _DiagnosticStreamState,
        decoded: str,
    ) -> None:
        remaining = decoded
        while remaining:
            newline = remaining.find("\n")
            if newline < 0:
                self._append_fragment(stream, state, remaining)
                return

            fragment = remaining[:newline]
            remaining = remaining[newline + 1 :]
            if fragment.endswith("\r"):
                fragment = fragment[:-1]
            self._append_fragment(stream, state, fragment)
            if state.pending or not state.emitted_at_boundary:
                self._retain(stream, state.pending)
            state.pending = ""
            state.emitted_at_boundary = False

    def _append_fragment(
        self,
        stream: _DiagnosticStream,
        state: _DiagnosticStreamState,
        fragment: str,
    ) -> None:
        remaining = fragment
        while remaining:
            capacity = _MAX_DIAGNOSTIC_LINE_BYTES - len(state.pending.encode("utf-8"))
            prefix = _utf8_prefix(remaining, capacity)
            if not prefix:
                self._retain(stream, state.pending)
                state.pending = ""
                state.emitted_at_boundary = True
                continue

            state.pending += prefix
            remaining = remaining[len(prefix) :]
            if len(state.pending.encode("utf-8")) >= _MAX_DIAGNOSTIC_LINE_BYTES:
                self._retain(stream, state.pending)
                state.pending = ""
                state.emitted_at_boundary = True
            else:
                state.emitted_at_boundary = False

    def _retain(self, stream: _DiagnosticStream, text: str) -> None:
        sanitized = self._sanitize(text)
        sanitized = _utf8_prefix(sanitized, _MAX_DIAGNOSTIC_LINE_BYTES)
        size = len(sanitized.encode("utf-8"))
        self._entries.append((AudioCppDiagnosticLine(stream, sanitized), size))
        self._retained_bytes += size

        while (
            len(self._entries) > _MAX_DIAGNOSTIC_LINES
            or self._retained_bytes > _MAX_DIAGNOSTIC_BYTES
        ):
            _line, evicted_size = self._entries.popleft()
            self._retained_bytes -= evicted_size
            self._dropped_lines += 1

    def _sanitize(self, text: str) -> str:
        value = _ANSI_ESCAPE_RE.sub("", text)
        value = _remove_unsafe_controls(value)
        value = _ASSIGNMENT_SECRET_RE.sub(rf"\1{_REDACTION}", value)
        value = _BEARER_SECRET_RE.sub(rf"\1{_REDACTION}", value)
        if self._home_directory and self._home_directory != os.sep:
            value = value.replace(self._home_directory, "~")
        return escape_rich_markup(value)


AudioCppProcessState = Literal[
    "stopped",
    "starting",
    "running",
    "unhealthy",
    "draining",
    "stopping",
    "unavailable",
]
AudioCppArtifactPrivacyPosture = Literal[
    "not_applicable",
    "unverified",
    "posix_owner_only",
    "windows_account_protected",
]
AudioCppTTSCapability = Literal["available", "not_configured", "unknown"]
AudioCppContractProbe = Callable[[], Awaitable[AudioCppTTSCapability]]
AudioCppHealthProbe = Callable[[], Awaitable[bool]]
AudioCppGenerationInvalidation = Callable[[], None]
AudioCppGenerationCleanup = Callable[[], Awaitable[None]]
_PortPreflightResult = Literal["available", "occupied", "ambiguous"]
_PortPreflight = Callable[[int, float], Awaitable[_PortPreflightResult]]
_Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class AudioCppReadyEndpoint:
    """One ready endpoint bound to an exact managed process generation."""

    base_url: str
    process_generation: int
    observation_version: int


@dataclass(frozen=True, slots=True)
class AudioCppProcessFailure:
    """Safe latest managed-process failure for UI and service projection."""

    process_generation: int | None
    code: TTSOperationCode
    message: str
    retryable: bool
    recovery_action: str | None


@dataclass(frozen=True, slots=True)
class AudioCppProcessAdmissionSnapshot:
    """Generation fence used while coordinating staged configuration."""

    lifecycle_epoch: int
    process_generation: int
    state: AudioCppProcessState
    stage_application_eligible: bool


@dataclass(frozen=True, slots=True)
class AudioCppGenerationHooks:
    """Adapter-owned HTTP hooks for exactly one process generation."""

    contract_probe: AudioCppContractProbe
    health_probe: AudioCppHealthProbe
    cleanup: AudioCppGenerationCleanup
    invalidate: AudioCppGenerationInvalidation | None = None


@dataclass(frozen=True, slots=True)
class AudioCppProcessSnapshot:
    """Immutable observable state of the one managed process slot."""

    state: AudioCppProcessState
    process_generation: int
    observation_version: int
    endpoint: str | None
    tts_capability: AudioCppTTSCapability
    consecutive_health_failures: int
    last_failure: AudioCppProcessFailure | None
    diagnostics: tuple[AudioCppDiagnosticLine, ...]
    dropped_diagnostic_lines: int
    generated_artifact_privacy_posture: AudioCppArtifactPrivacyPosture = (
        "not_applicable"
    )


@dataclass(frozen=True, slots=True)
class _OwnedAudioCppProcess:
    process: Any
    close_parent_pipes: Callable[[], None]
    close_native_transport: Callable[[], None] = lambda: None


_ProcessLauncher = Callable[
    [AudioCppManagedLaunchConfig, dict[str, str]], Awaitable[_OwnedAudioCppProcess]
]
_GenerationHooksFactory = Callable[[int], Awaitable[AudioCppGenerationHooks]]
_StepResult = TypeVar("_StepResult")


class _AudioCppGenerationChanged(RuntimeError):
    """Private signal requesting a fresh service preparation pass."""


class _ProcessExitedDuringStartup(RuntimeError):
    """Private marker for a child exit racing a bounded startup step."""


@dataclass(frozen=True, slots=True)
class _FailureSpec:
    code: TTSOperationCode
    message: str
    retryable: bool
    recovery_action: str | None


_CONFIGURATION_FAILURE = _FailureSpec(
    "configuration_invalid",
    "Managed audio.cpp configuration is invalid",
    False,
    "open_settings",
)
_PORT_FAILURE = _FailureSpec(
    "port_in_use",
    "The configured audio.cpp port is unavailable",
    True,
    "open_settings",
)
_SPAWN_FAILURE = _FailureSpec(
    "process_spawn_failed",
    "The audio.cpp server could not be started",
    True,
    "retry",
)
_STARTUP_TIMEOUT_FAILURE = _FailureSpec(
    "process_startup_timeout",
    "The audio.cpp server did not become ready in time",
    True,
    "open_diagnostics",
)
_PROCESS_EXITED_FAILURE = _FailureSpec(
    "process_exited",
    "The audio.cpp server exited unexpectedly",
    True,
    "open_diagnostics",
)
_CONTRACT_FAILURE = _FailureSpec(
    "contract_incompatible",
    "The audio.cpp server contract is incompatible",
    False,
    "open_settings",
)
_RUNTIME_UNHEALTHY_FAILURE = _FailureSpec(
    "runtime_unhealthy",
    "The managed audio.cpp server is unhealthy",
    True,
    "restart_managed",
)
_CLEANUP_FAILURE = _FailureSpec(
    "cleanup_failed",
    "Managed audio.cpp cleanup did not complete",
    False,
    "open_diagnostics",
)
_OPERATION_ID = "audio_cpp_managed"


@dataclass(slots=True)
class _ProcessGeneration:
    generation: int
    epoch: int
    launch: AudioCppManagedLaunchConfig
    owned: _OwnedAudioCppProcess
    hooks_ready: asyncio.Event
    process_exited: asyncio.Event
    stdout_drain: asyncio.Task[None] | None = None
    stderr_drain: asyncio.Task[None] | None = None
    exit_monitor: asyncio.Task[None] | None = None
    health_scheduler: asyncio.Task[None] | None = None
    health_probe: asyncio.Task[bool] | None = None
    output_failure_cleanup: asyncio.Task[None] | None = None
    hooks: AudioCppGenerationHooks | None = None
    expected_exit: bool = False
    terminal_state: AudioCppProcessState = "unavailable"
    failure: AudioCppProcessFailure | None = None
    cleanup_deadline: float | None = None
    cleanup_deadline_changed: asyncio.Event = field(default_factory=asyncio.Event)
    invalidation_called: bool = False
    hooks_cleanup_settled: bool = False
    hooks_cleanup_succeeded: bool = True
    artifact_cleanup_succeeded: bool = False
    cleanup_failure: AudioCppProcessFailure | None = None
    parent_pipes_closed: bool = False
    native_transport_closed: bool = False


def _failure_for(
    spec: _FailureSpec,
    process_generation: int | None,
) -> AudioCppProcessFailure:
    return AudioCppProcessFailure(
        process_generation=process_generation,
        code=spec.code,
        message=spec.message,
        retryable=spec.retryable,
        recovery_action=spec.recovery_action,
    )


def _operation_error(failure: AudioCppProcessFailure) -> TTSOperationError:
    return TTSOperationError(
        code=failure.code,
        message=failure.message,
        retryable=failure.retryable,
        operation_id=_OPERATION_ID,
        recovery_action=failure.recovery_action,
    )


def _remaining(deadline: float, monotonic: Callable[[], float]) -> float:
    return max(0.0, deadline - monotonic())


def _close_process_parent_pipes(process: Any) -> None:
    for reader_name in ("stdout", "stderr"):
        reader = getattr(process, reader_name, None)
        transport = getattr(reader, "_transport", None)
        close = getattr(transport, "close", None)
        if callable(close):
            close()


def _process_native_transport_closer(process: Any) -> Callable[[], None]:
    closed = False

    def close_once() -> None:
        nonlocal closed
        if closed:
            return
        transport = getattr(process, "_transport", None)
        close = getattr(transport, "close", None)
        if callable(close):
            close()
        closed = True

    return close_once


async def _default_process_launcher(
    launch: AudioCppManagedLaunchConfig,
    child_environment: dict[str, str],
) -> _OwnedAudioCppProcess:
    asyncio_logger = logging.getLogger("asyncio")
    if _ASYNCIO_SPAWN_PRIVACY_FILTER not in asyncio_logger.filters:
        asyncio_logger.addFilter(_ASYNCIO_SPAWN_PRIVACY_FILTER)
    suppression_token = _ASYNCIO_SPAWN_LOG_SUPPRESSION_ACTIVE.set(True)
    try:
        process = await asyncio.create_subprocess_exec(
            str(launch.binary_path),
            "--config",
            str(launch.server_json_path),
            cwd=str(launch.working_directory),
            env=child_environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    finally:
        _ASYNCIO_SPAWN_LOG_SUPPRESSION_ACTIVE.reset(suppression_token)
    return _OwnedAudioCppProcess(
        process=process,
        close_parent_pipes=lambda: _close_process_parent_pipes(process),
        close_native_transport=_process_native_transport_closer(process),
    )


async def _settle_process_launcher(
    launcher: Awaitable[_OwnedAudioCppProcess],
    *,
    timeout: float,
) -> tuple[
    asyncio.CancelledError | None,
    bool,
    _OwnedAudioCppProcess | None,
    BaseException | None,
]:
    """Settle one retained spawn task before exposing cancellation or timeout."""

    task: asyncio.Future[_OwnedAudioCppProcess] = asyncio.ensure_future(launcher)
    cancellation: asyncio.CancelledError | None = None
    timed_out = False
    waiter = asyncio.current_task()
    cancellation_requests = waiter.cancelling() if waiter is not None else 0
    try:
        done, _pending = await asyncio.wait({task}, timeout=max(0.0, timeout))
        timed_out = task not in done
    except asyncio.CancelledError as error:
        cancellation = error
    if (timed_out or cancellation is not None) and not task.done():
        task.cancel()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            next_requests = waiter.cancelling() if waiter is not None else 0
            if next_requests > cancellation_requests:
                cancellation = cancellation or error
                cancellation_requests = next_requests
        except BaseException:
            if not task.done():
                raise
            break
    try:
        return cancellation, timed_out, task.result(), None
    except BaseException as error:
        return cancellation, timed_out, None, error


async def _default_port_preflight(port: int, timeout: float) -> _PortPreflightResult:
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=timeout,
        )
    except ConnectionRefusedError:
        return "available"
    except (OSError, asyncio.TimeoutError):
        return "ambiguous"

    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return "occupied"


class AudioCppSupervisor:
    """Own and supervise at most one lazily launched audio.cpp process."""

    def __init__(
        self,
        *,
        source_environment: Mapping[str, Any] | None = None,
        provider_credential_names: AbstractSet[str] | None = None,
        process_launcher: _ProcessLauncher = _default_process_launcher,
        port_preflight: _PortPreflight = _default_port_preflight,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: _Sleep = asyncio.sleep,
    ) -> None:
        self._application_owner_token = _AUDIO_CPP_SUPERVISOR_OWNER_TOKEN
        credentials = collect_provider_credential_environment_names({})
        if provider_credential_names is not None:
            credentials = credentials.union(provider_credential_names)
        self._child_environment = build_audio_cpp_child_environment(
            os.environ if source_environment is None else source_environment,
            provider_credential_names=credentials,
        )
        self._process_launcher = process_launcher
        self._port_preflight = port_preflight
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._state: AudioCppProcessState = "stopped"
        self._process_generation = 0
        self._observation_version = 0
        self._lifecycle_epoch = 0
        self._endpoint: str | None = None
        self._tts_capability: AudioCppTTSCapability = "unknown"
        self._consecutive_health_failures = 0
        self._last_failure: AudioCppProcessFailure | None = None
        self._blocked_cleanup_failure: AudioCppProcessFailure | None = None
        self._diagnostics = _AudioCppDiagnosticRing()
        self._generation: _ProcessGeneration | None = None
        self._pre_spawn_launch: AudioCppManagedLaunchConfig | None = None
        self._startup_task: asyncio.Task[AudioCppReadyEndpoint] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._shutdown_started = False
        self._closed = False

    def snapshot(self) -> AudioCppProcessSnapshot:
        """Return the current immutable process observation."""
        diagnostics, dropped = self._diagnostics.snapshot()
        launch = (
            self._generation.launch
            if self._generation is not None
            else self._pre_spawn_launch
        )
        artifact = launch.generated_artifact if launch is not None else None
        posture = getattr(artifact, "privacy_posture", "not_applicable")
        if posture not in {
            "not_applicable",
            "unverified",
            "posix_owner_only",
            "windows_account_protected",
        }:
            posture = "unverified"
        return AudioCppProcessSnapshot(
            state=self._state,
            process_generation=self._process_generation,
            observation_version=self._observation_version,
            endpoint=self._endpoint,
            tts_capability=self._tts_capability,
            consecutive_health_failures=self._consecutive_health_failures,
            last_failure=self._last_failure,
            diagnostics=diagnostics,
            dropped_diagnostic_lines=dropped,
            generated_artifact_privacy_posture=cast(
                AudioCppArtifactPrivacyPosture,
                posture,
            ),
        )

    def _record_internal_diagnostic(
        self,
        phase: _InternalDiagnosticPhase,
        error: Exception,
    ) -> None:
        if isinstance(error, AssertionError):
            category = "assertion_error"
        elif isinstance(error, RuntimeError):
            category = "runtime_error"
        else:
            category = "unexpected_exception"
        self._diagnostics.feed(
            "stderr",
            (
                "Chatbook internal supervisor failure "
                f"(phase={phase}, category={category}).\n"
            ).encode("ascii"),
        )

    def admission_snapshot(self) -> AudioCppProcessAdmissionSnapshot:
        """Return the generation fence and staged-application eligibility."""
        eligible = (
            self._state in ("stopped", "unavailable")
            and self._generation is None
            and self._startup_task is None
            and self._stop_task is None
            and self._close_task is None
            and not self._shutdown_started
            and not self._closed
            and self._blocked_cleanup_failure is None
        )
        return AudioCppProcessAdmissionSnapshot(
            lifecycle_epoch=self._lifecycle_epoch,
            process_generation=self._process_generation,
            state=self._state,
            stage_application_eligible=eligible,
        )

    def suppress_clone_diagnostics(self, process_generation: int) -> bool:
        """Suppress child content for one exact live clone-capable generation."""
        record = self._generation
        if (
            type(process_generation) is not int
            or record is None
            or record.generation != process_generation
            or record.owned.process.returncode is not None
            or self._state not in {"running", "draining"}
        ):
            return False
        self._diagnostics.suppress_content()
        return True

    async def ensure_running(
        self,
        launch: AudioCppManagedLaunchConfig,
        *,
        generation_hooks_factory: _GenerationHooksFactory,
        require_existing: AudioCppProcessAdmissionSnapshot | None = None,
    ) -> AudioCppReadyEndpoint:
        """Return the current ready generation or share one lazy startup."""
        cleanup_monitor: asyncio.Task[None] | None = None
        async with self._lock:
            if self._closed or self._shutdown_started:
                raise _operation_error(
                    _failure_for(_PROCESS_EXITED_FAILURE, self._process_generation)
                )
            if self._blocked_cleanup_failure is not None:
                raise _operation_error(self._blocked_cleanup_failure)
            active_generation = self._generation
            if (
                active_generation is not None
                and self._state in ("starting", "running", "unhealthy", "draining")
                and not active_generation.expected_exit
                and (
                    active_generation.owned.process.returncode is not None
                    or active_generation.process_exited.is_set()
                )
            ):
                failure = self._publish_process_exited_locked(active_generation)
                raise _operation_error(failure)
            if require_existing is not None and not self._matches_admission_locked(
                require_existing
            ):
                raise _AudioCppGenerationChanged
            if self._state == "unavailable" and self._generation is not None:
                cleanup_monitor = self._generation.exit_monitor
                assert cleanup_monitor is not None
                generation = None
                startup_task = None
            elif self._state == "draining" and require_existing is not None:
                return self._ready_endpoint_locked()
            elif self._state in ("draining", "stopping"):
                raise TTSProviderReconfiguringError(
                    "The audio.cpp provider is reconfiguring"
                )
            elif self._state == "running" and self._generation is not None:
                return self._ready_endpoint_locked()
            elif self._state == "unhealthy" and self._generation is not None:
                generation = self._generation
                startup_task = None
            elif self._startup_task is not None:
                generation = None
                startup_task = self._startup_task
            else:
                if require_existing is not None:
                    raise _AudioCppGenerationChanged
                self._state = "starting"
                self._observation_version += 1
                epoch = self._lifecycle_epoch
                self._pre_spawn_launch = launch
                startup_task = asyncio.create_task(
                    self._start_generation(
                        launch,
                        epoch=epoch,
                        generation_hooks_factory=generation_hooks_factory,
                    )
                )
                self._startup_task = startup_task
                generation = None

        if cleanup_monitor is not None:
            await asyncio.shield(cleanup_monitor)
            return await self.ensure_running(
                launch,
                generation_hooks_factory=generation_hooks_factory,
                require_existing=require_existing,
            )

        if generation is not None:
            healthy = await asyncio.shield(self._shared_health_probe(generation))
            if not healthy:
                async with self._lock:
                    failure = self._last_failure or _failure_for(
                        _RUNTIME_UNHEALTHY_FAILURE,
                        generation.generation,
                    )
                raise _operation_error(failure)
            async with self._lock:
                if self._generation is not generation or self._state != "running":
                    raise _AudioCppGenerationChanged
                return self._ready_endpoint_locked()

        assert startup_task is not None
        return await asyncio.shield(startup_task)

    async def begin_draining(self) -> None:
        """Publish Draining for the current owned child, if any."""
        async with self._lock:
            if self._generation is not None and self._state not in (
                "draining",
                "stopping",
            ):
                self._state = "draining"
                self._observation_version += 1

    async def stop(
        self,
        *,
        application_shutdown: bool = False,
        expected_process_generation: int | None = None,
    ) -> None:
        """Stop only the accepted exact owned generation and join cleanup."""
        async with self._lock:
            if (
                self._blocked_cleanup_failure is not None
                and self._generation is None
                and self._pre_spawn_launch is None
            ):
                raise _operation_error(self._blocked_cleanup_failure)
            self._adopt_cleanup_deadline_locked(
                self._generation,
                current_shutdown_deadline(),
            )
            if expected_process_generation is not None and (
                self._generation is None
                or self._generation.generation != expected_process_generation
            ):
                return
            if self._stop_task is None:
                self._stop_task = asyncio.create_task(
                    self._stop_impl(
                        application_shutdown=application_shutdown,
                        expected_process_generation=expected_process_generation,
                    )
                )
            task = self._stop_task
        await join_retained_task(task)

    async def begin_terminal_shutdown(self, deadline: float | None) -> None:
        """Seal startup and propagate the outer deadline to retained cleanup."""
        async with self._lock:
            if not self._shutdown_started:
                self._shutdown_started = True
                self._lifecycle_epoch += 1
                self._state = "stopping"
                self._endpoint = None
                self._tts_capability = "unknown"
                self._consecutive_health_failures = 0
                self._observation_version += 1
            record = self._generation
            self._adopt_cleanup_deadline_locked(record, deadline)
            if record is not None:
                record.expected_exit = True
                record.terminal_state = "stopped"
                record.failure = None
            startup = self._startup_task
        if startup is not None and not startup.done():
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)

    async def wait_for_stage_application_boundary(self) -> None:
        """Wait for temporary Stopped/Unavailable cleanup ownership to settle."""
        while True:
            async with self._lock:
                if self._blocked_cleanup_failure is not None:
                    raise _operation_error(self._blocked_cleanup_failure)
                if self._state not in {"stopped", "unavailable"}:
                    return
                tasks = tuple(
                    task
                    for task in (
                        (
                            self._generation.exit_monitor
                            if self._generation is not None
                            else None
                        ),
                        self._startup_task,
                        self._stop_task,
                        self._close_task,
                    )
                    if task is not None and not task.done()
                )
                if not tasks:
                    return
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )

    async def close(self) -> None:
        """Perform retained terminal cleanup exactly once."""
        async with self._lock:
            if self._closed and self._close_task is None:
                return
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._close_impl())
            task = self._close_task
        await join_retained_task(task)

    async def wait_closed(self) -> None:
        """Wait until terminal cleanup owns no process or lifecycle task."""
        async with self._lock:
            task = self._close_task
            closed = self._closed
        if task is not None:
            await asyncio.shield(task)
        elif not closed:
            await self.close()

    def _matches_admission_locked(
        self, snapshot: AudioCppProcessAdmissionSnapshot
    ) -> bool:
        return (
            snapshot.lifecycle_epoch == self._lifecycle_epoch
            and snapshot.process_generation == self._process_generation
            and snapshot.state in ("starting", "running", "unhealthy")
            and (
                snapshot.state == self._state
                or (self._state == "draining" and self._generation is not None)
            )
            and (
                self._generation is None
                or (
                    self._generation.owned.process.returncode is None
                    and not self._generation.process_exited.is_set()
                )
            )
        )

    def _ready_endpoint_locked(self) -> AudioCppReadyEndpoint:
        assert self._endpoint is not None
        return AudioCppReadyEndpoint(
            base_url=self._endpoint,
            process_generation=self._process_generation,
            observation_version=self._observation_version,
        )

    async def _start_generation(
        self,
        launch: AudioCppManagedLaunchConfig,
        *,
        epoch: int,
        generation_hooks_factory: _GenerationHooksFactory,
    ) -> AudioCppReadyEndpoint:
        deadline = self._monotonic() + launch.startup_timeout_seconds
        record: _ProcessGeneration | None = None
        try:
            validated = self._revalidate_launch(launch)
            await self._assert_start_epoch(epoch)
            port = urlsplit(validated.base_url).port
            assert port is not None
            preflight_timeout = min(1.0, _remaining(deadline, self._monotonic))
            if preflight_timeout <= 0:
                raise _operation_error(_failure_for(_STARTUP_TIMEOUT_FAILURE, None))
            try:
                preflight = await asyncio.wait_for(
                    self._port_preflight(port, preflight_timeout),
                    timeout=preflight_timeout,
                )
            except asyncio.TimeoutError:
                preflight = "ambiguous"
            if preflight != "available":
                raise _operation_error(_failure_for(_PORT_FAILURE, None))

            await self._assert_start_epoch(epoch)
            spawn_validated = self._revalidate_launch(launch)
            if spawn_validated != validated:
                raise _operation_error(_failure_for(_CONFIGURATION_FAILURE, None))
            validated = spawn_validated
            spawn_timeout = _remaining(deadline, self._monotonic)
            if spawn_timeout <= 0:
                raise _operation_error(_failure_for(_STARTUP_TIMEOUT_FAILURE, None))
            (
                spawn_cancellation,
                spawn_timed_out,
                owned,
                spawn_error,
            ) = await _settle_process_launcher(
                self._process_launcher(validated, dict(self._child_environment)),
                timeout=spawn_timeout,
            )
            if owned is None:
                if spawn_cancellation is not None:
                    raise spawn_cancellation
                raise _operation_error(_failure_for(_SPAWN_FAILURE, None))

            async with self._lock:
                stale = (
                    spawn_cancellation is not None
                    or spawn_timed_out
                    or self._lifecycle_epoch != epoch
                    or self._state != "starting"
                )
                if self._generation is not None:
                    raise RuntimeError("audio.cpp process ownership invariant failed")
                self._process_generation += 1
                self._diagnostics.clear()
                self._last_failure = None
                if not stale:
                    self._endpoint = validated.base_url
                    self._tts_capability = "unknown"
                    self._consecutive_health_failures = 0
                record = _ProcessGeneration(
                    generation=self._process_generation,
                    epoch=epoch,
                    launch=validated,
                    owned=owned,
                    hooks_ready=asyncio.Event(),
                    process_exited=asyncio.Event(),
                )
                self._generation = record
                self._pre_spawn_launch = None
                record.stdout_drain = asyncio.create_task(
                    self._drain_output(record, "stdout", owned.process.stdout)
                )
                record.stderr_drain = asyncio.create_task(
                    self._drain_output(record, "stderr", owned.process.stderr)
                )
                record.exit_monitor = asyncio.create_task(self._monitor_exit(record))
                self._observation_version += 1
            if spawn_cancellation is not None:
                record.hooks_ready.set()
                raise spawn_cancellation
            if spawn_timed_out or spawn_error is not None:
                record.hooks_ready.set()
                failure = _failure_for(_SPAWN_FAILURE, record.generation)
                await self._rollback_generation(record, failure, deadline=deadline)
                raise _operation_error(failure)
            if stale:
                record.hooks_ready.set()
                raise asyncio.CancelledError
            assert record is not None

            if record.owned.process.returncode is not None:
                record.hooks_ready.set()
                failure = _failure_for(_PROCESS_EXITED_FAILURE, record.generation)
                await self._rollback_generation(record, failure, deadline=deadline)
                raise _operation_error(failure)

            hooks_failure: AudioCppProcessFailure | None = None
            try:
                hooks = await self._await_generation_step(
                    record,
                    generation_hooks_factory(record.generation),
                    timeout=_remaining(deadline, self._monotonic),
                )
            except asyncio.CancelledError:
                record.hooks_ready.set()
                raise
            except _ProcessExitedDuringStartup:
                hooks_failure = _failure_for(
                    _PROCESS_EXITED_FAILURE,
                    record.generation,
                )
            except BaseException:
                hooks_failure = _failure_for(_SPAWN_FAILURE, record.generation)
            if hooks_failure is not None:
                record.hooks_ready.set()
                await self._rollback_generation(
                    record,
                    hooks_failure,
                    deadline=deadline,
                )
                raise _operation_error(hooks_failure)
            record.hooks = hooks
            record.hooks_ready.set()

            await self._wait_for_startup_health(record, deadline)
            if record.owned.process.returncode is not None:
                failure = _failure_for(_PROCESS_EXITED_FAILURE, record.generation)
                await self._rollback_generation(record, failure, deadline=deadline)
                raise _operation_error(failure)

            contract_failure: AudioCppProcessFailure | None = None
            try:
                capability = await self._await_generation_step(
                    record,
                    hooks.contract_probe(),
                    timeout=_remaining(deadline, self._monotonic),
                )
            except asyncio.CancelledError:
                raise
            except _ProcessExitedDuringStartup:
                contract_failure = _failure_for(
                    _PROCESS_EXITED_FAILURE,
                    record.generation,
                )
            except asyncio.TimeoutError:
                contract_failure = _failure_for(
                    _STARTUP_TIMEOUT_FAILURE,
                    record.generation,
                )
            except TTSOperationError as error:
                spec = (
                    _CONTRACT_FAILURE
                    if error.code == "contract_incompatible"
                    else _STARTUP_TIMEOUT_FAILURE
                )
                contract_failure = _failure_for(spec, record.generation)
            except BaseException:
                contract_failure = _failure_for(
                    _CONTRACT_FAILURE,
                    record.generation,
                )
            if contract_failure is not None:
                await self._rollback_generation(
                    record,
                    contract_failure,
                    deadline=deadline,
                )
                raise _operation_error(contract_failure)

            if capability not in ("available", "not_configured"):
                failure = _failure_for(_CONTRACT_FAILURE, record.generation)
                await self._rollback_generation(record, failure, deadline=deadline)
                raise _operation_error(failure)

            startup_failure: AudioCppProcessFailure | None = None
            async with self._lock:
                process_failed = (
                    record.failure is not None
                    and record.failure.code == "process_exited"
                ) or (
                    not record.expected_exit
                    and (
                        record.owned.process.returncode is not None
                        or record.process_exited.is_set()
                    )
                )
                if process_failed:
                    startup_failure = self._publish_process_exited_locked(record)
                    stale = False
                elif (
                    self._generation is not record
                    or self._lifecycle_epoch != epoch
                    or self._state != "starting"
                ):
                    stale = True
                else:
                    stale = False
                    self._state = "running"
                    self._tts_capability = capability
                    self._last_failure = None
                    self._observation_version += 1
                    record.health_scheduler = asyncio.create_task(
                        self._health_scheduler(record)
                    )
                    endpoint = self._ready_endpoint_locked()
            if startup_failure is not None:
                await self._rollback_generation(
                    record,
                    startup_failure,
                    deadline=deadline,
                )
                raise _operation_error(startup_failure)
            if stale:
                raise asyncio.CancelledError
            return endpoint
        except asyncio.CancelledError:
            if record is not None:
                record.hooks_ready.set()
                await self._cancelled_start_rollback(record)
            raise
        except TTSOperationError as error:
            if record is None:
                await self._publish_pre_spawn_failure(epoch, error)
            raise
        finally:
            if record is None:
                artifact_succeeded = await self._cleanup_launch_artifact(launch)
                async with self._lock:
                    if self._pre_spawn_launch is launch and artifact_succeeded:
                        self._pre_spawn_launch = None
                if not artifact_succeeded:
                    await self._publish_cleanup_failure(None)
            current = asyncio.current_task()
            async with self._lock:
                if self._startup_task is current:
                    self._startup_task = None

    def _revalidate_launch(
        self, launch: AudioCppManagedLaunchConfig
    ) -> AudioCppManagedLaunchConfig:
        invalid = False
        artifact = launch.generated_artifact
        try:
            if artifact is not None:
                artifact.validate()
            config = AudioCppConfig.from_mapping(
                {
                    "mode": "managed",
                    "managed_binary_path": str(launch.binary_path),
                    "managed_server_json_path": str(launch.server_json_path),
                    "managed_startup_timeout_seconds": launch.startup_timeout_seconds,
                    "managed_health_check_interval_seconds": (
                        launch.health_check_interval_seconds
                    ),
                    "managed_termination_grace_seconds": (
                        launch.termination_grace_seconds
                    ),
                }
            )
            validated = validate_audio_cpp_managed_launch(config)
        except (TypeError, ValueError, OSError):
            invalid = True
        except Exception as error:
            self._record_internal_diagnostic("launch_revalidation", error)
            invalid = True
        if invalid:
            raise _operation_error(_failure_for(_CONFIGURATION_FAILURE, None))
        return replace(
            validated,
            expected_models=launch.expected_models,
            generated_artifact=artifact,
        )

    async def _assert_start_epoch(self, epoch: int) -> None:
        async with self._lock:
            if self._lifecycle_epoch != epoch or self._state != "starting":
                raise asyncio.CancelledError

    async def _wait_for_startup_health(
        self, record: _ProcessGeneration, deadline: float
    ) -> None:
        assert record.hooks is not None
        while True:
            if record.owned.process.returncode is not None:
                failure = _failure_for(_PROCESS_EXITED_FAILURE, record.generation)
                await self._rollback_generation(record, failure, deadline=deadline)
                raise _operation_error(failure)
            remaining = _remaining(deadline, self._monotonic)
            if remaining <= 0:
                failure = _failure_for(_STARTUP_TIMEOUT_FAILURE, record.generation)
                await self._rollback_generation(record, failure, deadline=deadline)
                raise _operation_error(failure)
            try:
                healthy = await self._await_generation_step(
                    record,
                    record.hooks.health_probe(),
                    timeout=remaining,
                )
            except asyncio.CancelledError:
                raise
            except _ProcessExitedDuringStartup:
                failure = _failure_for(_PROCESS_EXITED_FAILURE, record.generation)
                await self._rollback_generation(record, failure, deadline=deadline)
                raise _operation_error(failure) from None
            except BaseException:
                healthy = False
            if healthy:
                return
            delay = min(0.25, _remaining(deadline, self._monotonic))
            if delay <= 0:
                continue
            try:
                await self._await_generation_step(
                    record,
                    self._sleep(delay),
                    timeout=_remaining(deadline, self._monotonic),
                )
            except _ProcessExitedDuringStartup:
                failure = _failure_for(_PROCESS_EXITED_FAILURE, record.generation)
                await self._rollback_generation(record, failure, deadline=deadline)
                raise _operation_error(failure) from None
            except asyncio.TimeoutError:
                failure = _failure_for(_STARTUP_TIMEOUT_FAILURE, record.generation)
                await self._rollback_generation(record, failure, deadline=deadline)
                raise _operation_error(failure) from None

    async def _await_generation_step(
        self,
        record: _ProcessGeneration,
        step: Awaitable[_StepResult],
        *,
        timeout: float,
    ) -> _StepResult:
        step_task: asyncio.Future[_StepResult] = asyncio.ensure_future(step)
        exit_wait = asyncio.create_task(record.process_exited.wait())
        try:
            done, _pending = await asyncio.wait(
                (step_task, exit_wait),
                timeout=max(0.0, timeout),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if exit_wait in done:
                if not step_task.done():
                    step_task.cancel()
                await asyncio.gather(step_task, return_exceptions=True)
                raise _ProcessExitedDuringStartup
            if step_task not in done:
                step_task.cancel()
                await asyncio.gather(step_task, return_exceptions=True)
                raise asyncio.TimeoutError
            return step_task.result()
        except asyncio.CancelledError:
            if not step_task.done():
                step_task.cancel()
            await asyncio.gather(step_task, return_exceptions=True)
            raise
        finally:
            if not exit_wait.done():
                exit_wait.cancel()
            await asyncio.gather(exit_wait, return_exceptions=True)

    async def _publish_pre_spawn_failure(
        self, epoch: int, error: TTSOperationError
    ) -> None:
        async with self._lock:
            if self._lifecycle_epoch != epoch or self._state != "starting":
                return
            self._state = "unavailable"
            self._endpoint = None
            self._tts_capability = "unknown"
            self._last_failure = AudioCppProcessFailure(
                process_generation=None,
                code=error.code,
                message=str(error),
                retryable=error.retryable,
                recovery_action=error.recovery_action,
            )
            self._observation_version += 1

    async def _publish_cleanup_failure(
        self,
        process_generation: int | None,
    ) -> AudioCppProcessFailure:
        failure = _failure_for(_CLEANUP_FAILURE, process_generation)
        async with self._lock:
            self._blocked_cleanup_failure = failure
            self._state = "unavailable"
            self._endpoint = None
            self._tts_capability = "unknown"
            self._consecutive_health_failures = 0
            self._last_failure = failure
            self._observation_version += 1
        return failure

    async def _rollback_generation(
        self,
        record: _ProcessGeneration,
        failure: AudioCppProcessFailure,
        *,
        deadline: float | None = None,
    ) -> None:
        cleanup: asyncio.Task[None] | None = None
        async with self._lock:
            if self._generation is not record:
                return
            if self._state not in ("stopping", "draining"):
                record.terminal_state = "unavailable"
                record.failure = failure
                record.expected_exit = True
            cleanup = record.output_failure_cleanup
            if cleanup is not None:
                self._adopt_cleanup_deadline_locked(record, deadline)
        if cleanup is not None:
            await join_retained_task(cleanup)
        else:
            await self._terminate_and_join(record, deadline=deadline)

    async def _cancelled_start_rollback(self, record: _ProcessGeneration) -> None:
        async with self._lock:
            if self._generation is not record:
                return
            record.expected_exit = True
            if self._state not in ("stopping", "draining"):
                record.terminal_state = "unavailable"
                record.failure = _failure_for(
                    _PROCESS_EXITED_FAILURE, record.generation
                )
        await self._terminate_and_join(record)

    async def _terminate_and_join(
        self,
        record: _ProcessGeneration,
        *,
        deadline: float | None = None,
    ) -> None:
        process = record.owned.process
        outer_deadline = current_shutdown_deadline()
        effective_deadline = record.cleanup_deadline
        if deadline is not None:
            effective_deadline = (
                deadline
                if effective_deadline is None
                else min(effective_deadline, deadline)
            )
        if outer_deadline is not None:
            effective_deadline = (
                outer_deadline
                if effective_deadline is None
                else min(effective_deadline, outer_deadline)
            )
        self._adopt_cleanup_deadline_locked(record, effective_deadline)
        if process.returncode is None:
            try:
                process.terminate()
            except (OSError, ProcessLookupError):
                pass

        if not record.process_exited.is_set():
            grace_deadline = self._monotonic() + record.launch.termination_grace_seconds
            while not record.process_exited.is_set():
                record.cleanup_deadline_changed.clear()
                deadline = grace_deadline
                if record.cleanup_deadline is not None:
                    deadline = min(deadline, record.cleanup_deadline)
                remaining = _remaining(deadline, self._monotonic)
                if remaining <= 0:
                    break
                if record.process_exited.is_set():
                    break
                exited = asyncio.create_task(record.process_exited.wait())
                changed = asyncio.create_task(record.cleanup_deadline_changed.wait())
                done, pending = await asyncio.wait(
                    (exited, changed),
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if exited in done:
                    break
                if changed in done:
                    continue
                break
            if not record.process_exited.is_set() and process.returncode is None:
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
        monitor = record.exit_monitor
        if monitor is not None:
            await asyncio.shield(monitor)
        if record.cleanup_failure is not None:
            raise _operation_error(record.cleanup_failure) from None

    async def _drain_output(
        self,
        record: _ProcessGeneration,
        stream: _DiagnosticStream,
        reader: Any,
    ) -> None:
        try:
            while True:
                chunk = await reader.read(_MAX_DIAGNOSTIC_LINE_BYTES)
                if not chunk:
                    self._diagnostics.finish(stream)
                    return
                self._diagnostics.feed(stream, chunk)
        except asyncio.CancelledError:
            raise
        except BaseException:
            await self._schedule_output_failure(record)

    async def _schedule_output_failure(self, record: _ProcessGeneration) -> None:
        async with self._lock:
            if (
                self._generation is not record
                or record.owned.process.returncode is not None
                or record.output_failure_cleanup is not None
            ):
                return
            record.expected_exit = True
            record.terminal_state = "unavailable"
            record.failure = _failure_for(_PROCESS_EXITED_FAILURE, record.generation)
            self._lifecycle_epoch += 1
            self._state = "unavailable"
            self._endpoint = None
            self._tts_capability = "unknown"
            self._consecutive_health_failures = 0
            self._last_failure = record.failure
            self._observation_version += 1
            self._invalidate_generation(record)
            record.output_failure_cleanup = asyncio.create_task(
                self._terminate_after_output_failure(record)
            )

    async def _terminate_after_output_failure(
        self,
        record: _ProcessGeneration,
    ) -> None:
        try:
            await self._terminate_and_join(record)
        except TTSOperationError as error:
            if error.code != "cleanup_failed":
                raise

    async def _monitor_exit(self, record: _ProcessGeneration) -> None:
        await self._await_owned_process_exit(record)
        record.process_exited.set()

        tasks_to_cancel: list[asyncio.Task[Any]] = []
        async with self._lock:
            if record.health_scheduler is not None:
                tasks_to_cancel.append(record.health_scheduler)
            if record.health_probe is not None:
                tasks_to_cancel.append(record.health_probe)
            if self._generation is record:
                unexpected = not record.expected_exit
                self._endpoint = None
                self._tts_capability = "unknown"
                self._consecutive_health_failures = 0
                if unexpected:
                    self._state = "unavailable"
                    self._last_failure = _failure_for(
                        _PROCESS_EXITED_FAILURE,
                        record.generation,
                    )
                elif self._state not in ("draining", "stopping"):
                    self._state = record.terminal_state
                    self._last_failure = record.failure
                self._observation_version += 1

        await record.hooks_ready.wait()
        self._invalidate_generation(record)
        for task in tasks_to_cancel:
            if task is not asyncio.current_task():
                task.cancel()
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        await self._join_output_drains(record)
        self._close_parent_pipes(record)
        cleanup_succeeded, ownership_succeeded = await self._cleanup_generation(record)

        async with self._lock:
            if self._generation is not record:
                return
            if ownership_succeeded:
                self._generation = None
            if not cleanup_succeeded:
                cleanup_failure = _failure_for(
                    _CLEANUP_FAILURE,
                    record.generation,
                )
                record.cleanup_failure = cleanup_failure
                record.terminal_state = "unavailable"
                record.failure = cleanup_failure
                self._blocked_cleanup_failure = cleanup_failure
                self._state = "unavailable"
                self._last_failure = cleanup_failure
            elif record.expected_exit:
                self._state = record.terminal_state
                self._last_failure = record.failure
            self._observation_version += 1

    async def _await_owned_process_exit(self, record: _ProcessGeneration) -> None:
        """Run the sole wait while bounding inherited-pipe transport blocking."""
        process = record.owned.process
        waiter = asyncio.create_task(process.wait())
        try:
            while not waiter.done():
                if process.returncode is not None:
                    self._close_parent_pipes(record)
                    break
                await asyncio.wait({waiter}, timeout=0.05)
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            raise

    async def _join_output_drains(self, record: _ProcessGeneration) -> None:
        drains = [
            task
            for task in (record.stdout_drain, record.stderr_drain)
            if task is not None and task is not asyncio.current_task()
        ]
        if not drains:
            return
        timeout = 1.0
        if record.cleanup_deadline is not None:
            timeout = min(
                timeout,
                _remaining(record.cleanup_deadline, self._monotonic),
            )
        joined = asyncio.gather(*drains, return_exceptions=True)
        timer: asyncio.Future[None] = asyncio.ensure_future(self._sleep(timeout))
        done, _pending = await asyncio.wait(
            (joined, timer), return_when=asyncio.FIRST_COMPLETED
        )
        if joined not in done:
            self._close_parent_pipes(record)
            for task in drains:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*drains, return_exceptions=True)
        if not timer.done():
            timer.cancel()
        await asyncio.gather(timer, return_exceptions=True)

    def _close_parent_pipes(self, record: _ProcessGeneration) -> None:
        if record.parent_pipes_closed:
            return
        record.parent_pipes_closed = True
        try:
            record.owned.close_parent_pipes()
        except BaseException:
            pass

    async def _cleanup_generation(
        self,
        record: _ProcessGeneration,
    ) -> tuple[bool, bool]:
        self._invalidate_generation(record)
        control_flow: BaseException | None = None
        if not record.hooks_cleanup_settled:
            record.hooks_cleanup_settled = True
            try:
                if record.hooks is not None:
                    await record.hooks.cleanup()
            except BaseException as error:
                record.hooks_cleanup_succeeded = False
                if isinstance(error, Exception):
                    self._record_internal_diagnostic("generation_cleanup", error)
                else:
                    control_flow = error
        if not record.artifact_cleanup_succeeded:
            record.artifact_cleanup_succeeded = await self._cleanup_launch_artifact(
                record.launch
            )
        transport_succeeded = self._close_native_transport(record)
        if control_flow is not None:
            raise control_flow
        return (
            record.hooks_cleanup_succeeded
            and record.artifact_cleanup_succeeded
            and transport_succeeded,
            record.artifact_cleanup_succeeded and transport_succeeded,
        )

    @staticmethod
    def _close_native_transport(record: _ProcessGeneration) -> bool:
        if record.native_transport_closed:
            return True
        try:
            record.owned.close_native_transport()
        except Exception:
            return False
        record.native_transport_closed = True
        return True

    async def _cleanup_launch_artifact(
        self,
        launch: AudioCppManagedLaunchConfig,
    ) -> bool:
        artifact = launch.generated_artifact
        if artifact is None:
            return True
        try:
            await asyncio.to_thread(artifact.cleanup)
        except (TypeError, ValueError, OSError):
            return False
        except Exception as error:
            self._record_internal_diagnostic("artifact_cleanup", error)
            return False
        return True

    @staticmethod
    def _invalidate_generation(record: _ProcessGeneration) -> None:
        if record.invalidation_called or record.hooks is None:
            return
        record.invalidation_called = True
        invalidate = record.hooks.invalidate
        if invalidate is None:
            return
        try:
            invalidate()
        except BaseException:
            pass

    async def _health_scheduler(self, record: _ProcessGeneration) -> None:
        try:
            while True:
                await self._sleep(record.launch.health_check_interval_seconds)
                async with self._lock:
                    if self._generation is not record or self._state not in (
                        "running",
                        "unhealthy",
                    ):
                        return
                await self._shared_health_probe(record)
        except asyncio.CancelledError:
            raise

    async def _shared_health_probe(self, record: _ProcessGeneration) -> bool:
        async with self._lock:
            if self._generation is not record or record.hooks is None:
                return False
            if record.health_probe is None:
                record.health_probe = asyncio.create_task(
                    self._perform_health_probe(record)
                )
            task = record.health_probe
        return await asyncio.shield(task)

    async def _perform_health_probe(self, record: _ProcessGeneration) -> bool:
        current = asyncio.current_task()
        try:
            assert record.hooks is not None
            try:
                healthy = bool(await record.hooks.health_probe())
            except asyncio.CancelledError:
                raise
            except BaseException:
                healthy = False
            async with self._lock:
                if self._generation is not record or self._state not in (
                    "running",
                    "unhealthy",
                ):
                    return False
                if (
                    record.owned.process.returncode is not None
                    or record.process_exited.is_set()
                ):
                    self._publish_process_exited_locked(record)
                    return False
                if healthy:
                    recovered_from_failure = self._consecutive_health_failures > 0
                    self._consecutive_health_failures = 0
                    if self._state == "unhealthy":
                        self._state = "running"
                        if (
                            self._last_failure is not None
                            and self._last_failure.code == "runtime_unhealthy"
                        ):
                            self._last_failure = None
                    if recovered_from_failure:
                        self._observation_version += 1
                else:
                    self._consecutive_health_failures += 1
                    self._observation_version += 1
                    if self._consecutive_health_failures >= 2:
                        self._state = "unhealthy"
                        self._last_failure = _failure_for(
                            _RUNTIME_UNHEALTHY_FAILURE, record.generation
                        )
                return healthy
        finally:
            async with self._lock:
                if record.health_probe is current:
                    record.health_probe = None

    def _publish_process_exited_locked(
        self, record: _ProcessGeneration
    ) -> AudioCppProcessFailure:
        failure = record.failure
        if failure is None or failure.code != "process_exited":
            failure = _failure_for(_PROCESS_EXITED_FAILURE, record.generation)
            record.failure = failure
        record.terminal_state = "unavailable"
        if self._generation is record:
            changed = (
                self._state != "unavailable"
                or self._endpoint is not None
                or self._tts_capability != "unknown"
                or self._consecutive_health_failures != 0
                or self._last_failure != failure
            )
            self._state = "unavailable"
            self._endpoint = None
            self._tts_capability = "unknown"
            self._consecutive_health_failures = 0
            self._last_failure = failure
            self._invalidate_generation(record)
            if changed:
                self._observation_version += 1
        return failure

    async def _stop_impl(
        self,
        *,
        application_shutdown: bool,
        expected_process_generation: int | None,
    ) -> None:
        current = asyncio.current_task()
        try:
            async with self._lock:
                if expected_process_generation is not None and (
                    self._generation is None
                    or self._generation.generation != expected_process_generation
                ):
                    return
                self._lifecycle_epoch += 1
                self._state = "stopping"
                self._observation_version += 1
                startup = self._startup_task
                record = self._generation
                scheduler = record.health_scheduler if record is not None else None
                probe = record.health_probe if record is not None else None
                if record is not None:
                    shutdown_deadline = current_shutdown_deadline()
                    if shutdown_deadline is not None:
                        self._adopt_cleanup_deadline_locked(
                            record,
                            shutdown_deadline,
                        )
                    record.expected_exit = True
                    record.terminal_state = "stopped"
                    record.failure = None
            for task in (scheduler, probe):
                if task is not None:
                    task.cancel()
            if startup is not None and not startup.done():
                startup.cancel()
                await asyncio.gather(startup, return_exceptions=True)
            for task in (scheduler, probe):
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)

            async with self._lock:
                record = self._generation
                pre_spawn_launch = self._pre_spawn_launch
                if record is not None:
                    record.expected_exit = True
                    record.terminal_state = "stopped"
                    record.failure = None
            if record is not None:
                if (
                    record.process_exited.is_set()
                    and record.cleanup_failure is not None
                ):
                    (
                        cleanup_succeeded,
                        ownership_succeeded,
                    ) = await self._cleanup_generation(record)
                    async with self._lock:
                        if self._generation is record and ownership_succeeded:
                            self._generation = None
                        if cleanup_succeeded:
                            record.cleanup_failure = None
                            self._blocked_cleanup_failure = None
                    if not cleanup_succeeded:
                        failure = record.cleanup_failure or _failure_for(
                            _CLEANUP_FAILURE,
                            record.generation,
                        )
                        raise _operation_error(failure) from None
                else:
                    await self._terminate_and_join(record)
            elif pre_spawn_launch is not None:
                artifact_succeeded = await self._cleanup_launch_artifact(
                    pre_spawn_launch
                )
                async with self._lock:
                    if (
                        self._pre_spawn_launch is pre_spawn_launch
                        and artifact_succeeded
                    ):
                        self._pre_spawn_launch = None
                        self._blocked_cleanup_failure = None
                        self._last_failure = None
                if not artifact_succeeded:
                    failure = await self._publish_cleanup_failure(None)
                    raise _operation_error(failure) from None

            async with self._lock:
                if self._generation is None:
                    self._state = "stopped"
                    self._endpoint = None
                    self._tts_capability = "unknown"
                    self._consecutive_health_failures = 0
                    self._observation_version += 1
        finally:
            async with self._lock:
                if self._stop_task is current:
                    self._stop_task = None

    async def _close_impl(self) -> None:
        current = asyncio.current_task()
        succeeded = False
        try:
            await self.stop(application_shutdown=True)
            succeeded = True
        finally:
            async with self._lock:
                if succeeded:
                    self._closed = True
                    self._diagnostics.clear()
                    self._last_failure = None
                    self._blocked_cleanup_failure = None
                    self._generation = None
                    self._pre_spawn_launch = None
                    self._startup_task = None
                    self._stop_task = None
                    self._state = "stopped"
                    self._endpoint = None
                    self._tts_capability = "unknown"
                    self._consecutive_health_failures = 0
                    self._observation_version += 1
                if self._close_task is current:
                    self._close_task = None

    @staticmethod
    def _adopt_cleanup_deadline_locked(
        record: _ProcessGeneration | None,
        deadline: float | None,
    ) -> None:
        if record is None or deadline is None:
            return
        if record.cleanup_deadline is None or deadline < record.cleanup_deadline:
            record.cleanup_deadline = deadline
            record.cleanup_deadline_changed.set()
