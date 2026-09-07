"""Bounded dictation-next admission for the queue-less local STT executor."""

from __future__ import annotations

import threading
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .contracts import BufferAudioSource, TranscriptionFailureCode
from .executor import (
    ExecutorBusyError,
    ExecutorEvent,
    ExecutorFailure,
    ExecutorResult,
    ExecutorUnavailableError,
    LocalSTTExecutor,
    WorkerPhase,
)
from .parakeet_dispatch import ParakeetDispatch

DICTATION_MAX_SECONDS = 60.0
_RETRY_ACTION = "retry_faster_whisper"


def pcm_byte_limit(*, sample_rate: int, channels: int, sample_width: int) -> int:
    """Return the canonical 60-second PCM byte ceiling.

    Args:
        sample_rate: PCM sample rate in frames per second.
        channels: Number of interleaved PCM channels.
        sample_width: Bytes per PCM channel sample.

    Returns:
        Maximum frame-aligned bytes accepted for one dictation capture.

    Raises:
        ValueError: Any PCM format value is invalid.
    """

    _validate_format(sample_rate, channels, sample_width)
    return int(sample_rate * channels * sample_width * DICTATION_MAX_SECONDS)


class DictationAppendStatus(str, Enum):
    """Result of one bounded logical-segment append."""

    ACCEPTED = "accepted"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class RetryableDictationBuffer:
    """PCM and logical boundaries retained for one explicit retry."""

    source: BufferAudioSource
    segment_end_frames: tuple[int, ...]


class RetryableDictationFailure(RuntimeError):
    """Sanitized failure carrying bounded explicit-retry PCM."""

    def __init__(self, retry_buffer: RetryableDictationBuffer) -> None:
        self.retry_buffer = retry_buffer
        super().__init__("Parakeet transcription failed.")


@dataclass(slots=True, eq=False, weakref_slot=True)
class _Capture:
    generation: int
    dispatch: ParakeetDispatch
    sample_rate: int
    channels: int
    sample_width: int
    language: str
    callback: Callable[[int, str], None]
    done: threading.Event = field(default_factory=threading.Event)
    total_bytes: int = 0
    next_sequence: int = 0
    finished: bool = False
    cancelled: bool = False
    force_cancel: bool = False
    retrying: bool = False
    failure: TranscriptionFailureCode | None = None
    retry_audio: bytearray = field(default_factory=bytearray, repr=False)
    retry_ends: list[int] = field(default_factory=list, repr=False)
    retry_buffer: RetryableDictationBuffer | None = field(default=None, repr=False)
    last_payload: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def frame_bytes(self) -> int:
        return self.channels * self.sample_width


@dataclass(slots=True)
class _Pending:
    capture: _Capture
    sequence_start: int
    audio: bytearray = field(default_factory=bytearray, repr=False)
    ends: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Request:
    capture: _Capture
    attempt_id: str
    source: BufferAudioSource = field(repr=False)
    ends: tuple[int, ...]
    sequence_start: int


@dataclass(slots=True)
class _Handoff:
    running: bool = False
    deferred: tuple[_Callbacks, ExecutorResult | ExecutorFailure] | None = None


@dataclass(slots=True)
class _Callbacks:
    kind: Literal["library", "dictation"]
    attempt_id: str
    on_result: Callable[[ExecutorResult], None]
    on_failure: Callable[[ExecutorFailure], None]
    on_event: Callable[[ExecutorEvent], None] = lambda _event: None
    request: _Request | None = None
    generation: int | None = None
    started: bool = False
    handoff: _Handoff = field(default_factory=_Handoff)
    early: ExecutorResult | ExecutorFailure | None = None


def _validate_format(sample_rate: int, channels: int, sample_width: int) -> None:
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if type(channels) is not int or channels <= 0:
        raise ValueError("channels must be a positive integer")
    if type(sample_width) is not int or sample_width not in {1, 2, 3, 4}:
        raise ValueError("sample_width must be one of 1, 2, 3, or 4")


class DictationCaptureHandle:
    """One asynchronous bounded dictation capture."""

    __slots__ = ("_capture", "_coordinator")

    def __init__(
        self,
        coordinator: LocalSTTDispatchCoordinator,
        capture: _Capture,
    ) -> None:
        self._coordinator = coordinator
        self._capture = capture

    @property
    def waiting_for_executor(self) -> bool:
        """Return whether this capture waits behind active Library work."""

        return self._coordinator._waiting(self._capture)

    def append_segment(self, audio: bytes) -> DictationAppendStatus:
        """Append one logical PCM segment without waiting for inference."""

        return self._coordinator._append(self._capture, audio)

    def finish(self) -> None:
        """Seal the capture after its last accepted segment."""

        self._coordinator._finish(self._capture)

    def wait(self) -> None:
        """Wait for completion and raise only normalized failures."""

        self._capture.done.wait()
        with self._coordinator._lock:
            retry, failure = self._capture.retry_buffer, self._capture.failure
        if retry is not None:
            raise RetryableDictationFailure(retry)
        if failure is not None:
            raise RuntimeError(failure.value)

    def cancel(self, *, force: bool = False) -> bool:
        """Cancel pending work or the exact active executor attempt."""

        return self._coordinator._cancel(self._capture, force)

    def take_retry_buffer(self) -> RetryableDictationBuffer | None:
        """Transfer retained retry PCM at most once."""

        return self._coordinator._take_retry(self._capture)


class LocalSTTDispatchCoordinator:
    """Admit Library work and one dictation reservation in strict order."""

    def __init__(
        self,
        executor: LocalSTTExecutor,
        *,
        on_dictation_idle: Callable[[], None] | None = None,
    ) -> None:
        self._executor = executor
        self._on_dictation_idle = on_dictation_idle or (lambda: None)
        self._lock = threading.RLock()
        self._active_kind: Literal["library", "dictation"] | None = None
        self._active_attempt_id: str | None = None
        self._reservation: _Capture | None = None
        self._pending: _Pending | None = None
        self._retry_owners: weakref.WeakSet[_Capture] = weakref.WeakSet()
        self._closed = False

    @property
    def dictation_reserved(self) -> bool:
        """Return whether dictation gates future heavy Library work."""

        with self._lock:
            return self._reservation is not None

    def submit_library(self, **executor_kwargs: Any) -> int:
        """Submit one Library request immediately, without queueing it."""

        attempt_id = executor_kwargs.get("attempt_id")
        if type(attempt_id) is not str or not attempt_id.strip():
            raise ValueError("attempt_id must be a non-empty string")
        callbacks = _Callbacks(
            "library",
            attempt_id,
            executor_kwargs.pop("on_result", lambda _result: None),
            executor_kwargs.pop("on_failure", lambda _failure: None),
            on_event=executor_kwargs.pop("on_event", lambda _event: None),
        )
        with self._lock:
            if self._closed:
                raise ExecutorUnavailableError("Local STT dispatch is closed")
            if self._reservation is not None:
                raise ExecutorBusyError("Local STT dictation has the next slot")
            if self._active_kind is not None:
                raise ExecutorBusyError("Local STT already has active work")
            self._claim_locked("library", attempt_id)
        try:
            generation = self._executor.submit(
                **executor_kwargs,
                on_event=lambda value: self._event(callbacks, value),
                on_result=lambda value: self._terminal(callbacks, value),
                on_failure=lambda value: self._terminal(callbacks, value),
            )
        except Exception:
            next_request = done = None
            notify = False
            with self._lock:
                if self._matches_locked("library", attempt_id):
                    self._clear_active_locked()
                    next_request, done, notify = self._advance_locked()
            if next_request is not None:
                submit_done, submit_notify = self._submit_dictation(next_request)
                done, notify = submit_done or done, submit_notify or notify
            self._post(done, notify)
            raise
        self._publish_generation(callbacks, generation)
        return generation

    def transcribe_buffer(
        self,
        *,
        source: BufferAudioSource,
        dispatch: ParakeetDispatch,
        language: str,
    ) -> dict[str, Any]:
        """Submit one buffer through shared admission and block for its payload."""

        if type(source) is not BufferAudioSource:
            raise TypeError("source must be a BufferAudioSource")
        limit = pcm_byte_limit(
            sample_rate=source.sample_rate,
            channels=source.channels,
            sample_width=source.sample_width,
        )
        if len(source.audio) > limit:
            raise ValueError("source audio exceeds the 60-second PCM limit")
        handle = self.begin_dictation(
            capture_generation=uuid.uuid4().int,
            dispatch=dispatch,
            sample_rate=source.sample_rate,
            channels=source.channels,
            sample_width=source.sample_width,
            language=language,
            on_logical_segment=lambda _sequence, _text: None,
        )
        handle.append_segment(source.audio)
        handle.finish()
        handle.wait()
        with self._lock:
            payload = handle._capture.last_payload
        if payload is None:
            raise RuntimeError(TranscriptionFailureCode.INFERENCE_FAILED.value)
        return payload

    def begin_dictation(
        self,
        *,
        capture_generation: int,
        dispatch: ParakeetDispatch,
        sample_rate: int,
        channels: int,
        sample_width: int,
        language: str,
        on_logical_segment: Callable[[int, str], None],
    ) -> DictationCaptureHandle:
        """Reserve the next local-STT slot for one capture."""

        if type(capture_generation) is not int or capture_generation < 0:
            raise ValueError("capture_generation must be a non-negative integer")
        if type(dispatch) is not ParakeetDispatch:
            raise TypeError("dispatch must be a ParakeetDispatch")
        _validate_format(sample_rate, channels, sample_width)
        if type(language) is not str or not language.strip():
            raise ValueError("language must be a non-empty string")
        if not callable(on_logical_segment):
            raise TypeError("on_logical_segment must be callable")
        with self._lock:
            if self._closed:
                raise RuntimeError("Local STT dispatch is closed")
            existing = self._reservation
            match = existing is not None and (
                existing.generation,
                existing.dispatch.identity,
                existing.sample_rate,
                existing.channels,
                existing.sample_width,
                existing.language,
            ) == (
                capture_generation,
                dispatch.identity,
                sample_rate,
                channels,
                sample_width,
                language,
            )
            if existing is not None:
                if match and not existing.cancelled:
                    return DictationCaptureHandle(self, existing)
                raise ExecutorBusyError("Another dictation capture is already reserved")
            capture = _Capture(
                capture_generation,
                dispatch,
                sample_rate,
                channels,
                sample_width,
                language,
                on_logical_segment,
            )
            self._reservation = capture
            return DictationCaptureHandle(self, capture)

    def close(self) -> None:
        """Release coordinator-owned audio without closing or joining the executor."""

        done = None
        attempt_id = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for owner in tuple(self._retry_owners):
                self._release_retry_locked(owner)
            capture = self._reservation
            if capture is None:
                return
            self._mark_cancelled_locked(capture)
            if self._active_kind == "dictation":
                attempt_id = self._active_attempt_id
            else:
                self._reservation, done = None, capture.done
        if attempt_id is not None:
            self._executor.cancel(attempt_id)
        if done is not None:
            done.set()

    def _waiting(self, capture: _Capture) -> bool:
        with self._lock:
            return self._reservation is capture and self._active_kind == "library"

    def _append(self, capture: _Capture, audio: bytes) -> DictationAppendStatus:
        if type(audio) is not bytes:
            raise TypeError("audio must be bytes")
        if not audio:
            raise ValueError("audio must not be empty")
        if len(audio) % capture.frame_bytes:
            raise ValueError("audio must contain complete interleaved PCM frames")
        request = done = None
        notify = False
        with self._lock:
            if self._closed:
                raise RuntimeError("Local STT dispatch is closed")
            if self._reservation is not capture or capture.cancelled:
                raise RuntimeError("dictation capture is no longer active")
            if capture.finished:
                raise RuntimeError("dictation capture is already finished")
            limit = pcm_byte_limit(
                sample_rate=capture.sample_rate,
                channels=capture.channels,
                sample_width=capture.sample_width,
            )
            accepted = audio[: limit - capture.total_bytes]
            capture.total_bytes += len(accepted)
            if capture.retrying:
                capture.retry_audio.extend(accepted)
                capture.retry_ends.append(
                    len(capture.retry_audio) // capture.frame_bytes
                )
                capture.next_sequence += 1
            else:
                self._append_pending_locked(capture, accepted)
                if self._active_kind is None:
                    request = self._snapshot_locked()
            reached = len(accepted) < len(audio) or capture.total_bytes == limit
            if reached:
                capture.finished = True
                if capture.retrying:
                    self._finalize_retry_locked(capture)
                    notify = self._clear_reservation_locked(capture)
                    done = capture.done
        if request is not None:
            submit_done, submit_notify = self._submit_dictation(request)
            done, notify = submit_done or done, submit_notify or notify
        self._post(done, notify)
        return (
            DictationAppendStatus.LIMIT_REACHED
            if reached
            else DictationAppendStatus.ACCEPTED
        )

    def _finish(self, capture: _Capture) -> None:
        done = None
        notify = False
        with self._lock:
            if capture.done.is_set() or capture.cancelled:
                return
            if self._reservation is not capture:
                return
            capture.finished = True
            if capture.retrying:
                self._finalize_retry_locked(capture)
            if self._active_kind != "dictation" and self._pending is None:
                notify = self._clear_reservation_locked(capture)
                done = capture.done
        self._post(done, notify)

    def _cancel(self, capture: _Capture, force: bool) -> bool:
        done = None
        notify = False
        attempt_id = None
        with self._lock:
            if capture.cancelled:
                return False
            if capture.done.is_set():
                if capture.retry_buffer is None:
                    return False
                self._mark_cancelled_locked(capture)
                return True
            if self._reservation is not capture:
                return False
            capture.force_cancel = force
            self._mark_cancelled_locked(capture)
            if self._active_kind == "dictation":
                attempt_id = self._active_attempt_id
            else:
                notify = self._clear_reservation_locked(capture)
                done = capture.done
        if attempt_id is not None:
            method = self._executor.force_stop if force else self._executor.cancel
            method(attempt_id)
        self._post(done, notify)
        return True

    def _take_retry(self, capture: _Capture) -> RetryableDictationBuffer | None:
        with self._lock:
            value, capture.retry_buffer = capture.retry_buffer, None
            self._retry_owners.discard(capture)
            return value

    def _append_pending_locked(self, capture: _Capture, audio: bytes) -> None:
        if self._pending is None:
            self._pending = _Pending(capture, capture.next_sequence)
        elif self._pending.capture is not capture:
            raise ExecutorBusyError("Another dictation request is already pending")
        self._pending.audio.extend(audio)
        self._pending.ends.append(len(self._pending.audio) // capture.frame_bytes)
        capture.next_sequence += 1

    def _snapshot_locked(self) -> _Request:
        pending = self._pending
        if pending is None or not pending.audio:
            raise RuntimeError("no pending dictation audio")
        capture = pending.capture
        request = _Request(
            capture,
            uuid.uuid4().hex,
            BufferAudioSource(
                bytes(pending.audio),
                capture.sample_rate,
                capture.channels,
                capture.sample_width,
            ),
            tuple(pending.ends),
            pending.sequence_start,
        )
        self._pending = None
        self._claim_locked("dictation", request.attempt_id)
        return request

    def _submit_dictation(
        self,
        request: _Request,
        handoff: _Handoff | None = None,
    ) -> tuple[threading.Event | None, bool]:
        callbacks = _Callbacks(
            "dictation",
            request.attempt_id,
            lambda _result: None,
            lambda _failure: None,
            request=request,
            handoff=handoff or _Handoff(),
        )
        capture = request.capture
        options = dict(capture.dispatch.option_updates)
        options["language"] = capture.language
        try:
            generation = self._executor.submit(
                attempt_id=request.attempt_id,
                job_id=None,
                source=request.source,
                identity=capture.dispatch.identity,
                options=options,
                segment_end_frames=request.ends,
                local_source=capture.dispatch.local_source,
                managed_store_root=capture.dispatch.managed_store_root,
                managed_artifact_ref=capture.dispatch.managed_artifact_ref,
                managed_dependency_refs=capture.dispatch.managed_dependency_refs,
                on_result=lambda value: self._terminal(callbacks, value),
                on_failure=lambda value: self._terminal(callbacks, value),
            )
        except Exception:
            with self._lock:
                if not self._matches_locked("dictation", request.attempt_id):
                    return None, False
                self._clear_active_locked()
                capture.failure = (
                    TranscriptionFailureCode.CANCELLED
                    if capture.cancelled or self._closed
                    else TranscriptionFailureCode.PROVIDER_UNAVAILABLE
                )
                capture.finished = True
                self._discard_pending_locked(capture)
                return capture.done, self._clear_reservation_locked(capture)
        self._publish_generation(callbacks, generation)
        with self._lock:
            cancel_method = (
                self._executor.force_stop
                if capture.cancelled and capture.force_cancel
                else self._executor.cancel
            )
            cancel_after_submit = capture.cancelled or self._closed
        if cancel_after_submit:
            cancel_method(request.attempt_id)
        return None, False

    def _publish_generation(self, callbacks: _Callbacks, generation: int) -> None:
        with self._lock:
            if callbacks.generation is None:
                callbacks.generation = generation
            early = callbacks.early
            callbacks.early = None
        if early is not None:
            self._terminal(callbacks, early)

    def _event(self, callbacks: _Callbacks, event: ExecutorEvent) -> None:
        with self._lock:
            if (
                callbacks.started
                or event.attempt_id != callbacks.attempt_id
                or not self._matches_locked(callbacks.kind, callbacks.attempt_id)
            ):
                return
            generation = callbacks.generation
            if generation is None:
                if event.phase is not WorkerPhase.PREPARING:
                    return
                callbacks.generation = event.generation
            elif event.generation != generation:
                if (
                    event.phase is not WorkerPhase.PREPARING
                    or event.generation < generation
                ):
                    return
                callbacks.generation = event.generation
        self._deliver(callbacks.on_event, event)

    def _terminal(
        self,
        callbacks: _Callbacks,
        envelope: ExecutorResult | ExecutorFailure,
    ) -> None:
        with self._lock:
            if (
                callbacks.started
                or envelope.attempt_id != callbacks.attempt_id
                or not self._matches_locked(callbacks.kind, callbacks.attempt_id)
            ):
                return
            if callbacks.generation is None:
                callbacks.early = callbacks.early or envelope
                return
            if envelope.generation != callbacks.generation:
                return
            callbacks.started = True
            handoff = callbacks.handoff
            if handoff.running:
                handoff.deferred = handoff.deferred or (callbacks, envelope)
                return
            handoff.running = True
        threading.Thread(
            target=self._handoff,
            args=(handoff, callbacks, envelope),
            name=f"local-stt-dispatch-{callbacks.attempt_id[:12]}",
            daemon=True,
        ).start()

    def _handoff(
        self,
        handoff: _Handoff,
        callbacks: _Callbacks,
        envelope: ExecutorResult | ExecutorFailure,
    ) -> None:
        current = (callbacks, envelope)
        while True:
            self._transition(*current)
            with self._lock:
                if handoff.deferred is None:
                    handoff.running = False
                    return
                current, handoff.deferred = handoff.deferred, None

    def _transition(
        self,
        callbacks: _Callbacks,
        envelope: ExecutorResult | ExecutorFailure,
    ) -> None:
        next_request = done = None
        notify = deliver_segments = False
        with self._lock:
            if not self._matches_locked(callbacks.kind, callbacks.attempt_id):
                return
            self._clear_active_locked()
            if callbacks.kind == "dictation":
                request = callbacks.request
                assert request is not None
                capture = request.capture
                if capture.cancelled:
                    self._mark_cancelled_locked(capture)
                elif type(envelope) is ExecutorResult:
                    capture.last_payload = envelope.payload
                    deliver_segments = True
                elif (
                    envelope.code is not TranscriptionFailureCode.CANCELLED
                    and _RETRY_ACTION in envelope.recovery_actions
                ):
                    capture.retrying, capture.failure = True, None
                    self._merge_retry_locked(capture, request.source.audio, request.ends)
                    if self._pending is not None and self._pending.capture is capture:
                        self._merge_retry_locked(
                            capture,
                            bytes(self._pending.audio),
                            tuple(self._pending.ends),
                        )
                        self._pending = None
                else:
                    capture.failure, capture.finished = envelope.code, True
                    self._discard_pending_locked(capture)
            next_request, done, notify = self._advance_locked()
        if next_request is not None:
            submit_done, submit_notify = self._submit_dictation(
                next_request,
                callbacks.handoff,
            )
            done, notify = submit_done or done, submit_notify or notify
        if callbacks.kind == "library":
            callback = (
                callbacks.on_result
                if type(envelope) is ExecutorResult
                else callbacks.on_failure
            )
            self._deliver(callback, envelope)
        elif deliver_segments:
            assert callbacks.request is not None and type(envelope) is ExecutorResult
            segments = envelope.payload.get("logical_segments", ())
            if isinstance(segments, (tuple, list)):
                for offset, text in enumerate(segments):
                    if type(text) is str:
                        self._deliver(
                            callbacks.request.capture.callback,
                            callbacks.request.sequence_start + offset,
                            text,
                        )
        self._post(done, notify)

    def _advance_locked(
        self,
    ) -> tuple[_Request | None, threading.Event | None, bool]:
        capture = self._reservation
        if (
            capture is not None
            and not capture.cancelled
            and not capture.retrying
            and self._pending is not None
        ):
            return self._snapshot_locked(), None, False
        if capture is not None and capture.finished:
            if capture.retrying:
                self._finalize_retry_locked(capture)
            return None, capture.done, self._clear_reservation_locked(capture)
        return None, None, False

    def _merge_retry_locked(
        self,
        capture: _Capture,
        audio: bytes,
        ends: tuple[int, ...],
    ) -> None:
        base = len(capture.retry_audio) // capture.frame_bytes
        capture.retry_audio.extend(audio)
        capture.retry_ends.extend(base + end for end in ends)

    def _finalize_retry_locked(self, capture: _Capture) -> None:
        if capture.retry_buffer is None and capture.retry_audio:
            capture.retry_buffer = RetryableDictationBuffer(
                BufferAudioSource(
                    bytes(capture.retry_audio),
                    capture.sample_rate,
                    capture.channels,
                    capture.sample_width,
                ),
                tuple(capture.retry_ends),
            )
            self._retry_owners.add(capture)
        capture.retry_audio.clear()
        capture.retry_ends.clear()

    def _mark_cancelled_locked(self, capture: _Capture) -> None:
        capture.cancelled = capture.finished = True
        capture.failure = TranscriptionFailureCode.CANCELLED
        self._release_retry_locked(capture)
        self._discard_pending_locked(capture)

    def _release_retry_locked(self, capture: _Capture) -> None:
        capture.retry_audio.clear()
        capture.retry_ends.clear()
        capture.retry_buffer = None
        self._retry_owners.discard(capture)

    def _discard_pending_locked(self, capture: _Capture) -> None:
        if self._pending is not None and self._pending.capture is capture:
            self._pending = None

    def _clear_reservation_locked(self, capture: _Capture) -> bool:
        if self._reservation is not capture:
            return False
        self._reservation = None
        return True

    def _claim_locked(
        self,
        kind: Literal["library", "dictation"],
        attempt_id: str,
    ) -> None:
        self._active_kind, self._active_attempt_id = kind, attempt_id

    def _matches_locked(
        self,
        kind: Literal["library", "dictation"],
        attempt_id: str,
    ) -> bool:
        return self._active_kind == kind and self._active_attempt_id == attempt_id

    def _clear_active_locked(self) -> None:
        self._active_kind = self._active_attempt_id = None

    def _post(self, done: threading.Event | None, notify: bool) -> None:
        if done is not None:
            done.set()
        if notify and not self._closed:
            self._deliver(self._on_dictation_idle)

    @staticmethod
    def _deliver(callback: Callable[..., Any], *args: Any) -> None:
        try:
            callback(*args)
        except Exception:
            return


__all__ = [
    "DICTATION_MAX_SECONDS",
    "DictationAppendStatus",
    "DictationCaptureHandle",
    "LocalSTTDispatchCoordinator",
    "RetryableDictationBuffer",
    "RetryableDictationFailure",
    "pcm_byte_limit",
]
