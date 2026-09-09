"""Incremental transcript revisions for live and rolling-window STT.

Only downstream-admitted audio enters this engine. Native callbacks are
serialized onto the owning event loop; batch-only adapters receive one
bounded, truthfully timed PCM window at a time.
"""

from __future__ import annotations

import asyncio
import inspect
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from .duplex_contracts import AudioFrame

TranscriptMode = Literal["live", "rolling-window"]
TranscriptFailureCode = Literal["backend_failed", "fallback_failed"]

_PCM_SAMPLE_RATE = 48_000
_PCM_BYTES_PER_SAMPLE = 2
_NANOSECONDS_PER_SECOND = 1_000_000_000
_MAX_ROLLING_WINDOW_NS = 10_000_000_000
_MAX_ROLLING_PCM_BYTES = 1_000_000
_ADAPTER_SHUTDOWN_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class TranscriptTiming:
    """Content-free provider timing and usage counters for one revision."""

    capture_duration_ms: int = 0
    provider_latency_ms: int = 0
    processed_duration_ms: int = 0
    duplicated_duration_ms: int = 0
    usage_units: int = 0

    def __post_init__(self) -> None:
        values = (
            self.capture_duration_ms,
            self.provider_latency_ms,
            self.processed_duration_ms,
            self.duplicated_duration_ms,
            self.usage_units,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("transcript timing values must be integers")
        if any(value < 0 for value in values):
            raise ValueError("transcript timing values must be non-negative")


@dataclass(frozen=True, slots=True)
class TranscriptRevision:
    """One monotonic view of an incremental transcript."""

    turn_id: str
    revision_id: int
    stable_text: str
    revisable_text: str
    covered_through_ns: int
    mode: TranscriptMode
    is_final: bool = False
    failure_code: TranscriptFailureCode | None = None
    timing: TranscriptTiming = TranscriptTiming()

    def __post_init__(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id must be nonempty")
        if self.revision_id < 1:
            raise ValueError("revision_id must be positive")
        if self.covered_through_ns < 0:
            raise ValueError("transcript coverage must be non-negative")
        if self.mode not in ("live", "rolling-window"):
            raise ValueError("unknown transcript mode")
        if self.failure_code not in (None, "backend_failed", "fallback_failed"):
            raise ValueError("unknown transcript failure code")
        if self.failure_code is not None and not self.is_final:
            raise ValueError("a failed transcript revision must be final")


@dataclass(frozen=True, slots=True)
class TranscriptToken:
    """Timestamped display token returned by a rolling provider adapter."""

    text: str
    started_ns: int
    ended_ns: int

    def __post_init__(self) -> None:
        if self.started_ns < 0 or self.ended_ns <= self.started_ns:
            raise ValueError("transcript token timestamps must be positive and ordered")


@dataclass(frozen=True, slots=True)
class TranscriptHypothesis:
    """Provider-neutral input accepted from a native or rolling adapter."""

    stable_text: str = ""
    revisable_text: str = ""
    covered_through_ns: int = 0
    tokens: tuple[TranscriptToken, ...] = ()
    is_final: bool = False
    provider_latency_ms: int = 0
    usage_units: int = 0

    def __post_init__(self) -> None:
        if self.covered_through_ns < 0:
            raise ValueError("hypothesis coverage must be non-negative")
        if self.provider_latency_ms < 0 or self.usage_units < 0:
            raise ValueError("hypothesis timing and usage must be non-negative")
        if self.tokens and (self.stable_text or self.revisable_text):
            raise ValueError("a hypothesis uses timestamped tokens or text, not both")
        if any(token.ended_ns > self.covered_through_ns for token in self.tokens):
            raise ValueError("hypothesis coverage cannot precede any token")
        if any(
            current.started_ns < previous.ended_ns
            for previous, current in zip(self.tokens, self.tokens[1:])
        ):
            raise ValueError("hypothesis tokens must not overlap")


class TranscriptBackendFailure(RuntimeError):
    """Typed terminal failure retaining the user's editable transcript draft."""

    def __init__(self, code: TranscriptFailureCode, editable_draft: str) -> None:
        super().__init__(code)
        self.code = code
        self.editable_draft = editable_draft


class _LiveAdapter(Protocol):
    def start(
        self,
        publish: Callable[[TranscriptHypothesis], None],
        fail: Callable[[BaseException], None],
        settle: Callable[[int], None],
    ) -> None: ...

    def submit(self, frame: AudioFrame) -> None: ...

    async def close(self) -> None: ...


class _RollingAdapter(Protocol):
    async def transcribe_window(
        self, *, pcm16: bytes, started_ns: int, ended_ns: int
    ) -> TranscriptHypothesis: ...

    async def abort(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _Window:
    pcm16: bytes
    started_ns: int
    ended_ns: int
    duration_ns: int
    first_sequence: int
    last_sequence: int
    discarded_admitted_through_ns: int


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def is_material_transcript_change(before: str, after: str) -> bool:
    """Return whether a correction changes content, not its presentation.

    Args:
        before: Transcript text before the correction.
        after: Transcript text after the correction.

    Returns:
        ``True`` when normalized transcript content differs.
    """

    return _normalized_text(before) != _normalized_text(after)


def _pcm_duration_ns(pcm16: bytes) -> int:
    if len(pcm16) % _PCM_BYTES_PER_SAMPLE:
        raise ValueError("PCM16 audio must contain whole samples")
    samples = len(pcm16) // _PCM_BYTES_PER_SAMPLE
    return samples * _NANOSECONDS_PER_SECOND // _PCM_SAMPLE_RATE


def _silence_for_ns(duration_ns: int) -> bytes:
    return bytes(_pcm_bytes_for_ns(duration_ns))


def _pcm_bytes_for_ns(duration_ns: int) -> int:
    samples = (
        duration_ns * _PCM_SAMPLE_RATE + _NANOSECONDS_PER_SECOND // 2
    ) // _NANOSECONDS_PER_SECOND
    return samples * _PCM_BYTES_PER_SAMPLE


class TranscriptEngine:
    """Unify native-live and bounded rolling-window transcript revisions."""

    def __init__(
        self,
        *,
        turn_id: str,
        live_adapter: _LiveAdapter | None = None,
        rolling_adapter: _RollingAdapter | None = None,
        fallback_adapter: _RollingAdapter | None = None,
        frame_tolerance_ns: int = 10_000_000,
        rolling_window_ns: int = 4_000_000_000,
        rolling_min_window_ns: int = 0,
        rolling_debounce_seconds: float = 0.0,
        max_buffered_frames: int = 400,
        max_sequence_entries: int | None = None,
        on_revision: Callable[[TranscriptRevision], None] | None = None,
    ) -> None:
        if not turn_id:
            raise ValueError("turn_id must be nonempty")
        if live_adapter is None and rolling_adapter is None:
            raise ValueError("a live or rolling transcript adapter is required")
        if rolling_adapter is not None and rolling_adapter is fallback_adapter:
            raise ValueError("rolling_adapter and fallback_adapter must be distinct")
        if frame_tolerance_ns < 0:
            raise ValueError("frame_tolerance_ns must be non-negative")
        if rolling_window_ns <= 0 or max_buffered_frames <= 0:
            raise ValueError("rolling bounds must be positive")
        if rolling_window_ns > _MAX_ROLLING_WINDOW_NS:
            raise ValueError("rolling_window_ns exceeds the safe maximum")
        if rolling_min_window_ns < 0 or rolling_debounce_seconds < 0:
            raise ValueError("rolling coalescing bounds must be non-negative")
        if rolling_min_window_ns > rolling_window_ns:
            raise ValueError("rolling minimum cannot exceed its window")
        sequence_limit = (
            max(1_024, max_buffered_frames * 4)
            if max_sequence_entries is None
            else max_sequence_entries
        )
        if sequence_limit <= 0:
            raise ValueError("max_sequence_entries must be positive")
        try:
            owner_loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "TranscriptEngine must be constructed on its owning event loop"
            ) from exc

        self.turn_id = turn_id
        self._owner_loop = owner_loop
        self._live_adapter = live_adapter
        self._fallback_adapter = fallback_adapter
        self._active_rolling_adapter = rolling_adapter
        self._mode: TranscriptMode = (
            "live" if live_adapter is not None else "rolling-window"
        )
        self._frame_tolerance_ns = frame_tolerance_ns
        self._rolling_window_ns = rolling_window_ns
        self._rolling_min_window_ns = rolling_min_window_ns
        self._rolling_debounce_seconds = rolling_debounce_seconds
        self._frames: deque[AudioFrame] = deque(maxlen=max_buffered_frames)
        self._max_sequence_entries = sequence_limit
        self._sequence_coverage: dict[int, int] = {}
        self._sequence_pcm_cumulative: dict[int, int] = {}
        self._compacted_sequences: deque[int] = deque()
        self._compacted_sequence_set: set[int] = set()
        self._compacted_pcm_duration_ns = 0
        self._evicted_admitted_through_ns = 0
        self._on_revision = on_revision

        self._latest_revision: TranscriptRevision | None = None
        self._revision_id = 0
        self._capture_started_ns: int | None = None
        self._submitted_pcm_duration_ns = 0
        self._processed_duration_ns = 0
        self._duplicated_duration_ns = 0
        self._processed_through_ns = 0
        self._stable_tokens: list[TranscriptToken] = []
        self._tail_tokens: list[TranscriptToken] = []
        self._rolling_stable_prefix = ""
        self._rolling_fallback_base_coverage_ns = 0
        self._rolling_fallback_editable_base = ""
        self._rolling_fallback_proven_coverage_ns = 0
        self._rolling_fallback_first_result_pending = False
        self._rolling_has_native_base = False

        self._pending_window: _Window | None = None
        self._active_window: _Window | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._seal_waiters: list[
            tuple[int, int, asyncio.Future[TranscriptRevision]]
        ] = []

        self._active_epoch = 0
        self._fallback_attempted = False
        self._live_newest_coverage_ns = 0
        self._live_submitted_sequences: set[int] = set()
        self._live_settled_sequence = -1
        self._live_attempt_base_coverage_ns = 0
        self._live_attempt_base_text = ""
        self._terminal_failure: TranscriptBackendFailure | None = None
        self._automatic_dispatch_suspended = False
        self._status: Literal["transcribing", "ready", "failed"] = "transcribing"

        self._closed = False
        self._close_complete = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._supervised_shutdown_tasks: set[asyncio.Future[object]] = set()
        self._retiring_rolling_adapters: dict[asyncio.Future[object], object] = {}

        if self._live_adapter is not None:
            self._start_live_attempt()
        else:
            self._active_epoch = 1

    @property
    def latest_revision(self) -> TranscriptRevision | None:
        return self._latest_revision

    @property
    def editable_draft(self) -> str:
        if self._latest_revision is None:
            return ""
        return self._latest_revision.stable_text + self._latest_revision.revisable_text

    @property
    def automatic_dispatch_suspended(self) -> bool:
        return self._automatic_dispatch_suspended

    @property
    def status(self) -> str:
        return self._status

    @property
    def supervised_shutdown_task_count(self) -> int:
        """Return the content-free count of provider tasks still supervised."""

        return len(self._supervised_shutdown_tasks)

    def append_admitted_frame(self, frame: AudioFrame) -> None:
        """Append one post-AEC/VAD-admitted frame on the owner loop."""

        self._require_owner_loop()
        if self._closed:
            raise RuntimeError("transcript engine is closed")
        if self._sequence_is_known(frame.sequence):
            return

        self._compact_sequence_coverage()
        if len(self._sequence_coverage) >= self._max_sequence_entries:
            self._fail_capacity_closed()
            return
        if self._capture_started_ns is None:
            self._capture_started_ns = frame.started_ns
        frame_duration_ns = _pcm_duration_ns(frame.pcm16)
        if abs(frame_duration_ns - frame.duration_ns) > (
            _NANOSECONDS_PER_SECOND // _PCM_SAMPLE_RATE
        ):
            self._handle_adapter_failure(
                self._active_epoch,
                ValueError("admitted frame PCM duration disagrees with timestamps"),
            )
            return
        if len(self._frames) == self._frames.maxlen:
            self._evicted_admitted_through_ns = max(
                self._evicted_admitted_through_ns,
                self._frames[0].ended_ns,
            )
        self._frames.append(frame)
        self._sequence_coverage[frame.sequence] = frame.ended_ns
        self._submitted_pcm_duration_ns += frame_duration_ns
        self._sequence_pcm_cumulative[frame.sequence] = self._submitted_pcm_duration_ns

        resumed_from_failure = self._terminal_failure is not None
        live_retry_base: tuple[int, str] | None = None
        if resumed_from_failure:
            if frame.ended_ns <= self._terminal_failure_coverage():
                return
            if self._mode == "live" and self._live_adapter is not None:
                live_retry_base = self._live_retry_base()
                if live_retry_base is None:
                    return
            self._resume_after_terminal()

        if self._mode == "live" and self._live_adapter is not None:
            if resumed_from_failure:
                assert live_retry_base is not None
                self._start_live_attempt(
                    base_coverage_ns=live_retry_base[0],
                    base_text=live_retry_base[1],
                )
                return
            self._submit_live_frame(frame, self._active_epoch)
            return

        self._queue_newest_window()

    def fresh_for(self, speech_end_ns: int) -> bool:
        """Return whether the current healthy revision covers the speech tail."""

        revision = self._latest_revision
        return bool(
            not self._closed
            and self._terminal_failure is None
            and revision is not None
            and revision.failure_code is None
            and revision.covered_through_ns + self._frame_tolerance_ns >= speech_end_ns
        )

    async def seal_through(self, admitted_sequence: int) -> TranscriptRevision:
        """Wait for coverage and all work derived from a sequence to settle."""

        self._require_owner_loop()
        if self._terminal_failure is not None:
            raise self._terminal_failure
        if self._sequence_was_compacted(admitted_sequence):
            if self._can_seal(admitted_sequence, 0):
                assert self._latest_revision is not None
                return self._latest_revision
            future = self._owner_loop.create_future()
            self._seal_waiters.append((admitted_sequence, 0, future))
            return await future
        if admitted_sequence not in self._sequence_coverage:
            raise ValueError("cannot seal an unknown admitted sequence")

        target_ns = self._sequence_coverage[admitted_sequence]
        if self._can_seal(admitted_sequence, target_ns):
            assert self._latest_revision is not None
            return self._latest_revision

        future: asyncio.Future[TranscriptRevision] = self._owner_loop.create_future()
        self._seal_waiters.append((admitted_sequence, target_ns, future))
        return await future

    async def wait_idle(self) -> None:
        """Wait until the rolling provider has no in-flight or pending work."""

        self._require_owner_loop()
        while self._runner_task is not None:
            task = self._runner_task
            await task
            if task is self._runner_task:
                break

    def manual_retry(self) -> None:
        """Explicitly resume transcription after a terminal adapter failure."""

        self._require_owner_loop()
        if self._closed:
            raise RuntimeError("transcript engine is closed")
        if self._terminal_failure is None:
            self._automatic_dispatch_suspended = False
            return
        if self._mode == "rolling-window":
            self._resume_after_terminal()
            self._active_epoch += 1
            self._queue_newest_window()
        elif self._live_adapter is not None:
            live_retry_base = self._live_retry_base()
            if live_retry_base is None:
                return
            self._resume_after_terminal()
            self._start_live_attempt(
                base_coverage_ns=live_retry_base[0],
                base_text=live_retry_base[1],
            )

    async def close(self) -> None:
        """Fence callbacks, stop provider work, and release buffered audio."""

        self._require_owner_loop()
        async with self._close_lock:
            if self._close_complete:
                return
            if self._close_task is None:
                self._close_task = self._owner_loop.create_task(self._close_owned())
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _close_owned(self) -> None:
        self._closed = True
        self._active_epoch += 1
        self._pending_window = None
        self._terminal_failure = TranscriptBackendFailure(
            "backend_failed", self.editable_draft
        )
        self._automatic_dispatch_suspended = True
        self._status = "failed"
        self._resolve_seals()
        shutdown_deadline = self._owner_loop.time() + _ADAPTER_SHUTDOWN_TIMEOUT_SECONDS
        runner = self._runner_task
        if runner is not None:
            self._supervise_shutdown_task(runner)
        try:
            await self._bounded_adapter_shutdown(
                self._active_rolling_adapter, "abort", shutdown_deadline
            )
            if runner is not None:
                runner.cancel()
                await self._bounded_task_shutdown(runner, shutdown_deadline)
            await self._bounded_adapter_shutdown(
                self._live_adapter, "close", shutdown_deadline
            )
        finally:
            self._runner_task = None
            self._active_window = None
            self._pending_window = None
            self._frames.clear()
            self._sequence_coverage.clear()
            self._sequence_pcm_cumulative.clear()
            self._compacted_sequences.clear()
            self._compacted_sequence_set.clear()
            self._live_submitted_sequences.clear()
            self._stable_tokens.clear()
            self._tail_tokens.clear()
            self._close_complete = True

    async def _bounded_adapter_shutdown(
        self,
        adapter: object | None,
        method_name: str,
        deadline: float,
        *,
        retain_adapter_until_done: bool = False,
    ) -> asyncio.Future[object] | None:
        method = getattr(adapter, method_name, None)
        if method is None:
            return None
        try:
            result = method()
        except Exception:
            return None
        if not inspect.isawaitable(result):
            return None
        task = asyncio.ensure_future(result)
        if retain_adapter_until_done:
            self._retiring_rolling_adapters[task] = adapter
            task.add_done_callback(self._release_retiring_rolling_adapter)
        self._supervise_shutdown_task(task)
        await self._bounded_task_shutdown(task, deadline)
        return task

    async def _retire_rolling_adapter(self, adapter: object) -> None:
        deadline = self._owner_loop.time() + _ADAPTER_SHUTDOWN_TIMEOUT_SECONDS
        await self._bounded_adapter_shutdown(
            adapter,
            "abort",
            deadline,
            retain_adapter_until_done=True,
        )

    def _release_retiring_rolling_adapter(self, task: asyncio.Future[object]) -> None:
        self._retiring_rolling_adapters.pop(task, None)

    async def _bounded_task_shutdown(
        self, task: asyncio.Future[object], deadline: float
    ) -> None:
        self._supervise_shutdown_task(task)
        remaining = max(0.0, deadline - self._owner_loop.time())
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if task not in done:
            task.cancel()
            return
        self._release_supervised_shutdown_task(task)

    def _supervise_shutdown_task(self, task: asyncio.Future[object]) -> None:
        if task in self._supervised_shutdown_tasks:
            return
        self._supervised_shutdown_tasks.add(task)
        task.add_done_callback(self._release_supervised_shutdown_task)

    def _release_supervised_shutdown_task(self, task: asyncio.Future[object]) -> None:
        self._supervised_shutdown_tasks.discard(task)
        self._consume_shutdown_result(task)

    @staticmethod
    def _consume_shutdown_result(task: asyncio.Future[object]) -> None:
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            return

    def _require_owner_loop(self) -> None:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError("operation requires the transcript owner loop") from exc
        if current_loop is not self._owner_loop:
            raise RuntimeError("operation ran outside the transcript owner loop")

    def _post_owner(self, callback: Callable[..., None], *args: object) -> None:
        try:
            self._owner_loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            # The owner loop has already shut down; the fenced callback has no
            # safe recipient and must not mutate state elsewhere.
            return

    def _live_retry_base(self) -> tuple[int, str] | None:
        revision = self._latest_revision
        base_coverage_ns = revision.covered_through_ns if revision is not None else 0
        base_text = (
            revision.stable_text + revision.revisable_text
            if revision is not None
            else ""
        )
        if self._evicted_admitted_through_ns > base_coverage_ns:
            return None
        suffix_frames = [
            frame for frame in self._frames if frame.ended_ns > base_coverage_ns
        ]
        if not suffix_frames or suffix_frames[0].started_ns < base_coverage_ns:
            return None
        return base_coverage_ns, base_text

    def _start_live_attempt(
        self, *, base_coverage_ns: int = 0, base_text: str = ""
    ) -> None:
        if self._live_adapter is None or self._closed:
            return
        self._active_epoch += 1
        epoch = self._active_epoch
        self._live_attempt_base_coverage_ns = base_coverage_ns
        self._live_attempt_base_text = base_text
        self._live_newest_coverage_ns = 0
        self._live_submitted_sequences.clear()
        self._live_settled_sequence = -1
        try:
            self._live_adapter.start(
                lambda hypothesis: self._post_owner(
                    self._receive_live_hypothesis, epoch, hypothesis
                ),
                lambda failure: self._post_owner(
                    self._receive_adapter_failure, epoch, failure
                ),
                lambda sequence: self._post_owner(
                    self._receive_live_settlement, epoch, sequence
                ),
            )
        except Exception as exc:
            self._handle_adapter_failure(epoch, exc)
            return
        for frame in tuple(self._frames):
            if not self._epoch_is_active(epoch) or self._mode != "live":
                return
            if frame.ended_ns <= base_coverage_ns:
                continue
            self._submit_live_frame(frame, epoch)

    def _submit_live_frame(self, frame: AudioFrame, epoch: int) -> None:
        if (
            self._live_adapter is None
            or not self._epoch_is_active(epoch)
            or self._mode != "live"
        ):
            return
        self._live_newest_coverage_ns = max(
            self._live_newest_coverage_ns, frame.ended_ns
        )
        self._live_submitted_sequences.add(frame.sequence)
        try:
            self._live_adapter.submit(frame)
        except Exception as exc:
            self._handle_adapter_failure(epoch, exc)

    def _receive_live_hypothesis(
        self, epoch: int, hypothesis: TranscriptHypothesis
    ) -> None:
        if not self._epoch_is_active(epoch) or self._mode != "live":
            return
        if hypothesis.covered_through_ns > self._live_newest_coverage_ns:
            self._handle_adapter_failure(
                epoch, ValueError("live transcript coverage exceeds admitted audio")
            )
            return
        if hypothesis.covered_through_ns < self._live_attempt_base_coverage_ns:
            self._handle_adapter_failure(
                epoch, ValueError("live transcript coverage precedes its retry base")
            )
            return
        try:
            self._accept_text_hypothesis(hypothesis, mode="live")
        except (TypeError, ValueError) as exc:
            self._handle_adapter_failure(epoch, exc)

    def _receive_live_settlement(self, epoch: int, sequence: int) -> None:
        if not self._epoch_is_active(epoch) or self._mode != "live":
            return
        if sequence <= self._live_settled_sequence:
            return
        if sequence not in self._live_submitted_sequences:
            self._handle_adapter_failure(
                epoch, ValueError("live settlement exceeds submitted audio")
            )
            return
        self._live_settled_sequence = max(self._live_settled_sequence, sequence)
        self._resolve_seals()

    def _receive_adapter_failure(self, epoch: int, failure: BaseException) -> None:
        self._handle_adapter_failure(epoch, failure)

    def _accept_text_hypothesis(
        self, hypothesis: TranscriptHypothesis, *, mode: TranscriptMode
    ) -> None:
        if hypothesis.tokens:
            raise ValueError("native transcript hypotheses must use stable/tail text")
        stable_text = hypothesis.stable_text
        if mode == "live":
            stable_text = self._live_attempt_base_text + stable_text
        if mode == "live":
            self._processed_duration_ns = max(
                self._processed_duration_ns,
                self._pcm_duration_through(hypothesis.covered_through_ns),
            )
        self._publish_revision(
            stable_text=stable_text,
            revisable_text=hypothesis.revisable_text,
            covered_through_ns=hypothesis.covered_through_ns,
            mode=mode,
            is_final=hypothesis.is_final,
            provider_latency_ms=hypothesis.provider_latency_ms,
            usage_units=hypothesis.usage_units,
        )

    def _pcm_duration_through(self, covered_through_ns: int) -> int:
        duration_ns = self._compacted_pcm_duration_ns
        for sequence, ended_ns in self._sequence_coverage.items():
            if ended_ns <= covered_through_ns:
                duration_ns = self._sequence_pcm_cumulative[sequence]
        return duration_ns

    def _queue_newest_window(self) -> None:
        if (
            self._closed
            or self._terminal_failure is not None
            or self._active_rolling_adapter is None
            or not self._frames
        ):
            return
        try:
            self._pending_window = self._make_window()
        except (TypeError, ValueError) as exc:
            self._handle_adapter_failure(self._active_epoch, exc)
            return
        self._ensure_runner()

    def _make_window(self) -> _Window:
        latest_end = self._frames[-1].ended_ns
        earliest_start = latest_end - self._rolling_window_ns
        frames = [frame for frame in self._frames if frame.ended_ns > earliest_start]
        discarded_admitted_through_ns = self._evicted_admitted_through_ns
        for frame in self._frames:
            if frame.ended_ns > earliest_start:
                break
            discarded_admitted_through_ns = max(
                discarded_admitted_through_ns, frame.ended_ns
            )
        for previous, current in zip(frames, frames[1:]):
            gap_ns = current.started_ns - previous.ended_ns
            if gap_ns > self._rolling_window_ns:
                raise ValueError("admitted audio gap exceeds the safe rolling window")
        parts: list[bytes] = []
        pcm_bytes = 0
        cursor_ns = frames[0].started_ns
        for frame in frames:
            # Device timestamps are not on the PCM sample grid. Apply the
            # same nearest-sample rounding to tiny negative gaps as silence
            # insertion already applies to positive gaps; never drop audio.
            if frame.started_ns < cursor_ns and _pcm_bytes_for_ns(
                cursor_ns - frame.started_ns
            ):
                raise ValueError("admitted audio frames overlap or move backward")
            gap_ns = max(0, frame.started_ns - cursor_ns)
            if gap_ns:
                silence_bytes = _pcm_bytes_for_ns(gap_ns)
                if pcm_bytes + silence_bytes > _MAX_ROLLING_PCM_BYTES:
                    raise ValueError("rolling PCM exceeds the safe byte ceiling")
                silence = _silence_for_ns(gap_ns)
                parts.append(silence)
                pcm_bytes += len(silence)
                cursor_ns += _pcm_duration_ns(silence)
            frame_duration_ns = _pcm_duration_ns(frame.pcm16)
            if abs(frame_duration_ns - frame.duration_ns) > (
                _NANOSECONDS_PER_SECOND // _PCM_SAMPLE_RATE
            ):
                raise ValueError(
                    "admitted frame PCM duration disagrees with timestamps"
                )
            if pcm_bytes + len(frame.pcm16) > _MAX_ROLLING_PCM_BYTES:
                raise ValueError("rolling PCM exceeds the safe byte ceiling")
            parts.append(frame.pcm16)
            pcm_bytes += len(frame.pcm16)
            cursor_ns += frame_duration_ns

        pcm16 = b"".join(parts)
        duration_ns = _pcm_duration_ns(pcm16)
        return _Window(
            pcm16=pcm16,
            started_ns=frames[0].started_ns,
            ended_ns=frames[0].started_ns + duration_ns,
            duration_ns=duration_ns,
            first_sequence=frames[0].sequence,
            last_sequence=frames[-1].sequence,
            discarded_admitted_through_ns=discarded_admitted_through_ns,
        )

    def _ensure_runner(self) -> None:
        if self._runner_task is not None and not self._runner_task.done():
            return
        self._runner_task = self._owner_loop.create_task(self._run_rolling())

    async def _run_rolling(self) -> None:
        try:
            while self._pending_window is not None and not self._closed:
                window = self._pending_window
                self._pending_window = None
                adapter = self._active_rolling_adapter
                if adapter is None:
                    return
                epoch = self._active_epoch
                self._active_window = window
                try:
                    if (
                        window.duration_ns < self._rolling_min_window_ns
                        and self._rolling_debounce_seconds > 0
                    ):
                        await asyncio.sleep(self._rolling_debounce_seconds)
                        if not self._epoch_is_active(epoch):
                            continue
                        if self._pending_window is not None:
                            continue
                    hypothesis = await adapter.transcribe_window(
                        pcm16=window.pcm16,
                        started_ns=window.started_ns,
                        ended_ns=window.ended_ns,
                    )
                    if not self._epoch_is_active(epoch):
                        continue
                    if hypothesis.covered_through_ns != window.ended_ns:
                        raise ValueError(
                            "rolling transcript must cover its full request window"
                        )
                    if any(
                        token.started_ns < window.started_ns - self._frame_tolerance_ns
                        for token in hypothesis.tokens
                    ):
                        raise ValueError(
                            "rolling transcript token precedes its request window"
                        )
                    self._accept_rolling(hypothesis, window)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    outcome = self._handle_adapter_failure(epoch, exc)
                    if outcome != "fallback":
                        return
                    await self._retire_rolling_adapter(adapter)
                finally:
                    self._active_window = None
                    self._resolve_seals()
        finally:
            self._active_window = None
            self._runner_task = None
            self._resolve_seals()

    def _accept_rolling(
        self, hypothesis: TranscriptHypothesis, window: _Window
    ) -> None:
        proven_coverage_ns = (
            self._rolling_fallback_proven_coverage_ns
            if self._rolling_has_native_base
            else (
                self._latest_revision.covered_through_ns
                if self._latest_revision is not None
                else 0
            )
        )
        if (
            window.started_ns > proven_coverage_ns
            and window.discarded_admitted_through_ns > proven_coverage_ns
        ):
            self._fail_capacity_closed()
            return
        if (
            self._rolling_fallback_first_result_pending
            and window.started_ns >= self._rolling_fallback_base_coverage_ns
        ):
            self._rolling_stable_prefix = self._rolling_fallback_editable_base
            self._stable_tokens.clear()
            self._tail_tokens.clear()

        overlap_ns = max(
            0,
            min(window.ended_ns, self._processed_through_ns) - window.started_ns,
        )
        self._duplicated_duration_ns += overlap_ns
        self._processed_duration_ns += window.duration_ns
        self._processed_through_ns = max(self._processed_through_ns, window.ended_ns)

        if not hypothesis.tokens:
            previous = self._latest_revision
            if previous is None:
                self._accept_text_hypothesis(hypothesis, mode="rolling-window")
                self._mark_rolling_fallback_accepted(hypothesis)
                return
            if window.started_ns < previous.covered_through_ns:
                raise ValueError(
                    "overlapping rolling revisions require timestamped tokens"
                )
            self._publish_revision(
                stable_text=(
                    previous.stable_text
                    + previous.revisable_text
                    + hypothesis.stable_text
                ),
                revisable_text=hypothesis.revisable_text,
                covered_through_ns=hypothesis.covered_through_ns,
                mode="rolling-window",
                is_final=hypothesis.is_final,
                provider_latency_ms=hypothesis.provider_latency_ms,
                usage_units=hypothesis.usage_units,
            )
            self._mark_rolling_fallback_accepted(hypothesis)
            return

        new_tail = [
            token for token in hypothesis.tokens if token.ended_ns > window.started_ns
        ]
        if (
            _normalized_text(self._rolling_stable_prefix)
            and window.started_ns < self._rolling_fallback_base_coverage_ns
        ):
            consumed = self._matched_rolling_prefix_tokens(new_tail)
            new_tail = new_tail[consumed:]
        promotable = [
            token for token in self._tail_tokens if token.ended_ns <= window.started_ns
        ]
        aligned_count = self._boundary_replacement_count(promotable, new_tail)
        if aligned_count:
            promotable = promotable[:-aligned_count]
        self._stable_tokens.extend(promotable)
        self._tail_tokens = new_tail
        self._publish_revision(
            stable_text=self._rolling_stable_prefix
            + "".join(token.text for token in self._stable_tokens),
            revisable_text="".join(token.text for token in self._tail_tokens),
            covered_through_ns=hypothesis.covered_through_ns,
            mode="rolling-window",
            is_final=hypothesis.is_final,
            provider_latency_ms=hypothesis.provider_latency_ms,
            usage_units=hypothesis.usage_units,
        )
        self._mark_rolling_fallback_accepted(hypothesis)

    def _mark_rolling_fallback_accepted(self, hypothesis: TranscriptHypothesis) -> None:
        if not self._rolling_has_native_base:
            return
        self._rolling_fallback_first_result_pending = False
        self._rolling_fallback_proven_coverage_ns = hypothesis.covered_through_ns

    def _matched_rolling_prefix_tokens(self, tokens: list[TranscriptToken]) -> int:
        target = _normalized_text(self._rolling_stable_prefix)
        matched = ""
        for index, token in enumerate(tokens, start=1):
            matched += _normalized_text(token.text)
            if matched == target:
                return index
            if not target.startswith(matched):
                break
        raise ValueError("rolling fallback does not preserve native stable prefix")

    def _boundary_replacement_count(
        self,
        old_tokens: list[TranscriptToken],
        new_tokens: list[TranscriptToken],
    ) -> int:
        replaced = 0
        for old in reversed(old_tokens):
            if not any(
                self._tokens_are_temporally_aligned(old, new) for new in new_tokens
            ):
                break
            replaced += 1
        return replaced

    def _tokens_are_temporally_aligned(
        self, old: TranscriptToken, new: TranscriptToken
    ) -> bool:
        overlaps = new.started_ns < old.ended_ns and old.started_ns < new.ended_ns
        same_normalized_token = bool(
            _normalized_text(old.text)
            and _normalized_text(old.text) == _normalized_text(new.text)
        )
        within_boundary_tolerance = not (
            new.started_ns > old.ended_ns + self._frame_tolerance_ns
            or old.started_ns > new.ended_ns + self._frame_tolerance_ns
        )
        return overlaps or (same_normalized_token and within_boundary_tolerance)

    def _publish_revision(
        self,
        *,
        stable_text: str,
        revisable_text: str,
        covered_through_ns: int,
        mode: TranscriptMode,
        is_final: bool,
        provider_latency_ms: int,
        usage_units: int,
    ) -> None:
        previous = self._latest_revision
        if previous is not None:
            if not stable_text.startswith(previous.stable_text):
                raise ValueError("transcript stable prefix is immutable")
            if covered_through_ns < previous.covered_through_ns:
                raise ValueError("transcript coverage must be monotonic")

        self._revision_id += 1
        capture_start = self._capture_started_ns or 0
        capture_duration_ns = max(0, covered_through_ns - capture_start)
        revision = TranscriptRevision(
            turn_id=self.turn_id,
            revision_id=self._revision_id,
            stable_text=stable_text,
            revisable_text=revisable_text,
            covered_through_ns=covered_through_ns,
            mode=mode,
            is_final=is_final,
            timing=TranscriptTiming(
                capture_duration_ms=capture_duration_ns // 1_000_000,
                provider_latency_ms=provider_latency_ms,
                processed_duration_ms=self._processed_duration_ns // 1_000_000,
                duplicated_duration_ms=self._duplicated_duration_ns // 1_000_000,
                usage_units=usage_units,
            ),
        )
        self._latest_revision = revision
        self._status = "ready" if is_final else "transcribing"
        self._resolve_seals()
        self._notify_revision(revision)

    def _notify_revision(self, revision: TranscriptRevision) -> None:
        if self._on_revision is None:
            return
        try:
            self._on_revision(revision)
        except Exception:
            # Observers are outside the causal transcript state machine. Their
            # failure must not be mistaken for a provider failure.
            return

    def _handle_adapter_failure(
        self, epoch: int, failure: BaseException
    ) -> Literal["fallback", "terminal", "ignored"]:
        if not self._epoch_is_active(epoch) or self._terminal_failure is not None:
            return "ignored"
        from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetServiceUnavailable
        from tldw_chatbook.Audio.voice_process_types import VoiceTranscriptCapacityError

        if (
            self._fallback_adapter is not None
            and not self._fallback_attempted
            and not isinstance(
                failure, (ParakeetServiceUnavailable, VoiceTranscriptCapacityError)
            )
        ):
            self._fallback_attempted = True
            self._active_epoch += 1
            self._active_rolling_adapter = self._fallback_adapter
            if self._mode == "live":
                self._rolling_stable_prefix = (
                    self._latest_revision.stable_text
                    if self._latest_revision is not None
                    else ""
                )
                self._rolling_fallback_base_coverage_ns = (
                    self._latest_revision.covered_through_ns
                    if self._latest_revision is not None
                    else 0
                )
                self._rolling_fallback_editable_base = (
                    self._latest_revision.stable_text
                    + self._latest_revision.revisable_text
                    if self._latest_revision is not None
                    else ""
                )
                self._rolling_fallback_proven_coverage_ns = (
                    self._rolling_fallback_base_coverage_ns
                )
                self._rolling_fallback_first_result_pending = True
                self._rolling_has_native_base = True
                self._stable_tokens.clear()
                self._tail_tokens.clear()
            self._mode = "rolling-window"
            self._status = "transcribing"
            self._queue_newest_window()
            return "fallback"

        self._active_epoch += 1
        code: TranscriptFailureCode = (
            "fallback_failed" if self._fallback_attempted else "backend_failed"
        )
        self._publish_terminal_failure(code)
        return "terminal"

    def _publish_terminal_failure(self, code: TranscriptFailureCode) -> None:
        stable = self._latest_revision.stable_text if self._latest_revision else ""
        tail = self._latest_revision.revisable_text if self._latest_revision else ""
        coverage = (
            self._latest_revision.covered_through_ns if self._latest_revision else 0
        )
        self._revision_id += 1
        revision = TranscriptRevision(
            turn_id=self.turn_id,
            revision_id=self._revision_id,
            stable_text=stable,
            revisable_text=tail,
            covered_through_ns=coverage,
            mode=self._mode,
            is_final=True,
            failure_code=code,
            timing=self._latest_revision.timing
            if self._latest_revision
            else TranscriptTiming(),
        )
        self._latest_revision = revision
        self._terminal_failure = TranscriptBackendFailure(code, stable + tail)
        self._automatic_dispatch_suspended = True
        self._status = "failed"
        self._pending_window = None
        self._resolve_seals()
        self._notify_revision(revision)

    def _terminal_failure_coverage(self) -> int:
        if self._latest_revision is None:
            return 0
        return self._latest_revision.covered_through_ns

    def _resume_after_terminal(self) -> None:
        self._terminal_failure = None
        self._automatic_dispatch_suspended = False
        self._status = "transcribing"
        self._pending_window = None

    def _epoch_is_active(self, epoch: int) -> bool:
        return not self._closed and epoch == self._active_epoch

    def _can_seal(self, sequence: int, target_ns: int) -> bool:
        revision = self._latest_revision
        if (
            revision is None
            or revision.failure_code is not None
            or revision.covered_through_ns < target_ns
        ):
            return False
        if self._mode == "live":
            return self._live_settled_sequence >= sequence
        return not self._rolling_work_contains(sequence)

    def _rolling_work_contains(self, sequence: int) -> bool:
        return any(
            window is not None
            and window.first_sequence <= sequence <= window.last_sequence
            for window in (self._active_window, self._pending_window)
        )

    def _resolve_seals(self) -> None:
        remaining: list[tuple[int, int, asyncio.Future[TranscriptRevision]]] = []
        for sequence, target_ns, future in self._seal_waiters:
            if future.done():
                continue
            if self._terminal_failure is not None:
                future.set_exception(self._terminal_failure)
            elif self._can_seal(sequence, target_ns):
                assert self._latest_revision is not None
                future.set_result(self._latest_revision)
            else:
                remaining.append((sequence, target_ns, future))
        self._seal_waiters = remaining
        self._compact_sequence_coverage()

    def _compact_sequence_coverage(self) -> None:
        revision = self._latest_revision
        if revision is None or self._terminal_failure is not None:
            return
        while self._sequence_coverage:
            sequence = next(iter(self._sequence_coverage))
            target_ns = self._sequence_coverage[sequence]
            if not self._can_seal(sequence, target_ns):
                break
            del self._sequence_coverage[sequence]
            self._compacted_pcm_duration_ns = self._sequence_pcm_cumulative.pop(
                sequence
            )
            self._live_submitted_sequences.discard(sequence)
            if len(self._compacted_sequences) >= self._max_sequence_entries:
                evicted = self._compacted_sequences.popleft()
                self._compacted_sequence_set.discard(evicted)
            self._compacted_sequences.append(sequence)
            self._compacted_sequence_set.add(sequence)

    def _sequence_is_known(self, sequence: int) -> bool:
        return sequence in self._sequence_coverage or self._sequence_was_compacted(
            sequence
        )

    def _sequence_was_compacted(self, sequence: int) -> bool:
        return sequence in self._compacted_sequence_set

    def _fail_capacity_closed(self) -> None:
        if self._terminal_failure is not None:
            return
        self._active_epoch += 1
        code: TranscriptFailureCode = (
            "fallback_failed" if self._fallback_attempted else "backend_failed"
        )
        self._publish_terminal_failure(code)
