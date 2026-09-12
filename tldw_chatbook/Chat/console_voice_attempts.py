"""Attempt-local speculative provider generation and bounded cleanup."""

from __future__ import annotations

import asyncio
import threading
import contextlib
import inspect
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol, cast

from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome
from tldw_chatbook.Chat.console_prepared_request import (
    PreparedProviderRequest,
    freeze_json,
)
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderCallPurpose,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
    ProviderToolCalls,
    iter_voice_visible_blocks,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
)
from tldw_chatbook.Chat.console_voice_supervisor import (
    VoiceDispatchQuarantined,
    VoiceDispatchSupervisor,
)
from tldw_chatbook.Chat.console_voice_trace_gateway import (
    ProvisionalTraceAttempt,
    ProvisionalTraceEnvelope,
    ProvisionalTraceManifest,
    ProvisionalTraceUnavailable,
)


_WaitForExit = Callable[[asyncio.Task[Any], float], Awaitable[bool]]
_MAX_OBSOLETE_CLEANUPS = 2
_NO_TTS_CANCELLATION_RESULT = object()


def _reject_epoch(_epoch: int) -> bool:
    return False


def _reject_signal_event() -> bool:
    return False


def _reject_provider_work(
    _completion: asyncio.Future[Any],
    _force_close: Callable[[], Any],
) -> bool:
    return False


class _ProviderGateway(Protocol):
    def stream_chat(
        self,
        resolution: ConsoleProviderResolution,
        messages: PreparedProviderRequest,
        *,
        tools: object,
        signals: ConsoleProviderStreamSignals,
        dispatch_purpose: ConsoleProviderCallPurpose,
    ) -> Any: ...

    def seal_provisional_voice_trace(
        self,
        attempt: ProvisionalTraceAttempt,
    ) -> tuple[ProvisionalTraceManifest, tuple[ProvisionalTraceEnvelope, ...]]: ...

    def abandon_provisional_voice_trace(
        self,
        attempt: ProvisionalTraceAttempt,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class VoiceAttemptRequest:
    """One exact immutable provider request bound to an attempt epoch."""

    attempt_epoch: int
    resolution: ConsoleProviderResolution = field(repr=False)
    prepared: PreparedProviderRequest = field(repr=False)
    exchange_capture_enabled: bool = False
    capture_detail: CaptureDetail = field(default=CaptureDetail.SAFE, repr=False)
    provisional_trace_attempt: ProvisionalTraceAttempt | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.attempt_epoch) is not int or self.attempt_epoch < 0:
            raise ValueError("attempt_epoch must be a non-negative integer")
        if not isinstance(self.resolution, ConsoleProviderResolution):
            raise TypeError("resolution must be a ConsoleProviderResolution")
        if not isinstance(self.prepared, PreparedProviderRequest):
            raise TypeError("prepared must be a PreparedProviderRequest")
        if not isinstance(self.capture_detail, CaptureDetail):
            raise TypeError("capture_detail must be a CaptureDetail")
        if (
            self.provisional_trace_attempt is not None
            and type(self.provisional_trace_attempt) is not ProvisionalTraceAttempt
        ):
            raise TypeError("provisional_trace_attempt must be gateway-issued or None")


@dataclass(frozen=True, slots=True)
class VoiceAttemptDelta:
    """Current-epoch provisional response text."""

    attempt_epoch: int
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class VoiceAttemptToolRequest:
    """First complete inert native tool request observed by an attempt."""

    attempt_epoch: int
    tool_calls: tuple[Mapping[str, Any], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ProviderAttemptFailed:
    """Content-free ordinary provider terminal failure."""

    attempt_epoch: int
    error_class: str


@dataclass(frozen=True, slots=True)
class VoiceAttemptSnapshot:
    """Defensive attempt-local output snapshot for later winning promotion."""

    attempt_epoch: int
    response_text: str = field(repr=False)
    tool_request: VoiceAttemptToolRequest | None = field(default=None, repr=False)
    speech_frozen: bool = False
    usage_payloads: tuple[Mapping[str, object], ...] = field(default=(), repr=False)
    trace_manifest: ProvisionalTraceManifest | None = field(default=None, repr=False)
    trace_envelopes: tuple[ProvisionalTraceEnvelope, ...] = field(
        default=(),
        repr=False,
    )


class _CleanupObserver(asyncio.Future[AttemptCleanupOutcome]):
    """Caller-owned view whose cancellation cannot reach owned cleanup."""

    def __init__(self, cleanup: asyncio.Future[AttemptCleanupOutcome]) -> None:
        super().__init__()
        cleanup.add_done_callback(self._complete_from)

    def _complete_from(self, cleanup: asyncio.Future[AttemptCleanupOutcome]) -> None:
        if cleanup.cancelled():
            if not self.done():
                self.cancel()
            return
        error = cleanup.exception()
        if self.done():
            return
        if error is not None:
            self.set_exception(error)
            return
        self.set_result(cleanup.result())


@dataclass(frozen=True, slots=True)
class _CleanupState:
    """Shared caller-visible terminal outcome for one attempt."""

    outcome: asyncio.Future[AttemptCleanupOutcome]


class VoiceCleanupCapacityExceeded(RuntimeError):
    """Content-free refusal to overlap more obsolete cleanups."""

    code = "voice_cleanup_capacity_reached"
    recoverable = True

    def __init__(self) -> None:
        super().__init__(self.code)


class _AttemptProviderSignals(ConsoleProviderStreamSignals):
    """Provider signals whose callback admission is owned by one attempt."""

    __slots__ = (
        "_attempt_is_current",
        "_provider_work_callback",
        "_trace_observation_lock",
        "_trace_accumulator",
        "_trace_discarded",
    )

    def __init__(
        self,
        *,
        exchange_capture_enabled: bool,
        capture_detail: CaptureDetail,
        attempt_is_current: Callable[[], bool],
        provider_work_callback: Callable[
            [asyncio.Future[Any], Callable[[], Any]], bool
        ],
    ) -> None:
        super().__init__(
            exchange_capture_enabled=exchange_capture_enabled,
            capture_detail=capture_detail,
        )
        self._attempt_is_current = attempt_is_current
        self._provider_work_callback = provider_work_callback
        self._trace_observation_lock = threading.RLock()
        self._trace_accumulator = None
        self._trace_discarded = False

    def observe_trace_response(self, accumulator, item, *, synthetic: bool) -> bool:
        """Observe through the existing gateway accumulator under the epoch fence."""
        with self._trace_observation_lock:
            if self._trace_discarded or not self.accepts_events():
                accumulator._omit("voice_attempt_discarded")
                return False
            self._trace_accumulator = accumulator
            return accumulator.observe(item, synthetic=synthetic)

    def discard_trace_calls(self) -> None:
        """Synchronously clear and fence the current gateway observation owner."""
        with self._trace_observation_lock:
            self._trace_discarded = True
            if self._trace_accumulator is not None:
                self._trace_accumulator._omit("voice_attempt_discarded")
            self._trace_accumulator = None

    def release_trace_observation(self) -> None:
        with self._trace_observation_lock:
            self._trace_accumulator = None

    def accepts_events(self) -> bool:
        try:
            return self._attempt_is_current() is True
        except Exception:
            return False

    def register_provider_work(
        self,
        completion: asyncio.Future[Any],
        force_close: Callable[[], Any],
    ) -> bool:
        return self._provider_work_callback(completion, force_close)

    def detach_owner(self) -> None:
        """Sever every callback that can retain the attempt or coordinator."""

        self._attempt_is_current = _reject_signal_event
        self._provider_work_callback = _reject_provider_work


class VoiceAttempt:
    """Run one provider attempt without any durable or effectful owner."""

    def __init__(
        self,
        *,
        request: VoiceAttemptRequest,
        gateway: _ProviderGateway,
        is_epoch_current: Callable[[int], bool],
        on_delta: Callable[[VoiceAttemptDelta], None] | None = None,
        visible_delta_sink: Callable[[VoiceAttemptDelta], Awaitable[None]]
        | None = None,
        on_tool_request: Callable[[VoiceAttemptToolRequest], None] | None = None,
        on_failed: Callable[[ProviderAttemptFailed], None] | None = None,
    ) -> None:
        self._request: VoiceAttemptRequest | None = request
        self._gateway = gateway
        self._is_epoch_current = is_epoch_current
        self._on_delta = on_delta
        self._visible_delta_sink = visible_delta_sink
        self._on_tool_request = on_tool_request
        self._on_failed = on_failed
        self._invalidated = False
        self._speech_frozen = False
        self._response_parts: list[str] = []
        self._tool_request: VoiceAttemptToolRequest | None = None
        self._trace_manifest: ProvisionalTraceManifest | None = None
        self._trace_envelopes: tuple[ProvisionalTraceEnvelope, ...] = ()
        self._runner: asyncio.Task[None] | None = None
        self._stream: Any | None = None
        self._provider_work: set[asyncio.Future[Any]] = set()
        self._provider_closers: list[Callable[[], Any]] = []
        self._provider_cleanup_completion: asyncio.Task[None] | None = None
        self._cleanup_completion: asyncio.Task[None] | None = None
        self._force_close_task: asyncio.Task[None] | None = None
        self._tts_cancellers: list[Callable[[], Any]] = []
        self._tts_cancellation_work: set[asyncio.Future[Any]] = set()
        self._tts_cleanup_completion: asyncio.Task[None] | None = None
        self.signals = _AttemptProviderSignals(
            exchange_capture_enabled=request.exchange_capture_enabled,
            capture_detail=request.capture_detail,
            attempt_is_current=self._accepts_events,
            provider_work_callback=self._register_provider_work,
        )

    @property
    def attempt_epoch(self) -> int:
        request = self._request
        if request is None:
            raise RuntimeError("attempt content has been detached")
        return request.attempt_epoch

    @property
    def runner_task(self) -> asyncio.Task[None]:
        if self._runner is None:
            raise RuntimeError("attempt has not started")
        return self._runner

    @property
    def provider_cleanup_task(self) -> asyncio.Task[None]:
        """Return completion of the consumer and real provider transport work."""

        if self._runner is None:
            raise RuntimeError("attempt has not started")
        if not self._provider_work:
            return self._runner
        if self._provider_cleanup_completion is None:
            self._provider_cleanup_completion = asyncio.create_task(
                _await_attempt_handles(self._runner, self._provider_work)
            )
        return self._provider_cleanup_completion

    @property
    def tts_cleanup_task(self) -> asyncio.Task[None] | None:
        """Return finalization of directly registered speech cancellation work."""

        if not self._tts_cancellation_work:
            return None
        if self._tts_cleanup_completion is None:
            self._tts_cleanup_completion = asyncio.create_task(
                _await_future_handles(self._tts_cancellation_work)
            )
        return self._tts_cleanup_completion

    @property
    def cleanup_task(self) -> asyncio.Task[None]:
        """Return completion of provider and registered cancellation work."""

        provider = self.provider_cleanup_task
        tts = self.tts_cleanup_task
        if tts is None:
            return provider
        if self._cleanup_completion is None:
            self._cleanup_completion = asyncio.create_task(
                _await_survivors(provider, tts)
            )
        return self._cleanup_completion

    @property
    def snapshot(self) -> VoiceAttemptSnapshot:
        request = self._request
        if request is None:
            raise RuntimeError("attempt content has been detached")
        usage = tuple(
            cast(Mapping[str, object], freeze_json(payload))
            for payload in self.signals.usage_payloads()
        )
        return VoiceAttemptSnapshot(
            attempt_epoch=request.attempt_epoch,
            response_text="".join(self._response_parts),
            tool_request=self._tool_request,
            speech_frozen=self._speech_frozen,
            usage_payloads=usage,
            trace_manifest=self._trace_manifest,
            trace_envelopes=self._trace_envelopes,
        )

    def _accepts_events(self) -> bool:
        request = self._request
        if self._invalidated or request is None:
            return False
        try:
            return self._is_epoch_current(request.attempt_epoch) is True
        except Exception:
            return False

    def _register_provider_work(
        self,
        completion: asyncio.Future[Any],
        force_close: Callable[[], Any],
    ) -> bool:
        if all(existing is not force_close for existing in self._provider_closers):
            self._provider_closers.append(force_close)
        self._track_work(completion)
        return True

    def _track_work(self, completion: asyncio.Future[Any]) -> None:
        if completion.done():
            _consume_task_result(completion)
            return
        self._provider_work.add(completion)
        completion.add_done_callback(self._provider_work.discard)
        completion.add_done_callback(_consume_task_result)

    def _track_tts_cancellation_awaitable(
        self,
        result: Awaitable[Any],
    ) -> None:
        try:
            completion = asyncio.ensure_future(result)
        except BaseException:
            _close_awaitable(result)
            raise
        if completion.done():
            _consume_task_result(completion)
            return
        self._tts_cancellation_work.add(completion)
        completion.add_done_callback(self._tts_cancellation_work.discard)
        completion.add_done_callback(_consume_task_result)

    @staticmethod
    def _notify(callback: Callable[[Any], None] | None, event: object) -> None:
        if callback is None:
            return
        with contextlib.suppress(Exception):
            callback(event)

    def start(self) -> asyncio.Task[None]:
        """Start generation once and return its app-loop task."""

        if self._runner is not None:
            return self._runner
        if not self._accepts_events():
            raise RuntimeError("attempt epoch is not current")
        self._runner = asyncio.create_task(self._run())
        return self._runner

    async def wait(self) -> None:
        """Wait for the provider runner to exit."""

        await self.runner_task

    async def _run(self) -> None:
        request = self._request
        if request is None or not self._accepts_events():
            return
        stream = None
        try:
            stream_kwargs: dict[str, object] = {}
            if getattr(self._gateway, "supports_provisional_voice", False) or getattr(
                self._gateway, "token_gated_adapter_entry", False
            ):
                stream_kwargs["dispatch_purpose"] = (
                    ConsoleProviderCallPurpose.VOICE_PROVISIONAL
                )
                if (
                    request.provisional_trace_attempt is not None
                    and request.prepared.provenance is not None
                ):
                    stream_kwargs.update(
                        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                        route=ConsoleRequestRoute.FRESH,
                        provisional_trace_attempt=request.provisional_trace_attempt,
                    )
                else:
                    admit_capture_off = getattr(
                        self._gateway, "admit_capture_off_stream", None
                    )
                    if callable(admit_capture_off):
                        stream_kwargs["capture_off_admission"] = admit_capture_off(
                            purpose=ConsoleProviderCallPurpose.VOICE_PROVISIONAL,
                            route=None,
                        )
            stream = self._gateway.stream_chat(
                request.resolution,
                request.prepared,
                tools=request.prepared.tools,
                signals=self.signals,
                **stream_kwargs,
            )
            self._stream = stream
            async for item in stream:
                if not self._accepts_events():
                    continue
                if isinstance(item, ProviderToolCalls):
                    if not item.tool_calls:
                        continue
                    if self._tool_request is not None:
                        continue
                    frozen_calls = freeze_json(item.tool_calls)
                    if not isinstance(frozen_calls, tuple):  # pragma: no cover
                        raise TypeError("tool calls must freeze to a tuple")
                    event = VoiceAttemptToolRequest(
                        request.attempt_epoch,
                        cast(tuple[Mapping[str, Any], ...], frozen_calls),
                    )
                    self._tool_request = event
                    self._speech_frozen = True
                    self._abandon_provisional_trace()
                    self.signals.discard_trace_calls()
                    self._notify(self._on_tool_request, event)
                    continue
                if self._speech_frozen or not isinstance(item, str) or not item:
                    continue
                event = VoiceAttemptDelta(request.attempt_epoch, item)
                self._response_parts.append(item)
                if self._visible_delta_sink is not None:
                    for block in iter_voice_visible_blocks(item):
                        if not self._accepts_events():
                            return
                        await self._visible_delta_sink(
                            VoiceAttemptDelta(request.attempt_epoch, block)
                        )
                        if not self._accepts_events():
                            return
                self._notify(self._on_delta, event)
            if self._accepts_events() and self._tool_request is None:
                self._seal_provisional_trace()
        except asyncio.CancelledError:
            self.invalidate()
            raise
        except Exception as exc:
            if not self._accepts_events():
                return
            event = ProviderAttemptFailed(request.attempt_epoch, type(exc).__name__)
            self.invalidate()
            self._notify(self._on_failed, event)
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                await close()

    def register_tts_canceller(self, canceller: Callable[[], Any]) -> None:
        """Attach an attempt-owned speech cancellation handle."""

        if not callable(canceller):
            raise TypeError("canceller must be callable")
        if self._invalidated:
            return
        self._tts_cancellers.append(canceller)

    def invalidate(self) -> None:
        """Synchronously fence every later callback and destroy local bodies."""

        self._invalidated = True
        self._speech_frozen = True
        self._response_parts.clear()
        self._tool_request = None
        self._abandon_provisional_trace()
        self.signals.discard_usage_payloads()
        self.signals.discard_exchange_captures()
        self.signals.discard_trace_calls()

    def _seal_provisional_trace(self) -> None:
        request = self._request
        if request is None or request.provisional_trace_attempt is None:
            return
        try:
            manifest, envelopes = self._gateway.seal_provisional_voice_trace(
                request.provisional_trace_attempt
            )
        except ProvisionalTraceUnavailable:
            return
        self._trace_manifest = manifest
        self._trace_envelopes = envelopes

    def _abandon_provisional_trace(self) -> None:
        request = self._request
        if request is None or request.provisional_trace_attempt is None:
            return
        self._trace_manifest = None
        self._trace_envelopes = ()
        try:
            self._gateway.abandon_provisional_voice_trace(
                request.provisional_trace_attempt
            )
        except ProvisionalTraceUnavailable:
            pass

    def request_cancellation(self) -> None:
        """Fence first, then request cooperative provider and TTS cancellation."""

        self.invalidate()
        runner = self._runner
        if runner is not None and not runner.done():
            runner.cancel()
        cancellers, self._tts_cancellers = self._tts_cancellers, []
        for cancel in cancellers:
            result: Any = _NO_TTS_CANCELLATION_RESULT
            try:
                result = cancel()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            try:
                if result is not _NO_TTS_CANCELLATION_RESULT and inspect.isawaitable(
                    result
                ):
                    self._track_tts_cancellation_awaitable(result)
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if runner is not None:
            _ = self.cleanup_task

    async def force_close_transport(self) -> None:
        """Best-effort force close of this attempt's provider stream only."""

        await asyncio.shield(self.start_force_close_transport())

    def start_force_close_transport(self) -> asyncio.Task[None]:
        """Start close from captured handles without retaining this attempt."""

        if self._force_close_task is None:
            self._force_close_task = asyncio.create_task(
                _force_close_handles(tuple(self._provider_closers), self._stream)
            )
            self._force_close_task.add_done_callback(_consume_task_result)
        return self._force_close_task

    def detach_content(self) -> None:
        """Release all directly held content before an opaque orphan is retained."""

        self.invalidate()
        self.signals.detach_owner()
        self._is_epoch_current = _reject_epoch
        self._request = None
        self._gateway = cast(_ProviderGateway, None)
        self._stream = None
        self._provider_work = set()
        self._provider_closers.clear()
        self._provider_cleanup_completion = None
        self._cleanup_completion = None
        self._force_close_task = None
        self._on_delta = None
        self._visible_delta_sink = None
        self._on_tool_request = None
        self._on_failed = None
        self._tts_cancellers.clear()
        self._tts_cancellation_work = set()
        self._tts_cleanup_completion = None


class AttemptCleanupManager:
    """Bound obsolete-attempt cleanup and escalate on the approved timeline."""

    def __init__(
        self,
        supervisor: VoiceDispatchSupervisor,
        *,
        wait_for_exit: _WaitForExit | None = None,
        on_conservative: Callable[[int], None] | None = None,
        on_detached: Callable[[int], None] | None = None,
    ) -> None:
        self._supervisor = supervisor
        self._wait_for_exit = wait_for_exit or _wait_for_task_exit
        self._on_conservative = on_conservative
        self._on_detached = on_detached
        self._cleanups: set[asyncio.Task[AttemptCleanupOutcome]] = set()
        self._cleanup_by_attempt: weakref.WeakKeyDictionary[
            VoiceAttempt, _CleanupState
        ] = weakref.WeakKeyDictionary()

    @property
    def obsolete_cleanup_count(self) -> int:
        """Return cleanups still owned by the logical-turn layer."""

        return len(self._cleanups)

    def cancel(self, attempt: VoiceAttempt) -> _CleanupObserver:
        """Synchronously fence an attempt and start bounded cleanup."""

        existing = self._cleanup_by_attempt.get(attempt)
        if existing is not None:
            return _CleanupObserver(existing.outcome)
        if len(self._cleanups) >= _MAX_OBSOLETE_CLEANUPS:
            raise VoiceCleanupCapacityExceeded()
        loop = asyncio.get_running_loop()
        cancelled_at = loop.time()
        epoch = attempt.attempt_epoch
        cancellation_finished: asyncio.Future[None] = loop.create_future()
        outcome: asyncio.Future[AttemptCleanupOutcome] = loop.create_future()
        outcome.add_done_callback(_consume_task_result)
        cleanup = asyncio.create_task(
            self._cleanup(
                attempt,
                epoch,
                cancelled_at,
                cancellation_finished,
                outcome,
            )
        )
        try:
            self._supervisor.retain_pending_cleanup(cleanup)
        except VoiceDispatchQuarantined:
            cleanup.cancel()
            cleanup.add_done_callback(_consume_task_result)
            raise VoiceCleanupCapacityExceeded() from None
        self._cleanups.add(cleanup)
        self._cleanup_by_attempt[attempt] = _CleanupState(outcome)
        cleanup.add_done_callback(self._cleanups.discard)
        cleanup.add_done_callback(partial(_complete_cleanup_outcome, outcome))
        cleanup.add_done_callback(_consume_task_result)
        try:
            attempt.request_cancellation()
        finally:
            if not cancellation_finished.done():
                cancellation_finished.set_result(None)
        return _CleanupObserver(outcome)

    async def _cleanup(
        self,
        attempt: VoiceAttempt,
        epoch: int,
        cancelled_at: float,
        cancellation_finished: asyncio.Future[None],
        public_outcome: asyncio.Future[AttemptCleanupOutcome],
    ) -> AttemptCleanupOutcome:
        await cancellation_finished
        completion = attempt.provider_cleanup_task
        tts_completion = attempt.tts_cleanup_task
        loop = asyncio.get_running_loop()
        force_close: asyncio.Task[None] | None = None
        if completion.done() or await self._wait_for_exit(
            completion,
            _remaining_seconds(loop, cancelled_at + 2.0),
        ):
            outcome = AttemptCleanupOutcome.CLEAN
        else:
            self._notify(self._on_conservative, epoch)
            if completion.done() or await self._wait_for_exit(
                completion,
                _remaining_seconds(loop, cancelled_at + 5.0),
            ):
                outcome = AttemptCleanupOutcome.CLEAN
            else:
                force_close = attempt.start_force_close_transport()
                work_exited = completion.done() or await self._wait_for_exit(
                    completion,
                    _remaining_seconds(loop, cancelled_at + 5.5),
                )
                await asyncio.sleep(0)
                work_exited = work_exited or completion.done()
                if work_exited and not force_close.done():
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(
                            asyncio.shield(force_close),
                            _remaining_seconds(loop, cancelled_at + 5.5),
                        )
                close_exited = force_close.done()
                if work_exited and close_exited:
                    outcome = AttemptCleanupOutcome.FORCE_CLOSED
                else:
                    outcome = AttemptCleanupOutcome.DETACHED

        survivors: list[asyncio.Task[Any]] = []
        if outcome is AttemptCleanupOutcome.DETACHED:
            survivors.append(completion)
            if force_close is not None:
                survivors.append(force_close)
        if tts_completion is not None and not tts_completion.done():
            survivors.append(tts_completion)

        if survivors:
            survivor = asyncio.create_task(_await_survivors(*survivors))
            self._supervisor.retain_orphan(
                survivor, pending_cleanup=asyncio.current_task()
            )
            attempt.detach_content()
            self._notify(self._on_detached, epoch)

        # Only the actual owner reaches disposition. Cancelling an observer
        # cannot release this shared gate, and a transferred survivor keeps it.
        self._supervisor.release_pending_cleanup(asyncio.current_task())
        _publish_cleanup_outcome(public_outcome, outcome)
        return outcome

    @staticmethod
    def _notify(callback: Callable[[int], None] | None, epoch: int) -> None:
        if callback is None:
            return
        with contextlib.suppress(Exception):
            callback(epoch)


async def _await_attempt_handles(
    runner: asyncio.Task[Any],
    provider_work: set[asyncio.Future[Any]],
) -> None:
    await asyncio.gather(runner, return_exceptions=True)
    await _await_future_handles(provider_work)


async def _await_future_handles(work: set[asyncio.Future[Any]]) -> None:
    while work:
        pending = tuple(completion for completion in work if not completion.done())
        if not pending:
            work.clear()
            return
        await asyncio.gather(
            *(asyncio.shield(completion) for completion in pending),
            return_exceptions=True,
        )
        work.difference_update(
            completion for completion in pending if completion.done()
        )


def _close_awaitable(awaitable: Awaitable[Any]) -> None:
    close = getattr(awaitable, "close", None)
    if callable(close):
        with contextlib.suppress(BaseException):
            close()


def _publish_cleanup_outcome(
    outcome_future: asyncio.Future[AttemptCleanupOutcome],
    outcome: AttemptCleanupOutcome,
) -> None:
    if not outcome_future.done():
        outcome_future.set_result(outcome)


def _complete_cleanup_outcome(
    outcome_future: asyncio.Future[AttemptCleanupOutcome],
    cleanup: asyncio.Task[AttemptCleanupOutcome],
) -> None:
    if outcome_future.done():
        return
    if cleanup.cancelled():
        outcome_future.cancel()
        return
    error = cleanup.exception()
    if error is not None:
        outcome_future.set_exception(error)
        return
    outcome_future.set_result(cleanup.result())


def _remaining_seconds(
    loop: asyncio.AbstractEventLoop,
    deadline: float,
) -> float:
    return max(0.0, deadline - loop.time())


async def _run_transport_close(close: Callable[[], Any]) -> None:
    if inspect.iscoroutinefunction(close):
        result = close()
    else:
        result = await asyncio.to_thread(close)
    if inspect.isawaitable(result):
        await result


async def _force_close_handles(
    closers: tuple[Callable[[], Any], ...],
    stream: Any | None,
) -> None:
    pending: list[asyncio.Future[Any]] = []
    result: Any = None
    close: Callable[[], Any] | None = None
    for closer in closers:
        try:
            result = closer()
            if inspect.isawaitable(result):
                pending.append(asyncio.ensure_future(result))
        except Exception:
            continue
    closers = ()
    closer = None
    result = None

    if stream is not None:
        close = getattr(stream, "force_close", None)
        if not callable(close):
            close = getattr(stream, "aclose", None)
        if callable(close):
            pending.append(asyncio.create_task(_run_transport_close(close)))
    stream = None
    close = None

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _wait_for_task_exit(task: asyncio.Task[Any], timeout: float) -> bool:
    if task.done():
        return True
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        return False
    except asyncio.CancelledError:
        if not task.done():
            raise
    except Exception:
        pass
    return task.done()


async def _await_survivors(*tasks: asyncio.Task[Any]) -> None:
    await asyncio.gather(*tasks, return_exceptions=True)


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    with contextlib.suppress(BaseException):
        task.result()


__all__ = [
    "AttemptCleanupManager",
    "AttemptCleanupOutcome",
    "ProviderAttemptFailed",
    "VoiceAttempt",
    "VoiceAttemptDelta",
    "VoiceAttemptRequest",
    "VoiceAttemptSnapshot",
    "VoiceAttemptToolRequest",
    "VoiceCleanupCapacityExceeded",
]
