"""Audio-owned native and rolling transcription; model calls stay off the loop."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
import time
from array import array
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from tldw_chatbook.Audio.duplex_contracts import AudioFrame
from tldw_chatbook.Audio.rolling_transcript import TranscriptHypothesis, TranscriptToken
from tldw_chatbook.Audio.parakeet_voice_worker import (
    ParakeetFailure,
    ParakeetOwnershipTimeout,
    ParakeetServiceUnavailable,
    ParakeetVoiceProcess,
)
from tldw_chatbook.Audio.voice_process_types import VoiceTranscriptCapacityError

_PARAKEET_STREAM_CHUNK_FRAMES = 400
_PARAKEET_STREAM_QUIET_SECONDS = 0.7
_PARAKEET_OWNERSHIP_TIMEOUT_SECONDS = 30.0
_PARAKEET_STREAM_CONTEXT_SIZE = (64, 64)
_PARAKEET_STREAM_PREWARM_FRAMES = 150
_PARAKEET_STREAM_PREWARM_SECONDS = _PARAKEET_STREAM_PREWARM_FRAMES / 100
_ROLLING_DEBOUNCE_SECONDS = 0.12
_ROLLING_MIN_WINDOW_NS = 500_000_000
_PARAKEET_STREAM_PREWARM_PCM_BYTES = _PARAKEET_STREAM_PREWARM_FRAMES * 960
_UNPREPARED_STREAMING_CANDIDATE = object()
_MAX_PENDING_NATIVE_FRAMES = 1000


def _persist_voice_event(event: str, *, diagnostic_sink=None, **fields: Any) -> None:
    if diagnostic_sink is not None:
        with contextlib.suppress(Exception):
            diagnostic_sink(event, fields)


def _stt_failure_metadata(failure: BaseException) -> dict[str, str]:
    """Return fixed failure identity without inspecting exception messages."""

    if isinstance(failure, ParakeetFailure):
        exception_type = failure.native_exception_type or type(failure).__name__
        return {
            "error_category": failure.reason,
            "exception_type": exception_type,
        }
    return {
        "error_category": "unknown_native",
        "exception_type": type(failure).__name__,
    }


def _terminal_parakeet_failure(failure: Exception) -> ParakeetServiceUnavailable:
    if isinstance(failure, ParakeetServiceUnavailable):
        return failure
    if isinstance(failure, ParakeetFailure):
        return ParakeetServiceUnavailable(
            failure.reason,
            native_exception_type=failure.native_exception_type,
        )
    return ParakeetServiceUnavailable("unknown_native")


def _persist_stt_failure_once(
    reported: set[str],
    *,
    provider: str,
    failure: BaseException,
    duration_ms: int | None = None,
    diagnostic_sink=None,
) -> None:
    fields = _stt_failure_metadata(failure)
    category = fields["error_category"]
    if category in reported:
        return
    reported.add(category)
    _persist_voice_event(
        "stt_failed",
        diagnostic_sink=diagnostic_sink,
        provider=provider,
        status="failed",
        **({"duration_ms": duration_ms} if duration_ms is not None else {}),
        **fields,
    )


class _SerialSttWorker:
    """Keep thread-affine native STT model work on one executor thread."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="speculative-voice-stt",
        )
        self._closed = False
        self._model_lock = asyncio.Lock()
        self._poisoned = False
        self._receipts: set[asyncio.Future[Any]] = set()

    def _retain(self, future: asyncio.Future[Any]) -> asyncio.Future[Any]:
        self._receipts.add(future)
        future.add_done_callback(self._receipts.discard)
        return future

    @staticmethod
    async def _drain(future: asyncio.Future[Any]) -> Any:
        # A cancelled observer cannot cancel or abandon real executor cleanup.
        # Parakeet RPC timeouts and process reap bounds limit this receipt.
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
        return future.result()

    async def _observe(self, future: asyncio.Future[Any]) -> Any:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self._drain(future)
            raise

    def _submit(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> asyncio.Future[Any]:
        if self._closed:
            raise RuntimeError("serial_stt_worker_closed")
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(
            self._executor,
            partial(function, *args, **kwargs),
        )

    async def run(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await self._submit(function, *args, **kwargs)

    async def run_owned(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        receipt = self._retain(self._submit(function, *args, **kwargs))
        return await self._observe(receipt)

    @contextlib.asynccontextmanager
    async def lease(self, service: Any):
        if self._poisoned or (
            isinstance(service, ParakeetVoiceProcess) and service.closed
        ):
            raise ParakeetServiceUnavailable("closed")
        try:
            await asyncio.wait_for(
                self._model_lock.acquire(), _PARAKEET_OWNERSHIP_TIMEOUT_SECONDS
            )
        except TimeoutError:
            raise ParakeetOwnershipTimeout() from None
        try:
            # Cancellation after acquisition but before RPC creates no context.
            await asyncio.sleep(0)
            if self._poisoned or (
                isinstance(service, ParakeetVoiceProcess) and service.closed
            ):
                raise ParakeetServiceUnavailable("closed")
            yield
        finally:
            # Prewarm and rolling RPCs can fence the service without passing
            # through parakeet_context's retirement path. A failed reap still
            # owns the model lease regardless of which operation discovered it.
            if (
                isinstance(service, ParakeetVoiceProcess)
                and service.closed
                and not service.reaped
            ):
                self._poisoned = True
            if not self._poisoned or (
                isinstance(service, ParakeetVoiceProcess) and service.reaped
            ):
                self._model_lock.release()

    async def _retire(self, service: Any) -> None:
        self._poisoned = True
        if isinstance(service, ParakeetVoiceProcess):
            try:
                await asyncio.to_thread(service.close)
            except Exception:
                raise ParakeetServiceUnavailable("unknown_native") from None

    @contextlib.asynccontextmanager
    async def parakeet_context(self, service: Any, factory: Callable[..., Any]):
        async with self.lease(service):

            def enter():
                context = _parakeet_stream_context(factory)
                return context, context.__enter__()

            # Retain the result separately: late successful entry still owns a
            # context even when the adapter awaiting it has been cancelled.
            entering = self._retain(self._submit(enter))
            context = None
            try:
                try:
                    context, stream = await asyncio.shield(entering)
                except asyncio.CancelledError:
                    try:
                        context, _stream = await self._drain(entering)
                    except Exception:
                        await self._observe(
                            self._retain(asyncio.create_task(self._retire(service)))
                        )
                    raise
                except Exception as exc:
                    await self._observe(
                        self._retain(asyncio.create_task(self._retire(service)))
                    )
                    terminal_failure = _terminal_parakeet_failure(exc)
                    if terminal_failure is exc:
                        raise
                    raise terminal_failure from None
                yield stream
            finally:
                if context is not None:

                    async def checked_exit():
                        try:
                            await self.run(context.__exit__, None, None, None)
                        except Exception as exc:
                            await self._retire(service)
                            terminal_failure = _terminal_parakeet_failure(exc)
                            if terminal_failure is exc:
                                raise
                            raise terminal_failure from None

                    await self._observe(
                        self._retain(asyncio.create_task(checked_exit()))
                    )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )


async def _close_streaming_candidate(
    candidate: Any,
    *,
    serial_worker: _SerialSttWorker | None = None,
) -> None:
    if candidate is None or candidate is _UNPREPARED_STREAMING_CANDIDATE:
        return
    for name in ("aclose", "close", "finalize"):
        close = getattr(candidate, name, None)
        if not callable(close):
            continue
        if inspect.iscoroutinefunction(close):
            await close()
        elif serial_worker is not None:
            await serial_worker.run(close)
        else:
            await asyncio.to_thread(close)
        return


def _transcript_text(result: object) -> tuple[str, bool]:
    if isinstance(result, str):
        return result.strip(), False
    if not isinstance(result, dict):
        return "", False
    final = result.get("final")
    if isinstance(final, str) and final.strip():
        return final.strip(), True
    partial = result.get("partial")
    if isinstance(partial, str):
        return partial.strip(), False
    text = result.get("text")
    if not isinstance(text, str):
        return "", False
    return text.strip(), partial is False


def _merge_streaming_text(current: str, update: str) -> str:
    if not update:
        return current
    if update.startswith(current):
        return update
    if current.startswith(update):
        return current
    if current.endswith(update):
        return current
    return f"{current} {update}".strip()


def _normalized_pcm16_samples(pcm16: bytes) -> list[float]:
    if len(pcm16) % 2:
        raise ValueError("PCM16 audio must contain whole samples")
    samples = array("h")
    samples.frombytes(pcm16)
    if sys.byteorder != "little":
        samples.byteswap()
    return [sample / 32_768 for sample in samples]


def _parakeet_mlx_audio(pcm16: bytes) -> Any:
    """Convert 48 kHz PCM16 into the package's native 16 kHz MLX input."""

    import mlx.core as mx

    samples = _normalized_pcm16_samples(pcm16)
    return mx.array(samples[::3], dtype=mx.float32)


def _parakeet_stream_factory(candidate: Any) -> Callable[..., Any] | None:
    model = getattr(candidate, "model", None)
    factory = getattr(model, "transcribe_stream", None)
    return factory if callable(factory) else None


def _parakeet_stream_context(factory: Callable[..., Any]) -> Any:
    # The package's default local-attention streaming mode is materially
    # faster than applying the model's full original attention to every
    # incremental update. Idle contexts are released at each silence boundary
    # below, so retained turn objects do not keep conflicting attention modes
    # active on the shared model.
    return factory(context_size=_PARAKEET_STREAM_CONTEXT_SIZE)


def _native_stream_realtime_capable(prewarm_seconds: float) -> bool:
    """Return whether native inference kept pace with its prewarm audio."""

    return (
        not isinstance(prewarm_seconds, bool)
        and isinstance(prewarm_seconds, (int, float))
        and 0 <= prewarm_seconds <= _PARAKEET_STREAM_PREWARM_SECONDS
    )


def _prewarm_parakeet_stream(candidate: Any) -> float:
    """Prewarm the native stream and return its content-free elapsed time."""

    factory = _parakeet_stream_factory(candidate)
    if factory is None:
        raise RuntimeError("native_streaming_stt_unavailable")
    started_at = time.monotonic()
    with _parakeet_stream_context(factory) as stream:
        _feed_parakeet_stream(stream, bytes(_PARAKEET_STREAM_PREWARM_PCM_BYTES))
    return max(0.0, time.monotonic() - started_at)


def _feed_parakeet_stream(stream: Any, pcm16: bytes) -> None:
    remote_push = getattr(stream, "add_pcm16", None)
    if callable(remote_push):
        remote_push(pcm16)
    else:
        stream.add_audio(_parakeet_mlx_audio(pcm16))


async def _prepare_streaming_candidate(
    service: Any,
    *,
    provider: str,
    model: str | None,
    language: str,
    serial_worker: _SerialSttWorker,
    diagnostic_sink=None,
) -> tuple[Any | None, bool]:
    """Prefer usable native STT, closing it before capture when too slow."""

    candidate = await serial_worker.run(
        service.create_streaming_transcriber,
        provider=provider,
        model=model,
        source_lang=language,
    )
    native = callable(getattr(candidate, "process_audio", None))
    if _parakeet_stream_factory(candidate) is not None:
        try:
            async with serial_worker.lease(service):
                prewarm_seconds = await serial_worker.run_owned(
                    _prewarm_parakeet_stream,
                    candidate,
                )
        except Exception as exc:
            if isinstance(service, ParakeetVoiceProcess) and service.closed:
                # A killed worker cannot provide the rolling fallback. Keep
                # capture closed instead of advertising an unusable recognizer.
                raise
            native = False
            _persist_voice_event(
                "stt_native_prepare_failed",
                diagnostic_sink=diagnostic_sink,
                provider=provider,
                status="fallback",
                **_stt_failure_metadata(exc),
            )
        else:
            native = _native_stream_realtime_capable(prewarm_seconds)
            if not native:
                _persist_voice_event(
                    "stt_native_rejected",
                    diagnostic_sink=diagnostic_sink,
                    provider=provider,
                    status="fallback",
                    result_type="slower-than-realtime",
                    latency_ms=round(prewarm_seconds * 1_000),
                )
    if candidate is not None and not native:
        await _close_streaming_candidate(
            candidate,
            serial_worker=serial_worker,
        )
        candidate = None
    return candidate, native


class _RollingWindowStt:
    def __init__(
        self,
        service: Any,
        *,
        provider: str,
        model: str | None,
        language: str,
        serial_worker: _SerialSttWorker | None = None,
        diagnostic_sink=None,
    ) -> None:
        self._service = service
        self._provider = provider
        self._model = model
        self._language = language
        self._closed = False
        self._serial_worker = serial_worker
        self._reported_failure_categories: set[str] = set()
        self._diagnostic_sink = diagnostic_sink

    async def transcribe_window(
        self,
        *,
        pcm16: bytes,
        started_ns: int,
        ended_ns: int,
    ) -> Any:
        if self._closed:
            raise RuntimeError("rolling_stt_closed")
        duration_ms = max(0, (ended_ns - started_ns) // 1_000_000)
        _persist_voice_event(
            "stt_window_started",
            diagnostic_sink=self._diagnostic_sink,
            provider=self._provider,
            status="started",
            duration_ms=duration_ms,
        )
        try:
            transcribe = partial(
                self._service.transcribe_buffer,
                audio_data=pcm16,
                sample_rate=48_000,
                channels=1,
                sample_width=2,
                provider=self._provider,
                model=self._model,
                language=self._language,
            )
            if isinstance(self._service, ParakeetVoiceProcess):
                if self._serial_worker is None:
                    raise ParakeetServiceUnavailable("closed")
                async with self._serial_worker.lease(self._service):
                    result = await self._serial_worker.run_owned(transcribe)
            else:
                result = await asyncio.to_thread(transcribe)
        except Exception as exc:
            _persist_stt_failure_once(
                self._reported_failure_categories,
                provider=self._provider,
                failure=exc,
                duration_ms=duration_ms,
                diagnostic_sink=self._diagnostic_sink,
            )
            raise
        text, _final = _transcript_text(result)
        _persist_voice_event(
            "stt_revision" if text else "stt_empty",
            diagnostic_sink=self._diagnostic_sink,
            provider=self._provider,
            status="ok" if text else "empty",
            duration_ms=duration_ms,
        )
        return TranscriptHypothesis(
            tokens=(TranscriptToken(text, started_ns, ended_ns),),
            covered_through_ns=ended_ns,
        )

    async def abort(self) -> None:
        self._closed = True


class _NativeStreamingStt:
    """Prefer a native stream without blocking the audio/coordinator loop."""

    def __init__(
        self,
        service: Any,
        *,
        provider: str,
        model: str | None,
        language: str,
        prepared_candidate: Any = _UNPREPARED_STREAMING_CANDIDATE,
        serial_worker: _SerialSttWorker | None = None,
        quiet_seconds: float = _PARAKEET_STREAM_QUIET_SECONDS,
        diagnostic_sink=None,
    ) -> None:
        if (
            isinstance(quiet_seconds, bool)
            or not isinstance(quiet_seconds, (int, float))
            or quiet_seconds <= 0
        ):
            raise ValueError("native streaming quiet interval must be positive")
        self._service = service
        self._provider = provider
        self._model = model
        self._language = language
        self._prepared_candidate = prepared_candidate
        self._serial_worker = serial_worker or _SerialSttWorker()
        self._owns_serial_worker = serial_worker is None
        self._quiet_seconds = float(quiet_seconds)
        self._publish: Callable[[Any], None] | None = None
        self._fail: Callable[[BaseException], None] | None = None
        self._settle: Callable[[int], None] | None = None
        self._frames: asyncio.Queue[AudioFrame | None] = asyncio.Queue(
            maxsize=_MAX_PENDING_NATIVE_FRAMES
        )
        self._worker: asyncio.Task[None] | None = None
        self._candidate: Any | None = None
        self._text = ""
        self._published_revision = False
        self._reported_failure_categories: set[str] = set()
        self._diagnostic_sink = diagnostic_sink
        self.failure: VoiceTranscriptCapacityError | None = None

    def _publish_result(
        self,
        result: object,
        *,
        frame: AudioFrame,
        cumulative_revision: bool,
        publish_empty: bool = False,
    ) -> None:
        text, final = _transcript_text(result)
        if isinstance(result, dict) and type(result.get("cumulative")) is bool:
            cumulative_revision = result["cumulative"]
        self._text = (
            text or self._text
            if cumulative_revision
            else _merge_streaming_text(self._text, text)
        )
        if (self._text or final or publish_empty) and self._publish is not None:
            if not self._published_revision:
                self._published_revision = True
                _persist_voice_event(
                    "stt_revision",
                    diagnostic_sink=self._diagnostic_sink,
                    provider=self._provider,
                    status="ok",
                )
            self._publish(
                TranscriptHypothesis(
                    stable_text=self._text if final else "",
                    revisable_text="" if final else self._text,
                    covered_through_ns=frame.ended_ns,
                    is_final=final,
                )
            )

    @staticmethod
    def _push_parakeet_stream(
        stream: Any,
        pcm16: bytes,
        *,
        final: bool,
    ) -> dict[str, object]:
        _feed_parakeet_stream(stream, pcm16)
        result = getattr(stream, "result", None)
        text = getattr(result, "text", "")
        return {
            "text": text if isinstance(text, str) else str(text),
            "partial": not final,
        }

    async def _run_parakeet_stream(self, factory: Callable[..., Any]) -> None:
        while True:
            first = await self._frames.get()
            if first is None:
                self._frames.task_done()
                return

            # A context owns one continuous speech burst. Releasing it after
            # the configured quiet interval keeps the package's model-global
            # local-attention mutation from overlapping a retained prior turn.
            burst_prefix = self._text
            next_first: AudioFrame | None = first
            async with self._serial_worker.parakeet_context(
                self._service, factory
            ) as stream:
                while next_first is not None:
                    frames = [next_first]
                    quiet = False
                    close_after = False
                    while len(frames) < _PARAKEET_STREAM_CHUNK_FRAMES:
                        try:
                            frame = await asyncio.wait_for(
                                self._frames.get(),
                                timeout=self._quiet_seconds,
                            )
                        except TimeoutError:
                            quiet = True
                            break
                        if frame is None:
                            self._frames.task_done()
                            close_after = True
                            break
                        frames.append(frame)

                    duration_ms = (
                        sum(frame.duration_ns for frame in frames) // 1_000_000
                    )
                    _persist_voice_event(
                        "stt_window_started",
                        diagnostic_sink=self._diagnostic_sink,
                        provider=self._provider,
                        status="native",
                        duration_ms=duration_ms,
                    )
                    pcm16 = b"".join(frame.pcm16 for frame in frames)
                    started_at = asyncio.get_running_loop().time()
                    try:
                        result = await self._serial_worker.run_owned(
                            self._push_parakeet_stream,
                            stream,
                            pcm16,
                            final=quiet or close_after,
                        )
                    finally:
                        for _frame in frames:
                            self._frames.task_done()
                    result["text"] = _merge_streaming_text(
                        burst_prefix,
                        str(result["text"]),
                    )
                    _persist_voice_event(
                        "stt_window_completed",
                        diagnostic_sink=self._diagnostic_sink,
                        provider=self._provider,
                        status="native",
                        duration_ms=duration_ms,
                        latency_ms=round(
                            (asyncio.get_running_loop().time() - started_at) * 1_000
                        ),
                        result_size=len(str(result["text"])),
                    )
                    last = frames[-1]
                    self._publish_result(
                        result,
                        frame=last,
                        cumulative_revision=True,
                        publish_empty=True,
                    )
                    if self._settle is not None:
                        for frame in frames:
                            self._settle(frame.sequence)
                    if close_after:
                        return
                    if quiet:
                        break
                    try:
                        next_first = await asyncio.wait_for(
                            self._frames.get(), timeout=self._quiet_seconds
                        )
                    except TimeoutError:
                        self._publish_result(
                            {"text": self._text, "partial": False},
                            frame=last,
                            cumulative_revision=True,
                            publish_empty=True,
                        )
                        break
                    if next_first is None:
                        self._frames.task_done()
                        self._publish_result(
                            {"text": self._text, "partial": False},
                            frame=last,
                            cumulative_revision=True,
                            publish_empty=True,
                        )
                        return

    def start(
        self,
        publish: Callable[[Any], None],
        fail: Callable[[BaseException], None],
        settle: Callable[[int], None],
    ) -> None:
        self._publish = publish
        self._fail = fail
        self._settle = settle
        if self._worker is None or self._worker.done():
            self._text = ""
            self._worker = asyncio.create_task(self._run())

    def submit(self, frame: AudioFrame) -> None:
        if self.failure is not None:
            return
        try:
            self._frames.put_nowait(frame)
        except asyncio.QueueFull:
            self.failure = VoiceTranscriptCapacityError()
            if self._fail is not None:
                self._fail(self.failure)

    async def _run(self) -> None:
        try:
            candidate = self._prepared_candidate
            self._prepared_candidate = _UNPREPARED_STREAMING_CANDIDATE
            if candidate is _UNPREPARED_STREAMING_CANDIDATE:
                candidate = await self._serial_worker.run(
                    self._service.create_streaming_transcriber,
                    provider=self._provider,
                    model=self._model,
                    source_lang=self._language,
                )
            process_audio = getattr(candidate, "process_audio", None)
            parakeet_stream = _parakeet_stream_factory(candidate)
            if not callable(process_audio) and parakeet_stream is None:
                raise RuntimeError("native_streaming_stt_unavailable")
            self._candidate = candidate
            if parakeet_stream is not None:
                await self._run_parakeet_stream(parakeet_stream)
                return
            stream_context = getattr(candidate, "stream_context", None)
            if callable(stream_context):
                async with self._serial_worker.parakeet_context(
                    self._service, stream_context
                ) as stream:
                    await self._run_process_audio(stream.process_audio)
            else:
                await self._run_process_audio(process_audio)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if (
                isinstance(self._service, ParakeetVoiceProcess)
                and self._service.closed
                and not isinstance(exc, ParakeetServiceUnavailable)
            ):
                exc = ParakeetServiceUnavailable("closed")
            _persist_stt_failure_once(
                self._reported_failure_categories,
                provider=self._provider,
                failure=exc,
                diagnostic_sink=self._diagnostic_sink,
            )
            if self._fail is not None:
                self._fail(exc)

    async def _run_process_audio(self, process_audio: Callable[..., Any]) -> None:
        while True:
            frame = await self._frames.get()
            try:
                if frame is None:
                    return
                result = await self._serial_worker.run_owned(process_audio, frame.pcm16)
                self._publish_result(result, frame=frame, cumulative_revision=False)
                if self._settle is not None:
                    self._settle(frame.sequence)
            finally:
                self._frames.task_done()

    async def close(self) -> None:
        worker = self._worker
        if worker is not None and not worker.done():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        await _close_streaming_candidate(
            self._candidate,
            serial_worker=self._serial_worker,
        )
        await _close_streaming_candidate(
            self._prepared_candidate,
            serial_worker=self._serial_worker,
        )
        if self._owns_serial_worker:
            await self._serial_worker.close()
