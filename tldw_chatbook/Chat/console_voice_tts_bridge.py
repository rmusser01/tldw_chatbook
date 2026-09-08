"""Bounded audio handoff from an owner-loop TTS service to the voice loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import queue
import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from tldw_chatbook.Audio.voice_process_protocol import (
    CreditWindow,
    Mailbox,
    ProtocolError,
    Record,
    StreamKey,
    encode_record,
)
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse
from tldw_chatbook.TTS.pcm_stream import iter_normalized_pcm_frames


_BLOCK_BYTES = 65_536
_BLOCK_CAPACITY = 8
_POLL_SECONDS = 0.001
_MAX_PROCESS_PHRASES = 3


class OwnerLoopTtsError(RuntimeError):
    """Report a content-free failure at the cross-loop TTS boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class PcmProduction:
    """Retained owner cleanup for one phrase, independent of its observers."""

    def __init__(self, key: StreamKey, owner_loop: asyncio.AbstractEventLoop) -> None:
        self.key = key
        self._loop = owner_loop
        self._cancelled = threading.Event()
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._receipt: concurrent.futures.Future[str] = concurrent.futures.Future()
        self._retired: concurrent.futures.Future[str] = concurrent.futures.Future()

    def cancel(self) -> None:
        """Fence this producer immediately; its first cleanup remains owned."""
        self._cancelled.set()
        self._loop.call_soon_threadsafe(self._cancel_owner)

    def _cancel_owner(self) -> None:
        if (
            self._started
            and not self._receipt.done()
            and self._task is not None
            and not self._task.done()
            and not self._task.cancelling()
        ):
            self._task.cancel()

    async def wait_for_cleanup(self) -> str:
        """Return the categorical actual outcome without cancelling its owner."""
        return await asyncio.shield(asyncio.wrap_future(self._receipt))

    async def wait_for_retirement(self) -> str:
        """Observe receipt consumption or transport retirement, not resource close."""
        return await asyncio.shield(asyncio.wrap_future(self._retired))


class OwnerLoopPcmProducer:
    """Decode original responses on their owner, with one global IPC PCM window.

    The existing whole-WAV/raw decoder allocations are separate from IPC. Each
    ready 960-byte frame is sent immediately under a reservation acquired before
    pulling the decoder; neither encoded audio nor adapter metadata crosses.
    """

    def __init__(
        self,
        owner_loop: asyncio.AbstractEventLoop,
        synthesize: Callable[..., Awaitable[TTSAudioResponse]],
        outbound: Mailbox,
        *,
        cleanup_key: StreamKey,
        on_fault: Callable[[ProtocolError | OwnerLoopTtsError], None] | None = None,
    ) -> None:
        if not outbound.outbound or outbound.direction != "parent_to_child":
            raise ProtocolError()
        if cleanup_key != StreamKey(
            outbound.generation, cleanup_key.request_id, "tts_closed"
        ):
            raise ProtocolError()
        self._owner_loop = owner_loop
        self._synthesize = synthesize
        self._outbound = outbound
        self._on_fault = on_fault
        self.failure: ProtocolError | OwnerLoopTtsError | None = None
        self._transport_failure: ProtocolError | None = None
        self._cleanup_window = CreditWindow(cleanup_key)
        self._lock = threading.Lock()
        self._closed = False
        self._states: dict[StreamKey, PcmProduction] = {}
        self._tail: concurrent.futures.Future[str] = concurrent.futures.Future()
        self._tail.set_result("clean")
        self._retirement_tail = self._tail
        self._window: CreditWindow | None = None
        self._sequence = 0

    @property
    def outstanding(self) -> tuple[int, int]:
        """Return all retained IPC PCM, including written/consumer-owned data."""
        with self._lock:
            return (0, 0) if self._window is None else self._window.outstanding

    @property
    def cleanup_outstanding(self) -> tuple[int, int]:
        """Return the single session-wide cleanup receipt's retained custody."""
        return self._cleanup_window.outstanding

    def submit(self, *, key: StreamKey, text: str) -> PcmProduction:
        """Retain one bounded phrase request and schedule only its original owner."""
        if type(key) is not StreamKey or key.lane != "pcm":
            raise ProtocolError()
        if (
            key.generation != self._outbound.generation
            or key.request_id != self._cleanup_window.key.request_id
            or type(text) is not str
            or len(text) > 3072
        ):
            raise ProtocolError()
        try:
            payload = text.encode("utf-8")
        except UnicodeError:
            raise ProtocolError() from None
        encode_record(_pcm_record(key, "synthesize", 1, payload), "child_to_parent")
        with self._lock:
            if self._closed:
                raise OwnerLoopTtsError("tts_bridge_closed")
            if key in self._states:
                raise ProtocolError()
            if len(self._states) >= _MAX_PROCESS_PHRASES:
                raise OwnerLoopTtsError("tts_bridge_capacity_exceeded")
            state = PcmProduction(key, self._owner_loop)
            predecessor = self._tail
            previous_retirement = self._retirement_tail
            self._tail = state._receipt
            self._retirement_tail = state._retired
            self._states[key] = state
            self._sequence += 1
            sequence = self._sequence

        def launch() -> None:
            state._task = asyncio.create_task(
                self._produce(state, predecessor, previous_retirement, text, sequence)
            )

        try:
            self._owner_loop.call_soon_threadsafe(launch)
        except RuntimeError:
            with self._lock:
                self._closed = True
                self._states.pop(key)
            state._receipt.set_result("failed")
            state._retired.set_result("failed")
            raise OwnerLoopTtsError("tts_bridge_failed") from None
        return state

    def cancel(self, key: StreamKey) -> None:
        """Cancel only an exact retained phrase, never a replacement window."""
        with self._lock:
            state = self._states.get(key)
        if state is not None:
            state.cancel()

    def acknowledge(self, key: StreamKey, sequence: int, consumed_bytes: int) -> None:
        """Accept matching consumption/discard custody after real wire delivery."""
        with self._lock:
            window = self._cleanup_window if key.lane == "tts_closed" else self._window
            if window is None:
                raise ProtocolError()
            window.acknowledge(key, sequence, consumed_bytes)

    def accept_credit(self, record: Record) -> None:
        """Route a validated wire credit to the current PCM or cleanup window."""
        with self._lock:
            window = (
                self._cleanup_window
                if record.header.get("lane") == "tts_closed"
                else self._window
            )
            if window is None:
                raise ProtocolError()
            window.accept_credit(record)

    async def aclose(self) -> None:
        """Fence submissions and observe every actual response cleanup.

        Outstanding wire data stays charged until the child consumes/discards it;
        closing this owner cannot stand in for that receipt or device playback.
        """
        with self._lock:
            self._closed = True
            states = tuple(self._states.values())
        for state in states:
            state.cancel()
        for state in states:
            await state.wait_for_cleanup()

    def _retain_failure(self, error: ProtocolError | OwnerLoopTtsError) -> None:
        with self._lock:
            self._closed = True
            first = self.failure is None
            if first:
                self.failure = type(error)(error.code)
        if first and self._on_fault is not None:
            with contextlib.suppress(Exception):
                self._on_fault(self.failure)

    def transport_failed(self, error: ProtocolError) -> None:
        """Retire broken IPC without rewriting actual response cleanup results."""
        if type(error) is not ProtocolError:
            raise TypeError("invalid_voice_transport_failure")
        with self._lock:
            self._transport_failure = self._transport_failure or ProtocolError(
                error.code
            )
            states = tuple(self._states.values())
        self._retain_failure(error)
        self._outbound.close()
        self._cleanup_window.close()
        for state in states:
            state.cancel()

    async def _produce(
        self,
        state: PcmProduction,
        predecessor: concurrent.futures.Future[str],
        previous_retirement: concurrent.futures.Future[str],
        text: str,
        sequence: int,
    ) -> None:
        state._started = True
        previous = asyncio.wrap_future(predecessor)
        response = None
        frames = None
        outcome = "clean"
        last_sequence = 0
        try:
            outcome = await asyncio.shield(previous)
            if outcome != "clean":
                return
            if state._cancelled.is_set():
                return
            while self.outstanding != (0, 0):
                await asyncio.sleep(_POLL_SECONDS)
            if state._cancelled.is_set():
                return
            self._outbound.open_stream(state.key)
            with self._lock:
                self._window = window = CreditWindow(state.key)
            response = await self._synthesize(text=text)
            frames = iter_normalized_pcm_frames(
                audio_format=response.audio_format,
                sample_rate=response.sample_rate,
                channels=response.metadata.get("channels", 1),
                byte_stream=response.byte_stream,
            )
            while not state._cancelled.is_set():
                permit = await window.reserve()
                try:
                    frame = await anext(frames)
                except StopAsyncIteration:
                    permit.abandon()
                    self._outbound.put(
                        _pcm_record(
                            state.key, "pcm_end", sequence, last_sequence=last_sequence
                        )
                    )
                    break
                except BaseException:
                    permit.abandon()
                    raise
                if state._cancelled.is_set():
                    permit.abandon()
                    break
                record = _pcm_record(state.key, "pcm", permit.sequence, frame)
                permit.publish(record)
                self._outbound.put(record)
                last_sequence = permit.sequence
                del record, frame, permit
        except asyncio.CancelledError:
            if not state._cancelled.is_set():
                outcome = "failed"
        except ProtocolError as error:
            outcome = "failed"
            self.transport_failed(error)
        except Exception:
            outcome = "failed"
        finally:
            # A cancelled queued middle phrase must not bypass its predecessor.
            if await _await_cancellation_protected(previous) != "clean":
                outcome = "failed"
            if frames is not None:
                try:
                    await _await_cancellation_protected(
                        asyncio.create_task(frames.aclose())
                    )
                except BaseException:
                    outcome = "failed"
            if response is not None:
                # This is the first real aclose, retained before it can suspend.
                # TTSAudioResponse marks _closed before its awaited callbacks.
                cleanup = asyncio.create_task(response.aclose())
                try:
                    await _await_cancellation_protected(cleanup)
                except BaseException:
                    outcome = "failed"
            # Actual cleanup is observable even while the peer holds receipt credit.
            state._receipt.set_result(outcome)
            if outcome != "clean":
                self._retain_failure(OwnerLoopTtsError("tts_bridge_failed"))
            retired = "failed"
            try:
                await _await_cancellation_protected(
                    asyncio.wrap_future(previous_retirement)
                )
                if self._transport_failure is None:
                    permit = await self._cleanup_window.reserve()
                    record = _pcm_record(
                        state.key,
                        "tts_closed",
                        permit.sequence,
                        outcome=outcome,
                        last_sequence=last_sequence,
                    )
                    permit.publish(record)
                    self._outbound.put(record)
                    del record, permit
                    while (
                        self._cleanup_window.outstanding != (0, 0)
                        and self._transport_failure is None
                    ):
                        await asyncio.sleep(_POLL_SECONDS)
                    if self._transport_failure is None:
                        retired = "clean"
            except ProtocolError as error:
                self.transport_failed(error)
            finally:
                with self._lock:
                    self._states.pop(state.key)
                state._retired.set_result(retired)


def _pcm_record(
    key: StreamKey, op: str, sequence: int, payload: bytes = b"", **fields: object
) -> Record:
    return Record(
        dict(
            version=1,
            op=op,
            generation=key.generation,
            request_id=key.request_id,
            sequence=sequence,
            turn_id=key.turn_id,
            revision=key.revision,
            epoch=key.epoch,
            phrase_id=key.phrase_id,
            **fields,
        ),
        payload,
    )


class _PhraseState:
    def __init__(self) -> None:
        self.blocks: queue.Queue[bytes] = queue.Queue(maxsize=_BLOCK_CAPACITY)
        self.metadata: concurrent.futures.Future[dict[str, Any]] = (
            concurrent.futures.Future()
        )
        self.receipt: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._lock = threading.Lock()
        self._submitted: concurrent.futures.Future[None] | None = None
        self._started = False
        self._cancelled = False
        self._terminal = False
        self._failed = False

    def attach(self, submitted: concurrent.futures.Future[None]) -> None:
        with self._lock:
            self._submitted = submitted
            cancel_started = self._cancelled and self._started
        if cancel_started:
            submitted.cancel()

    def mark_started(self) -> None:
        with self._lock:
            self._started = True

    def publish_metadata(self, metadata: Mapping[str, Any]) -> bool:
        with self._lock:
            if self._cancelled:
                return False
        with contextlib.suppress(concurrent.futures.InvalidStateError):
            self.metadata.set_result(dict(metadata))
        return True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    async def put(self, block: bytes) -> bool:
        while True:
            with self._lock:
                if self._cancelled:
                    return False
            try:
                self.blocks.put_nowait(block)
            except queue.Full:
                await asyncio.sleep(_POLL_SECONDS)
            else:
                return True

    async def next_block(self) -> bytes:
        while True:
            with self._lock:
                if self._cancelled:
                    raise StopAsyncIteration
            try:
                block = self.blocks.get_nowait()
            except queue.Empty:
                with self._lock:
                    cancelled = self._cancelled
                    terminal = self._terminal
                    failed = self._failed
                if cancelled:
                    raise StopAsyncIteration
                if terminal:
                    if failed:
                        raise OwnerLoopTtsError("tts_bridge_failed")
                    raise StopAsyncIteration
                await asyncio.sleep(_POLL_SECONDS)
                continue
            with self._lock:
                if self._cancelled:
                    raise StopAsyncIteration
            return block

    def fail(self) -> None:
        with self._lock:
            self._failed = True
        if not self.metadata.done():
            with contextlib.suppress(concurrent.futures.InvalidStateError):
                self.metadata.set_exception(OwnerLoopTtsError("tts_bridge_failed"))

    def finish(self) -> None:
        with self._lock:
            self._terminal = True
        if not self.receipt.done():
            with contextlib.suppress(concurrent.futures.InvalidStateError):
                self.receipt.set_result(None)

    def cancel(self, *, metadata_error: str | None = None) -> None:
        with self._lock:
            self._cancelled = True
            submitted = self._submitted
            started = self._started
        if metadata_error is not None and not self.metadata.done():
            with contextlib.suppress(concurrent.futures.InvalidStateError):
                self.metadata.set_exception(OwnerLoopTtsError(metadata_error))
        if submitted is not None and started:
            submitted.cancel()

    async def close(self) -> None:
        self.cancel()
        await asyncio.shield(asyncio.wrap_future(self.receipt))
        with self._lock:
            failed = self._failed
        if failed:
            raise OwnerLoopTtsError("tts_bridge_failed")


class _ProxyByteStream:
    def __init__(self, state: _PhraseState) -> None:
        self._state = state

    def __aiter__(self) -> _ProxyByteStream:
        return self

    async def __anext__(self) -> bytes:
        return await self._state.next_block()

    async def aclose(self) -> None:
        await self._state.close()


class OwnerLoopTtsSynthesizer:
    """Run shared TTS response ownership on its original event loop."""

    def __init__(
        self,
        owner_loop: asyncio.AbstractEventLoop,
        synthesize: Callable[..., Awaitable[TTSAudioResponse]],
    ) -> None:
        self._owner_loop = owner_loop
        self._synthesize = synthesize
        self._lock = threading.Lock()
        self._closed = False
        self._states: set[_PhraseState] = set()
        self._tail: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._tail.set_result(None)

    async def synthesize_hands_free(self, *, text: str) -> TTSAudioResponse:
        """Synthesize one phrase and return a voice-loop-owned proxy response.

        Args:
            text: Phrase to synthesize on the service owner loop.

        Returns:
            A response whose copied byte stream is consumed on the caller loop.

        Raises:
            OwnerLoopTtsError: If the bridge is closed or owner work fails.
        """

        state = _PhraseState()
        with self._lock:
            if self._closed:
                raise OwnerLoopTtsError("tts_bridge_closed")
            predecessor = self._tail
            self._tail = state.receipt
            self._states.add(state)
        state.receipt.add_done_callback(lambda _done: self._discard(state))

        producer = self._produce(state, predecessor, text)
        try:
            submitted = asyncio.run_coroutine_threadsafe(producer, self._owner_loop)
        except BaseException:
            producer.close()
            state.fail()
            state.finish()
            raise OwnerLoopTtsError("tts_bridge_failed") from None
        state.attach(submitted)

        try:
            metadata = await asyncio.shield(asyncio.wrap_future(state.metadata))
        except asyncio.CancelledError:
            state.cancel()
            await asyncio.shield(asyncio.wrap_future(state.receipt))
            raise

        return TTSAudioResponse(
            provider_id=metadata["provider_id"],
            model_id=metadata["model_id"],
            audio_format=metadata["audio_format"],
            content_type=metadata["content_type"],
            byte_stream=_ProxyByteStream(state),
            sample_rate=metadata["sample_rate"],
            metadata=metadata["metadata"],
        )

    async def aclose(self) -> None:
        """Fence new work and await actual owner-loop cleanup for every phrase."""

        with self._lock:
            self._closed = True
            states = tuple(self._states)
        for state in states:
            state.cancel(metadata_error="tts_bridge_closed")
        for state in states:
            await asyncio.shield(asyncio.wrap_future(state.receipt))

    async def _produce(
        self,
        state: _PhraseState,
        predecessor: concurrent.futures.Future[None],
        text: str,
    ) -> None:
        state.mark_started()
        response: TTSAudioResponse | None = None
        failed = False
        predecessor_observation = asyncio.wrap_future(predecessor)
        try:
            await asyncio.shield(predecessor_observation)
            if state.cancelled:
                return
            response = await self._synthesize(text=text)
            metadata = {
                "provider_id": response.provider_id,
                "model_id": response.model_id,
                "audio_format": response.audio_format,
                "content_type": response.content_type,
                "sample_rate": response.sample_rate,
                "metadata": dict(response.metadata),
            }
            if not state.publish_metadata(metadata):
                return
            async for yielded in response.byte_stream:
                copied = bytes(yielded)
                for offset in range(0, len(copied), _BLOCK_BYTES):
                    if not await state.put(copied[offset : offset + _BLOCK_BYTES]):
                        return
        except asyncio.CancelledError:
            raise
        except BaseException:
            failed = True
            state.fail()
        finally:
            try:
                await _await_cancellation_protected(predecessor_observation)
            except BaseException:
                failed = True
                state.fail()
            if response is not None:
                cleanup = asyncio.create_task(response.aclose())
                try:
                    await _await_cancellation_protected(cleanup)
                except BaseException:
                    failed = True
                    state.fail()
            if failed:
                state.fail()
            state.finish()

    def _discard(self, state: _PhraseState) -> None:
        with self._lock:
            self._states.discard(state)


async def _await_cancellation_protected(
    future: asyncio.Future[Any],
) -> Any:
    """Await owner work to completion despite cancellation of its observer."""

    while True:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.done():
                return future.result()


__all__ = [
    "OwnerLoopPcmProducer",
    "OwnerLoopTtsError",
    "OwnerLoopTtsSynthesizer",
    "PcmProduction",
]
