from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from tldw_chatbook.TTS.adapter_types import TTSAudioResponse


class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread_id: int | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.thread_id = threading.get_ident()
        self.ready.set()
        self.loop.run_forever()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        assert not self.thread.is_alive()
        self.loop.close()


@pytest.fixture
def owner() -> AsyncIterator[_LoopThread]:
    owner = _LoopThread()
    assert owner.ready.wait(1)
    try:
        yield owner
    finally:
        owner.close()


async def _wait_for(predicate: Callable[[], bool]) -> None:
    for _ in range(1_000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    pytest.fail("cross-loop TTS operation did not settle")


async def _collect(response: TTSAudioResponse) -> tuple[bytes, ...]:
    async with response:
        return tuple([chunk async for chunk in response.byte_stream])


def _response(
    byte_stream: AsyncIterator[bytes],
    *,
    cleanup: Callable[[], Awaitable[None]] | None = None,
) -> TTSAudioResponse:
    return TTSAudioResponse(
        provider_id="test-provider",
        model_id="test-model",
        audio_format="pcm",
        content_type="audio/pcm",
        byte_stream=byte_stream,
        sample_rate=48_000,
        metadata={"channels": 1, "source": "owner"},
        cleanup=cleanup,
    )


@pytest.mark.asyncio
async def test_synthesis_iteration_and_close_stay_on_owner_and_copy_bounded_blocks(
    owner: _LoopThread,
) -> None:
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsSynthesizer

    observed_threads: set[int] = set()
    original = bytearray(b"a" * 131_075)

    async def stream() -> AsyncIterator[bytes]:
        try:
            observed_threads.add(threading.get_ident())
            yield original  # type: ignore[misc]
            original[:] = b"z" * len(original)
        finally:
            observed_threads.add(threading.get_ident())

    async def cleanup() -> None:
        observed_threads.add(threading.get_ident())

    async def synthesize(*, text: str) -> TTSAudioResponse:
        assert text == "test"
        observed_threads.add(threading.get_ident())
        return _response(stream(), cleanup=cleanup)

    bridge = OwnerLoopTtsSynthesizer(owner.loop, synthesize)
    proxy = await bridge.synthesize_hands_free(text="test")
    chunks = await _collect(proxy)

    assert b"".join(chunks) == b"a" * 131_075
    assert [len(chunk) for chunk in chunks] == [65_536, 65_536, 3]
    assert proxy.metadata == {"channels": 1, "source": "owner"}
    assert observed_threads == {owner.thread_id}
    await bridge.aclose()


@pytest.mark.asyncio
async def test_next_phrase_waits_for_real_delayed_close_without_overlap(
    owner: _LoopThread,
) -> None:
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsSynthesizer

    calls: list[str] = []
    active = 0
    maximum_active = 0
    first_close_started = threading.Event()
    release_first_close = threading.Event()

    async def synthesize(*, text: str) -> TTSAudioResponse:
        nonlocal active, maximum_active
        calls.append(text)
        active += 1
        maximum_active = max(maximum_active, active)

        async def stream() -> AsyncIterator[bytes]:
            yield text.encode()

        async def cleanup() -> None:
            nonlocal active
            if text == "first":
                first_close_started.set()
                while not release_first_close.is_set():
                    await asyncio.sleep(0.001)
            active -= 1

        return _response(stream(), cleanup=cleanup)

    bridge = OwnerLoopTtsSynthesizer(owner.loop, synthesize)
    first = await bridge.synthesize_hands_free(text="first")
    first_consumer = asyncio.create_task(_collect(first))
    assert await asyncio.to_thread(first_close_started.wait, 1)

    second_request = asyncio.create_task(bridge.synthesize_hands_free(text="second"))
    await asyncio.sleep(0.02)
    assert calls == ["first"]
    assert not first_consumer.done()
    assert not second_request.done()

    release_first_close.set()
    assert await first_consumer == (b"first",)
    second = await asyncio.wait_for(second_request, timeout=1)
    assert await _collect(second) == (b"second",)
    assert calls == ["first", "second"]
    assert maximum_active == 1
    await bridge.aclose()


@pytest.mark.asyncio
async def test_proxy_close_fences_a_full_channel_and_waits_for_owner_cleanup(
    owner: _LoopThread,
) -> None:
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsSynthesizer

    yielded = 0
    close_started = threading.Event()
    release_close = threading.Event()

    async def stream() -> AsyncIterator[bytes]:
        nonlocal yielded
        try:
            for index in range(100):
                yielded += 1
                yield bytes([index]) * 65_536
        finally:
            close_started.set()
            while not release_close.is_set():
                await asyncio.sleep(0.001)

    async def synthesize(*, text: str) -> TTSAudioResponse:
        return _response(stream())

    bridge = OwnerLoopTtsSynthesizer(owner.loop, synthesize)
    proxy = await bridge.synthesize_hands_free(text="fill")
    await _wait_for(lambda: yielded >= 9)
    await asyncio.sleep(0.02)
    assert yielded == 9

    closing = asyncio.create_task(proxy.aclose())
    assert await asyncio.to_thread(close_started.wait, 1)
    await asyncio.sleep(0)
    assert not closing.done()
    with pytest.raises(StopAsyncIteration):
        await anext(proxy.byte_stream)

    release_close.set()
    await asyncio.wait_for(closing, timeout=1)
    await bridge.aclose()


@pytest.mark.asyncio
async def test_cancel_before_owner_start_never_calls_synthesizer(
    owner: _LoopThread,
) -> None:
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsSynthesizer

    owner_blocked = threading.Event()
    release_owner = threading.Event()

    def block_owner() -> None:
        owner_blocked.set()
        assert release_owner.wait(1)

    owner.loop.call_soon_threadsafe(block_owner)
    assert await asyncio.to_thread(owner_blocked.wait, 1)
    calls: list[str] = []

    async def synthesize(*, text: str) -> TTSAudioResponse:
        calls.append(text)

        async def stream() -> AsyncIterator[bytes]:
            yield b"ok"

        return _response(stream())

    bridge = OwnerLoopTtsSynthesizer(owner.loop, synthesize)
    request = asyncio.create_task(bridge.synthesize_hands_free(text="cancelled"))
    await asyncio.sleep(0.01)
    request.cancel()
    await asyncio.sleep(0)
    release_owner.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(request, timeout=1)

    assert calls == []
    proxy = await bridge.synthesize_hands_free(text="current")
    assert await _collect(proxy) == (b"ok",)
    assert calls == ["current"]
    await bridge.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ("synthesize", "stream"))
async def test_owner_errors_cross_the_boundary_without_private_content(
    owner: _LoopThread,
    failure_phase: str,
) -> None:
    from tldw_chatbook.Chat.console_voice_tts_bridge import (
        OwnerLoopTtsError,
        OwnerLoopTtsSynthesizer,
    )

    secret = "private spoken text and provider-key-123"

    async def synthesize(*, text: str) -> TTSAudioResponse:
        if failure_phase == "synthesize":
            raise RuntimeError(secret)

        async def stream() -> AsyncIterator[bytes]:
            yield b"accepted"
            raise ValueError(secret)

        return _response(stream())

    bridge = OwnerLoopTtsSynthesizer(owner.loop, synthesize)
    if failure_phase == "synthesize":
        operation: Awaitable[object] = bridge.synthesize_hands_free(text="secret")
    else:
        proxy = await bridge.synthesize_hands_free(text="secret")

        async def consume() -> object:
            return tuple([chunk async for chunk in proxy.byte_stream])

        operation = consume()

    with pytest.raises(OwnerLoopTtsError) as caught:
        await operation
    assert str(caught.value) == "tts_bridge_failed"
    assert secret not in repr(caught.value)

    if failure_phase == "stream":
        with contextlib.suppress(OwnerLoopTtsError):
            await proxy.aclose()
    await bridge.aclose()


@pytest.mark.asyncio
async def test_bridge_close_cancels_active_response_and_rejects_new_work(
    owner: _LoopThread,
) -> None:
    from tldw_chatbook.Chat.console_voice_tts_bridge import (
        OwnerLoopTtsError,
        OwnerLoopTtsSynthesizer,
    )

    closed = threading.Event()

    async def stream() -> AsyncIterator[bytes]:
        try:
            yield b"one"
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def synthesize(*, text: str) -> TTSAudioResponse:
        return _response(stream())

    bridge = OwnerLoopTtsSynthesizer(owner.loop, synthesize)
    proxy = await bridge.synthesize_hands_free(text="active")
    first = await anext(proxy.byte_stream)

    await bridge.aclose()

    assert first == b"one"
    assert closed.is_set()
    with pytest.raises(StopAsyncIteration):
        await anext(proxy.byte_stream)
    with pytest.raises(OwnerLoopTtsError, match="tts_bridge_closed"):
        await bridge.synthesize_hands_free(text="late")


@pytest.mark.asyncio
async def test_cancelled_middle_phrase_cannot_bypass_predecessor_cleanup(
    owner: _LoopThread,
) -> None:
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsSynthesizer

    calls: list[str] = []
    first_close_started = threading.Event()
    release_first_close = threading.Event()

    async def synthesize(*, text: str) -> TTSAudioResponse:
        calls.append(text)

        async def stream() -> AsyncIterator[bytes]:
            yield text.encode()

        async def cleanup() -> None:
            if text == "first":
                first_close_started.set()
                while not release_first_close.is_set():
                    await asyncio.sleep(0.001)

        return _response(stream(), cleanup=cleanup)

    bridge = OwnerLoopTtsSynthesizer(owner.loop, synthesize)
    first = await bridge.synthesize_hands_free(text="first")
    first_consumer = asyncio.create_task(_collect(first))
    assert await asyncio.to_thread(first_close_started.wait, 1)
    middle_request = asyncio.create_task(
        bridge.synthesize_hands_free(text="cancelled-middle")
    )
    await asyncio.sleep(0.01)
    middle_request.cancel()
    last_request = asyncio.create_task(bridge.synthesize_hands_free(text="last"))
    await asyncio.sleep(0.02)
    try:
        assert calls == ["first"]
        assert not middle_request.done()
        assert not last_request.done()
    finally:
        release_first_close.set()

    assert await first_consumer == (b"first",)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(middle_request, timeout=1)
    last = await asyncio.wait_for(last_request, timeout=1)
    assert await _collect(last) == (b"last",)
    assert calls == ["first", "last"]
    await bridge.aclose()


@pytest.mark.asyncio
async def test_proxy_cancel_during_natural_close_waits_for_actual_cleanup(
    owner: _LoopThread,
) -> None:
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsSynthesizer

    close_started = threading.Event()
    close_completed = threading.Event()
    release_close = threading.Event()

    async def stream() -> AsyncIterator[bytes]:
        yield b"one"

    async def cleanup() -> None:
        close_started.set()
        while not release_close.is_set():
            await asyncio.sleep(0.001)
        close_completed.set()

    async def synthesize(*, text: str) -> TTSAudioResponse:
        return _response(stream(), cleanup=cleanup)

    bridge = OwnerLoopTtsSynthesizer(owner.loop, synthesize)
    proxy = await bridge.synthesize_hands_free(text="natural-close")
    assert await anext(proxy.byte_stream) == b"one"
    assert await asyncio.to_thread(close_started.wait, 1)

    closing = asyncio.create_task(proxy.aclose())
    await asyncio.sleep(0.02)
    try:
        assert not closing.done()
        assert not close_completed.is_set()
    finally:
        release_close.set()

    await asyncio.wait_for(closing, timeout=1)
    assert close_completed.is_set()
    await bridge.aclose()


@pytest.mark.asyncio
async def test_bridge_close_wakes_synthesis_waiting_for_metadata(
    owner: _LoopThread,
) -> None:
    from tldw_chatbook.Chat.console_voice_tts_bridge import (
        OwnerLoopTtsError,
        OwnerLoopTtsSynthesizer,
    )

    started = threading.Event()

    async def synthesize(*, text: str) -> TTSAudioResponse:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    bridge = OwnerLoopTtsSynthesizer(owner.loop, synthesize)
    request = asyncio.create_task(bridge.synthesize_hands_free(text="pending"))
    assert await asyncio.to_thread(started.wait, 1)

    await bridge.aclose()

    with pytest.raises(OwnerLoopTtsError, match="tts_bridge_closed"):
        await asyncio.wait_for(request, timeout=0.2)
