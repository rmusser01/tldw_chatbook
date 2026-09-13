"""Real synchronous gateway producer with private controlled provider iterators."""

import asyncio
import threading

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderCallPurpose,
    ConsoleProviderCallSignals,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)


class Source:
    def __init__(self, text="x" * 4096, *, fail=False):
        self.text = text
        self.pulls = 0
        self.closed = threading.Event()
        self.fail = fail

    def __iter__(self):
        return self

    def __next__(self):
        self.pulls += 1
        if self.pulls > 32:
            if self.fail:
                raise ValueError("private provider error")
            raise StopIteration
        return self.text

    def close(self):
        self.closed.set()


def stream_for(source, *, purpose=ConsoleProviderCallPurpose.VOICE_PROVISIONAL):
    gateway = ConsoleProviderGateway(
        http_client=object(),
        chat_api_call_fn=lambda **kwargs: source,
    )
    resolution = ConsoleProviderResolution(
        provider="qwencloud",
        base_url="",
        execution_key="qwencloud",
        model="private-fake",
        ready=True,
    )
    request = gateway.prepare_chat_request(
        resolution, [{"role": "user", "content": "private fixture"}]
    )
    signals = ConsoleProviderCallSignals(ConsoleProviderStreamSignals())
    return gateway.stream_chat(
        resolution, request, signals=signals, dispatch_purpose=purpose
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_voice_handoff_stops_source_before_next_pull_and_closes_when_full(fail):
    source = Source(fail=fail)
    stream = stream_for(source)
    try:
        assert await anext(stream, None) == "x" * 4096, source.pulls
        await asyncio.sleep(0.05)
        assert source.pulls == 1
    finally:
        await stream.aclose()
        assert await asyncio.to_thread(source.closed.wait, 1)


@pytest.mark.asyncio
async def test_voice_handoff_splits_utf8_without_advancing_source():
    source = Source("😀" * 3000)
    stream = stream_for(source)
    try:
        blocks = [await anext(stream) for _ in range(3)]
        assert [len(block.encode()) for block in blocks] == [4096, 4096, 3808]
        assert "".join(blocks) == source.text
        assert source.pulls == 1
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_voice_rejects_oversized_source_item_without_emitting_it():
    source = Source("x" * (262144 + 1))
    stream = stream_for(source)
    with pytest.raises(ChatProviderError):
        await anext(stream)
    assert source.closed.is_set()


@pytest.mark.asyncio
async def test_ordinary_typed_dispatch_keeps_original_chunk_shape():
    source = Source("😀" * 3000)
    stream = stream_for(source, purpose=ConsoleProviderCallPurpose.CONVERSATION)
    try:
        assert await anext(stream) == source.text
    finally:
        await stream.aclose()
