"""Real stream controls for bounded, complete Canvas JSON request reads."""

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import StreamReader, web
from aiohttp.base_protocol import BaseProtocol

from tldw_chatbook.Canvas.gateway import CanvasGateway


def _stream():
    loop = asyncio.get_running_loop()
    protocol = BaseProtocol(loop)
    protocol.connection_made(asyncio.Transport())
    return StreamReader(protocol, limit=1024, loop=loop)


def _read(stream):
    return CanvasGateway._read_json(
        SimpleNamespace(_max_request_bytes=64), SimpleNamespace(content=stream)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (1, 12, 13))
async def test_split_json_waits_for_the_complete_utf8_body(split):
    body = '{"value": "☃"}'.encode()
    stream = _stream()
    stream.feed_data(body[:split])
    task = asyncio.create_task(_read(stream))
    await asyncio.sleep(0)
    premature = task.done()
    stream.feed_data(body[split:])
    stream.feed_eof()
    assert await task == {"value": "☃"}
    assert not premature


@pytest.mark.asyncio
async def test_valid_json_prefix_does_not_hide_trailing_invalid_content():
    stream = _stream()
    stream.feed_data(b"{}")
    task = asyncio.create_task(_read(stream))
    await asyncio.sleep(0)
    stream.feed_data(b"trailing")
    stream.feed_eof()
    with pytest.raises(web.HTTPBadRequest):
        await task


@pytest.mark.asyncio
async def test_exact_limit_is_accepted_at_eof():
    stream = _stream()
    stream.feed_data(b'"' + b"x" * 62 + b'"')
    stream.feed_eof()
    assert await _read(stream) == "x" * 62


@pytest.mark.asyncio
async def test_oversized_chunked_body_is_refused_without_waiting_for_eof():
    stream = _stream()
    stream.feed_data(b'"' + b"x" * 63)
    task = asyncio.create_task(_read(stream))
    await asyncio.sleep(0)
    stream.feed_data(b"xTAIL")
    with pytest.raises(web.HTTPRequestEntityTooLarge):
        await asyncio.wait_for(task, 1)
    assert stream.read_nowait() == b"TAIL"


@pytest.mark.asyncio
@pytest.mark.parametrize("body", (b"", b"{", b'"\xff"'))
async def test_malformed_complete_body_remains_bad_request(body):
    stream = _stream()
    stream.feed_data(body)
    stream.feed_eof()
    with pytest.raises(web.HTTPBadRequest):
        await _read(stream)


@pytest.mark.asyncio
async def test_cancelled_partial_read_propagates_cancellation():
    stream = _stream()
    stream.feed_data(b"{")
    task = asyncio.create_task(_read(stream))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
