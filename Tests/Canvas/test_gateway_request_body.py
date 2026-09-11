"""Real stream controls for bounded, complete Canvas JSON request reads."""

import asyncio
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import aiohttp
import pytest
from aiohttp import StreamReader, web
from aiohttp.base_protocol import BaseProtocol

from tldw_chatbook.Canvas.gateway import CanvasGateway
from tldw_chatbook.Canvas import gateway as gateway_module


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
async def test_split_json_waits_for_the_complete_utf8_body(split: int) -> None:
    """Wait for the entire JSON body, including a split UTF-8 code point.

    Args:
        split: Byte offset separating the two incoming body chunks.
    """
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
async def test_valid_json_prefix_does_not_hide_trailing_invalid_content() -> None:
    """Reject trailing invalid content after an otherwise valid JSON prefix."""
    stream = _stream()
    stream.feed_data(b"{}")
    task = asyncio.create_task(_read(stream))
    await asyncio.sleep(0)
    stream.feed_data(b"trailing")
    stream.feed_eof()
    with pytest.raises(web.HTTPBadRequest):
        await task


@pytest.mark.asyncio
async def test_exact_limit_is_accepted_at_eof() -> None:
    """Accept a complete JSON body exactly at the request size limit."""
    stream = _stream()
    stream.feed_data(b'"' + b"x" * 62 + b'"')
    stream.feed_eof()
    assert await _read(stream) == "x" * 62


@pytest.mark.asyncio
async def test_oversized_chunked_body_is_refused_without_waiting_for_eof() -> None:
    """Reject oversized chunked input before EOF without consuming its tail."""
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
async def test_malformed_complete_body_remains_bad_request(body: bytes) -> None:
    """Reject complete bodies containing empty, invalid JSON, or invalid UTF-8 input.

    Args:
        body: Malformed bytes supplied as the complete request body.
    """
    stream = _stream()
    stream.feed_data(body)
    stream.feed_eof()
    with pytest.raises(web.HTTPBadRequest):
        await _read(stream)


@pytest.mark.asyncio
async def test_cancelled_partial_read_propagates_cancellation() -> None:
    """Propagate cancellation while waiting for the rest of a partial body."""
    stream = _stream()
    stream.feed_data(b"{")
    task = asyncio.create_task(_read(stream))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial,trickle", [(b"", False), (b"{", False), (b"{}", False), (b"{", True)]
)
async def test_unfinished_body_has_an_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch, initial: bytes, trickle: bool
) -> None:
    """Partial progress cannot extend the total body-read deadline.

    Args:
        monkeypatch: Shortens only the fixed body deadline for this control.
        initial: Body bytes available when reading starts.
        trickle: Whether more bytes arrive before and after the deadline.
    """
    monkeypatch.setattr(
        gateway_module, "_REQUEST_BODY_TIMEOUT_SECONDS", 0.1, raising=False
    )
    stream = _stream()
    stream.feed_data(initial)
    task = asyncio.create_task(_read(stream))
    receipts = []

    def feed():
        receipts.append(True)
        stream.feed_data(b" ")

    handles = (
        [
            asyncio.get_running_loop().call_later(index * 0.02, feed)
            for index in range(1, 13)
        ]
        if trickle
        else []
    )
    try:
        with pytest.raises(web.HTTPRequestTimeout) as caught:
            await asyncio.wait_for(task, 0.2)
        assert caught.value.keep_alive is False
        assert not stream.is_eof()
        if trickle:
            assert len(receipts) >= 2
    finally:
        for handle in handles:
            handle.cancel()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.loopback_network
async def test_gateway_stalled_body_returns_private_nonreusable_408(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real gateway sanitizes a timed-out body and preserves close policy.

    Args:
        monkeypatch: Shortens the fixed body deadline for the loopback control.
    """
    from Tests.Canvas.test_gateway import _Authority, _scope

    monkeypatch.setattr(
        gateway_module, "_REQUEST_BODY_TIMEOUT_SECONDS", 0.05, raising=False
    )
    authority = _Authority([])
    gateway = CanvasGateway(authority=authority)
    writer = None
    try:
        launch = await gateway.open_shell(_scope())
        url = urlsplit(launch.clean_url)
        origin = f"{url.scheme}://{url.netloc}"
        reader, writer = await asyncio.open_connection(url.hostname, url.port)
        partial = b'{"private":"CANARY-BODY'
        writer.write(
            f"POST {url.path}api/boot HTTP/1.1\r\nHost: {url.netloc}\r\n"
            f"Origin: {origin}\r\nContent-Type: application/json\r\n"
            "Transfer-Encoding: chunked\r\n\r\n".encode()
            + f"{len(partial):x}\r\n".encode()
            + partial
            + b"\r\n"
        )
        await writer.drain()
        headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 1)
        assert headers.startswith(b"HTTP/1.1 408 ")
        fields = dict(line.split(b": ", 1) for line in headers.split(b"\r\n")[1:-2])
        body = await asyncio.wait_for(
            reader.readexactly(int(fields[b"Content-Length"])), 1
        )
        assert json.loads(body) == {"error": "request_refused"}
        assert b"CANARY-BODY" not in headers + body
        assert fields[b"Connection"] == b"close"
        assert fields[b"Cache-Control"] == b"no-store"
        assert fields[b"X-Content-Type-Options"] == b"nosniff"
        assert fields[b"Referrer-Policy"] == b"no-referrer"
        assert b"Content-Security-Policy" in fields
        assert authority.calls == []
        writer.close()
        await writer.wait_closed()
        writer = None

        # The incomplete request did not consume its bootstrap capability.
        token = parse_qs(urlsplit(launch.browser_url).fragment)["boot"][0]
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{launch.clean_url}api/boot",
                json={"bootstrap": token},
                headers={"Origin": origin},
            ) as response:
                assert response.status == 200
                assert (await response.json())["browser_session_id"] == "browser-a"
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        await gateway.aclose()
