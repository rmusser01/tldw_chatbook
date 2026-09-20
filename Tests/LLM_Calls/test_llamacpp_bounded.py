"""Qualification of one captured llama.cpp request on an owned loopback peer."""

import asyncio
import gzip
import json
import logging
import socket
import traceback
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError

import httpx
import pytest


def _answer(content="review me", *, usage=True, finish_reason="stop"):
    document = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
    }
    if usage:
        document["usage"] = {"prompt_tokens": 3, "completion_tokens": 2}
    return json.dumps(document, ensure_ascii=False).encode()


def _response(body, *, status=200, headers=b""):
    return (
        f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n".encode()
        + f"Content-Length: {len(body)}\r\nConnection: close\r\n".encode()
        + headers
        + b"\r\n"
        + body
    )


@asynccontextmanager
async def _listener(response, *, ssl_context=None):
    """Own, retain and join every accepted connection, even after a failed test."""
    requests = []
    tasks = []

    async def handle(reader, writer):
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            first, *lines = header.decode("ascii").strip().split("\r\n")
            headers = dict(line.lower().split(": ", 1) for line in lines)
            body = await reader.readexactly(int(headers["content-length"]))
            requests.append((first, headers, json.loads(body)))
            if callable(response):
                await response(reader, writer)
            else:
                writer.write(response)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    def accept(reader, writer):
        tasks.append(asyncio.create_task(handle(reader, writer)))

    async with await asyncio.start_server(
        accept, "127.0.0.1", 0, ssl=ssl_context
    ) as server:
        port = server.sockets[0].getsockname()[1]
        try:
            scheme = "https" if ssl_context else "http"
            yield f"{scheme}://127.0.0.1:{port}", requests
        finally:
            server.close()
            await server.wait_closed()
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, (asyncio.CancelledError, ConnectionError)
                ):
                    raise result


def _request(origin, **changes):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import BoundedLlamaRequest

    values = {
        "provider_id": "my_custom_llama",
        "selected_url": origin + "/proxy/v1",
        "dispatch_url": origin + "/proxy/v1/chat/completions",
        "model": "captured-model",
        "prompt": "résumé",
        "max_tokens": 512,
        "request_timeout_seconds": 2.0,
        "sampling": (("temperature", 0.25), ("seed", 7)),
    }
    return BoundedLlamaRequest(**(values | changes))


def test_reservation_counts_utf8_and_full_output_allowance():
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import estimate_llama_reservation

    assert estimate_llama_reservation("é", max_tokens=512) == 578


@pytest.mark.loopback_network
async def test_captured_request_reaches_real_endpoint_despite_live_config_edits(
    monkeypatch,
):
    from tldw_chatbook import config
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import complete_llama_bounded

    async with _listener(_response(_answer())) as (origin, requests):
        request = _request(origin)
        monkeypatch.setitem(
            config.settings,
            "api_settings",
            {
                "llama_cpp": {
                    "api_url": "http://127.0.0.1:1",
                    "api_key": "KEY-CANARY",
                    "model": "changed-model",
                    "max_retries": 4,
                    "request_timeout": 0.001,
                    "top_p": 0.1,
                },
            },
        )
        monkeypatch.setitem(
            config.settings, "providers", {"llama_cpp": ["changed-model"]}
        )
        result = await complete_llama_bounded(
            request, deadline_at=asyncio.get_running_loop().time() + 3
        )

    assert result == {
        "text": "review me",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    assert request.provider_id == "my_custom_llama"
    assert len(requests) == 1
    first, headers, payload = requests[0]
    assert first == "POST /proxy/v1/chat/completions HTTP/1.1"
    assert headers["host"] == origin.removeprefix("http://")
    assert headers["accept-encoding"] == "identity"
    assert "authorization" not in headers
    assert payload == {
        "model": "captured-model",
        "messages": [{"role": "user", "content": "résumé"}],
        "max_tokens": 512,
        "stream": False,
        "temperature": 0.25,
        "seed": 7,
    }


def test_request_is_frozen_and_private_fields_are_absent_from_repr():
    request = _request("http://127.0.0.1:9099", prompt="PROMPT-CANARY")
    with pytest.raises(FrozenInstanceError):
        request.model = "other-model"
    assert "127.0.0.1" not in repr(request)
    assert "PROMPT-CANARY" not in repr(request)


@pytest.mark.parametrize(
    "max_tokens", [True, False, 0, -1, 1.5, float("inf"), float("nan"), "512"]
)
def test_reservation_rejects_invalid_output_budget(max_tokens):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        estimate_llama_reservation,
    )

    with pytest.raises(BoundedLlamaError, match="^invalid_request$"):
        estimate_llama_reservation("prompt", max_tokens=max_tokens)


@pytest.mark.parametrize("prompt", [None, 12, "\ud800"])
def test_reservation_rejects_non_utf8_prompt(prompt):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        estimate_llama_reservation,
    )

    with pytest.raises(BoundedLlamaError, match="^invalid_request$"):
        estimate_llama_reservation(prompt, max_tokens=512)


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ("127.0.0.1:9099", "http://127.0.0.1:9099/v1/chat/completions"),
        ("https://[::1]:9099/proxy/v1", "https://[::1]:9099/proxy/v1/chat/completions"),
        (
            "http://127.0.0.2:9099/completion",
            "http://127.0.0.2:9099/v1/chat/completions",
        ),
    ],
)
def test_numeric_setup_uses_endpoint_contract_without_dns(
    selected, expected, monkeypatch
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import resolve_llama_loopback_url

    def forbidden(*args, **kwargs):
        pytest.fail("numeric setup must not resolve DNS")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    assert resolve_llama_loopback_url(selected) == expected


@pytest.mark.parametrize("address", ["127.0.0.1", "::1"])
def test_localhost_setup_pins_one_address(address, monkeypatch):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import resolve_llama_loopback_url

    calls = []

    def resolve(host, port, **kwargs):
        calls.append((host, port))
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    host = f"[{address}]" if ":" in address else address
    assert (
        resolve_llama_loopback_url("localhost:9099/proxy")
        == f"http://{host}:9099/proxy/v1/chat/completions"
    )
    assert calls == [("localhost", 9099)]


@pytest.mark.parametrize(
    "addresses",
    [
        [],
        ["127.0.0.1", "192.0.2.1"],
        ["::1", "127.0.0.1", "192.0.2.1"],
        ["::ffff:127.0.0.1"],
    ],
)
def test_localhost_setup_refuses_any_nonloopback_answer(addresses, monkeypatch):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        resolve_llama_loopback_url,
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 9099))
            for address in addresses
        ],
    )
    with pytest.raises(BoundedLlamaError, match="^invalid_endpoint$"):
        resolve_llama_loopback_url("localhost:9099")


@pytest.mark.parametrize(
    "selected",
    [
        "http://example.test",
        "http://192.0.2.1",
        "http://localhost.example.test",
        "http://user:SECRET@127.0.0.1",
        "http://127.0.0.1?",
        "http://127.0.0.1#",
        "http://[::1%25lo0]",
        "file:///tmp/model",
        "http://127.1",
        "http://2130706433",
    ],
)
def test_unsafe_setup_url_is_rejected_without_dns(selected, monkeypatch):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        resolve_llama_loopback_url,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("only literal localhost can use setup DNS")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    with pytest.raises(BoundedLlamaError, match="^invalid_endpoint$"):
        resolve_llama_loopback_url(selected)


@pytest.mark.parametrize(
    "changes",
    [
        {"provider_id": ""},
        {"provider_id": None},
        {"model": ""},
        {"model": "\ud800"},
        {"prompt": "\ud800"},
        {"max_tokens": True},
        {"max_tokens": 0},
        {"request_timeout_seconds": True},
        {"request_timeout_seconds": 0},
        {"request_timeout_seconds": float("nan")},
        {"request_timeout_seconds": float("inf")},
        {"request_timeout_seconds": 10**1000},
        {"sampling": (("temperature", True),)},
        {"sampling": (("temperature", float("nan")),)},
        {"sampling": (("temperature", 10**1000),)},
        {"sampling": (("temperature", 3),)},
        {"sampling": (("top_p", 1.1),)},
        {"sampling": (("min_p", -0.1),)},
        {"sampling": (("seed", -1),)},
        {"sampling": (("top_k", 2.5),)},
        {"sampling": (("tools", 1),)},
        {"sampling": (("max_tokens", 8),)},
        {"sampling": (("reasoning_effort", "low"),)},
        {"sampling": (("presence_penalty", 0),)},
        {"sampling": (("seed", 1), ("seed", 2))},
        {"sampling": [("seed", 1)]},
        {"sampling": (["seed", 1],)},
        {"sampling": (([], 1),)},
        {"dispatch_url": "http://localhost:9099/proxy/v1/chat/completions"},
        {"dispatch_url": "http://192.0.2.1:9099/proxy/v1/chat/completions"},
        {"dispatch_url": "http://127.0.0.1:9099/proxy"},
        {"dispatch_url": "http://127.0.0.1:9099/other/v1/chat/completions"},
        {"dispatch_url": "https://127.0.0.1:9099/proxy/v1/chat/completions"},
        {"dispatch_url": "http://127.0.0.1:9098/proxy/v1/chat/completions"},
        {"dispatch_url": "http://user:SECRET@127.0.0.1:9099/proxy/v1/chat/completions"},
        {"dispatch_url": "http://127.0.0.1:9099/proxy/v1/chat/completions?SECRET"},
        {"selected_url": "http://192.0.2.1:9099/proxy"},
    ],
)
async def test_invalid_request_fails_before_transport_construction(
    changes, monkeypatch
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid request reached transport construction")

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", forbidden)
    with pytest.raises(BoundedLlamaError):
        await complete_llama_bounded(
            _request("http://127.0.0.1:9099", **changes),
            deadline_at=asyncio.get_running_loop().time() + 3,
        )


@pytest.mark.parametrize(
    "deadline", [True, float("nan"), float("inf"), 10**1000, "soon", -1.0]
)
async def test_invalid_or_expired_deadline_never_constructs_transport(
    deadline, monkeypatch
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid/expired deadline reached transport construction")

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", forbidden)
    with pytest.raises(BoundedLlamaError):
        await complete_llama_bounded(
            _request("http://127.0.0.1:9099"), deadline_at=deadline
        )


@pytest.mark.loopback_network
@pytest.mark.parametrize("status", [429, 500, 307])
async def test_failure_status_causes_exactly_one_post_without_following_redirect(
    status,
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    async with _listener(_response(_answer())) as (redirect, redirected):
        headers = f"Location: {redirect}/redirected\r\n".encode()
        async with _listener(
            _response(b"ERROR-BODY-CANARY", status=status, headers=headers)
        ) as (origin, requests):
            with pytest.raises(BoundedLlamaError, match="^http_status$") as caught:
                await complete_llama_bounded(
                    _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
                )
    assert len(requests) == 1
    assert redirected == []
    assert "ERROR-BODY-CANARY" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.loopback_network
async def test_disconnect_is_payload_free_and_does_not_retry():
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    async with _listener(b"") as (origin, requests):
        with pytest.raises(BoundedLlamaError, match="^transport$"):
            await complete_llama_bounded(
                _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
            )
    assert len(requests) == 1


@pytest.mark.loopback_network
async def test_poisoned_proxy_environment_is_ignored(monkeypatch):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import complete_llama_bounded

    async with _listener(_response(b"proxy should not be used", status=500)) as (
        proxy,
        proxied,
    ):
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            monkeypatch.setenv(name, proxy)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        async with _listener(_response(_answer())) as (origin, requests):
            result = await complete_llama_bounded(
                _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
            )
    assert result["text"] == "review me"
    assert len(requests) == 1
    assert proxied == []


@pytest.mark.loopback_network
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("extra", [0, 1])
async def test_raw_body_limit_at_one_mib_before_json_parsing(chunked, extra):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    body = _answer()
    body += b" " * (1024 * 1024 + extra - len(body))
    if chunked:
        response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        for start in range(0, len(body), 65536):
            chunk = body[start : start + 65536]
            response += f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"
        response += b"0\r\n\r\n"
    else:
        response = _response(body)
    async with _listener(response) as (origin, requests):
        call = complete_llama_bounded(
            _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
        )
        if extra:
            with pytest.raises(BoundedLlamaError, match="^response_too_large$"):
                await call
        else:
            assert (await call)["text"] == "review me"
    assert len(requests) == 1


@pytest.mark.loopback_network
async def test_compressed_body_is_rejected_without_decompression():
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    response = _response(
        gzip.compress(_answer()), headers=b"Content-Encoding: gzip\r\n"
    )
    async with _listener(response) as (origin, _):
        with pytest.raises(BoundedLlamaError, match="^response_encoding$"):
            await complete_llama_bounded(
                _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
            )


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("café 🍵", "café 🍵"),
        ("<think>PRIVATE</think>\nreview me", "review me"),
        (" \n<thinking>PRIVATE</thinking>\nreview me", "review me"),
        ("explain <think>literal</think>", "explain <think>literal</think>"),
    ],
)
async def test_visible_answer_uses_only_start_anchored_thinking_filter(
    content, expected
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import complete_llama_bounded

    async with _listener(_response(_answer(content))) as (origin, _):
        result = await complete_llama_bounded(
            _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
        )
    assert result["text"] == expected


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"\xff",
        b"[]",
        b"{}",
        b'{"choices":[]}',
        b'{"choices":[null]}',
        b'{"choices":[{"text":"legacy only"}]}',
        _answer(None),
        _answer([]),
        _answer(""),
        _answer(" \n"),
        _answer("<think>unfinished"),
        _answer("<think>private</think>"),
        _answer("truncated", finish_reason="length"),
        _answer("maybe", finish_reason=None),
        _answer("refused", finish_reason="content_filter"),
        b'{"choices":[{"message":{"content":"answer", "tool_calls":[{}]},"finish_reason":"stop"}]}',
        b'{"choices":[{"message":{"content":"answer", "function_call":{}},"finish_reason":"stop"}]}',
        b'{"choices":[{"message":{"content":"\\ud800"},"finish_reason":"stop"}]}',
    ],
)
async def test_malformed_or_incomplete_answer_cannot_succeed(body):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    async with _listener(_response(body)) as (origin, _):
        with pytest.raises(BoundedLlamaError):
            await complete_llama_bounded(
                _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
            )


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        [],
        {"prompt_tokens": 1},
        {"prompt_tokens": True, "completion_tokens": 2},
        {"prompt_tokens": 1, "completion_tokens": -2},
        {"prompt_tokens": 1.5, "completion_tokens": 2},
        {"prompt_tokens": "1", "completion_tokens": 2},
    ],
)
async def test_missing_or_invalid_usage_is_none_not_zero(usage):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import complete_llama_bounded

    document = json.loads(_answer(usage=False))
    if usage is not None:
        document["usage"] = usage
    async with _listener(_response(json.dumps(document).encode())) as (origin, _):
        result = await complete_llama_bounded(
            _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
        )
    assert result == {"text": "review me", "usage": None}


@pytest.mark.loopback_network
@pytest.mark.parametrize("shorter", ["attempt", "request"])
async def test_continuous_trickle_hits_absolute_deadline_and_closes_connection(shorter):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    closed = asyncio.Event()
    chunks = []

    async def trickle(reader, writer):
        writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        while True:
            writer.write(b"1\r\n \r\n")
            await writer.drain()
            chunks.append(1)
            try:
                if await asyncio.wait_for(reader.read(1), 0.02) == b"":
                    closed.set()
                    return
            except TimeoutError:
                continue

    async with _listener(trickle) as (origin, requests):
        start = asyncio.get_running_loop().time()
        request = _request(
            origin, request_timeout_seconds=0.25 if shorter == "request" else 2.0
        )
        deadline = start + (0.25 if shorter == "attempt" else 2.0)
        with pytest.raises(BoundedLlamaError, match="^deadline$"):
            await complete_llama_bounded(request, deadline_at=deadline)
        await asyncio.wait_for(closed.wait(), 1)
        assert asyncio.get_running_loop().time() - start < 1.5
    assert len(chunks) >= 2
    assert len(requests) == 1


@pytest.mark.loopback_network
async def test_cancelled_body_closes_real_connection_before_return():
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import complete_llama_bounded

    started, closed = asyncio.Event(), asyncio.Event()

    async def hold(reader, writer):
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n ")
        await writer.drain()
        started.set()
        assert await reader.read() == b""
        closed.set()

    async with _listener(hold) as (origin, requests):
        task = asyncio.create_task(
            complete_llama_bounded(
                _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closed.wait(), 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert len(requests) == 1


@pytest.mark.loopback_network
@pytest.mark.parametrize("trigger", ["success", "deadline", "cancel"])
@pytest.mark.parametrize("cancel_during_cleanup", [False, True])
async def test_held_cleanup_retains_operation_through_repeated_cancellation(
    trigger, cancel_during_cleanup, monkeypatch
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    started, cleanup_started, release, cleaned = (asyncio.Event() for _ in range(4))
    original_close = httpx.AsyncClient.aclose

    async def held_close(client):
        cleanup_started.set()
        await release.wait()
        await original_close(client)
        cleaned.set()

    async def peer(reader, writer):
        started.set()
        if trigger == "success":
            writer.write(_response(_answer()))
            await writer.drain()
        else:
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n ")
            await writer.drain()
            await reader.read()

    monkeypatch.setattr(httpx.AsyncClient, "aclose", held_close)
    async with _listener(peer) as (origin, _):
        task = asyncio.create_task(
            complete_llama_bounded(
                _request(origin, request_timeout_seconds=0.15),
                deadline_at=asyncio.get_running_loop().time() + 3,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            if trigger == "cancel":
                task.cancel()
            await asyncio.wait_for(cleanup_started.wait(), 1)
            # The request deadline covers the body, not final cleanup; a body
            # completed in time must still succeed after delayed final cleanup.
            await asyncio.wait({task}, timeout=0.2)
            assert not task.done()
            assert not cleaned.is_set()
            if cancel_during_cleanup:
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done()
                    assert not cleaned.is_set()
            release.set()
            if cancel_during_cleanup or trigger == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif trigger == "deadline":
                with pytest.raises(BoundedLlamaError, match="^deadline$"):
                    await task
            else:
                assert (await task)["text"] == "review me"
            assert cleaned.is_set()
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.loopback_network
@pytest.mark.parametrize("status", [200, 500])
async def test_private_transport_canaries_do_not_leak_while_concurrent_logs_survive(
    status, caplog
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    caplog.set_level(logging.DEBUG)
    levels = {
        name: logging.getLogger(name).level for name in ("httpx", "httpcore.http11")
    }

    async def peer(reader, writer):
        logging.getLogger("httpcore.http11").debug("unrelated-concurrent-diagnostic")
        writer.write(
            _response(
                _answer("RESPONSE-CANARY"),
                status=status,
                headers=b"X-Private: HEADER-CANARY\r\n",
            )
        )
        await writer.drain()

    async with _listener(peer) as (origin, _):
        request = _request(
            origin,
            selected_url=origin + "/PATH-CANARY",
            dispatch_url=origin + "/PATH-CANARY/v1/chat/completions",
            prompt="PROMPT-CANARY",
        )
        try:
            await complete_llama_bounded(
                request, deadline_at=asyncio.get_running_loop().time() + 3
            )
        except BoundedLlamaError:
            assert status == 500
    assert "unrelated-concurrent-diagnostic" in caplog.text
    for canary in ("PROMPT-CANARY", "RESPONSE-CANARY", "PATH-CANARY", "HEADER-CANARY"):
        assert canary not in caplog.text
    assert levels == {name: logging.getLogger(name).level for name in levels}


@pytest.mark.loopback_network
async def test_deadline_during_physical_close_does_not_orphan_a_keepalive_connection(
    monkeypatch,
):
    from httpcore._backends.anyio import AnyIOStream

    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    entered, release, peer_closed = (asyncio.Event() for _ in range(3))
    streams = []
    original_close = AnyIOStream.aclose

    async def held_close(stream):
        streams.append(stream)
        entered.set()
        await release.wait()
        await original_close(stream)

    async def peer(reader, writer):
        writer.write(
            _response(_answer()).replace(
                b"Connection: close", b"Connection: keep-alive"
            )
        )
        await writer.drain()
        assert await reader.read() == b""
        peer_closed.set()

    monkeypatch.setattr(AnyIOStream, "aclose", held_close)
    async with _listener(peer) as (origin, _):
        task = asyncio.create_task(
            complete_llama_bounded(
                _request(origin, request_timeout_seconds=0.15),
                deadline_at=asyncio.get_running_loop().time() + 3,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0.25)
            assert not task.done(), (
                "request returned while its actual socket close was held"
            )
            assert not peer_closed.is_set()
            release.set()
            with pytest.raises(BoundedLlamaError, match="^deadline$"):
                await task
            await asyncio.wait_for(peer_closed.wait(), 1)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            for stream in streams:
                await original_close(stream)


@pytest.mark.loopback_network
async def test_deadline_waiting_for_headers_closes_the_peer():
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    closed = asyncio.Event()

    async def hold_headers(reader, writer):
        assert await reader.read() == b""
        closed.set()

    async with _listener(hold_headers) as (origin, requests):
        with pytest.raises(BoundedLlamaError, match="^deadline$"):
            await complete_llama_bounded(
                _request(origin, request_timeout_seconds=0.15),
                deadline_at=asyncio.get_running_loop().time() + 3,
            )
        await asyncio.wait_for(closed.wait(), 1)
    assert len(requests) == 1


@pytest.mark.loopback_network
async def test_setup_pin_survives_later_dns_change_without_resolving_at_dispatch(
    monkeypatch,
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        complete_llama_bounded,
        resolve_llama_loopback_url,
    )

    def initial_dns(host, port, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.2", port)),
        ]

    def poisoned_dns(*args, **kwargs):
        pytest.fail("dispatch must use its previously approved numeric address")

    async with _listener(_response(_answer())) as (origin, requests):
        selected = origin.replace("127.0.0.1", "localhost") + "/proxy"
        monkeypatch.setattr(socket, "getaddrinfo", initial_dns)
        pinned = resolve_llama_loopback_url(selected)
        request = _request(origin, selected_url=selected, dispatch_url=pinned)
        monkeypatch.setattr(socket, "getaddrinfo", poisoned_dns)
        assert (
            await complete_llama_bounded(
                request, deadline_at=asyncio.get_running_loop().time() + 3
            )
        )["text"] == "review me"
    assert len(requests) == 1
    assert requests[0][1]["host"] == origin.removeprefix("http://")


@pytest.mark.loopback_network
async def test_https_does_not_disable_certificate_verification(tmp_path):
    import ssl
    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "certificate.pem", tmp_path / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)

    async with _listener(_response(_answer()), ssl_context=server_context) as (
        origin,
        requests,
    ):
        # A test-only unverified control proves this exact TLS peer can serve a
        # completion. Production must refuse the same untrusted/mismatched cert.
        async with httpx.AsyncClient(verify=False, trust_env=False) as control:
            assert (await control.post(origin + "/control", json={})).status_code == 200
        with pytest.raises(BoundedLlamaError, match="^transport$"):
            await complete_llama_bounded(
                _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
            )
    assert len(requests) == 1
    assert requests[0][0] == "POST /control HTTP/1.1"


@pytest.mark.loopback_network
async def test_response_cleanup_failure_still_closes_client_and_has_no_private_error(
    monkeypatch,
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    closed = asyncio.Event()

    async def peer(reader, writer):
        writer.write(b"HTTP/1.1 500 Test\r\nContent-Length: 100\r\n\r\n ")
        await writer.drain()
        assert await reader.read() == b""
        closed.set()

    async def broken_response_close(response):
        raise RuntimeError("CLOSE-ERROR-CANARY")

    monkeypatch.setattr(httpx.Response, "aclose", broken_response_close)
    async with _listener(peer) as (origin, _):
        with pytest.raises(BoundedLlamaError, match="^cleanup$") as caught:
            await complete_llama_bounded(
                _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
            )
        await asyncio.wait_for(closed.wait(), 1)
    assert "CLOSE-ERROR-CANARY" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    "body",
    [
        b"RESPONSE-CANARY not json",
        b"\xffRESPONSE-CANARY",
        b'{"choices":[{"message":{"content":"RESPONSE-CANARY\\ud800"},"finish_reason":"stop"}]}',
    ],
)
async def test_parse_errors_retain_no_payload_bearing_exception_chain(body):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    async with _listener(_response(body)) as (origin, _):
        with pytest.raises(BoundedLlamaError) as caught:
            await complete_llama_bounded(
                _request(origin), deadline_at=asyncio.get_running_loop().time() + 3
            )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.args == ("malformed_response",)


async def test_invalid_model_retains_no_unicode_exception_payload():
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    with pytest.raises(BoundedLlamaError) as caught:
        await complete_llama_bounded(
            _request("http://127.0.0.1:9099", model="MODEL-CANARY\ud800"),
            deadline_at=asyncio.get_running_loop().time() + 3,
        )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("addresses", [("::1", "127.0.0.1"), ("127.0.0.1", "::1")])
def test_localhost_dual_stack_setup_prefers_ipv4_without_probing(
    addresses, monkeypatch
):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import resolve_llama_loopback_url

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (address, 9099),
            )
            for address in addresses
        ],
    )
    # The ordinary no-network fixture prohibits probing either answer here.
    assert (
        resolve_llama_loopback_url("localhost:9099")
        == "http://127.0.0.1:9099/v1/chat/completions"
    )


@pytest.mark.loopback_network
@pytest.mark.parametrize("phase", ["headers", "body"])
@pytest.mark.parametrize("trigger", ["caller", "request_deadline", "attempt_deadline"])
@pytest.mark.parametrize("late_cancellation", [False, True])
async def test_repeated_cancellation_retains_internal_physical_close(
    phase, trigger, late_cancellation, monkeypatch
):
    from httpcore._backends.anyio import AnyIOStream

    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    waiting, closing, release, peer_closed = (asyncio.Event() for _ in range(4))
    streams = []
    read_count = 0
    original_read, original_close = AnyIOStream.read, AnyIOStream.aclose

    async def observed_read(stream, *args, **kwargs):
        nonlocal read_count
        read_count += 1
        if phase == "headers" or read_count >= 2:
            waiting.set()
        return await original_read(stream, *args, **kwargs)

    async def held_close(stream):
        streams.append(stream)
        closing.set()
        await release.wait()
        await original_close(stream)

    async def peer(reader, writer):
        if phase == "body":
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n ")
            await writer.drain()
        assert await reader.read() == b""
        peer_closed.set()

    monkeypatch.setattr(AnyIOStream, "read", observed_read)
    monkeypatch.setattr(AnyIOStream, "aclose", held_close)
    async with _listener(peer) as (origin, requests):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (0.2 if trigger == "attempt_deadline" else 0.4)
        task = asyncio.create_task(
            complete_llama_bounded(
                _request(
                    origin,
                    request_timeout_seconds=0.2
                    if trigger == "request_deadline"
                    else 0.4,
                ),
                deadline_at=deadline,
            )
        )
        try:
            await asyncio.wait_for(waiting.wait(), 1)
            if trigger == "caller":
                task.cancel()
            await asyncio.wait_for(closing.wait(), 1)
            if late_cancellation:
                task.cancel()
            # Give the cancellation its full cleanup path without cancelling the
            # observed task through the observer. Also cross the attempt deadline.
            await asyncio.wait({task}, timeout=max(0, deadline - loop.time()) + 0.05)
            if late_cancellation:
                task.cancel()
            await asyncio.wait({task}, timeout=0.05)
            observed = {
                "returned_before_release": task.done(),
                "peer_closed_before_release": peer_closed.is_set(),
                "close_calls": len(streams),
            }
            assert observed == {
                "returned_before_release": False,
                "peer_closed_before_release": False,
                "close_calls": 1,
            }
            release.set()
            if trigger == "caller" or late_cancellation:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(BoundedLlamaError, match="^deadline$"):
                    await task
            await asyncio.wait_for(peer_closed.wait(), 1)
            assert len(requests) == 1
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            for stream in streams:
                await original_close(stream)


@pytest.mark.loopback_network
@pytest.mark.parametrize("trigger", ["caller", "deadline"])
async def test_stopped_transport_keeps_cleanup_failure_observable(trigger, monkeypatch):
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
        BoundedLlamaError,
        complete_llama_bounded,
    )

    started, closing, release, peer_closed = (asyncio.Event() for _ in range(4))
    original_close = httpx.AsyncClient.aclose

    async def failed_close(client):
        await original_close(client)
        closing.set()
        await release.wait()
        raise RuntimeError("CLEANUP-DETAIL-CANARY")

    async def peer(reader, writer):
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n ")
        await writer.drain()
        started.set()
        assert await reader.read() == b""
        peer_closed.set()

    monkeypatch.setattr(httpx.AsyncClient, "aclose", failed_close)
    async with _listener(peer) as (origin, _):
        task = asyncio.create_task(
            complete_llama_bounded(
                _request(origin, request_timeout_seconds=0.15),
                deadline_at=asyncio.get_running_loop().time() + 3,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            if trigger == "caller":
                task.cancel()
            await asyncio.wait_for(closing.wait(), 1)
            task.cancel()
            await asyncio.wait({task}, timeout=0.05)
            assert not task.done()
            release.set()
            with pytest.raises(BoundedLlamaError, match="^cleanup$") as caught:
                await task
            await asyncio.wait_for(peer_closed.wait(), 1)
            assert "CLEANUP-DETAIL-CANARY" not in "".join(
                traceback.format_exception(caught.value)
            )
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
