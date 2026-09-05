"""Contract tests for the bounded, loopback-only llama.cpp snapshot client."""

from __future__ import annotations

import asyncio
import gzip
import json
from collections.abc import Callable
from typing import Self

import httpx
import pytest
from pydantic import ValidationError

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)
from tldw_chatbook.LLM_Management.snapshot_models import (
    LaunchDescriptor,
    SlotReceipt,
    SnapshotError,
)


def _descriptor(
    base_url: str = "http://127.0.0.1:8080", *, token: str | None = "test-token"
) -> LaunchDescriptor:
    return LaunchDescriptor(
        launch_id="launch-test",
        claim=ServerLaunchClaim(provider="llamacpp", authority="External GGUF"),
        base_url=base_url,
        bearer_token=token,
        child_env={},
        files=(),
        compatibility=None,
        disabled_reason=None,
    )


def _response(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok", "prompt": "do-not-retain"})
    if request.url.path == "/props":
        return httpx.Response(
            200,
            json={
                "build_info": "build-427291b",
                "model_path": "/models/test.gguf",
                "default_generation_settings": {"prompt": "do-not-retain"},
                "flash_attn": True,
                "device": "invented-runtime-evidence",
            },
        )
    if request.url.path == "/slots":
        return httpx.Response(
            200,
            json=[
                {
                    "id": 0,
                    "is_processing": False,
                    "n_ctx": 4096,
                    "params": {"prompt": "do-not-retain"},
                },
                {"id": 1},
            ],
        )
    raise AssertionError(f"unexpected test route: {request.url.path}")


def _exception_graph_text(error: BaseException) -> str:
    values: list[str] = []
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        values.extend((str(current), repr(current), repr(current.args)))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return "\n".join(values)


@pytest.mark.asyncio
async def test_readiness_uses_exact_get_routes_and_projects_only_documented_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    observation = await client.readiness()
    await client.aclose()

    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/health"),
        ("GET", "/props"),
        ("GET", "/slots"),
    ]
    assert all(
        request.headers["authorization"] == "Bearer test-token" for request in requests
    )
    assert observation.build_info == "build-427291b"
    assert observation.model_path == "/models/test.gguf"
    assert observation.runtime_values == ()
    assert [slot.model_dump() for slot in observation.slots] == [
        {
            "slot_id": 0,
            "busy": False,
            "tokens": None,
            "context_size": 4096,
            "observed_at": observation.slots[0].observed_at,
        },
        {
            "slot_id": 1,
            "busy": None,
            "tokens": None,
            "context_size": None,
            "observed_at": observation.slots[1].observed_at,
        },
    ]
    retained = repr(observation) + observation.model_dump_json() + caplog.text
    assert "do-not-retain" not in retained
    assert "invented-runtime-evidence" not in retained


@pytest.mark.asyncio
async def test_slots_accepts_absent_optional_metrics_as_unknown() -> None:
    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(_response))
    slots = await client.slots()
    await client.aclose()

    assert len(slots) == 2
    assert slots[1].busy is None
    assert slots[1].tokens is None
    assert slots[1].context_size is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "action", "counter_fields"),
    [
        ("save", "save", {"n_saved": 17, "n_written": 8192}),
        ("restore", "restore", {"n_restored": 17, "n_read": 8192}),
    ],
)
async def test_mutation_uses_exact_path_query_body_and_receipt_mapping(
    operation: str,
    action: str,
    counter_fields: dict[str, int],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id_slot": 7,
                "filename": "owned.bin",
                **counter_fields,
                "timings": {"cache_n": 999999, "prompt": "secret-prompt"},
            },
        )

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    receipt = await getattr(client, operation)(7, "owned.bin")
    await client.aclose()

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/slots/7"
    assert request.url.query == b"action=" + action.encode("ascii")
    assert json.loads(request.content) == {"filename": "owned.bin"}
    assert receipt.model_dump() == {
        "slot_id": 7,
        "filename": "owned.bin",
        "tokens": 17,
        "bytes": 8192,
    }
    assert "secret-prompt" not in repr(receipt) + receipt.model_dump_json()
    assert "999999" not in repr(receipt) + receipt.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b'{"prompt":"response-secret-canary"', "invalid_response"),
        (b"[1, 2, 3]", "invalid_response"),
        (
            b'"response-secret-canary' + b"x" * (1024 * 1024) + b'"',
            "response_too_large",
        ),
    ],
    ids=("malformed-object", "wrong-top-level", "over-one-mib"),
)
async def test_malformed_and_oversized_json_are_bounded(
    payload: bytes,
    expected_code: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "response-secret-canary"

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=payload)

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    with pytest.raises(SnapshotError) as raised:
        await client.slots()
    await client.aclose()

    assert raised.value.code == expected_code
    assert raised.value.submission_possible is False
    assert canary not in _exception_graph_text(raised.value) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (401, "authentication_failed"),
        (403, "authentication_failed"),
        (404, "unsupported_route"),
        (405, "unsupported_route"),
        (501, "unsupported_route"),
        (307, "unexpected_redirect"),
        (500, "request_failed"),
    ],
)
async def test_http_errors_become_fixed_codes_without_body_disclosure(
    status: int,
    expected_code: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "provider-secret-and-prompt-canary"

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status, json={"error": canary, "prompt": canary})

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    with pytest.raises(SnapshotError) as raised:
        await client.slots()
    await client.aclose()

    assert raised.value.code == expected_code
    assert raised.value.submission_possible is False
    assert canary not in _exception_graph_text(raised.value) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_overrides",
    [
        {"id_slot": 8},
        {"filename": "different.bin"},
        {"filename": "../owned.bin"},
        {"n_saved": True},
        {"n_written": -1},
    ],
)
async def test_save_rejects_mismatched_or_invalid_receipts(
    response_overrides: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        body: dict[str, object] = {
            "id_slot": 7,
            "filename": "owned.bin",
            "n_saved": 17,
            "n_written": 8192,
        }
        body.update(response_overrides)
        return httpx.Response(200, json=body)

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    with pytest.raises(SnapshotError) as raised:
        await client.save(7, "owned.bin")
    await client.aclose()

    assert raised.value.code == "outcome_unknown"
    assert raised.value.submission_possible is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b'{"prompt":"mutation-secret-canary"',
        b'"mutation-secret-canary' + b"x" * (1024 * 1024) + b'"',
    ],
    ids=("malformed", "over-one-mib"),
)
async def test_mutation_without_valid_terminal_response_is_unknown(
    payload: bytes,
) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, content=payload)

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    with pytest.raises(SnapshotError) as raised:
        await client.save(0, "owned.bin")
    await client.aclose()

    assert requests == 1
    assert raised.value.code == "outcome_unknown"
    assert raised.value.submission_possible is True
    assert "mutation-secret-canary" not in _exception_graph_text(raised.value)


@pytest.mark.asyncio
async def test_probe_protocol_failure_has_a_fixed_safe_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("protocol-secret-canary", request=request)

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    with pytest.raises(SnapshotError) as raised:
        await client.slots()
    await client.aclose()

    assert raised.value.code == "protocol_error"
    assert raised.value.submission_possible is False
    assert "protocol-secret-canary" not in _exception_graph_text(raised.value)


def test_mutation_timeouts_do_not_inherit_probe_timeout() -> None:
    from tldw_chatbook.LLM_Management.snapshot_client import MUTATION_TIMEOUT

    assert isinstance(MUTATION_TIMEOUT, httpx.Timeout)
    assert MUTATION_TIMEOUT.connect == 5
    assert MUTATION_TIMEOUT.pool == 5
    assert MUTATION_TIMEOUT.write == 30
    assert MUTATION_TIMEOUT.read == 600


def test_slot_receipt_rejects_nul_in_server_filename() -> None:
    with pytest.raises(ValidationError):
        SlotReceipt(slot_id=0, filename="owned\0.bin", tokens=1, bytes=2)


@pytest.mark.asyncio
async def test_probe_deadline_is_independent_from_slow_valid_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        await asyncio.sleep(0.03)
        return httpx.Response(
            200,
            json={
                "id_slot": 0,
                "filename": "owned.bin",
                "n_saved": 1,
                "n_written": 2,
            },
        )

    import tldw_chatbook.LLM_Management.snapshot_client as module

    monkeypatch.setattr(module, "PROBE_SECONDS", 0.01)
    monkeypatch.setattr(module, "MUTATION_SECONDS", 0.2)
    client = module.SnapshotClient(
        _descriptor(), transport=httpx.MockTransport(handler)
    )
    receipt = await client.save(0, "owned.bin")
    await client.aclose()

    assert receipt.bytes == 2
    assert requests == 1


@pytest.mark.asyncio
async def test_probe_timeout_is_fixed_and_pre_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        entered.set()
        await release.wait()
        return httpx.Response(200, json=[])

    import tldw_chatbook.LLM_Management.snapshot_client as module

    monkeypatch.setattr(module, "PROBE_SECONDS", 0.01)
    client = module.SnapshotClient(
        _descriptor(), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(SnapshotError) as raised:
        await client.slots()
    release.set()
    await client.aclose()

    assert entered.is_set()
    assert raised.value.code == "probe_timeout"
    assert raised.value.submission_possible is False


@pytest.mark.asyncio
async def test_readiness_has_one_aggregate_probe_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        await asyncio.sleep(0.008)
        return _response(request)

    import tldw_chatbook.LLM_Management.snapshot_client as module

    monkeypatch.setattr(module, "PROBE_SECONDS", 0.02)
    client = module.SnapshotClient(
        _descriptor(), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(SnapshotError) as raised:
        await client.readiness()
    await client.aclose()

    assert requests == ["/health", "/props", "/slots"]
    assert raised.value.code == "probe_timeout"
    assert raised.value.submission_possible is False


@pytest.mark.asyncio
async def test_mutation_timeout_after_dispatch_is_unknown_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0
    dispatched = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        dispatched.set()
        await release.wait()
        return httpx.Response(200, json={})

    import tldw_chatbook.LLM_Management.snapshot_client as module

    monkeypatch.setattr(module, "MUTATION_SECONDS", 0.01)
    client = module.SnapshotClient(
        _descriptor(), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(SnapshotError) as raised:
        await client.save(0, "owned.bin")
    release.set()
    await client.aclose()

    assert dispatched.is_set()
    assert requests == 1
    assert raised.value.code == "outcome_unknown"
    assert raised.value.submission_possible is True
    assert not any(
        isinstance(value, BaseException) for value in client.__dict__.values()
    )


@pytest.mark.asyncio
async def test_connect_failure_is_pre_submission_and_never_retried() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise httpx.ConnectError("raw transport canary", request=request)

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    with pytest.raises(SnapshotError) as raised:
        await client.restore(0, "owned.bin")
    await client.aclose()

    assert requests == 1
    assert raised.value.code == "connection_failed"
    assert raised.value.submission_possible is False
    assert "raw transport canary" not in _exception_graph_text(raised.value)
    assert not any(
        isinstance(value, BaseException) for value in client.__dict__.values()
    )


@pytest.mark.asyncio
async def test_invalid_request_basename_is_rejected_before_post() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(500)

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    with pytest.raises(SnapshotError) as raised:
        await client.save(0, "../foreign.bin")
    await client.aclose()

    assert requests == 0
    assert raised.value.code == "invalid_filename"
    assert raised.value.submission_possible is False


class _CloseTrackingStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.closed = False
        self.read = False

    async def __aiter__(self):
        self.read = True
        yield self.body

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["slots", "save"])
async def test_requests_negotiate_identity_encoding(operation: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = (
            []
            if operation == "slots"
            else {"id_slot": 0, "filename": "owned.bin", "n_saved": 1, "n_written": 2}
        )
        return httpx.Response(
            200, headers={"Content-Encoding": "identity"}, json=payload
        )

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    try:
        if operation == "slots":
            assert await client.slots() == ()
        else:
            assert (await client.save(0, "owned.bin")).bytes == 2
    finally:
        await client.aclose()

    assert len(requests) == 1
    assert requests[0].headers["accept-encoding"] == "identity"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["slots", "save"])
@pytest.mark.parametrize(
    ("encoding", "body"),
    [
        ("gzip", b"corrupt-gzip-secret-canary"),
        ("gzip", gzip.compress(b"x" * (2 * 1024 * 1024))),
        ("unknown-encoding-secret-canary", b"[]"),
    ],
    ids=["corrupt-gzip", "inflating-gzip", "unknown-encoding"],
)
async def test_unexpected_encoding_is_rejected_before_reading_or_decoding(
    operation: str,
    encoding: str,
    body: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _CloseTrackingStream(body)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, headers={"Content-Encoding": encoding}, stream=stream
        )

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(SnapshotError) as raised:
            if operation == "slots":
                await client.slots()
            else:
                await client.save(0, "owned.bin")
    finally:
        await client.aclose()

    assert len(requests) == 1
    assert stream.read is False
    assert stream.closed is True
    assert raised.value.code == (
        "invalid_response" if operation == "slots" else "outcome_unknown"
    )
    assert raised.value.submission_possible is (operation == "save")
    assert "secret-canary" not in _exception_graph_text(raised.value) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["slots", "save"])
async def test_deeply_nested_json_has_safe_error_classification(
    operation: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _CloseTrackingStream(
        b"[" * 50000 + b'"nested-json-secret-canary"' + b"]" * 50000
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, stream=stream)

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(SnapshotError) as raised:
            if operation == "slots":
                await client.slots()
            else:
                await client.save(0, "owned.bin")
    finally:
        await client.aclose()

    assert len(requests) == 1
    assert stream.closed is True
    assert raised.value.code == (
        "invalid_response" if operation == "slots" else "outcome_unknown"
    )
    assert raised.value.submission_possible is (operation == "save")
    assert "secret-canary" not in _exception_graph_text(raised.value) + caplog.text


class _CloseTrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self, stream: _CloseTrackingStream) -> None:
        self.stream = stream
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=self.stream, request=request)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_response_and_owned_client_are_closed() -> None:
    stream = _CloseTrackingStream(b"[]")
    transport = _CloseTrackingTransport(stream)

    from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

    client = SnapshotClient(_descriptor(), transport=transport)
    assert await client.slots() == ()
    assert stream.closed is True
    assert transport.closed is False
    await client.aclose()
    assert transport.closed is True


class _OwnedListener:
    def __init__(
        self,
        responder: Callable[[bytes], tuple[int, list[tuple[str, str]], bytes]],
    ) -> None:
        self.responder = responder
        self.requests: list[bytes] = []
        self.server: asyncio.AbstractServer | None = None

    async def __aenter__(self) -> Self:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *args: object) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    @property
    def base_url(self) -> str:
        assert self.server is not None
        port = self.server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=1)
            self.requests.append(request)
            status, headers, body = self.responder(request)
            reason = {200: "OK", 302: "Found"}[status]
            head = [
                f"HTTP/1.1 {status} {reason}\r\n",
                f"Content-Length: {len(body)}\r\n",
                "Connection: close\r\n",
                *(f"{name}: {value}\r\n" for name, value in headers),
                "\r\n",
            ]
            writer.write("".join(head).encode("ascii") + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.mark.asyncio
@pytest.mark.loopback_network
async def test_proxy_environment_cannot_reroute_numeric_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owned numeric-loopback listeners prove proxy variables cannot reroute traffic."""
    real = _OwnedListener(lambda request: (200, [], b"[]"))
    decoy = _OwnedListener(lambda request: (200, [], b"[]"))
    async with real, decoy:
        monkeypatch.setenv("HTTP_PROXY", decoy.base_url)
        monkeypatch.setenv("ALL_PROXY", decoy.base_url)
        monkeypatch.setenv("NO_PROXY", "")

        from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

        client = SnapshotClient(_descriptor(real.base_url))
        assert await client.slots() == ()
        await client.aclose()

    assert len(real.requests) == 1
    assert len(decoy.requests) == 0
    assert b"GET /slots HTTP/1.1" in real.requests[0]


@pytest.mark.asyncio
@pytest.mark.loopback_network
async def test_redirect_never_reaches_decoy_or_forwards_credentials() -> None:
    """An owned loopback redirect target must receive no request or credentials."""
    decoy = _OwnedListener(lambda request: (200, [], b"[]"))
    async with decoy:
        real = _OwnedListener(
            lambda request: (
                302,
                [("Location", f"{decoy.base_url}/capture")],
                b'{"prompt":"redirect-secret-canary"}',
            )
        )
        async with real:
            from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient

            client = SnapshotClient(
                _descriptor(real.base_url, token="credential-canary")
            )
            with pytest.raises(SnapshotError) as raised:
                await client.slots()
            await client.aclose()

    assert raised.value.code == "unexpected_redirect"
    assert len(real.requests) == 1
    assert len(decoy.requests) == 0
    assert b"Authorization: Bearer credential-canary" in real.requests[0]
    assert "redirect-secret-canary" not in _exception_graph_text(raised.value)
