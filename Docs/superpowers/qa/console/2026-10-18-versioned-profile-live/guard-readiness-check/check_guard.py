import asyncio
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import requests
import urllib3
from transport_guard import install

WORKTREE = "/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook"
URL = "http://127.0.0.1:8010/v1/chat/completions"
BODY = b'{"choices":[{"message":{"content":"ok","reasoning_content":"private"},"finish_reason":"stop"}]}'
SSE = b"data: " + BODY + b"\n\ndata: [DONE]\n\n"
forwarded = []
socket_attempts = []


def refuse_socket(*args, **kwargs):
    socket_attempts.append(True)
    raise AssertionError("offline guard check attempted network")


socket.socket.connect = refuse_socket
socket.socket.connect_ex = refuse_socket
socket.create_connection = refuse_socket
socket.getaddrinfo = refuse_socket


class AsyncChunks(httpx.AsyncByteStream):
    async def __aiter__(self):
        for chunk in (SSE[:17], SSE[17:]):
            yield chunk


class AsyncJSON(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield BODY


class SyncChunks(httpx.SyncByteStream):
    def __iter__(self):
        yield SSE


async def fake_async(self, request):
    forwarded.append(("async", request.content))
    if request.method == "GET":
        return httpx.Response(404, content=b"{}")
    payload = json.loads(request.content)
    if payload.get("test_fail"):
        raise httpx.ReadError("offline simulated stream failure")
    if payload.get("stream"):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=AsyncChunks()
        )
    return httpx.Response(
        200, headers={"content-type": "application/json"}, stream=AsyncJSON()
    )


def fake_sync(self, request):
    forwarded.append(("sync", request.content))
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=SyncChunks()
    )


def fake_requests(self, request, **kw):
    forwarded.append(("requests", request.body))
    streaming = json.loads(request.body).get("stream")
    response = requests.Response()
    response.status_code = 200
    response.headers["content-type"] = (
        "text/event-stream" if streaming else "application/json"
    )
    response.raw = urllib3.response.HTTPResponse(
        body=io.BytesIO(SSE if streaming else BODY), preload_content=False
    )
    return response


httpx.AsyncHTTPTransport.handle_async_request = fake_async
httpx.HTTPTransport.handle_request = fake_sync
requests.adapters.HTTPAdapter.send = fake_requests
root = (
    Path(sys.argv[2])
    if len(sys.argv) > 1
    else Path(tempfile.mkdtemp(prefix="archive-recovery-guard-offline-"))
)
for name, suffix in {
    "HOME": "home",
    "XDG_CONFIG_HOME": "xdg-config",
    "XDG_DATA_HOME": "xdg-data",
    "XDG_CACHE_HOME": "xdg-cache",
}.items():
    (root / suffix).mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(root / suffix)
(root / "config.toml").write_text(
    "[general]\nuser_data_dir = " + json.dumps(str(root / "data")) + "\n"
)
install(root, consumed=17, cap=24)


async def check():
    async with httpx.AsyncClient(trust_env=False) as client:
        assert (await client.get("http://127.0.0.1:8010/props")).status_code == 404
        assert json.loads((root / "counter.json").read_text())["consumed"] == 17
        payload = {
            "messages": [{"role": "user", "content": "synthetic"}],
            "stream": True,
        }
        expected = client.build_request("POST", URL, json=payload).content
        async with client.stream("POST", URL, json=payload) as response:
            assert b"".join([chunk async for chunk in response.aiter_raw()]) == SSE
        assert forwarded[-1] == ("async", expected)
        # This is the production sensitive nonstream/auxiliary helper, which bypasses client.send.
        os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
        os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, WORKTREE)
        from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

        response = await ConsoleProviderGateway._post_without_high_level_http_log(
            client, URL, json_payload={"messages": [], "stream": False}
        )
        assert response.content == BODY
        before = len(forwarded)
        for bad_url in (
            "http://example.invalid/v1/chat/completions",
            "http://127.0.0.1:8010/embeddings",
        ):
            try:
                await client.post(bad_url, json=payload)
            except RuntimeError:
                pass
            else:
                raise AssertionError("route not refused")
        assert len(forwarded) == before
        try:
            await client.post(URL, json={"stream": True, "test_fail": True})
        except httpx.ReadError:
            pass
        else:
            raise AssertionError("simulated stream failed to fail")
        # A fallback is a distinct transport attempt and must consume another slot.
        assert (await client.post(URL, json={"stream": False})).content == BODY
    with httpx.Client(trust_env=False) as client:
        assert client.post(URL, json={"stream": True}).content == SSE
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(URL, json={"stream": True}, stream=True)
        assert b"".join(response.iter_content(chunk_size=7)) == SSE
        response.close()
        response = session.post(URL, json={"stream": False})
        assert response.content == BODY
        response.close()
    assert len(forwarded) == 8
    assert json.loads((root / "counter.json").read_text()) == {
        "consumed": 24,
        "authorized_total": 24,
    }
    before = len(forwarded)
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            await client.post(URL, json={"stream": False})
        except RuntimeError as error:
            assert "cap reached" in str(error)
        else:
            raise AssertionError("cap not enforced")
    assert len(forwarded) == before
    with requests.Session() as session:
        session.trust_env = False
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=1))
        try:
            session.post(URL, json={"stream": False})
        except RuntimeError as error:
            assert "route/retry" in str(error)
        else:
            raise AssertionError("requests retry not refused")
    async with httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(retries=1)
    ) as client:
        try:
            await client.post(URL, json={"stream": False})
        except RuntimeError as error:
            assert "route/retry" in str(error)
        else:
            raise AssertionError("HTTPX retry not refused")
    events = [
        json.loads(line)
        for line in (root / "transport-metadata.jsonl").read_text().splitlines()
    ]
    attempts = [row["attempt"] for row in events if row["event"] == "request"]
    assert attempts == list(range(18, 25))
    observed = [row for row in events if row["event"] == "response_observation"]
    assert len(observed) == 6
    assert all(
        row["reasoning_bytes"] == 7 and row["content_bytes"] == 2 for row in observed
    )
    assert not socket_attempts
    print(
        json.dumps(
            {
                "passed": True,
                "actual_model_calls": 0,
                "socket_attempts": 0,
                "fake_transports": len(forwarded),
                "observed_responses": len(observed),
                "exact_request_and_response_bytes": True,
                "production_sensitive_nonstream_helper": True,
                "fallback_counted_separately": True,
                "cap_denied_before_transport": True,
                "retry_and_route_denial": True,
                "state_path": str(root / "counter.json"),
            },
            indent=2,
        )
    )


if len(sys.argv) > 1:
    assert sys.argv[1] == "--resume"
    with requests.Session() as session:
        session.trust_env = False
        try:
            session.post(URL, json={"stream": False})
        except RuntimeError as error:
            assert "cap reached" in str(error)
        else:
            raise AssertionError("restart reset the counter")
    assert not forwarded and not socket_attempts
    assert json.loads((root / "counter.json").read_text())["consumed"] == 24
    print("Restart retained counter and refused before transport.")
else:
    asyncio.run(check())
    subprocess.run([sys.executable, __file__, "--resume", str(root)], check=True)
