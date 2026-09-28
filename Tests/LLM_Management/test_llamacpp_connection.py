from __future__ import annotations

import asyncio
import json

import httpx
import pytest


def connection_module():
    from tldw_chatbook.LLM_Management import llamacpp_connection

    return llamacpp_connection


@pytest.mark.asyncio
async def test_local_ready_requires_health_exact_alias_and_live_claim():
    module = connection_module()
    owner = module.LlamaCppConnectionOwner()
    alive = True
    request = owner.begin(
        "http://127.0.0.1:8080",
        runtime_owner="lab_process",
        live_check=lambda: alive,
    )
    paths = []

    def respond(incoming):
        paths.append(incoming.url.path)
        return httpx.Response(
            200,
            json={"status": "ok"}
            if incoming.url.path == "/health"
            else {"data": [{"id": "chatbook-llamacpp"}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await module.probe_llamacpp_target(request, client=client)
    assert paths == ["/health", "/v1/models"]
    assert owner.accept(result)
    target = owner.snapshot().target
    assert target.model_id == "chatbook-llamacpp"
    assert target.provider_key == "llama_cpp"
    assert owner.is_current(target)
    alive = False
    assert owner.snapshot().target is None
    assert not owner.is_current(target)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "health,models,expected",
    [
        (503, {"data": [{"id": "chatbook-llamacpp"}]}, "loading_model"),
        (401, {}, "credential_required"),
        (302, {}, "health_failed"),
        (200, {"data": [{"id": "wrong-model"}]}, "model_missing"),
        (200, {"data": [{"id": "/private/sentinel/model.gguf"}]}, "unsafe_model_id"),
        (200, {"data": [{"id": "name\u202ehidden"}]}, "unsafe_model_id"),
        (200, {"data": [{"id": " two  spaces"}]}, "unsafe_model_id"),
        (
            200,
            {"data": [{"id": "chatbook-llamacpp"}, {"id": "../secret"}]},
            "unsafe_model_id",
        ),
        (200, {"data": "bad"}, "invalid_models_response"),
    ],
)
async def test_failure_never_publishes_a_target(health, models, expected):
    module = connection_module()
    owner = module.LlamaCppConnectionOwner()
    request = owner.begin(
        "http://localhost:8080", runtime_owner="lab_process", live_check=lambda: True
    )

    def respond(incoming):
        return httpx.Response(
            health if incoming.url.path == "/health" else 200, json=models
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await module.probe_llamacpp_target(request, client=client)
    assert result.code == expected
    assert owner.accept(result)
    assert owner.snapshot().target is None
    assert "sentinel" not in repr(owner.snapshot())
    assert "hidden" not in repr(owner.snapshot())


@pytest.mark.asyncio
async def test_external_model_choice_is_explicit_when_multiple_models_exist():
    module = connection_module()
    owner = module.LlamaCppConnectionOwner()

    def respond(incoming):
        return httpx.Response(
            200, json={"data": [{"id": "org/one"}, {"id": "org/two"}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        first = owner.begin(
            "https://example.test/prefix/v1", runtime_owner="external_server"
        )
        result = await module.probe_llamacpp_target(first, client=client)
        assert owner.accept(result)
        assert owner.snapshot().model_ids == ("org/one", "org/two")
        assert owner.snapshot().target is None
        second = owner.begin(
            "https://example.test/prefix",
            runtime_owner="external_server",
            model_id="org/two",
        )
        result = await module.probe_llamacpp_target(second, client=client)
        assert owner.accept(result)
        assert owner.snapshot().target.model_id == "org/two"
        assert owner.snapshot().target.base_url == "https://example.test/prefix"


@pytest.mark.asyncio
async def test_stale_probe_cannot_replace_new_request():
    module = connection_module()
    owner = module.LlamaCppConnectionOwner()
    old = owner.begin("http://localhost:8080", runtime_owner="external_server")
    started, release = asyncio.Event(), asyncio.Event()

    async def respond(incoming):
        started.set()
        await release.wait()
        return httpx.Response(200, json={"data": [{"id": "safe"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(module.probe_llamacpp_target(old, client=client))
        await started.wait()
        current = owner.begin("http://localhost:8081", runtime_owner="external_server")
        release.set()
        result = await task
    assert not owner.accept(result)
    assert owner.snapshot().request == current
    assert owner.snapshot().target is None


@pytest.mark.asyncio
async def test_oversized_body_and_credential_model_id_are_not_published():
    module = connection_module()
    request = module.LlamaCppConnectionOwner().begin(
        "http://localhost:8080", runtime_owner="external_server"
    )
    for body, code in [
        (b"x" * (64 * 1024 + 1), "invalid_models_response"),
        (b"[" * 20000 + b"]" * 20000, "invalid_models_response"),
        (json.dumps({"data": [{"id": "secret-sentinel"}]}).encode(), "unsafe_model_id"),
    ]:

        def respond(incoming, body=body):
            return httpx.Response(
                200, content=b"{}" if incoming.url.path == "/health" else body
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await module.probe_llamacpp_target(
                request, client=client, credential="secret-sentinel"
            )
        assert result.code == code
        assert not result.model_ids
        assert "secret-sentinel" not in repr(result)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://user:secret@localhost:8080",
        "http://localhost:8080/?key=secret",
        "file:///private/model",
        "",
    ],
)
def test_rejects_unsafe_endpoint_before_probe(endpoint):
    module = connection_module()
    with pytest.raises(ValueError):
        module.LlamaCppConnectionOwner().begin(
            endpoint, runtime_owner="external_server"
        )


def test_claim_readiness_exception_fails_closed():
    module = connection_module()

    def failed_poll():
        raise OSError("private sentinel")

    owner = module.LlamaCppConnectionOwner()
    request = owner.begin(
        "http://localhost:8080", runtime_owner="lab_process", live_check=failed_poll
    )
    assert not request.process_alive()


def test_connection_target_rejects_path_model_and_wrong_provider():
    module = connection_module()
    with pytest.raises(ValueError):
        module.LlamaCppConnectionTarget(
            "http://localhost:8080", "./secret.gguf", "external_server", 1
        )


@pytest.mark.asyncio
@pytest.mark.loopback_network
async def test_actual_loopback_health_then_models_round_trip():
    import asyncio
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from tldw_chatbook.LLM_Management.llamacpp_connection import (
        LlamaCppConnectionOwner,
        probe_llamacpp_target,
    )

    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            payload = (
                b'{"status":"ok"}'
                if self.path == "/health"
                else b'{"data":[{"id":"chatbook-llamacpp"}]}'
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        owner = LlamaCppConnectionOwner()
        request = owner.begin(
            f"http://127.0.0.1:{server.server_port}",
            runtime_owner="lab_process",
            live_check=lambda: True,
        )
        result = await probe_llamacpp_target(request)
        assert owner.accept(result)
        assert owner.snapshot().target.model_id == "chatbook-llamacpp"
        assert paths == ["/health", "/v1/models"]
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_compressed_models_response_is_rejected_before_decoding():
    import gzip

    module = connection_module()
    request = module.LlamaCppConnectionOwner().begin(
        "http://localhost:8080", runtime_owner="external_server"
    )

    def respond(incoming):
        if incoming.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            content=gzip.compress(b'{"data":[{"id":"model"}]}'),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await module.probe_llamacpp_target(request, client=client)
    assert result.code == "invalid_models_response"
