"""Native helper limits reach installed provider dispatch and physical HTTP."""

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from Tests.Chat.response_rules_fixtures import source
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionRequest,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.response_rules.resources import RuleHelperPool
from tldw_chatbook.Provider_Contract import ChatAPIError
from tldw_chatbook.provider_registry import AUDITED_ENDPOINT_KEYS

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@contextmanager
def _endpoint(*, stall=False, status=200, anthropic_cache=False):
    seen = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            if stall:
                release.wait(15)
            data = json.dumps(
                {
                    "id": "offline",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "deepseek-chat",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            response_status = status
            if anthropic_cache:
                response_status = 400 if len(seen) == 1 else 200
                data = json.dumps(
                    {"error": {"message": "cache_control unsupported"}}
                    if response_status == 400
                    else {
                        "id": "offline",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "answer"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()
            try:
                self.send_response(response_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def _request(provider, base):
    return AuxiliaryCompletionRequest(
        ConsoleProviderResolution(
            provider=provider,
            execution_key=provider,
            readiness_key=provider,
            model="deepseek-chat" if provider == "deepseek" else "gpt-test",
            base_url=base,
            api_key="offline-fixture",
            ready=True,
            streaming=False,
            max_tokens=64,
        ),
        ({"role": "user", "content": "offline test"},),
        None,
        64,
    )


@pytest.mark.asyncio
@pytest.mark.loopback_network
async def test_native_helper_deadline_settles_actual_stalled_http(monkeypatch):
    """Dropping the deadline must leave the physical worker occupied and fail."""
    observed = []
    original = requests.adapters.HTTPAdapter.send

    def observe(adapter, request, **kwargs):
        observed.append((kwargs.get("timeout"), adapter.max_retries.total))
        return original(adapter, request, **kwargs)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", observe)
    pool = RuleHelperPool(
        usage_sink=lambda *_: None, current=lambda _: True, clock=time.monotonic
    )
    gateway = ConsoleProviderGateway(environ={})
    with _endpoint(stall=True) as (base, seen):
        try:
            lease = pool.try_acquire(
                source(), purpose="checking", deadline=time.monotonic() + 30
            )
            with pytest.raises((TimeoutError, ChatAPIError)):
                request = _request("deepseek", base)
                request = replace(
                    request, resolution=replace(request.resolution, request_timeout=5)
                )
                await gateway.complete_auxiliary(request, rule_lease=lease)
            # The real server still holds its response. A cancelled awaiter is
            # insufficient: the actual HTTP timeout must retire the worker.
            async with asyncio.timeout(1.5):
                while pool.unsettled_count:
                    await asyncio.sleep(0.01)
            assert len(seen) == 1
            assert len(observed) == 1 and 0 < observed[0][0] <= 5
            assert observed[0][1] == 0
        finally:
            pool.close_admission()
            await gateway.aclose()
    await pool.wait_for_cleanup()


@pytest.mark.parametrize("provider", sorted(AUDITED_ENDPOINT_KEYS))
def test_every_installed_sync_route_clamps_at_the_http_boundary(monkeypatch, provider):
    """Real adapters must neither lose the deadline nor install hidden retries."""
    from tldw_chatbook.Chat import Chat_Functions as dispatch
    from tldw_chatbook.Utils.sensitive_llm_logging import (
        native_rule_transport_timeout,
        sensitive_llm_request,
    )

    seen = []

    def stop_at_transport(adapter, request, **kwargs):
        seen.append((kwargs["timeout"], adapter.max_retries.total))
        raise requests.exceptions.ReadTimeout("offline transport stop")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", stop_at_transport)
    # Give cold configuration preparation the actual allowed helper budget;
    # the separate stalled-server test exercises a tighter pinned deadline.
    with sensitive_llm_request(), native_rule_transport_timeout(30):
        with pytest.raises((ChatAPIError, requests.RequestException, ValueError)):
            dispatch.chat_api_call(
                api_endpoint=provider,
                messages_payload=[{"role": "user", "content": "offline test"}],
                api_key="offline-fixture",
                model="test-model",
                streaming=False,
                api_base_url="http://127.0.0.1:18765/v1",
                max_tokens=64,
            )
    assert len(seen) == 1, f"{provider} did not reach the installed transport once"
    assert 0 < seen[0][0] <= 30
    assert seen[0][1] == 0


def test_native_unreviewed_dispatch_route_refuses_before_handler(monkeypatch):
    from tldw_chatbook.Chat import Chat_Functions as dispatch
    from tldw_chatbook.Provider_Contract import ChatConfigurationError
    from tldw_chatbook.Utils.sensitive_llm_logging import native_rule_transport_timeout

    called = []

    def unbounded_handler(**_kwargs):
        called.append(True)
        return "unbounded"

    monkeypatch.setitem(dispatch.API_CALL_HANDLERS, "unreviewed", unbounded_handler)
    with native_rule_transport_timeout(2), pytest.raises(ChatConfigurationError):
        dispatch.chat_api_call(api_endpoint="unreviewed", messages_payload=[])
    assert called == []
    # An unrelated ordinary route retains its existing extension behavior.
    assert (
        dispatch.chat_api_call(api_endpoint="unreviewed", messages_payload=[])
        == "unbounded"
    )


@pytest.mark.asyncio
@pytest.mark.loopback_network
@pytest.mark.parametrize("status", [200, 500])
async def test_native_http_is_one_shot_and_does_not_change_ordinary_timeouts(
    monkeypatch, status
):
    observed = []
    original = requests.adapters.HTTPAdapter.send

    def observe(adapter, request, **kwargs):
        observed.append((kwargs["timeout"], adapter.max_retries.total))
        return original(adapter, request, **kwargs)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", observe)
    pool = RuleHelperPool(
        usage_sink=lambda *_: None, current=lambda _: True, clock=time.monotonic
    )
    gateway = ConsoleProviderGateway(environ={})
    with _endpoint(status=status) as (base, seen):
        try:
            for native in (False, True, False):
                lease = (
                    pool.try_acquire(
                        source(), purpose="checking", deadline=time.monotonic() + 30
                    )
                    if native
                    else None
                )
                request = _request("deepseek", base)
                if native:
                    request = replace(
                        request,
                        resolution=replace(request.resolution, request_timeout=5),
                    )
                if status == 500:
                    with pytest.raises(ChatAPIError):
                        await gateway.complete_auxiliary(request, rule_lease=lease)
                else:
                    assert (
                        await gateway.complete_auxiliary(request, rule_lease=lease)
                    ).text == "answer"
            assert len(seen) == len(observed) == 3
            assert observed[0][0] == observed[2][0] == 120
            assert 0 < observed[1][0] <= 5
            assert all(retries == 0 for _, retries in observed)
        finally:
            pool.close_admission()
            await pool.wait_for_cleanup()
            await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.loopback_network
@pytest.mark.parametrize("native", [False, True])
async def test_anthropic_cache_degradation_is_ordinary_only(native):
    pool = RuleHelperPool(
        usage_sink=lambda *_: None, current=lambda _: True, clock=time.monotonic
    )
    gateway = ConsoleProviderGateway(environ={})
    with _endpoint(anthropic_cache=True) as (base, seen):
        request = _request("anthropic", base)
        request = replace(
            request,
            resolution=replace(request.resolution, model="claude-sonnet-4-20250514"),
            messages=(
                {"role": "system", "content": "Keep answers concise."},
                *request.messages,
            ),
        )
        lease = (
            pool.try_acquire(
                source(), purpose="checking", deadline=time.monotonic() + 30
            )
            if native
            else None
        )
        try:
            if native:
                with pytest.raises(ChatAPIError):
                    await gateway.complete_auxiliary(request, rule_lease=lease)
                assert len(seen) == 1
            else:
                assert (await gateway.complete_auxiliary(request)).text == "answer"
                assert len(seen) == 2
                assert "cache_control" not in json.dumps(seen[1][1])
            assert "cache_control" in json.dumps(seen[0][1])
        finally:
            pool.close_admission()
            await pool.wait_for_cleanup()
            await gateway.aclose()
