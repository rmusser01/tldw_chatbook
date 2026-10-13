"""Actual installed dispatch to a private loopback HTTP owner; no paid calls."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic

import pytest

from Tests.Chat.response_rules_fixtures import source
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionRequest,
    ConsoleProviderGateway,
)
from tldw_chatbook.Chat.response_rules.resources import RuleHelperPool

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
    pytest.mark.loopback_network,
]


@contextmanager
def endpoint():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            response = {
                "id": "offline",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "fixture answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", [None, "none", "low", "high", "max"])
@pytest.mark.parametrize("configured", [False, True])
async def test_deepseek_explicit_effort_primary_and_leased_helper(effort, configured):
    usage = []
    pool = RuleHelperPool(
        usage_sink=lambda *args: usage.append(args),
        current=lambda _: True,
        clock=monotonic,
    )
    with endpoint() as (base, seen):
        settings = {
            "api_key": "offline-fixture",
            "model": "deepseek-v4-pro",
            "api_base_url": base,
        }
        if configured:
            settings["reasoning_effort"] = effort
        gateway = ConsoleProviderGateway(
            environ={}, config_provider=lambda: {"api_settings": {"deepseek": settings}}
        )
        try:
            resolution = await gateway.resolve_for_send(
                ConsoleProviderSelection(
                    provider="deepseek",
                    explicit_model="deepseek-v4-pro",
                    base_url=base,
                    streaming=False,
                    max_tokens=4096,
                    reasoning_effort=None if configured else effort,
                )
            )
            assert resolution.ready
            messages = [{"role": "user", "content": "offline fixture"}]
            result = [
                chunk async for chunk in gateway.stream_chat(resolution, messages)
            ]
            assert result == ["fixture answer"]
            lease = pool.try_acquire(
                source(), purpose="checking", deadline=monotonic() + 30
            )
            answer = await gateway.complete_auxiliary(
                AuxiliaryCompletionRequest(resolution, tuple(messages), None, 4096),
                rule_lease=lease,
            )
            assert answer.text == "fixture answer"
            assert len(seen) == 2 and all(
                path == "/chat/completions" for path, _ in seen
            )
            for _, body in seen:
                assert body.get("reasoning_effort") == effort
                assert ("reasoning_effort" in body) == (effort is not None)
                assert "thinking" not in body
            assert len(usage) == 1 and usage[0][2].output == 2
        finally:
            pool.close_admission()
            await pool.wait_for_cleanup()
            await gateway.aclose()
        assert pool.unsettled_count == 0 and gateway.http_client.is_closed


@pytest.mark.asyncio
async def test_deepseek_responses_refuses_before_transport():
    with endpoint() as (base, seen):
        gateway = ConsoleProviderGateway(
            environ={},
            config_provider=lambda: {
                "api_settings": {
                    "deepseek": {
                        "api_key": "offline-fixture",
                        "model": "deepseek-v4-pro",
                        "api_base_url": base,
                        "api_mode": "responses",
                    }
                }
            },
        )
        try:
            resolution = await gateway.resolve_for_send(
                ConsoleProviderSelection(
                    provider="deepseek", explicit_model="deepseek-v4-pro"
                )
            )
            assert not resolution.ready
            assert "unsupported" in resolution.visible_copy.lower()
            assert seen == []
        finally:
            await gateway.aclose()


@pytest.mark.parametrize(
    "model,effort",
    [
        ("deepseek-v4-pro", "medium"),
        ("deepseek-v4-pro", {"unsafe": "value"}),
        ("deepseek-chat", "none"),
        ("deepseek-v4-pro", ""),
        ("deepseek-v4-pro", "none\n"),
    ],
)
def test_deepseek_invalid_effort_refuses_without_http(model, effort):
    from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    with endpoint() as (base, seen):
        with pytest.raises(ChatConfigurationError):
            chat_api_call(
                "deepseek",
                messages_payload=[{"role": "user", "content": "x"}],
                model=model,
                api_key="offline-fixture",
                api_base_url=base,
                reasoning_effort=effort,
            )
        assert not seen


def test_direct_deepseek_responses_refuses_without_http():
    from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    with endpoint() as (base, seen):
        with pytest.raises(ChatConfigurationError):
            chat_api_call(
                "deepseek",
                messages_payload=[{"role": "user", "content": "x"}],
                model="deepseek-v4-pro",
                api_key="offline-fixture",
                api_base_url=base,
                api_mode="responses",
            )
        assert not seen


@pytest.mark.asyncio
async def test_deepseek_selection_precedence_and_frozen_helper_setting():
    with endpoint() as (base, seen):
        settings = {
            "api_key": "offline-fixture",
            "model": "deepseek-flash",
            "api_base_url": base,
            "reasoning_effort": "high",
        }
        gateway = ConsoleProviderGateway(
            environ={}, config_provider=lambda: {"api_settings": {"deepseek": settings}}
        )
        pool = RuleHelperPool(
            usage_sink=lambda *args: None, current=lambda _: True, clock=monotonic
        )
        try:
            resolution = await gateway.resolve_for_send(
                ConsoleProviderSelection(
                    provider="deepseek",
                    explicit_model="deepseek-flash",
                    base_url=base,
                    reasoning_effort="none",
                    max_tokens=4096,
                )
            )
            assert resolution.reasoning_effort == "none"
            settings.update(reasoning_effort="max", api_mode="responses")
            lease = pool.try_acquire(
                source(), purpose="checking", deadline=monotonic() + 30
            )
            result = await gateway.complete_auxiliary(
                AuxiliaryCompletionRequest(
                    resolution, ({"role": "user", "content": "frozen"},), None, 4096
                ),
                rule_lease=lease,
            )
            assert result.text == "fixture answer"
            assert seen[0][1]["reasoning_effort"] == "none"
            assert seen[0][0] == "/chat/completions"
        finally:
            pool.close_admission()
            await pool.wait_for_cleanup()
            await gateway.aclose()


def test_deepseek_editor_support_matches_the_exact_adapter_values():
    from tldw_chatbook.Chat.console_provider_support import (
        console_generation_control_support,
        reasoning_effort_values_sent,
        supported_generation_fields,
    )

    for model in ("deepseek-flash", "deepseek-v4-pro"):
        assert "reasoning_effort" in supported_generation_fields("deepseek", model)
        assert (
            console_generation_control_support("deepseek", model, "reasoning_effort")
            == "supported"
        )
    assert "reasoning_effort" not in supported_generation_fields(
        "deepseek", "deepseek-chat"
    )
    assert reasoning_effort_values_sent(
        "deepseek", ("none", "low", "medium", "high", "max", "ultra")
    ) == ("none", "low", "high", "max")


@pytest.mark.asyncio
async def test_deepseek_all_dispatch_builders_pin_resolved_mode():
    settings = {
        "api_key": "offline-fixture",
        "model": "deepseek-flash",
        "api_mode": "chat_completions",
    }
    gateway = ConsoleProviderGateway(
        environ={}, config_provider=lambda: {"api_settings": {"deepseek": settings}}
    )
    try:
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider="deepseek",
                explicit_model="deepseek-flash",
                reasoning_effort="none",
            )
        )
        assert resolution.ready
        settings["api_mode"] = "responses"
        messages = [{"role": "user", "content": "frozen fixture"}]
        request = AuxiliaryCompletionRequest(resolution, tuple(messages), None, 4096)
        for kwargs in (
            gateway._chat_api_kwargs(resolution, messages),
            gateway._auxiliary_chat_api_kwargs(request, resolution),
        ):
            assert kwargs.get("api_mode") == "chat_completions"
            assert kwargs["reasoning_effort"] == "none"
    finally:
        await gateway.aclose()
