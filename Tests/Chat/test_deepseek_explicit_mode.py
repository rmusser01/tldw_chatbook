"""Actual installed dispatch to a private loopback HTTP owner; no paid calls."""

import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic

import pytest

from Tests.Chat.response_rules_fixtures import source
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleProviderSelection,
    ConsoleRunStatus,
)
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
@pytest.mark.parametrize("base_url", [None, "https://deepseek.example.test/v1"])
@pytest.mark.parametrize("effort", [None, "none"])
@pytest.mark.parametrize("mode", [None, "chat_completions"])
async def test_deepseek_all_dispatch_builders_pin_resolved_mode(base_url, effort, mode):
    from Tests.Chat.test_console_provider_gateway import _capture_on_prepared_request
    from tldw_chatbook.Chat.console_trace_final_values import (
        reconstruct_provider_gateway_kwargs,
    )

    settings = {"api_key": "offline-fixture", "model": "deepseek-flash"}
    if mode is not None:
        settings["api_mode"] = mode
    gateway = ConsoleProviderGateway(
        environ={}, config_provider=lambda: {"api_settings": {"deepseek": settings}}
    )
    try:
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider="deepseek",
                explicit_model="deepseek-flash",
                reasoning_effort=effort,
                base_url=base_url,
                streaming=False,
                max_tokens=4096,
            )
        )
        assert resolution.ready
        settings.update(api_mode="responses", api_base_url="https://changed.test")
        messages = [{"role": "user", "content": "frozen fixture"}]
        prepared = _capture_on_prepared_request(gateway, resolution, messages)
        request = AuxiliaryCompletionRequest(resolution, tuple(messages), None, 4096)
        expected = reconstruct_provider_gateway_kwargs(resolution, prepared)
        for kwargs in (
            gateway._chat_api_kwargs(resolution, messages),
            gateway._chat_api_kwargs_from_prepared(resolution, prepared),
            gateway._auxiliary_chat_api_kwargs(request, resolution),
        ):
            assert kwargs["api_mode"] == "chat_completions"
            assert kwargs["api_base_url"] == resolution.base_url
            assert kwargs.get("reasoning_effort") == effort
            assert ("reasoning_effort" in kwargs) == (effort is not None)
            assert kwargs == expected
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_deepseek_capture_on_stop_then_next_send_keeps_revision_surface(
    tmp_path,
    monkeypatch,
):
    """Real admission/SQLite/trace owners; only the external adapter is controlled."""
    from Tests.Chat.test_console_first_send_atomicity import _controller
    from Tests.Chat.test_console_trace_first_send_atomicity import _force_capture_on
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory

    db, store, controller, _ = _controller(
        tmp_path,
        initial_settings=ConsoleSessionSettings(
            provider="deepseek",
            model="deepseek-flash",
            streaming=True,
            reasoning_effort="none",
            max_tokens=4096,
        ),
    )
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def adapter(**kwargs):
        calls.append(kwargs)

        def chunks():
            if len(calls) == 1:
                entered.set()
                assert release.wait(10), (
                    "test must release the stopped physical adapter"
                )
            yield {"choices": [{"delta": {"content": "done"}}]}

        return chunks()

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=adapter,
        environ={},
        config_provider=lambda: {
            "api_settings": {
                "deepseek": {
                    "api_key": "offline-fixture",
                    "model": "deepseek-flash",
                }
            }
        },
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(db),
    )
    controller.provider_gateway = gateway
    _force_capture_on(monkeypatch)
    first = None
    try:
        first = asyncio.create_task(
            controller.submit_draft("first", session_id="session-1")
        )
        assert await asyncio.to_thread(entered.wait, 10)
        assert controller.stop_active_run()
        release.set()
        await first
        assert controller.run_state.status is ConsoleRunStatus.STOPPED
        first_call = dict(
            db.get_connection().execute("SELECT * FROM console_trace_calls").fetchone()
        )
        second = await controller.submit_draft("second", session_id="session-1")
        rows = [
            dict(row)
            for row in db.get_connection().execute(
                "SELECT * FROM console_trace_calls ORDER BY rowid"
            )
        ]
        nodes = [
            dict(row)
            for row in db.get_connection().execute(
                "SELECT * FROM console_trace_surface_nodes"
            )
        ]
        assert first_call["integrity_state"] == "complete", first_call
        # Before the first response chunk, Stop retains unknown transport outcome.
        assert first_call["state"] == "dispatch_unknown", first_call
        first_node = next(
            node for node in nodes if node["node_id"] == first_call["surface_node_id"]
        )
        assert first_node["component_kind"] == "message"
        assert first_node["reference_kind"] == "revision"
        assert first_node["semantic_revision_id"] is not None
        saved = (
            db.get_connection()
            .execute(
                "SELECT m.content FROM console_trace_semantic_revisions r "
                "JOIN messages m ON m.id = r.source_message_id WHERE r.revision_id = ?",
                (first_node["semantic_revision_id"],),
            )
            .fetchone()
        )
        assert saved["content"] == "first"
        assert second.accepted and second.provider_started, second
        assert len(calls) == 2 and len(rows) == 2, rows
        assert [item["content"] for item in calls[1]["messages_payload"]] == [
            "first",
            "second",
        ]
        assert store.get_message(second.assistant_message_id).content == "done"
        assert all(row["integrity_state"] == "complete" for row in rows), rows
        assert rows[1]["state"] == "complete"
        assert rows[1]["dispatch_started_at"] is not None
        assert rows[1]["reservation_provenance"] == "crash_durable_reserved"
        second_node = next(
            node for node in nodes if node["node_id"] == rows[1]["surface_node_id"]
        )
        assert second_node["predecessor_node_id"] == first_node["node_id"]
        assert not any(node["reference_kind"] == "omission" for node in nodes), nodes
    finally:
        release.set()
        if first is not None:
            await asyncio.gather(first, return_exceptions=True)
        try:
            await controller.shutdown()
        finally:
            try:
                await gateway.aclose()
            finally:
                db.close()
