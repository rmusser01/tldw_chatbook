"""Real local transports and SQLite enforce every fallback generation."""

import asyncio

import pytest

from Tests.Chat.test_automatic_provider_budget import collect, context_for
from Tests.Chat.test_console_refused_stream_retry import (
    _resolve,
    _server,
    _REPLY,
    _USAGE,
)
from Tests.DB.test_automatic_work_budget import _automatic_work_db  # noqa: F401
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderStreamSignals

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.loopback_network]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["custom", "vllm"])
@pytest.mark.parametrize("allowance", [1, 2])
async def test_retry_has_separate_automatic_call_admission(db, provider, allowance):
    context = context_for(db, model_calls=allowance, output_tokens=13)
    first = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    signals = ConsoleProviderStreamSignals()
    with _server(empty_stream=True, empty_stream_usage=first) as server:
        gateway, selected = await asyncio.to_thread(_resolve, provider, server.base_url)
        try:
            with context.scope():
                if allowance == 1:
                    with pytest.raises(AutomaticWorkRefused, match="model_call_budget"):
                        await collect(gateway, selected, signals=signals)
                else:
                    assert await collect(gateway, selected, signals=signals) == [_REPLY]
            assert server.stream_flags == ([True] if allowance == 1 else [True, False])
            snapshot = db.automatic_work.snapshot(context.chain_id)
            assert snapshot.used["model_call"] == allowance
            assert snapshot.used["tokens"] == (3 if allowance == 1 else 29)
            assert snapshot.reserved["tokens"] == 0
            assert signals.usage_payloads() == (
                [first] if allowance == 1 else [first, _USAGE]
            )
        finally:
            await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["custom", "vllm"])
async def test_unknown_first_usage_keeps_reservation_and_blocks_retry(db, provider):
    context = context_for(db, model_calls=2, output_tokens=13)
    with _server(empty_stream=True) as server:
        gateway, selected = await asyncio.to_thread(_resolve, provider, server.base_url)
        try:
            with context.scope(), pytest.raises(
                AutomaticWorkRefused, match="usage_unknown"
            ):
                await collect(gateway, selected)
            assert server.stream_flags == [True]
            snapshot = db.automatic_work.snapshot(context.chain_id)
            assert snapshot.used["model_call"] == 1
            assert snapshot.reserved["tokens"] > 13
            assert snapshot.status == "review_required"
        finally:
            await gateway.aclose()


@pytest.mark.parametrize("bad", [True, -1, 1.5, "7", float("inf")])
def test_malformed_physical_usage_cannot_be_normalized_into_valid_budget(bad):
    from tldw_chatbook.Chat.console_agent_bridge import (
        _openai_usage_from_provider_calls,
    )

    first = {"prompt_tokens": bad, "completion_tokens": 0}
    assert (
        _openai_usage_from_provider_calls(
            (first, _USAGE), provider="custom", model="local"
        )
        is None
    )
    first = {
        "prompt_tokens": 21,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": bad},
    }
    assert (
        _openai_usage_from_provider_calls(
            (first, _USAGE), provider="custom", model="local"
        )
        is None
    )


def test_zero_unknown_and_cache_write_legs_keep_disjoint_usage():
    from tldw_chatbook.Chat.console_agent_bridge import (
        _openai_usage_from_provider_calls,
    )

    zero = {"prompt_tokens": 0, "completion_tokens": 0}
    assert (
        _openai_usage_from_provider_calls(
            (zero, _USAGE), provider="custom", model="local"
        )["total_tokens"]
        == 26
    )
    assert (
        _openai_usage_from_provider_calls(
            (None, _USAGE), provider="custom", model="local"
        )["total_tokens"]
        == 26
    )
    assert (
        _openai_usage_from_provider_calls(
            (_USAGE, None), provider="custom", model="local"
        )["total_tokens"]
        == 26
    )
    assert (
        _openai_usage_from_provider_calls((zero,), provider="custom", model="local")
        is None
    )
    first = {"input_tokens": 5, "output_tokens": 0, "cache_creation_input_tokens": 7}
    result = _openai_usage_from_provider_calls(
        (first, {"input_tokens": 3, "output_tokens": 2}),
        provider="anthropic",
        model="local",
    )
    assert result == {
        "prompt_tokens": 15,
        "completion_tokens": 2,
        "total_tokens": 17,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 7},
    }


@pytest.mark.parametrize(
    "legs,expected",
    [
        ((None, {"completion_tokens": 5}), 5),
        (({"completion_tokens": 2}, {"completion_tokens": 5}), 7),
        (({"completion_tokens": 0}, {"prompt_tokens": 21}), None),
        (({"completion_tokens": 0}, None), None),
        (({"completion_tokens": 0}, {"completion_tokens": 0}), 0),
    ],
)
def test_live_logical_count_does_not_invent_missing_output_zero(legs, expected):
    from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter

    assert _StreamingModelAdapter._provider_calls_output_count(legs) == expected


def test_retry_usage_keeps_physical_identity_and_ignores_unrelated_call():
    aggregate = ConsoleProviderStreamSignals()
    first = aggregate.new_usage_call()
    first.record_usage_payload({"prompt_tokens": 2, "completion_tokens": 0})
    unrelated = aggregate.new_usage_call()
    unrelated.record_usage_payload({"prompt_tokens": 100, "completion_tokens": 50})
    retry = first.begin_retry_usage_call()
    first.record_usage_payload({"prompt_tokens": 999})  # late first-call event
    retry.record_usage_payload(_USAGE)
    assert first.usage_snapshot() == {"prompt_tokens": 2, "completion_tokens": 0}
    assert retry.usage_snapshot() == _USAGE
    assert first.logical_usage_snapshots() == (first.usage_snapshot(), _USAGE)
    retry.close_usage_call()
    first.close_usage_call()
    retry.close_usage_call()
    assert len(aggregate.usage_payloads()) == 3
    assert aggregate.usage_payloads().count(_USAGE) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["custom", "vllm"])
async def test_whole_json_preserves_native_tools_without_retry(provider):
    from tldw_chatbook.Chat.console_provider_gateway import ProviderToolCalls

    signals = ConsoleProviderStreamSignals()
    with _server(ignore_stream=True, json_tools=True) as server:
        gateway, selected = await asyncio.to_thread(_resolve, provider, server.base_url)
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "parameters": {"type": "object"},
                    },
                }
            ]
            chunks = [
                chunk
                async for chunk in gateway.stream_chat(
                    selected,
                    [{"role": "user", "content": "calculate"}],
                    tools=tools,
                    signals=signals,
                )
            ]
            calls = [chunk for chunk in chunks if isinstance(chunk, ProviderToolCalls)]
            assert len(calls) == 1
            assert calls[0].tool_calls[0]["function"] == {
                "name": "calculator",
                "arguments": '{"expression":"1+1"}',
            }
            assert server.stream_flags == [True]
            assert signals.usage_payloads() == [_USAGE]
        finally:
            await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["custom", "vllm"])
async def test_capture_on_retry_retains_both_calls_usage_and_project_context(
    tmp_path, monkeypatch, provider
):
    import json
    from Tests.Chat.test_console_project_instruction_traces import (
        _project_console,
        GUIDANCE,
    )
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader

    first_usage = {"prompt_tokens": 21, "completion_tokens": 0, "total_tokens": 21}
    with _server(empty_stream=True, empty_stream_usage=first_usage) as server:
        async with _project_console(tmp_path, monkeypatch) as app:
            helper, selected = await asyncio.to_thread(
                _resolve, provider, server.base_url
            )
            await helper.aclose()

            async def resolve(_selection):
                return selected

            monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
            result = await app.controller.submit_draft(
                "Hello", session_id=app.session.id
            )
            assert result.accepted
            assistant = app.store.get_message(result.assistant_message_id)
            assert assistant.content == _REPLY, app.reservation_errors
            assert assistant.status == "complete"
            assert server.stream_flags == [True, False]
            assert all(GUIDANCE in json.dumps(payload) for payload in server.payloads)
            assert app.reservation_errors == []
            user = next(
                row
                for row in app.store.messages_for_session(app.session.id)
                if row.content == "Hello"
            )
            captures = ConsoleTraceNativeReader(app.db).read_calls(
                user.persisted_message_id
            )
            assert len(captures) == 2
            assert [json.loads(call.capture.usage_json) for call in captures] == [
                first_usage,
                _USAGE,
            ]
            assert all(
                GUIDANCE in json.dumps(call.capture.request) for call in captures
            )
            with app.db.transaction() as cursor:
                calls = cursor.execute(
                    "SELECT run_id, call_sequence, state FROM console_trace_calls ORDER BY rowid"
                ).fetchall()
            assert len({row[0] for row in calls}) == 1
            assert [row[1] for row in calls] == [0, 1]
            assert [row[2] for row in calls] == ["error", "complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "capture_off", "foreign_owner"])
@pytest.mark.parametrize("agent_mode", [True, False])
async def test_fallback_trace_recovery_keeps_accepted_owner(
    tmp_path, monkeypatch, action, agent_mode
):
    from Tests.Chat.test_console_project_instruction_traces import _project_console

    first_usage = {"prompt_tokens": 21, "completion_tokens": 0, "total_tokens": 21}
    with _server(empty_stream=True, empty_stream_usage=first_usage) as server:
        async with _project_console(tmp_path, monkeypatch) as app:
            helper, selected = await asyncio.to_thread(
                _resolve, "custom", server.base_url
            )
            await helper.aclose()

            async def resolve(_selection):
                return selected

            monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
            app.controller.update_agent_runtime(
                enabled=agent_mode, bridge=app.controller._agent_bridge
            )
            original_bind = app.factory.repository.bind_call
            binds = 0

            def fail_fallback_bind_once(*args, **kwargs):
                nonlocal binds
                binds += 1
                result = original_bind(*args, **kwargs)
                if binds == 2:
                    raise RuntimeError("owned fallback bind failure")
                return result

            monkeypatch.setattr(
                app.factory.repository, "bind_call", fail_fallback_bind_once
            )
            submitted = await app.controller.submit_draft(
                "Hello", session_id=app.session.id
            )
            assert submitted.accepted
            preparation = app.controller.trace_call_recovery_preparation()
            assert preparation is not None, submitted
            preparation_id = preparation.preparation_id
            boundary = app.controller._trace_call_boundaries_by_preparation[
                preparation_id
            ]
            continuation = app.controller._durable_postcommit_continuations[
                preparation_id
            ]
            assert boundary._accepted_preparation is continuation
            assert server.stream_flags == [True]
            if action == "foreign_owner":
                boundary._accepted_preparation = object()
            if action == "capture_off":
                result = await app.controller.send_without_capture(preparation_id)
            else:
                result = await app.controller.retry_library_preparation(preparation_id)
            if action == "foreign_owner":
                assert server.stream_flags == [True]
                assert (
                    app.store.get_message(submitted.assistant_message_id).status
                    != "complete"
                )
                return
            assistant = app.store.get_message(submitted.assistant_message_id)
            assert assistant.status == "complete", (
                result,
                app.reservation_errors,
            )
            assert assistant.content == _REPLY
            assert server.stream_flags == (
                [True, False] if action == "retry" else [True, True, False]
            )
            if action == "retry":
                assert continuation.stream_signals.usage_payloads() == [
                    first_usage,
                    _USAGE,
                ]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_phase", ["before_retry", "after_retry"])
async def test_capture_on_post_tool_stream_retry_keeps_tool_chain(
    tmp_path, monkeypatch, tool_phase
):
    from Tests.Chat.test_console_project_instruction_traces import _project_console

    with _server(
        empty_stream=True,
        empty_stream_usage={"prompt_tokens": 21, "completion_tokens": 0},
        tool_first=tool_phase == "before_retry",
        tool_on_retry=tool_phase == "after_retry",
    ) as server:
        async with _project_console(tmp_path, monkeypatch, tools=1) as app:
            helper, selected = await asyncio.to_thread(
                _resolve, "custom", server.base_url
            )
            await helper.aclose()

            async def resolve(_selection):
                return selected

            monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
            result = await app.controller.submit_draft(
                "Use calculator once", session_id=app.session.id
            )
            assistant = app.store.get_message(result.assistant_message_id)
            assert assistant.status == "complete", (
                result,
                app.reservation_errors,
                server.stream_flags,
            )
            assert assistant.content.endswith(_REPLY)
            assert server.stream_flags == (
                [True, True, False]
                if tool_phase == "before_retry"
                else [True, False, True]
            )
            import json

            assert "42" in json.dumps(server.payloads[-1])
            with app.db.transaction() as cursor:
                calls = cursor.execute(
                    "SELECT run_id, call_sequence, state FROM console_trace_calls ORDER BY rowid"
                ).fetchall()
            assert len({row[0] for row in calls}) == 1
            assert [row[1] for row in calls] == [0, 1, 2]
            assert [row[2] for row in calls] == (
                ["complete", "error", "complete"]
                if tool_phase == "before_retry"
                else ["error", "complete", "complete"]
            )


@pytest.mark.parametrize(
    "previous,current,overlay,allowed",
    [
        (
            {"streaming": True, "temp": 0.2},
            {"streaming": False, "temp": 0.2},
            "structural",
            True,
        ),
        (
            {"streaming": True, "temp": 0.2},
            {"streaming": False, "temp": 0.3},
            "structural",
            False,
        ),
        (
            {"streaming": True},
            {"streaming": False, "max_tokens": 100},
            "structural",
            False,
        ),
        ({"streaming": True}, {"streaming": False}, None, False),
        ({"streaming": True}, {"streaming": False}, "configuration", False),
        ({"streaming": False}, {"streaming": True}, "structural", False),
        ({"streaming": 1}, {"streaming": False}, "structural", False),
        ({"streaming": True}, {"streaming": 0}, "structural", False),
        ({}, {"streaming": False}, "structural", False),
        ({"streaming": True}, {}, "structural", False),
        (
            {"streaming": False, "temp": 0.2},
            {"streaming": False, "temp": 0.2},
            None,
            True,
        ),
    ],
)
def test_failed_call_retry_only_allows_verified_transport_change(
    previous, current, overlay, allowed
):
    from tldw_chatbook.Chat.console_trace_final_values import ProviderOverlayProvenance
    from tldw_chatbook.Chat.console_trace_service import _retry_generation_matches

    overlays = (
        ()
        if overlay is None
        else (ProviderOverlayProvenance("llama_fallback_retry", overlay),)
    )
    assert _retry_generation_matches(previous, current, overlays) is allowed
