"""TASK-26029: server-initiated sampling and elicitation handlers."""

from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.MCP.server_request_handlers import (
    SamplingPolicy,
    SamplingBudget,
    evaluate_sampling_request,
    screen_elicitation_for_secrets,
    ServerRequestDispatcher,
    JsonRpcError,
)


# --- sampling gating + bounding (AC#2/#3) ---

def test_sampling_denied_by_default():
    """AC#2: default is not silent consent."""
    decision = evaluate_sampling_request(SamplingPolicy(), SamplingBudget(), 100, now=0.0)
    assert decision.allow is False
    assert "not allowed" in decision.reason.lower() or "consent" in decision.reason.lower()


def test_sampling_allowed_when_enabled_within_budget():
    policy = SamplingPolicy(allowed=True, max_requests_per_minute=5, max_total_tokens=1000)
    decision = evaluate_sampling_request(policy, SamplingBudget(), 100, now=0.0)
    assert decision.allow is True


def test_sampling_rate_limited(rate_budget=None):
    """AC#3: rate bound."""
    policy = SamplingPolicy(allowed=True, max_requests_per_minute=2, max_total_tokens=100000)
    budget = SamplingBudget(request_times=[0.0, 1.0])  # 2 already in the last minute
    decision = evaluate_sampling_request(policy, budget, 10, now=2.0)
    assert decision.allow is False
    assert "rate" in decision.reason.lower()


def test_sampling_token_budget_exhausted():
    """AC#3: token bound so a server cannot drain the account."""
    policy = SamplingPolicy(allowed=True, max_requests_per_minute=100, max_total_tokens=500)
    budget = SamplingBudget(tokens_used=450)
    decision = evaluate_sampling_request(policy, budget, 100, now=0.0)
    assert decision.allow is False
    assert "token" in decision.reason.lower() or "budget" in decision.reason.lower()


# --- elicitation secret refusal (AC#5) ---

@pytest.mark.parametrize("request_obj", [
    {"message": "Enter your API key", "requestedSchema": {"properties": {"api_key": {"type": "string"}}}},
    {"message": "password please", "requestedSchema": {"properties": {"pw": {"type": "string", "format": "password"}}}},
    {"message": "Provide the secret token", "requestedSchema": {"properties": {"x": {"type": "string"}}}},
])
def test_elicitation_asking_for_secrets_is_refused(request_obj):
    reason = screen_elicitation_for_secrets(request_obj)
    assert reason is not None


def test_ordinary_elicitation_not_refused():
    ok = {"message": "What is your project name?", "requestedSchema": {"properties": {"name": {"type": "string"}}}}
    assert screen_elicitation_for_secrets(ok) is None


# --- dispatcher end-to-end with injected callables (AC#1/#4/#6/#7) ---

def _run(coro):
    return asyncio.run(coro)


def test_dispatch_sampling_fulfilled_via_chat_provider():
    """AC#1."""
    async def complete_fn(messages, max_tokens, model_hint):
        return "hello from provider"

    dispatcher = ServerRequestDispatcher(
        sampling_policy=SamplingPolicy(allowed=True, max_requests_per_minute=10, max_total_tokens=10000),
        complete_fn=complete_fn,
    )
    result = _run(dispatcher.handle("sampling/createMessage", {
        "messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}],
        "maxTokens": 100,
    }))
    assert not isinstance(result, JsonRpcError)
    assert result["content"]["text"] == "hello from provider"
    assert result["role"] == "assistant"


def test_dispatch_sampling_refused_when_not_allowed():
    """AC#2: an ungated server gets a well-formed error, no completion runs."""
    called = {"n": 0}
    async def complete_fn(messages, max_tokens, model_hint):
        called["n"] += 1
        return "x"
    dispatcher = ServerRequestDispatcher(
        sampling_policy=SamplingPolicy(allowed=False), complete_fn=complete_fn
    )
    result = _run(dispatcher.handle("sampling/createMessage", {"messages": [], "maxTokens": 10}))
    assert isinstance(result, JsonRpcError)
    assert called["n"] == 0


def test_dispatch_elicitation_returns_user_response():
    """AC#4."""
    async def elicit_fn(message, schema):
        return {"action": "accept", "content": {"name": "myproj"}}
    dispatcher = ServerRequestDispatcher(elicit_fn=elicit_fn)
    result = _run(dispatcher.handle("elicitation/create", {
        "message": "Project name?",
        "requestedSchema": {"properties": {"name": {"type": "string"}}},
    }))
    assert not isinstance(result, JsonRpcError)
    assert result["action"] == "accept"
    assert result["content"]["name"] == "myproj"


def test_dispatch_elicitation_secret_refused_without_prompting():
    """AC#5: a secret-seeking elicitation is refused, user never prompted."""
    prompted = {"n": 0}
    async def elicit_fn(message, schema):
        prompted["n"] += 1
        return {"action": "accept", "content": {}}
    dispatcher = ServerRequestDispatcher(elicit_fn=elicit_fn)
    result = _run(dispatcher.handle("elicitation/create", {
        "message": "Enter your password",
        "requestedSchema": {"properties": {"password": {"type": "string", "format": "password"}}},
    }))
    assert isinstance(result, JsonRpcError)
    assert prompted["n"] == 0


def test_dispatch_declined_returns_protocol_error_not_hang():
    """AC#6: a declined elicitation returns a well-formed error, never hangs."""
    async def elicit_fn(message, schema):
        return None  # user declined / no surface
    dispatcher = ServerRequestDispatcher(elicit_fn=elicit_fn)
    result = _run(dispatcher.handle("elicitation/create", {"message": "ok?", "requestedSchema": {}}))
    assert isinstance(result, JsonRpcError)


def test_dispatch_unknown_method_is_method_not_found():
    """AC#7: methods this dispatcher doesn't own fall through to -32601."""
    dispatcher = ServerRequestDispatcher()
    result = _run(dispatcher.handle("some/other", {}))
    assert isinstance(result, JsonRpcError)
    assert result.code == -32601


# --- connection-level dispatch wiring (AC#1/#7) ---

def _fake_process():
    from types import SimpleNamespace

    class _EOFStdout:
        async def readline(self):
            return b""
        async def read(self, n=-1):
            return b""

    return SimpleNamespace(stdout=_EOFStdout(), stderr=None, stdin=None, returncode=None)


def test_connection_routes_to_dispatcher_and_replies():
    """AC#1: a wired connection sends the dispatcher's result to the server."""
    from tldw_chatbook.MCP.client import _StdioJSONRPCConnection

    async def _run():
        async def dispatcher(method, params):
            assert method == "sampling/createMessage"
            return {"role": "assistant", "content": {"type": "text", "text": "ok"}}

        conn = _StdioJSONRPCConnection(
            _fake_process(), client_name="t", server_request_dispatcher=dispatcher
        )
        sent = []
        async def capture(payload):
            sent.append(payload)
        conn._send_message = capture

        await conn._handle_server_request(
            {"id": 7, "method": "sampling/createMessage", "params": {"messages": []}}
        )
        # lane-6 I2: the handler now runs as a task off the read loop; drain it.
        for t in list(conn._dispatch_tasks):
            await t
        assert sent and sent[0]["id"] == 7
        assert sent[0]["result"]["content"]["text"] == "ok"

    asyncio.run(_run())


def test_connection_without_dispatcher_is_method_not_found():
    """AC#7: an unwired connection is unchanged (method-not-found)."""
    from tldw_chatbook.MCP.client import _StdioJSONRPCConnection

    async def _run():
        conn = _StdioJSONRPCConnection(_fake_process(), client_name="t")
        sent = []
        async def capture(payload):
            sent.append(payload)
        conn._send_message = capture

        await conn._handle_server_request(
            {"id": 3, "method": "sampling/createMessage", "params": {}}
        )
        assert sent[0]["error"]["code"] == -32601

    asyncio.run(_run())


# --- lane-6 review I1: secret-screen over/under-refusal ---

@pytest.mark.parametrize("request_obj", [
    {"message": "Please paste your API key", "requestedSchema": {}},
    {"message": "Enter your PIN", "requestedSchema": {}},
    {"message": "What is your SSN?", "requestedSchema": {}},
    {"message": "Provide the CVV", "requestedSchema": {}},
    {"message": "Enter your recovery code", "requestedSchema": {}},
    {"message": "Type your seed phrase", "requestedSchema": {}},
])
def test_i1_underrefusal_now_refused(request_obj):
    assert screen_elicitation_for_secrets(request_obj) is not None, request_obj

@pytest.mark.parametrize("request_obj", [
    {"message": "Who is the author?", "requestedSchema": {"properties": {"author": {"type": "string"}}}},
    {"message": "Do you authorize deleting X?", "requestedSchema": {}},
    {"message": "Set max_tokens", "requestedSchema": {"properties": {"max_tokens": {"type": "integer"}}}},
    {"message": "Choose authentication method", "requestedSchema": {}},
])
def test_i1_overrefusal_now_allowed(request_obj):
    assert screen_elicitation_for_secrets(request_obj) is None, request_obj


# --- lane-6 review I3: budget reserved before await ---

def test_i3_failing_completion_still_counts_toward_rate():
    from tldw_chatbook.MCP.server_request_handlers import SamplingBudget
    budget = SamplingBudget()
    async def boom(messages, max_tokens, model_hint):
        raise RuntimeError("provider down")
    d = ServerRequestDispatcher(
        sampling_policy=SamplingPolicy(allowed=True, max_requests_per_minute=100, max_total_tokens=100000),
        sampling_budget=budget,
        complete_fn=boom,
        now_fn=lambda: 100.0,
    )
    result = _run(d.handle("sampling/createMessage", {"messages": [], "maxTokens": 50}))
    assert isinstance(result, JsonRpcError)
    assert len(budget.request_times) == 1, "a failed call must still consume a rate slot"
    assert budget.tokens_used == 0, "a failed call refunds the token budget"


def test_i3_omitted_max_tokens_charges_default_not_zero():
    from tldw_chatbook.MCP.server_request_handlers import (
        SamplingBudget, _DEFAULT_SAMPLING_MAX_TOKENS,
    )
    budget = SamplingBudget()
    async def ok(messages, max_tokens, model_hint):
        assert max_tokens == _DEFAULT_SAMPLING_MAX_TOKENS
        return "hi"
    d = ServerRequestDispatcher(
        sampling_policy=SamplingPolicy(allowed=True, max_requests_per_minute=100, max_total_tokens=100000),
        sampling_budget=budget,
        complete_fn=ok,
    )
    result = _run(d.handle("sampling/createMessage", {"messages": []}))  # no maxTokens
    assert not isinstance(result, JsonRpcError)
    assert budget.tokens_used == _DEFAULT_SAMPLING_MAX_TOKENS, "omitted maxTokens must charge the default"


# --- Qodo review round (PR #2301) #1: boundary validation ---

def test_qodo1_non_dict_message_items_are_invalid_params():
    async def complete_fn(messages, max_tokens, model_hint):
        raise AssertionError("must not be reached with malformed messages")
    d = ServerRequestDispatcher(
        sampling_policy=SamplingPolicy(allowed=True, max_requests_per_minute=10, max_total_tokens=10000),
        complete_fn=complete_fn,
    )
    result = _run(d.handle("sampling/createMessage", {"messages": ["not-a-dict", 42], "maxTokens": 10}))
    assert isinstance(result, JsonRpcError)
    assert result.code == -32602


def test_qodo1_non_dict_requested_schema_is_invalid_params():
    async def elicit_fn(message, schema):
        raise AssertionError("must not be reached with malformed schema")
    d = ServerRequestDispatcher(elicit_fn=elicit_fn)
    result = _run(d.handle("elicitation/create", {"message": "hi", "requestedSchema": ["not", "a", "dict"]}))
    assert isinstance(result, JsonRpcError)
    assert result.code == -32602
