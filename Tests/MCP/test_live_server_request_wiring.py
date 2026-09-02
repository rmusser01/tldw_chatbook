"""TASK-27019: wire server-initiated sampling/elicitation to live surfaces."""

from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.MCP.live_server_request_wiring import (
    sampling_policy_for_server,
    build_live_complete_fn,
    build_live_elicit_fn,
    build_server_request_dispatcher_factory,
)


def _run(coro):
    return asyncio.run(coro)


# --- AC#3: policy from config, default deny ---

def test_policy_default_deny(monkeypatch):
    from tldw_chatbook.MCP import live_server_request_wiring as w
    monkeypatch.setattr(w, "get_cli_setting", lambda s, k, d=None: d)
    policy = sampling_policy_for_server("some-server")
    assert policy.allowed is False


def test_policy_allowlist_and_caps(monkeypatch):
    from tldw_chatbook.MCP import live_server_request_wiring as w
    settings = {
        ("mcp", "sampling_allowed_servers"): ["docs", "linear"],
        ("mcp", "sampling_max_requests_per_minute"): 3,
        ("mcp", "sampling_max_total_tokens"): 9000,
    }
    monkeypatch.setattr(w, "get_cli_setting", lambda s, k, d=None: settings.get((s, k), d))
    allowed = sampling_policy_for_server("docs")
    assert allowed.allowed is True
    assert allowed.max_requests_per_minute == 3
    assert allowed.max_total_tokens == 9000
    assert sampling_policy_for_server("other").allowed is False


# --- AC#1: sampling through the live chat provider ---

def test_complete_fn_calls_chat_api_call_and_extracts(monkeypatch):
    from tldw_chatbook.MCP import live_server_request_wiring as w
    captured = {}
    def fake_chat_api_call(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "hello back"}}]}
    monkeypatch.setattr(w, "chat_api_call", fake_chat_api_call)
    settings = {
        ("mcp", "sampling_provider"): "anthropic",
        ("mcp", "sampling_model"): "claude-sonnet-5",
    }
    monkeypatch.setattr(w, "get_cli_setting", lambda s, k, d=None: settings.get((s, k), d))

    complete = build_live_complete_fn()
    text = _run(complete(
        [{"role": "user", "content": {"type": "text", "text": "hi"}}], 128, None
    ))
    assert text == "hello back"
    assert captured["api_endpoint"] == "anthropic"
    assert captured["model"] == "claude-sonnet-5"
    assert captured["streaming"] is False
    # MCP message shape converted to plain chat shape
    assert captured["messages_payload"] == [{"role": "user", "content": "hi"}]


def test_complete_fn_model_hint_wins(monkeypatch):
    from tldw_chatbook.MCP import live_server_request_wiring as w
    captured = {}
    monkeypatch.setattr(w, "chat_api_call", lambda **k: (captured.update(k), {"text": "x"})[1])
    monkeypatch.setattr(w, "get_cli_setting", lambda s, k, d=None: d)
    complete = build_live_complete_fn()
    _run(complete([{"role": "user", "content": "plain"}], 10, "hinted-model"))
    assert captured["model"] == "hinted-model"


# --- AC#2: elicitation through the approval store (confirmation slice) ---

def _store(tmp_path):
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    return LocalMCPStore(tmp_path / "mcp.json")


def test_elicit_approved_returns_accept(tmp_path, monkeypatch):
    store = _store(tmp_path)
    elicit = build_live_elicit_fn(store, poll_seconds=0.02, timeout_seconds=2.0)

    async def drive():
        task = asyncio.create_task(elicit("Proceed with the thing?", {}))
        # wait for the pending request to appear, then approve it
        for _ in range(100):
            pending = [r for r in store.list_approval_requests() if r.status == "pending"]
            if pending:
                store.resolve_approval_request(pending[0].request_id, "approved")
                break
            await asyncio.sleep(0.02)
        return await task

    result = _run(drive())
    assert result == {"action": "accept", "content": {}}


def test_elicit_denied_returns_none(tmp_path):
    store = _store(tmp_path)
    elicit = build_live_elicit_fn(store, poll_seconds=0.02, timeout_seconds=2.0)

    async def drive():
        task = asyncio.create_task(elicit("ok?", {}))
        for _ in range(100):
            pending = [r for r in store.list_approval_requests() if r.status == "pending"]
            if pending:
                store.resolve_approval_request(pending[0].request_id, "denied")
                break
            await asyncio.sleep(0.02)
        return await task

    assert _run(drive()) is None


def test_elicit_timeout_raises_and_cancels(tmp_path):
    store = _store(tmp_path)
    elicit = build_live_elicit_fn(store, poll_seconds=0.02, timeout_seconds=0.1)
    with pytest.raises(TimeoutError):
        _run(elicit("ok?", {}))
    # the abandoned request must not stay pending forever
    stale = [r for r in store.list_approval_requests() if r.status == "pending"]
    assert stale == []


def test_elicit_complex_schema_refused_before_prompting(tmp_path):
    store = _store(tmp_path)
    elicit = build_live_elicit_fn(store, poll_seconds=0.02, timeout_seconds=1.0)
    with pytest.raises(ValueError):
        _run(elicit("fill this", {"properties": {"name": {"type": "string"}}}))
    assert store.list_approval_requests() == [], "unsupported schema must not create a request"


# --- AC#4: factory set at the creation site; per-server budgets isolated ---

def test_factory_builds_per_server_dispatchers(monkeypatch, tmp_path):
    from tldw_chatbook.MCP import live_server_request_wiring as w
    settings = {("mcp", "sampling_allowed_servers"): ["a"]}
    monkeypatch.setattr(w, "get_cli_setting", lambda s, k, d=None: settings.get((s, k), d))
    factory = build_server_request_dispatcher_factory(_store(tmp_path))
    da, db = factory("a"), factory("b")
    assert da.sampling_policy.allowed is True
    assert db.sampling_policy.allowed is False
    assert da.sampling_budget is not db.sampling_budget, "budgets are per-server"
    assert factory("a").sampling_budget is da.sampling_budget, "budget survives reconnect"


def test_get_client_sets_the_factory(tmp_path):
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    svc = LocalMCPControlService.__new__(LocalMCPControlService)
    svc.client = None
    svc.store = _store(tmp_path)
    client = svc._get_client()
    assert client._server_request_dispatcher_factory is not None
    d = client._server_request_dispatcher_factory("some-server")
    assert d is not None and d.sampling_policy.allowed is False
