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


@pytest.mark.bootstrap_profile
def test_get_client_sets_the_factory(tmp_path):
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    svc = LocalMCPControlService.__new__(LocalMCPControlService)
    svc.client = None
    svc.store = _store(tmp_path)
    client = svc._get_client()
    assert client._server_request_dispatcher_factory is not None
    d = client._server_request_dispatcher_factory("some-server")
    assert d is not None and d.sampling_policy.allowed is False


def test_qodo8_boolean_schema_now_refused(tmp_path):
    """Qodo #8 (PR #2313): approve/deny cannot fabricate boolean field values;
    only an empty schema is a representable confirmation."""
    store = _store(tmp_path)
    elicit = build_live_elicit_fn(store, poll_seconds=0.02, timeout_seconds=1.0)
    with pytest.raises(ValueError):
        _run(elicit("confirm?", {"properties": {"confirm": {"type": "boolean"}}, "required": ["confirm"]}))
    assert store.list_approval_requests() == []


def test_qodo9_expired_request_cannot_be_approved(tmp_path):
    """Qodo #9 (PR #2313): expiry is terminal — a late approval must not
    overwrite it."""
    store = _store(tmp_path)
    elicit = build_live_elicit_fn(store, poll_seconds=0.02, timeout_seconds=0.1)
    with pytest.raises(TimeoutError):
        _run(elicit("ok?", {}))
    expired = [r for r in store.list_approval_requests() if r.status == "expired"]
    assert len(expired) == 1
    assert store.resolve_approval_request(expired[0].request_id, "approved") is None
    still = [r for r in store.list_approval_requests() if r.request_id == expired[0].request_id]
    assert still[0].status == "expired"


def test_qodo2_oversized_sampling_request_refused(monkeypatch):
    """Qodo #2 hardening (PR #2313): a server cannot stuff an unbounded prompt."""
    from tldw_chatbook.MCP import live_server_request_wiring as w
    monkeypatch.setattr(w, "get_cli_setting", lambda s, k, d=None: d)
    monkeypatch.setattr(w, "chat_api_call", lambda **k: (_ for _ in ()).throw(AssertionError("must not reach provider")))
    complete = build_live_complete_fn()
    huge = [{"role": "user", "content": "x" * 250_000}]
    with pytest.raises(ValueError):
        _run(complete(huge, 10, None))


# --- TASK-28228: end-to-end through the REAL factory-built dispatcher ---
# The per-component builders are covered above; these drive
# build_server_request_dispatcher_factory(store)(server_id).handle(...) for a
# real sampling and elicitation request, i.e. the "fulfilled and returned"
# behavior AC#1/#2 describe, through the same object _get_client wires in prod.

def _allow(monkeypatch, servers):
    from tldw_chatbook.MCP import live_server_request_wiring as w
    settings = {("mcp", "sampling_allowed_servers"): list(servers)}
    monkeypatch.setattr(w, "get_cli_setting", lambda s, k, d=None: settings.get((s, k), d))


def test_factory_dispatcher_fulfills_sampling_end_to_end(tmp_path, monkeypatch):
    """AC#1: an allowlisted server's sampling request routes through the live
    provider (chat_api_call) and comes back as a well-formed MCP result."""
    from tldw_chatbook.MCP import live_server_request_wiring as w
    _allow(monkeypatch, ["srv"])
    seen = {}

    def fake_chat_api_call(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "sampled-answer"}}]}

    monkeypatch.setattr(w, "chat_api_call", fake_chat_api_call)
    factory = build_server_request_dispatcher_factory(_store(tmp_path))
    dispatcher = factory("srv")

    result = _run(dispatcher.handle(
        "sampling/createMessage",
        {
            "messages": [{"role": "user", "content": {"type": "text", "text": "hi there"}}],
            "maxTokens": 32,
        },
    ))
    assert isinstance(result, dict)
    assert result["role"] == "assistant"
    assert result["content"] == {"type": "text", "text": "sampled-answer"}
    assert result["stopReason"] == "endTurn"
    assert seen.get("streaming") is False  # non-streaming, bounded (AC#1)
    assert seen.get("max_tokens") == 32


def test_factory_dispatcher_denies_sampling_for_unlisted_server(tmp_path, monkeypatch):
    """AC#3 end-to-end: default-deny stops a non-allowlisted server's sampling
    request at the dispatcher, before the provider is ever called."""
    from tldw_chatbook.MCP import live_server_request_wiring as w
    from tldw_chatbook.MCP.server_request_handlers import JsonRpcError
    _allow(monkeypatch, ["srv"])
    monkeypatch.setattr(
        w, "chat_api_call",
        lambda **k: (_ for _ in ()).throw(AssertionError("provider must not be reached")),
    )
    dispatcher = build_server_request_dispatcher_factory(_store(tmp_path))("other")
    result = _run(dispatcher.handle(
        "sampling/createMessage",
        {"messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}], "maxTokens": 8},
    ))
    assert isinstance(result, JsonRpcError)
    assert result.code == -32001  # request refused (policy deny)


@pytest.mark.bootstrap_profile
def test_factory_dispatcher_fulfills_elicitation_end_to_end(tmp_path):
    """AC#2: an elicitation request routes through the live approval surface
    (this store) and the out-of-band approval is returned to the server."""
    store = _store(tmp_path)
    # short poll so the test does not wait on the default cadence
    factory = build_server_request_dispatcher_factory(store)
    dispatcher = factory("srv")

    async def drive():
        task = asyncio.create_task(dispatcher.handle(
            "elicitation/create", {"message": "Proceed?", "requestedSchema": {}}
        ))
        for _ in range(200):
            pending = [r for r in store.list_approval_requests() if r.status == "pending"]
            if pending:
                store.resolve_approval_request(pending[0].request_id, "approved")
                break
            await asyncio.sleep(0.02)
        return await task

    result = _run(drive())
    assert result == {"action": "accept", "content": {}}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_id", [1, "1"])
async def test_wire_cancellation_expires_only_the_matching_confirmation(
    tmp_path, monkeypatch, cancel_id
):
    """Route real stdio cancellation frames through the live dispatcher/store.

    Args:
        tmp_path: Private directory for the real approval store.
        monkeypatch: Configure missing settings without touching user config.
        cancel_id: Integer or string inbound ID, kept distinct on the wire.
    """
    import json
    from types import SimpleNamespace

    from tldw_chatbook.MCP import live_server_request_wiring as wiring
    from tldw_chatbook.MCP.client import _StdioJSONRPCConnection

    monkeypatch.setattr(
        wiring, "get_cli_setting", lambda section, key, default=None: default
    )
    store = _store(tmp_path)
    dispatcher = wiring.build_server_request_dispatcher_factory(store)("server")
    reader = asyncio.StreamReader()
    sent = []

    class Writer:
        def write(self, data):
            """Capture bytes written by the production JSON-RPC sender.

            Args:
                data: Encoded outgoing JSON-RPC frame.
            """
            sent.append(json.loads(data))

        async def drain(self):
            """Expose the transport drain boundary."""

        def close(self):
            """Expose the transport closure boundary."""

    conn = _StdioJSONRPCConnection(
        SimpleNamespace(stdout=reader, stderr=None, stdin=Writer(), returncode=0),
        client_name="test",
        server_request_dispatcher=dispatcher.handle,
    )

    def frame(payload):
        """Deliver a server frame through the production read loop.

        Args:
            payload: JSON-RPC server request or notification.
        """
        reader.feed_data(json.dumps(payload).encode() + b"\n")

    async def wait_for(predicate):
        """Bound a wait for the observed transport/store state.

        Args:
            predicate: State condition to observe.
        """
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.005)

    # Opposite-direction IDs belong to a separate namespace.
    outgoing = asyncio.get_running_loop().create_future()
    conn._pending_requests[1] = outgoing
    try:
        for request_id in (1, "1"):
            frame(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "elicitation/create",
                    "params": {
                        "message": f"Confirm {type(request_id).__name__}?",
                        "requestedSchema": {},
                    },
                }
            )
        await wait_for(lambda: len(store.list_approval_requests()) == 2)
        assert all(r.status == "pending" for r in store.list_approval_requests())
        frame(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "elicitation/create",
                "params": {"message": "Duplicate?", "requestedSchema": {}},
            }
        )
        for malformed in (None, True, 1.0, [], {}, "unknown"):
            frame(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": malformed},
                }
            )
        frame({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": []})
        await asyncio.sleep(0.02)
        assert not conn._read_task.done()
        assert all(r.status == "pending" for r in store.list_approval_requests())
        frame(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": cancel_id},
            }
        )
        await wait_for(
            lambda: any(r.status == "expired" for r in store.list_approval_requests())
        )
        (expired,) = [
            r for r in store.list_approval_requests() if r.status == "expired"
        ]
        assert expired.payload["message"] == f"Confirm {type(cancel_id).__name__}?"
        assert store.resolve_approval_request(expired.request_id, "approved") is None
        assert not outgoing.done()
        (pending,) = [
            r for r in store.list_approval_requests() if r.status == "pending"
        ]
        store.resolve_approval_request(pending.request_id, "approved")
        await wait_for(lambda: not conn._dispatch_tasks)
        assert len(sent) == 1
        assert sent[0]["id"] == ("1" if type(cancel_id) is int else 1)
        assert sent[0]["result"] == {"action": "accept", "content": {}}
        assert not conn._server_request_tasks
        # Disconnect still aborts an indefinite human wait and drains ownership.
        frame(
            {
                "jsonrpc": "2.0",
                "id": "disconnect",
                "method": "elicitation/create",
                "params": {"message": "Disconnect?", "requestedSchema": {}},
            }
        )
        await wait_for(
            lambda: any(r.status == "pending" for r in store.list_approval_requests())
        )
        (abandoned,) = [
            r for r in store.list_approval_requests() if r.status == "pending"
        ]
        conn._pending_requests.pop(1, None)
        await conn.close()
        await wait_for(
            lambda: all(r.status != "pending" for r in store.list_approval_requests())
        )
        assert store.resolve_approval_request(abandoned.request_id, "approved") is None
        assert not conn._dispatch_tasks and not conn._server_request_tasks
        assert len(sent) == 1
    finally:
        conn._pending_requests.pop(1, None)
        outgoing.cancel()
        await conn.close()
        await asyncio.sleep(0)
