from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Chat import ServerChatLoopScopeService, ServerChatLoopService


class FakePolicyEnforcer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def start_chat_loop_run(self, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("start_chat_loop_run", payload))
        return {"run_id": "run_1"}

    async def list_chat_loop_events(self, run_id: str, after_seq: int = 0):
        self.calls.append(("list_chat_loop_events", run_id, after_seq))
        return {
            "run_id": run_id,
            "events": [
                {
                    "run_id": run_id,
                    "seq": 1,
                    "ts": "2026-04-23T18:00:00Z",
                    "event": "run_started",
                    "data": {"messages_count": 1},
                }
            ],
        }

    async def approve_chat_loop_call(self, run_id: str, approval_id: str):
        self.calls.append(("approve_chat_loop_call", run_id, approval_id))
        return {"ok": True}

    async def reject_chat_loop_call(self, run_id: str, approval_id: str):
        self.calls.append(("reject_chat_loop_call", run_id, approval_id))
        return {"ok": True}

    async def cancel_chat_loop_run(self, run_id: str):
        self.calls.append(("cancel_chat_loop_run", run_id))
        return {"ok": True}


class FakeCachingProvider:
    def __init__(self, client_factory):
        self.client_factory = client_factory
        self.client = None
        self.build_calls = 0
        self.constructed_clients = 0

    def build_client(self):
        self.build_calls += 1
        if self.client is None:
            self.client = self.client_factory()
            self.constructed_clients += 1
        return self.client


class ExplodingProvider:
    def __init__(self):
        self.calls = 0

    def build_client(self):
        self.calls += 1
        raise AssertionError("provider should not be used")


@pytest.mark.asyncio
async def test_server_chat_loop_service_reuses_provider_cached_client_across_operations():
    provider = FakeCachingProvider(FakeClient)
    service = ServerChatLoopService.from_server_context_provider(provider)

    started = await service.start_run(messages=[{"role": "user", "content": "hello"}])
    events = await service.list_events("run_1", after_seq=1)

    assert started == {"run_id": "run_1"}
    assert events["run_id"] == "run_1"
    assert provider.build_calls == 2
    assert provider.constructed_clients == 1
    assert provider.client.calls == [
        ("start_chat_loop_run", {"messages": [{"role": "user", "content": "hello"}]}),
        ("list_chat_loop_events", "run_1", 1),
    ]


@pytest.mark.asyncio
async def test_server_chat_loop_service_direct_client_takes_precedence_over_provider():
    client = FakeClient()
    provider = ExplodingProvider()
    service = ServerChatLoopService(client=client, client_provider=provider)

    await service.cancel("run_1")

    assert provider.calls == 0
    assert client.calls == [("cancel_chat_loop_run", "run_1")]


@pytest.mark.asyncio
async def test_server_chat_loop_service_routes_typed_client_calls():
    client = FakeClient()
    service = ServerChatLoopService(client=client)

    started = await service.start_run(
        messages=[{"role": "user", "content": "hello"}],
        conversation_id="conv-1",
    )
    events = await service.list_events("run_1", after_seq=3)
    approved = await service.approve("run_1", "approval-1")
    rejected = await service.reject("run_1", "approval-2")
    cancelled = await service.cancel("run_1")

    assert started == {"run_id": "run_1"}
    assert events["events"][0]["event"] == "run_started"
    assert approved == {"ok": True}
    assert rejected == {"ok": True}
    assert cancelled == {"ok": True}
    assert client.calls == [
        (
            "start_chat_loop_run",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "conversation_id": "conv-1",
            },
        ),
        ("list_chat_loop_events", "run_1", 3),
        ("approve_chat_loop_call", "run_1", "approval-1"),
        ("reject_chat_loop_call", "run_1", "approval-2"),
        ("cancel_chat_loop_run", "run_1"),
    ]


@pytest.mark.asyncio
async def test_chat_loop_scope_service_enforces_server_mode_policy_and_normalizes_ids():
    policy = FakePolicyEnforcer()
    service = ServerChatLoopService(client=FakeClient())
    scope = ServerChatLoopScopeService(server_service=service, policy_enforcer=policy)

    started = await scope.start_run(
        mode="server",
        messages=[{"role": "user", "content": "hello"}],
        conversation_id="conv-1",
    )
    events = await scope.list_events(mode="server", run_id="run_1", after_seq=3)
    approved = await scope.approve(mode="server", run_id="run_1", approval_id="approval-1")
    rejected = await scope.reject(mode="server", run_id="run_1", approval_id="approval-2")
    cancelled = await scope.cancel(mode="server", run_id="run_1")

    assert started["id"] == "server:chat_loop_run:run_1"
    assert started["backend"] == "server"
    assert started["entity_kind"] == "chat_loop_run"
    assert events["events"][0]["id"] == "server:chat_loop_event:run_1:1"
    assert events["events"][0]["backend"] == "server"
    assert approved == {"ok": True}
    assert rejected == {"ok": True}
    assert cancelled == {"ok": True}
    assert policy.calls == [
        "chat.launch.server",
        "chat.detail.server",
        "chat.launch.server",
        "chat.launch.server",
        "chat.launch.server",
    ]


@pytest.mark.asyncio
async def test_chat_loop_scope_service_rejects_local_mode_before_policy_dispatch():
    policy = FakePolicyEnforcer()
    scope = ServerChatLoopScopeService(server_service=ServerChatLoopService(client=FakeClient()), policy_enforcer=policy)

    with pytest.raises(ValueError, match="Server chat loop requires server mode"):
        await scope.start_run(mode="local", messages=[{"role": "user", "content": "hello"}])

    assert policy.calls == []
