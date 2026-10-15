"""MobileChatService — parent-process Console engine binding (ADR-234).

Task 3 of the served-mobile Phase 1b plan. Real engine over tmp_path SQLite
DBs; the provider is the Task-2 ``AgentModeGateway`` double (agent-capable:
emits one native ``fs_list`` tool call when tools are advertised, then the
final answer once the tool result returns). The service test IS the spike,
productized: the four tests below restate the Binding-1 probe's assertions
against the ``MobileChatService`` surface the HTTP/WS layer will consume.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from Tests.console_provider_doubles import AgentModeGateway

from tldw_chatbook.Web_Server.mobile_service import MobileChatService, MobileSendRefused

# The real engine's submit path reads the guarded config loader (turn
# configuration, [tools] gates, hooks admission). Under pytest's per-test
# config redirect those reads fail closed with RecoveryRequired — the same
# admission signature Tests/conftest.py documents for the console suites
# (TASK-19425/22302) — so this module keeps the bootstrap config profile,
# like the real-engine Tests/Chat suites (e.g. test_automatic_wake_
# recovery.py). All databases stay tmp_path-owned.
pytestmark = pytest.mark.bootstrap_profile

FINAL_ANSWER = "MOBILE_AGENT_ROUND_TRIP_COMPLETE"
WORKSPACE_ID = "ws-mobile-1b"


async def _subscribe(service: MobileChatService, sink: list[Any]) -> None:
    """Background subscriber: append every fanned-out event to ``sink``."""
    async for event in service.events():
        sink.append(event)


async def _wait_for(predicate, timeout: float, what: str) -> None:
    """Poll ``predicate`` at 0.1 s until true; fail with ``what`` on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def _bound_workspace(service: MobileChatService, tmp_path: Path) -> str:
    """One workspace + one folder binding through the service's registry.

    Mirrors the spike probe's ``ws-mobile-spike`` fixture: a workspace-bound
    chat is the realistic mobile case (default-workspace sessions get no
    ``fs_*``/``git_*`` tools at all).
    """
    folder = tmp_path / "code"
    (folder / "src").mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text("# mobile\n", encoding="utf-8")
    registry = service.workspace_registry
    registry.create_workspace(workspace_id=WORKSPACE_ID, name="Mobile 1b")
    registry.add_folder_binding(WORKSPACE_ID, folder, allow_write=True)
    return WORKSPACE_ID


async def _shutdown(service: MobileChatService, subscriber: asyncio.Task) -> None:
    subscriber.cancel()
    try:
        await subscriber
    except asyncio.CancelledError:
        pass
    await asyncio.wait_for(service.shutdown(), timeout=15)


async def test_agent_round_trip_with_approval(tmp_path: Path) -> None:
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    # Two subscribers: the first holds its own queue (fan-out sanity), the
    # second collects the events the asserts read.
    subscriber = asyncio.create_task(_subscribe(service, []))
    got: list[Any] = []
    collector = asyncio.create_task(_subscribe(service, got))
    try:
        workspace_id = _bound_workspace(service, tmp_path)
        session = await service.create_session(
            title="phone chat", workspace_id=workspace_id
        )
        assert session["workspace_id"] == workspace_id
        assert session["has_folder_tools"] is True

        send_task = asyncio.create_task(
            service.send(session["id"], "list the folder")
        )
        await _wait_for(
            lambda: any(e.type == "approval_requested" for e in got),
            timeout=30,
            what="approval_requested event",
        )
        # Non-blocking send (live-check round 1): the send call must have
        # returned already — the turn is parked on this approval, which
        # only this test can resolve, so a blocking send would still hang.
        assert send_task.done() and not send_task.exception()
        approvals = [e for e in got if e.type == "approval_requested"]
        payload = approvals[0].payload
        assert payload["calls"][0]["llm_name"] == "fs_list"
        assert payload["round_id"]

        # The pending round is also visible on the polling surface.
        listed = service.list_approvals()
        assert any(a["round_id"] == payload["round_id"] for a in listed)

        call = payload["calls"][0]
        assert (
            service.decide_approval(
                payload["round_id"], {call["call_id"]: "approve_once"}
            )
            is True
        )
        # Deciding an unknown round is reported, not raised.
        assert service.decide_approval("no-such-round", {"x": "deny"}) is False

        await asyncio.wait_for(send_task, timeout=60)
        await _wait_for(
            lambda: any(
                e.type == "run_completed" and e.session_id == session["id"]
                for e in got
            ),
            timeout=15,
            what="run_completed event",
        )

        messages = service.session_messages(session["id"])
        assert any(
            m["role"] == "assistant" and FINAL_ANSWER in m["content"]
            for m in messages["messages"]
        )
        # The tool really ran: the provider saw the fs_list result.
        assert gateway.tool_result_messages, "fs_list result never reached provider"

        deltas = [e for e in got if e.type == "delta" and e.session_id == session["id"]]
        assert deltas, "no streaming deltas observed during the run"

        # Wire shape: monotonic per-service seq, JSON-serializable payloads.
        seqs = [e.seq for e in got]
        assert seqs == sorted(seqs)
        for event in got:
            json.dumps(event.to_wire())
    finally:
        await _shutdown(service, collector)
        await _shutdown(service, subscriber)


async def test_send_is_serialized_per_session(tmp_path: Path) -> None:
    """Turns serialize per session; a concurrent send is refused (round 1).

    The non-blocking contract: ``send`` schedules the turn and returns, so
    a second send while the first turn is pending raises ``MobileSendRefused``
    (the engine's own one-run-per-session admission rule, applied at the
    service boundary instead of queueing on the lock). Once the first turn
    completes, the next send is accepted and the transcript order is
    preserved: each user row lands only after the previous turn's
    assistant row.
    """
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    got: list[Any] = []
    subscriber = asyncio.create_task(_subscribe(service, got))
    try:
        session = await service.create_session(title="serial")
        await service.send(session["id"], "first question")
        with pytest.raises(MobileSendRefused):
            await service.send(session["id"], "second question too early")
        await _wait_for(
            lambda: any(e.type == "run_completed" for e in got),
            timeout=15,
            what="first run_completed event",
        )

        await service.send(session["id"], "second question")
        await _wait_for(
            lambda: len([e for e in got if e.type == "run_completed"]) == 2,
            timeout=15,
            what="two run_completed events",
        )

        messages = service.session_messages(session["id"])["messages"]
        user_texts = [m["content"] for m in messages if m["role"] == "user"]
        assert user_texts == ["first question", "second question"]
        # The second user row landed only after the first turn's assistant
        # row — the turns really did serialize.
        roles = [m["role"] for m in messages]
        assert roles.count("user") == 2
        second_user_at = roles.index("user", 1)
        assert any(r == "assistant" for r in roles[:second_user_at])
        # Two full provider round-trips (one per turn; each turn's first
        # call attempts the gateway's fs_list call and fails fast because
        # the default workspace never advertises it, then the follow-up
        # call answers).
        assert len(gateway.stream_calls) >= 2
    finally:
        await _shutdown(service, subscriber)


async def test_send_returns_while_the_turn_is_pending(tmp_path: Path) -> None:
    """The 202-shape contract: send schedules and returns (finding 1).

    The gateway's first provider call parks on a gate only this test opens,
    so a send that blocked on ``submit_draft`` (the old shape — the one the
    live check caught) would hang until the timeout. Also proves the
    unknown-session pre-validation still raises before anything is
    scheduled, and that deltas + terminal events flow once the gate opens.
    """

    class _GatedAgentGateway(AgentModeGateway):
        """AgentModeGateway whose provider calls park on ``gate``."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = asyncio.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
            await self.gate.wait()
            async for chunk in super().stream_chat(resolution, messages, **kwargs):
                yield chunk

    gateway = _GatedAgentGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    got: list[Any] = []
    subscriber = asyncio.create_task(_subscribe(service, got))
    try:
        session = await service.create_session(title="pending turn")
        with pytest.raises(KeyError):
            await service.send("no-such-session", "gone")

        loop = asyncio.get_running_loop()
        started = loop.time()
        await asyncio.wait_for(
            service.send(session["id"], "held turn"), timeout=5
        )
        assert loop.time() - started < 2.0, "send blocked on the turn"

        with pytest.raises(MobileSendRefused):
            await service.send(session["id"], "too soon")

        gateway.gate.set()
        await _wait_for(
            lambda: any(
                e.type == "run_completed" and e.session_id == session["id"]
                for e in got
            ),
            timeout=30,
            what="run_completed after gate release",
        )
        deltas = [
            e for e in got if e.type == "delta" and e.session_id == session["id"]
        ]
        assert deltas, "no streaming deltas observed during the run"
        messages = service.session_messages(session["id"])["messages"]
        assert any(
            m["role"] == "user" and m["content"] == "held turn" for m in messages
        )
        assert any(
            m["role"] == "assistant" and FINAL_ANSWER in m["content"]
            for m in messages
        )
    finally:
        gateway.gate.set()  # never leave the parked turn holding shutdown
        await _shutdown(service, subscriber)


async def test_default_workspace_session_has_no_folder_tools(tmp_path: Path) -> None:
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    got: list[Any] = []
    subscriber = asyncio.create_task(_subscribe(service, got))
    try:
        session = await service.create_session(title="no folder")
        listing = service.sessions
        assert listing[0]["workspace_id"] == "global"
        assert listing[0]["has_folder_tools"] is False

        await service.send(session["id"], "hello without tools")
        await _wait_for(
            lambda: any(e.type == "run_completed" for e in got),
            timeout=15,
            what="run_completed event",
        )
        # Real evidence: builtin tools may still be advertised, but the
        # bound-folder path-tool family (fs_list) never is in the default
        # workspace.
        assert gateway.stream_calls, "provider was never called"
        assert all(not c["tool_advertised"] for c in gateway.stream_calls)
    finally:
        await _shutdown(service, subscriber)


async def test_messages_cursor(tmp_path: Path) -> None:
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    got: list[Any] = []
    subscriber = asyncio.create_task(_subscribe(service, got))
    try:
        session = await service.create_session(title="cursor")

        await service.send(session["id"], "turn one")
        await _wait_for(
            lambda: any(e.type == "run_completed" for e in got),
            timeout=15,
            what="first run_completed event",
        )
        first = service.session_messages(session["id"])
        first_ids = [m["id"] for m in first["messages"]]
        assert first["cursor"] == first_ids[-1]

        got.clear()
        await service.send(session["id"], "turn two")
        await _wait_for(
            lambda: len([e for e in got if e.type == "run_completed"]) == 1,
            timeout=15,
            what="second run_completed event",
        )

        after = service.session_messages(
            session["id"], after_message_id=first["cursor"]
        )
        after_ids = [m["id"] for m in after["messages"]]
        assert after_ids, "cursor returned no later messages"
        assert not (set(after_ids) & set(first_ids))
        assert after["cursor"] == after_ids[-1]
        # The second turn's user row is the first row past the cursor.
        assert after["messages"][0]["role"] == "user"
        assert after["messages"][0]["content"] == "turn two"
        assert any(
            m["role"] == "assistant" and FINAL_ANSWER in m["content"]
            for m in after["messages"]
        )
    finally:
        await _shutdown(service, subscriber)
