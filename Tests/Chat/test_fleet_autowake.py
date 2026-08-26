"""PR 3a-2 Task 5: auto-wake -- a settling survivor re-invokes its supervisor.

Spec §3 invariant 5 (corrected 2026-08-11): a finished background child
wakes its supervisor so it can act on the result, instead of the result
sitting in ``agent_runs.result`` until the user happens to send another
message in that conversation. The wake is machine-origin end to end: no
USER transcript row, the composer untouched, a SYSTEM-class notice row
marked ``origin="agent_wake"``, and the model payload carrying the fenced
results as an explicitly-labelled not-user-input injection.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from Tests.Chat.test_child_run_scope_ordering import _survivor_bridge
from Tests.Chat.test_console_agent_bridge import (
    _fence,
    _join_fleet_threads,
    _run,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from Tests.console_provider_doubles import provider_resolution


async def _settle(predicate, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


class _WakeStreamingGateway:
    """Controller-side provider fake for the wake turn's plain path."""

    def __init__(self):
        self.messages_seen: list[list[dict]] = []

    async def resolve_for_send(self, selection):
        return provider_resolution(
                   ready=True,
                   provider="llama_cpp",
                   model="test-model",
                   base_url="http://127.0.0.1:9099",
                   visible_copy="",
               )

    async def stream_chat(self, resolution, messages, **kwargs):
        self.messages_seen.append(list(messages))
        yield "acting on the sub-agent result"


# ---------------------------------------------------------------------------
# The gap, red-first: a survivor settles and the supervisor is woken.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_settling_survivor_wakes_its_supervisor_with_the_result(tmp_path):
    """The full chain (spec §3 invariant 5): a fleet child that outlived
    its turn settles -> the supervisor is re-invoked in that conversation
    with the child's DB-sourced result, via a machine-origin turn:

    - the transcript gains a SYSTEM-class notice row marked
      ``origin="agent_wake"`` and NO new USER row;
    - the model payload's final message carries the fenced result text and
      the explicit not-user-input labelling;
    - a new assistant reply streams in the same session.
    """
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["turn final"],
        ],
        needed=1,
    )
    wake_gateway = _WakeStreamingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=wake_gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=False,
    )
    pre_turn_user_rows = sum(
        1
        for message in store.messages_for_session(session.id)
        if message.role is ConsoleMessageRole.USER
    )
    try:
        outcome = _run(
            bridge,
            store,
            session,
            aid,
            conversation_id=controller._agent_conversation_id(session.id),
        )
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"
    finally:
        gate.set()
    _join_fleet_threads()

    def _wake_notice_rows():
        return [
            message
            for message in store.messages_for_session(session.id)
            if message.role is ConsoleMessageRole.SYSTEM
            and message.metadata is not None
            and message.metadata.origin == "agent_wake"
        ]

    woke = await _settle(lambda: bool(_wake_notice_rows()))
    assert woke, (
        "the survivor settled and no wake turn ever fired: the transcript "
        "gained no machine-origin notice row -- the result is sitting "
        "unread in agent_runs.result"
    )
    notice = _wake_notice_rows()[-1]
    assert "not user input" in notice.content
    assert "child answer" in notice.content, (
        "the notice must carry the finished child's DB result"
    )

    replied = await _settle(
        lambda: any(
            "acting on the sub-agent result" in message.content
            for message in store.messages_for_session(session.id)
            if message.role is ConsoleMessageRole.ASSISTANT
        )
    )
    assert replied, "the wake turn produced no assistant reply"

    user_rows = sum(
        1
        for message in store.messages_for_session(session.id)
        if message.role is ConsoleMessageRole.USER
    )
    assert user_rows == pre_turn_user_rows, (
        "a wake send must write NO USER transcript row"
    )

    assert wake_gateway.messages_seen, "the wake turn never reached the provider"
    final_payload = wake_gateway.messages_seen[-1][-1]
    assert final_payload["role"] == "user"
    assert "child answer" in final_payload["content"]
    assert "not user input" in final_payload["content"]
