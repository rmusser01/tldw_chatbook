"""TASK-33621.80: automatic turns over the Ask compaction trigger.

TASK-33621.4 stopped every send a user starts before commit at the Ask
trigger. Two automatic turns were left: an agent wake and a goal iteration.

An agent wake used to be accepted, commit its notice and an empty assistant
row, and then hit the post-commit Ask block: the row was marked Failed, a
System row said "Conversation context reached its compaction threshold", and
the wake was completed (its results consumed) without the agent ever seeing
them. Now the wake stops before acceptance: its chain pauses with the
``compaction_ask`` code (no retry timer, results kept), and it goes on once
the user compacts the chat.

A goal iteration sends fresh context -- the goal handoff, never the chat's
history -- so the Ask trigger, which counts history, cannot fire for it. The
pin below fails if that ever changes.

Real ChaChaNotes DB, store, controller, request preparation, compaction
service, fleet-wake coordinator, automatic-work ledger and Console runtime;
only the provider network is doubled.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from Tests.Chat.test_console_compaction_live_session import (
    _MODEL,
    _OVERRIDES,
    _LiveProviderGateway,
)
from Tests.Chat.test_console_fleet_wake import (
    _drain,
    _FakeWakeBridge,
    _survivor,
    _terminal_subagent_run,
)
from Tests.Chat.test_fleet_attention import _AppStub
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.bootstrap_profile

_SESSION = "session-1"
_OFF = replace(_OVERRIDES, compaction_mode=ContextCompactionMode.OFF)
_ASK = replace(_OVERRIDES, compaction_mode=ContextCompactionMode.ASK)


async def _settle(predicate, seconds: float = 8.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


class _WakeRig:
    """A saved chat over its Ask trigger with one finished background child."""

    def __init__(self, tmp_path: Path) -> None:
        self.db = CharactersRAGDB(tmp_path / "chat.sqlite", client_id="task33621-80")
        self.runs_db = AgentRunsDB(tmp_path / "runs.db", client_id="task33621-80")
        self.app = _AppStub(self.db)
        self.app.local_chat_conversation_service = ChatConversationService(self.db)
        self.store = ConsoleChatStore(persistence=ChatPersistenceService(self.db))
        self.store.create_session(session_id=_SESSION, title="Research")
        self.store.set_session_context_policy_overrides(_SESSION, _OFF)
        self.gateway = _LiveProviderGateway()
        self.controller = ConsoleChatController(
            store=self.store,
            provider_gateway=self.gateway,
            provider="openai",
            model=_MODEL,
            agent_bridge=_FakeWakeBridge(self.runs_db),
            agent_runtime_enabled=False,
        )
        self.controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
        self.runtime: ConsoleRuntime | None = None
        self.cid = ""
        self.chain = ""
        self.child = ""

    async def over_the_ask_trigger(self) -> None:
        """Send real turns under Off, then switch the chat to Ask."""

        self.runtime = ConsoleRuntime(self.app)
        self.runtime.set_chat_store(self.store)
        self.runtime.set_chat_controller(self.controller)
        self.controller.fleet_wake.wire(app=self.app)
        for index in range(12):
            result = await self.controller.submit_draft(
                f"question-{index}: explain step {index} in detail.",
                session_id=_SESSION,
            )
            assert result.accepted is True, result.visible_copy
            self.store.set_session_context_policy_overrides(_SESSION, _ASK)
            # A composer send on this chat would now be held at Ask.
            if await self._composer_send_would_hold():
                break
            self.store.set_session_context_policy_overrides(_SESSION, _OFF)
        else:
            raise AssertionError("the custom budget was never crossed")
        session = next(s for s in self.store.sessions() if s.id == _SESSION)
        self.cid = str(session.persisted_conversation_id)

    async def _composer_send_would_hold(self) -> bool:
        resolution = await self.gateway.resolve_for_send(None)
        hold, _alert = await self.controller._assess_context_compaction(
            session_id=_SESSION,
            resolution=resolution,
            provider_messages=[
                *self.controller._provider_messages_for_session(
                    _SESSION, annotate_ids=True
                ),
                {"role": "user", "content": "one more question"},
            ],
            uncommitted_user_message_id=None,
        )
        return hold is not None

    def finish_a_background_child(self) -> None:
        self.chain = self.runs_db.automatic_work.create_chain(
            self.cid, root_submission_id=uuid4().hex
        )
        _parent, self.child = _terminal_subagent_run(
            self.runs_db,
            self.cid,
            result="Saved research result",
            work_chain_id=self.chain,
        )
        self.controller.fleet_wake.on_fleet_drained(
            _drain(self.cid, _survivor(self.child, session_id=_SESSION))
        )

    async def wake_paused(self) -> bool:
        """Wait until the wake is paused and its delivery has fully ended."""

        wake = self.controller.fleet_wake
        return await _settle(
            lambda: (
                wake.pause_reason(self.cid) == "compaction_ask"
                and not wake._active
                and not wake._delivery_tasks
                and not self.runtime._turn_custody
            )
        )

    def attempts(self) -> list[str]:
        with self.runs_db.connection() as conn:
            return [
                row["state"]
                for row in conn.execute(
                    "SELECT state FROM automatic_wake_attempts "
                    "WHERE conversation_id = ? ORDER BY rowid",
                    (self.cid,),
                )
            ]

    def failed_rows(self) -> list[str]:
        return [
            message.id
            for message in self.store.messages_for_session(_SESSION)
            if message.status == "failed"
        ]

    def wake_notices(self) -> list[str]:
        return [
            message.content
            for message in self.store.messages_for_session(_SESSION)
            if getattr(message.metadata, "origin", "") == "agent_wake"
        ]

    async def close(self) -> None:
        if self.runtime is not None:
            await self.runtime.dispose()
        self.runs_db.close()
        self.db.close_connection()


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
    return _WakeRig(tmp_path)


@pytest.mark.asyncio
async def test_an_agent_wake_over_the_ask_trigger_pauses_before_commit(rig) -> None:
    """AC#2: no Failed turn, no unattended model call, no retry loop."""

    try:
        await rig.over_the_ask_trigger()
        streams = rig.gateway.stream_calls
        messages_before = len(rig.store.messages_for_session(_SESSION))
        rig.finish_a_background_child()
        wake = rig.controller.fleet_wake

        assert await _settle(lambda: rig.attempts() and not wake._active), (
            "the wake never ran"
        )
        # Longer than the coordinator's 1 s pre-acceptance retry delay.
        await asyncio.sleep(2.5)

        assert rig.failed_rows() == [], "a policy pause must not mark a turn Failed"
        assert rig.wake_notices() == [], "nothing is committed before acceptance"
        assert not any(
            message.role is ConsoleMessageRole.ASSISTANT
            for message in rig.store.messages_for_session(_SESSION)[messages_before:]
        )
        assert rig.gateway.stream_calls == streams, "no model call for the wake"
        assert rig.gateway.auxiliary_calls == 0, "an unattended wake never compacts"
        assert rig.attempts() == ["aborted"], "one refused attempt, then no retry"
        assert wake.pause_reason(rig.cid) == "compaction_ask"
        assert wake.has_pending(rig.cid), "the results stay pending"
        assert rig.runs_db.get_run(rig.child)["wake_delivered_at"] is None
        snapshot = rig.runs_db.automatic_work.snapshot(rig.chain)
        assert (snapshot.status, snapshot.pause_reason) == ("paused", "compaction_ask")
        assert snapshot.used["generation"] == 0, "a refused wake spends no turn"
        copy = rig.controller.run_state_for(_SESSION).visible_copy
        assert "/compact" in copy and "saved" in copy, copy
    finally:
        await rig.close()


@pytest.mark.asyncio
async def test_compacting_the_chat_lets_the_paused_wake_go_on_once(rig) -> None:
    """The user's decision resumes the wake: /compact, then one delivery."""

    try:
        await rig.over_the_ask_trigger()
        rig.finish_a_background_child()
        wake = rig.controller.fleet_wake
        assert await rig.wake_paused(), (
            rig.failed_rows(),
            rig.controller.run_state_for(_SESSION).visible_copy,
        )
        streams = rig.gateway.stream_calls

        compacted, copy = await rig.controller.compact_context_now(_SESSION)

        assert compacted is True, copy
        assert rig.gateway.auxiliary_calls == 1, "only the user's compaction"
        assert await _settle(lambda: not wake.has_pending(rig.cid)), (
            rig.attempts(),
            wake.pause_reason(rig.cid),
        )
        assert await _settle(lambda: not wake._active)
        assert rig.gateway.stream_calls == streams + 1, "the wake replied once"
        assert len(rig.wake_notices()) == 1
        assert "Saved research result" in rig.wake_notices()[0]
        assistant = [
            message
            for message in rig.store.messages_for_session(_SESSION)
            if message.role is ConsoleMessageRole.ASSISTANT
        ][-1]
        assert assistant.status == "complete"
        assert rig.failed_rows() == []
        assert wake.pause_reason(rig.cid) is None
        assert rig.runs_db.get_run(rig.child)["wake_delivered_at"]
        assert rig.attempts() == ["aborted", "completed"]
        assert rig.runs_db.automatic_work.snapshot(rig.chain).status == "active"
    finally:
        await rig.close()


@pytest.mark.asyncio
async def test_rewind_summarize_also_lets_the_paused_wake_go_on(rig) -> None:
    """/rewind Summarize up to here is a compaction too."""

    try:
        await rig.over_the_ask_trigger()
        rig.finish_a_background_child()
        wake = rig.controller.fleet_wake
        assert await rig.wake_paused()
        streams = rig.gateway.stream_calls
        last_prompt = [
            message
            for message in rig.store.messages_for_session(_SESSION)
            if message.role is ConsoleMessageRole.USER
        ][-1]

        summarized = await rig.controller.summarize_up_to(last_prompt.id)

        assert summarized.accepted is True, summarized.visible_copy
        assert await _settle(lambda: not wake.has_pending(rig.cid)), (
            rig.attempts(),
            wake.pause_reason(rig.cid),
        )
        assert await _settle(lambda: not wake._active)
        assert rig.gateway.stream_calls == streams + 1
        assert len(rig.wake_notices()) == 1
        assert rig.failed_rows() == []
        assert rig.attempts() == ["aborted", "completed"]
    finally:
        await rig.close()


@pytest.mark.asyncio
async def test_sending_your_own_message_without_compacting_leaves_the_wake_paused(
    rig,
) -> None:
    """Send without compacting is the user's own turn; the results wait."""

    try:
        await rig.over_the_ask_trigger()
        rig.finish_a_background_child()
        wake = rig.controller.fleet_wake
        assert await rig.wake_paused()
        streams = rig.gateway.stream_calls

        held = await rig.controller.submit_draft(
            "carry on with the saved result", session_id=_SESSION
        )
        assert held.accepted is False and held.preparation_id is not None
        result = await rig.controller.send_without_compacting(held.preparation_id)

        assert result.accepted is True, result.visible_copy
        assert await _settle(lambda: rig.gateway.stream_calls == streams + 1)
        assert await _settle(lambda: not wake._active)
        await asyncio.sleep(1.5)
        assert rig.gateway.stream_calls == streams + 1, "no wake turn followed"
        assert rig.wake_notices() == []
        assert wake.pause_reason(rig.cid) == "compaction_ask"
        assert wake.has_pending(rig.cid)
        assert rig.attempts() == ["aborted"]
        assert rig.failed_rows() == []
    finally:
        await rig.close()


@pytest.mark.asyncio
async def test_turning_ask_off_lets_the_paused_wake_go_on_after_the_next_turn(
    rig,
) -> None:
    """Changing When limit nears is a decision too; the next settle retries."""

    try:
        await rig.over_the_ask_trigger()
        rig.finish_a_background_child()
        wake = rig.controller.fleet_wake
        assert await rig.wake_paused()
        streams = rig.gateway.stream_calls
        rig.store.set_session_context_policy_overrides(_SESSION, _OFF)

        result = await rig.controller.submit_draft(
            "carry on with the saved result", session_id=_SESSION
        )

        assert result.accepted is True, result.visible_copy
        assert await _settle(lambda: not wake.has_pending(rig.cid)), (
            rig.attempts(),
            wake.pause_reason(rig.cid),
        )
        assert await _settle(lambda: not wake._active)
        assert rig.gateway.stream_calls == streams + 2, "the send, then the wake"
        assert rig.gateway.auxiliary_calls == 0, "Off never compacts"
        assert len(rig.wake_notices()) == 1
        assert rig.failed_rows() == []
        assert rig.attempts() == ["aborted", "completed"]
    finally:
        await rig.close()


# -- goal iterations -------------------------------------------------------

from Tests.Chat.test_automatic_provider_budget import (  # noqa: E402
    resolution as _goal_resolution,
)
from Tests.Chat.test_console_goal_dispatch import build_goal_rig  # noqa: E402
from Tests.Chat.test_goal_conversation_provisioning import (  # noqa: E402
    stores as _stores_fixture,
)

stores = _stores_fixture


@pytest.mark.asyncio
async def test_a_goal_iteration_over_the_ask_trigger_sends_fresh_context(
    stores, monkeypatch
) -> None:
    """AC#1: neither held nor Failed, because the goal never sends history.

    The Ask trigger counts the chat's history; a goal iteration sends only its
    handoff. If a goal iteration ever starts carrying history, this pin fails:
    it must then stop before commit and the goal must say why and how to go
    on, the way an agent wake pauses (see the tests above).
    """

    goal, store, session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    try:
        store.set_session_context_policy_overrides(session.id, _ASK)
        for index in range(10):
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content=f"history-{index}: explain step {index}.",
                persist=True,
            )
            store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content=f"history-answer-{index} " + "detail " * 150,
                persist=True,
            )
        hold, _alert = await controller._assess_context_compaction(
            session_id=session.id,
            resolution=_goal_resolution(),
            provider_messages=[
                *controller._provider_messages_for_session(
                    session.id, annotate_ids=True
                ),
                {"role": "user", "content": "one more question"},
            ],
            uncommitted_user_message_id=None,
        )
        assert hold is not None, "the chat must be over its Ask trigger"

        result = await coordinator.dispatch_once(goal.id)

        assert result.outcome is not None and result.outcome.status == "done", result
        assert len(calls) == 1, "the goal iteration reached the provider once"
        assert "history-" not in str(calls[0]), "a goal sends fresh context"
        assert "Repair fixture" in str(calls[0])
        rows = store.messages_for_session(session.id)
        assert [m.id for m in rows if m.status == "failed"] == []
        assert not any(
            "compaction threshold" in m.content
            for m in rows
            if m.role is ConsoleMessageRole.SYSTEM
        )
        assert store.preparation_for_session(session.id) is None
        assert stores[0].goal_runs.get(goal.id).pause_reason is None
    finally:
        await gateway.aclose()
