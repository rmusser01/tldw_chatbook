"""TASK-22000: the restored ADR-046 queue cannot race a durable commit.

The owner decided that a *healthy* in-flight durable turn must not block Send
(ADR-046 / TASK-14808: the button re-labels to "Queue" and admits the draft as
a FIFO follow-up). TASK-19900.3's ``dispatch_recovery_blocks_submission``
previously refused for the app's own live run, so the queue was unreachable.

Narrowing that predicate is only safe if a queued follow-up genuinely cannot
overlap the first turn's durable commit -- two durable owners in one
conversation are refused at the SQLite level ("active dispatch checkpoint"),
so an overlap would surface as a raw ``RuntimeError``. These tests drive the
real interleavings rather than reasoning about them:

* admission attempted *while the commit transaction is open* (refused: there
  is no chain to admit into until the turn is accepted);
* admission accepted mid-stream, then drained (strictly after the first turn
  settles, never alongside it -- at most one checkpoint row exists at a time);
* admission accepted mid-run whose owner then becomes genuinely unhealthy
  (still refused, with a visible reason and the entry returned to the queue).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _restored_store,
)
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_prompt_queue import (
    PromptQueueMode,
    PromptQueuePauseReason,
    QueueMutationStatus,
)
from tldw_chatbook.Chat.console_prompt_queue_coordinator import _PromptChain


class _HoldingGateway:
    """Stream one chunk, then hold the turn open until explicitly released."""

    def __init__(self, db) -> None:
        self.db = db
        self.calls = 0
        self.prompts: list[str] = []
        self.checkpoint_counts: list[int] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.hold = True

    async def resolve_for_send(self, _selection: object) -> object:
        return SimpleNamespace(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            visible_copy="",
            resolved_destination=ConsoleResolvedDestination(
                provider="llama_cpp",
                model="test-model",
                endpoint_identity="http://127.0.0.1:9099",
                egress_class=ConsoleEgressClass.ON_DEVICE,
            ),
        )

    async def stream_chat(
        self, _resolution: object, messages: list[dict[str, Any]], **_kwargs: Any
    ):
        self.calls += 1
        self.prompts.append(str(messages[-1].get("content", "")))
        self.checkpoint_counts.append(_checkpoint_rows(self.db))
        self.started.set()
        yield "done"
        if self.hold:
            await self.release.wait()


def _checkpoint_rows(db) -> int:
    return int(
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
    )


def _message_rows(db) -> int:
    return int(
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    )


@pytest.mark.asyncio
async def test_queued_follow_up_cannot_be_admitted_while_the_durable_commit_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draft offered mid-commit is refused admission, not silently queued."""

    db, store, controller, _gateway = _controller(tmp_path)
    persistence = store.persistence
    assert persistence is not None
    gateway = _HoldingGateway(db)
    gateway.hold = False
    controller.provider_gateway = gateway
    create_conversation = persistence.create_conversation
    observed: list[tuple[bool, bool, bool, QueueMutationStatus, int]] = []

    def observe_inside_the_open_transaction(**kwargs: Any):
        # `commit_durable_turn` creates the conversation inside its own
        # `BEGIN IMMEDIATE`, so this runs with the durable turn half-written.
        snapshot = controller.prompt_queue_registry.snapshot("session-1")
        activity = controller.activity_for("session-1")
        admitted = controller.queue_prompt(
            "session-1",
            text="follow-up offered mid-commit",
            expected_revision=snapshot.revision,
        )
        observed.append(
            (
                db.get_connection().in_transaction,
                activity.accepted_live_turn,
                store.dispatch_recovery_blocks_submission("session-1"),
                admitted.status,
                snapshot.total_count,
            )
        )
        return create_conversation(**kwargs)

    monkeypatch.setattr(
        persistence, "create_conversation", observe_inside_the_open_transaction
    )

    result = await controller.run_prompt_chain("first", session_id="session-1")

    assert result.accepted is True
    # The hook fired exactly once, genuinely inside the commit transaction.
    assert len(observed) == 1
    in_transaction, accepted_live, blocks, status, queued_before = observed[0]
    assert in_transaction is True
    # Nothing is accepted yet, so there is no chain and no owner at all --
    # admission cannot happen, and there is nothing to block submission with.
    assert accepted_live is False
    assert blocks is False
    assert status is QueueMutationStatus.REROUTE_NORMAL_SEND
    assert queued_before == 0
    # The rejected mid-commit draft left no entry behind.
    assert controller.prompt_queue_registry.snapshot("session-1").total_count == 0
    assert gateway.calls == 1
    assert _checkpoint_rows(db) == 0
    assert _message_rows(db) == 2


@pytest.mark.asyncio
async def test_healthy_live_owner_admits_a_queued_follow_up_drained_strictly_after(
    tmp_path: Path,
) -> None:
    """The ADR-046 queue works, and never holds two durable owners at once."""

    db, store, controller, _gateway = _controller(tmp_path)
    gateway = _HoldingGateway(db)
    controller.provider_gateway = gateway

    task = asyncio.create_task(
        controller.run_prompt_chain("first", session_id="session-1")
    )
    await asyncio.wait_for(gateway.started.wait(), timeout=5)

    # A healthy in-flight durable owner exists...
    recovery = store.dispatch_recovery_for_session("session-1")
    assert recovery is not None
    assert recovery.runtime_active is True
    assert recovery.recovery_needed is False
    # ...and, per the TASK-22000 owner decision, does NOT block submission or
    # surface as a recovery card.
    assert store.dispatch_recovery_blocks_submission("session-1") is False
    assert store.dispatch_recovery_for_presentation("session-1") is None
    # A *manual* second turn is still refused -- by the live run, not by the
    # recovery owner. The distinction is the whole point: the queue is the
    # affordance for a follow-up, and it is now reachable.
    refusal = controller.send_refusal_copy("session-1")
    assert refusal
    assert "pending response" not in refusal.lower()
    assert _checkpoint_rows(db) == 1

    snapshot = controller.prompt_queue_registry.snapshot("session-1")
    admitted = controller.queue_prompt(
        "session-1", text="second", expected_revision=snapshot.revision
    )
    assert admitted.status is QueueMutationStatus.APPLIED
    # Admission is a memory-only registry write: the follow-up has NOT been
    # submitted, and the first turn still owns the only checkpoint row.
    assert gateway.calls == 1
    assert _checkpoint_rows(db) == 1

    gateway.hold = False
    gateway.release.set()
    result = await asyncio.wait_for(task, timeout=10)

    assert result.accepted is True
    assert gateway.calls == 2
    assert gateway.prompts == ["first", "second"]
    # The decisive evidence: at each provider entry exactly one durable owner
    # existed, so the queued turn never overlapped the first turn's commit.
    assert gateway.checkpoint_counts == [1, 1]
    assert _checkpoint_rows(db) == 0
    assert _message_rows(db) == 4
    users = [
        message.content
        for message in store.messages_for_session("session-1")
        if message.role is ConsoleMessageRole.USER
    ]
    assert users == ["first", "second"]
    assert controller.prompt_queue_registry.snapshot("session-1").total_count == 0


@pytest.mark.asyncio
async def test_unhealthy_recovery_owner_still_blocks_submission_and_a_queued_turn(
    tmp_path: Path,
) -> None:
    """The block TASK-19900.3 actually needed survives the narrowing."""

    db, conversation_id, repository = _database(tmp_path / "unhealthy.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _HoldingGateway(db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )

    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert recovery.recovery_needed is True
    assert store.dispatch_recovery_blocks_submission(session_id) is True
    assert store.dispatch_recovery_for_presentation(session_id) is recovery
    assert "pending response" in (
        controller.send_refusal_copy(session_id) or ""
    ).lower()

    manual = await controller.submit_draft("manual retry", session_id=session_id)
    assert manual.accepted is False
    assert "pending response" in manual.visible_copy.lower()

    # And the same refusal reaches a genuinely coordinator-authorized QUEUED
    # send, driven through the real drain rather than a hand-built origin.
    coordinator = controller.prompt_queue_coordinator
    registry = coordinator.registry
    begun = registry.begin_chain(session_id, context_epoch=0, expected_revision=0)
    entry = registry.admit(
        session_id,
        text="queued retry",
        expected_revision=begun.snapshot.revision,
    )
    assert entry.entry_id is not None
    coordinator._chains[session_id] = _PromptChain()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id=session_id
    )
    queued_results: list[Any] = []
    submit_queued = coordinator._submit_queued

    async def capture(text: str, **kwargs: Any):
        result = await submit_queued(text, **kwargs)
        queued_results.append(result)
        return result

    coordinator._submit_queued = capture

    await coordinator._drain_waiting(session_id, ConsoleRunStatus.COMPLETED)

    assert len(queued_results) == 1
    assert queued_results[0].accepted is False
    assert "pending response" in queued_results[0].visible_copy.lower()
    snapshot = registry.snapshot(session_id)
    assert [item.entry_id for item in snapshot.entries] == [entry.entry_id]
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.pause_reason is PromptQueuePauseReason.DISPATCH_REFUSED

    assert gateway.calls == 0
    assert _checkpoint_rows(db) == 1
    assert _message_rows(db) == 2


@pytest.mark.asyncio
async def test_follow_up_admitted_mid_run_is_refused_when_the_owner_turns_unhealthy(
    tmp_path: Path,
) -> None:
    """A drain that meets a failed settlement refuses visibly and keeps the entry."""

    db, store, controller, _gateway = _controller(tmp_path)
    gateway = _HoldingGateway(db)
    controller.provider_gateway = gateway

    task = asyncio.create_task(
        controller.run_prompt_chain("first", session_id="session-1")
    )
    await asyncio.wait_for(gateway.started.wait(), timeout=5)

    snapshot = controller.prompt_queue_registry.snapshot("session-1")
    admitted = controller.queue_prompt(
        "session-1", text="second", expected_revision=snapshot.revision
    )
    assert admitted.status is QueueMutationStatus.APPLIED

    # Break terminal settlement AFTER the follow-up is already queued, so the
    # first turn's owner is left genuinely unhealthy with work behind it.
    db.get_connection().execute(
        "CREATE TRIGGER task22000_block_settlement "
        "BEFORE DELETE ON console_dispatch_checkpoints "
        "BEGIN SELECT RAISE(ABORT, 'task22000 settlement failure'); END"
    )
    db.get_connection().commit()

    gateway.hold = False
    gateway.release.set()
    await asyncio.wait_for(task, timeout=10)

    recovery = store.dispatch_recovery_for_session("session-1")
    assert recovery is not None
    assert recovery.recovery_needed is True
    assert store.dispatch_recovery_blocks_submission("session-1") is True
    # The queued follow-up never reached the provider, and it is still queued.
    assert gateway.calls == 1
    queue = controller.prompt_queue_registry.snapshot("session-1")
    assert queue.total_count == 1
    assert queue.mode is PromptQueueMode.PAUSED
    assert queue.pause_reason is not PromptQueuePauseReason.MANUAL
    assert _checkpoint_rows(db) == 1

    # Forcing the drain anyway is refused by the narrowed gate, visibly, and
    # the entry is returned to the head rather than consumed.
    await controller.prompt_queue_coordinator.resume_and_drain("session-1")

    assert gateway.calls == 1
    queue = controller.prompt_queue_registry.snapshot("session-1")
    assert queue.total_count == 1
    assert queue.mode is PromptQueueMode.PAUSED
    assert queue.pause_reason is PromptQueuePauseReason.DISPATCH_REFUSED
    assert _checkpoint_rows(db) == 1
