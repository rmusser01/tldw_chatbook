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
import threading
from pathlib import Path
import threading
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
from tldw_chatbook.Tools.raw_cli_executor import RawCliResult
from tldw_chatbook.UI.Console_Modules.raw_cli import ConsoleRawCliController
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash


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


class _ImmediateRawRuntime:
    permitted = True
    armed = True

    def __init__(self) -> None:
        self.started = threading.Event()
        self.requests: list[Any] = []

    def execute(self, request: Any, _on_event: Any) -> RawCliResult:
        self.requests.append(request)
        self.started.set()
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell="bash",
            initial_directory=request.initial_directory,
            elapsed_seconds=0.01,
            stdout_preview="raw\n",
            stderr_preview="",
            record_output="raw\n",
            exit_code=0,
            terminal_state="exited",
            truncated=False,
            cleanup_proven=True,
        )


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

    commit_is_open = threading.Event()
    draft_was_offered = threading.Event()
    # Sampled on the COMMIT's thread; the offer is made on the event loop.
    transaction_open: list[bool] = []
    offered_inside_the_window: list[bool] = []
    observed: list[tuple[bool, bool, QueueMutationStatus, int]] = []

    def hold_the_open_transaction(**kwargs: Any):
        # Worker thread. `commit_durable_turn` creates the conversation
        # inside its own `BEGIN IMMEDIATE`, so the durable turn is
        # half-written here -- and since TASK-22205 (#2091) that whole
        # transaction runs under `asyncio.to_thread`, off the event loop.
        # `get_connection()` is thread-local, so the open transaction can
        # only be observed HERE; the loop thread holds a different
        # connection that is not in a transaction at all.
        transaction_open.append(db.get_connection().in_transaction)
        commit_is_open.set()
        # Hold the write transaction open across the whole offer. Bounded,
        # so a broken observer fails an assertion instead of hanging the
        # run; whether the offer landed inside the window is recorded
        # rather than assumed.
        offered_inside_the_window.append(draft_was_offered.wait(timeout=10))
        transaction_open.append(db.get_connection().in_transaction)
        return create_conversation(**kwargs)

    monkeypatch.setattr(
        persistence, "create_conversation", hold_the_open_transaction
    )

    async def offer_a_draft_while_the_commit_is_open() -> None:
        """Press Send from the thread a real user's Send runs on.

        Before #2091 the commit occupied the event loop, so the only way to
        interleave with it was to reach into the queue from inside the
        transaction -- which happened to be the same thread. That is now
        both unfaithful and impossible: the queue registry asserts its
        owner thread (`_assert_owner_thread`, added 2026-08-23), and no
        user submission ever originates on a persistence worker thread.
        Since #2091 the loop is FREE while the commit runs, so this
        interleaving is one a user can now actually produce -- the race got
        more reachable, not less.
        """
        try:
            if not await asyncio.to_thread(commit_is_open.wait, 10):
                return
            snapshot = controller.prompt_queue_registry.snapshot("session-1")
            activity = controller.activity_for("session-1")
            admitted = controller.queue_prompt(
                "session-1",
                text="follow-up offered mid-commit",
                expected_revision=snapshot.revision,
            )
            observed.append(
                (
                    activity.accepted_live_turn,
                    store.dispatch_recovery_blocks_submission("session-1"),
                    admitted.status,
                    snapshot.total_count,
                )
            )
        finally:
            draft_was_offered.set()

    observer = asyncio.create_task(offer_a_draft_while_the_commit_is_open())
    result = await controller.run_prompt_chain("first", session_id="session-1")
    # Deliberately longer than both inner waits combined. A tighter bound
    # cancels the observer mid-wait and reports `CancelledError` instead of
    # the assertion that names what actually went wrong -- which is how a
    # staging failure gets mistaken for a flake.
    await asyncio.wait_for(observer, timeout=30)

    assert result.accepted is True
    # POSITIVE proof that the race was actually staged, asserted before
    # anything about its outcome. Without these three, a turn refused early
    # -- for any unrelated reason -- leaves `observed` empty and every
    # "nothing bad happened" assertion below passes vacuously. That is not
    # hypothetical: between #2088 and #2091 this test went red for exactly
    # that shape, and a careless repair would have made it green and blind.
    assert transaction_open == [True, True], (
        "the commit transaction was not open across the whole offer"
    )
    assert offered_inside_the_window == [True], (
        "the draft was not offered before the commit hold timed out"
    )
    assert len(observed) == 1
    accepted_live, blocks, status, queued_before = observed[0]
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


@pytest.mark.asyncio
async def test_raw_cli_worker_runs_while_model_owner_is_active_without_queueing(
    tmp_path: Path,
) -> None:
    """A direct raw command never waits behind the provider prompt chain."""

    db, store, chat_controller, _gateway = _controller(tmp_path)
    gateway = _HoldingGateway(db)
    chat_controller.provider_gateway = gateway
    model_task = asyncio.create_task(
        chat_controller.run_prompt_chain("first", session_id="session-1")
    )
    await asyncio.wait_for(gateway.started.wait(), timeout=5)

    runtime = _ImmediateRawRuntime()
    worker_threads: list[threading.Thread] = []

    def start_worker(work: Any, **options: Any) -> threading.Thread:
        assert options["thread"] is True
        assert options["exclusive"] is False
        thread = threading.Thread(target=work, name=options["name"])
        worker_threads.append(thread)
        thread.start()
        return thread

    raw_controller = ConsoleRawCliController(
        raw_cli_runtime=lambda: runtime,
        active_session_id=lambda: "session-1",
        persisted_leaf_anchor=lambda _session_id: None,
        selected_local_root=lambda _session_id: tmp_path,
        private_scratch_root=lambda _session_id: tmp_path,
        restore_stash=lambda _session_id, _stash: None,
        append_local_error=lambda _session_id, _text: None,
        append_store_marker=lambda *args, **kwargs: None,
        update_store_marker=lambda *args, **kwargs: None,
        agent_runs_db=lambda: None,
        run_log_access=lambda: None,
        start_worker=start_worker,
        marshal_to_ui=lambda callback, *args: callback(*args),
    )
    stash = ConsoleDraftStash(
        segments=[],
        text="! printf raw",
        has_paste=False,
        raw_cli_prefix_typed=True,
    )

    assert raw_controller.start_user_command(stash) is True
    assert await asyncio.to_thread(runtime.started.wait, 5)

    assert gateway.calls == 1
    assert chat_controller.prompt_queue_registry.snapshot("session-1").total_count == 0
    assert all(
        message.usage is None
        for message in store.messages_for_session("session-1")
    )
    assert runtime.requests[0].command == "printf raw"

    gateway.hold = False
    gateway.release.set()
    await asyncio.wait_for(model_task, timeout=10)
    for thread in worker_threads:
        await asyncio.to_thread(thread.join, 5)
