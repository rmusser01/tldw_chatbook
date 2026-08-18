"""PR 3a-2 Task 5: auto-wake -- a finished background sub-agent re-invokes
its supervisor (spec §3 invariant 5, corrected 2026-08-11).

The gap, reproduced RED against unmodified production before this module
existed (probe archived in the task-5 ledger report, run twice, stable):

- ``test_probe_a_survivor_settle_wakes_the_supervisor`` -- "the survivor
  settled and NO wake turn ever reached the provider";
- ``test_probe_mount_claim_delivers_a_marked_conversations_result`` --
  "a marked conversation with a terminal, undelivered survivor run was
  claimed by NOTHING at mount".

Under test here:

1. the full chain (real bridge, real gated child, real drain): a
   survivor settles -> ONE wake turn reaches the provider carrying the
   child's ``agent_runs.result`` as a machine-labelled trailing
   user-role payload entry; the transcript gains a SYSTEM row with
   ``MessageMetadata.origin == "agent_wake"`` and NO new USER row; the
   composer hook is never invoked; the ``FLEET_UNSEEN`` mark clears
   through the named seam only after acceptance;
2. coalescing + exactly-once: one wake bundles every undelivered
   completion; a refused wake loses nothing and is retried; a child
   settling DURING a wake turn rides the NEXT wake, and the mark
   survives until nothing undelivered remains;
3. scheduling: the manual-send gate (busy session, queue ownership, the
   ``max_parallel_runs`` cap) defers a wake, terminal transitions retry
   it, user-wins-ties defers behind a composer draft, and a conversation
   with no open session stays staged on the durable mark;
4. the mount-claim: ``seed_from_marks`` reconstructs the undelivered set
   from ``agent_runs`` + the mark's stable ``created_at``, excluding
   within-turn children and previously-delivered survivors;
5. authority: ``submit_draft(origin=AGENT_WAKE)`` is unreachable without
   the coordinator-issued token, and OFF (``[agents] autowake_enabled``)
   silences the wake at both fire points without losing anything.
"""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Chat.test_child_run_scope_ordering import _survivor_bridge
from Tests.Chat.test_console_agent_bridge import (
    _fence,
    _join_fleet_threads,
    _run,
)
from Tests.Chat.test_fleet_attention import _AppStub
from tldw_chatbook.Chat import console_fleet_wake
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    FleetDrained,
    SettledChild,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_fleet_attention import register_fleet_attention
from tldw_chatbook.Chat.console_fleet_wake import (
    WAKE_NOTICE_DISCLAIMER,
    WAKE_NOTICE_HEADER,
    WAKE_NOTICE_TRAILER,
    AgentWakeAuthorization,
    ConsoleFleetWakeCoordinator,
    compose_wake_notice,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


async def _settle(predicate, seconds: float = 5.0) -> bool:
    """Yield the loop until ``predicate()`` is true (delivery hops arrive
    via ``call_soon_threadsafe`` and need the loop to run)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


async def _quiet(predicate, seconds: float = 0.4) -> bool:
    """Give the loop a real window and assert ``predicate()`` NEVER fired.

    The no-op direction of ``_settle``: returns True only if the predicate
    stayed false for the whole window."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return False
        await asyncio.sleep(0.02)
    return True


class _RecordingWakeGateway:
    """Plain-path provider double recording every payload it streams --
    the seam a wake turn's model payload must arrive at."""

    def __init__(self, reply: str = "wake reply"):
        self.payloads: list[list[dict]] = []
        self.reply = reply
        #: When False, `resolve_for_send` reports not-ready (the refusal
        #: shape for the exactly-once tests); flip back to retry.
        self.ready = True
        #: Optional gate the stream waits on before yielding (slot tests).
        self.stream_gate: asyncio.Event | None = None
        #: Optional hook invoked once per stream, before yielding.
        self.on_stream: object | None = None

    async def resolve_for_send(self, selection):
        return SimpleNamespace(
            ready=self.ready,
            provider="llama_cpp",
            model="test-model",
            base_url=None,
            visible_copy="" if self.ready else "WIP: provider warming up",
        )

    async def stream_chat(self, resolution, messages, **kwargs):
        self.payloads.append([dict(m) for m in messages])
        hook = self.on_stream
        if hook is not None:
            hook()
        if self.stream_gate is not None:
            await self.stream_gate.wait()
        yield self.reply


class _FakeWakeBridge:
    """Registration + runs-db seams only -- no real fleet. Used by the
    controller-level tests that deliver drain events directly; the
    full-path tests use a real ``ConsoleAgentBridge``."""

    def __init__(self, runs_db):
        self.registered: dict[str, object] = {}
        self._runs_db = runs_db

    def on_fleet_drained(self, name, consumer):
        self.registered[name] = consumer

    def has_unsettled_children(self, conversation_id):
        return False

    @property
    def runs_db(self):
        return self._runs_db


def _drain(conversation_id, *children):
    return FleetDrained(conversation_id=conversation_id, children=tuple(children))


def _survivor(run_id, *, status="done", session_id="s-1", aid="aid-1"):
    return SettledChild(
        run_id=run_id,
        status=status,
        session_id=session_id,
        assistant_message_id=aid,
        settled_after_turn=True,
    )


def _within_turn(run_id, **kwargs):
    child = _survivor(run_id, **kwargs)
    return SettledChild(
        run_id=child.run_id,
        status=child.status,
        session_id=child.session_id,
        assistant_message_id=child.assistant_message_id,
        settled_after_turn=False,
    )


def _terminal_subagent_run(
    runs_db, conversation_id, *, parent_id=None, result="child answer", status="done"
):
    """A terminal subagent row settled AFTER its (terminal) parent."""
    if parent_id is None:
        parent_id = runs_db.create_run(
            conversation_id=conversation_id, agent_kind="primary"
        )
        runs_db.set_status(parent_id, "done", "turn final")
    run_id = runs_db.create_run(
        conversation_id=conversation_id,
        agent_kind="subagent",
        task="long job",
        parent_run_id=parent_id,
    )
    runs_db.set_status(run_id, status, result)
    return parent_id, run_id


def _controller_rig(tmp_path, *, session_title="Research"):
    """Controller + fake bridge + real runs DB + real marks DB + app stub."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    app = _AppStub(chacha)
    runs_db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session(title=session_title)
    gateway = _RecordingWakeGateway()
    bridge = _FakeWakeBridge(runs_db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=False,
    )
    controller.fleet_wake.wire(app=app)
    return chacha, app, runs_db, store, session, gateway, bridge, controller


# ---------------------------------------------------------------------------
# 1. The full chain (real bridge, real gated child) -- the headline red.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_survivor_settle_wakes_the_supervisor_with_a_machine_notice(
    tmp_path,
):
    """The reproduced gap, green: turn returns while the child runs; the
    child settles; ONE wake turn reaches the provider with the child's
    result as a machine-labelled trailing user-role entry; the transcript
    gains a machine-origin SYSTEM row and NO new USER row; the composer
    hook never fires; the mark clears only after acceptance."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        app = _AppStub(chacha)
        gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
            tmp_path,
            parent_script=[
                [_fence("spawn_subagent", {"task": "long job"})],
                ["turn final"],
            ],
            needed=1,
        )
        register_fleet_attention(bridge, app)
        wake_gateway = _RecordingWakeGateway()
        controller = ConsoleChatController(
            store=store,
            provider_gateway=wake_gateway,
            agent_bridge=bridge,
            agent_runtime_enabled=False,
        )
        controller.fleet_wake.wire(app=app)
        accepted_hook_calls: list[str] = []
        controller.on_submission_accepted = lambda: accepted_hook_calls.append(
            "composer-clear"
        )
        user_rows_before = len(
            [
                m
                for m in store.messages_for_session(session.id)
                if m.role is ConsoleMessageRole.USER
            ]
        )
        try:
            outcome = _run(bridge, store, session, aid, conversation_id=session.id)
            assert outcome.status == "done"
            assert gateway.entered_event.wait(5), "the child never started"
        finally:
            gate.set()
        _join_fleet_threads()

        woke = await _settle(lambda: wake_gateway.payloads)
        assert woke, (
            "the survivor settled and NO wake turn ever reached the provider"
        )
        assert len(wake_gateway.payloads) == 1

        # The model payload: the notice is the TRAILING user-role entry,
        # fully labelled, carrying the child's agent_runs.result.
        tail = wake_gateway.payloads[0][-1]
        assert tail["role"] == "user"
        assert WAKE_NOTICE_HEADER in tail["content"]
        assert "not user input" in tail["content"]
        assert "not approval" in tail["content"]
        assert "child answer" in tail["content"]

        # The transcript: NO new USER row; one SYSTEM row, machine-origin,
        # BEFORE the wake's own assistant reply.
        messages = store.messages_for_session(session.id)
        user_rows_after = [
            m for m in messages if m.role is ConsoleMessageRole.USER
        ]
        assert len(user_rows_after) == user_rows_before, (
            "a wake must never write a USER transcript row"
        )
        notice_rows = [
            m
            for m in messages
            if m.role is ConsoleMessageRole.SYSTEM
            and getattr(m.metadata, "origin", "") == "agent_wake"
        ]
        assert len(notice_rows) == 1
        assert WAKE_NOTICE_HEADER in notice_rows[0].content
        notice_index = messages.index(notice_rows[0])
        reply_rows = [m for m in messages if m.content == "wake reply"]
        assert reply_rows and messages.index(reply_rows[0]) > notice_index, (
            "the wake reply must follow its own notice row"
        )

        # Never the composer's business.
        assert accepted_hook_calls == []

        # Delivered: mark cleared through the named seam, pending drained,
        # and the durable per-run ledger stamped (exactly-once survives a
        # restart because THIS row, not memory, is the delivered bit).
        assert not app.conversation_local_marks_service.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        assert not controller.fleet_wake.has_pending(session.id)
        stamped = [
            run
            for run in db.list_runs(session.id, agent_kind="subagent")
            if run.get("wake_delivered_at")
        ]
        assert len(stamped) == 1, (
            "the accepted wake must stamp its delivered run in the ledger"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_children_finishing_inside_their_turn_never_wake(tmp_path):
    """A within-turn child is the turn's own news. Survivor-proofing (the
    Task 3 M10 lesson): the same live coordinator then receives a REAL
    after-turn drain and does fire -- so the no-op was a decision, not a
    dead consumer."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _within_turn(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "a within-turn child must not wake anyone"
        )
        assert not wake.has_pending(session.id)

        # A run-id-less survivor (child died before create_run: no row, no
        # result, nothing to wake on -- Task 2's consumer contract).
        wake.on_fleet_drained(
            _drain(session.id, _survivor(None, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "a child with no run row has nothing deliverable"
        )
        assert not wake.has_pending(session.id)

        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: gateway.payloads), (
            "the SAME coordinator must fire for a real survivor drain"
        )
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# 2. Coalescing + exactly-once.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_wake_bundles_every_undelivered_completion(tmp_path):
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        parent_id, run_a = _terminal_subagent_run(
            runs_db, session.id, result="answer alpha"
        )
        _, run_b = _terminal_subagent_run(
            runs_db,
            session.id,
            parent_id=parent_id,
            result="answer beta",
            status="error",
        )
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(
                session.id,
                _survivor(run_a, session_id=session.id),
                _survivor(run_b, status="error", session_id=session.id),
            )
        )
        assert await _settle(lambda: gateway.payloads)
        assert len(gateway.payloads) == 1, "N completions coalesce into ONE wake"
        notice = gateway.payloads[0][-1]["content"]
        assert "2 background sub-agents finished" in notice
        assert "answer alpha" in notice
        assert "answer beta" in notice
        assert "error" in notice, "a failed child's status is delivered honestly"
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_redelivered_drain_cannot_double_deliver(tmp_path):
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        wake = controller.fleet_wake
        event = _drain(session.id, _survivor(run_id, session_id=session.id))
        wake.on_fleet_drained(event)
        wake.on_fleet_drained(event)  # re-delivered (idempotence contract)
        assert await _settle(lambda: gateway.payloads)
        await _quiet(lambda: len(gateway.payloads) > 1)
        assert len(gateway.payloads) == 1
        assert gateway.payloads[0][-1]["content"].count("child answer") == 1
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_refused_wake_loses_nothing_and_is_retried(tmp_path):
    """Refusal direction of exactly-once: the provider blocks the first
    attempt -> no notice row, pending + mark retained; the retry
    delivers, and only THEN does the mark clear."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        app.conversation_local_marks_service.set_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        gateway.ready = False
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "a not-ready provider must refuse the wake before any stream"
        )
        assert wake.has_pending(session.id), "a refused wake keeps its pending bit"
        assert not runs_db.get_run(run_id).get("wake_delivered_at"), (
            "a refused wake must never stamp the durable delivered ledger"
        )
        assert app.conversation_local_marks_service.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "a refused wake must not clear the durable mark"
        notice_rows = [
            m
            for m in store.messages_for_session(session.id)
            if getattr(m.metadata, "origin", "") == "agent_wake"
        ]
        assert notice_rows == [], "a refused wake leaves no orphaned notice row"

        gateway.ready = True
        wake.retry_soon()
        assert await _settle(lambda: gateway.payloads), "the retry never delivered"
        assert await _settle(lambda: not wake.has_pending(session.id))
        assert not app.conversation_local_marks_service.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "delivery must clear the mark through the named seam"
        assert runs_db.get_run(run_id).get("wake_delivered_at"), (
            "the retried delivery must stamp the ledger"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_children_settling_during_a_wake_turn_ride_the_next_wake(tmp_path):
    """No double-delivery ACROSS the wake boundary: a child settling while
    the wake turn streams joins the NEXT wake, and the mark -- re-written
    by the attention consumer for that new settle -- survives the FIRST
    wake's delivery commit (clear-only-when-nothing-undelivered)."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        parent_id, run_a = _terminal_subagent_run(
            runs_db, session.id, result="first answer"
        )
        _, run_b = _terminal_subagent_run(
            runs_db, session.id, parent_id=parent_id, result="second answer"
        )
        marks = app.conversation_local_marks_service
        marks.set_mark(session.id, ConversationLocalMarksService.FLEET_UNSEEN)
        wake = controller.fleet_wake
        #: has_mark observed the instant wake #2's stream begins -- i.e.
        #: AFTER wake #1's delivery commit, BEFORE #2 delivers. The mark
        #: must still be set there: run_b is undelivered, and a cleared
        #: badge over an undelivered completion is the exact lie the
        #: clear-only-when-nothing-undelivered guard exists to prevent
        #: (first-attempt survivor M4 of the mutation round: the
        #: final-state asserts below could not see this window).
        mark_between_wakes: list[bool] = []

        def mid_stream_settle():
            if len(gateway.payloads) == 1:
                # The second survivor settles while wake #1 streams: the
                # attention consumer would re-write the mark on its thread;
                # mirror both halves here.
                marks.set_mark(
                    session.id, ConversationLocalMarksService.FLEET_UNSEEN
                )
                wake.on_fleet_drained(
                    _drain(session.id, _survivor(run_b, session_id=session.id))
                )
            elif len(gateway.payloads) == 2:
                mark_between_wakes.append(
                    marks.has_mark(
                        session.id, ConversationLocalMarksService.FLEET_UNSEEN
                    )
                )

        gateway.on_stream = mid_stream_settle
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_a, session_id=session.id))
        )
        assert await _settle(lambda: len(gateway.payloads) >= 2), (
            "the mid-turn settle never got its own follow-up wake"
        )
        assert len(gateway.payloads) == 2
        first = gateway.payloads[0][-1]["content"]
        second = gateway.payloads[1][-1]["content"]
        assert "first answer" in first and "second answer" not in first
        assert "second answer" in second and "first answer" not in second, (
            "wake #2 must carry ONLY what wake #1 had not delivered"
        )
        assert mark_between_wakes == [True], (
            "between the wakes the mark must still point at the "
            "undelivered second completion -- wake #1's commit may clear "
            "it only when NOTHING undelivered remains"
        )
        # After everything delivered: mark gone, pending gone.
        assert await _settle(lambda: not wake.has_pending(session.id))
        assert not marks.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# 3. Scheduling: gates, retries, ties, staging.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_busy_session_defers_the_wake_until_its_terminal_transition(
    tmp_path,
):
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "manual turn running"),
            session_id=session.id,
        )
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "a wake must never fire into a session whose run is in flight"
        )
        assert wake.has_pending(session.id)

        # The manual run ends: the terminal transition ITSELF is the retry
        # trigger (the production hook in _set_run_state, not a test poke).
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"),
            session_id=session.id,
        )
        assert await _settle(lambda: gateway.payloads), (
            "the terminal transition never retried the deferred wake"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_the_global_cap_defers_a_wake_like_any_other_send(tmp_path):
    """max_parallel_runs applies to a wake exactly as to a manual send
    (spec: a wake turn is a normal turn under every cap)."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        busy_sessions = [store.create_session(title=f"busy {i}") for i in range(3)]
        for busy in busy_sessions:
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING, "busy"),
                session_id=busy.id,
            )
        assert controller.send_refusal_copy(session.id) is not None, (
            "harness must actually saturate the cap"
        )
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "a wake must wait for a cap slot like any send"
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"),
            session_id=busy_sessions[0].id,
        )
        assert await _settle(lambda: gateway.payloads), (
            "freeing a cap slot never retried the wake"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_queue_owned_session_defers_the_wake(tmp_path):
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        # Instance-attr replacement of the real coordinator's gate: queued
        # work owns the next generation for this session.
        controller.prompt_queue_coordinator.controls_generation = (
            lambda sid: sid == session.id
        )
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "queued messages own the next turn; a wake must not steal it"
        )
        assert wake.has_pending(session.id)

        controller.prompt_queue_coordinator.controls_generation = lambda sid: False
        # Chain end fires the production retry hook.
        controller._publish_queue_chain_terminal(
            session.id, ConsoleRunStatus.COMPLETED
        )
        assert await _settle(lambda: gateway.payloads), (
            "queue-chain end never retried the deferred wake"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_user_wins_ties_a_composer_draft_defers_the_wake(tmp_path):
    """The pinned tie-break: while the user-priority probe reports a claim
    (screen wiring: a non-empty composer draft -- which also covers the
    dispatch gap, since the composer clears only on ACCEPTED sends), a due
    wake defers; a RAISING probe defers too (user wins on uncertainty);
    the claim ending plus any retry trigger delivers."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        draft_present = True
        probed: list[str] = []

        def probe(session_id: str) -> bool:
            probed.append(session_id)
            return draft_present

        controller.wake_user_priority_probe = probe
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "the user's pending draft must never lose its slot to a wake"
        )
        assert probed and probed[0] == session.id
        assert wake.has_pending(session.id)

        controller.wake_user_priority_probe = lambda sid: (_ for _ in ()).throw(
            RuntimeError("broken probe")
        )
        wake.retry_soon()
        assert await _quiet(lambda: gateway.payloads), (
            "a raising probe must defer -- user wins on uncertainty"
        )

        draft_present = False
        controller.wake_user_priority_probe = probe
        wake.retry_soon()
        assert await _settle(lambda: gateway.payloads), (
            "the cleared draft never let the wake through"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_no_open_session_means_the_mark_is_the_staged_wake(tmp_path):
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        foreign_conversation = "conv-not-open-anywhere"
        _parent, run_id = _terminal_subagent_run(runs_db, foreign_conversation)
        app.conversation_local_marks_service.set_mark(
            foreign_conversation, ConversationLocalMarksService.FLEET_UNSEEN
        )
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(foreign_conversation, _survivor(run_id, session_id="gone"))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "no open session -> nothing to submit into"
        )
        assert wake.has_pending(foreign_conversation)
        assert app.conversation_local_marks_service.has_mark(
            foreign_conversation, ConversationLocalMarksService.FLEET_UNSEEN
        ), "the durable mark stays the staged wake"
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_wake_delivery_is_serialized_one_conversation_at_a_time(tmp_path):
    """Two conversations owed wakes deliver sequentially: while wake #1
    streams, conversation #2 waits; #1 finishing chains #2."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        second = store.create_session(title="Second")
        _parent_a, run_a = _terminal_subagent_run(
            runs_db, session.id, result="alpha result"
        )
        _parent_b, run_b = _terminal_subagent_run(
            runs_db, second.id, result="beta result"
        )
        gate = asyncio.Event()
        gateway.stream_gate = gate
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_a, session_id=session.id))
        )
        wake.on_fleet_drained(
            _drain(second.id, _survivor(run_b, session_id=second.id))
        )
        assert await _settle(lambda: len(gateway.payloads) == 1)
        assert await _quiet(lambda: len(gateway.payloads) > 1, seconds=0.3), (
            "the second conversation's wake must wait for the first"
        )
        gateway.stream_gate = None
        gate.set()
        assert await _settle(lambda: len(gateway.payloads) == 2), (
            "finishing wake #1 must chain wake #2"
        )
        contents = [p[-1]["content"] for p in gateway.payloads]
        assert any("alpha result" in c for c in contents)
        assert any("beta result" in c for c in contents)
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_wake_turn_occupies_the_sessions_send_slot(tmp_path):
    """A woken turn is a normal turn under the caps: while it streams, the
    session refuses a manual send exactly as any in-flight run does."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        gate = asyncio.Event()
        gateway.stream_gate = gate
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: gateway.payloads)
        assert controller.send_refusal_copy(session.id) is not None, (
            "a streaming wake turn must hold the session's slot"
        )
        assert controller.run_state_for(session.id).status is (
            ConsoleRunStatus.STREAMING
        )
        gate.set()
        assert await _settle(
            lambda: controller.run_state_for(session.id).is_send_allowed
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_one_conversations_failed_delivery_does_not_strand_anothers(
    tmp_path,
):
    """The chaining half `_deliver`'s own trailing retry owns (mutation
    M23's first run SURVIVED without this): a delivery that RAISES lands
    no terminal run-state transition -- the session sticks at VALIDATING
    -- so nothing else would ever re-attempt the OTHER conversation that
    deferred behind the serialization gate."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        second = store.create_session(title="Second")
        _pa, run_a = _terminal_subagent_run(runs_db, session.id, result="alpha")
        _pb, run_b = _terminal_subagent_run(runs_db, second.id, result="beta")
        real_resolve = gateway.resolve_for_send
        blew_up = []

        async def exploding_resolve(selection):
            if not blew_up:
                blew_up.append(True)
                raise RuntimeError("probe blew up mid-wake")
            return await real_resolve(selection)

        gateway.resolve_for_send = exploding_resolve
        wake = controller.fleet_wake
        # Both pending before the first delivery task runs: attempt #2
        # defers behind the serialization gate, so only the failed
        # delivery's own completion can ever revive it.
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_a, session_id=session.id))
        )
        wake.on_fleet_drained(
            _drain(second.id, _survivor(run_b, session_id=second.id))
        )
        assert await _settle(lambda: gateway.payloads), (
            "the surviving conversation's wake never fired after the "
            "other's delivery raised"
        )
        assert any(
            "beta" in p[-1]["content"] for p in gateway.payloads
        ), "the delivered wake must be the SECOND conversation's"
        # The failed conversation keeps its pending bit for a later retry.
        assert wake.has_pending(session.id)
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# 4. Mount-claim: the durable mark IS the staged wake.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mount_claim_delivers_a_marked_conversations_result_from_the_db(
    tmp_path,
):
    """The second reproduced red, green: nothing in memory survived (the
    settle happened with Console closed); the mark names the conversation
    and the durable per-run ``wake_delivered_at`` ledger defines what is
    still owed. Within-turn children and survivors an earlier wake
    already stamped stay excluded."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        conversation_id = session.id
        parent_id = runs_db.create_run(
            conversation_id=conversation_id, agent_kind="primary"
        )
        # A within-turn child: terminal BEFORE its parent -> excluded.
        within_id = runs_db.create_run(
            conversation_id=conversation_id,
            agent_kind="subagent",
            task="quick job",
            parent_run_id=parent_id,
        )
        runs_db.set_status(within_id, "done", "the within-turn answer")
        runs_db.set_status(parent_id, "done", "turn final")
        # A survivor an EARLIER wake already carried: a genuine survivor
        # by every timing rule -- ONLY its ledger stamp can exclude it,
        # which is exactly the pin (a timestamp rule against the mark
        # could never recover this, per the ledger commit's own record).
        old_id = runs_db.create_run(
            conversation_id=conversation_id,
            agent_kind="subagent",
            task="old job",
            parent_run_id=parent_id,
        )
        runs_db.set_status(old_id, "done", "the already-delivered answer")
        assert runs_db.mark_wake_delivered([old_id]) == 1
        app.conversation_local_marks_service.set_mark(
            conversation_id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        # The undelivered survivor: terminal after the parent, at mark time.
        fresh_id = runs_db.create_run(
            conversation_id=conversation_id,
            agent_kind="subagent",
            task="long job",
            parent_run_id=parent_id,
        )
        runs_db.set_status(fresh_id, "done", "the staged child answer")

        wake = controller.fleet_wake
        assert wake.seed_from_marks() == 1
        wake.retry_soon()
        assert await _settle(lambda: gateway.payloads), (
            "the staged wake never fired at mount-claim"
        )
        notice = gateway.payloads[0][-1]["content"]
        assert "the staged child answer" in notice
        assert "the within-turn answer" not in notice, (
            "a within-turn child was already delivered by its own turn"
        )
        assert "the already-delivered answer" not in notice, (
            "a ledger-stamped survivor was already carried by an earlier wake"
        )
        assert await _settle(
            lambda: not app.conversation_local_marks_service.has_mark(
                conversation_id, ConversationLocalMarksService.FLEET_UNSEEN
            )
        ), "the claimed mark must clear after delivery"
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_pending_run_the_ledger_shows_delivered_is_dropped_not_reannounced(
    tmp_path,
):
    """The restart-race belt: the drain re-delivers (or a mount seeds) a
    run some earlier wake already stamped in the durable ledger. Compose
    drops it -- from the notice AND from the registry -- instead of
    announcing the same result twice."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        parent_id, delivered_id = _terminal_subagent_run(
            runs_db, session.id, result="already announced"
        )
        assert runs_db.mark_wake_delivered([delivered_id]) == 1
        _, fresh_id = _terminal_subagent_run(
            runs_db,
            session.id,
            parent_id=parent_id,
            result="genuinely new",
        )
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(
                session.id,
                _survivor(delivered_id, session_id=session.id),
                _survivor(fresh_id, session_id=session.id),
            )
        )
        assert await _settle(lambda: gateway.payloads)
        notice = gateway.payloads[0][-1]["content"]
        assert "genuinely new" in notice
        assert "already announced" not in notice, (
            "a ledger-stamped run must never be re-announced"
        )
        assert await _settle(lambda: not wake.has_pending(session.id)), (
            "the stale entry must leave the registry, not strand it"
        )
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# 5. Authority + the kill switch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_wake_origin_is_unreachable_without_the_coordinator_token(
    tmp_path,
):
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        with pytest.raises(PermissionError):
            await controller.submit_draft(
                "forged notice",
                session_id=session.id,
                origin=ConsoleSubmissionOrigin.AGENT_WAKE,
            )
        # A token minted by a FOREIGN coordinator is refused too.
        foreign = ConsoleFleetWakeCoordinator(controller)
        forged = AgentWakeAuthorization(
            foreign,
            session.id,
            _key=console_fleet_wake._WAKE_AUTHORIZATION_KEY,
        )
        with pytest.raises(PermissionError):
            await controller.submit_draft(
                "forged notice",
                session_id=session.id,
                origin=ConsoleSubmissionOrigin.AGENT_WAKE,
                wake_authorization=forged,
            )
        # And the coordinator's own token is live ONLY while it delivers.
        idle_token = AgentWakeAuthorization(
            controller.fleet_wake,
            session.id,
            _key=console_fleet_wake._WAKE_AUTHORIZATION_KEY,
        )
        assert not controller.fleet_wake.authorizes(idle_token, session.id)
        assert store.messages_for_session(session.id) == [], (
            "refused wake submissions must leave no transcript trace"
        )
    finally:
        chacha.close()


def test_wake_authority_key_is_module_private():
    with pytest.raises(PermissionError):
        AgentWakeAuthorization(object(), "s-1", _key=object())


@pytest.mark.asyncio
async def test_autowake_off_records_everything_and_fires_nothing(
    tmp_path, monkeypatch
):
    """OFF must not lose completions (the brief's own wording): the mark
    and the pending record still land; the wake simply never fires -- at
    the drain AND at the mount-claim -- and flipping the switch back on
    delivers what was recorded, breaking any wake chain only while OFF."""
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        app.conversation_local_marks_service.set_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "autowake_enabled=false must silence the immediate fire point"
        )
        assert wake.has_pending(session.id), "OFF still records the completion"
        assert app.conversation_local_marks_service.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "OFF keeps the indicator working"

        # The second fire point: mount-claim seeds nothing while OFF.
        assert wake.seed_from_marks() == 0

        monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
        wake.retry_soon()
        assert await _settle(lambda: gateway.payloads), (
            "flipping the switch back on must deliver what OFF recorded"
        )
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# 6. The notice text.
# ---------------------------------------------------------------------------


def test_compose_wake_notice_labels_fences_and_truncates():
    long_result = "x" * 5000
    fencey_result = "before ```python\nprint(1)\n``` after"
    notice = compose_wake_notice(
        [
            {
                "id": "run-1",
                "status": "done",
                "result": long_result,
                "agent_definition": "researcher",
                "task": "dig",
            },
            {"id": "run-2", "status": "cancelled", "result": fencey_result},
            {"id": "run-3", "status": "error", "result": None},
        ]
    )
    assert notice.startswith(WAKE_NOTICE_HEADER)
    assert WAKE_NOTICE_DISCLAIMER in notice
    assert notice.endswith(WAKE_NOTICE_TRAILER)
    assert "not user input" in notice and "not approval" in notice
    assert "[run-1] researcher — done — task: dig" in notice
    # Per-child truncation at the same constant the in-turn collection
    # uses (max_subagent_result_chars = 4000).
    assert "x" * 4000 in notice
    assert "x" * 4001 not in notice
    assert "truncated" in notice
    # A result carrying its own fences cannot break out of its block.
    assert "````\nbefore ```python" in notice
    # A resultless failure is delivered honestly, not blank.
    assert "(no result recorded; the run ended 'error')" in notice
    assert compose_wake_notice([]) == ""


def test_compose_wake_notice_splits_the_total_budget_evenly():
    """Five over-cap results must be shortened FAIRLY to the 16000-char
    total (wait_agents' own discipline), not cut mid-notice downstream.
    Bodies are 5000 chars each -- over BOTH the per-child cap and the
    even share -- so the split provably engages (the first version of
    this test used 3000-char bodies under every cap and could not kill a
    truncation-removed mutant)."""
    rows = [
        {"id": f"run-{i}", "status": "done", "result": ("r%d " % i) * 1250}
        for i in range(5)
    ]
    notice = compose_wake_notice(rows)
    from tldw_chatbook.Agents.agent_models import RunBudget

    budget = RunBudget()
    assert len(notice) <= budget.max_tool_result_chars + 1500, (
        "the combined notice must respect the total result budget "
        "(small constant slack for the wrapper text)"
    )
    assert notice.count("truncated to share") == 5, (
        "every over-share child is shortened, none silently dropped"
    )
    for i in range(5):
        assert f"[run-{i}]" in notice, "every child still appears"


def test_a_failed_delivery_task_never_wedges_the_delivering_flag(monkeypatch):
    """Qodo audit minor batch: `_attempt` set `_delivering`/`_delivering_
    session` BEFORE `loop.create_task(...)`. If create_task raises (the
    loop closing between the `is_closed()` check and the call is the live
    shape), the flags stayed set forever: every later `_attempt` for the
    whole process early-returned at `self._delivering is not None`, so no
    wake could ever fire again. The flags must still be set before the
    UI hook and the task (the poll-beat race the comment there pins), so
    the fix is clear-on-failure, not set-after.
    """
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
    session = SimpleNamespace(id="s-1", persisted_conversation_id="conv-1")
    controller = SimpleNamespace(
        _disposed=False,
        store=SimpleNamespace(sessions=lambda: [session]),
        send_refusal_copy=lambda session_id: None,
    )
    wake = ConsoleFleetWakeCoordinator(controller)

    class _RaisingLoop:
        def __init__(self):
            self.calls = 0

        def is_closed(self):
            return False

        def create_task(self, coro):
            self.calls += 1
            coro.close()
            raise RuntimeError("Event loop is closed")

    raising = _RaisingLoop()
    wake._loop = raising
    with wake._registry_lock:
        wake._pending["conv-1"] = {"run-1": "done"}

    wake._attempt("conv-1")
    assert raising.calls == 1, (
        "harness precondition: the attempt never reached create_task"
    )
    assert wake.delivering_conversation_id() is None, (
        "a failed create_task left `_delivering` set forever -- every "
        "future wake in the process is silently refused"
    )
    assert wake.delivering_session_id() is None

    # The wake is deferred, not lost: a later attempt on a healthy loop
    # schedules the delivery.
    class _FakeTask:
        def add_done_callback(self, cb):
            return None

    class _RecordingLoop:
        def __init__(self):
            self.tasks: list = []

        def is_closed(self):
            return False

        def create_task(self, coro):
            self.tasks.append(coro)
            coro.close()
            return _FakeTask()

    healthy = _RecordingLoop()
    wake._loop = healthy
    wake._attempt("conv-1")
    assert healthy.tasks, (
        "the pending wake was lost after the failed attempt"
    )
