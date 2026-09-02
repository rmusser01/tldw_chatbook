"""PR 3a-2 Task 4: the durable unseen-completion mark + app-wide toast.

A fleet SURVIVOR (a child that outlived its spawning turn, PR 3a-1)
settles on its own daemon thread, possibly after the Console screen died.
Before this task nothing durable or app-wide recorded that: the result sat
in ``agent_runs.result``, invisible unless the user happened to reopen the
right conversation and look at the rail.

Under test here:

1. the **survivor discriminator** -- ``SettledChild.settled_after_turn``,
   recorded AT SETTLE TIME from the bridge's in-flight-turn window
   (``_inflight_turn_message_ids``): a child settling while its turn's
   ``run_reply`` still executes is within-turn (False); one settling after
   is a survivor (True);
2. the **durable mark** (``ConversationLocalMarksService.FLEET_UNSEEN``)
   written from the drain consumer on the CHILD's thread -- restart-proof
   by construction, proven by reading it back through a FRESH service
   handle over a FRESH DB handle on the same file;
3. the **app-wide toast**: one per drain (N children = 1 toast), hopped to
   the loop captured at registration, honest about ``error``/``cancelled``;
4. the **deep link**: staged only when Console is NOT the active screen;
5. the **named clear seam** (``clear_fleet_unseen_completion``) Task 5's
   delivery and the view-clear both call.

Full-path tests drive a real ``ConsoleAgentBridge`` + gated fleet child
(the Task 1/3 harness); matrix tests deliver synthetic ``FleetDrained``
events to the consumer directly (the Task 3 pattern).
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from Tests.Chat.test_child_run_scope_ordering import _survivor_bridge
from Tests.Chat.test_console_agent_bridge import (
    _fence,
    _FleetTwoChildGateway,
    _join_fleet_threads,
    _run,
)
from tldw_chatbook.Chat.console_agent_bridge import FleetDrained, SettledChild
from tldw_chatbook.Chat.console_fleet_attention import (
    FLEET_UNSEEN_REVISION_ATTR,
    ConsoleFleetAttentionConsumer,
    clear_fleet_unseen_completion,
    fleet_completion_toast_copy,
    fleet_unseen_conversation_ids,
    register_fleet_attention,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


async def _settle(predicate, seconds: float = 5.0) -> bool:
    """Yield the loop until ``predicate()`` is true (the announce arrives
    via ``call_soon_threadsafe`` and needs the loop to run)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


class _AppStub:
    """The app-object surface the attention consumer touches, nothing more."""

    def __init__(self, chachanotes_db=None):
        self.chachanotes_db = chachanotes_db
        self.conversation_local_marks_service = (
            ConversationLocalMarksService(chachanotes_db)
            if chachanotes_db is not None
            else None
        )
        #: (thread name, message, severity) per notify call.
        self.notifications: list[tuple[str, str, str]] = []
        #: (channel, value) per stage call.
        self.staged: list[tuple[object, object]] = []
        self.screen = object()  # not a Console screen by default
        self.pending_handoffs = self._Handoffs(self.staged)

    class _Handoffs:
        def __init__(self, staged):
            self._staged = staged

        def stage(self, channel, value):
            self._staged.append((channel, value))
            return 1

    def notify(self, message, *, severity="information", **_kwargs):
        self.notifications.append(
            (threading.current_thread().name, message, severity)
        )


def _drain(conversation_id="conv-att", children=()):
    return FleetDrained(conversation_id=conversation_id, children=tuple(children))


def _child(status="done", *, after_turn=True, run_id="run-1", session_id="s-1"):
    return SettledChild(
        run_id=run_id,
        status=status,
        session_id=session_id,
        assistant_message_id="aid-1",
        settled_after_turn=after_turn,
    )


class _ParentWaitsForDrainGateway(_FleetTwoChildGateway):
    """The within-turn shape, deterministic: the parent's FINAL turn does
    not stream until the drain has already fired -- so the child provably
    settled while ``run_reply`` was still executing."""

    def __init__(self, *args, wait_before_final: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self._wait_before_final = wait_before_final

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX

        if not system.startswith(SUBAGENT_PROMPT_PREFIX) and len(self._parent) == 1:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._wait_before_final.wait)
        async for chunk in super().stream_chat(
            resolution, messages, tools=tools, **kwargs
        ):
            yield chunk


# ---------------------------------------------------------------------------
# Full path: a real bridge, a real gated child, the real drain.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_survivor_settle_writes_the_durable_mark_and_toasts_once(
    tmp_path,
):
    """The headline chain: the turn returns while the child still runs;
    the child settles; the drain's consumer writes the FLEET_UNSEEN mark
    (readable through a FRESH service handle over a FRESH DB handle -- the
    restart shape) and fires exactly one toast, on the captured loop's
    thread, naming the conversation."""
    chacha_path = str(tmp_path / "chacha.sqlite")
    chacha = CharactersRAGDB(chacha_path, client_id="t")
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
        events: list[FleetDrained] = []
        bridge.on_fleet_drained("test-recorder", events.append)
        register_fleet_attention(bridge, app)  # captures this test's loop
        conversation_id = "conv-att"
        try:
            outcome = _run(
                bridge, store, session, aid, conversation_id=conversation_id
            )
            assert outcome.status == "done"
            assert gateway.entered_event.wait(5), "the child never started"
            # The turn is over, the child is not: nothing may be marked or
            # toasted yet.
            assert app.notifications == []
            assert not app.conversation_local_marks_service.has_mark(
                conversation_id,
                ConversationLocalMarksService.FLEET_UNSEEN,
            )
        finally:
            gate.set()
        _join_fleet_threads()

        toasted = await _settle(lambda: app.notifications)
        assert toasted, "the survivor settled and no toast ever arrived"
        assert len(app.notifications) == 1
        thread_name, message, severity = app.notifications[0]
        assert thread_name == threading.current_thread().name, (
            "the toast must hop to the captured loop's thread, never fire "
            f"on the child's: {thread_name}"
        )
        assert message == "Background sub-agent finished in “an unsaved Console chat”."
        assert severity == "information"

        # The discriminator, on the real event: this child settled after
        # its turn returned.
        assert [c.settled_after_turn for e in events for c in e.children] == [True]

        # Restart-proof by construction: a FRESH DB handle + FRESH service
        # on the same file reads the mark back.
        fresh_db = CharactersRAGDB(chacha_path, client_id="verifier")
        try:
            fresh_service = ConversationLocalMarksService(fresh_db)
            assert fresh_service.has_mark(
                conversation_id, ConversationLocalMarksService.FLEET_UNSEEN
            ), "the durable mark must survive into a fresh handle"
        finally:
            fresh_db.close()
        assert getattr(app, FLEET_UNSEEN_REVISION_ATTR, 0) >= 1, (
            "the announce must bump the badge-cache revision"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_child_that_finishes_inside_its_turn_marks_and_toasts_nothing(
    tmp_path,
):
    """The other half of the discriminator, deterministic: the parent's
    final turn is gated on the drain itself, so the child provably settled
    while ``run_reply`` was still executing -- within-turn news the
    per-turn notify already covers. No mark, no toast."""
    chacha_path = str(tmp_path / "chacha.sqlite")
    chacha = CharactersRAGDB(chacha_path, client_id="t")
    try:
        app = _AppStub(chacha)
        gate = threading.Event()
        gate.set()  # the child runs immediately
        drain_fired = threading.Event()
        gateway = _ParentWaitsForDrainGateway(
            parent_script=[
                [_fence("spawn_subagent", {"task": "quick job"})],
                ["turn final"],
            ],
            child_result=["child answer"],
            gate=gate,
            needed=1,
            wait_before_final=drain_fired,
        )
        from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
        from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        store = ConsoleChatStore()
        session = store.ensure_session()
        store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=db, store=store, provider_gateway=gateway
        )
        events: list[FleetDrained] = []
        bridge.on_fleet_drained("test-recorder", events.append)
        bridge.on_fleet_drained("test-wait", lambda _e: drain_fired.set())
        register_fleet_attention(bridge, app)
        conversation_id = "conv-within"
        outcome = _run(
            bridge, store, session, assistant.id, conversation_id=conversation_id
        )
        assert outcome.status == "done"
        _join_fleet_threads()
        assert drain_fired.is_set(), "precondition: the drain fired mid-turn"
        assert [c.settled_after_turn for e in events for c in e.children] == [
            False
        ], "a child settling while its turn still runs is within-turn"

        await asyncio.sleep(0.2)  # give a wrong hop time to land
        assert app.notifications == [], (
            "a within-turn child must not toast -- the per-turn notify "
            "already covers its session"
        )
        assert not app.conversation_local_marks_service.has_mark(
            conversation_id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "a within-turn child must not write the unseen mark"
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_two_survivors_in_one_drain_coalesce_into_one_toast(tmp_path):
    """N children of one conversation settling together = one toast, not N."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        app = _AppStub(chacha)
        gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
            tmp_path,
            parent_script=[
                [_fence("spawn_subagent", {"task": "job A"})],
                [_fence("spawn_subagent", {"task": "job B"})],
                ["turn final"],
            ],
            needed=2,
        )
        register_fleet_attention(bridge, app)
        try:
            outcome = _run(bridge, store, session, aid, conversation_id="conv-two")
            assert outcome.status == "done"
            assert gateway.entered_event.wait(5)
        finally:
            gate.set()
        _join_fleet_threads()
        toasted = await _settle(lambda: app.notifications)
        assert toasted
        assert len(app.notifications) == 1, (
            f"two survivors must coalesce into ONE toast: {app.notifications}"
        )
        message = app.notifications[0][1]
        assert message == (
            "2 background sub-agents in “an unsaved Console chat”: 2 finished."
        )
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# Consumer matrix (direct events -- the Task 3 pattern).
# ---------------------------------------------------------------------------


def test_a_closed_loop_still_writes_the_mark_and_announces_inline(tmp_path):
    """The app-exit shape: the captured loop is closed; the durable mark
    must still land (child thread, DB only) and the announce runs inline
    as a last chance -- ``app.notify`` is documented thread-safe."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        app = _AppStub(chacha)
        closed_loop = asyncio.new_event_loop()
        closed_loop.close()
        consumer = ConsoleFleetAttentionConsumer(app, loop=closed_loop)
        worker = threading.Thread(
            target=consumer,
            args=(_drain(children=[_child()]),),
            name="fake-child-thread",
        )
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
        assert app.conversation_local_marks_service.has_mark(
            "conv-att", ConversationLocalMarksService.FLEET_UNSEEN
        ), "the mark write must not depend on the loop at all"
        assert [n[0] for n in app.notifications] == ["fake-child-thread"], (
            "with a dead loop the announce runs inline as a best effort"
        )
    finally:
        chacha.close()


def test_error_and_cancelled_children_are_reported_honestly(tmp_path):
    """Failure outcomes say so -- severity and verb both."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        app = _AppStub(chacha)
        consumer = ConsoleFleetAttentionConsumer(app, loop=None)
        consumer(_drain("conv-err", children=[_child("error")]))
        consumer(_drain("conv-cxl", children=[_child("cancelled")]))
        consumer(
            _drain(
                "conv-mix",
                children=[_child("done"), _child("error"), _child("cancelled")],
            )
        )
        assert [(n[1], n[2]) for n in app.notifications] == [
            ("Background sub-agent failed in “an unsaved Console chat”.", "error"),
            (
                "Background sub-agent was cancelled in “an unsaved Console chat”.",
                "warning",
            ),
            (
                "3 background sub-agents in “an unsaved Console chat”: "
                "1 finished, 1 failed, 1 cancelled.",
                "error",
            ),
        ]
        # Every failed conversation is marked too -- an errored survivor is
        # still unseen news.
        assert fleet_unseen_conversation_ids(app) == frozenset(
            {"conv-err", "conv-cxl", "conv-mix"}
        )
    finally:
        chacha.close()


def test_the_toast_names_the_persisted_conversations_title(tmp_path):
    """A persisted conversation's DB title is used, not the generic stub."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        conv_id = chacha.add_conversation(
            {"title": "Quarterly report", "character_id": 1}
        )
        app = _AppStub(chacha)
        consumer = ConsoleFleetAttentionConsumer(app, loop=None)
        consumer(_drain(conv_id, children=[_child()]))
        assert app.notifications[0][1] == (
            "Background sub-agent finished in “Quarterly report”."
        )
    finally:
        chacha.close()


def test_a_drain_with_no_after_turn_children_is_a_strict_no_op(tmp_path):
    """A drain that only carries within-turn children touches nothing."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        app = _AppStub(chacha)
        consumer = ConsoleFleetAttentionConsumer(app, loop=None)
        consumer(
            _drain(children=[_child(after_turn=False), _child("error", after_turn=False)])
        )
        assert app.notifications == []
        assert app.staged == []
        assert fleet_unseen_conversation_ids(app) == frozenset()
    finally:
        chacha.close()


def test_run_id_none_children_still_count(tmp_path):
    """A child that died before ``create_run`` has no row, but its settle
    is still unseen news -- the consumer must tolerate ``run_id=None``."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        app = _AppStub(chacha)
        consumer = ConsoleFleetAttentionConsumer(app, loop=None)
        consumer(_drain(children=[_child("error", run_id=None)]))
        assert len(app.notifications) == 1
        assert fleet_unseen_conversation_ids(app) == frozenset({"conv-att"})
    finally:
        chacha.close()


def test_incomplete_receipt_publication_falls_back_to_coarse_mark(tmp_path):
    """A partial durable publish cannot erase the compatibility signal."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        app = _AppStub(chacha)

        class _IncompleteReceipts:
            def publish_fleet_drain(self, _event):
                return type(
                    "Publication", (), {"activity_ids": (), "complete": False}
                )()

            def ensure_fleet_mark(self, conversation_id):
                app.conversation_local_marks_service.set_mark(
                    conversation_id,
                    ConversationLocalMarksService.FLEET_UNSEEN,
                )
                return True

        consumer = ConsoleFleetAttentionConsumer(
            app, loop=None, receipt_service=_IncompleteReceipts()
        )
        consumer(_drain(children=[_child("error")]))

        assert fleet_unseen_conversation_ids(app) == frozenset({"conv-att"})
        assert getattr(app, FLEET_UNSEEN_REVISION_ATTR) == 1
    finally:
        chacha.close()


def test_deep_link_is_staged_only_while_console_is_not_the_active_screen(
    tmp_path,
):
    """Console mounted -> the badge + toast suffice; switching the session
    under a user already in Console would be hostile. Not mounted -> stage
    the mount-claimable target."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleFleetCompletionTarget
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        app = _AppStub(chacha)
        consumer = ConsoleFleetAttentionConsumer(app, loop=None)
        consumer(_drain(children=[_child(session_id="sess-9")]))
        assert len(app.staged) == 1
        channel, value = app.staged[0]
        assert channel is HandoffChannel.CONSOLE_FLEET_COMPLETION
        assert isinstance(value, ConsoleFleetCompletionTarget)
        assert value.conversation_id == "conv-att"
        assert value.session_id == "sess-9"

        class _ConsoleishScreen:
            def _ensure_console_chat_controller(self):  # pragma: no cover - seam only
                return None

        app.screen = _ConsoleishScreen()
        consumer(_drain(children=[_child()]))
        assert len(app.staged) == 1, "no staging while Console is active"
    finally:
        chacha.close()


def test_clear_seam_clears_and_bumps_exactly_when_a_mark_existed(tmp_path):
    """The named clear function Task 5's delivery calls: True + revision
    bump when a mark was cleared, False (no bump) when nothing existed."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        app = _AppStub(chacha)
        app.conversation_local_marks_service.set_mark(
            "conv-att", ConversationLocalMarksService.FLEET_UNSEEN
        )
        before = getattr(app, FLEET_UNSEEN_REVISION_ATTR, 0)
        assert clear_fleet_unseen_completion(app, "conv-att") is True
        assert not app.conversation_local_marks_service.has_mark(
            "conv-att", ConversationLocalMarksService.FLEET_UNSEEN
        )
        assert getattr(app, FLEET_UNSEEN_REVISION_ATTR, 0) == before + 1
        # Nothing left: strict False, no bump.
        assert clear_fleet_unseen_completion(app, "conv-att") is False
        assert getattr(app, FLEET_UNSEEN_REVISION_ATTR, 0) == before + 1
        # Starred marks are untouched territory: clearing FLEET_UNSEEN
        # never reaches another mark type.
        app.conversation_local_marks_service.star_conversation("conv-att")
        app.conversation_local_marks_service.set_mark(
            "conv-att", ConversationLocalMarksService.FLEET_UNSEEN
        )
        assert clear_fleet_unseen_completion(app, "conv-att") is True
        assert app.conversation_local_marks_service.is_starred("conv-att")
    finally:
        chacha.close()


def test_registration_is_tolerant_and_replaces_by_name():
    """No bridge / no seam degrade silently; re-registration replaces."""
    register_fleet_attention(None, object())  # no bridge: no raise

    class _Seamless:
        pass

    register_fleet_attention(_Seamless(), object())  # no seam: no raise

    class _Bridge:
        def __init__(self):
            self.registrations: list[tuple[str, object]] = []

        def on_fleet_drained(self, name, consumer):
            self.registrations.append((name, consumer))

    bridge = _Bridge()
    register_fleet_attention(bridge, object())
    register_fleet_attention(bridge, object())
    assert [name for name, _ in bridge.registrations] == [
        ConsoleFleetAttentionConsumer.NAME,
        ConsoleFleetAttentionConsumer.NAME,
    ]


def test_toast_copy_grammar_is_exact():
    """The copy helper, pinned -- the toast and the report must not drift."""
    assert fleet_completion_toast_copy("T", ["done"]) == (
        "Background sub-agent finished in “T”.",
        "information",
    )
    assert fleet_completion_toast_copy("T", ["error"]) == (
        "Background sub-agent failed in “T”.",
        "error",
    )
    assert fleet_completion_toast_copy("T", ["cancelled"]) == (
        "Background sub-agent was cancelled in “T”.",
        "warning",
    )
    assert fleet_completion_toast_copy("T", ["done", "done"]) == (
        "2 background sub-agents in “T”: 2 finished.",
        "information",
    )
    assert fleet_completion_toast_copy("T", ["done", "cancelled"]) == (
        "2 background sub-agents in “T”: 1 finished, 1 cancelled.",
        "warning",
    )
