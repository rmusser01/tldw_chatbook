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
        self.notifications.append((threading.current_thread().name, message, severity))


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
@pytest.mark.bootstrap_profile
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
            outcome = _run(bridge, store, session, aid, conversation_id=conversation_id)
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
@pytest.mark.bootstrap_profile
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
        assert [c.settled_after_turn for e in events for c in e.children] == [False], (
            "a child settling while its turn still runs is within-turn"
        )

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
@pytest.mark.bootstrap_profile
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
            _drain(
                children=[_child(after_turn=False), _child("error", after_turn=False)]
            )
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


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("mode", ["receipt", "lazy", "borrowed", "fallback"])
def test_native_attention_returns_only_its_new_worker_handles(tmp_path, mode):
    import sqlite3
    import sys
    from contextlib import ExitStack

    from tldw_chatbook.Backup_Recovery import participants, storage_admission
    from tldw_chatbook.Chat.console_activity_receipts import (
        ConsoleActivityReceiptService,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    chacha = CharactersRAGDB(str(tmp_path / "attention.sqlite"), client_id="attention")
    runs = AgentRunsDB(tmp_path / "attention-runs.sqlite")
    app = _AppStub(chacha)
    receipts = ConsoleActivityReceiptService(runs, app.conversation_local_marks_service)
    if mode == "lazy":
        from tldw_chatbook.Chat.console_runtime import (
            _LazyConsoleActivityReceiptService,
        )

        receipts = _LazyConsoleActivityReceiptService(
            runs, app.conversation_local_marks_service
        )
    loop = asyncio.new_event_loop()
    loop.close()
    consumer = ConsoleFleetAttentionConsumer(
        app, loop=loop, receipt_service=receipts if mode != "fallback" else None
    )
    if mode == "lazy":
        assert receipts._delegate is None
        consumer(_drain(children=[_child(after_turn=False)]))
        assert receipts._delegate is None
    connections, errors = [], []

    def observe(frame, event, result):
        if (
            event == "return"
            and frame.f_code is participants._register_core_connection.__code__
            and frame.f_locals.get("repository") in (chacha, runs)
        ):
            connection = frame.f_locals["connection"]
            participant, lease = frame.f_locals["participant"], frame.f_locals["lease"]
            assert (
                result is connection
                and lease.resource_thread is threading.current_thread()
            )
            connections.append((connection, participant, lease))

    def worker():
        sys.setprofile(observe)
        try:
            with ExitStack() as borrowed:
                if mode == "borrowed":
                    chacha_connection = borrowed.enter_context(
                        chacha.transaction()
                    ).connection
                    runs_connection = borrowed.enter_context(runs.connection())
                    originals = tuple(connections)
                consumer(_drain(children=[_child()]))
                assert connections
                if mode == "borrowed":
                    assert tuple(connections) == originals
                    for connection, participant, lease in originals:
                        assert (
                            connection is chacha_connection
                            or connection is runs_connection
                        )
                        assert connection.in_transaction is (
                            connection is chacha_connection
                        )
                        assert connection.execute("SELECT 1").fetchone()[0] == 1
                        with storage_admission._lock:
                            assert participant.connections[connection] is lease
                            assert lease in storage_admission._live_leases
                else:
                    for connection, participant, lease in connections:
                        with storage_admission._lock:
                            assert connection not in participant.connections
                            assert lease not in storage_admission._live_leases
                        with pytest.raises(sqlite3.ProgrammingError):
                            connection.execute("SELECT 1")
            if mode == "borrowed":
                # The test's original worker acquired these before the callback.
                chacha.close_connection()
                runs.close()
        except BaseException as error:  # noqa: BLE001 - return the exact worker failure.
            errors.append(error)
        finally:
            sys.setprofile(None)

    thread = threading.Thread(target=worker, name="original-attention-worker")
    try:
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
        if errors:
            raise errors[0]
        assert connections
        if mode == "lazy":
            assert type(receipts._delegate) is ConsoleActivityReceiptService
        with storage_admission._lock:
            assert all(
                connection not in participant.connections
                and lease not in storage_admission._live_leases
                for connection, participant, lease in connections
            )
        assert app.conversation_local_marks_service.has_mark("conv-att", "fleet_unseen")
        if mode != "fallback":
            assert len(receipts.unseen_snapshot()) == 1
            assert receipts.unseen_snapshot()[0].run_id == "run-1"
        assert len(app.notifications) == 1
    finally:
        runs.close()
        chacha.close()


def test_attention_preserves_lazy_custom_getter_boundaries():
    from types import SimpleNamespace

    calls, queued = [], []

    class Title:
        def get_conversation_by_id(self, _conversation):
            calls.append("title-query")
            return {"title": "Original title"}

    class Marks:
        FLEET_UNSEEN = "fleet_unseen"

        @property
        def db(self):
            raise AssertionError("custom marks storage must not be inspected")

        def set_mark(self, *_args):
            calls.append("mark")

    class App(_AppStub):
        def __init__(self):
            self.notifications, self.staged = [], []
            self.screen = object()
            self.pending_handoffs = self._Handoffs(self.staged)

        @property
        def chachanotes_db(self):
            calls.append("title-getter")
            return Title()

        @property
        def conversation_local_marks_service(self):
            calls.append("marks-getter")
            return Marks()

    class Receipts:
        @property
        def _db(self):
            raise AssertionError("custom receipt storage must not be inspected")

        @property
        def _marks(self):
            raise AssertionError("custom receipt marks must not be inspected")

        def publish_fleet_drain(self, _event):
            calls.append("publish")
            return SimpleNamespace(complete=True)

    loop = SimpleNamespace(
        is_closed=lambda: False,
        call_soon_threadsafe=lambda callback, *args: queued.append((callback, args)),
    )
    app = App()
    consumer = ConsoleFleetAttentionConsumer(app, loop=loop, receipt_service=Receipts())
    consumer(_drain(children=[_child(after_turn=False)]))
    assert calls == [] and queued == []
    consumer(_drain(children=[_child()]))
    assert calls == ["publish"] and len(queued) == 1
    callback, args = queued.pop()
    callback(*args)
    assert calls == ["publish", "title-getter", "title-query"]
    calls.clear()
    fallback = ConsoleFleetAttentionConsumer(
        app, loop=SimpleNamespace(is_closed=lambda: True)
    )
    fallback(_drain(children=[_child()]))
    assert calls == ["marks-getter", "mark", "title-getter", "title-query"]


@pytest.mark.bootstrap_profile
def test_attention_preserves_native_memory_database_owner():
    database = CharactersRAGDB(":memory:", client_id="attention-memory")
    connection = database.get_connection()
    try:
        app = _AppStub(database)
        consumer = ConsoleFleetAttentionConsumer(app, loop=None)
        consumer(_drain(children=[_child()]))
        assert database.get_connection() is connection
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert app.conversation_local_marks_service.has_mark("conv-att", "fleet_unseen")
    finally:
        database.close_connection()


def _native_attention_close_failure(root, *, fail_on_fallback=False):
    """An isolated injected close error keeps the actual native lease visible."""
    import os
    import sys

    from tldw_chatbook.Backup_Recovery import participants, storage_admission
    from tldw_chatbook.Chat.console_activity_receipts import (
        ConsoleActivityReceiptService,
    )
    from tldw_chatbook.Chat.console_agent_bridge import FleetDrainFanout
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    database = CharactersRAGDB(str(root / "failed-close.sqlite"), client_id="close")
    runs = AgentRunsDB(root / "failed-close-runs.sqlite")
    app = _AppStub(database)
    receipts = ConsoleActivityReceiptService(runs, app.conversation_local_marks_service)
    consumer = ConsoleFleetAttentionConsumer(app, loop=None, receipt_service=receipts)
    fanout = FleetDrainFanout()
    boundary_errors = []

    def attention(event):
        try:
            consumer(event)
        except Exception as error:
            boundary_errors.append(error)
            raise

    fanout.register("attention", attention)
    later, errors, registrations, calls = [], [], [], []
    fanout.register("after-attention", later.append)
    failure = RuntimeError("original attention close did not return")

    close_count = 0
    original_close = database.close_connection

    def refuse_close():
        nonlocal close_count
        close_count += 1
        if not fail_on_fallback or close_count == 2:
            calls.append(failure)
            raise failure
        original_close()

    database.close_connection = refuse_close

    def observe(frame, event, result):
        if (
            event == "return"
            and frame.f_code is participants._register_core_connection.__code__
            and frame.f_locals.get("repository") is database
        ):
            connection = frame.f_locals["connection"]
            participant, lease = frame.f_locals["participant"], frame.f_locals["lease"]
            assert (
                result is connection
                and lease.resource_thread is threading.current_thread()
            )
            registrations.append((connection, participant, lease))

    event = _drain(
        children=[_child(), _child("unsupported")] if fail_on_fallback else [_child()]
    )

    def worker():
        sys.setprofile(observe)
        try:
            fanout.fire(event)
            assert calls == [failure] and later == [event]
            assert len(boundary_errors) == 1
            assert (
                boundary_errors[0] is failure
                or boundary_errors[0].__cause__ is failure
            )
            assert registrations
            with storage_admission._lock:
                if fail_on_fallback:
                    assert len(registrations) == 2
                    connection, participant, lease = registrations[0]
                    assert connection not in participant.connections
                    assert lease not in storage_admission._live_leases
                    retained = registrations[1:]
                else:
                    retained = registrations
                assert all(
                    participant.connections.get(connection) is lease
                    and lease in storage_admission._live_leases
                    for connection, participant, lease in retained
                )
            assert app.notifications == [] and app.staged == []
            assert len(receipts.unseen_snapshot()) == 1
        except BaseException as error:  # noqa: BLE001 - preserve original diagnostic failure.
            errors.append(error)
        finally:
            sys.setprofile(None)

    thread = threading.Thread(target=worker, name="original-attention-close-failure")
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    if errors:
        raise errors[0]
    assert all(lease.resource_thread is thread for _, _, lease in registrations)
    print("ATTENTION_ORIGINAL_CLOSE_RETAINED; no creator drain", flush=True)
    os._exit(17)


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("fail_on_fallback", [False, True])
def test_attention_close_failure_keeps_actual_lease_and_fanout_continues(
    tmp_path, fail_on_fallback
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = """
import os
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
with user_fixture_default_owner():
    from Tests.Chat.test_fleet_attention import _native_attention_close_failure
    _native_attention_close_failure(
        Path(os.environ["XDG_DATA_HOME"]).parent,
        fail_on_fallback=FAIL_ON_FALLBACK,
    )
""".replace("FAIL_ON_FALLBACK", repr(fail_on_fallback))
    with pytest.raises(AssertionError, match="ATTENTION_ORIGINAL_CLOSE_RETAINED"):
        _run(tmp_path, "attention-close", "retained", script=script, timeout=60)


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("operation", ["publish_fleet_drain", "ensure_fleet_mark"])
@pytest.mark.parametrize("change", ["helper", "helper-and-record", "module"])
def test_native_receipt_storage_refuses_replaced_preparation_source(
    tmp_path, monkeypatch, operation, change
):
    import sys
    from types import ModuleType

    from tldw_chatbook.Chat import console_attention_preparation as preparation
    from tldw_chatbook.Chat.console_activity_receipts import (
        ConsoleActivityReceiptService,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    database = CharactersRAGDB(str(tmp_path / "marks.sqlite"), client_id="attention")
    runs = AgentRunsDB(tmp_path / "runs.sqlite")
    marks = ConversationLocalMarksService(database)
    receipts = ConsoleActivityReceiptService(runs, marks)
    called = []

    def foreign_helper(*args):
        called.append(args)

    try:
        with monkeypatch.context() as patch:
            if change == "module":
                foreign_module = ModuleType(preparation.__name__)
                vars(foreign_module).update(vars(preparation))
                patch.setitem(sys.modules, preparation.__name__, foreign_module)
            else:
                patch.setattr(
                    preparation, "_enter_receipt_storage", foreign_helper, raising=False
                )
                if change == "helper-and-record":
                    patch.setattr(
                        preparation,
                        "_RECEIPT_STORAGE_SOURCE",
                        (
                            foreign_helper,
                            foreign_helper.__code__,
                            foreign_helper.__defaults__,
                            foreign_helper.__kwdefaults__,
                            foreign_helper.__closure__,
                        ),
                        raising=False,
                    )
            argument = (
                _drain(children=[_child()])
                if operation == "publish_fleet_drain"
                else "conv-att"
            )
            with pytest.raises(
                RuntimeError, match="console_attention_storage_source_changed"
            ):
                getattr(receipts, operation)(argument)
        assert called == []
        assert runs.list_unseen_console_activity().rows == ()
        assert not marks.has_mark("conv-att", marks.FLEET_UNSEEN)
    finally:
        runs.close()
        database.close_connection()


@pytest.mark.parametrize("operation", ["publish_fleet_drain", "ensure_fleet_mark"])
@pytest.mark.parametrize("receiver", ["foreign", "subclass"])
def test_borrowed_receipt_methods_do_not_inspect_custom_storage(operation, receiver):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_activity_receipts import (
        ConsoleActivityReceiptService,
    )

    class CustomStorage:
        _lock = threading.RLock()

        @property
        def _db(self):
            raise AssertionError("borrowed method inspected custom database")

        @property
        def _marks(self):
            raise AssertionError("borrowed method inspected custom marks")

        def _set_degraded(self, *_args):
            pass

    class Inherited(CustomStorage, ConsoleActivityReceiptService):
        pass

    value = object.__new__(Inherited if receiver == "subclass" else CustomStorage)
    argument = (
        SimpleNamespace(conversation_id="", drain_id="")
        if operation == "publish_fleet_drain"
        else ""
    )
    result = getattr(ConsoleActivityReceiptService, operation)(value, argument)
    if operation == "publish_fleet_drain":
        assert result.complete is False
    else:
        assert result is False


def test_foreign_receipt_class_export_cannot_grant_storage_discovery(monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.Chat import console_activity_receipts, console_fleet_attention

    published = []

    class ForeignReceipts:
        @property
        def _db(self):
            raise AssertionError("foreign receipt database inspected")

        @property
        def _marks(self):
            raise AssertionError("foreign receipt marks inspected")

        def publish_fleet_drain(self, event):
            published.append(event)
            return SimpleNamespace(complete=True)

    monkeypatch.setattr(
        console_activity_receipts, "ConsoleActivityReceiptService", ForeignReceipts
    )
    monkeypatch.setattr(
        console_fleet_attention,
        "ConsoleActivityReceiptService",
        ForeignReceipts,
        raising=False,
    )
    app = _AppStub()
    event = _drain(children=[_child()])
    ConsoleFleetAttentionConsumer(app, receipt_service=ForeignReceipts())(event)
    assert published == [event]
    assert len(app.notifications) == 1


@pytest.mark.parametrize("receiver", ["foreign", "subclass"])
def test_borrowed_mark_method_can_complete_without_custom_database_inspection(receiver):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_activity_receipts import (
        ConsoleActivityReceiptService,
    )

    written = []

    class CustomStorage:
        _lock = threading.RLock()
        _marks = SimpleNamespace(
            FLEET_UNSEEN="fleet_unseen",
            set_mark=lambda conversation, mark: written.append((conversation, mark)),
        )

        @property
        def _db(self):
            raise AssertionError("borrowed mark method inspected custom database")

    class Inherited(CustomStorage, ConsoleActivityReceiptService):
        pass

    value = object.__new__(Inherited if receiver == "subclass" else CustomStorage)
    assert ConsoleActivityReceiptService.ensure_fleet_mark(value, "conv-att") is True
    assert written == [("conv-att", "fleet_unseen")]


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("change", ["helper-and-record", "module"])
def test_runtime_pin_refuses_matching_helper_and_record_before_first_receipt_import(
    tmp_path, change,
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = """
import asyncio
import os
import sys
from pathlib import Path
from types import ModuleType
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
with user_fixture_default_owner():
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    runtime = ConsoleRuntime(None)
    from tldw_chatbook.Chat import console_attention_preparation as preparation
    from tldw_chatbook.Chat.conversation_local_marks_service import ConversationLocalMarksService
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    root = Path(os.environ["XDG_DATA_HOME"])
    database = CharactersRAGDB(str(root / "marks.sqlite"), client_id="attention")
    runs = AgentRunsDB(root / "runs.sqlite")
    marks = ConversationLocalMarksService(database)
    was_cold = "tldw_chatbook.Chat.console_activity_receipts" not in sys.modules
    called = []
    def foreign_helper(*args):
        called.append(args)
    original_helper = preparation._enter_receipt_storage
    original_record = preparation._RECEIPT_STORAGE_SOURCE
    parent = sys.modules["tldw_chatbook.Chat"]
    if CHANGE == "module":
        clone = ModuleType(preparation.__name__)
        vars(clone).update(vars(preparation))
        sys.modules[preparation.__name__] = clone
        parent.console_attention_preparation = clone
    else:
        preparation._enter_receipt_storage = foreign_helper
        preparation._RECEIPT_STORAGE_SOURCE = (
            foreign_helper, foreign_helper.__code__, foreign_helper.__defaults__,
            foreign_helper.__kwdefaults__, foreign_helper.__closure__,
        )
    from tldw_chatbook.Chat.console_activity_receipts import ConsoleActivityReceiptService
    receipts = ConsoleActivityReceiptService(runs, marks)
    def observe(frame, event, result):
        if event == "call" and frame.f_code is original_helper.__code__:
            called.append("original helper")
    sys.setprofile(observe)
    try:
        receipts.ensure_fleet_mark("conv-att")
    except RuntimeError as error:
        assert str(error) == "console_attention_storage_source_changed", error
    else:
        raise AssertionError("changed preparation source was accepted")
    finally:
        sys.setprofile(None)
    assert called == []
    assert was_cold, "receipt implementation loaded before its first operation"
    preparation._enter_receipt_storage = original_helper
    preparation._RECEIPT_STORAGE_SOURCE = original_record
    sys.modules[preparation.__name__] = preparation
    parent.console_attention_preparation = preparation
    assert receipts.ensure_fleet_mark("conv-att") is True
    assert marks.has_mark("conv-att", marks.FLEET_UNSEEN)
    asyncio.run(runtime.dispose())
    runs.close()
    database.close_connection()
print("retired and reopened", flush=True)
""".replace("CHANGE", repr(change))
    _run(tmp_path, "attention-source", "refused", script=script, timeout=60)


@pytest.mark.bootstrap_profile
def test_receipt_storage_uses_type_identity_before_foreign_database_getters():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_activity_receipts import (
        ConsoleActivityReceiptService,
    )

    class EqualToNative(type):
        def __eq__(cls, other):
            return True

        __hash__ = type.__hash__

    class ForeignDatabase(metaclass=EqualToNative):
        receipt_capability_available = True

        @property
        def is_memory_db(self):
            raise AssertionError("equality spoof inspected foreign database")

    written = []
    marks = SimpleNamespace(
        FLEET_UNSEEN="fleet_unseen",
        set_mark=lambda conversation, mark: written.append((conversation, mark)),
    )
    receipts = ConsoleActivityReceiptService(ForeignDatabase(), marks)
    assert receipts.ensure_fleet_mark("conv-att") is True
    assert written == [("conv-att", "fleet_unseen")]
