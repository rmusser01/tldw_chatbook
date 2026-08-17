"""Wake at LAUNCH -- delivering what was owed while the app was closed.

task-15860 plan Task 6, and the last fire point in the arc.

The wake-fires-headless landing made a survivor settling with no Console
mounted deliver a full supervisor turn -- inside a process that had opened
Console at least once. `ensure_chat_controller`/`ensure_agent_bridge` are
lazy and their only callers were `ChatScreen` and its Console modules, so
a child that finished while the app was CLOSED waited for the next Console
visit, however many launches later that was. (That was recorded as an
inference in the fires report's §9; `Tests/UI/test_probe_launch_wake.py`
executed it: with Console never opened the runtime holds
`chat_controller=None, chat_store=None, agent_bridge=None`.)

The owner's ruling, implemented literally and pinned here:

* wake at launch is **YES**, and **mark-gated** -- one cheap indexed read,
  and delivery only for a conversation that already carries a
  `FLEET_UNSEEN` mark AND an owed `agent_runs` row;
* it stays behind the existing `[agents] autowake_enabled`; there is no
  separate launch switch;
* **when there are no marks, construct NOTHING** -- startup is
  byte-identical to before. That is
  `test_a_launch_with_no_marks_constructs_nothing_and_reads_once`, and it
  is the pin that protects every user who never touches the fleet.

Rig notes:

* a "restart" is a genuinely SECOND `TldwCli` over the SAME on-disk DBs
  (`tmp_path/chacha.sqlite` + its sibling `agent_runs.db`), never a
  re-used app object;
* the second process boots with `default_tab="library"`, so the real
  routing never constructs a `ChatScreen` at all -- every test asserts
  that as a precondition rather than assuming it;
* the delivery is produced by the app's own deferred-startup path, not by
  calling the launch module directly, except where a test says otherwise.
"""

from __future__ import annotations

import time

import pytest

from Tests.Chat.test_console_fleet_wake import _terminal_subagent_run
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import (
    StaticConversationTreeService,
    _configure_native_ready_console,
)
from Tests.UI.test_console_store_continuity import (
    _StallingWakeGateway,
    _db_chain,
    _seed_console,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_fleet_wake import WAKE_NOTICE_HEADER
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


CHILD_RESULT = "the child's finished answer"
LAUNCH_REPLY = "LAUNCH-WAKE-REPLY"
FLEET_UNSEEN = ConversationLocalMarksService.FLEET_UNSEEN


async def _settle(pilot, predicate, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return bool(predicate())


async def _quiet(pilot, predicate, seconds: float = 2.0) -> bool:
    """True when `predicate` stayed False for the whole window."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return False
        await pilot.pause(0.05)
    return not predicate()


def _launch_app(tmp_path, *, tree=None, gateway_reply=LAUNCH_REPLY):
    """A SECOND process over the same durable state, Console never opened.

    `default_tab="library"` is what keeps Console out of it: the real
    routing builds the Library screen and nothing ever constructs a
    `ChatScreen`.
    """
    app = _build_test_app("library")
    marks = _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    gateway = _StallingWakeGateway(reply=gateway_reply)
    app.console_provider_gateway_factory = lambda: gateway
    if tree is not None:
        app.chat_conversation_scope_service = StaticConversationTreeService(tree)
    return app, marks, gateway


async def _seed_a_finished_background_job(tmp_path):
    """Process ONE: a real Console conversation plus a survivor that
    finished after its turn, marked ◈ and never delivered.

    Everything here goes through production: a mounted `ChatScreen`, a real
    accepted send that PERSISTS the conversation, and the real runs DB the
    app-owned bridge opens.

    Returns:
        (conversation_id, run_id, the seeded conversation's persisted rows)
    """
    app = _build_test_app()
    marks = _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway
    async with app.run_test(size=(160, 48)) as pilot:
        _chat, controller, _store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        runs_db = controller._agent_bridge.runs_db
        _parent, run_id = _terminal_subagent_run(
            runs_db, conversation_id, result=CHILD_RESULT
        )
        marks.set_mark(conversation_id, FLEET_UNSEEN)
        rows = _db_chain(app.chachanotes_db, conversation_id)
        assert len(rows) == 2, (
            "harness precondition: the seeded turn must have persisted two "
            f"rows; got {[(r[1], r[2][:24]) for r in rows]}"
        )
        assert session_id
    return conversation_id, run_id, rows


def _fixture_tree(conversation_id, rows):
    """A conversation tree matching the rows process one persisted."""
    nodes = []
    parent = None
    for row in rows:
        node = {
            "id": str(row[0]),
            "sender": row[1],
            "content": row[2],
            "children": [],
        }
        if parent is None:
            nodes.append(node)
        else:
            parent["children"].append(node)
        parent = node
    return {
        conversation_id: {
            "conversation": {"id": conversation_id, "title": "Seeded"},
            "root_threads": nodes,
        }
    }


def _assert_console_never_mounted(app):
    assert not any(isinstance(s, ChatScreen) for s in app.screen_stack), (
        "harness precondition: this test is about a launch with Console "
        f"NEVER opened; stack is {[type(s).__name__ for s in app.screen_stack]}"
    )


# ---------------------------------------------------------------------------
# The red: a launch delivers what a previous process left owed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_launch_delivers_a_wake_owed_from_a_previous_process(tmp_path):
    """RED before this task: the second process boots, reads nothing,
    builds nothing, and the owed wake sits until the user opens Console.

    Asserted here (the whole chain, not just the send):

    1. exactly ONE wake turn reaches the provider, its payload ending on a
       machine-labelled `user` entry carrying the child's result -- and
       carrying the conversation's REAL prior history, which is what makes
       hydration load-bearing rather than decorative;
    2. the machine-origin SYSTEM notice and the assistant reply are in the
       app-owned store AND in ChaChaNotes;
    3. `agent_runs.wake_delivered_at` is stamped;
    4. no USER row is added anywhere -- a wake is never user input;
    5. the ◈ mark SURVIVES: a launch delivery has no view, so nobody
       watched it.
    """
    conversation_id, run_id, rows = await _seed_a_finished_background_job(tmp_path)
    app, marks, gateway = _launch_app(
        tmp_path, tree=_fixture_tree(conversation_id, rows)
    )

    async with app.run_test(size=(120, 40)) as pilot:
        delivered = await _settle(pilot, lambda: bool(gateway.payloads))
        _assert_console_never_mounted(app)
        assert delivered, (
            "a background sub-agent finished before this process started, its "
            "◈ mark and its owed ledger row both survived -- and the launch "
            "delivered no wake turn at all"
        )
        assert await _quiet(pilot, lambda: len(gateway.payloads) > 1), (
            f"one launch delivered more than one wake turn: {len(gateway.payloads)}"
        )

        # (1) the payload: machine-labelled trailing user entry, real history.
        payload = gateway.payloads[-1]
        trailing = payload[-1]
        assert trailing["role"] == ConsoleMessageRole.USER.value, (
            "the notice must be the payload's trailing user-role entry (a "
            f"payload ending on an assistant row is a prefill); got {trailing['role']!r}"
        )
        assert WAKE_NOTICE_HEADER in str(trailing["content"])
        assert CHILD_RESULT in str(trailing["content"]), (
            "the child's result never reached the supervisor"
        )
        seeded_user = rows[0][2]
        assert any(seeded_user in str(entry.get("content") or "") for entry in payload), (
            "the launch-hydrated session carried NO prior history -- the "
            "supervisor was woken with no idea what it had been doing: "
            f"{[(e.get('role'), str(e.get('content'))[:32]) for e in payload]}"
        )

        # (2) both rows landed in the app-owned store AND in ChaChaNotes.
        store = app.console_runtime.chat_store
        session = next(
            s
            for s in store.sessions()
            if s.persisted_conversation_id == conversation_id
        )
        assert await _settle(
            pilot,
            lambda: any(
                m.content == LAUNCH_REPLY
                for m in store.messages_for_session(session.id)
            ),
        ), (
            "the launch wake turn never landed in the app-owned store: "
            f"{[(m.role.value, m.content[:28]) for m in store.messages_for_session(session.id)]}"
        )
        notices = [
            m
            for m in store.messages_for_session(session.id)
            if getattr(m.metadata, "origin", "") == "agent_wake"
        ]
        assert len(notices) == 1, f"expected one machine-origin notice, got {len(notices)}"
        assert notices[0].role is ConsoleMessageRole.SYSTEM

        assert await _settle(
            pilot,
            lambda: len(_db_chain(app.chachanotes_db, conversation_id)) == 4,
        ), (
            "the launch wake turn never PERSISTED: "
            f"{[(r[1], r[2][:28]) for r in _db_chain(app.chachanotes_db, conversation_id)]}"
        )
        db_rows = _db_chain(app.chachanotes_db, conversation_id)
        senders = [row[1] for row in db_rows]
        assert senders == ["user", "assistant", "system", "assistant"], (
            f"unexpected persisted row shape: {senders}"
        )
        assert db_rows[2][2].startswith(WAKE_NOTICE_HEADER)
        assert db_rows[3][2] == LAUNCH_REPLY

        # (3) the ledger is stamped.
        runs_db = app.console_runtime.agent_bridge.runs_db
        assert (runs_db.get_run(run_id) or {}).get("wake_delivered_at"), (
            "the launch delivery never stamped agent_runs.wake_delivered_at"
        )

        # (4) no USER row was added.
        assert senders.count("user") == 1, f"the launch wake persisted a USER row: {senders}"
        store_user_rows = [
            m
            for m in store.messages_for_session(session.id)
            if m.role is ConsoleMessageRole.USER
        ]
        assert [m.content for m in store_user_rows] == [rows[0][2]], (
            f"the launch wake wrote a USER transcript row: {store_user_rows!r}"
        )

        # (5) the ◈ mark survives -- nobody could have watched this.
        assert marks.has_mark(conversation_id, FLEET_UNSEEN), (
            "a wake delivered at launch with no Console mounted cleared the ◈ "
            "mark -- the user has no way to learn the supervisor turn ever ran"
        )


@pytest.mark.asyncio
async def test_a_second_launch_does_not_re_announce_a_delivered_wake(tmp_path):
    """Exactly-once across launches. The ◈ mark deliberately SURVIVES the
    first launch's delivery (nobody watched it), so the second launch reads
    a mark and must still stay silent -- the ledger, not the mark, is what
    says a result was delivered.

    The control that stops this passing vacuously: a genuinely undelivered
    SECOND run in the same conversation IS delivered by a third launch, and
    its notice names the second child, never the first.
    """
    conversation_id, _run_id, rows = await _seed_a_finished_background_job(tmp_path)
    tree = _fixture_tree(conversation_id, rows)

    app1, marks1, gateway1 = _launch_app(tmp_path, tree=tree)
    async with app1.run_test(size=(120, 40)) as pilot:
        assert await _settle(pilot, lambda: bool(gateway1.payloads)), (
            "precondition: the first launch must deliver"
        )
    assert marks1.has_mark(conversation_id, FLEET_UNSEEN), (
        "precondition: the unwatched delivery must KEEP the mark, which is "
        "exactly what makes the second launch a real test"
    )

    app2, _marks2, gateway2 = _launch_app(tmp_path, tree=tree)
    async with app2.run_test(size=(120, 40)) as pilot:
        assert await _quiet(pilot, lambda: bool(gateway2.payloads), seconds=6.0), (
            "a second launch re-announced a wake the first launch already "
            f"delivered: {gateway2.payloads!r}"
        )
        _assert_console_never_mounted(app2)
        assert len(_db_chain(app2.chachanotes_db, conversation_id)) == 4, (
            "the second launch persisted more rows"
        )

    # The control: a genuinely new completion IS delivered on a later launch.
    runs_db = AgentRunsDB(tmp_path / "agent_runs.db", client_id="seed-2")
    _terminal_subagent_run(runs_db, conversation_id, result="second child result")
    runs_db.close()
    app3, _marks3, gateway3 = _launch_app(tmp_path, tree=tree)
    async with app3.run_test(size=(120, 40)) as pilot:
        assert await _settle(pilot, lambda: bool(gateway3.payloads)), (
            "the exactly-once drop is too wide: a genuinely undelivered "
            "completion was never announced"
        )
        notice = str(gateway3.payloads[-1][-1]["content"])
        assert "second child result" in notice
        assert CHILD_RESULT not in notice, (
            "the launch re-announced the already-delivered first child"
        )


# ---------------------------------------------------------------------------
# The startup-cost pin -- the one that protects everyone who never uses
# the fleet.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_launch_with_no_marks_constructs_nothing_and_reads_once(tmp_path):
    """With no marks, a launch does ONE indexed read and nothing else.

    Four independent observations, because "nothing was constructed" is
    exactly the claim a weak test states and never checks:

    1. the marks service saw exactly ONE `list_marked_conversation_ids`
       call, for `fleet_unseen`;
    2. the runtime holds no store, no gateway, no bridge, no controller;
    3. **no `agent_runs.db` file exists on disk** -- constructing the
       bridge opens (and creates) it, so the filesystem is an observer the
       production code cannot lie to;
    4. no `deferred_launch_wake` task was ever created.

    The control against vacuity is
    `test_the_startup_cost_pin_is_not_vacuous` below, which runs the same
    probes WITH a mark and sees every one of them flip.
    """
    app = _build_test_app("library")
    marks = _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    calls: list[str] = []
    real_list = marks.list_marked_conversation_ids

    def recording_list(mark_type=None, **kwargs):
        calls.append(str(mark_type))
        return real_list(mark_type, **kwargs)

    marks.list_marked_conversation_ids = recording_list
    task_names: list[str] = []
    real_create = type(app)._create_deferred_startup_task

    def recording_create(self, coroutine, *, name):
        task_names.append(name)
        return real_create(self, coroutine, name=name)

    type(app)._create_deferred_startup_task = recording_create
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _settle(pilot, lambda: getattr(app, "_ui_ready", False)), (
                "harness precondition: the app never became ready, so the "
                "deferred startup work never ran and this pin proves nothing"
            )
            await pilot.pause(0.2)
            _assert_console_never_mounted(app)
            assert calls == [ConversationLocalMarksService.FLEET_UNSEEN], (
                "a launch with no marks must cost exactly one indexed mark "
                f"listing; got {calls}"
            )
            runtime = app.console_runtime
            assert runtime.chat_store is None, "a launch with no marks built a store"
            assert runtime.provider_gateway is None, (
                "a launch with no marks built a provider gateway"
            )
            assert runtime.agent_bridge is None, (
                "a launch with no marks built an agent bridge"
            )
            assert runtime.chat_controller is None, (
                "a launch with no marks built a chat controller"
            )
            assert not (tmp_path / "agent_runs.db").exists(), (
                "a launch with no marks opened the agent runs DB -- the bridge "
                "was constructed after all"
            )
            assert "deferred_launch_wake" not in task_names, (
                f"a launch with no marks scheduled the wake task: {task_names}"
            )
    finally:
        type(app)._create_deferred_startup_task = real_create


@pytest.mark.asyncio
async def test_the_startup_cost_pin_is_not_vacuous(tmp_path):
    """The control for the pin above: with a mark present, every one of its
    four observations flips. Without this, a launch hook that never ran at
    all would satisfy the pin perfectly."""
    conversation_id, _run_id, rows = await _seed_a_finished_background_job(tmp_path)
    app, _marks, gateway = _launch_app(
        tmp_path, tree=_fixture_tree(conversation_id, rows)
    )
    task_names: list[str] = []
    real_create = type(app)._create_deferred_startup_task

    def recording_create(self, coroutine, *, name):
        task_names.append(name)
        return real_create(self, coroutine, name=name)

    type(app)._create_deferred_startup_task = recording_create
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _settle(pilot, lambda: bool(gateway.payloads))
            runtime = app.console_runtime
            assert runtime.chat_store is not None
            assert runtime.provider_gateway is not None
            assert runtime.agent_bridge is not None
            assert runtime.chat_controller is not None
            assert (tmp_path / "agent_runs.db").exists()
            assert "deferred_launch_wake" in task_names, task_names
    finally:
        type(app)._create_deferred_startup_task = real_create


# ---------------------------------------------------------------------------
# AC#3's phantom-wake case, at the launch fire point.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_crash_killed_child_swept_to_error_wakes_nobody_at_launch(tmp_path):
    """A child left `running` by a crash is swept to `error` by the next
    `AgentRunsDB.__init__`, which makes it terminal, undelivered and
    therefore OWED by the ledger's own definition -- and it carries no
    mark, because nothing ever settled it through the fan-out.

    Seeding from the ledger alone would manufacture a wake here. The claim
    is marks-indexed, so nothing fires. The assertion that stops this
    passing for the wrong reason: `undelivered_wake_runs` genuinely DOES
    report the orphan, so the silence is the mark gate's doing, not an
    empty ledger.
    """
    app0 = _build_test_app("library")
    _attach_real_dbs(app0, tmp_path)
    app0.chachanotes_db.add_conversation({"id": "conv-crashed", "title": "Crashed"})
    runs_db = AgentRunsDB(tmp_path / "agent_runs.db", client_id="seed")
    parent_id = runs_db.create_run(conversation_id="conv-crashed", agent_kind="primary")
    runs_db.set_status(parent_id, "done", "turn final")
    child_id = runs_db.create_run(
        conversation_id="conv-crashed",
        agent_kind="subagent",
        task="killed mid-flight",
        parent_run_id=parent_id,
    )
    runs_db.close()
    # The restart sweep: a fresh AgentRunsDB over the same file, with the
    # per-process guard discarded the way Tests/DB/test_agent_runs_db.py
    # models a restart.
    AgentRunsDB._swept_paths.discard(str(tmp_path / "agent_runs.db"))
    swept = AgentRunsDB(tmp_path / "agent_runs.db", client_id="sweep")
    assert (swept.get_run(child_id) or {}).get("status") == "error", (
        "harness precondition: the reconcile sweep must have marked the "
        "crash-killed child as error"
    )
    assert [row["id"] for row in swept.undelivered_wake_runs("conv-crashed")] == [
        child_id
    ], (
        "harness precondition: the ledger must consider this orphan OWED -- "
        "otherwise the silence below proves nothing about the mark gate"
    )
    swept.close()

    app, marks, gateway = _launch_app(tmp_path)
    assert marks.list_marked_conversation_ids(FLEET_UNSEEN) == (), (
        "harness precondition: a crash-killed child leaves no ◈ mark"
    )
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _quiet(pilot, lambda: bool(gateway.payloads), seconds=5.0), (
            "a crash-killed child with no ◈ mark woke the supervisor at "
            f"launch -- a phantom wake: {gateway.payloads!r}"
        )
        _assert_console_never_mounted(app)
        assert app.console_runtime.chat_controller is None, (
            "an unmarked owed row still built the whole Console runtime at "
            "launch"
        )


# ---------------------------------------------------------------------------
# The kill switch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_kill_switch_silences_the_launch_fire_point_and_loses_nothing(
    tmp_path, monkeypatch
):
    """`autowake_enabled = false` seeds nothing at launch and loses nothing
    durable: the ◈ mark and the owed ledger row are both still there
    afterwards -- and a later launch with the switch back ON delivers
    exactly what OFF recorded."""
    conversation_id, run_id, rows = await _seed_a_finished_background_job(tmp_path)
    tree = _fixture_tree(conversation_id, rows)

    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    app_off, marks_off, gateway_off = _launch_app(tmp_path, tree=tree)
    async with app_off.run_test(size=(120, 40)) as pilot:
        assert await _quiet(pilot, lambda: bool(gateway_off.payloads), seconds=5.0), (
            "the kill switch is OFF and a launch still woke the supervisor: "
            f"{gateway_off.payloads!r}"
        )
        assert app_off.console_runtime.chat_controller is None, (
            "the kill switch is OFF and the launch still built the runtime"
        )
    assert marks_off.has_mark(conversation_id, FLEET_UNSEEN), (
        "the kill switch lost the ◈ mark"
    )
    off_runs = AgentRunsDB(tmp_path / "agent_runs.db", client_id="check")
    assert not (off_runs.get_run(run_id) or {}).get("wake_delivered_at"), (
        "the kill switch stamped the ledger for a wake it never delivered"
    )
    off_runs.close()

    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
    app_on, _marks_on, gateway_on = _launch_app(tmp_path, tree=tree)
    async with app_on.run_test(size=(120, 40)) as pilot:
        assert await _settle(pilot, lambda: bool(gateway_on.payloads)), (
            "flipping the kill switch back on did not deliver the wake that "
            "OFF recorded"
        )
        assert CHILD_RESULT in str(gateway_on.payloads[-1][-1]["content"])


# ---------------------------------------------------------------------------
# Stale / unresolvable marks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unresolvable_ephemeral_mark_is_cleared_at_launch(tmp_path):
    """A fleet turn in an UNSAVED session is keyed by the ephemeral
    `session.id` (`_agent_conversation_id` returns
    `persisted_conversation_id or session_id`), so its ◈ mark names no
    ChaChaNotes conversation once that process is gone. Executed in
    `Tests/UI/test_probe_launch_wake.py`: the mark survives the restart and
    resolves to nothing, forever.

    A launch that owes that conversation a wake it can never deliver clears
    the mark instead of retrying it every boot for the life of the install.
    The owed ledger rows are deliberately untouched -- with the mark gone,
    nothing indexes them, so nothing re-announces them.
    """
    app0 = _build_test_app("library")
    marks0 = _attach_real_dbs(app0, tmp_path)
    ephemeral_id = "console-session-9f3ac1"
    runs_db = AgentRunsDB(tmp_path / "agent_runs.db", client_id="seed")
    _parent, run_id = _terminal_subagent_run(
        runs_db, ephemeral_id, result="work nobody can be shown"
    )
    runs_db.close()
    marks0.set_mark(ephemeral_id, FLEET_UNSEEN)
    assert app0.chachanotes_db.get_conversation_by_id(ephemeral_id) is None, (
        "harness precondition: an ephemeral session id names no conversation"
    )

    app, marks, gateway = _launch_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _settle(
            pilot, lambda: not marks.has_mark(ephemeral_id, FLEET_UNSEEN)
        ), (
            "a ◈ mark that can never be resolved again survived the launch -- "
            "it will be retried on every boot for the life of the install"
        )
        assert await _quiet(pilot, lambda: bool(gateway.payloads), seconds=2.0), (
            "the launch tried to deliver into a conversation that does not "
            f"exist: {gateway.payloads!r}"
        )
    check = AgentRunsDB(tmp_path / "agent_runs.db", client_id="check")
    assert not (check.get_run(run_id) or {}).get("wake_delivered_at"), (
        "clearing an unresolvable mark must not stamp the ledger -- the "
        "result was never delivered to anyone"
    )
    check.close()


@pytest.mark.asyncio
async def test_a_mark_with_nothing_owed_is_left_alone_at_launch(tmp_path):
    """A ◈ mark with no owed ledger row is NOT stale: it is the
    delivered-but-unseen badge the wake sets when it delivers off-view
    (task-15971). A launch must not deliver anything for it, and must not
    clear it -- the user has still not seen that result."""
    app0 = _build_test_app("library")
    marks0 = _attach_real_dbs(app0, tmp_path)
    app0.chachanotes_db.add_conversation({"id": "conv-seen-later", "title": "Delivered"})
    marks0.set_mark("conv-seen-later", FLEET_UNSEEN)

    app, marks, gateway = _launch_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _quiet(pilot, lambda: bool(gateway.payloads), seconds=4.0), (
            f"a launch delivered a wake for a conversation owing none: {gateway.payloads!r}"
        )
        assert marks.has_mark("conv-seen-later", FLEET_UNSEEN), (
            "the launch cleared a delivered-but-unseen ◈ badge -- the user "
            "loses the only pointer they had to a result they never saw"
        )
