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

import asyncio
from contextlib import closing, contextmanager
import gc
import time
import weakref
from types import SimpleNamespace

import pytest

from Tests.agent_runs_owner import owned_agent_runs_db

# These owners exercise native config participants selected at collection.
pytestmark = pytest.mark.bootstrap_profile

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
    _navigate,
    _seed_console,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_fleet_wake import WAKE_NOTICE_HEADER
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.console_execution_owner import ConsoleExecutionOwner
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


CHILD_RESULT = "the child's finished answer"
LAUNCH_REPLY = "LAUNCH-WAKE-REPLY"
FLEET_UNSEEN = ConversationLocalMarksService.FLEET_UNSEEN


@contextmanager
def _fixture_runs(path, *, client_id):
    """Return the native store and lease before the next simulated process."""
    with ConsoleExecutionOwner.acquire(path) as owner:
        with closing(
            owned_agent_runs_db(path, client_id=client_id, execution_owner=owner)
        ) as database:
            yield database


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


def _launch_app(
    tmp_path, *, tree=None, gateway_reply=LAUNCH_REPLY, real_service=False, tab="library"
):
    """A SECOND process over the same durable state, Console never opened.

    `default_tab="library"` is what keeps Console out of it: the real
    routing builds the Library screen and nothing ever constructs a
    `ChatScreen`.

    Args:
        tree: Trees for a `StaticConversationTreeService` double.
        real_service: Re-run the app's own
            `_wire_chat_conversation_services()` instead, so the tree comes
            from the PRODUCTION `ChatConversationService` reading the real
            on-disk rows. `_build_test_app` patches
            `_open_profile_chachanotes` to `None`, so that wiring ran at
            `__init__` with no DB and left the local service `None` — which
            is why every resume test in this suite otherwise injects a
            double. Used by the headline e2e so its "the payload carried
            real prior history" claim is not a claim about a fixture.
    """
    app = _build_test_app(tab)
    marks = _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    gateway = _StallingWakeGateway(reply=gateway_reply)
    app.console_provider_gateway_factory = lambda: gateway
    if real_service:
        app._wire_chat_conversation_services()
        assert app.local_chat_conversation_service is not None, (
            "harness precondition: the real conversation service must be wired"
        )
    elif tree is not None:
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
    # The PRODUCTION tree loader, not a double: this test's history claim
    # must be about the real persisted rows, not about a fixture.
    app, marks, gateway = _launch_app(tmp_path, real_service=True)

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
        assert any(
            seeded_user in str(entry.get("content") or "") for entry in payload
        ), (
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
        assert len(notices) == 1, (
            f"expected one machine-origin notice, got {len(notices)}"
        )
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
        assert senders.count("user") == 1, (
            f"the launch wake persisted a USER row: {senders}"
        )
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

        # The app handle the launch-built controller needs. Deliberately not
        # a view-hook slot (it is the APP, which outlives every view), and
        # nothing else here would notice it missing: it is what carries a
        # headless approval round's app-wide toast and its
        # `call_from_thread` bridge (the headless-approval landing). A
        # wiring assertion, labelled as one -- the consequence itself is
        # owned by `Tests/UI/test_console_headless_approval.py`.
        assert app.console_runtime.chat_controller.app is app, (
            "the launch-built controller has no app handle, so a headless "
            "approval round in a launch wake could surface nothing"
        )


def _deny_local_skill_listing(monkeypatch):
    """Deny ``skills.context.list.local`` alone, as a server runtime source does."""
    from tldw_chatbook.runtime_policy import enforcement
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError

    original = enforcement.ServicePolicyEnforcer.require_allowed

    def require_allowed(self, *, action_id, **kwargs):
        if action_id == "skills.context.list.local":
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code="wrong_source",
                user_message="Local skill listing requires local mode.",
                effective_source="server",
                authority_owner="local",
            )
        return original(self, action_id=action_id, **kwargs)

    monkeypatch.setattr(enforcement.ServicePolicyEnforcer, "require_allowed", require_allowed)


@pytest.mark.asyncio
@pytest.mark.parametrize("listing_denied", [False, True])
async def test_a_launch_into_home_builds_the_skill_trust_its_wake_captures(
    tmp_path, monkeypatch, listing_denied
):
    """TASK-41507: Library builds the skill trust service as it starts, which
    hid that a launch wake captures its configuration before anything else
    does. Home builds none, so the capture found the slot empty and refused
    with ``console_capture_source_not_ready``; nothing reached the provider.

    ``listing_denied``: a server runtime source denies the local skill
    listing, which the capture never asks. A gate on it before the build
    refused this wake in a loop (r16co-rev1); it is still delivered.
    """
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError

    conversation_id, run_id, _rows = await _seed_a_finished_background_job(tmp_path)
    if listing_denied:
        _deny_local_skill_listing(monkeypatch)
    app, _marks, gateway = _launch_app(tmp_path, real_service=True, tab="home")
    assert vars(app)["_local_skill_trust_service"] is None, "precondition: cold"
    if listing_denied:  # the denial is live for this app's own scope
        with pytest.raises(PolicyDeniedError):
            app.skills_scope_service._enforce_policy("skills.context.list.local")

    async with app.run_test(size=(120, 40)) as pilot:
        delivered = await _settle(pilot, lambda: bool(gateway.payloads))
        _assert_console_never_mounted(app)
        assert delivered, "the cold launch delivered no wake turn"
        assert CHILD_RESULT in str(gateway.payloads[-1][-1]["content"])
        runs_db = app.console_runtime.agent_bridge.runs_db
        assert await _settle(
            pilot, lambda: (runs_db.get_run(run_id) or {}).get("wake_delivered_at")
        ), f"the wake for {conversation_id} was never stamped delivered"


@pytest.mark.asyncio
async def test_a_wake_that_fails_before_its_claim_retries_after_the_delay(
    tmp_path, monkeypatch
):
    """Review r16co-rev1: a delivery that raised before choosing a chain was
    "paused" with no chain, which saved nothing, blocked nothing and cleared
    its retry deadline, so ``retry_soon`` ran it again at once (5,239
    deliveries in 22 s). It now waits ``RETRY_DELAY_SECONDS`` each time."""
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

    await _seed_a_finished_background_job(tmp_path)
    tries = []

    async def fail_before_the_claim(_app):
        tries.append(time.monotonic())
        raise RuntimeError("forced before the claim")

    monkeypatch.setattr(preparation, "publish_cold_skill_trust", fail_before_the_claim)
    app, _marks, gateway = _launch_app(tmp_path, real_service=True, tab="home")

    async with app.run_test(size=(120, 40)) as pilot:
        assert await _settle(pilot, lambda: bool(tries)), "no delivery was tried"
        window = 2.5
        await _quiet(pilot, lambda: False, seconds=window)
        delay = ConsoleFleetWakeCoordinator.RETRY_DELAY_SECONDS
        gaps = [later - earlier for earlier, later in zip(tries, tries[1:], strict=False)]
        assert 2 <= len(tries) <= window / delay + 2, f"{len(tries)} tries in {window} s"
        assert all(gap >= 0.9 * delay for gap in gaps), gaps
        assert not gateway.payloads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("slot", "factory", "deny", "events", "built"),
    [
        ("ready", True, False, [], "ready"),
        (None, False, False, [], None),
        (None, True, False, ["build"], "built"),
        (None, True, True, ["build"], "built"),
    ],
)
async def test_the_cold_trust_build_runs_only_where_capture_refuses(
    slot, factory, deny, events, built
):
    """TASK-41507: the shared build mirrors ``_ready_attribute`` (an empty slot
    with a factory) and any other app is left alone. A denied local skill
    listing (a server runtime source) still builds: the capture never asks
    that policy (r16co-rev1)."""
    from tldw_chatbook.Chat.console_configuration_preparation import (
        publish_cold_skill_trust,
    )

    seen = []

    class Local:
        def __init__(self):
            self._trust_service = slot
            self._trust_service_factory = object() if factory else None

        @property
        def trust_service(self):
            return self._trust_service

        @trust_service.setter
        def trust_service(self, value):
            self._trust_service = value

    class Scope:
        local_service = Local()

        def _enforce_policy(self, action_id):
            seen.append(action_id)
            if deny:
                raise PermissionError(action_id)

    class App:
        def __init__(self):
            self._skills_scope_service = Scope()  # the helper reads vars(app)

        async def ensure_local_skill_trust_service(self):
            seen.append("build")
            return "built"

    app = App()
    await publish_cold_skill_trust(app)
    assert seen == events
    assert app._skills_scope_service.local_service._trust_service == built
    await publish_cold_skill_trust(None)  # no app: nothing to build


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
    conversation_id, _run_id, _rows = await _seed_a_finished_background_job(tmp_path)

    app1, marks1, gateway1 = _launch_app(tmp_path, real_service=True)
    async with app1.run_test(size=(120, 40)) as pilot:
        assert await _settle(pilot, lambda: bool(gateway1.payloads)), (
            "precondition: the first launch must deliver"
        )
        assert await _settle(
            pilot,
            lambda: (
                not app1.console_runtime.chat_controller.fleet_wake.delivering_conversation_ids()
            ),
        )
        assert marks1.has_mark(conversation_id, FLEET_UNSEEN)
    # Shutdown may clear the view mark; seed it again to prove that attention
    # alone cannot replay a result whose exact delivery receipt already exists.
    marks1.set_mark(conversation_id, FLEET_UNSEEN)

    app2, _marks2, gateway2 = _launch_app(tmp_path, real_service=True)
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
    with _fixture_runs(tmp_path / "agent_runs.db", client_id="seed-2") as runs_db:
        _terminal_subagent_run(runs_db, conversation_id, result="second child result")
    app3, _marks3, gateway3 = _launch_app(tmp_path, real_service=True)
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


@pytest.mark.asyncio
async def test_a_launch_into_console_delivers_without_stealing_the_active_tab(
    tmp_path,
    monkeypatch,
):
    """The OTHER launch shape, and the shipped default: `default_tab` is
    Console, so the startup screen IS a `ChatScreen` with its own fresh
    session.

    The owed wake must still be delivered -- the User Guide's promise is
    "at the next start", not "at the next start if you land somewhere
    else" -- and it must NOT move the user off the tab they landed on.
    `restore_persisted_session` activates what it creates, so the second
    half is a real hazard rather than a hypothetical.
    """
    conversation_id, _run_id, rows = await _seed_a_finished_background_job(tmp_path)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    app = _build_test_app("chat")
    app.app_config.pop("_first_run", None)
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    marks = _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    gateway = _StallingWakeGateway(reply=LAUNCH_REPLY)
    app.console_provider_gateway_factory = lambda: gateway
    app._wire_chat_conversation_services()

    async with app.run_test(size=(160, 48)) as pilot:
        assert await _settle(
            pilot, lambda: any(isinstance(s, ChatScreen) for s in app.screen_stack)
        ), (
            "harness precondition: this test is about launching INTO Console; "
            f"stack is {[type(s).__name__ for s in app.screen_stack]}"
        )
        store = app.console_runtime.chat_store
        assert await _settle(pilot, lambda: store.active_session_id is not None), (
            "harness precondition: the Console mount must have made a session "
            "active, or there is no tab to steal"
        )
        landed_on = store.active_session_id
        assert await _settle(pilot, lambda: bool(gateway.payloads)), (
            "launching INTO Console delivered no owed wake at all -- the "
            "shipped default tab would get nothing"
        )
        assert CHILD_RESULT in str(gateway.payloads[-1][-1]["content"])
        assert store.active_session_id == landed_on, (
            "the launch wake moved the user off the tab they landed on: "
            f"{landed_on} -> {store.active_session_id}"
        )
        assert await _settle(
            pilot,
            lambda: (
                not app.console_runtime.chat_controller.fleet_wake.delivering_conversation_ids()
            ),
        )
        assert marks.has_mark(conversation_id, FLEET_UNSEEN), (
            "the wake landed in a tab the user was not viewing, so the ◈ mark "
            "must survive to point them at it"
        )


@pytest.mark.asyncio
async def test_a_launch_built_controller_is_not_sticky_when_console_opens(tmp_path):
    """The hazard this task introduces, pinned.

    `ConsoleRuntime.ensure_chat_controller` is idempotent and IGNORES its
    parameters after the first call. A launch now makes that first call --
    with config defaults, because there is no control bar to read -- so if
    `ChatScreen` did not re-apply its own selection on every
    `_ensure_console_chat_controller()`, a user whose launch happened to
    wake a supervisor would run the whole session on the launch's guesses.

    Executed rather than read: the config's model is CHANGED between the
    launch and the Console mount, and the controller must end up on the new
    one.
    """
    conversation_id, _run_id, _rows = await _seed_a_finished_background_job(tmp_path)
    app, _marks, gateway = _launch_app(tmp_path, real_service=True)

    async with app.run_test(size=(160, 48)) as pilot:
        assert await _settle(pilot, lambda: bool(gateway.payloads)), (
            "precondition: the launch must have built the controller"
        )
        controller = app.console_runtime.chat_controller
        assert controller is not None
        _assert_console_never_mounted(app)

        # The user changes their model in Settings, then opens Console.
        # `configured_model` is derived from `[api_settings.<provider>]`,
        # not `[chat_defaults]` -- the first draft changed the wrong key and
        # the test's own vacuity guard caught it.
        _configure_native_ready_console(app, "mounted-model")
        chat = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert isinstance(chat, ChatScreen)
        await pilot.pause()
        assert controller is app.console_runtime.chat_controller, (
            "harness precondition: the mount must REUSE the launch-built "
            "controller, or this test is not about stickiness at all"
        )
        expected = chat._build_console_provider_selection()
        assert await _settle(
            pilot,
            lambda: controller.configured_model == expected.configured_model,
        ), (
            "the launch-built controller kept the launch's model after Console "
            f"opened: controller={controller.configured_model!r} "
            f"screen={expected.configured_model!r}"
        )
        assert expected.configured_model == "mounted-model", (
            "the fixture never changed anything, so the comparison above is "
            f"vacuous: {expected.configured_model!r}"
        )
        del conversation_id


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
            assert calls == [], (
                "durable wake discovery must not treat attention marks as authority"
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


@pytest.mark.asyncio
async def test_quit_settles_an_in_flight_launch_wake_before_retiring_agent_runs(
    tmp_path, monkeypatch
):
    """TASK-42028: a quit cancels the launch wake's delivery mid-turn, and the
    delivery's `finally` still settles its attempt in AgentRuns. Disposal
    used to drop that task and retire the file under it, so the retirement
    refused (`sqlite_retirement_incomplete:AgentRuns`) or beat the
    settlement. The settlement is held 0.5 s so the race cannot be won by
    luck; the quit must wait for it and then retire cleanly."""
    from tldw_chatbook.Chat import console_activity_receipts
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

    conversation_id, _run_id, rows = await _seed_a_finished_background_job(tmp_path)
    app, _marks, gateway = _launch_app(
        tmp_path, tree=_fixture_tree(conversation_id, rows)
    )
    gateway.stall_stream = True
    settle = ConsoleFleetWakeCoordinator._finish_attempt

    async def held_settle(self, authorization, returned):
        await asyncio.sleep(0.5)
        await settle(self, authorization, returned)

    monkeypatch.setattr(ConsoleFleetWakeCoordinator, "_finish_attempt", held_settle)
    retire = console_activity_receipts.retire_agent_runs_db
    deliveries: tuple = ()
    unsettled_at_retire: list = []

    async def recording_retire(database):
        unsettled_at_retire.append([task for task in deliveries if not task.done()])
        return await retire(database)

    monkeypatch.setattr(
        console_activity_receipts, "retire_agent_runs_db", recording_retire
    )
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _settle(pilot, gateway.entered_stream.is_set)
        deliveries = tuple(
            app.console_runtime.chat_controller.fleet_wake._delivery_tasks
        )
        assert deliveries, "precondition: the wake's turn is in flight at quit"
    assert unsettled_at_retire and unsettled_at_retire[0] == [], (
        "the quit retired AgentRuns while the cancelled wake delivery was "
        f"still settling its attempt: {unsettled_at_retire!r}"
    )
    app._console_runtime_shutdown_task.result()


@pytest.mark.asyncio
async def test_quit_settles_a_wake_delivery_a_session_close_cancelled_before_retiring_agent_runs(
    tmp_path, monkeypatch
):
    """TASK-42074: closing the session fences and cancels the launch wake's
    delivery, and `fence_conversation` drops it from `_delivery_tasks`, so
    the quit's drain (TASK-42028) never saw it and retired AgentRuns under
    its settlement. The settlement is held until the quit starts, plus
    0.5 s; the quit must wait for it."""
    from tldw_chatbook.Chat import console_activity_receipts
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

    conversation_id, _run_id, rows = await _seed_a_finished_background_job(tmp_path)
    app, _marks, gateway = _launch_app(
        tmp_path, tree=_fixture_tree(conversation_id, rows)
    )
    gateway.stall_stream = True
    quit_started = asyncio.Event()
    dispose = ConsoleFleetWakeCoordinator.dispose
    settle = ConsoleFleetWakeCoordinator._finish_attempt

    def recording_dispose(self):
        quit_started.set()
        dispose(self)

    async def held_settle(self, authorization, returned):
        await quit_started.wait()
        await asyncio.sleep(0.5)
        await settle(self, authorization, returned)

    monkeypatch.setattr(ConsoleFleetWakeCoordinator, "dispose", recording_dispose)
    monkeypatch.setattr(ConsoleFleetWakeCoordinator, "_finish_attempt", held_settle)
    retire = console_activity_receipts.retire_agent_runs_db
    deliveries: tuple = ()
    unsettled_at_retire: list = []

    async def recording_retire(database):
        unsettled_at_retire.append([task for task in deliveries if not task.done()])
        return await retire(database)

    monkeypatch.setattr(
        console_activity_receipts, "retire_agent_runs_db", recording_retire
    )
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _settle(pilot, gateway.entered_stream.is_set)
        runtime = app.console_runtime
        controller = runtime.chat_controller
        wake = controller.fleet_wake
        deliveries = tuple(wake._delivery_tasks)
        (session_id,) = wake.delivering_session_ids()
        await runtime.close_session(
            session_id,
            expected_revision=controller.lifecycle_impact(
                session_id=session_id
            ).revision,
        )
        assert deliveries and not wake._delivery_tasks, (
            "precondition: the close fenced the delivery and dropped it"
        )
        assert not any(task.done() for task in deliveries), (
            "precondition: the cancelled delivery is still settling at quit"
        )
    assert unsettled_at_retire and unsettled_at_retire[0] == [], (
        "the quit retired AgentRuns while the wake delivery a session close "
        f"cancelled was still settling its attempt: {unsettled_at_retire!r}"
    )
    app._console_runtime_shutdown_task.result()


@pytest.mark.asyncio
async def test_session_fence_keeps_only_a_weak_handle_on_a_cancelled_delivery():
    """TASK-42074: quit can still join a delivery a session close cancelled,
    but the fence never keeps that delivery alive (see
    `test_session_fence_releases_uncooperative_delivery_task_reference`)."""
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

    coordinator = ConsoleFleetWakeCoordinator(SimpleNamespace())
    task = asyncio.create_task(asyncio.Event().wait())
    coordinator._delivery_tasks[task] = "closing"

    coordinator.fence_conversation("closing", generation=1)

    assert set(coordinator._fenced_delivery_tasks) == {task}
    with pytest.raises(asyncio.CancelledError):
        await task
    handle = weakref.ref(task)
    del task
    await asyncio.sleep(0)  # drop the loop handle that resumed this await
    gc.collect()
    assert handle() is None, "the fence kept a strong reference to its delivery"


@pytest.mark.asyncio
async def test_maintenance_drain_waits_for_a_delivery_a_session_close_cancelled():
    """TASK-42152: the session fence drops the delivery it cancels from
    `_delivery_tasks`, and the delivery still settles its attempt from its
    `finally`. A backup's maintenance drain must wait for that before it
    reports drained. One that never settles still ends the drain False at its
    deadline, and the drain keeps no reference to it."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    release = asyncio.Event()
    stop = []

    async def settles_when_released():
        try:
            await asyncio.Event().wait()
        finally:
            await release.wait()

    async def never_settles():
        while not stop:  # the test's teardown ends it after a failure
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                continue

    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    wake = controller.fleet_wake
    try:
        controller.maintenance_close_admission()
        settling = asyncio.create_task(settles_when_released())
        await asyncio.sleep(0)
        wake._delivery_tasks[settling] = "closing"
        wake.fence_conversation("closing", generation=1)
        await asyncio.sleep(0)
        assert not wake._delivery_tasks and not settling.done(), (
            "precondition: the fence dropped the delivery, which is still settling"
        )
        drain = asyncio.create_task(
            controller.maintenance_drain(time.monotonic() + 5)
        )
        await asyncio.sleep(0.2)
        assert not drain.done(), (
            "the drain reported drained while the delivery a session close "
            "cancelled was still settling its attempt"
        )
        release.set()
        assert await drain is True
        assert settling.done()

        stuck = asyncio.create_task(never_settles())
        await asyncio.sleep(0)
        wake._delivery_tasks[stuck] = "stuck"
        wake.fence_conversation("stuck", generation=1)
        assert await controller.maintenance_drain(time.monotonic() + 0.1) is False
        handle = weakref.ref(stuck)
        del stuck
        await asyncio.sleep(0)  # drop the loop handle that resumed this await
        gc.collect()
        assert handle() is None, "the drain kept a strong reference to the delivery"
    finally:
        stop.append(True)
        release.set()
        await controller.shutdown()


# ---------------------------------------------------------------------------
# AC#3's phantom-wake case, at the launch fire point.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_crash_killed_child_swept_to_error_wakes_nobody_at_launch(tmp_path):
    """A child left `running` by a crash is swept to `error` by the next
    `AgentRunsDB.__init__`, which makes it terminal, undelivered and
    therefore OWED by the ledger's own definition -- and it carries no
    mark, because nothing ever settled it through the fan-out.

    Since 827b622638 launch discovery reads the ledger, not the marks ("a
    missing badge never hides a pending result"), so the orphan is found and
    the Console runtime is built for it. Its result has no work chain, so it
    is paused for manual review (`legacy_lineage`) and nothing fires. The
    assertion that stops this passing for the wrong reason:
    `undelivered_wake_runs` genuinely DOES report the orphan, so the silence
    is the review pause's doing, not an empty ledger (TASK-39044).
    """
    app0 = _build_test_app("library")
    _attach_real_dbs(app0, tmp_path)
    app0.chachanotes_db.add_conversation({"id": "conv-crashed", "title": "Crashed"})
    with _fixture_runs(tmp_path / "agent_runs.db", client_id="seed") as runs_db:
        parent_id = runs_db.create_run(
            conversation_id="conv-crashed", agent_kind="primary"
        )
        runs_db.set_status(parent_id, "done", "turn final")
        child_id = runs_db.create_run(
            conversation_id="conv-crashed",
            agent_kind="subagent",
            task="killed mid-flight",
            parent_run_id=parent_id,
        )
    # The restart sweep: a fresh AgentRunsDB over the same file, with the
    # per-process guard discarded the way Tests/DB/test_agent_runs_db.py
    # models a restart.
    AgentRunsDB._swept_paths.discard(str(tmp_path / "agent_runs.db"))
    with _fixture_runs(tmp_path / "agent_runs.db", client_id="sweep") as swept:
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
        wake = app.console_runtime.chat_controller.fleet_wake
        assert wake.pause_reason("conv-crashed") == "legacy_lineage", (
            "a chainless crash-killed result must wait for manual review"
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
    conversation_id, run_id, _rows = await _seed_a_finished_background_job(tmp_path)

    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    app_off, marks_off, gateway_off = _launch_app(tmp_path, real_service=True)
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
    with _fixture_runs(tmp_path / "agent_runs.db", client_id="check") as off_runs:
        assert not (off_runs.get_run(run_id) or {}).get("wake_delivered_at"), (
            "the kill switch stamped the ledger for a wake it never delivered"
        )

    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
    app_on, _marks_on, gateway_on = _launch_app(tmp_path, real_service=True)
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
    with _fixture_runs(tmp_path / "agent_runs.db", client_id="seed") as runs_db:
        _parent, run_id = _terminal_subagent_run(
            runs_db, ephemeral_id, result="work nobody can be shown"
        )
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
    with _fixture_runs(tmp_path / "agent_runs.db", client_id="check") as check:
        assert not (check.get_run(run_id) or {}).get("wake_delivered_at"), (
            "clearing an unresolvable mark must not stamp the ledger -- the "
            "result was never delivered to anyone"
        )


@pytest.mark.asyncio
async def test_a_mark_with_nothing_owed_is_left_alone_at_launch(tmp_path):
    """A ◈ mark with no owed ledger row is NOT stale: it is the
    delivered-but-unseen badge the wake sets when it delivers off-view
    (task-15971). A launch must not deliver anything for it, and must not
    clear it -- the user has still not seen that result."""
    app0 = _build_test_app("library")
    marks0 = _attach_real_dbs(app0, tmp_path)
    app0.chachanotes_db.add_conversation(
        {"id": "conv-seen-later", "title": "Delivered"}
    )
    marks0.set_mark("conv-seen-later", FLEET_UNSEEN)

    app, marks, gateway = _launch_app(
        tmp_path,
        tree={
            "conv-seen-later": {
                "conversation": {"id": "conv-seen-later", "title": "Delivered"},
                "root_threads": [
                    {"id": "m1", "sender": "user", "content": "earlier work"}
                ],
            }
        },
    )
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _quiet(pilot, lambda: bool(gateway.payloads), seconds=4.0), (
            f"a launch delivered a wake for a conversation owing none: {gateway.payloads!r}"
        )
        assert marks.has_mark("conv-seen-later", FLEET_UNSEEN), (
            "the launch cleared a delivered-but-unseen ◈ badge -- the user "
            "loses the only pointer they had to a result they never saw"
        )
        store = app.console_runtime.chat_store
        opened = (
            []
            if store is None
            else [s.persisted_conversation_id for s in store.sessions()]
        )
        assert "conv-seen-later" not in opened, (
            "the launch hydrated a session for a conversation owing nothing -- "
            "the user opens Console to an unexplained tab, and the work was "
            f"pointless: {opened}"
        )


@pytest.mark.asyncio
async def test_a_launch_hydrates_only_the_conversations_that_are_owed(tmp_path):
    """Two marks, one owed. The owed one is hydrated and delivered; the
    other is not opened at all and its receipt-free stale mark is reconciled.

    This is the case that makes the per-conversation `has_pending` guard
    load-bearing rather than decorative: with a SINGLE unowed mark the
    launch returns before the loop, so removing that guard changes nothing
    observable.
    """
    conversation_id, _run_id, rows = await _seed_a_finished_background_job(tmp_path)
    app0 = _build_test_app("library")
    marks0 = _attach_real_dbs(app0, tmp_path)
    app0.chachanotes_db.add_conversation({"id": "conv-quiet", "title": "Quiet"})
    marks0.set_mark("conv-quiet", FLEET_UNSEEN)

    tree = _fixture_tree(conversation_id, rows)
    tree["conv-quiet"] = {
        "conversation": {"id": "conv-quiet", "title": "Quiet"},
        "root_threads": [{"id": "q1", "sender": "user", "content": "quiet work"}],
    }
    app, marks, gateway = _launch_app(tmp_path, tree=tree)
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _settle(pilot, lambda: bool(gateway.payloads)), (
            "precondition: the OWED conversation must still be delivered"
        )
        opened = [
            s.persisted_conversation_id
            for s in app.console_runtime.chat_store.sessions()
        ]
        assert conversation_id in opened, opened
        assert "conv-quiet" not in opened, (
            "a marked-but-unowed conversation was hydrated alongside an owed "
            f"one: {opened}"
        )
        assert not marks.has_mark("conv-quiet", FLEET_UNSEEN), (
            "receipt hydration must remove a stale compatibility mark that "
            "has neither an unseen receipt nor an owed wake"
        )
