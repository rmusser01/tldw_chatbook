"""The headless-wake invariants, proven TOGETHER on current dev (task-15860, plan Task 7).

Each of AC#3's four invariants was asserted somewhere while the arc was
landing -- but each against the tree that existed at ITS merge-base, and
three of the four at a seam rather than at the surfaces a user's durable
state actually lives on. Eight landings later this file re-proves them on
one tree, and closes the three gaps the per-landing coverage left:

* **No USER transcript row, asserted on the DB rows.** The nav-away path
  pins this (`Tests/UI/test_console_headless_wake_fires.py`) and so does
  the launch path (`Tests/UI/test_console_launch_wake.py`). Neither pins
  it for the case where the wake was *withheld* and then released by the
  kill switch -- the path that writes the rows LAST and through a
  different fire point (`retry_soon`, not `on_fleet_drained`).
* **`autowake_enabled = false` loses nothing DURABLE.** The existing
  kill-switch tests assert the registry, the ◈ mark and the ledger. None
  asserts the conversation's persisted rows -- the thing a user would
  actually lose.
* **Deliveries stay serialized app-wide.** The runtime is app-owned now,
  so there is one `_delivering` for one app. The failure this guards
  against is structural (a per-screen coordinator starting a second,
  concurrent wake turn), so it is asserted across a real screen
  REPLACEMENT: leave Console mid-delivery, come back to a brand-new
  `ChatScreen`, and prove the fresh screen neither reset the flag nor
  started a second turn.
* **Exactly-once across a restart mid-commit.** `_deliver` stamps the
  ledger *after* `submit_draft` returns accepted, so there is a real
  window where the rows are committed and the ledger is not. Every
  existing exactly-once test restarts either side of that window. This
  one restarts INSIDE it and measures what the user gets.

Rig: the real app, real on-disk ChaChaNotes + `agent_runs` DBs, the real
navigation API, and the production wake chain
(`on_fleet_drained` -> `_attempt` -> `_deliver` -> `submit_draft`). No
gate is monkeypatched.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_fleet_wake import _drain, _quiet, _settle, _survivor
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_store_continuity import (
    CHILD_RESULT,
    SEEDED_USER,
    _StallingWakeGateway,
    _db_chain,
    _drain_from_child_thread,
    _navigate,
    _seed_console,
    _terminal_survivor_run,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_fleet_wake import WAKE_NOTICE_HEADER
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from Tests.UI.test_console_launch_wake import (
    _assert_console_never_mounted,
    _launch_app,
    _quiet as _launch_quiet,
    _settle as _launch_settle,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from Tests.UI.app_factory import attach_chachanotes_db


WAKE_REPLY = "GATE-WAKE-REPLY"
SECOND_RESULT = "the second child's finished answer"


class _CountingStallGateway(_StallingWakeGateway):
    """`_StallingWakeGateway` that also counts readiness-probe ENTRIES.

    Payload count cannot tell "a second wake turn was refused" from "a
    second wake turn started and parked at the same stall", because the
    stall is a property of the gateway, not of a turn. Measured: with
    `_attempt`'s `_delivering` guard mutated away, a second conversation's
    wake turn DOES start -- and streams nothing, because it parks at the
    probe exactly like the first. Counting entries is what distinguishes
    them, so it is what the serialization assertion reads.
    """

    def __init__(self, reply: str | None = None) -> None:
        super().__init__(*(() if reply is None else (reply,)))
        #: Every call into the readiness probe, stalled or not.
        self.probe_entries = 0

    async def resolve_for_send(self, selection):
        self.probe_entries += 1
        return await super().resolve_for_send(selection)


def _build_console_app(tmp_path):
    """A real app with real DBs and a recording provider gateway."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _CountingStallGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    return app, gateway


def _senders(app, conversation_id):
    return [row[1] for row in _db_chain(app.chachanotes_db, conversation_id)]


# ---------------------------------------------------------------------------
# Invariants 1 + 3: the kill switch silences the headless fire point, loses
# nothing durable, and what it releases still writes no USER row.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_autowake_off_silences_the_headless_fire_point_and_loses_no_rows(
    tmp_path, monkeypatch
):
    """AC#3, on the two surfaces the user actually keeps.

    OFF: the survivor settles with no Console mounted and the conversation's
    PERSISTED rows do not move -- not the registry, not the mark, the rows.
    ON (same live coordinator, no restart): what OFF recorded is delivered,
    and the rows it finally writes are a SYSTEM notice and an ASSISTANT
    reply. Not a USER row, on the DB, on the release path.
    """
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    app, gateway = _build_console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_id = _terminal_survivor_run(runs_db, conversation_id)
        marks = app.conversation_local_marks_service
        marks.set_mark(conversation_id, ConversationLocalMarksService.FLEET_UNSEEN)
        rows_before = _db_chain(app.chachanotes_db, conversation_id)
        assert len(rows_before) == 2, (
            "harness precondition: the seeded turn must have persisted two rows; "
            f"got {[(r[1], r[2][:24]) for r in rows_before]}"
        )

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack, "Console must actually unmount"
        assert controller is app.console_runtime.chat_controller, (
            "harness precondition: the runtime must OUTLIVE the screen"
        )
        assert controller._disposed is False, (
            "harness precondition: a navigation is not an app exit"
        )

        # -- OFF ----------------------------------------------------------
        gateway.reply = WAKE_REPLY
        # The seeding send already streamed once, so every payload
        # assertion below is against THIS baseline, never against zero.
        before = len(gateway.payloads)
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
        )
        assert await _quiet(lambda: len(gateway.payloads) > before), (
            "autowake_enabled=false must silence the HEADLESS fire point too"
        )
        assert _db_chain(app.chachanotes_db, conversation_id) == rows_before, (
            "the silenced wake still wrote to the conversation: "
            f"{[(r[1], r[2][:28]) for r in _db_chain(app.chachanotes_db, conversation_id)]}"
        )
        # ...and nothing durable was lost while OFF.
        assert wake.has_pending(conversation_id), "OFF still records the completion"
        assert marks.has_mark(
            conversation_id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "OFF keeps the ◈ indicator working"
        assert not (runs_db.get_run(run_id) or {}).get("wake_delivered_at"), (
            "OFF stamped the ledger for a wake that never ran"
        )
        assert wake.seed_from_marks() == 0, "OFF seeds nothing at the mount claim"

        # -- ON, same coordinator, still headless -------------------------
        monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
        wake.retry_soon()
        assert await _settle(lambda: len(gateway.payloads) > before, seconds=10.0), (
            "flipping the kill switch ON did not deliver what OFF recorded"
        )
        assert await _settle(
            lambda: len(_db_chain(app.chachanotes_db, conversation_id)) == 4,
            seconds=10.0,
        ), (
            "the released wake never PERSISTED: "
            f"{[(r[1], r[2][:28]) for r in _db_chain(app.chachanotes_db, conversation_id)]}"
        )

        # (1) no USER row on the DB, on the RELEASE path.
        senders = _senders(app, conversation_id)
        assert senders == ["user", "assistant", "system", "assistant"], (
            f"unexpected persisted row shape after the release: {senders}"
        )
        assert senders.count("user") == 1, (
            f"the released headless wake persisted a USER row: {senders}"
        )
        rows = _db_chain(app.chachanotes_db, conversation_id)
        assert rows[2][2].startswith(WAKE_NOTICE_HEADER)
        assert rows[3][2] == WAKE_REPLY
        store_user_rows = [
            m
            for m in store.messages_for_session(session_id)
            if m.role is ConsoleMessageRole.USER
        ]
        assert [m.content for m in store_user_rows] == [SEEDED_USER], (
            f"the released headless wake wrote a USER transcript row: {store_user_rows!r}"
        )
        assert await _settle(
            lambda: bool((runs_db.get_run(run_id) or {}).get("wake_delivered_at"))
        ), "the released delivery never stamped the ledger"


# ---------------------------------------------------------------------------
# Invariant 4: one `_delivering` for one APP, across a screen replacement.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_delivery_at_a_time_app_wide_across_a_screen_replacement(
    tmp_path,
):
    """AC#3's serialization half, at the structure that used to break it.

    Ownership was per-screen: each `ChatScreen` built its own controller,
    so its own `ConsoleFleetWakeCoordinator`, so its own `_delivering`.
    Two screens over one conversation could each hold a wake turn.

    Driven here by consequence, with a wake genuinely in flight:

    1. a wake turn parks at the provider readiness probe (`_delivering`
       set, nothing written yet);
    2. Console is left and RE-ENTERED through the real navigation API --
       a brand-new `ChatScreen`, which runs the mount claim
       (`seed_from_marks`) and a `retry_soon` of its own;
    3. a SECOND completion settles into the same conversation while the
       first is still parked;
    4. the probe is released.

    A per-screen coordinator delivers twice (the fresh screen's flag is
    `None`). One app-owned coordinator delivers once, then picks the
    second completion up on the next attempt -- and both children's
    results are accounted for, so serializing is not the same as losing.

    Honesty note (measured): the second completion here lands in the SAME
    conversation, so `_attempt` refuses it at the per-session busy gate
    before it ever reaches the `_delivering` check -- mutating
    `_delivering` away leaves this test green. The APP-WIDE half of the
    invariant, the one only `_delivering` can enforce, is a SECOND
    conversation whose own session is idle; that is
    `test_a_second_conversations_wake_waits_while_another_is_delivering`
    below, and it is the one that dies under that mutation. This test
    owns the structural half: one runtime, one coordinator, one flag,
    across a real screen replacement.
    """
    app, gateway = _build_console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_one = _terminal_survivor_run(runs_db, conversation_id)
        marks = app.conversation_local_marks_service
        marks.set_mark(conversation_id, ConversationLocalMarksService.FLEET_UNSEEN)

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack, "Console must actually unmount"

        # (1) park a real wake turn at the readiness probe.
        gateway.reply = WAKE_REPLY
        gateway.stall = True
        # The seeding send already streamed once: every payload assertion
        # below counts from THIS baseline, never from zero.
        before = len(gateway.payloads)
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_one, session_id=session_id))
        )
        assert await _settle(lambda: gateway.entered_stall.is_set(), seconds=10.0), (
            "the headless wake never reached the provider readiness probe"
        )
        assert wake.delivering_conversation_id() == conversation_id, (
            "the coordinator does not consider a delivery in flight, so the "
            "serialization this test asserts would be vacuous"
        )

        # (2) a real screen REPLACEMENT while that delivery is in flight.
        chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert isinstance(chat2, ChatScreen), type(chat2).__name__
        assert chat2 is not chat, "screens are never cached"
        controller2 = chat2._ensure_console_chat_controller()
        assert controller2 is controller, (
            "the new screen built its OWN controller -- ownership is not app-wide"
        )
        assert controller2.fleet_wake is wake, (
            "the new screen has its own wake coordinator, so it has its own "
            "`_delivering`: two screens could each run a wake turn"
        )
        assert wake.delivering_conversation_id() == conversation_id, (
            "mounting a fresh Console screen RESET the in-flight delivery flag"
        )

        # (3) a second child settles while the first turn is still parked.
        run_two = _terminal_survivor_run(
            runs_db, conversation_id, result=SECOND_RESULT
        )
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_two, session_id=session_id))
        )
        assert await _quiet(lambda: len(gateway.payloads) > before, seconds=1.0), (
            "a second wake turn reached the provider while one was in flight"
        )

        # (4) release: exactly one turn had been composed, and the second
        #     completion rides the next one rather than being lost.
        gateway.stall = False
        gateway.release.set()
        assert await _settle(
            lambda: len(gateway.payloads) > before, seconds=10.0
        ), "releasing the probe never delivered the parked wake"
        first_notice = str(gateway.payloads[before][-1]["content"])
        assert CHILD_RESULT in first_notice, first_notice[:200]

        assert await _settle(
            lambda: bool((runs_db.get_run(run_two) or {}).get("wake_delivered_at")),
            seconds=15.0,
        ), (
            "the completion that settled during a serialized delivery was "
            "never delivered at all -- serializing must defer, not drop"
        )
        assert (runs_db.get_run(run_one) or {}).get("wake_delivered_at"), (
            "the first delivery never stamped the ledger"
        )
        # Both results reached the supervisor, across however many turns
        # the serialization needed.
        every_notice = "\n".join(
            str(payload[-1]["content"]) for payload in gateway.payloads[before:]
        )
        assert CHILD_RESULT in every_notice and SECOND_RESULT in every_notice, (
            "a child's result never reached the supervisor: "
            f"{[p[-1]['content'][:60] for p in gateway.payloads]}"
        )
        assert wake.delivering_conversation_id() is None, (
            "the coordinator never cleared its in-flight flag"
        )


@pytest.mark.asyncio
async def test_a_second_conversations_wake_waits_while_another_is_delivering(
    tmp_path,
):
    """The APP-WIDE half: one `_delivering`, two conversations.

    The per-session busy gate cannot serialize this case -- conversation
    B's session is idle the whole time. Only `_attempt`'s `_delivering`
    check stands between a wake in flight for A and a second, concurrent
    wake turn for B on the same app-owned runtime. With ownership still
    per-screen there was no single flag to hold, which is precisely what
    "serialized app-wide" had to become true of.

    Mutation-tested: neutering `if self._delivering is not None: return`
    makes the "no second wake turn" assertion below fail.
    """
    app, gateway = _build_console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_a, conversation_a = await _seed_console(
            app, pilot, gateway
        )
        # A second Console session, persisted to its own conversation.
        # `settings=` is required: the screen's provider-selection path
        # raises `Unknown Console session` for a session created without
        # them (measured), which has nothing to do with the wake.
        session_b_obj = store.create_session(
            title="Second",
            settings=chat._session._default_console_session_settings(),
        )
        session_b = session_b_obj.id
        outcome = await controller.submit_draft(
            "second conversation user message", session_id=session_b
        )
        assert outcome.accepted, "harness precondition: session B must send"
        conversation_b = next(
            s.persisted_conversation_id
            for s in store.sessions()
            if s.id == session_b
        )
        assert conversation_b and conversation_b != conversation_a, (
            "harness precondition: two DISTINCT persisted conversations; got "
            f"{conversation_a!r} and {conversation_b!r}"
        )

        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_a = _terminal_survivor_run(runs_db, conversation_a)
        run_b = _terminal_survivor_run(
            runs_db, conversation_b, result=SECOND_RESULT
        )
        marks = app.conversation_local_marks_service
        marks.set_mark(conversation_a, ConversationLocalMarksService.FLEET_UNSEEN)
        marks.set_mark(conversation_b, ConversationLocalMarksService.FLEET_UNSEEN)

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack, "Console must actually unmount"

        gateway.reply = WAKE_REPLY
        gateway.stall = True
        probes_before = gateway.probe_entries
        _drain_from_child_thread(
            wake, _drain(conversation_a, _survivor(run_a, session_id=session_a))
        )
        assert await _settle(lambda: gateway.entered_stall.is_set(), seconds=10.0), (
            "conversation A's headless wake never reached the readiness probe"
        )
        assert gateway.probe_entries == probes_before + 1, (
            f"expected exactly one wake probe; got {gateway.probe_entries - probes_before}"
        )
        assert wake.delivering_conversation_id() == conversation_a, (
            "no delivery is in flight, so this test would prove nothing"
        )

        # B settles now. Its own session is idle -- nothing but the
        # app-wide flag can hold it.
        _drain_from_child_thread(
            wake, _drain(conversation_b, _survivor(run_b, session_id=session_b))
        )
        assert await _quiet(
            lambda: gateway.probe_entries > probes_before + 1, seconds=1.5
        ), (
            "a second conversation's wake turn STARTED while one was already "
            "in flight -- deliveries are not serialized app-wide"
        )
        assert not (runs_db.get_run(run_b) or {}).get("wake_delivered_at"), (
            "conversation B's ledger row was stamped by a turn that never ran"
        )
        assert wake.has_pending(conversation_b), (
            "the deferred completion was dropped rather than held"
        )

        # Releasing A's probe lets both run, one after the other.
        gateway.stall = False
        gateway.release.set()
        assert await _settle(
            lambda: bool((runs_db.get_run(run_a) or {}).get("wake_delivered_at"))
            and bool((runs_db.get_run(run_b) or {}).get("wake_delivered_at")),
            seconds=20.0,
        ), (
            "serializing lost a delivery: A="
            f"{(runs_db.get_run(run_a) or {}).get('wake_delivered_at')!r} "
            f"B={(runs_db.get_run(run_b) or {}).get('wake_delivered_at')!r}"
        )
        assert len(_db_chain(app.chachanotes_db, conversation_a)) == 4, (
            f"conversation A: {_senders(app, conversation_a)}"
        )
        assert len(_db_chain(app.chachanotes_db, conversation_b)) == 4, (
            f"conversation B: {_senders(app, conversation_b)}"
        )
        assert _senders(app, conversation_b).count("user") == 1, (
            f"the second conversation's wake persisted a USER row: "
            f"{_senders(app, conversation_b)}"
        )
        assert wake.delivering_conversation_id() is None


# ---------------------------------------------------------------------------
# Invariant 2: exactly-once across a restart INSIDE the commit window.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_restart_mid_commit_never_re_announces_more_than_once(tmp_path):
    """AC#3's exactly-once bit, restarted INSIDE the window it depends on.

    `_deliver` commits in this order: `submit_draft` returns accepted (the
    notice and the reply are already appended AND persisted), and only
    then is `agent_runs.wake_delivered_at` stamped. A process that dies
    between those two leaves durable state the ledger cannot describe:
    rows written, ledger unstamped, ◈ mark set. Every other exactly-once
    test in this arc restarts either side of that window.

    Process one dies inside it (the stamp raises -- durable state is
    identical to the kill). Process two is a real launch with Console
    never opened. Process three is another. This test measures what the
    conversation actually ends up holding, and pins the bound: the window
    may cost at most ONE re-announce, never an unbounded loop, and never
    a lost result.

    **MEASURED on dev 524194c15 (2026-08-17): process two DOES re-announce.**
    The conversation ends on six rows -- `user, assistant, system,
    assistant, system, assistant` -- i.e. the same child result announced
    to the supervisor twice, and paid for twice. Process two's own stamp
    commits, so a third launch adds nothing; the cost of the window is
    bounded at one duplicate. `_deliver`'s own comment predicts exactly
    this ("a lost stamp risks one re-announce at a later claim, never a
    lost result"), so the behaviour is deliberate -- but the User Guide
    claimed the stronger thing ("a restart between a wake being accepted
    and the app exiting does not re-announce anything at the next
    launch"), which is false in this window and was corrected alongside
    this test. The assertions below deliberately accept EITHER outcome so
    that closing the window later is not a test failure; what they pin is
    the bound, the row shape, and that no USER row appears on any of it.
    """
    # -- process one: die between acceptance and the stamp -----------------
    app = _build_test_app()
    attach_chachanotes_db(app)
    marks = _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    gateway = _CountingStallGateway()
    app.console_provider_gateway_factory = lambda: gateway

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_id = _terminal_survivor_run(runs_db, conversation_id)
        marks.set_mark(conversation_id, ConversationLocalMarksService.FLEET_UNSEEN)

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack, "Console must actually unmount"

        def _die_mid_commit(_run_ids):
            raise RuntimeError("process died between acceptance and the stamp")

        runs_db.mark_wake_delivered = _die_mid_commit

        gateway.reply = WAKE_REPLY
        before = len(gateway.payloads)
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
        )
        assert await _settle(
            lambda: len(_db_chain(app.chachanotes_db, conversation_id)) == 4,
            seconds=15.0,
        ), (
            "the wake never got as far as committing its rows, so the crash "
            "window this test is about was never entered: "
            f"{_senders(app, conversation_id)}"
        )
        assert len(gateway.payloads) == before + 1
        assert not (runs_db.get_run(run_id) or {}).get("wake_delivered_at"), (
            "harness precondition: the stamp must NOT have committed -- "
            "otherwise this is the ordinary already-covered restart"
        )
        assert marks.has_mark(
            conversation_id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "harness precondition: the ◈ mark must survive the crash"

    # -- process two: a real launch over that durable state ----------------
    app2, marks2, gateway2 = _launch_app(tmp_path, real_service=True)
    async with app2.run_test(size=(120, 40)) as pilot2:
        # Give the launch path the same window the launch suite gives it.
        await _launch_settle(pilot2, lambda: bool(gateway2.payloads), 15.0)
        _assert_console_never_mounted(app2)
        rows_after_two = _db_chain(app2.chachanotes_db, conversation_id)
        runs_db2 = (
            app2.console_runtime.agent_bridge.runs_db
            if app2.console_runtime.agent_bridge is not None
            else None
        )
        stamped_after_two = (
            bool((runs_db2.get_run(run_id) or {}).get("wake_delivered_at"))
            if runs_db2 is not None
            else False
        )
        re_announced = len(rows_after_two) > 4
        assert len(rows_after_two) in (4, 6), (
            "a single mid-commit crash produced more than one re-announced "
            f"wake turn: {[(r[1], r[2][:28]) for r in rows_after_two]}"
        )
        if re_announced:
            # The honest cost of the window: one duplicate notice, and the
            # ledger closed this time so it cannot recur.
            assert _senders(app2, conversation_id) == [
                "user",
                "assistant",
                "system",
                "assistant",
                "system",
                "assistant",
            ], _senders(app2, conversation_id)
            assert stamped_after_two, (
                "the re-announcing launch did not close the ledger either, so "
                "the re-announce is UNBOUNDED, not a one-off"
            )
        assert _senders(app2, conversation_id).count("user") == 1, (
            "the mid-commit restart persisted a USER row: "
            f"{_senders(app2, conversation_id)}"
        )

    # -- process three: it must not recur ---------------------------------
    app3, _marks3, gateway3 = _launch_app(tmp_path, real_service=True)
    async with app3.run_test(size=(120, 40)) as pilot3:
        assert await _launch_quiet(pilot3, lambda: bool(gateway3.payloads), 3.0), (
            "a third launch re-announced again: the mid-commit window is not "
            "bounded at one re-announce"
        )
        _assert_console_never_mounted(app3)
        assert len(_db_chain(app3.chachanotes_db, conversation_id)) == len(
            rows_after_two
        ), (
            "a third launch grew the conversation: "
            f"{_senders(app3, conversation_id)}"
        )
