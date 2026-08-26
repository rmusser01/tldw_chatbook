"""The app-owned Console store is the continuity owner (task-15860, Task 3).

Console message history used to travel across a navigation as an in-memory
`ScreenStateStore` SNAPSHOT: `ChatScreen.save_state` serialized `sessions` +
`messages_by_session`, and the next Console mount rebuilt the store from that
payload without ever re-reading ChaChaNotes. Task 0's probes executed what
that costs once the runtime outlives the screen:

* **P1** -- rows appended to the conversation while Console was unmounted
  were absent from the restored transcript, absent from the next send's
  provider payload, and the next persisted append FORKED the tree away from
  them. Maintaining the durable active-leaf pointer changed nothing.
* **P3b** -- a wake turn that genuinely ran, spent money and stamped the
  ledger persisted four rows; the user returning to Console saw two. The
  wake notice was invisible.

The mechanism under test here is the one the 2026-07-11 UI-freeze incident
hardened, so this file carries its own rapid-switch soak that asserts the app
is still INTERACTIVE (a real keypress reaches a live widget, a real click
lands, the transcript repaints) after the churn -- not merely that nothing
raised.

Everything runs through the real navigation API
(`app.handle_screen_navigation(NavigateToScreen(...))`), the real `ChatScreen`,
a real on-disk ChaChaNotes DB and the production wake chain
(`on_fleet_drained` -> `_attempt` -> `_deliver` -> `submit_draft`). No gate is
monkeypatched: `_attempt`'s `_shutdown_requested` refusal is untouched, and the
headless half of the wake turn is produced the way production produces it
today -- a wake that starts while Console is mounted, stalls on the provider
readiness probe, and COMPLETES after the user has navigated away (the owner
ruling that `leave_console` never cancels an in-flight `AGENT_WAKE` turn).
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from loguru import logger

from Tests.Chat.test_console_fleet_wake import _drain, _settle, _survivor
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_fleet_wake import WAKE_NOTICE_HEADER
from tldw_chatbook.Chat.console_library_destination import (
    resolve_console_destination,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


SEEDED_USER = "first user message"
SEEDED_REPLY = "assistant one"
WAKE_REPLY = "WAKE-REPLY-ROW"
CHILD_RESULT = "the child's finished answer"


class _StallingWakeGateway:
    """A provider double whose READINESS PROBE can be held open.

    The wake's SYSTEM notice row is appended at the acceptance point --
    *after* `provider_gateway.resolve_for_send` returns
    (`console_chat_controller.py`, the `AGENT_WAKE` branch). Holding the
    probe therefore parks the whole wake turn before it has written
    anything, which is what lets a test navigate away and have BOTH wake
    rows land with no Console mounted. A cold llama.cpp probe is the
    everyday shape of this stall.
    """

    def __init__(self, reply: str = SEEDED_REPLY) -> None:
        self.payloads: list[list[dict]] = []
        self.reply = reply
        self.ready = True
        #: Set while the probe should block.
        self.stall = False
        #: Set by the probe when it starts blocking (the test's cue to leave).
        self.entered_stall = asyncio.Event()
        #: Released by the test to let the parked turn finish.
        self.release = asyncio.Event()

    async def resolve_for_send(self, selection):
        if self.stall:
            self.entered_stall.set()
            await self.release.wait()
        resolution = SimpleNamespace(
            ready=self.ready,
            provider="llama_cpp",
            model="test-model",
            base_url=None,
            visible_copy="" if self.ready else "WIP: provider warming up",
        )
        if resolution.ready:
            # TASK-21590 made the typed destination mandatory on a READY
            # resolution: `_resolved_destination_for_context` raises
            # ValueError without it, and the submit is refused with the
            # generic "Provider destination is incomplete." copy. Derive it
            # through the production classifier the real gateway uses rather
            # than hand-building one, so this double cannot drift from it.
            resolution.resolved_destination = resolve_console_destination(
                resolution
            )
        return resolution

    async def stream_chat(self, resolution, messages, **kwargs):
        self.payloads.append([dict(m) for m in messages])
        yield self.reply

    async def aclose(self) -> None:
        return None


def _drain_from_child_thread(wake, drain) -> None:
    """Deliver the drain the way production does: from a plain thread."""
    thread = threading.Thread(target=lambda: wake.on_fleet_drained(drain))
    thread.start()
    thread.join(5)


def _terminal_survivor_run(runs_db, conversation_id, *, result=CHILD_RESULT):
    """A sub-agent run that finished AFTER its (terminal) parent turn."""
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
    runs_db.set_status(run_id, "done", result)
    return run_id



async def _navigate(app, pilot, target: str, *, expect: str, timeout: float = 15.0):
    """Navigate through the real routing, answering Console's busy-fleet gate.

    `ChatScreen.confirm_navigation` opens a real "Leave Console?"
    `ConfirmationDialog` whenever the fleet is busy -- which a wake turn in
    flight makes it. Awaiting `handle_screen_navigation` directly therefore
    deadlocks the test against a dialog nobody answers (measured: the run
    hangs forever with the app's pump alive). Production's answer is a user
    pressing "Leave", so that is what this does -- posting the navigation the
    way the nav bar does and pressing the real button.
    """
    app.post_message(NavigateToScreen(target))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        screen = app.screen
        name = type(screen).__name__
        if name == "ConfirmationDialog":
            try:
                screen.query_one("#confirm-button", Button).press()
            except Exception:  # noqa: BLE001 -- the dialog may still be settling
                pass
            continue
        if name == expect:
            await pilot.pause()
            return screen
    raise AssertionError(
        f"navigating to {target!r} never reached {expect}; "
        f"stuck on {type(app.screen).__name__}"
    )


def _step(label: str) -> None:
    """Progress marker; a hang names its own step.

    Written through loguru, not `print`: Textual REDIRECTS stdout for the
    whole of `App.run_test()`, so a `print` from inside the pilot block is
    swallowed and a hang looks like it never started.
    """
    logger.warning("STEP {}", label)


async def _seed_console(app, pilot, gateway):
    """Mount a real Console, send once, return the live handles."""
    _step("seed: constructing screen")
    chat = ChatScreen(app)
    await app.push_screen(chat)
    _step("seed: pushed screen")
    app._initial_screen_pushed = True
    app.current_tab = "chat"
    await pilot.pause()
    await _wait_for_selector(chat, pilot, "#console-native-composer")
    _step("seed: composer present")
    controller = chat._ensure_console_chat_controller()
    store = chat._console_chat_store
    session_id = store.sessions()[0].id
    _step("seed: submitting")
    outcome = await controller.submit_draft(SEEDED_USER, session_id=session_id)
    _step("seed: submitted")
    assert outcome.accepted, "harness precondition: the seeding send must be accepted"
    conversation_id = store.sessions()[0].persisted_conversation_id
    assert conversation_id, "harness precondition: the conversation must PERSIST"
    return chat, controller, store, session_id, conversation_id


def _transcript_text(store, session_id: str) -> str:
    return "\n".join(m.content for m in store.messages_for_session(session_id))


def _rendered_text(screen) -> str:
    """Text from the transcript's MOUNTED row widgets.

    Deliberately NOT `ConsoleTranscript._messages`: that is the widget's
    model, assigned by `set_messages` before a single row is built, so an
    assertion against it proves the data arrived -- not that anything
    repainted. Measured, not assumed: with `_messages` in this helper, a
    mutation that removed `await transcript.refresh_messages()` outright
    left all four tests in this file GREEN. `_row_widgets` is what
    `_reconcile_rows` actually mounts, so it is what a "the user can see
    it" assertion has to read.
    """
    chunks: list[str] = []
    for transcript in screen.query(ConsoleTranscript):
        for row in transcript._row_widgets.values():
            for node in row.walk_children(with_self=True):
                renderable = getattr(node, "renderable", None)
                if renderable is not None:
                    chunks.append(str(renderable))
    return "\n".join(chunks)


def _db_chain(db, conversation_id):
    """The conversation's rows as (id, sender, content, parent) tuples."""
    return [
        (
            row["id"],
            row["sender"],
            row["content"],
            row.get("parent_message_id"),
        )
        for row in db.get_messages_for_conversation(conversation_id, limit=200)
    ]


async def _run_headless_wake_turn(app, pilot, gateway, tmp_path):
    """Drive P3b: a wake turn that COMPLETES with no Console mounted.

    Returns ``(chat2, store, session_id, conversation_id, runs_db, run_id)``
    with Console re-entered through the real navigation API.
    """
    chat, controller, store, session_id, conversation_id = await _seed_console(
        app, pilot, gateway
    )
    wake = controller.fleet_wake
    runs_db = controller._agent_bridge.runs_db
    run_id = _terminal_survivor_run(runs_db, conversation_id)

    # Park the next turn inside the readiness probe, then fire the survivor's
    # settle through the production fan-out.
    _step("wake: arming stall")
    gateway.stall = True
    gateway.reply = WAKE_REPLY
    _drain_from_child_thread(
        wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
    )
    _step("wake: drained, waiting for stall")
    assert await _settle(lambda: gateway.entered_stall.is_set()), (
        "the wake never reached the provider readiness probe while mounted"
    )

    # Leave Console. The wake turn is in flight and (owner ruling) survives.
    _step("nav: leaving console")
    await _navigate(app, pilot, "library", expect="LibraryScreen")
    _step("nav: left console")
    assert chat not in app.screen_stack, "Console must actually unmount"

    # ...and only NOW let the turn finish: both wake rows land headless.
    _step("wake: releasing provider")
    seeded_payloads = len(gateway.payloads)
    gateway.release.set()
    # Measure GROWTH, not truthiness: `_seed_console` already sent once, so
    # `gateway.payloads` is non-empty before the wake ever starts and a bare
    # truthiness check here can never go red.
    assert await _settle(lambda: len(gateway.payloads) > seeded_payloads), (
        "the parked wake turn never reached the provider after the nav-away"
    )
    assert await _settle(
        lambda: bool((runs_db.get_run(run_id) or {}).get("wake_delivered_at"))
    ), "the headless wake turn never committed (no ledger stamp)"

    _step("nav: returning to console")
    chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
    _step("nav: returned")
    assert isinstance(chat2, ChatScreen), type(chat2).__name__
    assert chat2 is not chat, "screens are never cached"
    await _wait_for_selector(chat2, pilot, "#console-native-composer")
    await pilot.pause()
    return chat2, store, session_id, conversation_id, runs_db, run_id


# ---------------------------------------------------------------------------
# The red: P3b as a regression test.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_wake_that_ran_while_console_was_unmounted_is_in_the_transcript(
    tmp_path,
):
    """P3b. Executed on unmodified production: the user saw 2 of 4 rows.

    The wake turn genuinely ran -- it reached the provider, appended its
    machine-origin SYSTEM notice and its assistant reply, persisted both to
    ChaChaNotes and stamped `agent_runs.wake_delivered_at`. Returning to
    Console rebuilt the store from a snapshot taken BEFORE any of it.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False

    async with app.run_test(size=(160, 48)) as pilot:
        (
            chat2,
            _store,
            session_id,
            conversation_id,
            _runs_db,
            _run_id,
        ) = await _run_headless_wake_turn(app, pilot, gateway, tmp_path)

        db_rows = _db_chain(app.chachanotes_db, conversation_id)
        db_text = "\n".join(row[2] for row in db_rows)
        assert WAKE_NOTICE_HEADER in db_text, (
            "harness precondition: the headless wake must have PERSISTED its "
            f"notice; rows were {[(r[1], r[2][:30]) for r in db_rows]}"
        )
        assert WAKE_REPLY in db_text, (
            "harness precondition: the headless wake must have PERSISTED its "
            f"reply; rows were {[(r[1], r[2][:30]) for r in db_rows]}"
        )

        transcript = _transcript_text(chat2._console_chat_store, session_id)
        rendered = _rendered_text(chat2)
        rows = [
            (m.role.value, m.content[:32])
            for m in chat2._console_chat_store.messages_for_session(session_id)
        ]
        assert WAKE_NOTICE_HEADER in transcript, (
            "the wake notice is missing from the transcript the user sees on "
            f"returning (P3b). Transcript was {rows}; the DB has "
            f"{len(db_rows)} rows."
        )
        assert WAKE_REPLY in transcript, (
            f"the wake's reply is missing from the returned transcript: {rows}"
        )
        assert WAKE_NOTICE_HEADER in rendered, (
            "the wake notice is in the store but was never RENDERED on return"
        )
        assert WAKE_REPLY in rendered, (
            "the wake reply is in the store but was never RENDERED on return"
        )


# ---------------------------------------------------------------------------
# The centrepiece: no second source of truth.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_payload_db_and_active_leaf_all_agree(tmp_path):
    """The four things P1 found disagreeing must agree, by identity.

    P1's failure was not "a row is missing" but four surfaces telling four
    different stories: the transcript, the next send's provider payload, the
    persisted rows and the durable active-leaf pointer. This asserts all four
    against each other -- on message IDs, not merely on content, so a
    look-alike rebuild cannot pass -- and then proves the next append extends
    the same chain instead of forking away from it.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False

    async with app.run_test(size=(160, 48)) as pilot:
        (
            chat2,
            _store,
            session_id,
            conversation_id,
            _runs_db,
            _run_id,
        ) = await _run_headless_wake_turn(app, pilot, gateway, tmp_path)

        db = app.chachanotes_db
        store2 = chat2._console_chat_store
        controller2 = chat2._ensure_console_chat_controller()

        # (1) TRANSCRIPT -- the in-memory rows, by id and in order.
        transcript = list(store2.messages_for_session(session_id))
        # `persisted_message_id`, not the local `id`: the store mints its own
        # in-memory ids, and the DURABLE identity is what has to match, or a
        # transcript rebuilt from look-alike text would pass.
        transcript_ids = [m.persisted_message_id for m in transcript]
        assert all(transcript_ids), (
            "a visible row carries no persisted identity at all: "
            f"{[(m.role.value, m.content[:24], m.persisted_message_id) for m in transcript]}"
        )
        assert len(transcript) == 4, (
            "expected user + assistant + wake notice + wake reply; got "
            f"{[(m.role.value, m.content[:30]) for m in transcript]}"
        )

        # (2) DB ROWS -- same ids, same order, and ONE unforked chain.
        db_rows = _db_chain(db, conversation_id)
        assert [row[0] for row in db_rows] == transcript_ids, (
            "the persisted rows and the transcript are not the same messages: "
            f"db={[(r[1], r[2][:24]) for r in db_rows]} vs "
            f"transcript={[(m.role.value, m.content[:24]) for m in transcript]}"
        )
        parents = [row[3] for row in db_rows]
        assert parents[0] is None, f"the first row must be a root, got {parents[0]!r}"
        for index in range(1, len(db_rows)):
            assert parents[index] == db_rows[index - 1][0], (
                "the persisted conversation FORKED: row "
                f"{index} ({db_rows[index][1]}) parents to {parents[index]!r}, "
                f"not to its predecessor {db_rows[index - 1][0]!r}"
            )

        # (3) ACTIVE LEAF -- the durable pointer names the chain's last row.
        leaf = db.get_conversation_active_leaf(conversation_id)
        assert leaf == db_rows[-1][0], (
            f"active leaf {leaf!r} does not point at the last persisted row "
            f"{db_rows[-1][0]!r} ({db_rows[-1][1]})"
        )

        # (4) PROVIDER PAYLOAD -- the next send carries every row, in order.
        gateway.payloads.clear()
        outcome = await controller2.submit_draft(
            "second user message", session_id=session_id
        )
        assert outcome.accepted, "the follow-up send was refused"
        assert gateway.payloads, "the follow-up send never reached the provider"
        payload = gateway.payloads[-1]
        payload_text = "\n".join(str(entry["content"]) for entry in payload)
        narrated = [
            message
            for message in transcript
            if message.role in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}
        ]
        assert len(narrated) == 3, (
            "expected the two seeded rows plus the headless wake reply: "
            f"{[(m.role.value, m.content[:24]) for m in narrated]}"
        )
        for message in narrated:
            assert message.content in payload_text, (
                f"a row the user can see ({message.role.value}: "
                f"{message.content[:32]!r}) never reached the provider payload: "
                f"{[(e['role'], str(e['content'])[:24]) for e in payload]}"
            )
        # ...and in the transcript's own order, not merely present somewhere.
        positions = [payload_text.index(m.content) for m in narrated]
        assert positions == sorted(positions), (
            "the provider payload reorders the transcript: "
            f"{[(e['role'], str(e['content'])[:24]) for e in payload]}"
        )
        # The other direction, so "agreement" cannot be bought by narrating
        # everything: a Console SYSTEM row is UI chrome and is deliberately
        # NOT sent to the model (`_provider_message_payloads` keeps only
        # USER/ASSISTANT), and the wake notice is a SYSTEM row.
        assert WAKE_NOTICE_HEADER not in payload_text, (
            "the machine-origin SYSTEM notice was narrated to the model as a "
            "conversation turn"
        )

        # (5) The next persisted append EXTENDS the chain (P1's fork).
        after_rows = _db_chain(db, conversation_id)
        assert len(after_rows) > len(db_rows), "the follow-up send persisted nothing"
        new_rows = after_rows[len(db_rows) :]
        assert new_rows[0][3] == db_rows[-1][0], (
            "the next persisted append FORKED away from the headless rows: it "
            f"parented to {new_rows[0][3]!r}, not to the chain's leaf "
            f"{db_rows[-1][0]!r}"
        )
        assert db.get_conversation_active_leaf(conversation_id) == after_rows[-1][0]


# ---------------------------------------------------------------------------
# Session identity across navigation (an unsaved conversation).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_identity_survives_a_navigation_for_an_unsaved_chat(tmp_path):
    """A staged wake targets a session id; that id must not change on nav.

    The snapshot preserved session ids explicitly
    (`_console_session_to_state`), and that is the only reason today's
    mount-claim can deliver into an UNSAVED chat -- one with no
    `persisted_conversation_id` at all, which no DB re-read could ever
    recover. Asserted by execution on a session that has never persisted:
    the id, the tab set, the active session and the draft all survive, and
    the runtime hands the second screen the SAME session object.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        store = chat._ensure_console_chat_store()
        first = store.sessions()[0]
        second = chat._ensure_console_chat_controller().new_session(title="Second")
        store.switch_session(first.id)
        await pilot.pause()
        # Type the half-thought the way a user does: `save_state` flushes the
        # COMPOSER into the store, so a draft written straight to the store
        # would be overwritten by the empty composer on the way out.
        chat.query_one("#console-native-composer").focus()
        await pilot.pause()
        await pilot.press("h", "a", "l", "f")
        await pilot.pause()
        before_ids = [session.id for session in store.sessions()]
        assert first.persisted_conversation_id in (None, ""), (
            "this test is about an UNSAVED conversation"
        )

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert chat2 is not chat
        await _wait_for_selector(chat2, pilot, "#console-native-composer")

        store2 = chat2._console_chat_store
        after = {session.id: session for session in store2.sessions()}
        assert list(after) == before_ids, (
            f"session identity changed across the navigation: {before_ids} -> "
            f"{list(after)}"
        )
        assert store2.active_session_id == first.id, (
            "the active session changed across the navigation"
        )
        assert after[first.id].draft == "half", (
            "an unsaved session's draft did not survive the navigation: "
            f"{after[first.id].draft!r}"
        )
        assert after[second.id].title == "Second"
        assert after[first.id] is first, (
            "the returning Console got a REBUILT session object for an unsaved "
            "conversation -- a staged wake holding the old reference would "
            "target a dead id"
        )


# ---------------------------------------------------------------------------
# The freeze-incident gate: a rapid-switch soak that asserts INTERACTIVITY.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rapid_route_switching_leaves_console_interactive(tmp_path):
    """The 2026-07-11 freeze shape, asserted as interactivity, not silence.

    That incident was total and exception-free: re-mounting a torn-down
    screen left child pumps permanently stopped while widgets still reported
    `mounted=True`, the compositor kept presenting a stale frame, and every
    click was hit-tested into the dead tree. A soak proving "no exception"
    would have passed straight through it. So after the churn this asserts
    the app still RESPONDS: a real keypress reaches a live widget and changes
    its state, a real click lands, and the transcript repaints new content
    appended after the churn.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )

        for index in range(9):
            target, expect = (
                ("library", "LibraryScreen"),
                ("chat", "ChatScreen"),
                ("settings", "SettingsScreen"),
                ("chat", "ChatScreen"),
            )[index % 4]
            await _navigate(app, pilot, target, expect=expect)

        screen = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert isinstance(screen, ChatScreen), type(screen).__name__
        await _wait_for_selector(screen, pilot, "#console-native-composer")

        # (a) A REAL keypress reaches a live widget and changes its state.
        composer = screen.query_one("#console-native-composer")
        composer.focus()
        await pilot.pause()
        before_draft = composer.draft_text()
        await pilot.press("z", "q")
        await pilot.pause()
        after_draft = composer.draft_text()
        assert after_draft != before_draft and after_draft.endswith("zq"), (
            "a keypress after the churn never reached the composer "
            f"({before_draft!r} -> {after_draft!r}) -- the widget tree is not live"
        )

        # (b) A REAL click lands on a live widget (the composer regains focus
        #     after we move it away, which only a live hit-test can do).
        transcript_widget = screen.query_one("#console-transcript-surface")
        transcript_widget.focus()
        await pilot.pause()
        await pilot.click("#console-native-composer")
        await pilot.pause()
        focused = getattr(app.screen, "focused", None) or getattr(app, "focused", None)
        assert getattr(focused, "id", None) == "console-native-composer", (
            "a click after the churn was hit-tested into a dead tree "
            f"(focus landed on {getattr(focused, 'id', None)!r})"
        )

        # (c) The transcript REPAINTS content appended after the churn.
        live_store = screen._console_chat_store
        assert SEEDED_USER in _rendered_text(screen), (
            "the pre-churn transcript is not being presented after the churn"
        )
        # Sent through the SCREEN's own Enter path, not `submit_draft`: only
        # the screen dispatch arms the transcript poll, and "does the view
        # repaint" is exactly what this soak is for.
        gateway.reply = "post-churn reply"
        await pilot.press("p", "q")
        await pilot.press("enter")
        assert await _settle(
            lambda: "post-churn reply" in _rendered_text(app.screen), seconds=8.0
        ), "the transcript never repainted the turn sent after the churn"
        assert "zqpq" in _transcript_text(live_store, session_id), (
            "the typed message never reached the store: "
            f"{[m.content[:24] for m in live_store.messages_for_session(session_id)]}"
        )

        # (d) Nothing accumulated: one runtime, one controller, one store.
        assert app.console_runtime.chat_controller is controller
        assert app.console_runtime.chat_store is store
        assert screen._console_chat_store is store
