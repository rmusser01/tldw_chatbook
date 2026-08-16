"""Headless approval: surface it app-wide, keep it claimable (task-15860, plan Task 5).

**Step 1 measured, not assumed** (probe
`Tests/UI/test_probe_headless_approval_behaviour.py`, and the report):
the plan's 120.43s (P4) is stale in two independent ways.

* ADR-067 dropped `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS` 120.0 -> **0.0**:
  no deadline is armed by default at all.
* `request_mcp_approvals` binds the VISIT cancellation Event at ARM time
  (`_bind_visit_cancel_signal`), and while Console is detached that Event
  is already SET -- so a round armed headless was denied at the FIRST
  1.0s poll.

Measured through the production path: **1.01s to `deny`**, silently. Not
120s of silence -- a second of silence, and then every risk-tagged tool
in a headless wake turn is auto-refused with the user never told.

That is fail-closed, but it makes plan Task 5's actual requirement
unreachable: a round must be *resolvable by opening Console*. This file
is the red for that, plus the safety pins around it.

The distinction the fix rests on is the same one the wake-fires landing
made for `_attempt`: the visit Event means "the visit that armed this
round ended". A round armed while NO view is attached was not armed
during any visit -- reading that Event for it is the same category error,
one layer down. `_disposed` (app exit), the run's own cancel event and a
CONFIGURED deadline all still deny, unchanged.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _quiet,
    _settle,
    _survivor,
    _terminal_subagent_run,
)
from Tests.Chat.test_console_runtime_lifetime import _View, _pending_call
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_mcp_approval import _pending
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_store_continuity import (
    _StallingWakeGateway,
    _drain_from_child_thread,
    _navigate,
    _seed_console,
    _terminal_survivor_run,
)
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard


# ---------------------------------------------------------------------------
# rig
# ---------------------------------------------------------------------------


class _ThreadApp:
    """The app surface `request_mcp_approvals` actually touches.

    `call_from_thread` runs inline (the convention every controller-level
    approval test uses) and `notify` records, so an app-wide announcement
    is observable without a live Textual app.

    **Wiring `app` at all matters:** ADR-067 added a no-`app` guard that
    denies a round on the spot, so a round test built on a controller with
    `app is None` never reaches the poll loop and proves nothing about
    cancellation. Two of the lifetime landing's own AC#2 pins were in that
    state; see the report.
    """

    def __init__(self) -> None:
        self.notifications: list[tuple[str, str]] = []

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def notify(self, message, *, severity="information", **_kwargs) -> None:
        self.notifications.append((str(message), severity))


class _StalledGateway:
    """Never resolves a send; approval rounds are what these tests drive."""

    async def resolve_for_send(self, selection):
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        return None


def _detached_rig(*, timeout_seconds: float | None = None):
    """A controller whose Console visit has genuinely ENDED.

    Built through the production seam (`ConsoleRuntime.attach_view` then
    `leave_console`), which is what the real navigation does -- not by
    poking `_shutdown_requested`.
    """
    store = ConsoleChatStore()
    session = store.ensure_session(title="Headless")
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    app = _ThreadApp()
    controller.app = app
    if timeout_seconds is not None:
        controller.mcp_approval_timeout_seconds = lambda: timeout_seconds
    runtime = ConsoleRuntime(app=app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    runtime.attach_view(_View())
    return runtime, controller, store, session, app


async def _leave(runtime) -> None:
    await asyncio.wait_for(runtime.leave_console(), timeout=5)


def _arm(controller, session_id, *, call=None) -> tuple[threading.Thread, dict]:
    """Arm one approval round on a plain worker thread, as production does."""
    box: dict[str, object] = {}
    started = threading.Event()

    def _run() -> None:
        started.set()
        box["decisions"] = controller.request_mcp_approvals(
            [call or _pending_call()], session_id=session_id
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert started.wait(timeout=2), "the arming thread never started"
    box["thread"] = thread
    return thread, box


def _armed_round_ids(controller, session_id) -> list[str]:
    with controller._approval_state_lock:
        return [
            round_id
            for round_id, state in controller._pending_approval_rounds.items()
            if state.get("session_id") == session_id
        ]


def _round_is_claimable(controller, session_id) -> bool:
    """Registered AND payload-retained -- the two writes a mount needs.

    `request_mcp_approvals` registers the round in
    `_pending_approval_rounds` and only afterwards retains its payload in
    `_parked_approval_payloads`, with `_resolve_mcp_approval_timeout_
    seconds()` (a `get_cli_setting` read, which on a cold test config
    CREATES the file) in between. Waiting on the registration alone
    therefore returns inside that window, and an attach that follows
    finds no payload and mounts nothing.

    Measured, not theorised: mutation M3 made
    `test_attaching_a_view_mounts_a_round_armed_while_detached` fail
    deterministically for exactly this reason -- a mutation "kill" that
    was really a timing artefact. The mount depends on the PAYLOAD, so
    that is what the precondition has to observe.
    """
    if not _armed_round_ids(controller, session_id):
        return False
    with controller._approval_state_lock:
        return controller._parked_approval_payloads.get(session_id) is not None


async def _wait_for_round(controller, session_id, *, seconds: float = 3.0) -> bool:
    return await _settle(
        lambda: _round_is_claimable(controller, session_id), seconds=seconds
    )


def _risk_row():
    """The shape `build_tool_review_hook` emits for a risk-tagged tool."""
    return _pending(
        server_key="agent:builtin",
        tool_name="write_file",
        llm_name="builtin__write_file",
        reason="risk_floored",
    )


def _toast_text(app) -> str:
    """Text from the app's MOUNTED `Toast` widgets, whatever screen is up.

    Reads `Toast.render()`, not `renderable`: a `Toast` is a `Static`
    that never calls `update()`, so its `renderable` is empty and a
    helper reading that attribute reports "no toast" for a toast that is
    on screen. Measured -- that is what this helper did first.
    """
    chunks: list[str] = []
    for screen in app.screen_stack:
        for node in screen.walk_children(with_self=True):
            if type(node).__name__ != "Toast":
                continue
            render = getattr(node, "render", None)
            if callable(render):
                try:
                    chunks.append(str(render()))
                except Exception:  # noqa: BLE001 -- a mid-mount toast
                    pass
    return "\n".join(chunks)


def _build_console_app(tmp_path):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    return app, gateway


# ---------------------------------------------------------------------------
# THE RED -- app-wide surfacing + resolvable by opening Console
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_headless_risk_tagged_round_toasts_app_wide_and_is_resolvable(
    tmp_path,
):
    """The whole of plan Task 5's first two bullets, through production.

    Console is left via the REAL navigation API (`NavigateToScreen` + the
    real "Leave Console?" dialog) with a wake turn parked in flight, so
    the runtime is in exactly the state a headless wake runs in. Then a
    risk-tagged round arms from a plain worker thread.

    RED before the fix, measured (1.01s): the round self-denied at the
    first poll, no toast was raised on the Library screen, and opening
    Console showed no card -- the user was never told and could not have
    answered.

    Asserted:

    1. an app-wide toast names the pending approval **while the user is on
       another screen** -- read off the MOUNTED toast widgets, not the
       notification list;
    2. the round is STILL armed well past the 1.0s poll granularity that
       used to kill it;
    3. opening Console MOUNTS the card (the payload survived detachment);
    4. answering it on that card resolves the worker thread's round with
       the human's verdict.
    """
    app, gateway = _build_console_app(tmp_path)

    # `notifications=True` is REQUIRED and is not decoration: Textual's
    # `run_test` defaults it to False, which makes `Screen._extend_compose`
    # skip the `ToastRack` entirely -- no toast can ever mount, so a test
    # that asserted on the RENDERED toast under the default would fail
    # forever and one that asserted on `app._notifications` would pass
    # without proving anything reached the screen.
    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_id = _terminal_survivor_run(runs_db, conversation_id)

        gateway.stall = True
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
        )
        assert await _settle(lambda: gateway.entered_stall.is_set(), seconds=10.0), (
            "harness precondition: the wake turn must be in flight"
        )

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack, "Console must actually unmount"
        assert controller is app.console_runtime.chat_controller, (
            "harness precondition: the runtime must OUTLIVE the screen"
        )
        assert controller._shutdown_requested.is_set(), (
            "harness precondition: the visit must really have ended -- that set "
            "Event is exactly what used to deny the round at the first poll"
        )
        assert controller._disposed is False, "a navigation is not an app exit"
        assert controller.set_pending_approval is None, (
            "harness precondition: the card seam must be detached"
        )

        thread, box = _arm(controller, session_id, call=_risk_row())

        # (1) the toast reaches the user on the screen they are ACTUALLY on.
        assert await _settle(
            lambda: "approval" in _toast_text(app).lower(), seconds=5.0
        ), (
            "a risk-tagged tool armed an approval round with no Console mounted "
            "and nothing surfaced app-wide; the user is on "
            f"{type(app.screen).__name__} and sees: {_toast_text(app)!r}"
        )

        # (2) it did not self-deny at the first poll.
        assert await _quiet(lambda: "decisions" in box, seconds=2.5), (
            "the headless round resolved itself before the user could possibly "
            f"open Console: {box.get('decisions')}"
        )
        assert _armed_round_ids(controller, session_id), (
            "the round is no longer registered, so opening Console cannot claim it"
        )

        # (3) opening Console MOUNTS the card.
        chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert chat2 is not chat, "screens are never cached"
        await pilot.pause()
        assert await _settle(
            lambda: bool(list(chat2.query(".approval-row"))), seconds=5.0
        ), (
            "the round armed while Console was closed did not mount its card on "
            "attach -- it was silently re-parked and the user still cannot answer"
        )
        card = chat2.query_one(ChatApprovalCard)
        assert "write_file" in _rendered(card), (
            f"the mounted card is not this round's: {_rendered(card)!r}"
        )

        # (4) the human's verdict resolves the worker's round.
        chat2.query_one(".approval-row-fast-approve", Button).press()
        await pilot.pause()
        assert await _settle(lambda: "decisions" in box, seconds=10.0), (
            "answering the card never resolved the headless round"
        )
        thread.join(timeout=5)
        assert box["decisions"] == {"builtin__write_file": "approve_once"}, box[
            "decisions"
        ]

        gateway.release.set()
        await pilot.pause()


def _rendered(widget) -> str:
    chunks: list[str] = []
    for node in widget.walk_children(with_self=True):
        renderable = getattr(node, "renderable", None)
        if renderable is not None:
            chunks.append(str(renderable))
        label = getattr(node, "label", None)
        if label is not None:
            chunks.append(str(label))
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# The announcement seam, without a live app
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_headless_round_announces_through_the_app_not_the_screen():
    """The announcement must not depend on a screen hook that is None.

    `park_pending_approval`/`set_pending_approval` are both cleared by
    `detach_view`, so the screen's own toast seam
    (`ChatScreen._park_console_approval`) is unreachable. The controller
    announces through `app.notify` -- the app-wide seam, which renders on
    whatever screen the user is on.
    """
    runtime, controller, _store, session, app = _detached_rig()
    await _leave(runtime)
    assert controller.park_pending_approval is None
    assert controller.set_pending_approval is None

    thread, box = _arm(controller, session.id, call=_risk_row())
    assert await _settle(lambda: bool(app.notifications), seconds=3.0), (
        "a round armed with no view announced nothing app-wide"
    )
    message, _severity = app.notifications[0]
    assert "approval" in message.lower(), message
    assert "console" in message.lower(), (
        f"the notice must tell the user WHERE to answer: {message!r}"
    )
    assert len(app.notifications) == 1, (
        f"one round, one announcement: {app.notifications}"
    )

    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"},
        round_id=_armed_round_ids(controller, session.id)[0],
    )
    thread.join(timeout=5)


@pytest.mark.asyncio
async def test_a_round_armed_with_a_view_attached_does_not_double_announce():
    """The app-wide notice is the DETACHED path only.

    With a view attached the screen's own card/park seams do the
    surfacing; adding a second app-level toast on top would double every
    mounted round's announcement.
    """
    runtime, controller, store, session, app = _detached_rig()
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append

    thread, box = _arm(controller, session.id, call=_risk_row())
    assert await _settle(lambda: bool(mounted), seconds=3.0), "the card never mounted"
    assert app.notifications == [], (
        f"a mounted round raised a second, app-level toast: {app.notifications}"
    )

    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"},
        round_id=_armed_round_ids(controller, session.id)[0],
    )
    thread.join(timeout=5)
    await _leave(runtime)


# ---------------------------------------------------------------------------
# Attach mounts the card (plan Task 5 bullet 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attaching_a_view_mounts_a_round_armed_while_detached():
    """A round still armed at attach must MOUNT, not sit invisible.

    Nothing else would ever mount it: the screen's card is derived from
    `_task_resume_state`, which a fresh screen starts empty, and
    `switch_session`'s re-derive only runs when the user switches
    sessions -- which they have no reason to do, having never seen a card.
    """
    runtime, controller, _store, session, _app = _detached_rig()
    await _leave(runtime)
    thread, box = _arm(controller, session.id, call=_risk_row())
    assert await _wait_for_round(controller, session.id), "the round never armed"

    mounted: list[dict | None] = []
    view = _View({"set_pending_approval": mounted.append})
    runtime.attach_view(view)

    assert mounted and mounted[-1] is not None, (
        "attaching a view left the armed round invisible"
    )
    payload = mounted[-1]
    assert payload["session_id"] == session.id
    assert [c["llm_name"] for c in payload["calls"]] == ["builtin__write_file"]

    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=payload["round_id"]
    )
    thread.join(timeout=5)


@pytest.mark.asyncio
async def test_attaching_a_view_with_no_armed_round_mounts_nothing():
    """The mirror: attach must not push an empty/stale card at every mount.

    `attach_view` runs on EVERY `_ensure_console_chat_controller()` call,
    so a re-derive that fired unconditionally would repaint the card
    surface on every tick that touches the controller.
    """
    runtime, controller, _store, _session, _app = _detached_rig()
    await _leave(runtime)
    mounted: list[dict | None] = []
    runtime.attach_view(_View({"set_pending_approval": mounted.append}))
    assert mounted == [], f"attach pushed a card with nothing armed: {mounted}"


# ---------------------------------------------------------------------------
# The two-round limitation -- asserted AGAINST, not around (task-15661)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_headless_rounds_share_one_payload_slot_and_only_one_mounts():
    """KNOWN LIMITATION, pinned: `_parked_approval_payloads` is per-SESSION.

    Both rounds are independently registered (the badge and the verdicts
    are round-keyed), and both stay claimable in the sense that neither is
    denied -- but the retained payload slot holds only the LAST-armed
    round, so an attach can only ever mount ONE card. That is
    `console_chat_controller.py`'s own documented "accepted scope
    limitation" on that slot, filed as task-15661.

    **This task does NOT fix it.** Per-round payload storage changes the
    card's mount/park contract for every caller (mounted rounds included),
    which is a wider change than making the headless case surfaceable.
    Pinned here so the limitation is a measured fact with a failing test
    the day someone fixes it, rather than folklore in a comment.
    """
    runtime, controller, _store, session, app = _detached_rig()
    await _leave(runtime)

    first_call = _pending(
        server_key="agent:builtin",
        tool_name="write_file",
        llm_name="builtin__write_file",
        reason="risk_floored",
    )
    second_call = _pending(
        server_key="agent:builtin",
        tool_name="delete_note",
        llm_name="builtin__delete_note",
        reason="risk_floored",
    )
    thread_a, box_a = _arm(controller, session.id, call=first_call)
    assert await _wait_for_round(controller, session.id), "round A never armed"
    round_a = _armed_round_ids(controller, session.id)[0]
    thread_b, box_b = _arm(controller, session.id, call=second_call)
    assert await _settle(
        lambda: len(_armed_round_ids(controller, session.id)) == 2, seconds=3.0
    ), "round B never armed alongside A"

    # Both announced: a sibling round must never silence a new one.
    assert len(app.notifications) == 2, app.notifications

    mounted: list[dict | None] = []
    runtime.attach_view(_View({"set_pending_approval": mounted.append}))
    assert mounted and mounted[-1] is not None, "attach mounted nothing at all"
    names = [c["llm_name"] for c in mounted[-1]["calls"]]
    assert names == ["builtin__delete_note"], (
        "THE LIMITATION: the single per-session payload slot means the "
        f"LAST-armed round is the only one an attach can mount; got {names}"
    )
    assert mounted[-1]["round_id"] != round_a, (
        "round A's payload was overwritten by B's -- that IS the limitation"
    )

    for round_id, name in (
        (round_a, "builtin__write_file"),
        (mounted[-1]["round_id"], "builtin__delete_note"),
    ):
        controller.resolve_pending_approval({name: "deny"}, round_id=round_id)
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert box_a["decisions"] == {"builtin__write_file": "deny"}
    assert box_b["decisions"] == {"builtin__delete_note": "deny"}


# ---------------------------------------------------------------------------
# SAFETY PINS -- nothing below may change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leaving_console_still_denies_a_round_armed_during_the_visit():
    """AC#2, for real: the visit Event still denies ITS OWN visit's rounds.

    The distinction the fix rests on -- "armed during a visit" vs "armed
    with no visit open" -- is only sound if this half keeps working.
    """
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=60.0)
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the round never armed"
    # Let the poll loop reach its first `event.wait(1.0)`.
    await asyncio.sleep(0.2)

    await _leave(runtime)
    thread.join(timeout=10)

    assert not thread.is_alive(), "the round never resolved after leaving Console"
    assert box["decisions"] == {"write_file": "deny"}, box["decisions"]


@pytest.mark.asyncio
async def test_leaving_console_denies_a_round_armed_while_detached_too():
    """The headless round is deferred, never immortal.

    A round armed with no view waits for the user -- but once the user
    HAS opened Console and left again, they have seen it and declined to
    answer, so the same AC#2 rule applies.
    """
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=60.0)
    await _leave(runtime)
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the round never armed"
    assert await _quiet(lambda: "decisions" in box, seconds=2.0), (
        "the headless round self-denied instead of waiting"
    )

    runtime.attach_view(_View())
    await _leave(runtime)
    thread.join(timeout=10)
    assert box["decisions"] == {"write_file": "deny"}, box["decisions"]


@pytest.mark.asyncio
async def test_a_second_headless_round_after_a_leave_is_not_born_denied():
    """The deferred Event must not be REUSED once it has fired.

    Written because mutation M8 (stop dropping `_headless_visit_cancel`
    after setting it) SURVIVED the rest of this file. Investigating it
    showed the drop is redundant with `_bind_visit_cancel_signal`'s own
    `event.is_set()` guard -- two independent defences of one property,
    so neither line is individually killable. This test pins the
    PROPERTY, which is what actually matters: if both defences went, a
    headless round armed after any previous leave would inherit a
    pre-set Event and self-deny in ~1s, silently restoring the exact
    behaviour this task removed, and nothing else here would notice.
    """
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=60.0)
    await _leave(runtime)

    first, box_first = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "round 1 never armed"
    runtime.attach_view(_View())
    await _leave(runtime)
    first.join(timeout=10)
    assert box_first["decisions"] == {"write_file": "deny"}, box_first["decisions"]

    second, box_second = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "round 2 never armed"
    assert await _quiet(lambda: "decisions" in box_second, seconds=2.0), (
        "the SECOND headless round was denied on arrival -- it inherited the "
        f"first round's already-fired Event: {box_second.get('decisions')}"
    )

    controller.resolve_pending_approval(
        {"write_file": "deny"},
        round_id=_armed_round_ids(controller, session.id)[0],
    )
    second.join(timeout=5)


@pytest.mark.asyncio
async def test_app_exit_denies_a_round_armed_while_detached():
    """`_disposed` is the signal a headless round DOES answer to."""
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=60.0)
    await _leave(runtime)
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the round never armed"

    await asyncio.wait_for(runtime.dispose(), timeout=10)
    thread.join(timeout=10)
    assert controller._disposed is True
    assert box["decisions"] == {"write_file": "deny"}, box["decisions"]


@pytest.mark.asyncio
async def test_a_configured_deadline_still_expires_a_headless_round():
    """Plan Task 5 bullet 2: the clock is NOT paused or extended while detached.

    A positive `[mcp] approval_timeout_seconds` is a fail-closed ceiling
    the user opted into; detachment does not buy the round more time.
    """
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=2.0)
    await _leave(runtime)
    started = time.monotonic()
    thread, box = _arm(controller, session.id)
    assert await _settle(lambda: "decisions" in box, seconds=10.0), (
        "a configured deadline never expired the headless round"
    )
    elapsed = time.monotonic() - started
    thread.join(timeout=5)
    assert box["decisions"] == {"write_file": "timeout"}, box["decisions"]
    assert elapsed < 6.0, (
        f"the 2s deadline took {elapsed:.2f}s -- detachment must not extend it"
    )


@pytest.mark.asyncio
async def test_no_headless_path_returns_an_approval_without_a_human():
    """Every automatic resolution of a headless round fails CLOSED.

    Deadline, app exit and leave-after-attach are the three ways a round
    can end with nobody answering; none of them may produce an
    `approve_*` verdict.
    """
    verdicts: list[str] = []

    # deadline
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=1.0)
    await _leave(runtime)
    thread, box = _arm(controller, session.id)
    assert await _settle(lambda: "decisions" in box, seconds=10.0)
    thread.join(timeout=5)
    verdicts.extend(box["decisions"].values())

    # app exit
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=60.0)
    await _leave(runtime)
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id)
    await asyncio.wait_for(runtime.dispose(), timeout=10)
    thread.join(timeout=10)
    verdicts.extend(box["decisions"].values())

    assert verdicts, "no verdicts were collected"
    assert all(not v.startswith("approve") and v != "always_allow" for v in verdicts), (
        f"a headless round resolved to an APPROVAL with no human: {verdicts}"
    )


@pytest.mark.asyncio
async def test_a_wake_delivery_cannot_resolve_a_pending_headless_round(tmp_path):
    """A wake notice is never user input and never an approval.

    A full wake turn delivers into the same controller while a headless
    round is armed; the round must be untouched by it.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, store, session, gateway, _bridge, controller = rig
    try:
        thread_app = _ThreadApp()
        controller.app = thread_app
        controller.mcp_approval_timeout_seconds = lambda: 60.0
        runtime = ConsoleRuntime(app=app)
        runtime.set_chat_store(store)
        runtime.set_chat_controller(controller)
        runtime.attach_view(_View())
        await _leave(runtime)

        thread, box = _arm(controller, session.id, call=_risk_row())
        assert await _wait_for_round(controller, session.id), "the round never armed"
        round_ids_before = set(_armed_round_ids(controller, session.id))

        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        controller.fleet_wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: bool(gateway.payloads), seconds=10.0), (
            "harness precondition: the wake turn must actually deliver"
        )

        assert "decisions" not in box, (
            f"the wake delivery resolved the pending approval round: {box}"
        )
        assert set(_armed_round_ids(controller, session.id)) == round_ids_before, (
            "the wake delivery disturbed the armed round's registration"
        )

        controller.resolve_pending_approval(
            {"builtin__write_file": "deny"}, round_id=round_ids_before.pop()
        )
        thread.join(timeout=5)
        assert box["decisions"] == {"builtin__write_file": "deny"}
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_the_risk_floor_still_raises_a_card_in_a_headless_turn(
    tmp_path, monkeypatch
):
    """The approval FLOOR is unchanged headless: a real risk-tagged tool asks.

    Drives the REAL `ReadFileTool` (risk_tags `("reads",)`) through the
    REAL `BuiltinToolGate` and the REAL `build_tool_review_hook`, wired to
    a DETACHED controller exactly as `_run_agent_reply` wires it
    (`functools.partial(self.request_mcp_approvals, session_id=...)`).
    The hook runs on a worker thread, as it does inside a turn.

    A wake turn composes this hook the same way a manual turn does -- the
    composition never consults the submission origin -- so "the floor
    still applies in a woken turn" is exactly this.
    """
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
    from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY
    from tldw_chatbook.Tools import file_operation_tools as fot
    from tldw_chatbook.Tools import workspace_file_roots as wfr
    from tldw_chatbook.Tools.file_operation_tools import ReadFileTool

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())

    def _no_registry():
        raise RuntimeError("no workspace registry in this test")

    monkeypatch.setattr(wfr, "_registry_factory", _no_registry)

    class _SessionApprovalService:
        def get_kill_switch(self) -> bool:
            return False

        def approve_for_session(self, server_key, tool_name) -> None:
            return None

        def is_session_approved(self, server_key, tool_name) -> bool:
            return False

    class _RealToolProvider:
        def __init__(self, *tools) -> None:
            self._tools = {tool.name: tool for tool in tools}

        def tool_for(self, name):
            return self._tools.get(name)

    tool = ReadFileTool()
    assert "reads" in tool.risk_tags, "precondition: the real tag set"

    runtime, controller, _store, session, app = _detached_rig(timeout_seconds=60.0)
    await _leave(runtime)

    hook = build_tool_review_hook(
        BuiltinToolGate(_SessionApprovalService()),
        _RealToolProvider(tool),
        None,
        functools.partial(controller.request_mcp_approvals, session_id=session.id),
    )

    verdicts: dict = {}

    def _run() -> None:
        verdicts.update(
            hook(
                [ToolCall(name="read_file", args={"file_path": "notes.md"}, call_id="c1")],
                "run-1",
            )
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert await _wait_for_round(controller, session.id, seconds=5.0), (
        "a REAL risk-tagged tool ran headlessly WITHOUT raising an approval "
        "round -- the floor stopped applying with no Console mounted"
    )
    assert app.notifications, "the floored round was never announced app-wide"
    round_id = _armed_round_ids(controller, session.id)[0]
    with controller._approval_state_lock:
        state = controller._pending_approval_rounds[round_id]
    assert state["names"] == ("read_file",), state["names"]
    payload = controller._parked_approval_payloads[session.id]
    assert payload["calls"][0]["server_key"] == BUILTIN_TOOL_SERVER_KEY
    assert payload["calls"][0]["reason"] == "risk_floored"

    controller.resolve_pending_approval({"read_file": "deny"}, round_id=round_id)
    thread.join(timeout=10)
    assert verdicts.get("c1") not in (None, "proceed"), (
        f"the refusal did not reach the runtime: {verdicts}"
    )
