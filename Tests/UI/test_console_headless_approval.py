"""Headless approval: surface it app-wide, keep it claimable (task-15860, plan Task 5).

Approval rounds are app-owned work. Console attachment controls only their
projection: navigation never resolves a round, while exact user decisions,
the owning run's cancellation event, configured deadlines, and app exit
remain authoritative.
"""

from __future__ import annotations

import asyncio
import functools
import threading

import pytest
from textual.widgets import Button

from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _quiet,
    _settle,
    _survivor,
    _terminal_subagent_run,
)
from Tests.Chat.test_console_runtime_lifetime import _pending_call, _View
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_mcp_approval import _pending
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_store_continuity import (
    _drain_from_child_thread,
    _navigate,
    _seed_console,
    _StallingWakeGateway,
    _terminal_survivor_run,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
    SkillInstallConfirmCard,
)

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
        self.notification_posted = threading.Event()

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def notify(self, message, *, severity="information", **_kwargs) -> None:
        self.notifications.append((str(message), severity))
        self.notification_posted.set()


class _StalledGateway:
    """Never resolves a send; approval rounds are what these tests drive."""

    async def resolve_for_send(self, selection):
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        return None


class _DecisionClock:
    """Deterministic monotonic clock for answerable-time assertions."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _detached_rig(*, timeout_seconds: float | None = None):
    """Build an app-owned controller with one attached Console projection."""
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


def _never_visited_rig(*, timeout_seconds: float | None = 60.0):
    """Build the wake-at-launch shape with no Console projection yet."""
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


def _arm_install(controller, session_id) -> tuple[threading.Thread, dict]:
    box: dict[str, object] = {}

    def _run() -> None:
        box["allowed"] = controller.request_skill_install_confirm(
            "https://example.invalid/skill", session_id=session_id
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, box


def _arm_script(controller, session_id) -> tuple[threading.Thread, dict]:
    box: dict[str, object] = {}

    def _run() -> None:
        box["decision"] = controller.request_skill_script_confirm(
            {
                "skill_name": "example",
                "script_path": "run.py",
                "mechanism": "python",
                "args": [],
            },
            session_id=session_id,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, box


def _armed_round_ids(controller, session_id) -> list[str]:
    with controller._approval_state_lock:
        return [
            round_id
            for round_id, state in controller._pending_approval_rounds.items()
            if state.get("session_id") == session_id
        ]


def _pending_ids_for_type(controller, session_id: str, decision_type: str):
    if decision_type == "approval":
        return _armed_round_ids(controller, session_id)
    if decision_type == "skill_install":
        return controller.pending_skill_install_ids()
    return controller.pending_skill_script_ids()


def _resolve_stale_allow(controller, decision_type: str, decision_id: str) -> None:
    if decision_type == "approval":
        controller.resolve_pending_approval(
            {"builtin__write_file": "approve_once"}, round_id=decision_id
        )
    elif decision_type == "skill_install":
        controller.resolve_pending_skill_install(True, request_id=decision_id)
    else:
        controller.resolve_pending_skill_script(
            True, True, request_id=decision_id
        )


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
    # PR0 (task-15661): the map is keyed by ROUND now, so "this session has
    # a retained payload" is a head lookup, not a `.get(session_id)`.
    return (
        controller._head_round_payload(
            controller._parked_approval_payloads, session_id
        )
        is not None
    )


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
        assert not controller._shutdown_requested.is_set(), (
            "ordinary navigation must detach the Console projection without "
            "signalling domain cancellation"
        )
        assert controller._disposed is False, "a navigation is not an app exit"
        assert controller.set_pending_approval is None, (
            "harness precondition: the card seam must be detached"
        )

        thread, box = _arm(controller, session_id, call=_risk_row())

        # (1) the toast reaches the user on the screen they are ACTUALLY on.
        assert await _settle(
            lambda: "needs approval" in _toast_text(app).lower(), seconds=5.0
        ), (
            "a risk-tagged tool armed an approval round with no Console mounted "
            "and nothing surfaced app-wide; the user is on "
            f"{type(app.screen).__name__} and sees: {_toast_text(app)!r}"
        )
        assert type(app.screen).__name__ == "LibraryScreen", (
            "the toast must have been read while the user was on ANOTHER screen"
        )
        # ...and the badge half, which needed no new machinery:
        # `add_pending_round` already ran unconditionally.
        assert controller.has_pending_approval_round(session_id), (
            "the session carries no NEEDS_APPROVAL badge for the armed round"
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


@pytest.mark.asyncio
async def test_widget_promotion_and_navigation_gate_exact_answerable_head(
    tmp_path, monkeypatch
):
    """Only a successfully synced production card consumes active time."""
    app, gateway = _build_console_app(tmp_path)
    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        chat, controller, _store, session_id, _conversation_id = await _seed_console(
            app, pilot, gateway
        )
        clock = _DecisionClock()
        controller.decision_monotonic_clock = clock
        controller.mcp_approval_timeout_seconds = lambda: 60.0
        controller.skill_install_confirm_timeout_seconds = lambda: 60.0

        approval_thread, approval_box = _arm(
            controller, session_id, call=_risk_row()
        )
        assert await _settle(lambda: bool(list(chat.query(".approval-row"))))
        approval_id = _armed_round_ids(controller, session_id)[0]
        assert controller._answerable_decision_by_session == {
            session_id: approval_id
        }

        install_thread, install_box = _arm_install(controller, session_id)
        assert await _settle(lambda: bool(controller.pending_skill_install_ids()))
        install_id = controller.pending_skill_install_ids()[0]
        task_cards = chat.query_one("#console-task-surface", ChatTaskCards)
        install_card = task_cards.query_one(SkillInstallConfirmCard)
        original_set_install = install_card.set_install
        failed_install_mount = threading.Event()

        def _fail_first_install_sync(payload) -> None:
            if payload is not None:
                failed_install_mount.set()
                raise RuntimeError("injected install-card render failure")
            original_set_install(payload)

        monkeypatch.setattr(install_card, "set_install", _fail_first_install_sync)
        controller.resolve_pending_approval(
            {"builtin__write_file": "deny"}, round_id=approval_id
        )
        assert await asyncio.to_thread(failed_install_mount.wait, 5.0)
        await asyncio.to_thread(approval_thread.join, 5.0)
        assert approval_box["decisions"] == {"builtin__write_file": "deny"}
        projection = controller.pending_decision_projection(session_id)
        assert projection is not None
        assert projection.decision_id == install_id
        assert projection.remaining_active_seconds == pytest.approx(60.0)
        assert session_id not in controller._answerable_decision_by_session
        approval_card = chat.query_one(ChatApprovalCard)
        assert approval_card.display is False
        assert approval_card._batch_names == [], (
            "the hidden predecessor retained actionable decision membership"
        )
        assert install_thread.is_alive()

        # A meaningful reconciliation retry mounts the same exact successor.
        monkeypatch.setattr(install_card, "set_install", original_set_install)
        assert controller.project_pending_decision_for_active_session() is True
        assert controller._answerable_decision_by_session == {
            session_id: install_id
        }
        assert chat._task_resume_state.pending_skill_install is not None
        assert (
            chat._task_resume_state.pending_skill_install["request_id"]
            == install_id
        )

        clock.advance(2.0)
        await _navigate(app, pilot, "library", expect="LibraryScreen")
        projection = controller.pending_decision_projection(session_id)
        assert projection is not None
        assert projection.decision_id == install_id
        assert projection.remaining_active_seconds == pytest.approx(58.0)
        assert session_id not in controller._answerable_decision_by_session
        clock.advance(600.0)
        assert controller.expire_pending_decisions() == ()

        successor = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert successor is not chat
        assert await _settle(
            lambda: (
                successor._task_resume_state.pending_skill_install is not None
                and successor._task_resume_state.pending_skill_install.get(
                    "request_id"
                )
                == install_id
            )
        )
        assert controller._answerable_decision_by_session == {
            session_id: install_id
        }
        controller.resolve_pending_skill_install(False, request_id=install_id)
        await asyncio.to_thread(install_thread.join, 5.0)
        assert install_box["allowed"] is False


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
async def test_positive_timeout_counts_only_successfully_answerable_time():
    """A hidden round keeps its allowance; only an answerable head consumes it."""
    runtime, controller, _store, session, _app = _detached_rig(
        timeout_seconds=5.0
    )
    clock = _DecisionClock()
    controller.decision_monotonic_clock = clock
    await _leave(runtime)

    thread, box = _arm(controller, session.id, call=_risk_row())
    assert await _wait_for_round(controller, session.id), "the round never armed"
    projection = controller.pending_decision_projection(session.id)
    assert projection is not None
    decision_id = projection.decision_id
    assert projection.remaining_active_seconds == pytest.approx(5.0)

    # A successful card mount starts the clock. Detach/card replacement clears
    # answerability before any projection state changes.
    assert controller.set_answerable_decision(session.id, decision_id) is True
    clock.advance(2.0)
    assert controller.set_answerable_decision(session.id, None) is True
    projection = controller.pending_decision_projection(session.id)
    assert projection is not None
    assert projection.remaining_active_seconds == pytest.approx(3.0)

    # Arbitrarily long hidden time is free.
    clock.advance(600.0)
    assert controller.expire_pending_decisions() == ()
    assert "decisions" not in box

    # Re-mount resumes the same stable ID from the exact remainder.
    assert controller.set_answerable_decision(session.id, decision_id) is True
    clock.advance(3.0)
    assert controller.expire_pending_decisions() == (decision_id,)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert box["decisions"] == {"builtin__write_file": "timeout"}
    assert controller._announced_pending_decision_ids == set()


@pytest.mark.parametrize(
    "decision_type", ["approval", "skill_install", "skill_script"]
)
@pytest.mark.parametrize("lifecycle", ["detach", "session_switch"])
@pytest.mark.parametrize("elapsed", [5.0, 6.0], ids=["exact", "over"])
@pytest.mark.asyncio
async def test_pausing_at_exhaustion_settles_exact_round_fail_closed(
    decision_type, lifecycle, elapsed
):
    """Detach/switch cannot orphan or later approve an exhausted card."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Timed")
    other = store.create_session(title="Other", activate=False)
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    app = _ThreadApp()
    runtime = ConsoleRuntime(app=app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    clock = _DecisionClock()
    controller.decision_monotonic_clock = clock
    controller.mcp_approval_timeout_seconds = lambda: 5.0
    controller.skill_install_confirm_timeout_seconds = lambda: 5.0
    controller.skill_script_confirm_timeout_seconds = lambda: 5.0
    mounted: list[object | None] = []
    view = _View(
        {"set_pending_decision": lambda projection: mounted.append(projection) or True}
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)

    pending_ids = functools.partial(
        _pending_ids_for_type, controller, session.id, decision_type
    )
    stale_allow = functools.partial(
        _resolve_stale_allow, controller, decision_type
    )
    if decision_type == "approval":
        worker, box = _arm(controller, session.id, call=_risk_row())
    elif decision_type == "skill_install":
        worker, box = _arm_install(controller, session.id)
    else:
        worker, box = _arm_script(controller, session.id)
    assert await _settle(lambda: bool(pending_ids()))
    decision_id = pending_ids()[0]
    assert await _settle(
        lambda: controller._answerable_decision_by_session
        == {session.id: decision_id}
    )

    clock.advance(elapsed)
    mounted.clear()
    if lifecycle == "detach":
        assert runtime.detach_view(view, generation)
    else:
        controller.switch_session(other.id)

    # A click already queued for the exhausted card must lose to the exact
    # fail-closed transition performed before detach/switch returns.
    stale_allow(decision_id)
    worker.join(timeout=5)
    assert not worker.is_alive()
    if decision_type == "approval":
        assert box["decisions"] == {"builtin__write_file": "timeout"}
    elif decision_type == "skill_install":
        assert box["allowed"] is False
    else:
        assert box["decision"] == {"allow": False, "remember": False}
    assert pending_ids() == []
    assert decision_id not in controller._parked_approval_payloads
    assert decision_id not in controller._parked_skill_install_payloads
    assert decision_id not in controller._parked_skill_script_payloads
    assert decision_id not in controller._announced_pending_decision_ids
    assert session.id not in controller._answerable_decision_by_session

    mounted.clear()
    if lifecycle == "detach":
        successor = _View(
            {
                "set_pending_decision": lambda projection: (
                    mounted.append(projection) or True
                )
            }
        )
        successor_generation = runtime.attach_view(successor)
        assert runtime.finish_view_reconciliation(successor, successor_generation)
    else:
        controller.switch_session(session.id)
    assert all(projection is None for projection in mounted)
    assert controller.pending_decision_projection(session.id) is None


@pytest.mark.parametrize(
    "decision_type", ["approval", "skill_install", "skill_script"]
)
@pytest.mark.parametrize("elapsed", [5.0, 6.0], ids=["exact", "over"])
@pytest.mark.asyncio
async def test_reprojection_never_renders_an_exhausted_head(
    decision_type, elapsed
):
    """Production re-projection settles before exposing a zero-budget card."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Timed reproject")
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    app = _ThreadApp()
    runtime = ConsoleRuntime(app=app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    clock = _DecisionClock()
    controller.decision_monotonic_clock = clock
    controller.mcp_approval_timeout_seconds = lambda: 5.0
    controller.skill_install_confirm_timeout_seconds = lambda: 5.0
    controller.skill_script_confirm_timeout_seconds = lambda: 5.0
    rendered: list[object | None] = []
    view = _View(
        {"set_pending_decision": lambda projection: rendered.append(projection) or True}
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)

    if decision_type == "approval":
        worker, box = _arm(controller, session.id, call=_risk_row())
    elif decision_type == "skill_install":
        worker, box = _arm_install(controller, session.id)
    else:
        worker, box = _arm_script(controller, session.id)
    pending_ids = functools.partial(
        _pending_ids_for_type, controller, session.id, decision_type
    )
    assert await _settle(lambda: bool(pending_ids()))
    decision_id = pending_ids()[0]
    assert controller._answerable_decision_by_session == {
        session.id: decision_id
    }

    clock.advance(elapsed)
    rendered.clear()
    controller.project_pending_decision_for_active_session()
    _resolve_stale_allow(controller, decision_type, decision_id)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert rendered and all(projection is None for projection in rendered)
    assert app.notifications == []
    assert pending_ids() == []
    assert decision_id not in controller._parked_approval_payloads
    assert decision_id not in controller._parked_skill_install_payloads
    assert decision_id not in controller._parked_skill_script_payloads
    assert decision_id not in controller._announced_pending_decision_ids
    assert session.id not in controller._answerable_decision_by_session
    if decision_type == "approval":
        assert box["decisions"] == {"builtin__write_file": "timeout"}
    elif decision_type == "skill_install":
        assert box["allowed"] is False
    else:
        assert box["decision"] == {"allow": False, "remember": False}

    rendered.clear()
    controller.project_pending_decision_for_active_session()
    assert all(projection is None for projection in rendered)
    assert controller.pending_decision_projection(session.id) is None


@pytest.mark.asyncio
async def test_cross_type_publication_keeps_first_admitted_round_as_fifo_head():
    """An MCP admission paused before publication blocks a later install."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="FIFO")
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    app = _ThreadApp()
    controller.app = app
    controller.mcp_approval_timeout_seconds = lambda: 60.0
    controller.skill_install_confirm_timeout_seconds = lambda: 60.0
    mounted: list[object | None] = []
    controller.set_pending_decision = lambda projection: (
        mounted.append(projection) or True
    )
    controller.set_pending_skill_install = lambda _payload: True

    first_admitted = threading.Event()
    release_first = threading.Event()
    original_add_pending_round = controller.add_pending_round
    first_call = True

    def _barrier_add_pending_round(session_id: str, round_id: str) -> None:
        nonlocal first_call
        if first_call:
            first_call = False
            first_admitted.set()
            assert release_first.wait(timeout=5)
        original_add_pending_round(session_id, round_id)

    controller.add_pending_round = _barrier_add_pending_round
    mcp_thread, _mcp_box = _arm(controller, session.id, call=_risk_row())
    assert first_admitted.wait(timeout=3)
    mcp_id = _armed_round_ids(controller, session.id)[0]

    install_thread, _install_box = _arm_install(controller, session.id)
    assert await _settle(
        lambda: bool(controller.pending_skill_install_ids())
        and (bool(mounted) or bool(app.notifications)),
        seconds=3.0,
    )
    install_id = controller.pending_skill_install_ids()[0]
    assert mounted == [], "later install mounted before the admitted MCP round"

    release_first.set()
    assert await _settle(
        lambda: bool(mounted)
        and getattr(mounted[-1], "decision_id", None) == mcp_id,
        seconds=3.0,
    )
    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=mcp_id
    )
    mcp_thread.join(timeout=5)
    assert await _settle(
        lambda: bool(mounted)
        and getattr(mounted[-1], "decision_id", None) == install_id,
        seconds=3.0,
    )
    controller.resolve_pending_skill_install(False, request_id=install_id)
    install_thread.join(timeout=5)
    assert not mcp_thread.is_alive()
    assert not install_thread.is_alive()


@pytest.mark.asyncio
async def test_mixed_rounds_project_only_one_stable_fifo_head():
    """MCP/install/script share one ordered projection and one answerable head."""
    runtime, controller, _store, session, _app = _detached_rig(
        timeout_seconds=60.0
    )
    controller.skill_install_confirm_timeout_seconds = lambda: 60.0
    controller.skill_script_confirm_timeout_seconds = lambda: 60.0
    await _leave(runtime)

    mcp_thread, _mcp_box = _arm(controller, session.id, call=_risk_row())
    assert await _wait_for_round(controller, session.id)
    mcp_id = _armed_round_ids(controller, session.id)[0]
    install_thread, _install_box = _arm_install(controller, session.id)
    assert await _settle(lambda: bool(controller.pending_skill_install_ids()))
    install_id = controller.pending_skill_install_ids()[0]
    script_thread, _script_box = _arm_script(controller, session.id)
    assert await _settle(lambda: bool(controller.pending_skill_script_ids()))
    script_id = controller.pending_skill_script_ids()[0]

    mounted: list[tuple[str, dict | None]] = []
    view = _View(
        {
            "set_pending_approval": lambda payload: mounted.append(
                ("approval", payload)
            )
            or True,
            "set_pending_skill_install": lambda payload: mounted.append(
                ("skill_install", payload)
            )
            or True,
            "set_pending_skill_script": lambda payload: mounted.append(
                ("skill_script", payload)
            )
            or True,
        }
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)

    visible = [(kind, payload) for kind, payload in mounted if payload is not None]
    assert len(visible) == 1, f"mixed rounds mounted concurrently: {visible}"
    assert visible[0][0] == "approval"
    assert visible[0][1]["round_id"] == mcp_id
    projection = controller.pending_decision_projection(session.id)
    assert projection is not None
    assert projection.decision_id == mcp_id
    assert controller._answerable_decision_by_session == {session.id: mcp_id}

    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=mcp_id
    )
    mcp_thread.join(timeout=5)
    assert await _settle(
        lambda: any(
            kind == "skill_install"
            and payload is not None
            and payload.get("request_id") == install_id
            for kind, payload in mounted
        )
    ), f"install was not promoted after MCP head: {mounted}"
    assert controller._answerable_decision_by_session == {
        session.id: install_id
    }
    controller.resolve_pending_skill_install(False, request_id=install_id)
    install_thread.join(timeout=5)
    assert await _settle(
        lambda: any(
            kind == "skill_script"
            and payload is not None
            and payload.get("request_id") == script_id
            for kind, payload in mounted
        )
    ), f"script was not promoted after install head: {mounted}"
    assert controller._answerable_decision_by_session == {session.id: script_id}
    controller.resolve_pending_skill_script(False, False, request_id=script_id)
    script_thread.join(timeout=5)


@pytest.mark.asyncio
async def test_render_failure_keeps_exact_round_paused_and_retryable():
    """A projection exception cannot consume time or tear down domain work."""
    runtime, controller, _store, session, app = _detached_rig(
        timeout_seconds=5.0
    )
    clock = _DecisionClock()
    controller.decision_monotonic_clock = clock
    failed = threading.Event()

    def _fail_projection(_projection) -> bool:
        failed.set()
        raise RuntimeError("render failed")

    outgoing = runtime.view
    generation = runtime._attached_generation
    assert await runtime.leave_console(outgoing, generation)
    failing_view = _View(
        {
            "set_pending_decision": _fail_projection,
            "set_pending_approval": lambda _payload: True,
        }
    )
    failing_generation = runtime.attach_view(failing_view)
    assert runtime.finish_view_reconciliation(failing_view, failing_generation)

    thread, box = _arm(controller, session.id, call=_risk_row())
    assert failed.wait(timeout=3), "projection failure barrier was never reached"
    assert thread.is_alive(), "render failure tore down the pending round"
    assert _round_is_claimable(controller, session.id)
    projection = controller.pending_decision_projection(session.id)
    assert projection is not None
    assert projection.remaining_active_seconds == pytest.approx(5.0)
    assert app.notification_posted.wait(timeout=3)
    assert app.notifications, "hidden render failure raised no safe notice"

    # A later successful reconciliation mounts the SAME ID and starts time.
    assert runtime.detach_view(failing_view, failing_generation)
    mounted: list[dict | None] = []
    successor = _View(
        {
            "set_pending_approval": lambda payload: (
                mounted.append(payload) or True
            )
        }
    )
    successor_generation = runtime.attach_view(successor)
    assert runtime.finish_view_reconciliation(successor, successor_generation)
    assert mounted[-1]["round_id"] == projection.decision_id
    clock.advance(5.0)
    assert controller.expire_pending_decisions() == (projection.decision_id,)
    thread.join(timeout=5)
    assert box["decisions"] == {"builtin__write_file": "timeout"}
    assert controller._announced_pending_decision_ids == set()


@pytest.mark.parametrize(
    ("decision_type", "arm", "pending_ids", "resolve"),
    [
        (
            "approval",
            lambda controller, session_id: _arm(
                controller, session_id, call=_risk_row()
            ),
            lambda controller, session_id: _armed_round_ids(
                controller, session_id
            ),
            lambda controller, decision_id: controller.resolve_pending_approval(
                {"builtin__write_file": "deny"}, round_id=decision_id
            ),
        ),
        (
            "skill_install",
            _arm_install,
            lambda controller, _session_id: controller.pending_skill_install_ids(),
            lambda controller, decision_id: controller.resolve_pending_skill_install(
                False, request_id=decision_id
            ),
        ),
        (
            "skill_script",
            _arm_script,
            lambda controller, _session_id: controller.pending_skill_script_ids(),
            lambda controller, decision_id: controller.resolve_pending_skill_script(
                False, False, request_id=decision_id
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_late_projection_failure_cannot_resurrect_terminal_announcement(
    decision_type, arm, pending_ids, resolve
):
    """A pop that wins the race makes a later projection failure inert."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Race")
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    app = _ThreadApp()
    runtime = ConsoleRuntime(app=app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    controller.mcp_approval_timeout_seconds = lambda: 60.0
    controller.skill_install_confirm_timeout_seconds = lambda: 60.0
    controller.skill_script_confirm_timeout_seconds = lambda: 60.0

    projection_entered = threading.Event()
    release_projection = threading.Event()
    late_projection_thread: list[threading.Thread] = []

    def _project(_projection) -> bool:
        if (
            late_projection_thread
            and threading.current_thread() is late_projection_thread[0]
        ):
            projection_entered.set()
            assert release_projection.wait(timeout=5)
            raise RuntimeError("injected late render failure")
        return True

    view = _View({"set_pending_decision": _project})
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    worker, _box = arm(controller, session.id)
    assert await _settle(lambda: bool(pending_ids(controller, session.id)))
    decision_id = pending_ids(controller, session.id)[0]
    projection = controller.pending_decision_projection(session.id)
    assert projection is not None
    assert projection.decision_type == decision_type
    assert app.notifications == []
    controller._announce_hidden_decision(
        decision_type, "wrong-session", decision_id
    )
    controller._announce_hidden_decision(
        decision_type, session.id, "missing-decision-id"
    )
    assert app.notifications == []
    assert controller._announced_pending_decision_ids == set()

    def _late_projection() -> None:
        runtime._project_pending_decision_to_attached_view(projection)

    late_thread = threading.Thread(target=_late_projection, daemon=True)
    late_projection_thread.append(late_thread)
    late_thread.start()
    assert projection_entered.wait(timeout=3)

    resolve(controller, decision_id)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert pending_ids(controller, session.id) == []
    assert decision_id not in controller._announced_pending_decision_ids

    release_projection.set()
    late_thread.join(timeout=5)
    assert not late_thread.is_alive()
    assert app.notifications == []
    assert decision_id not in controller._announced_pending_decision_ids


@pytest.mark.asyncio
async def test_hidden_announcer_rejects_session_type_and_id_mismatch():
    """Only the exact live registry record can admit an app-wide notice."""
    runtime, controller, _store, session, app = _detached_rig(
        timeout_seconds=60.0
    )
    outgoing = runtime.view
    outgoing_generation = runtime._attached_generation
    assert await runtime.leave_console(outgoing, outgoing_generation)
    view = _View({"set_pending_decision": lambda _projection: True})
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    worker, _box = _arm(controller, session.id, call=_risk_row())
    assert await _settle(
        lambda: bool(_armed_round_ids(controller, session.id))
    )
    decision_id = _armed_round_ids(controller, session.id)[0]

    controller._announce_hidden_decision(
        "approval", "wrong-session", decision_id
    )
    controller._announce_hidden_decision(
        "skill_install", session.id, decision_id
    )
    controller._announce_hidden_decision(
        "approval", session.id, "missing-decision-id"
    )
    assert app.notifications == []
    assert controller._announced_pending_decision_ids == set()

    controller._announce_hidden_decision("approval", session.id, decision_id)
    assert len(app.notifications) == 1
    assert controller._announced_pending_decision_ids == {decision_id}
    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=decision_id
    )
    worker.join(timeout=5)
    assert controller._announced_pending_decision_ids == set()


@pytest.mark.parametrize(
    ("decision_type", "arm", "pending_ids", "resolve"),
    [
        (
            "approval",
            lambda controller, session_id: _arm(
                controller, session_id, call=_risk_row()
            ),
            lambda controller, session_id: _armed_round_ids(
                controller, session_id
            ),
            lambda controller, decision_id: controller.resolve_pending_approval(
                {"builtin__write_file": "deny"}, round_id=decision_id
            ),
        ),
        (
            "skill_install",
            _arm_install,
            lambda controller, _session_id: controller.pending_skill_install_ids(),
            lambda controller, decision_id: controller.resolve_pending_skill_install(
                False, request_id=decision_id
            ),
        ),
        (
            "skill_script",
            _arm_script,
            lambda controller, _session_id: controller.pending_skill_script_ids(),
            lambda controller, decision_id: controller.resolve_pending_skill_script(
                False, False, request_id=decision_id
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_hidden_notice_marker_records_only_successful_delivery(
    decision_type, arm, pending_ids, resolve
):
    """A failed app notice remains retryable and does not claim the stable ID."""
    runtime, controller, _store, session, app = _detached_rig(
        timeout_seconds=60.0
    )
    controller.skill_install_confirm_timeout_seconds = lambda: 60.0
    controller.skill_script_confirm_timeout_seconds = lambda: 60.0
    await _leave(runtime)

    delivered_notify = app.notify
    attempts = 0

    def _flaky_notify(message, *, severity="information", **kwargs) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected notification failure")
        delivered_notify(message, severity=severity, **kwargs)

    app.notify = _flaky_notify
    worker, _box = arm(controller, session.id)
    assert await _settle(lambda: bool(pending_ids(controller, session.id)))
    decision_id = pending_ids(controller, session.id)[0]
    assert await _settle(lambda: attempts == 1)
    assert app.notifications == []
    assert decision_id not in controller._announced_pending_decision_ids

    controller._announce_hidden_decision(
        decision_type, session.id, decision_id
    )
    assert attempts == 2
    assert len(app.notifications) == 1
    assert controller._announced_pending_decision_ids == {decision_id}
    controller._announce_hidden_decision(
        decision_type, session.id, decision_id
    )
    assert attempts == 2

    resolve(controller, decision_id)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert controller._announced_pending_decision_ids == set()


@pytest.mark.asyncio
async def test_background_decision_updates_runtime_attention_until_resolution():
    """A background round updates mounted shell state without a view remount."""
    runtime, controller, store, active_session, app = _detached_rig(
        timeout_seconds=60.0
    )
    background_session = store.create_session(title="Background", activate=False)
    assert store.active_session_id == active_session.id
    app.console_attention_updates = []
    app.set_console_attention_projection = (
        lambda value: app.console_attention_updates.append(bool(value))
    )

    worker, _box = _arm(
        controller, background_session.id, call=_risk_row()
    )
    assert await _settle(
        lambda: bool(_armed_round_ids(controller, background_session.id))
    )
    decision_id = _armed_round_ids(controller, background_session.id)[0]

    assert runtime.console_needs_attention is True
    assert app.console_attention_updates[-1] is True

    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=decision_id
    )
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert runtime.console_needs_attention is False
    assert app.console_attention_updates[-1] is False


@pytest.mark.asyncio
async def test_each_hidden_decision_id_emits_one_sanitized_notice():
    """All three types notify once without decision bodies or opaque IDs."""
    runtime, controller, _store, session, app = _detached_rig(
        timeout_seconds=60.0
    )
    controller.skill_install_confirm_timeout_seconds = lambda: 60.0
    controller.skill_script_confirm_timeout_seconds = lambda: 60.0
    await _leave(runtime)

    mcp_thread, _mcp_box = _arm(
        controller,
        session.id,
        call=_pending(
            server_key="agent:builtin",
            tool_name="SECRET_TOOL",
            llm_name="builtin__SECRET_TOOL",
            reason="SECRET_REASON",
        ),
    )
    assert await _wait_for_round(controller, session.id)
    install_thread, _install_box = _arm_install(controller, session.id)
    assert await _settle(lambda: bool(controller.pending_skill_install_ids()))
    script_thread, _script_box = _arm_script(controller, session.id)
    assert await _settle(lambda: bool(controller.pending_skill_script_ids()))
    assert len(app.notifications) == 3

    ids = (
        _armed_round_ids(controller, session.id)[0],
        controller.pending_skill_install_ids()[0],
        controller.pending_skill_script_ids()[0],
    )
    combined = "\n".join(message for message, _severity in app.notifications)
    for forbidden in (
        "SECRET_TOOL",
        "SECRET_REASON",
        "example.invalid",
        "run.py",
        "example",
        *ids,
    ):
        assert forbidden not in combined
    assert combined.count("Return to Console") == 3

    for kind, decision_id in zip(
        ("approval", "skill_install", "skill_script"), ids, strict=True
    ):
        controller._announce_hidden_decision(kind, session.id, decision_id)
    assert len(app.notifications) == 3, "a stable ID was announced twice"

    controller.resolve_pending_approval(
        {"builtin__SECRET_TOOL": "deny"}, round_id=ids[0]
    )
    mcp_thread.join(timeout=5)
    controller.resolve_pending_skill_install(False, request_id=ids[1])
    install_thread.join(timeout=5)
    controller.resolve_pending_skill_script(False, False, request_id=ids[2])
    script_thread.join(timeout=5)
    assert controller._announced_pending_decision_ids == set()


async def _background_decision_rig():
    """Keep Console attached to A while decisions arm for hostile-named B."""
    runtime, controller, store, active, app = _detached_rig(
        timeout_seconds=60.0
    )
    controller.skill_install_confirm_timeout_seconds = lambda: 60.0
    controller.skill_script_confirm_timeout_seconds = lambda: 60.0
    background = store.create_session(
        title="https://evil.invalid/private/body SECRET_TITLE",
        workspace_id="/Users/victim/private/SECRET_WORKSPACE",
        activate=False,
    )
    legacy_park_calls: list[str] = []
    outgoing = runtime.view
    outgoing_generation = runtime._attached_generation
    assert await runtime.leave_console(outgoing, outgoing_generation)
    view = _View({"park_pending_approval": legacy_park_calls.append})
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    assert store.active_session_id == active.id
    return controller, background, app, legacy_park_calls


@pytest.mark.asyncio
async def test_two_rapid_background_rounds_announce_exact_ids_without_legacy_park():
    """Same-type background rounds neither coalesce nor expose session labels."""
    controller, background, app, legacy_park_calls = (
        await _background_decision_rig()
    )

    thread_a, _box_a = _arm(controller, background.id, call=_risk_row())
    thread_b, _box_b = _arm(
        controller,
        background.id,
        call=_pending(
            server_key="agent:builtin",
            tool_name="delete_note",
            llm_name="builtin__delete_note",
            reason="SECRET_BODY",
        ),
    )
    assert await _settle(
        lambda: len(_armed_round_ids(controller, background.id)) == 2,
        seconds=3.0,
    )
    round_a, round_b = _armed_round_ids(controller, background.id)
    assert legacy_park_calls == []
    assert len(app.notifications) == 2
    assert controller._announced_pending_decision_ids == {round_a, round_b}
    rendered = "\n".join(message for message, _severity in app.notifications)
    for secret in (
        "evil.invalid",
        "/Users/victim",
        "SECRET_TITLE",
        "SECRET_WORKSPACE",
        "SECRET_BODY",
        background.id,
        round_a,
        round_b,
    ):
        assert secret not in rendered
    assert rendered.count("Return to Console") == 2

    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=round_a
    )
    assert await _settle(
        lambda: controller._announced_pending_decision_ids == {round_b},
        seconds=3.0,
    )
    controller.resolve_pending_approval(
        {"builtin__delete_note": "deny"}, round_id=round_b
    )
    assert await _settle(
        lambda: not controller._announced_pending_decision_ids,
        seconds=3.0,
    )
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert controller._announced_pending_decision_ids == set()

    # Stable IDs are never recycled; a genuinely new round can announce.
    thread_c, _box_c = _arm(controller, background.id, call=_risk_row())
    assert await _settle(
        lambda: bool(_armed_round_ids(controller, background.id)), seconds=3.0
    )
    round_c = _armed_round_ids(controller, background.id)[0]
    assert round_c not in {round_a, round_b}
    assert len(app.notifications) == 3
    assert controller._announced_pending_decision_ids == {round_c}
    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=round_c
    )
    thread_c.join(timeout=5)
    assert controller._announced_pending_decision_ids == set()


@pytest.mark.asyncio
async def test_mixed_background_rounds_announce_individually_and_cleanup_exactly():
    """Mixed rounds share no screen notice registry and clean only their ID."""
    controller, background, app, legacy_park_calls = (
        await _background_decision_rig()
    )

    mcp_thread, _mcp_box = _arm(controller, background.id, call=_risk_row())
    install_thread, _install_box = _arm_install(controller, background.id)
    script_thread, _script_box = _arm_script(controller, background.id)
    assert await _settle(
        lambda: (
            len(_armed_round_ids(controller, background.id)) == 1
            and len(controller.pending_skill_install_ids()) == 1
            and len(controller.pending_skill_script_ids()) == 1
        ),
        seconds=3.0,
    )
    mcp_id = _armed_round_ids(controller, background.id)[0]
    install_id = controller.pending_skill_install_ids()[0]
    script_id = controller.pending_skill_script_ids()[0]
    live_ids = {mcp_id, install_id, script_id}
    assert legacy_park_calls == []
    assert len(app.notifications) == 3
    assert controller._announced_pending_decision_ids == live_ids
    rendered = "\n".join(message for message, _severity in app.notifications)
    assert rendered.count("Return to Console") == 3
    for secret in (
        "evil.invalid",
        "/Users/victim",
        "SECRET_TITLE",
        "SECRET_WORKSPACE",
        background.id,
        *live_ids,
    ):
        assert secret not in rendered

    # Resolving the non-head install round cannot clear either live sibling.
    controller.resolve_pending_skill_install(False, request_id=install_id)
    install_thread.join(timeout=5)
    assert controller._announced_pending_decision_ids == {mcp_id, script_id}

    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=mcp_id
    )
    controller.resolve_pending_skill_script(False, False, request_id=script_id)
    mcp_thread.join(timeout=5)
    script_thread.join(timeout=5)
    assert controller._announced_pending_decision_ids == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["session_close", "app_dispose"])
async def test_terminal_scope_releases_all_three_announcement_ids(terminal):
    """Destructive terminal scopes release every exact mixed decision ID."""
    runtime, controller, _store, session, _app = _detached_rig(
        timeout_seconds=60.0
    )
    controller.skill_install_confirm_timeout_seconds = lambda: 60.0
    controller.skill_script_confirm_timeout_seconds = lambda: 60.0
    await _leave(runtime)

    mcp_thread, _mcp_box = _arm(controller, session.id, call=_risk_row())
    install_thread, _install_box = _arm_install(controller, session.id)
    script_thread, _script_box = _arm_script(controller, session.id)
    assert await _settle(
        lambda: len(controller._announced_pending_decision_ids) == 3,
        seconds=3.0,
    )

    if terminal == "session_close":
        impact = controller.lifecycle_impact(session_id=session.id)
        await runtime.close_session(
            session.id,
            expected_revision=impact.revision,
        )
    else:
        await asyncio.wait_for(runtime.dispose(), timeout=10)
    for thread in (mcp_thread, install_thread, script_thread):
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert controller._announced_pending_decision_ids == set()


def test_unified_router_never_invokes_legacy_type_setters():
    """Runtime-owned mixed projection is exclusive across all three types."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Unified")
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    controller.app = _ThreadApp()
    projected: list[object] = []
    projected_event = threading.Event()

    def _project(projection) -> bool:
        projected.append(projection)
        projected_event.set()
        return True

    def _legacy_called(_payload) -> None:
        raise AssertionError("legacy type setter ran while unified router was bound")

    controller.set_pending_decision = _project
    controller.set_pending_approval = _legacy_called
    controller.set_pending_skill_install = _legacy_called
    controller.set_pending_skill_script = _legacy_called

    approval_thread, _ = _arm(controller, session.id, call=_risk_row())
    assert projected_event.wait(timeout=2)
    approval_id = projected[-1].decision_id
    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=approval_id
    )
    approval_thread.join(timeout=2)
    assert not approval_thread.is_alive()

    projected_event.clear()
    install_thread, _ = _arm_install(controller, session.id)
    assert projected_event.wait(timeout=2)
    install_id = projected[-1].decision_id
    controller.resolve_pending_skill_install(False, request_id=install_id)
    install_thread.join(timeout=2)
    assert not install_thread.is_alive()

    projected_event.clear()
    script_thread, _ = _arm_script(controller, session.id)
    assert projected_event.wait(timeout=2)
    script_id = projected[-1].decision_id
    controller.resolve_pending_skill_script(False, False, request_id=script_id)
    script_thread.join(timeout=2)
    assert not script_thread.is_alive()


@pytest.mark.asyncio
async def test_a_round_armed_with_a_view_attached_does_not_double_announce():
    """The app-wide notice is the DETACHED path only.

    With a view attached the screen's own card/park seams do the
    surfacing; adding a second app-level toast on top would double every
    mounted round's announcement.
    """
    runtime, controller, store, session, app = _detached_rig()
    mounted: list[dict | None] = []
    outgoing = runtime.view
    outgoing_generation = runtime._attached_generation
    assert await runtime.leave_console(outgoing, outgoing_generation)
    view = _View(
        {
            "set_pending_approval": lambda payload: (
                mounted.append(payload) or True
            )
        }
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)

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
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)

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
    view = _View({"set_pending_approval": mounted.append})
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    assert mounted == [], f"attach pushed a card with nothing armed: {mounted}"


# ---------------------------------------------------------------------------
# Two same-session rounds each get their turn (task-15661, FIXED in PR0)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_headless_rounds_each_mount_in_turn():
    """`_parked_approval_payloads` is keyed by ROUND: no round is stranded.

    Pre-PR0 this slot was per-SESSION and last-armed-wins, so of two rounds
    armed while detached only the newest had a surviving payload and an
    attach could mount only ONE card -- the older round stayed registered,
    badge lit, with no way to answer it until its own timeout. That was
    `console_chat_controller.py`'s documented "accepted scope limitation",
    filed as task-15661, and this test pinned it as a measured fact.

    task-15661 is now FIXED. Each round retains its own payload, and the
    mounted card is the session's FIFO HEAD -- the oldest-armed round.
    Attaching mounts round A; once A resolves, B is promoted and mounts in
    its turn. Everything above the mount assertions is unchanged: both
    rounds still register independently and both still announce.
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

    round_b = [
        r for r in _armed_round_ids(controller, session.id) if r != round_a
    ][0]

    # Attaching mounts the FIFO HEAD -- round A, armed FIRST. Pre-PR0 this
    # was round B, because B's arm had overwritten A's payload.
    mounted: list[dict | None] = []
    view = _View({"set_pending_approval": mounted.append})
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    assert mounted and mounted[-1] is not None, "attach mounted nothing at all"
    names = [c["llm_name"] for c in mounted[-1]["calls"]]
    assert names == ["builtin__write_file"], (
        f"attach must mount the oldest-armed round's card; got {names}"
    )
    assert mounted[-1]["round_id"] == round_a

    # Resolving the head promotes B. Round A's own teardown re-derives the
    # card from the session's remaining head, so B mounts without anyone
    # asking -- and a fresh remount (the attach path) agrees.
    controller.resolve_pending_approval(
        {"builtin__write_file": "deny"}, round_id=round_a
    )
    thread_a.join(timeout=5)
    assert await _settle(
        lambda: bool(mounted) and (mounted[-1] or {}).get("round_id") == round_b,
        seconds=3.0,
    ), f"round B was never promoted after the head resolved: {mounted[-1]}"

    mounted.clear()
    assert controller.remount_pending_approval_for_active_session() is True
    assert mounted[-1]["round_id"] == round_b
    assert [c["llm_name"] for c in mounted[-1]["calls"]] == ["builtin__delete_note"]

    controller.resolve_pending_approval(
        {"builtin__delete_note": "deny"}, round_id=round_b
    )
    thread_b.join(timeout=5)
    assert box_a["decisions"] == {"builtin__write_file": "deny"}
    assert box_b["decisions"] == {"builtin__delete_note": "deny"}

    # Nothing left armed -> the card clears rather than showing a corpse.
    assert await _settle(lambda: mounted[-1] is None, seconds=3.0), mounted[-1]


# ---------------------------------------------------------------------------
# DETACH AND CANCELLATION SAFETY PINS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_navigation_leaves_an_armed_round_pending():
    """Detaching the sole projection never decides app-owned work."""
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=60.0)
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the round never armed"
    view = runtime.view
    generation = runtime._attached_generation

    assert await runtime.leave_console(view, generation) is True
    assert await _quiet(lambda: "decisions" in box, seconds=2.0), (
        "navigation resolved an app-owned approval round"
    )
    assert _round_is_claimable(controller, session.id)

    controller.resolve_pending_approval(
        {"write_file": "deny"},
        round_id=_armed_round_ids(controller, session.id)[0],
    )
    thread.join(timeout=5)
    assert box["decisions"] == {"write_file": "deny"}, box["decisions"]


@pytest.mark.asyncio
async def test_exact_user_resolution_works_after_detach_and_reattach():
    """A preserved round keeps its exact identity across view replacement."""
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=60.0)
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the round never armed"
    round_id = _armed_round_ids(controller, session.id)[0]
    outgoing = runtime.view
    outgoing_generation = runtime._attached_generation
    assert await runtime.leave_console(outgoing, outgoing_generation) is True

    successor = _View()
    successor_generation = runtime.attach_view(successor)
    assert successor_generation is not None
    assert runtime.view is successor
    controller.resolve_pending_approval(
        {"write_file": "deny"}, round_id=round_id
    )
    thread.join(timeout=5)
    assert box["decisions"] == {"write_file": "deny"}, box["decisions"]


@pytest.mark.asyncio
async def test_repeated_detach_never_predenies_a_later_round():
    """Projection churn cannot leave a cancellation latch behind."""
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=60.0)
    for _index in range(2):
        view = runtime.view
        generation = runtime._attached_generation
        assert await runtime.leave_console(view, generation) is True
        assert not controller._shutdown_requested.is_set()
        assert runtime.attach_view(_View()) is not None

    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the later round never armed"
    assert await _quiet(lambda: "decisions" in box, seconds=2.0), (
        "detach history pre-denied a later round"
    )

    controller.resolve_pending_approval(
        {"write_file": "deny"},
        round_id=_armed_round_ids(controller, session.id)[0],
    )
    thread.join(timeout=5)
    assert box["decisions"] == {"write_file": "deny"}


@pytest.mark.asyncio
async def test_never_visited_round_survives_its_first_attach_and_detach():
    """A wake-at-launch decision is unaffected by the first Console visit."""
    runtime, controller, _store, session, _app = _never_visited_rig()
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the round never armed"

    view = _View()
    generation = runtime.attach_view(view)
    assert generation is not None
    assert await runtime.leave_console(view, generation) is True
    assert await _quiet(lambda: "decisions" in box, seconds=2.0), (
        "the first attach/detach resolved a never-visited round"
    )
    controller.resolve_pending_approval(
        {"write_file": "deny"},
        round_id=_armed_round_ids(controller, session.id)[0],
    )
    thread.join(timeout=5)
    assert box["decisions"] == {"write_file": "deny"}


@pytest.mark.asyncio
async def test_app_exit_denies_a_round_armed_before_any_visit_ever_opened():
    """The same orphaned binding also beat `begin_shutdown` (app exit).

    Once a visit had opened (replacing the constructor Event), app exit
    set the CURRENT `_shutdown_requested` and the headless-cancel Event --
    neither of which the born-headless round had bound. A worker poll
    that only dies at its own deadline, through the app's shutdown.
    """
    runtime, controller, _store, session, _app = _never_visited_rig()
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the round never armed"

    runtime.attach_view(_View())
    await asyncio.wait_for(runtime.dispose(), timeout=10)
    thread.join(timeout=10)
    assert controller._disposed is True
    assert not thread.is_alive(), (
        "app exit never reached the born-headless round"
    )
    assert box["decisions"] == {"write_file": "deny"}, box.get("decisions")


@pytest.mark.asyncio
async def test_a_born_headless_round_waits_and_app_exit_without_a_visit_denies_it():
    """The mirror pin: with NO visit ever opened, `begin_shutdown` still
    denies the born-headless round (whatever Event it bound, exit sets
    it), and the round WAITS rather than self-denying before that."""
    runtime, controller, _store, session, _app = _never_visited_rig()
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the round never armed"
    assert await _quiet(lambda: "decisions" in box, seconds=2.0), (
        "the born-headless round self-denied instead of waiting"
    )

    await asyncio.wait_for(runtime.dispose(), timeout=10)
    thread.join(timeout=10)
    assert not thread.is_alive(), "exit-with-no-visit never reached the round"
    assert box["decisions"] == {"write_file": "deny"}, box.get("decisions")


@pytest.mark.asyncio
async def test_round_binding_answers_to_explicit_run_cancel_not_view_lifecycle():
    """The owning run can cancel a round after its projection detaches."""
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=60.0)
    cancel_event = threading.Event()
    controller._active_cancel_events[session.id] = cancel_event
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id), "the round never armed"
    view = runtime.view
    generation = runtime._attached_generation

    assert await runtime.leave_console(view, generation) is True
    assert await _quiet(lambda: "decisions" in box, seconds=2.0), (
        "view detachment resolved the run-owned round"
    )

    cancel_event.set()
    thread.join(timeout=5)
    assert not thread.is_alive(), "the owning run cancellation was ignored"
    assert box["decisions"] == {"write_file": "deny"}
    assert controller._announced_pending_decision_ids == set()


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
    assert controller._announced_pending_decision_ids == set()


@pytest.mark.asyncio
async def test_a_configured_deadline_waits_for_answerable_time():
    """Positive timeout is a mounted/answerable allowance, not wall time."""
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=2.0)
    clock = _DecisionClock()
    controller.decision_monotonic_clock = clock
    await _leave(runtime)
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id)

    clock.advance(600.0)
    assert controller.expire_pending_decisions() == ()
    assert "decisions" not in box

    mounted: list[dict | None] = []
    view = _View(
        {
            "set_pending_approval": lambda payload: (
                mounted.append(payload) or True
            )
        }
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    round_id = mounted[-1]["round_id"]
    clock.advance(2.0)
    assert controller.expire_pending_decisions() == (round_id,)
    thread.join(timeout=5)
    assert box["decisions"] == {"write_file": "timeout"}, box["decisions"]


@pytest.mark.asyncio
async def test_no_headless_path_returns_an_approval_without_a_human():
    """Every automatic resolution of a headless round fails CLOSED.

    Deadline, app exit and leave-after-attach are the three ways a round
    can end with nobody answering; none of them may produce an
    `approve_*` verdict.
    """
    verdicts: list[str] = []

    # answerable-time expiry
    runtime, controller, _store, session, _app = _detached_rig(timeout_seconds=1.0)
    clock = _DecisionClock()
    controller.decision_monotonic_clock = clock
    await _leave(runtime)
    thread, box = _arm(controller, session.id)
    assert await _wait_for_round(controller, session.id)
    mounted: list[dict | None] = []
    view = _View(
        {
            "set_pending_approval": lambda payload: (
                mounted.append(payload) or True
            )
        }
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    clock.advance(1.0)
    assert controller.expire_pending_decisions() == (mounted[-1]["round_id"],)
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
        # Runtime ownership makes its app authoritative for every worker-
        # thread UI marshal.  Use the thread-safe app surface this test
        # already provides; the fleet coordinator remains wired to ``app``.
        runtime = ConsoleRuntime(app=thread_app)
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

        def approve_for_session(self, server_key, tool_name, **kwargs) -> None:
            return None

        def is_session_approved(self, server_key, tool_name, **kwargs) -> bool:
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
    # Native calls retain their per-call verdict key even when the round is
    # parked; the payload below still carries the human-facing tool name.
    assert state["names"] == ("c1",), state["names"]
    payload = controller._head_round_payload(
        controller._parked_approval_payloads, session.id
    )
    assert payload["calls"][0]["server_key"] == BUILTIN_TOOL_SERVER_KEY
    assert payload["calls"][0]["reason"] == "risk_floored"

    controller.resolve_pending_approval({"c1": "deny"}, round_id=round_id)
    thread.join(timeout=10)
    assert verdicts.get("c1") not in (None, "proceed"), (
        f"the refusal did not reach the runtime: {verdicts}"
    )
