"""The teardown split: leaving a Console visit vs destroying the runtime.

task-15860 (headless wake), the lifetime landing. `ConsoleChatController.
shutdown()` used to mean both "cancel the user's work" and "this instance
is finished forever", and one flag -- `_shutdown_requested` -- carried
both meanings. With the runtime app-owned and surviving every navigation,
those are different events:

* `ConsoleRuntime.leave_console()` -> `controller.leave_console()` ends ONE
  visit. AC#2's two documented screen-scoped semantics still happen here.
* `ConsoleRuntime.dispose()` -> `controller.shutdown()` is the permanent,
  app-exit form and keeps its old behaviour exactly.

The two AC#2 tests below (`test_leaving_console_still_cancels_a_streaming_
user_turn`, `test_leaving_console_still_denies_a_parked_approval_round`)
are genuine reds for the intermediate state this landing passes through:
with the runtime surviving and no `leave_console()` on the unmount path,
BOTH fail. They were run in that state and observed failing before the
split was wired -- see the lifetime report.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore as _ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_prompt_queue import QueueMutationStatus
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


class ConsoleChatStore(_ConsoleChatStore):
    """Test store whose intentionally db-less sessions are explicitly ephemeral."""

    def create_session(self, **kwargs):
        kwargs.setdefault("ephemeral", self.persistence is None)
        return super().create_session(**kwargs)


class _StalledGateway:
    """Streams one chunk, then never finishes until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.never_release = asyncio.Event()

    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "",
                "resolved_destination": ConsoleResolvedDestination(
                    provider="llama_cpp",
                    model="test-model",
                    endpoint_identity="http://127.0.0.1:9099",
                    egress_class=ConsoleEgressClass.ON_DEVICE,
                ),
            },
        )()

    async def stream_chat(self, resolution, messages, **kwargs):
        self.started.set()
        yield "partial"
        await self.never_release.wait()
        yield "never"


class _ThreadApp:
    """The app surface `request_mcp_approvals` needs to reach its poll loop.

    **Not decoration.** ADR-067 added a no-`app` guard to
    `request_mcp_approvals` that denies every name on the spot when
    `controller.app is None` -- so the two approval-round tests below,
    written before that guard existed, stopped exercising the poll loop
    at all and passed on the guard's verdict instead. Measured
    (task-15860 Task 5): with the visit-cancel check deleted outright --
    fail-open for every session-scoped round -- this whole file was
    still 14/14 green in 0.98s, which is less than one poll interval.
    Wiring an app restores what these tests claim to pin.
    """

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class _View:
    """The smallest thing `ConsoleRuntime` accepts as a view."""

    def __init__(self, hooks: dict | None = None) -> None:
        self._hooks = hooks or {}

    def console_view_hooks(self) -> dict:
        return dict(self._hooks)


def _runtime_with(controller: ConsoleChatController, view: _View) -> ConsoleRuntime:
    """A runtime holding `controller`, attached to `view`."""
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(controller.store)
    runtime.set_chat_controller(controller)
    runtime.attach_view(view)
    return runtime


def _pending_call() -> MCPPendingCall:
    return MCPPendingCall(
        llm_name="write_file",
        server_key="agent:builtin",
        tool_name="write_file",
        server_label="Built-in",
        arguments={},
        reason="risk_floored",
    )


# -- AC#2, semantic 1: leaving still cancels a streaming USER turn ----------


@pytest.mark.asyncio
async def test_leaving_console_still_cancels_a_streaming_user_turn():
    """AC#2 RED: nav-away must still kill the user's in-flight turn.

    Goes red the moment the runtime survives unmount without a per-visit
    teardown: the turn simply keeps streaming into a screen that is gone.
    """
    gateway = _StalledGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = _runtime_with(controller, _View())

    turn = asyncio.create_task(controller.submit_draft("hello"))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)
    session_id = store.active_session_id
    assert controller._active_stream_tasks.get(session_id) is not None

    assert await asyncio.wait_for(runtime.leave_console(), timeout=2) is True
    await asyncio.wait_for(turn, timeout=1)

    assert controller._active_stream_tasks.get(session_id) is None
    assert store.messages_for_session(session_id)[-1].status == "stopped"
    # ...and the runtime itself is emphatically still alive.
    assert runtime.chat_controller is controller
    assert runtime.chat_store is store
    assert runtime.generation == 0


# -- AC#2, semantic 2: leaving still denies a parked approval round --------


@pytest.mark.asyncio
async def test_leaving_console_still_denies_a_parked_approval_round():
    """AC#2 RED: an undecided round at nav-away resolves to `deny`.

    The round polls on a worker thread exactly as production does; nothing
    ever answers it, so only the visit's cancellation signal can end it
    inside the (deliberately generous) timeout below.
    """
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store, provider_gateway=_StalledGateway()
    )
    controller.mcp_approval_timeout_seconds = lambda: 60.0
    # Without an app the round never reaches the poll loop -- see `_ThreadApp`.
    controller.app = _ThreadApp()
    runtime = _runtime_with(controller, _View())

    decisions: dict[str, str] = {}
    armed = threading.Event()

    def _run_round() -> None:
        armed.set()
        decisions.update(
            controller.request_mcp_approvals(
                [_pending_call()], session_id=session.id
            )
        )

    worker = threading.Thread(target=_run_round, daemon=True)
    worker.start()
    assert armed.wait(timeout=2), "the round never armed"
    # Let the poll loop reach its first `event.wait(1.0)`.
    await asyncio.sleep(0.2)

    await asyncio.wait_for(runtime.leave_console(), timeout=5)
    worker.join(timeout=10)

    assert not worker.is_alive(), "the round never resolved after leaving"
    assert decisions == {"write_file": "deny"}, decisions


# -- the owner ruling: a wake turn is NOT a user turn ----------------------


@pytest.mark.asyncio
async def test_leaving_console_does_not_cancel_an_in_flight_wake_turn():
    """Owner ruling: `leave_console` never cancels an `AGENT_WAKE` turn.

    Cancelling it would re-create the "only completes if you stay" gap
    this whole arc exists to close, and a wake turn is structurally the
    same class of work as the fleet survivor AC#2 keeps running.
    """
    gateway = _StalledGateway()
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = _runtime_with(controller, _View())

    # Stand in for a wake turn in flight: the exemption registry and a real
    # (stalled) stream task for the same session, exactly as `submit_draft`
    # leaves them mid-turn.
    async def _stalled_turn() -> None:
        await gateway.never_release.wait()

    task = asyncio.create_task(_stalled_turn())
    await asyncio.sleep(0)
    controller._active_stream_tasks[session.id] = task
    controller._agent_wake_turn_sessions.add(session.id)

    await asyncio.wait_for(runtime.leave_console(), timeout=2)

    assert not task.cancelled() and not task.done(), (
        "leaving Console cancelled a wake turn -- the owner ruling is that "
        "it must not"
    )
    assert controller._active_stream_tasks.get(session.id) is task
    gateway.never_release.set()
    task.cancel()


@pytest.mark.asyncio
async def test_the_wake_exemption_never_outlives_its_turn():
    """A wake turn that finishes leaves nothing exempt behind."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=_StalledGateway()
    )
    with pytest.raises(PermissionError):
        # No coordinator-issued token -> refused before anything runs, which
        # is enough to prove the registry is not populated by the attempt.
        await controller.submit_draft(
            "notice", origin=ConsoleSubmissionOrigin.AGENT_WAKE
        )
    assert controller._agent_wake_turn_sessions == set()


# -- the per-visit Event, captured at ARM time -----------------------------


def test_begin_visit_installs_a_fresh_cancellation_event():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=_StalledGateway()
    )
    first = controller._shutdown_requested
    first.set()

    controller.begin_visit()

    assert controller._shutdown_requested is not first
    assert not controller._shutdown_requested.is_set()
    assert first.is_set(), "the previous visit's Event must stay set forever"


def test_a_disposed_controller_is_never_re_opened():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=_StalledGateway()
    )
    controller.begin_shutdown()
    event = controller._shutdown_requested

    controller.begin_visit()

    assert controller._shutdown_requested is event
    assert controller._shutdown_requested.is_set()


@pytest.mark.asyncio
async def test_a_round_from_the_previous_visit_is_not_resurrected():
    """The arm-time capture, proven by its consequence.

    A round armed during visit 1 keeps polling while the user is on visit
    2. Visit 2 installed a FRESH, unset `_shutdown_requested`. If the poll
    site re-read that attribute instead of the Event it captured at arm
    time, the round would silently un-deny itself and go on waiting (and
    then approving, or timing out into a UI that no longer exists).
    """
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store, provider_gateway=_StalledGateway()
    )
    controller.mcp_approval_timeout_seconds = lambda: 60.0
    # Without an app the round never reaches the poll loop -- see `_ThreadApp`.
    controller.app = _ThreadApp()
    runtime = _runtime_with(controller, _View())

    decisions: dict[str, str] = {}
    armed = threading.Event()
    resolved = threading.Event()

    def _run_round() -> None:
        armed.set()
        decisions.update(
            controller.request_mcp_approvals(
                [_pending_call()], session_id=session.id
            )
        )
        resolved.set()

    worker = threading.Thread(target=_run_round, daemon=True)
    worker.start()
    assert armed.wait(timeout=2)
    await asyncio.sleep(0.2)

    # Visit 1 ends, then visit 2 begins -- and visit 2's fresh Event is
    # installed BEFORE the round has finished unwinding.
    await asyncio.wait_for(runtime.leave_console(), timeout=5)
    runtime.attach_view(_View())
    assert not controller._shutdown_requested.is_set(), (
        "visit 2 must start with a clean cancellation Event"
    )

    assert resolved.wait(timeout=10), (
        "the visit-1 round never resolved -- a poll site re-read "
        "`_shutdown_requested` and saw visit 2's fresh Event"
    )
    assert decisions == {"write_file": "deny"}, decisions


# -- per-visit queue admission --------------------------------------------


@pytest.mark.asyncio
async def test_the_prompt_queue_admits_again_on_the_next_visit():
    """Leaving Console tombstones the queue; it must not stay dead.

    The coordinator's `shutdown()` latch is permanent by construction, and
    with one app-owned controller serving every visit that made the prompt
    queue unusable for the rest of the app's life after the first
    navigation away.
    """
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store, provider_gateway=_StalledGateway()
    )
    runtime = _runtime_with(controller, _View())
    coordinator = controller.prompt_queue_coordinator

    def _admit_status():
        revision = coordinator.registry.snapshot(session.id).revision
        return coordinator.admit(
            session.id, text="queued", expected_revision=revision
        ).status

    # An empty queue reroutes to a normal send -- the "admission is open"
    # answer, and emphatically not `SHUTTING_DOWN`.
    assert _admit_status() is QueueMutationStatus.REROUTE_NORMAL_SEND

    await asyncio.wait_for(runtime.leave_console(), timeout=2)
    assert _admit_status() is QueueMutationStatus.SHUTTING_DOWN, (
        "leaving Console must still tombstone this visit's queue"
    )

    runtime.attach_view(_View())

    assert _admit_status() is QueueMutationStatus.REROUTE_NORMAL_SEND, (
        "the prompt queue is permanently dead after the first navigation "
        "away -- the coordinator's shutdown latch was never re-opened"
    )


# -- dispose keeps today's behaviour exactly -------------------------------


@pytest.mark.asyncio
async def test_dispose_shuts_the_controller_down_and_closes_the_gateway():
    closed: list[bool] = []

    class _Gateway(_StalledGateway):
        async def aclose(self) -> None:
            closed.append(True)

    gateway = _Gateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    runtime.attach_view(_View())

    await asyncio.wait_for(runtime.dispose(), timeout=2)

    assert controller._shutdown_requested.is_set()
    assert controller._disposed is True
    assert closed == [True], "dispose must close the app-owned gateway"
    assert runtime.view is None
    assert runtime.generation == 1


@pytest.mark.asyncio
async def test_dispose_does_not_let_a_late_ensure_rebuild_the_runtime():
    """A quit-time tick must not resurrect what dispose just tore down.

    `_shutdown_app_owned_lifecycles` runs BEFORE Textual closes screen
    state, so a Console screen and its 0.2s timers can still be live while
    `dispose()` runs -- and ~75 `_ensure_console_chat_*` call sites are
    reachable from them. If `ensure_*` built a fresh object then, quit
    would leave a brand-new controller alive that nothing ever shuts down.
    """
    store = ConsoleChatStore()
    gateway = _StalledGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    runtime.attach_view(_View())

    await asyncio.wait_for(runtime.dispose(), timeout=2)

    # (a) an ALREADY-BUILT slot hands back the torn-down object rather than
    #     a fresh one -- dispose keeps its references precisely for this.
    assert runtime.ensure_chat_controller() is controller, (
        "dispose must not let a late ensure build a SECOND controller"
    )
    assert runtime.ensure_chat_store() is store
    assert runtime.ensure_provider_gateway() is gateway
    # ...and what it hands back is genuinely torn down, so it refuses work.
    assert controller._shutdown_requested.is_set()


@pytest.mark.asyncio
async def test_dispose_does_not_let_a_late_ensure_build_an_unbuilt_slot():
    """The `_disposed` latch itself, on the case references cannot cover.

    A slot that was never built before quit has no reference to hand back,
    so only the latch stops `ensure_*` constructing a brand-new store (and,
    with it, a fresh `ChatPersistenceService`) while the app is exiting.
    """
    runtime = ConsoleRuntime(app=None)
    runtime.attach_view(_View())
    assert runtime.chat_store is None, "nothing built yet -- that is the point"

    await asyncio.wait_for(runtime.dispose(), timeout=2)

    assert runtime.ensure_chat_store() is None, (
        "a late tick built a fresh store DURING QUIT"
    )
    assert runtime.ensure_provider_gateway() is None
    assert runtime.ensure_chat_controller() is None
    assert runtime.chat_store is None


@pytest.mark.asyncio
async def test_leaving_console_does_not_close_the_provider_gateway():
    """The gateway is app-owned now; a surviving turn still needs it."""
    closed: list[bool] = []

    class _Gateway(_StalledGateway):
        async def aclose(self) -> None:
            closed.append(True)

    gateway = _Gateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    runtime.attach_view(_View())

    await asyncio.wait_for(runtime.leave_console(), timeout=2)

    assert closed == []
    assert runtime.provider_gateway is gateway


# -- the wake gate is NOT relaxed here -------------------------------------


@pytest.mark.asyncio
async def test_the_visit_cancellation_event_is_set_between_visits():
    """The per-visit Event's lifecycle: set on leave, fresh on the next attach.

    **Renamed, deliberately, by the wake-fires-headless slice** -- the
    assertions below are unchanged, but the old name
    (`test_the_wake_gate_still_refuses_between_visits`) now describes a
    property that is no longer true. `ConsoleFleetWakeCoordinator._attempt`
    reads `_disposed`, not this Event, so a wake DOES fire between visits;
    the tests that own that are
    `Tests/Chat/test_console_headless_wake_invariants.py::
    test_a_visit_that_merely_ended_does_not_refuse_the_wake` and its
    disposed-direction sibling.

    What this Event still is, and what is pinned here: the signal that
    denies every parked approval/confirm round armed during the visit that
    is ending (each captured it at arm time), replaced by a fresh, unset
    one when the next view attaches so a returning Console works.
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=_StalledGateway()
    )
    runtime = _runtime_with(controller, _View())
    assert not controller._shutdown_requested.is_set()

    await asyncio.wait_for(runtime.leave_console(), timeout=2)
    assert controller._shutdown_requested.is_set(), (
        "leaving Console must leave this visit's Event set -- it is what "
        "keeps every round armed during the visit denied"
    )

    runtime.attach_view(_View())
    assert not controller._shutdown_requested.is_set()


@pytest.mark.asyncio
async def test_leave_console_is_idempotent_and_cheap_with_no_work():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=_StalledGateway()
    )
    runtime = _runtime_with(controller, _View())

    started = time.monotonic()
    assert await runtime.leave_console() is True
    # A second leave finds no view attached and does nothing.
    assert await runtime.leave_console(object()) is False
    assert time.monotonic() - started < 2.0
