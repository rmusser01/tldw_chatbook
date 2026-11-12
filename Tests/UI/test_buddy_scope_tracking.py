"""Buddy scope tracking stops with its presentation and no-ops when unchanged."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Navigation.buddy_management import BuddyManagementCoordinator
from tldw_chatbook.UI.Navigation.persona_buddy_overlay import PersonaBuddyOverlay

pytestmark = pytest.mark.bootstrap_profile


class FakeTimer:
    def __init__(self, interval: float, callback) -> None:
        self.interval = interval
        self.callback = callback
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakeApp:
    def __init__(self) -> None:
        self.app_config = {"buddy_interaction": {}}
        self.console_runtime = None
        self.timers: list[FakeTimer] = []

    def set_interval(self, interval: float, callback) -> FakeTimer:
        timer = FakeTimer(interval, callback)
        self.timers.append(timer)
        return timer


class FakeSink:
    def __init__(self) -> None:
        self.scope_calls: list[tuple[frozenset, frozenset]] = []

    def set_scope(self, *, session_ids, conversation_ids) -> bool:
        self.scope_calls.append((session_ids, conversation_ids))
        return False


def _runtime(store: ConsoleChatStore, sink: FakeSink) -> SimpleNamespace:
    return SimpleNamespace(
        chat_store=store,
        chat_controller=object(),
        persona_buddy_sink=sink,
    )


def _coordinator(app: FakeApp, *, binding=None, speak_responses=False):
    coordinator = BuddyManagementCoordinator(app)
    coordinator.preferences = SimpleNamespace(
        binding=binding, speak_responses=speak_responses
    )
    return coordinator


def test_stop_scope_tracking_stops_the_app_timer_and_is_idempotent():
    app = FakeApp()
    coordinator = _coordinator(app)

    coordinator.start_scope_tracking()
    assert len(app.timers) == 1 and not app.timers[0].stopped

    coordinator.stop_scope_tracking()
    assert app.timers[0].stopped
    assert coordinator._scope_timer is None

    coordinator.stop_scope_tracking()  # no timer armed: stays a no-op
    assert coordinator._scope_timer is None


def test_reconcile_scope_no_ops_while_state_is_unchanged():
    app = FakeApp()
    store = ConsoleChatStore()
    sink = FakeSink()
    app.console_runtime = _runtime(store, sink)
    coordinator = _coordinator(app)

    coordinator.reconcile_scope()
    assert len(sink.scope_calls) == 1

    for _ in range(5):
        coordinator.reconcile_scope()
    assert len(sink.scope_calls) == 1, "unchanged ticks must not re-drive the sink"


def test_reconcile_scope_replays_scope_after_a_runtime_rebind():
    app = FakeApp()
    store = ConsoleChatStore()
    first_sink = FakeSink()
    app.console_runtime = _runtime(store, first_sink)
    coordinator = _coordinator(app)
    coordinator.reconcile_scope()

    rebound_sink = FakeSink()
    app.console_runtime = _runtime(store, rebound_sink)
    coordinator.reconcile_scope()
    assert len(rebound_sink.scope_calls) == 1, "a rebind must replay sink scope"


def test_reconcile_scope_follows_session_changes():
    app = FakeApp()
    store = ConsoleChatStore()
    store.create_session(title="One", ephemeral=True)
    sink = FakeSink()
    app.console_runtime = _runtime(store, sink)
    binding = SimpleNamespace(kind="workspace", includes=lambda session: True)
    coordinator = _coordinator(app, binding=binding)
    coordinator.reconcile_scope()
    assert len(sink.scope_calls[-1][0]) == 1

    store.create_session(title="Two", ephemeral=True)
    coordinator.reconcile_scope()
    assert len(sink.scope_calls) == 2
    assert len(sink.scope_calls[-1][0]) == 2


def test_reconcile_scope_promotion_retry_bypasses_the_unchanged_gate():
    app = FakeApp()
    store = ConsoleChatStore()
    session = store.create_session(title="Unpromoted", ephemeral=True)
    sink = FakeSink()
    app.console_runtime = _runtime(store, sink)
    binding = SimpleNamespace(
        kind="conversation",
        conversation_id=None,
        ephemeral=False,
        resolve_session=lambda sessions: session,
    )
    coordinator = _coordinator(app, binding=binding)

    coordinator.reconcile_scope()
    coordinator.reconcile_scope()
    assert len(sink.scope_calls) == 2, "deadline-driven retries must bypass the gate"


def test_overlay_retire_stops_scope_tracking():
    app = FakeApp()
    app.screen = SimpleNamespace(is_attached=False)
    app.persona_buddy_controller = None
    coordinator = _coordinator(app)
    app._buddy_management = coordinator
    coordinator.start_scope_tracking()

    overlay = PersonaBuddyOverlay(app)
    assert asyncio.run(overlay.reconcile()) is True
    assert coordinator._scope_timer is None
    assert app.timers[0].stopped


def test_overlay_shutdown_stops_scope_tracking():
    app = FakeApp()
    coordinator = _coordinator(app)
    app._buddy_management = coordinator
    coordinator.start_scope_tracking()

    overlay = PersonaBuddyOverlay(app)
    asyncio.run(overlay.shutdown())
    assert overlay.closed
    assert coordinator._scope_timer is None
    assert app.timers[0].stopped
