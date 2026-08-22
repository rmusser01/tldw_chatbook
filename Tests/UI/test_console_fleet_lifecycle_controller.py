"""No-mount contracts for the Console fleet lifecycle controller."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleFleetCompletionTarget,
    ConsoleRunMarker,
)
from tldw_chatbook.UI.Console_Modules.fleet import ConsoleFleetLifecycleController
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel


_CALLBACK_NAMES = (
    "pending_handoffs_accessor",
    "ensure_chat_store",
    "ensure_chat_controller",
    "activate_workspace_for_session",
    "switch_chat_session",
    "schedule_native_console_sync",
    "ensure_agent_bridge",
    "wire_wake_coordinator",
    "seed_wake_from_marks",
    "retry_wake_soon",
    "wake_has_pending",
    "wake_delivering_conversation_id",
    "displayed_composer_draft_accessor",
    "screen_displayed_accessor",
    "screen_mounted_accessor",
    "active_session_id_accessor",
    "chat_sessions_accessor",
    "defer_on_message_pump",
    "start_transcript_sync_timer",
    "transcript_sync_timer_active",
    "sync_native_console_ui",
    "create_interval",
    "record_timer_created",
    "record_timer_stopped",
    "chat_controller_available",
    "fleet_has_unsettled_children",
    "run_marker_for_session",
    "fleet_teardown_split",
    "leave_runtime",
    "stage_teardown_notices",
    "fleet_unseen_revision_accessor",
    "read_fleet_unseen_ids",
    "clear_fleet_unseen",
)


class _Edges:
    """Callable-backed controller fixture whose targets remain replaceable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._targets: dict[str, Callable[..., Any]] = {
            name: lambda *args, **kwargs: None for name in _CALLBACK_NAMES
        }
        callbacks = {name: self._callback(name) for name in _CALLBACK_NAMES}
        self.controller = ConsoleFleetLifecycleController(**callbacks)

    def _callback(self, name: str) -> Callable[..., Any]:
        def invoke(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args))
            return self._targets[name](*args, **kwargs)

        return invoke

    def replace(self, name: str, target: Callable[..., Any]) -> None:
        self._targets[name] = target

    @property
    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]


class _PendingHandoffs:
    def __init__(self, calls: list[tuple[str, tuple[Any, ...]]], claim: Any) -> None:
        self._calls = calls
        self._claim = claim

    def claim(self, channel: HandoffChannel) -> Any:
        self._calls.append(("claim", (channel,)))
        return self._claim

    def acknowledge(self, claim: Any) -> None:
        self._calls.append(("acknowledge", (claim,)))

    def release(self, claim: Any) -> None:
        self._calls.append(("release", (claim,)))


class _Timer:
    def __init__(self, calls: list[tuple[str, tuple[Any, ...]]]) -> None:
        self._calls = calls

    def stop(self) -> None:
        self._calls.append(("timer.stop", ()))


def _session(
    session_id: str,
    *,
    conversation_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=session_id,
        persisted_conversation_id=conversation_id,
    )


def _completion_edges(
    *,
    sessions: tuple[SimpleNamespace, ...],
    active_session_id: str | None,
    target: ConsoleFleetCompletionTarget,
) -> tuple[_Edges, Any]:
    edges = _Edges()
    claim = SimpleNamespace(value=target, revision=17)
    handoffs = _PendingHandoffs(edges.calls, claim)
    store = SimpleNamespace(
        active_session_id=active_session_id,
        sessions=lambda: sessions,
    )
    edges.replace("pending_handoffs_accessor", lambda: handoffs)
    edges.replace("ensure_chat_store", lambda: store)
    return edges, claim


def test_completion_claim_prefers_exact_session_and_acknowledges_once() -> None:
    sessions = (
        _session("conversation-first", conversation_id="conversation-a"),
        _session("exact-session", conversation_id="conversation-b"),
        _session("conversation-last", conversation_id="conversation-a"),
    )
    edges, claim = _completion_edges(
        sessions=sessions,
        active_session_id="other-session",
        target=ConsoleFleetCompletionTarget(
            conversation_id="conversation-a",
            session_id="exact-session",
        ),
    )

    result = edges.controller.consume_pending_console_fleet_completion()

    assert result is True
    assert edges.calls == [
        ("pending_handoffs_accessor", ()),
        ("claim", (HandoffChannel.CONSOLE_FLEET_COMPLETION,)),
        ("ensure_chat_store", ()),
        ("ensure_chat_controller", ()),
        ("activate_workspace_for_session", ("exact-session",)),
        ("switch_chat_session", ("exact-session",)),
        ("schedule_native_console_sync", ()),
        ("acknowledge", (claim,)),
    ]


def test_completion_claim_uses_last_conversation_match() -> None:
    sessions = (
        _session("first", conversation_id="conversation-a"),
        _session("unrelated", conversation_id="conversation-b"),
        _session("last", conversation_id="conversation-a"),
    )
    edges, claim = _completion_edges(
        sessions=sessions,
        active_session_id="unrelated",
        target=ConsoleFleetCompletionTarget(conversation_id="conversation-a"),
    )

    result = edges.controller.consume_pending_console_fleet_completion()

    assert result is True
    assert edges.calls[-5:] == [
        ("ensure_chat_controller", ()),
        ("activate_workspace_for_session", ("last",)),
        ("switch_chat_session", ("last",)),
        ("schedule_native_console_sync", ()),
        ("acknowledge", (claim,)),
    ]


def test_already_active_completion_returns_true_without_side_effects() -> None:
    active = _session("active", conversation_id="conversation-a")
    edges, claim = _completion_edges(
        sessions=(active,),
        active_session_id=active.id,
        target=ConsoleFleetCompletionTarget(
            conversation_id="conversation-a",
            session_id=active.id,
        ),
    )

    result = edges.controller.consume_pending_console_fleet_completion()

    assert result is True
    assert edges.calls == [
        ("pending_handoffs_accessor", ()),
        ("claim", (HandoffChannel.CONSOLE_FLEET_COMPLETION,)),
        ("ensure_chat_store", ()),
        ("acknowledge", (claim,)),
    ]


def test_completion_exception_releases_claim_without_acknowledging() -> None:
    edges, claim = _completion_edges(
        sessions=(),
        active_session_id=None,
        target=ConsoleFleetCompletionTarget(conversation_id="conversation-a"),
    )

    def raise_store_error() -> None:
        raise RuntimeError("store unavailable")

    edges.replace("ensure_chat_store", raise_store_error)

    result = edges.controller.consume_pending_console_fleet_completion()

    assert result is False
    assert edges.calls == [
        ("pending_handoffs_accessor", ()),
        ("claim", (HandoffChannel.CONSOLE_FLEET_COMPLETION,)),
        ("ensure_chat_store", ()),
        ("release", (claim,)),
    ]


def test_completion_controller_failure_releases_before_workspace_activation() -> None:
    target = _session("target", conversation_id="conversation-a")
    edges, claim = _completion_edges(
        sessions=(target,),
        active_session_id="other-session",
        target=ConsoleFleetCompletionTarget(
            conversation_id="conversation-a",
            session_id=target.id,
        ),
    )

    def raise_controller_error() -> None:
        raise RuntimeError("controller unavailable")

    edges.replace("ensure_chat_controller", raise_controller_error)

    result = edges.controller.consume_pending_console_fleet_completion()

    assert result is False
    assert edges.calls == [
        ("pending_handoffs_accessor", ()),
        ("claim", (HandoffChannel.CONSOLE_FLEET_COMPLETION,)),
        ("ensure_chat_store", ()),
        ("ensure_chat_controller", ()),
        ("release", (claim,)),
    ]


@pytest.mark.parametrize(
    "sessions",
    [
        (),
        (_session("unrelated", conversation_id="conversation-b"),),
    ],
    ids=["empty", "unrelated"],
)
def test_completion_claim_acknowledges_missing_session_without_side_effects(
    sessions: tuple[SimpleNamespace, ...],
) -> None:
    edges, claim = _completion_edges(
        sessions=sessions,
        active_session_id="unrelated",
        target=ConsoleFleetCompletionTarget(conversation_id="conversation-a"),
    )

    result = edges.controller.consume_pending_console_fleet_completion()

    assert result is False
    assert edges.calls == [
        ("pending_handoffs_accessor", ()),
        ("claim", (HandoffChannel.CONSOLE_FLEET_COMPLETION,)),
        ("ensure_chat_store", ()),
        ("acknowledge", (claim,)),
    ]


def test_mount_claim_reads_uncached_marks_before_wiring_and_retry() -> None:
    edges = _Edges()
    edges.replace("read_fleet_unseen_ids", lambda: frozenset({"conversation-a"}))
    edges.replace("ensure_agent_bridge", object)
    edges.replace("wire_wake_coordinator", lambda: True)
    edges.replace("seed_wake_from_marks", lambda: True)

    edges.controller._claim_console_fleet_wake_marks()

    assert edges.call_names == [
        "read_fleet_unseen_ids",
        "ensure_agent_bridge",
        "wire_wake_coordinator",
        "seed_wake_from_marks",
        "retry_wake_soon",
    ]
    assert "fleet_unseen_revision_accessor" not in edges.call_names


def test_user_priority_propagates_a_selected_composer_draft_error() -> None:
    edges = _Edges()

    def raise_draft_error() -> None:
        raise RuntimeError("selected composer disappeared")

    edges.replace("displayed_composer_draft_accessor", raise_draft_error)

    with pytest.raises(RuntimeError, match="selected composer disappeared"):
        edges.controller._console_wake_user_priority("session-a")


def test_delivery_start_distinguishes_unmounted_from_hidden_mounted() -> None:
    edges = _Edges()
    edges.replace("screen_mounted_accessor", lambda: False)
    edges.controller._on_console_wake_delivery_started("session-a")
    unmounted_calls = edges.call_names

    edges.calls.clear()
    edges.replace("screen_mounted_accessor", lambda: True)
    edges.replace("screen_displayed_accessor", lambda: False)
    edges.replace(
        "defer_on_message_pump",
        lambda callback: callback(),
    )
    edges.controller._on_console_wake_delivery_started("session-a")
    hidden_mounted_calls = edges.call_names

    assert unmounted_calls == ["screen_mounted_accessor"]
    assert hidden_mounted_calls == [
        "screen_mounted_accessor",
        "defer_on_message_pump",
        "start_transcript_sync_timer",
    ]


def test_missing_chat_controller_returns_none_markers() -> None:
    edges = _Edges()
    edges.replace("chat_controller_available", lambda: False)

    markers = edges.controller.prepare_session_run_markers(
        (_session("session-a"),),
        "session-a",
    )

    assert markers is None


def test_missing_chat_controller_still_clears_a_displayed_unseen_mark() -> None:
    edges = _Edges()
    active = _session("active", conversation_id="conversation-active")
    revision = 4
    durable_ids = frozenset({"conversation-active"})
    edges.replace("chat_controller_available", lambda: False)
    edges.replace("fleet_unseen_revision_accessor", lambda: revision)
    edges.replace("read_fleet_unseen_ids", lambda: durable_ids)
    edges.replace("wake_has_pending", lambda conversation_id: False)
    edges.replace("screen_displayed_accessor", lambda: True)

    def clear_unseen(conversation_id: str) -> bool:
        nonlocal revision, durable_ids
        revision = 5
        durable_ids = frozenset()
        return True

    edges.replace("clear_fleet_unseen", clear_unseen)

    markers = edges.controller.prepare_session_run_markers((active,), active.id)

    assert markers is None
    assert edges.calls == [
        ("chat_controller_available", ()),
        ("fleet_unseen_revision_accessor", ()),
        ("read_fleet_unseen_ids", ()),
        ("wake_has_pending", ("conversation-active",)),
        ("screen_displayed_accessor", ()),
        ("clear_fleet_unseen", ("conversation-active",)),
        ("fleet_unseen_revision_accessor", ()),
        ("read_fleet_unseen_ids", ()),
    ]
    assert "run_marker_for_session" not in edges.call_names


def test_pending_wake_defers_view_clear_and_live_marker_outranks_unseen() -> None:
    edges = _Edges()
    active = _session("active", conversation_id="conversation-active")
    live = _session("live", conversation_id="conversation-live")
    edges.replace("chat_controller_available", lambda: True)
    edges.replace("fleet_unseen_revision_accessor", lambda: 4)
    edges.replace(
        "read_fleet_unseen_ids",
        lambda: frozenset({"conversation-active", "conversation-live"}),
    )
    edges.replace(
        "wake_has_pending",
        lambda conversation_id: conversation_id == "conversation-active",
    )
    edges.replace("screen_displayed_accessor", lambda: True)
    edges.replace(
        "run_marker_for_session",
        lambda session_id: (
            ConsoleRunMarker.RUNNING if session_id == "live" else ConsoleRunMarker.NONE
        ),
    )

    markers = edges.controller.prepare_session_run_markers(
        (active, live),
        active.id,
    )

    assert markers == {
        "active": ConsoleRunMarker.SUBAGENT_UNSEEN,
        "live": ConsoleRunMarker.RUNNING,
    }
    assert ("wake_has_pending", ("conversation-active",)) in edges.calls
    assert "clear_fleet_unseen" not in edges.call_names


def test_fleet_unseen_cache_reuses_revision_and_refreshes_after_change() -> None:
    edges = _Edges()
    revision = 4
    durable_ids = frozenset({"conversation-a"})
    edges.replace("fleet_unseen_revision_accessor", lambda: revision)
    edges.replace("read_fleet_unseen_ids", lambda: durable_ids)

    first = edges.controller._console_fleet_unseen_ids()
    same_revision = edges.controller._console_fleet_unseen_ids()
    revision = 5
    durable_ids = frozenset({"conversation-b"})
    changed_revision = edges.controller._console_fleet_unseen_ids()

    assert edges.call_names == [
        "fleet_unseen_revision_accessor",
        "read_fleet_unseen_ids",
        "fleet_unseen_revision_accessor",
        "fleet_unseen_revision_accessor",
        "read_fleet_unseen_ids",
    ]
    assert (first, same_revision, changed_revision) == (
        frozenset({"conversation-a"}),
        frozenset({"conversation-a"}),
        frozenset({"conversation-b"}),
    )


@pytest.mark.asyncio
async def test_teardown_stages_counts_only_after_a_truthy_leave() -> None:
    false_edges = _Edges()
    false_edges.replace("fleet_teardown_split", lambda: (2, 3))

    async def leave_false() -> bool:
        return False

    false_edges.replace("leave_runtime", leave_false)
    await false_edges.controller._record_console_fleet_teardown()

    true_edges = _Edges()
    true_edges.replace("fleet_teardown_split", lambda: (2, 3))

    async def leave_true() -> bool:
        return True

    true_edges.replace("leave_runtime", leave_true)
    await true_edges.controller._record_console_fleet_teardown()

    assert false_edges.call_names == ["fleet_teardown_split", "leave_runtime"]
    assert true_edges.call_names == [
        "fleet_teardown_split",
        "leave_runtime",
        "stage_teardown_notices",
    ]
    assert true_edges.calls[-1] == ("stage_teardown_notices", (2, 3))


@pytest.mark.asyncio
async def test_survivor_tick_is_idempotent_and_final_paints_after_stop() -> None:
    edges = _Edges()
    timer = _Timer(edges.calls)
    survivors_live = True
    edges.replace("chat_controller_available", lambda: True)
    edges.replace("fleet_has_unsettled_children", lambda: survivors_live)
    edges.replace("create_interval", lambda seconds, callback: timer)
    edges.replace("transcript_sync_timer_active", lambda: False)

    async def sync_ui() -> None:
        return None

    edges.replace("sync_native_console_ui", sync_ui)

    edges.controller._maybe_start_console_fleet_survivor_tick()
    edges.controller._maybe_start_console_fleet_survivor_tick()
    survivors_live = False
    await edges.controller._console_fleet_survivor_tick()

    assert [call for call in edges.calls if call[0] == "create_interval"] == [
        (
            "create_interval",
            (1.0, edges.controller._console_fleet_survivor_tick),
        )
    ]
    assert [call for call in edges.calls if call[0] == "record_timer_created"] == [
        ("record_timer_created", ("console-fleet-survivor-tick",))
    ]
    assert edges.call_names[-3:] == [
        "timer.stop",
        "record_timer_stopped",
        "sync_native_console_ui",
    ]
    assert edges.calls[-2] == (
        "record_timer_stopped",
        ("console-fleet-survivor-tick",),
    )
    assert edges.controller._console_fleet_survivor_timer is None
