"""Focused contracts for Roleplay's ID-only Console resume navigation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Constants import (
    CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def test_resume_navigation_context_captures_only_normalized_local_id() -> None:
    screen = ChatScreen.__new__(ChatScreen)

    screen.apply_navigation_context(
        {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "  abc  "}
    )

    assert screen._pending_resume_local_conversation_id == "abc"


@pytest.mark.parametrize("value", [None, 7, "", "   ", "x" * 257])
def test_resume_navigation_context_ignores_invalid_ids(value: object) -> None:
    screen = ChatScreen.__new__(ChatScreen)

    screen.apply_navigation_context(
        {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: value}
    )

    assert screen._pending_resume_local_conversation_id is None


def test_resume_navigation_context_is_capture_only_before_mount() -> None:
    screen = ChatScreen.__new__(ChatScreen)

    def unexpected_side_effect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("navigation capture touched Console runtime state")

    screen.query_one = unexpected_side_effect
    screen._ensure_console_chat_store = unexpected_side_effect
    screen._sync_native_console_chat_ui = unexpected_side_effect
    screen._restore_console_workbench_focus = unexpected_side_effect
    screen.run_worker = unexpected_side_effect
    screen.set_timer = unexpected_side_effect

    screen.apply_navigation_context(
        {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "conversation-7"}
    )

    assert screen._pending_resume_local_conversation_id == "conversation-7"


def test_resume_navigation_context_has_class_level_unset_default() -> None:
    screen = ChatScreen.__new__(ChatScreen)

    assert screen._pending_resume_local_conversation_id is None
    assert screen._resume_navigation_startup_in_progress is False


def _async_spy(events: list[str], label: str):
    async def callback() -> None:
        events.append(label)

    callback.__name__ = label
    return callback


def _sync_spy(events: list[str], label: str, result=None):
    def callback():
        events.append(label)
        return result

    callback.__name__ = label
    return callback


def _resume_mount_screen() -> tuple[
    ChatScreen,
    list[str],
    list[tuple[float, object]],
    list[object],
    list[tuple[object, dict[str, object]]],
    list[str],
    list[str],
]:
    screen = ChatScreen.__new__(ChatScreen)
    screen.apply_navigation_context(
        {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "  resume-target  "}
    )
    events: list[str] = []
    timers: list[tuple[float, object]] = []
    after_refresh: list[object] = []
    workers: list[tuple[object, dict[str, object]]] = []
    opener_ids: list[str] = []
    ordinary_events: list[str] = []
    first_chat_calls = 0

    def first_chat() -> None:
        nonlocal first_chat_calls
        first_chat_calls += 1
        if first_chat_calls == 1:
            events.append("first-chat")

    async def fleet_completion() -> None:
        events.append("fleet-completion")

    async def opener(conversation_id: str) -> bool:
        opener_ids.append(conversation_id)
        events.extend(("resume-selected-conversation", "final-presentation-focus"))
        return True

    async def refresh_skills() -> None:
        ordinary_events.append("skills")

    screen.app_instance = SimpleNamespace()
    screen._apply_focus_chrome = lambda: None
    screen._session = SimpleNamespace(
        consume_pending_console_first_chat_intent=first_chat,
    )
    screen._notify_console_fleet_teardown_if_any = lambda: None
    screen._fleet = SimpleNamespace(
        _claim_console_fleet_wake_marks=lambda: None,
        consume_pending_console_fleet_completion=fleet_completion,
        _maybe_start_console_fleet_survivor_tick=_sync_spy(
            ordinary_events, "survivor"
        ),
    )
    screen._console_auto_speak = SimpleNamespace(mount=lambda: None)
    screen._restore_collapsible_states = _sync_spy(
        ordinary_events, "collapsibles"
    )
    screen.sync_task_resume_state = _sync_spy(ordinary_events, "task-resume")
    screen._consume_pending_chat_handoff = _async_spy(events, "chat-handoff")
    screen._consume_pending_console_roleplay_repair = _sync_spy(
        events, "roleplay-repair", True
    )
    screen._consume_pending_console_prompt_insert = _async_spy(
        events, "prompt-insert"
    )
    screen.consume_pending_console_provider_intent = _sync_spy(
        events, "provider-intent", True
    )
    screen._sync_console_dictation_availability = _sync_spy(
        ordinary_events, "dictation"
    )
    screen._sync_native_console_chat_ui = lambda: None
    screen._image = SimpleNamespace(
        _reconcile_h3_image_edit_completions=_sync_spy(
            ordinary_events, "image"
        ),
    )
    screen._restore_console_workbench_focus = lambda: None
    screen._skill = SimpleNamespace(
        _refresh_console_skill_candidates=refresh_skills,
    )
    screen._workspace = SimpleNamespace(
        open_console_workspace_conversation=opener,
        _reconcile_console_session_with_registry=_sync_spy(
            events, "registry-reconcile"
        ),
    )
    screen._sync_console_transcript_guidance = lambda: None
    screen._register_console_footer_shortcuts = lambda: None
    screen._consume_pending_console_identity_refresh = lambda: True
    screen._dispatch_active_console_roleplay_refresh = lambda: None
    screen.set_timer = lambda delay, callback: timers.append((delay, callback))
    screen.call_after_refresh = after_refresh.append
    screen.run_worker = lambda coroutine, **kwargs: workers.append(
        (coroutine, kwargs)
    )
    return (
        screen,
        events,
        timers,
        after_refresh,
        workers,
        opener_ids,
        ordinary_events,
    )


@pytest.mark.asyncio
async def test_resume_navigation_mount_orders_pending_consumers_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen, events, timers, after_refresh, workers, opener_ids, ordinary_events = (
        _resume_mount_screen()
    )
    monkeypatch.setattr(chat_screen_module, "apply_status_chips_position", lambda _: None)

    ChatScreen.on_mount(screen)

    assert screen._resume_navigation_startup_in_progress is True
    competing = {
        "chat-handoff",
        "roleplay-repair",
        "prompt-insert",
        "provider-intent",
        "fleet-completion",
    }
    assert not competing.intersection(
        getattr(callback, "__name__", "")
        for delay, callback in timers
        if delay == 0.15
    )
    assert {(delay, getattr(callback, "__name__", "")) for delay, callback in timers} >= {
        (0.1, "collapsibles"),
        (0.05, "task-resume"),
        (0.15, "dictation"),
        (0.3, "survivor"),
    }
    assert screen._sync_console_dictation_availability in after_refresh
    assert screen._image._reconcile_h3_image_edit_completions in after_refresh
    assert screen._sync_native_console_chat_ui not in after_refresh
    assert screen._restore_console_workbench_focus not in after_refresh
    assert not any(
        delay == 0.2 and callback is screen._restore_console_workbench_focus
        for delay, callback in timers
    )

    timer_count = len(timers)
    after_refresh_count = len(after_refresh)
    ChatScreen.on_screen_resume(screen)
    assert len(timers) == timer_count
    assert len(after_refresh) == after_refresh_count

    for _delay, callback in timers:
        if getattr(callback, "__name__", "") != "_start_resume_navigation_startup":
            callback()
    for callback in after_refresh:
        callback()
    await workers[0][0]
    assert ordinary_events == [
        "collapsibles",
        "task-resume",
        "survivor",
        "dictation",
        "dictation",
        "image",
        "skills",
    ]

    ordered_timer = next(
        callback
        for delay, callback in timers
        if delay == 0.15
        and getattr(callback, "__name__", "")
        == "_start_resume_navigation_startup"
    )
    ordered_timer()
    ordered_work, ordered_options = workers[-1]
    assert ordered_options == {
        "exclusive": True,
        "group": "console-resume-navigation-startup",
    }
    await ordered_work

    assert events == [
        "first-chat",
        "chat-handoff",
        "roleplay-repair",
        "prompt-insert",
        "provider-intent",
        "fleet-completion",
        "resume-selected-conversation",
        "final-presentation-focus",
    ]
    assert opener_ids == ["resume-target"]
    assert screen._pending_resume_local_conversation_id is None
    assert screen._resume_navigation_startup_in_progress is False


@pytest.mark.asyncio
async def test_resume_navigation_continues_after_chat_handoff_release() -> None:
    screen = ChatScreen.__new__(ChatScreen)
    handoffs = PendingHandoffStore()
    handoffs.stage(
        HandoffChannel.CHAT,
        ChatHandoffPayload(
            source="Library",
            item_type="note",
            title="Pending note",
            body="body",
        ),
    )
    events: list[str] = []

    async def release_handoff(_payload: ChatHandoffPayload) -> bool:
        events.append("chat-handoff-released")
        raise RuntimeError("not ready")

    async def opener(conversation_id: str) -> bool:
        events.append(f"resume:{conversation_id}")
        return True

    screen.app_instance = SimpleNamespace(pending_handoffs=handoffs)
    screen._handoff_consumption_in_progress = False
    screen._session = SimpleNamespace(
        _start_character_console_session=release_handoff,
    )
    screen._stage_handoff_as_console_live_work = lambda _payload: None
    screen._consume_pending_console_roleplay_repair = lambda: False
    screen._consume_pending_console_prompt_insert = _async_spy(events, "prompt")
    screen.consume_pending_console_provider_intent = lambda: False
    screen._fleet = SimpleNamespace(
        consume_pending_console_fleet_completion=lambda: False,
    )
    screen._workspace = SimpleNamespace(
        open_console_workspace_conversation=opener,
    )
    screen._pending_resume_local_conversation_id = "resume-target"
    screen._resume_navigation_startup_in_progress = True

    await screen._consume_resume_navigation_startup()

    assert handoffs.has_pending(HandoffChannel.CHAT)
    assert events == ["chat-handoff-released", "prompt", "resume:resume-target"]


@pytest.mark.asyncio
async def test_resume_navigation_worker_cancellation_stops_and_propagates() -> None:
    screen = ChatScreen.__new__(ChatScreen)
    started = asyncio.Event()
    blocker = asyncio.Event()
    events: list[str] = []
    tasks: list[asyncio.Task[None]] = []

    async def pending_handoff() -> None:
        events.append("chat-handoff")
        started.set()
        await blocker.wait()

    async def opener(_conversation_id: str) -> bool:
        events.append("resume")
        return True

    screen._consume_pending_chat_handoff = pending_handoff
    screen._consume_pending_console_roleplay_repair = lambda: False
    screen._consume_pending_console_prompt_insert = _async_spy(events, "prompt")
    screen.consume_pending_console_provider_intent = lambda: False
    screen._fleet = SimpleNamespace(
        consume_pending_console_fleet_completion=lambda: False,
    )
    screen._workspace = SimpleNamespace(
        open_console_workspace_conversation=opener,
    )
    screen._pending_resume_local_conversation_id = "resume-target"
    screen._resume_navigation_startup_in_progress = True

    def run_worker(coroutine, **_kwargs) -> None:
        tasks.append(asyncio.create_task(coroutine))

    screen.run_worker = run_worker
    screen._start_resume_navigation_startup()
    await started.wait()
    tasks[0].cancel()

    with pytest.raises(asyncio.CancelledError):
        await tasks[0]

    assert events == ["chat-handoff"]
    assert screen._pending_resume_local_conversation_id is None
    assert screen._resume_navigation_startup_in_progress is False
