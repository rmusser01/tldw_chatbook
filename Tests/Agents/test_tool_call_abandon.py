"""task-31386: abandon one in-flight tool call and let the turn continue."""

from __future__ import annotations

import threading

from tldw_chatbook.Agents.agent_models import TOOL_OUTCOME_CANCELLED
from tldw_chatbook.Agents.agent_service import (
    _call_with_timeout,
    _clear_tool_call_inflight,
    _mark_tool_call_inflight,
    inflight_tool_call,
    request_tool_call_abandon,
    tool_call_abandon_requested,
)


def test_a_request_with_no_call_in_flight_is_refused_and_never_queued():
    _clear_tool_call_inflight("run-x")
    assert request_tool_call_abandon("run-x") is None
    assert tool_call_abandon_requested("run-x") is False
    _mark_tool_call_inflight("run-x", "c1", "grep_files")
    assert inflight_tool_call("run-x") == ("c1", "grep_files")
    assert tool_call_abandon_requested("run-x") is False  # the refusal left nothing behind
    _clear_tool_call_inflight("run-x")


def test_an_abandon_request_cancels_only_the_wrapped_call_and_is_cleared_after():
    run_id = "run-abandon"
    release = threading.Event()

    def _hung():
        release.wait(10)
        return None

    _mark_tool_call_inflight(run_id, "c1", "slow_tool")
    try:
        assert request_tool_call_abandon(run_id) == "slow_tool"
        run_wide_stop = False
        result = _call_with_timeout(
            _hung,
            seconds=30.0,
            tool_name="slow_tool",
            should_cancel=lambda: run_wide_stop or tool_call_abandon_requested(run_id),
        )
    finally:
        _clear_tool_call_inflight(run_id)
        release.set()
    assert result.ok is False and result.outcome == TOOL_OUTCOME_CANCELLED
    assert "cancelled" in (result.error or "")
    # The run itself was never asked to stop, and the flag did not outlive the call.
    assert run_wide_stop is False
    assert tool_call_abandon_requested(run_id) is False
    assert inflight_tool_call(run_id) is None


def test_the_screen_action_reaches_the_service_and_cancels_only_the_call():
    """The action -> controller -> service chain, with a hung call in flight."""
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    store = ConsoleChatStore()
    session = store.ensure_session()
    session.persisted_conversation_id = "conv-1"
    controller = ConsoleChatController(store=store, provider_gateway=None)
    controller._agent_bridge = SimpleNamespace(
        live_primary_run_id=lambda conversation_id: "run-ui" if conversation_id == "conv-1" else None
    )
    notices = []
    screen = SimpleNamespace(
        _ensure_console_chat_controller=lambda: controller,
        app_instance=SimpleNamespace(notify=lambda message, **kw: notices.append(message)),
    )
    # Nothing in flight: the click is refused, not queued.
    ChatScreen.action_abandon_console_tool_call(screen)
    assert notices == ["No tool call is running."]
    assert tool_call_abandon_requested("run-ui") is False

    release = threading.Event()

    def _hung():
        release.wait(10)
        return None

    _mark_tool_call_inflight("run-ui", "c1", "slow_tool")
    try:
        ChatScreen.action_abandon_console_tool_call(screen)
        assert notices[-1] == "Abandoning slow_tool… the turn continues."
        run_wide_stop = False
        result = _call_with_timeout(
            _hung,
            seconds=30.0,
            tool_name="slow_tool",
            should_cancel=lambda: run_wide_stop or tool_call_abandon_requested("run-ui"),
        )
    finally:
        _clear_tool_call_inflight("run-ui")
        release.set()
    assert result.ok is False and result.outcome == TOOL_OUTCOME_CANCELLED
    assert run_wide_stop is False and inflight_tool_call("run-ui") is None
