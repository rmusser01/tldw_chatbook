"""Tests for the live Console MCP batch-approval flow (Phase-5 task-5).

Covers the widget half (``ChatApprovalCard.set_batch``/``ApprovalDecided``)
and the controller half (``ConsoleChatController.request_mcp_approvals``/
``resolve_pending_approval``/context-change denial) of the worker-thread
<-> UI-thread approval round-trip described in
``.superpowers/sdd/task-5-brief.md``. ``Tests/UI/test_chat_approvals_and_
resume.py`` covers the legacy single-approval API and must stay green
unmodified -- this file only exercises the new batch path.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import Mock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Static

from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard


def _text(widget: Static) -> str:
    return str(widget.render())


class _CardHarnessApp(App[None]):
    """Minimal host for `ChatApprovalCard` that records `ApprovalDecided`."""

    def __init__(self) -> None:
        super().__init__()
        self.decided: list[dict[str, str]] = []

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard()

    @on(ChatApprovalCard.ApprovalDecided)
    def _capture_decision(self, event: ChatApprovalCard.ApprovalDecided) -> None:
        self.decided.append(event.decisions)


def _sample_calls() -> list[dict]:
    """Three raw pending-call dicts: two share an llm_name (collapse to one row)."""
    return [
        {
            "llm_name": "mcp__srv_a__search",
            "server_key": "local:srv_a",
            "tool_name": "search",
            "server_label": "Srv A",
            "arguments": {"query": "hello"},
            "reason": "ask",
        },
        {
            "llm_name": "mcp__srv_a__search",
            "server_key": "local:srv_a",
            "tool_name": "search",
            "server_label": "Srv A",
            "arguments": {"query": "hello"},
            "reason": "ask",
        },
        {
            "llm_name": "mcp__srv_b__write",
            "server_key": "local:srv_b",
            "tool_name": "write",
            "server_label": "Srv B",
            "arguments": {"path": "/tmp/x" * 10},
            "reason": "config_changed",
        },
    ]


# ---------------------------------------------------------------------------
# ChatApprovalCard.set_batch / ApprovalDecided
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_batch_renders_one_row_per_unique_name_with_tooltips():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=45.0)
        await pilot.pause()

        assert card.display is True
        rows = list(app.query(".approval-row"))
        assert len(rows) == 2  # collapsed by llm_name (T3 contract)

        headers = [_text(row.query_one(".approval-row-header", Static)) for row in rows]
        assert "Srv A · search ×2" in headers[0]
        assert "Srv B · write" in headers[1]
        assert "(definition changed)" in headers[1]
        assert "(definition changed)" not in headers[0]

        args_summaries = [_text(row.query_one(".approval-row-args", Static)) for row in rows]
        assert all(len(summary) <= 80 for summary in args_summaries)

        for select in app.query(Select):
            assert select.value == "approve_once"

        for button_id in ("#approval-approve-all", "#approval-submit", "#approval-deny-all"):
            button = app.query_one(button_id, Button)
            assert button.tooltip, f"{button_id} must be tooltipped"


@pytest.mark.asyncio
async def test_approve_all_and_deny_all_bulk_set_every_row():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=45.0)
        await pilot.pause()

        app.query_one("#approval-deny-all", Button).press()
        await pilot.pause()
        assert all(select.value == "deny" for select in card._batch_selects)

        app.query_one("#approval-approve-all", Button).press()
        await pilot.pause()
        assert all(select.value == "approve_once" for select in card._batch_selects)


@pytest.mark.asyncio
async def test_submit_posts_approval_decided_with_per_row_decisions():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=45.0)
        await pilot.pause()

        card._batch_selects[1].value = "deny"
        app.query_one("#approval-submit", Button).press()
        await pilot.pause()

        assert app.decided == [
            {"mcp__srv_a__search": "approve_once", "mcp__srv_b__write": "deny"}
        ]


@pytest.mark.asyncio
async def test_set_batch_with_no_calls_hides_the_card():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=45.0)
        await pilot.pause()
        assert card.display is True

        card.set_batch([], timeout_seconds=45.0)
        await pilot.pause()
        assert card.display is False


@pytest.mark.asyncio
async def test_set_batch_remount_does_not_duplicate_rows():
    """Calling set_batch twice in a row must not raise or leave stale rows.

    Exercises the fire-and-forget remove/mount discipline documented on
    `ChatApprovalCard.set_batch` (unique per-generation row ids rather than
    an awaited `remove_children()`).
    """
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=45.0)
        card.set_batch(_sample_calls()[:1], timeout_seconds=45.0)
        await pilot.pause()

        rows = list(app.query(".approval-row"))
        assert len(rows) == 1
        assert card._batch_names == ["mcp__srv_a__search"]


# ---------------------------------------------------------------------------
# ConsoleChatController.request_mcp_approvals / resolve_pending_approval
# ---------------------------------------------------------------------------


def _pending(
    *,
    llm_name: str = "mcp__srv__tool",
    server_key: str = "local:srv",
    tool_name: str = "tool",
    server_label: str = "Srv",
    reason: str = "ask",
    arguments: dict | None = None,
) -> MCPPendingCall:
    return MCPPendingCall(
        llm_name=llm_name,
        server_key=server_key,
        tool_name=tool_name,
        server_label=server_label,
        arguments=arguments or {"a": 1},
        reason=reason,
    )


class _FakeApp:
    """`call_from_thread` stand-in: invokes the callback immediately."""

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _build_controller() -> tuple[ConsoleChatController, ConsoleChatStore]:
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=object())
    return controller, store


@pytest.mark.asyncio
async def test_request_mcp_approvals_round_trip_resolves_from_ui_thread():
    """A real worker thread blocks in `request_mcp_approvals`; the pilot
    (event-loop) thread resolves it via `resolve_pending_approval`, mirroring
    the real `ApprovalDecided` message-handler path."""
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    pending = [_pending()]

    async def resolve_soon() -> None:
        await asyncio.sleep(0.05)
        assert received and received[0] is not None
        assert received[0]["calls"][0]["llm_name"] == "mcp__srv__tool"
        assert received[0]["timeout_seconds"] == 30.0
        controller.resolve_pending_approval({"mcp__srv__tool": "approve_session"})

    decisions_task = asyncio.create_task(asyncio.to_thread(controller.request_mcp_approvals, pending))
    await resolve_soon()
    decisions = await decisions_task

    assert decisions == {"mcp__srv__tool": "approve_session"}
    # The card is always cleared afterwards, regardless of resolution path.
    assert received[-1] is None


def test_request_mcp_approvals_collapses_duplicate_llm_names_in_payload():
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    pending = [_pending(), _pending()]

    def _resolve_soon() -> None:
        time.sleep(0.05)
        controller.resolve_pending_approval({"mcp__srv__tool": "always_allow"})

    threading.Thread(target=_resolve_soon).start()
    decisions = controller.request_mcp_approvals(pending)

    assert decisions == {"mcp__srv__tool": "always_allow"}


def test_request_mcp_approvals_timeout_denies_with_timeout_for_all_undecided():
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 0.05

    started = time.monotonic()
    decisions = controller.request_mcp_approvals([_pending()])
    elapsed = time.monotonic() - started

    assert decisions == {"mcp__srv__tool": "timeout"}
    # Poll granularity is 1s (binding contract) -- deadline + one poll's slack.
    assert elapsed < 2.5
    assert received[-1] is None


def test_request_mcp_approvals_cancellation_denies_undecided():
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    def _cancel_soon() -> None:
        time.sleep(0.05)
        controller._stop_requested = True

    canceller = threading.Thread(target=_cancel_soon)
    canceller.start()
    decisions = controller.request_mcp_approvals([_pending()])
    canceller.join()

    assert decisions == {"mcp__srv__tool": "deny"}
    assert received[-1] is None


def test_request_mcp_approvals_active_cancel_event_denies_undecided():
    """The per-run `_active_cancel_event` (stop/close_session/shutdown's
    `_signal_stop`) is observed even when `_stop_requested` has already
    been reset by the coroutine side (task-227's own documented race)."""
    controller, _ = _build_controller()
    controller.mcp_approval_timeout_seconds = lambda: 30.0
    cancel_event = threading.Event()
    controller._active_cancel_event = cancel_event

    def _cancel_soon() -> None:
        time.sleep(0.05)
        cancel_event.set()

    canceller = threading.Thread(target=_cancel_soon)
    canceller.start()
    decisions = controller.request_mcp_approvals([_pending()])
    canceller.join()

    assert decisions == {"mcp__srv__tool": "deny"}


def test_switch_session_denies_a_pending_approval_round():
    controller, store = _build_controller()
    other_session = store.ensure_session(title="Other")
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    def _switch_soon() -> None:
        time.sleep(0.05)
        controller.switch_session(other_session.id)

    switcher = threading.Thread(target=_switch_soon)
    switcher.start()
    decisions = controller.request_mcp_approvals([_pending()])
    switcher.join()

    assert decisions == {"mcp__srv__tool": "deny"}


def test_resolve_pending_approval_without_active_round_is_a_noop():
    controller, _ = _build_controller()
    controller.resolve_pending_approval({"mcp__srv__tool": "deny"})  # must not raise


def test_request_mcp_approvals_with_no_pending_calls_returns_empty_and_never_surfaces_card():
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append

    assert controller.request_mcp_approvals([]) == {}
    assert received == []


# ---------------------------------------------------------------------------
# ChatScreen wiring: pending-approval state bridge + ApprovalDecided handler
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_chat_host():
    host = Mock()
    host.app_config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.7,
        }
    }
    host.chat_sidebar_collapsed = False
    host.chat_right_sidebar_collapsed = False
    host.notify = Mock()
    host.run_worker = Mock()
    host.bell = Mock()
    return host


def test_set_console_pending_approval_preserves_other_resume_fields(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    screen.chat_window = Mock()
    screen.chat_state.task_resume_state = TaskResumeState(
        summary="Keep me", last_step="Also keep"
    )

    payload = {"calls": [{"llm_name": "mcp__a__b"}], "timeout_seconds": 30.0}
    screen._set_console_pending_approval(payload)

    state = screen.chat_state.task_resume_state
    assert state.summary == "Keep me"
    assert state.last_step == "Also keep"
    assert state.pending_approval == payload
    screen.chat_window.sync_task_resume_state.assert_called_once_with(state)

    screen._set_console_pending_approval(None)
    assert screen.chat_state.task_resume_state.pending_approval is None
    assert screen.chat_state.task_resume_state.summary == "Keep me"


def test_chat_screen_forwards_approval_decided_to_controller(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    controller = Mock()
    screen._console_chat_controller = controller

    event = ChatApprovalCard.ApprovalDecided({"mcp__a__b": "deny"})
    screen.handle_console_approval_decided(event)

    controller.resolve_pending_approval.assert_called_once_with({"mcp__a__b": "deny"})


def test_chat_screen_approval_decided_handler_tolerates_no_controller(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    screen._console_chat_controller = None

    event = ChatApprovalCard.ApprovalDecided({"mcp__a__b": "deny"})
    screen.handle_console_approval_decided(event)  # must not raise
