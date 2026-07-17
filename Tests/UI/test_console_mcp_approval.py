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
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Static

import tldw_chatbook
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

_CSS_ROOT = Path(tldw_chatbook.__file__).parent / "css"
_AGENTIC_TERMINAL_TCSS = _CSS_ROOT / "components" / "_agentic_terminal.tcss"
_BUNDLED_STYLESHEET = _CSS_ROOT / "tldw_cli_modular.tcss"


def _text(widget: Static) -> str:
    return str(widget.render())


def _assert_rule_pinned_in_bundle_source_and_bundle(
    selector: str, expected_declarations: tuple[str, ...]
) -> None:
    """Shared pin-test body (T9, MCP Hub Phase 5) -- mirrors the identical
    helper in test_mcp_audit_mode.py: asserts ``selector``'s block carries
    every one of ``expected_declarations`` in BOTH the bundle-source
    component file (`_agentic_terminal.tcss`) and the generated bundle
    (`tldw_cli_modular.tcss`), proving `build_css.py` was re-run after the
    source edit."""
    agentic_terminal = _AGENTIC_TERMINAL_TCSS.read_text(encoding="utf-8")
    bundled_stylesheet = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text, label in (
        (agentic_terminal, "_agentic_terminal.tcss"),
        (bundled_stylesheet, "tldw_cli_modular.tcss"),
    ):
        start = text.find(selector)
        assert start != -1, f"{label} is missing {selector!r}"
        end = text.find("}", start)
        block = text[start:end]
        for declaration in expected_declarations:
            assert declaration in block, (
                f"{label}'s {selector!r} block is missing {declaration!r}"
            )


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
# _summarize_arguments -- redaction parity (Minor 4)
# ---------------------------------------------------------------------------


def test_summarize_arguments_redacts_secret_looking_values():
    """Minor 4: the approval card must apply the same `redact_mapping`
    boundary as every other MCP display/log surface -- pre-fix, a raw
    `api_key` argument value rendered verbatim on the card."""
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import _summarize_arguments

    text = _summarize_arguments({"api_key": "sk-super-secret-value", "q": "hello"})

    assert "sk-super-secret-value" not in text
    assert "***" in text
    assert '"q":"hello"' in text


@pytest.mark.asyncio
async def test_set_batch_redacts_secret_arguments_in_rendered_row():
    """End-to-end through the real widget row, not just the helper."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        calls = [
            {
                "llm_name": "mcp__srv__auth",
                "server_key": "local:srv",
                "tool_name": "auth",
                "server_label": "Srv",
                "arguments": {"api_key": "sk-super-secret-value"},
                "reason": "ask",
            }
        ]
        card.set_batch(calls, timeout_seconds=45.0)
        await pilot.pause()

        row = app.query_one(".approval-row-args", Static)
        rendered = _text(row)
        assert "sk-super-secret-value" not in rendered
        assert "***" in rendered


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
# CSS / geometry (T9, MCP Hub Phase 5) -- T5 deferred `.approval-row*`
# styling to this task's phase gate; `ChatApprovalCard` carries no
# `DEFAULT_CSS` of its own at all, so these bundle-source rules are the
# ONLY styling this card has anywhere.
# ---------------------------------------------------------------------------


class _CardHarnessAppWithBundledCSS(App[None]):
    """Mirrors `_CardHarnessApp` but loads the real generated bundle as
    CSS_PATH, so a batch row's header/args/decision-Select contest their
    actual CSS priority battle exactly as they do in the live Console
    screen -- mirrors `AuditModeAppWithBundledCSS` (test_mcp_audit_mode.py)
    / `ToolsModeAppWithBundledCSS` (test_mcp_tools_mode.py)."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard()


@pytest.mark.asyncio
async def test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css():
    """Without an explicit width, `_conversations.tcss`'s bare `Select {
    width: 100%; }` rule would size a row's decision Select to the FULL
    row width (not just its own share), overlapping/clipping it behind the
    header and args Statics laid out before it in the row's Horizontal --
    verified empirically before landing the fix. Asserts all three
    per-row widgets render with real size AND stay within the row's own
    bounds in left-to-right order, under the real bundled stylesheet.

    T9 (MCP Hub Phase 5): also asserts HEIGHT bounds: each `.approval-row`
    stays compact (height <= 3, should be ~1-2 lines), the `#approval-batch-rows`
    container doesn't balloon (height <= rows*3 + slack), and the
    `#approval-batch-actions` bar sits close after the rows (region.y within
    a few rows of the last row's bottom), matching the audit-mode geometry
    tests' discipline so all Horizontals/Verticals in the bundle stay compact."""
    app = _CardHarnessAppWithBundledCSS()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=45.0)
        await pilot.pause()

        rows = list(app.query(".approval-row"))
        assert len(rows) == 2
        for row in rows:
            header = row.query_one(".approval-row-header", Static)
            args = row.query_one(".approval-row-args", Static)
            select = row.query_one(".approval-row-decision", Select)

            assert header.size.width > 0 and header.size.height > 0, (
                "approval row header collapsed to zero size under bundled CSS"
            )
            assert args.size.width > 0 and args.size.height > 0, (
                "approval row args summary collapsed to zero size under bundled CSS"
            )
            assert select.size.width > 0 and select.size.height > 0, (
                "approval row decision Select collapsed to zero size under bundled CSS"
            )
            # The decision Select must not claim the row's FULL width (the
            # actual bug this CSS fixes) -- it gets a definite, bounded
            # share instead.
            assert select.size.width < row.size.width, (
                f"decision Select width {select.size.width} claimed the "
                f"entire row width {row.size.width} under bundled CSS"
            )
            assert select.size.width == 26, (
                f"decision Select width {select.size.width} != pinned 26"
            )
            # Left-to-right order, no overlap: header, then args, then the
            # decision Select, each starting no earlier than the previous
            # widget's right edge, and the Select stays inside the row.
            assert header.region.x <= args.region.x
            assert args.region.x >= header.region.right
            assert select.region.x >= args.region.right
            assert select.region.right <= row.region.right

            # T9: height bounds -- each row must stay compact (height: auto;
            # min-height: 1) instead of ballooning to 1fr (which would balloon
            # to fill the card height and push the actions bar far down).
            # Empirically measured before this fix: rows ballooning to height 9-10.
            assert row.size.height <= 4, (
                f"approval row ballooned to height {row.size.height} under "
                "bundled CSS -- height: auto; min-height: 1; is not winning"
            )

        # T9: container height bound -- the Vertical wrapping all rows must
        # also stay compact (height: auto; min-height: 0) instead of balloning
        # to 1fr and claiming the full card height, which would push the
        # #approval-batch-actions bar far down. Empirically measured before
        # this fix: container ballooning to height 19, actions pushed to y=20.
        batch_rows = app.query_one("#approval-batch-rows")
        assert batch_rows.size.height <= len(rows) * 3 + 2, (
            f"approval-batch-rows container ballooned to height "
            f"{batch_rows.size.height} (with {len(rows)} rows) under bundled CSS "
            "-- height: auto; min-height: 0; is not winning"
        )

        # T9: action bar positioning -- must sit close after the rows,
        # not far below due to container ballooning. Within a few rows'
        # worth of lines from the last row's bottom edge.
        batch_actions = app.query_one("#approval-batch-actions")
        last_row = rows[-1]
        max_y_gap = 3  # generous slack: a few rows worth of lines
        assert batch_actions.region.y <= last_row.region.bottom + max_y_gap, (
            f"approval-batch-actions bar at y={batch_actions.region.y} is too far "
            f"below last row's bottom ({last_row.region.bottom}) -- should be "
            f"within {max_y_gap} lines"
        )


def test_approval_row_decision_select_width_rule_pinned_in_bundle_source_and_bundle() -> None:
    """T9: id-scoped bundle rule directly on `.approval-row-decision`
    (a class selector -- higher specificity than `_conversations.tcss`'s
    bare `Select { width: 100%; }` type selector, so it wins regardless of
    the two files' relative concatenation order in build_css.py) -- same
    Defect-1 Select-width lesson as `#mcp-tools-filter-server-slot Select`
    / `#mcp-audit-filter-decision` above, applied to the approval card."""
    _assert_rule_pinned_in_bundle_source_and_bundle(
        ".approval-row-decision {", ("width: 26;",)
    )


def test_approval_row_height_rule_pinned_in_bundle_source_and_bundle() -> None:
    """T9: `.approval-row` (a Horizontal, which defaults to `height: 1fr`)
    needs an explicit `height: auto` -- otherwise each row would try to
    claim a `1fr` share of its `#approval-batch-rows` parent's remaining
    space instead of hugging its own single-line content, the same
    fr-inside-auto-parent collapse class documented on
    `MCPAuditMode.DEFAULT_CSS`'s Findings-view comment."""
    _assert_rule_pinned_in_bundle_source_and_bundle(
        ".approval-row {", ("height: auto;", "width: 1fr;")
    )


def test_approval_batch_rows_height_rule_pinned_in_bundle_source_and_bundle() -> None:
    """T9: `#approval-batch-rows` is a bare Vertical (default `height:
    1fr`) that would otherwise balloon to fill the card and push the
    sibling Approve-all/Submit/Deny-all action bar
    (`#approval-batch-actions`) far below the visible rows -- same bug
    class as `#mcp-perm-preview`/`#mcp-detail-builtin-toggles`/
    `#mcp-import-list`."""
    _assert_rule_pinned_in_bundle_source_and_bundle(
        "#approval-batch-rows {", ("height: auto;",)
    )


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


def test_request_mcp_approvals_cancellation_records_denied_decision_to_execution_log(tmp_path):
    """Finding I3: a stop/unmount that resolves this round via cancellation
    must still leave an audit record. Pre-fix, `run_agent_loop`'s own
    `should_cancel()` check fires for every call in the batch BEFORE any
    of them reaches `invoke()` once cancellation has resolved the round,
    so the "deny" verdict `request_mcp_approvals` hands back is never
    consumed/logged downstream -- the JSONL execution log would otherwise
    have NO record at all for a call denied this way (contrast with a
    timeout, whose calls DO still reach `invoke()`'s own gate and get
    logged there, since a timeout is not a cancellation). Uses the REAL
    `UnifiedMCPControlPlaneService` + JSONL-backed execution log (not the
    lighter `FakeMCPService`) so this proves the fix end-to-end through
    the actual persistence path."""
    from types import SimpleNamespace

    from tldw_chatbook.MCP.execution_log import MCPExecutionLog
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    store = LocalMCPStore(tmp_path / "store.json")
    store.save_profile(
        LocalExternalMCPProfile(profile_id="docs", command="python", args=("-m", "demo"))
    )
    service = UnifiedMCPControlPlaneService(
        local_service=SimpleNamespace(store=store),
        server_service=None, target_store=None, context_store=None,
    )

    controller, _ = _build_controller()
    controller.app = SimpleNamespace(
        call_from_thread=lambda fn, *a, **kw: fn(*a, **kw),
        unified_mcp_service=service,
    )
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    def _cancel_soon() -> None:
        time.sleep(0.05)
        controller._stop_requested = True

    canceller = threading.Thread(target=_cancel_soon)
    canceller.start()
    decisions = controller.request_mcp_approvals(
        [_pending(server_key="local:docs", tool_name="search", llm_name="mcp__docs__search")]
    )
    canceller.join()

    assert decisions == {"mcp__docs__search": "deny"}

    log_path = Path(store.path).with_name("mcp_execution_log.jsonl")
    records = MCPExecutionLog(log_path).read_recent()
    assert records, "the stop-mid-approval path left no audit record at all"
    assert records[0]["server_key"] == "local:docs"
    assert records[0]["tool_name"] == "search"
    assert records[0]["decision"] == "denied"
    assert records[0]["ok"] is False
    assert "run stopped while approval pending" in (records[0].get("error") or "")


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


def test_request_mcp_approvals_survives_marshal_failure_during_teardown():
    """The finally-block clear must not raise (nor destroy the computed
    decisions) when `call_from_thread` fails mid-teardown — e.g. the app
    stopped between resolution and cleanup. Regression for the `self.logger`
    AttributeError found in review: the teardown guard itself must not blow
    up."""
    controller, _ = _build_controller()
    calls: list[dict | None] = []

    class _TeardownApp:
        def call_from_thread(self, fn, *args, **kwargs):
            # Surface the card normally, then fail on the clearing call.
            if args and args[0] is None:
                raise RuntimeError("App is not running")
            calls.append(args[0] if args else None)
            return fn(*args, **kwargs)

    controller.app = _TeardownApp()
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    def _resolve_soon() -> None:
        time.sleep(0.05)
        controller.resolve_pending_approval({"mcp__srv__tool": "approve_once"})

    resolver = threading.Thread(target=_resolve_soon)
    resolver.start()
    try:
        decisions = controller.request_mcp_approvals([_pending()])
    finally:
        resolver.join()

    assert decisions == {"mcp__srv__tool": "approve_once"}
