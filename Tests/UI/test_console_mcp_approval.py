"""Tests for the live Console MCP batch-approval flow (Phase-5 task-5).

Covers the widget half (``ChatApprovalCard.set_batch``/``ApprovalDecided``)
and the controller half (``ConsoleChatController.request_mcp_approvals``/
``resolve_pending_approval``/context-change denial) of the worker-thread
<-> UI-thread approval round-trip described in
``.superpowers/sdd/task-5-brief.md``. (task-914: the legacy single-approval
API's own pinning suite, ``Tests/UI/test_chat_approvals_and_resume.py``,
was already deleted by task-649 when its last caller -- the
``Chat_Window_Enhanced`` composition -- was retired; the API itself and
its dead card body were removed here.)
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual import on

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.app import ComposeResult
from textual.widgets import Button, Select, Static, TextArea
from textual.widgets._select import SelectOverlay

import tldw_chatbook
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.run_context import current_run_id, use_run_id
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards

from Tests.UI.app_factory import _build_test_app

#: PR2a Task 5: the review hook takes the id of the run whose batch it is
#: reviewing (both gates key their per-turn verdicts by run). These
#: hook-level tests drive ONE run each, so they name it once here; every
#: assertion below is unchanged.
RUN = "run-1"

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


class _CardHarnessApp(ConsolidatedCSSApp):
    """Minimal host for `ChatApprovalCard` that records `ApprovalDecided`."""

    def __init__(self) -> None:
        super().__init__()
        self.decided: list[dict[str, str]] = []
        self.decided_round_ids: list[str | None] = []

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard()

    @on(ChatApprovalCard.ApprovalDecided)
    def _capture_decision(self, event: ChatApprovalCard.ApprovalDecided) -> None:
        self.decided.append(event.decisions)
        self.decided_round_ids.append(event.round_id)


def _raw_shell_call(command: str) -> dict:
    return {
        "llm_name": "shell_exec",
        "server_key": "local:__local__",
        "tool_name": "shell_exec",
        "server_label": "Raw CLI (unsafe host shell)",
        "arguments": {
            "command": command,
            "shell": "bash",
            "initial_directory": "/tmp/raw-shell-test",
            "timeout_seconds": 17.0,
        },
        "reason": "ask",
        "options": ["approve_once", "approve_session", "deny"],
        "call_id": "raw-call-1",
        "full_command": command,
        "warning": (
            "Runs with the full authority of the OS user. Command and output "
            "may persist in a local log."
        ),
        "scope_notice": (
            "Allow for session covers future raw shell commands in this Console session."
        ),
    }


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
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _summarize_arguments,
    )

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


def test_legacy_single_approval_api_was_removed():
    """task-914: `set_approval`/`#approval-single-body` were confirmed dead
    (no production caller ever passed a non-batch payload -- the sole
    legacy caller, the pre-task-649 `Chat_Window_Enhanced` composition,
    was already fully retired) and removed rather than wired. This pins
    the removal so the method can't quietly come back."""
    assert not hasattr(ChatApprovalCard, "set_approval")


@pytest.mark.asyncio
async def test_card_never_renders_the_retired_single_approval_buttons():
    """The card must never mount `#approval-allow-once`/`#approval-deny`
    (the retired single-approval body's buttons) -- neither in its
    default unmounted state, nor while showing a batch, nor after being
    cleared back to empty. `set_batch` is the sole production entry
    point, so this covers every state it can put the card in."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)

        for retired_id in (
            "#approval-single-body",
            "#approval-allow-once",
            "#approval-deny",
        ):
            assert not list(app.query(retired_id)), retired_id

        card.set_batch(_sample_calls(), timeout_seconds=45.0)
        await pilot.pause()
        for retired_id in (
            "#approval-single-body",
            "#approval-allow-once",
            "#approval-deny",
        ):
            assert not list(app.query(retired_id)), retired_id

        card.set_batch([], timeout_seconds=45.0)
        await pilot.pause()
        for retired_id in (
            "#approval-single-body",
            "#approval-allow-once",
            "#approval-deny",
        ):
            assert not list(app.query(retired_id)), retired_id


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

        args_summaries = [
            _text(row.query_one(".approval-row-args", Static)) for row in rows
        ]
        assert all(len(summary) <= 80 for summary in args_summaries)

        for select in app.query(Select):
            assert select.value == "approve_once"

        for button_id in (
            "#approval-approve-all",
            "#approval-submit",
            "#approval-deny-all",
        ):
            button = app.query_one(button_id, Button)
            assert button.tooltip, f"{button_id} must be tooltipped"


@pytest.mark.asyncio
async def test_raw_shell_row_shows_complete_command_and_danger_context():
    command = "printf 'first line\\n'\nprintf 'second line with [markup]\\n'"
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch([_raw_shell_call(command)], timeout_seconds=45.0)
        await pilot.pause()

        row = app.query_one(".approval-row")
        full_command = row.query_one(".approval-row-full-command", TextArea)
        assert full_command.text == command
        assert full_command.read_only is True
        assert not list(row.query(".approval-row-args"))

        metadata = _text(row.query_one(".approval-row-raw-metadata", Static))
        warning = _text(row.query_one(".approval-row-raw-warning", Static))
        scope = _text(row.query_one(".approval-row-raw-scope", Static))
        assert "Shell: bash" in metadata
        assert "Directory: /tmp/raw-shell-test" in metadata
        assert "Timeout: 17" in metadata
        assert "full authority of the OS user" in warning
        assert "local log" in warning
        assert "future raw shell commands" in scope


@pytest.mark.asyncio
async def test_raw_shell_row_defaults_to_deny_and_enter_does_not_submit():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch([_raw_shell_call("printf safe")], timeout_seconds=45.0)
        await pilot.pause()

        select = app.query_one(".approval-row-decision", Select)
        assert [value for _label, value in select._options] == [
            "approve_once",
            "approve_session",
            "deny",
        ]
        assert select.value == "deny"
        assert card.first_focus_widget_id() == select.id

        card.focus_first_decision()
        await pilot.pause()
        assert app.focused is select
        await pilot.press("enter")
        await pilot.pause()
        assert app.decided == []


def test_raw_shell_command_view_has_bounded_scrollable_geometry():
    _assert_rule_pinned_in_bundle_source_and_bundle(
        ".approval-row-full-command",
        ("width: 1fr", "height: 6", "min-height: 3"),
    )


@pytest.mark.asyncio
async def test_risk_floored_row_header_carries_a_why_affordance_tooltip():
    """Fleet-UX expert review F5/F7 (task-1234, item g): "(high risk)" on a
    plain read reads as alarmist with no explanation -- the row header
    Static now carries a tooltip naming why. `config_changed` rows (no
    risk badge) get no tooltip at all; this is scoped to `risk_floored`."""
    app = _CardHarnessApp()
    calls = [
        {
            "llm_name": "read_file",
            "server_key": "builtin",
            "tool_name": "read_file",
            "server_label": "Built-in",
            "arguments": {"path": "notes.txt"},
            "reason": "risk_floored",
        },
        {
            "llm_name": "mcp__srv_b__write",
            "server_key": "local:srv_b",
            "tool_name": "write",
            "server_label": "Srv B",
            "arguments": {"path": "/tmp/x"},
            "reason": "config_changed",
        },
    ]
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(calls, timeout_seconds=45.0)
        await pilot.pause()

        rows = list(app.query(".approval-row"))
        risk_header = rows[0].query_one(".approval-row-header", Static)
        changed_header = rows[1].query_one(".approval-row-header", Static)

        assert "(high risk)" in _text(risk_header)
        assert risk_header.tooltip == (
            "Reads can exfiltrate file contents; built-in file tools "
            "always ask before running."
        )
        assert "(definition changed)" in _text(changed_header)
        assert not changed_header.tooltip


@pytest.mark.asyncio
async def test_set_batch_row_with_options_key_narrows_the_select_and_stays_valid_default():
    """Task-5: a row's ``options`` key narrows its ``Select`` choices.

    Mirrors what task-6 will do for built-in tools (offer only the
    session-scoped choices, since persistent decisions for them cannot yet
    be undone in the UI). ``approve_once`` (the global default) is
    deliberately excluded here to prove the row-default guard: when the
    module default isn't among the narrowed options, the ``Select``'s
    initial value must fall back to the FIRST narrowed option rather than
    an out-of-list value (which `Select` would reject/misrender).
    """
    calls = _sample_calls()
    calls[0]["options"] = ["approve_session", "deny"]

    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(calls, timeout_seconds=45.0)
        await pilot.pause()

        rows = list(app.query(".approval-row"))
        narrowed_select = rows[0].query_one(".approval-row-decision", Select)
        unfiltered_select = rows[1].query_one(".approval-row-decision", Select)

        assert [value for _label, value in narrowed_select._options] == [
            "approve_session",
            "deny",
        ]
        assert narrowed_select.value == "approve_session"

        # The row with no `options` key is untouched: full four choices,
        # default `approve_once` (MCP behavior unchanged, byte-identical).
        assert [value for _label, value in unfiltered_select._options] == [
            "approve_once",
            "approve_session",
            "always_allow",
            "deny",
        ]
        assert unfiltered_select.value == "approve_once"


@pytest.mark.asyncio
async def test_bulk_approve_and_deny_all_are_row_aware_and_never_raise():
    """Important-severity fix: a batch containing narrowed rows must survive
    BOTH bulk buttons without Textual's ``InvalidSelectValueError``, and each
    row must always land on a value it legally offers -- never an
    out-of-list value silently assigned to its own ``Select``.

    Before this fix, ``_set_all_batch_decisions`` unconditionally assigned
    the bulk target to every row's ``Select.value``, which is safe only when
    every row carries all four options (true before task-5). Three rows
    here exercise every relevant shape:

      - row A: narrowed to exclude ``approve_once`` (but keeps ``deny``) --
        "Approve all" must fall back to ``approve_session`` for this row.
      - row B: narrowed to exclude ``deny`` (but keeps ``approve_once``) --
        "Deny all" must leave this row on its current legal value instead
        of crashing or assigning an illegal one.
      - row C: unnarrowed -- gets the bulk target directly, exactly as
        before task-5 (regression guard for the common MCP case).
    """
    calls = [
        {
            "llm_name": "mcp__srv_a__search",
            "server_key": "local:srv_a",
            "tool_name": "search",
            "server_label": "Srv A",
            "arguments": {},
            "reason": "ask",
            "options": ["approve_session", "deny"],  # no approve_once
        },
        {
            "llm_name": "mcp__srv_b__write",
            "server_key": "local:srv_b",
            "tool_name": "write",
            "server_label": "Srv B",
            "arguments": {},
            "reason": "ask",
            "options": ["approve_once", "approve_session", "always_allow"],  # no deny
        },
        {
            "llm_name": "mcp__srv_c__read",
            "server_key": "local:srv_c",
            "tool_name": "read",
            "server_label": "Srv C",
            "arguments": {},
            "reason": "ask",
            # no `options` key at all -- unnarrowed, all four legal.
        },
    ]

    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(calls, timeout_seconds=45.0)
        await pilot.pause()

        row_a, row_b, row_c = card._batch_selects
        assert row_a.value == "approve_session"  # default-guard fallback (task-5)
        assert row_b.value == "approve_once"
        assert row_c.value == "approve_once"

        # "Deny all" must not raise even though row B cannot legally hold "deny".
        app.query_one("#approval-deny-all", Button).press()
        await pilot.pause()
        assert row_a.value == "deny"
        assert row_b.value == "approve_once"  # untouched: no legal deny candidate
        assert row_c.value == "deny"

        # "Approve all" must not raise even though row A cannot legally hold
        # "approve_once" -- it must fall back to "approve_session" instead.
        app.query_one("#approval-approve-all", Button).press()
        await pilot.pause()
        assert row_a.value == "approve_session"
        assert row_b.value == "approve_once"
        assert row_c.value == "approve_once"


def _row_options_excluding_every_bulk_candidate() -> list[str]:
    """The only decision value that is neither an "Approve all" candidate
    (``approve_once``/``approve_session``) nor "Deny all"'s ``deny`` --
    unreachable by either bulk button. Unlike today's only shipped narrowed
    shape (``approve_once``/``approve_session``/``deny``, which BOTH bulk
    buttons can reach), a row narrowed to just this genuinely exercises the
    skip path in ``_set_all_batch_decisions``."""
    return ["always_allow"]


@pytest.mark.asyncio
async def test_bulk_actions_flag_a_row_they_could_not_apply_to():
    """Task 5: a row a bulk button could not touch must not look identical
    to a row nobody has decided on yet -- it gets a `needs-decision` class
    on its row container so the user notices it still needs an explicit
    choice. Covers both bulk buttons and confirms an applied row (here,
    the unnarrowed row) never carries the class."""
    calls = _sample_calls()
    calls[0]["options"] = _row_options_excluding_every_bulk_candidate()

    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(calls, timeout_seconds=45.0)
        await pilot.pause()

        rows = list(app.query(".approval-row"))
        row_a, row_b = rows  # row A: narrowed & unreachable; row B: unnarrowed
        assert not row_a.has_class("needs-decision")  # untouched card starts clean
        assert not row_b.has_class("needs-decision")

        app.query_one("#approval-approve-all", Button).press()
        await pilot.pause()
        assert row_a.has_class("needs-decision")
        assert not row_b.has_class("needs-decision")  # applied row: never flagged

        app.query_one("#approval-deny-all", Button).press()
        await pilot.pause()
        assert row_a.has_class("needs-decision")  # still unreachable
        assert not row_b.has_class("needs-decision")


@pytest.mark.asyncio
async def test_changing_a_flagged_rows_select_clears_needs_decision():
    """Task 5: once the user gives a flagged row its own explicit decision,
    the flag must clear -- the row's own `Select.Changed` is what a real
    interaction with the (only legal, single-option) overlay would fire.
    Posted directly (mirrors ``test_mcp_rail.py``'s
    ``rail.on_select_changed(Select.Changed(select, "local"))`` pattern)
    since this row's sole legal value is already its current value, so a
    real drive-the-overlay interaction wouldn't itself change `.value` --
    the flag-clearing contract is "the row got a Select.Changed", not
    "the value differs from its previous value"."""
    calls = _sample_calls()
    calls[0]["options"] = _row_options_excluding_every_bulk_candidate()

    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(calls, timeout_seconds=45.0)
        await pilot.pause()

        rows = list(app.query(".approval-row"))
        row_a = rows[0]
        select_a = row_a.query_one(".approval-row-decision", Select)

        app.query_one("#approval-approve-all", Button).press()
        await pilot.pause()
        assert row_a.has_class("needs-decision")

        select_a.post_message(Select.Changed(select_a, select_a.value))
        await pilot.pause()
        assert not row_a.has_class("needs-decision")


@pytest.mark.asyncio
async def test_unrelated_select_changed_does_not_touch_batch_rows():
    """A `Select.Changed` from a `Select` that isn't one of this card's
    batch-row selects must be a no-op -- guards the membership check in
    `_on_batch_row_select_changed` (it must not, say, clear every row's
    flag on any incoming event)."""
    calls = _sample_calls()
    calls[0]["options"] = _row_options_excluding_every_bulk_candidate()

    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(calls, timeout_seconds=45.0)
        await pilot.pause()

        rows = list(app.query(".approval-row"))
        row_a = rows[0]

        app.query_one("#approval-approve-all", Button).press()
        await pilot.pause()
        assert row_a.has_class("needs-decision")

        foreign_select = Select([("x", "x")], value="x", allow_blank=False)
        card._on_batch_row_select_changed(Select.Changed(foreign_select, "x"))
        await pilot.pause()
        assert row_a.has_class("needs-decision")  # unaffected by a foreign Select


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
async def test_submit_echoes_back_the_round_id_stamped_by_set_batch():
    """Task 9 fix round 1: `set_batch`'s `round_id` must round-trip
    unchanged through `ApprovalDecided` -- this is what lets
    `ConsoleChatController.resolve_pending_approval` resolve the EXACT
    round the user decided, rather than guessing from whatever session
    happens to be active when the message is handled."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=45.0, round_id="round-xyz")
        await pilot.pause()

        app.query_one("#approval-submit", Button).press()
        await pilot.pause()

        assert app.decided_round_ids == ["round-xyz"]


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


@pytest.mark.asyncio
async def test_identical_approval_round_sync_preserves_mounted_controls():
    """Resume-state sync is idempotent without suppressing real updates."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        calls = _single_call()
        card.set_batch(
            calls,
            timeout_seconds=45.0,
            round_id="round-stable",
            phase="approval",
        )
        await pilot.pause()

        generation = card._batch_generation
        select = app.query_one(".approval-row-decision", Select)
        fast_approve = app.query_one(".approval-row-fast-approve", Button)
        for _ in range(5):
            card.set_batch(
                [dict(calls[0], arguments={"query": "hello"})],
                timeout_seconds=45.0,
                round_id="round-stable",
                phase="approval",
            )
        await pilot.pause()

        assert card._batch_generation == generation
        assert app.query_one(".approval-row-decision", Select) is select
        assert app.query_one(".approval-row-fast-approve", Button) is fast_approve

        changed_calls = [dict(calls[0], arguments={"query": "changed"})]
        card.set_batch(
            changed_calls,
            timeout_seconds=45.0,
            round_id="round-stable",
            phase="approval",
        )
        await pilot.pause()
        assert card._batch_generation == generation + 1

        card.set_batch(
            changed_calls,
            timeout_seconds=45.0,
            round_id="round-stable",
            phase="finishing",
        )
        await pilot.pause()
        assert card._batch_generation == generation + 2

        card.set_batch(
            changed_calls,
            timeout_seconds=45.0,
            round_id="round-next",
            phase="approval",
        )
        await pilot.pause()
        assert card._batch_generation == generation + 3


@pytest.mark.asyncio
async def test_changed_ordinary_one_row_reuses_only_noncommitting_widgets():
    """A new ordinary one-row round updates in place without reusing commit buttons.

    Replacing the row and Select forces Textual to register, style, lay out, and
    paint the whole subtree again.  The decision controls deliberately remain
    round-scoped: reusing them would let queued old-round interaction reach the
    new round.
    """
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        first = [
            dict(
                _single_call()[0],
                effects=["network"],
                rationale="Checking the old target",
            )
        ]
        card.set_batch(first, timeout_seconds=45.0, round_id="round-reuse-old")
        await pilot.pause()

        row = app.query_one(".approval-row")
        header = row.query_one(".approval-row-header", Static)
        args = row.query_one(".approval-row-args", Static)
        select = app.query_one(".approval-row-decision", Select)
        old_fast_approve = app.query_one(".approval-row-fast-approve", Button)
        card.set_batch([], timeout_seconds=0, round_id=None)
        assert card.display is False

        changed = [
            {
                "llm_name": "mcp__srv_b__write",
                "server_key": "local:srv_b",
                "tool_name": "write",
                "server_label": "Srv B",
                "arguments": {"path": "/tmp/new.txt"},
                "reason": "ask",
                "options": ["approve_session", "deny"],
                "effects": ["private_read"],
                "rationale": "Writing the new target",
            }
        ]
        card.set_batch(changed, timeout_seconds=45.0, round_id="round-reuse-new")
        await pilot.pause()

        assert app.query_one(".approval-row") is row
        assert row.query_one(".approval-row-header", Static) is header
        assert row.query_one(".approval-row-args", Static) is args
        new_select = app.query_one(".approval-row-decision", Select)
        assert new_select is not select
        assert app.query_one(".approval-row-fast-approve", Button) is not (
            old_fast_approve
        )
        assert [value for _label, value in new_select._options] == [
            "approve_session",
            "deny",
        ]
        assert new_select.value == "approve_session"
        assert "Srv B · write" in _text(row.query_one(".approval-row-header", Static))
        assert "/tmp/new.txt" in _text(row.query_one(".approval-row-args", Static))
        assert "may read private local data" in _text(
            row.query_one(".approval-row-effects", Static)
        )
        context = next(
            widget
            for widget in row.query(Static)
            if (widget.id or "").startswith("approval-context-")
        )
        assert "Writing the new target" in _text(context)

        app.query_one(".approval-row-fast-approve", Button).press()
        await pilot.pause()
        assert app.decided == [{"mcp__srv_b__write": "approve_once"}]
        assert app.decided_round_ids == ["round-reuse-new"]


@pytest.mark.asyncio
async def test_queued_old_select_event_cannot_change_the_new_round():
    """An old overlay message must remain bound to its old decision control."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_single_call(), timeout_seconds=45.0, round_id="round-old")
        await pilot.pause()

        old_select = app.query_one(".approval-row-decision", Select)
        old_select.expanded = True
        old_select.query_one(SelectOverlay).post_message(
            SelectOverlay.UpdateSelection(1)
        )
        card.set_batch([], timeout_seconds=0, round_id=None)
        card.set_batch(
            [
                {
                    "llm_name": "mcp__srv_b__write",
                    "server_key": "local:srv_b",
                    "tool_name": "write",
                    "server_label": "Srv B",
                    "arguments": {"path": "/tmp/new.txt"},
                    "reason": "ask",
                    "options": ["approve_once", "always_allow", "deny"],
                }
            ],
            timeout_seconds=45.0,
            round_id="round-new",
        )

        await pilot.pause()
        new_select = app.query_one(".approval-row-decision", Select)
        assert new_select is not old_select
        assert new_select.value == "approve_once"
        assert new_select.expanded is False


@pytest.mark.asyncio
async def test_back_to_back_changed_rounds_leave_only_latest_controls():
    """Deferred pruning cannot leave an intermediate decision row visible."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_single_call(), timeout_seconds=45.0, round_id="round-first")
        await pilot.pause()

        def changed_call(suffix: str) -> list[dict]:
            return [
                {
                    "llm_name": f"mcp__srv__{suffix}",
                    "server_key": "local:srv",
                    "tool_name": suffix,
                    "server_label": "Srv",
                    "arguments": {"path": f"/tmp/{suffix}.txt"},
                    "reason": "ask",
                }
            ]

        card.set_batch(
            changed_call("second"),
            timeout_seconds=45.0,
            round_id="round-second",
        )
        card.set_batch(
            changed_call("third"),
            timeout_seconds=45.0,
            round_id="round-third",
        )
        await pilot.pause()

        row = app.query_one(".approval-row")
        assert len(row.query(".approval-row-controls")) == 1
        assert len(row.query(".approval-row-decision")) == 1
        assert len(row.query(".approval-row-fast-approve")) == 1
        assert len(row.query(".approval-row-fast-deny")) == 1
        assert "third" in _text(row.query_one(".approval-row-header", Static))

        row.query_one(".approval-row-decision", Select).value = "deny"
        app.query_one("#approval-submit", Button).press()
        await pilot.pause()
        assert app.decided == [{"mcp__srv__third": "deny"}]
        assert app.decided_round_ids == ["round-third"]


# ---------------------------------------------------------------------------
# Single-row fast-approval path (Fleet-UX expert review F5, task-1234)
# ---------------------------------------------------------------------------


def _single_call() -> list[dict]:
    """One unique pending call -- exercises the single-row fast-path gate."""
    return [
        {
            "llm_name": "mcp__srv_a__search",
            "server_key": "local:srv_a",
            "tool_name": "search",
            "server_label": "Srv A",
            "arguments": {"query": "hello"},
            "reason": "ask",
        }
    ]


@pytest.mark.asyncio
async def test_single_row_batch_renders_fast_approve_and_deny_buttons():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_single_call(), timeout_seconds=45.0)
        await pilot.pause()

        rows = list(app.query(".approval-row"))
        assert len(rows) == 1

        fast_approve = app.query_one(".approval-row-fast-approve", Button)
        fast_deny = app.query_one(".approval-row-fast-deny", Button)
        assert str(fast_approve.label) == "Approve once"
        assert str(fast_deny.label) == "Deny"
        assert fast_approve.tooltip
        assert fast_deny.tooltip

        # The Select+Submit path stays fully available alongside the fast
        # buttons -- this is an addition, not a replacement (e.g. "Approve
        # for session" is still only reachable through the row's Select).
        assert list(app.query(Select))
        assert app.query_one("#approval-submit", Button)


@pytest.mark.asyncio
async def test_multi_row_batch_omits_fast_buttons():
    """Multi-row cards keep the Select+Submit-only flow -- the fast path
    is gated on exactly one row (`ChatApprovalCard.set_batch`'s
    `single_row`)."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=45.0)
        await pilot.pause()

        assert len(list(app.query(".approval-row"))) == 2
        assert not list(app.query(".approval-row-fast-approve"))
        assert not list(app.query(".approval-row-fast-deny"))


@pytest.mark.asyncio
async def test_fast_approve_button_submits_approve_once_immediately():
    """SAFETY: the fast Approve button maps to `approve_once` ONLY, and
    submits through the exact same `ApprovalDecided`/`round_id` seam a
    normal Select+Submit round trip uses -- no new resolution path."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_single_call(), timeout_seconds=45.0, round_id="round-fast-1")
        await pilot.pause()

        # Never touched the Select or clicked Submit.
        app.query_one(".approval-row-fast-approve", Button).press()
        await pilot.pause()

        assert app.decided == [{"mcp__srv_a__search": "approve_once"}]
        assert app.decided_round_ids == ["round-fast-1"]


@pytest.mark.asyncio
async def test_fast_deny_button_submits_deny_immediately():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_single_call(), timeout_seconds=45.0, round_id="round-fast-2")
        await pilot.pause()

        app.query_one(".approval-row-fast-deny", Button).press()
        await pilot.pause()

        assert app.decided == [{"mcp__srv_a__search": "deny"}]
        assert app.decided_round_ids == ["round-fast-2"]


@pytest.mark.asyncio
async def test_stale_generation_fast_button_press_is_a_noop():
    """SAFETY (task-1234 review round 1): a fast button from a SUPERSEDED
    batch must never resolve the NEW round it was superseded by.
    `set_batch`'s row remount is fire-and-forget (`remove_children()`
    defers the actual detachment to the next event-loop tick -- its own
    docstring), so a stale-generation button stays mounted and pressable
    for a narrow window after a new batch arrives. Without a guard, that
    press would resolve the NEW round using whatever `self._batch_names`/
    `self._batch_round_id` the newer `set_batch` call just overwrote them
    with -- silently deciding a tool call the user never reviewed.
    `on_button_pressed` now guards on `self._batch_fast_buttons`
    membership, mirroring `_on_batch_row_select_changed`'s
    `self._batch_selects` guard."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_single_call(), timeout_seconds=45.0, round_id="round-old")
        await pilot.pause()

        stale_button = app.query_one(".approval-row-fast-approve", Button)
        assert stale_button in card._batch_fast_buttons

        new_call = [
            {
                "llm_name": "mcp__srv_c__delete",
                "server_key": "local:srv_c",
                "tool_name": "delete",
                "server_label": "Srv C",
                "arguments": {"path": "/tmp/y"},
                "reason": "ask",
            }
        ]
        # No pause between these two calls -- the stale button is still
        # mounted (remove_children's detachment is deferred) at the exact
        # moment it is pressed below, matching the real race.
        card.set_batch(new_call, timeout_seconds=45.0, round_id="round-new")
        assert stale_button not in card._batch_fast_buttons

        stale_button.press()
        await pilot.pause()

        assert app.decided == []
        assert app.decided_round_ids == []

        # The new round's OWN fast button still resolves normally --
        # the guard blocks stale presses, not the whole fast path.
        fresh_button = app.query_one(".approval-row-fast-approve", Button)
        assert fresh_button is not stale_button
        fresh_button.press()
        await pilot.pause()

        assert app.decided == [{"mcp__srv_c__delete": "approve_once"}]
        assert app.decided_round_ids == ["round-new"]


@pytest.mark.asyncio
async def test_fast_approve_button_disables_after_press_to_prevent_double_submit():
    """task-1234 review round 1: Submit/fast buttons only resolve the
    round ONCE -- disabling immediately after a press closes the
    double-submit window rather than relying on a re-render to happen
    first (previously incidental safety only)."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_single_call(), timeout_seconds=45.0, round_id="round-once")
        await pilot.pause()

        fast_approve = app.query_one(".approval-row-fast-approve", Button)
        fast_deny = app.query_one(".approval-row-fast-deny", Button)
        submit = app.query_one("#approval-submit", Button)
        assert fast_approve.disabled is False
        assert submit.disabled is False

        fast_approve.press()
        await pilot.pause()

        assert app.decided == [{"mcp__srv_a__search": "approve_once"}]
        assert fast_approve.disabled is True
        assert fast_deny.disabled is True
        assert submit.disabled is True

        # A second press is a no-op: `Button.press()` itself refuses once
        # `disabled` is True, so no second `Button.Pressed` is ever posted.
        fast_approve.press()
        await pilot.pause()
        assert app.decided == [{"mcp__srv_a__search": "approve_once"}]


@pytest.mark.asyncio
async def test_new_batch_reenables_submit_after_a_prior_round_disabled_it():
    """A NEW round must never inherit a disabled Submit button from its
    predecessor -- `set_batch` re-enables `#approval-submit` (and installs
    fresh, enabled fast buttons) at the start of every call."""
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=45.0, round_id="round-a")
        await pilot.pause()

        app.query_one("#approval-submit", Button).press()
        await pilot.pause()
        assert app.query_one("#approval-submit", Button).disabled is True

        card.set_batch(_sample_calls(), timeout_seconds=45.0, round_id="round-b")
        await pilot.pause()
        assert app.query_one("#approval-submit", Button).disabled is False


# ---------------------------------------------------------------------------
# CSS / geometry (T9, MCP Hub Phase 5) -- T5 deferred `.approval-row*`
# styling to this task's phase gate; `ChatApprovalCard` carries no
# `DEFAULT_CSS` of its own at all, so these bundle-source rules are the
# ONLY styling this card has anywhere.
# ---------------------------------------------------------------------------


def _settings_without_splash(section, key=None, default=None):
    """Disable only the splash while retaining the production application."""
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def _show_production_approval_batch(
    app,
    pilot,
    calls: list[dict],
) -> ChatApprovalCard:
    """Render an approval batch through the mounted production Console."""
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        screen = app.screen
        if (
            isinstance(screen, ChatScreen)
            and screen.is_mounted
            and screen.query("#console-task-surface")
        ):
            break
        await pilot.pause(0.05)
    else:
        raise AssertionError("Production Console did not finish mounting")

    screen.set_task_resume_state(
        TaskResumeState(
            pending_approval={
                "calls": calls,
                "timeout_seconds": 45.0,
            }
        )
    )
    expected_rows = len({str(call.get("llm_name") or "") for call in calls})
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        cards = screen.query("#chat-approval-card")
        if cards:
            card = cards.first()
            if card.display and len(card.query(".approval-row")) == expected_rows:
                await pilot.pause()
                return card
        await pilot.pause(0.05)
    raise AssertionError("Production approval batch did not finish rendering")


class _ControllerCardsHarness(ConsolidatedCSSApp):
    """Production task-card hierarchy with the real consolidated stylesheet."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self) -> None:
        super().__init__()
        self.controller: ConsoleChatController | None = None

    def compose(self) -> ComposeResult:
        yield ChatTaskCards(id="console-task-surface")

    @on(ChatApprovalCard.ApprovalDecided)
    def _resolve_controller_approval(
        self, event: ChatApprovalCard.ApprovalDecided
    ) -> None:
        if self.controller is not None:
            self.controller.resolve_pending_approval(
                event.decisions,
                round_id=event.round_id,
            )


@pytest.mark.asyncio
async def test_descriptor_effects_reach_the_mounted_production_approval_card(tmp_path):
    """A controller-marshaled local descriptor reaches the real card unchanged."""
    app = _ControllerCardsHarness()
    assert _BUNDLED_STYLESHEET.resolve() in {
        path.resolve() for path in app.css_path
    }
    async with app.run_test(size=(200, 40)) as pilot:
        cards = app.query_one(ChatTaskCards)
        gate = LocalToolProvider(
            workspace_root=tmp_path,
            resolve_state=lambda _hub: EffectiveToolState(
                state="ask", origin="global_default"
            ),
        ).pending_gate_for("fs_list", {"effects": ["network"], "path": "."})
        assert gate is not None
        call = MCPPendingCall(
            llm_name=gate.llm_name,
            server_key=gate.server_key,
            tool_name=gate.tool_name,
            server_label=gate.server_label,
            arguments=gate.arguments,
            reason=gate.reason,
            effects=gate.effects,
        )
        controller, store = _build_controller()
        session = store.ensure_session()
        controller.app = app
        controller.set_pending_approval = lambda payload: (
            cards.sync_state(TaskResumeState(pending_approval=payload))
            if payload is not None
            else None
        )
        controller.park_pending_approval = lambda _session_id: None

        pending = asyncio.create_task(
            asyncio.to_thread(
                controller.request_mcp_approvals, [call], session_id=session.id
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            card = cards.query_one(ChatApprovalCard)
            effects = card.query(".approval-row-effects")
            if card.display and len(effects) == 1:
                break
            await pilot.pause(0.05)
        else:
            controller.begin_shutdown()
            await pending
            raise AssertionError("Production approval card did not render effects")

        assert _text(effects.first()) == "Effects: may read private local data"
        assert "network" not in _text(effects.first())
        controller.resolve_pending_approval(
            {call.llm_name: "deny"}, round_id=card._batch_round_id
        )
        assert await pending == {call.llm_name: "deny"}


@pytest.mark.parametrize("crash_after_release", [False, True], ids=("success", "crash"))
@pytest.mark.asyncio
async def test_approved_definitive_tool_stays_mounted_until_real_terminal(
    tmp_path, crash_after_release
):
    """Approval becomes a disabled finishing card until the keyed terminal.

    This uses the production task-card hierarchy, a real local descriptor,
    the real review bridge, and ``AgentService._make_invoke_tool``.  The
    handler is event-blocked so the assertion cannot race a fast mutation;
    the crash case proves the same terminal cleanup runs for ``BaseException``.
    """
    from tldw_chatbook.Agents.agent_models import (
        AgentConfig,
        RunBudget,
        ToolCall,
        ToolResult,
    )
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_chat_controller import build_local_review_hook
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )
    from tldw_chatbook.Tools.watchlists_command_service import (
        WatchlistsCommandService,
    )

    app = _ControllerCardsHarness()
    entered = threading.Event()
    release = threading.Event()
    result_box: dict[str, ToolResult] = {}
    subscriptions = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    bundles = WatchlistBundleService(subscriptions)

    def blocked_create(**kwargs):
        entered.set()
        if not release.wait(5):
            raise AssertionError("test never released definitive handler")
        if crash_after_release:
            raise SystemExit("secret definitive crash detail")
        return bundles.create_with_sources(**kwargs)

    def unavailable(*_args, **_kwargs):
        return None

    commands = WatchlistsCommandService(
        runtime_source_loader=lambda: "local",
        create_sources_batch=unavailable,
        create_collection=blocked_create,
        update_collection_sources=unavailable,
    )
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        watchlists_command_service=commands,
        resolve_state=lambda _hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    controller, store = _build_controller()
    session = store.ensure_session()
    run_id = "run-definitive-card"
    call = ToolCall(
        name="watchlists_create_collection",
        args={"name": "Threat intel", "if_exists": "auto_suffix"},
        call_id="call-definitive-card",
    )

    async with app.run_test(size=(200, 40)) as pilot:
        cards = app.query_one(ChatTaskCards)
        app.controller = controller
        controller.app = app
        controller.set_pending_approval = lambda payload: cards.sync_state(
            TaskResumeState(pending_approval=payload)
        )
        controller.park_pending_approval = lambda _session_id: None
        controller.mcp_approval_timeout_seconds = lambda: 30.0
        review = build_local_review_hook(
            provider,
            lambda pending: controller.request_mcp_approvals(
                pending, session_id=session.id
            ),
        )
        service = AgentService(
            db=AgentRunsDB(tmp_path / "runs.db", "test"),
            registry=registry,
            chat_call=lambda **_kwargs: {"choices": [{"message": {"content": "x"}}]},
            on_tool_terminal=controller.complete_definitive_tool,
            on_run_terminal=controller.complete_definitive_run,
        )
        invoke = service._make_invoke_tool(
            AgentConfig(
                model="test",
                system_prompt="s",
                allowed_tools=(call.name,),
                budget=RunBudget(max_tool_call_seconds=0.001),
            ),
            disclosed_names={call.name},
            run_id=run_id,
        )

        def run_tool() -> None:
            with use_run_id(run_id):
                verdicts = review([call], run_id)
                assert verdicts.get(call.name) == "proceed"
                result_box["result"] = invoke(call)

        worker = threading.Thread(target=run_tool)
        worker.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            card = cards.query_one(ChatApprovalCard)
            fast = list(card.query(".approval-row-fast-approve"))
            if card.display and fast:
                break
            await pilot.pause(0.05)
        else:
            controller.begin_shutdown()
            worker.join(2)
            raise AssertionError("approval card did not mount")

        await pilot.click(f"#{fast[0].id}")
        assert entered.wait(2), "approved handler never started"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if _text(card.query_one("#approval-title", Static)).startswith(
                "Finishing"
            ):
                break
            await pilot.pause(0.05)
        else:
            release.set()
            await asyncio.to_thread(worker.join, 2)
            raise AssertionError("approval card never entered finishing state")

        assert _text(card.query_one("#approval-title", Static)) == (
            "Finishing — Stop will not cancel"
        )
        assert card.display is True
        assert all(select.disabled for select in card.query(Select))
        assert all(button.disabled for button in card.query(Button))
        assert worker.is_alive(), "runtime returned before the real tool terminal"

        release.set()
        await asyncio.to_thread(worker.join, 3)
        assert not worker.is_alive()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and card.display:
            await pilot.pause(0.05)
        assert card.display is False

    result = result_box["result"]
    if crash_after_release:
        assert result.ok is False
        assert result.error == "tool call failed: watchlists_create_collection"
        assert "secret" not in result.error
        assert bundles.list_watchlists() == []
    else:
        assert result.ok is True
        assert [row["name"] for row in bundles.list_watchlists()] == ["Threat intel"]


@pytest.mark.parametrize(
    "failure_type",
    [SystemExit, asyncio.CancelledError],
    ids=("system-exit", "cancelled-error"),
)
def test_run_terminal_sweeps_approved_undispatched_row_after_base_exception(
    tmp_path, monkeypatch, failure_type
):
    """The loop terminal observer runs once even when control flow escapes."""
    import tldw_chatbook.Agents.agent_service as agent_service_module
    from tldw_chatbook.Agents.agent_models import AgentConfig
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import (
        ToolCatalogRegistry,
        ToolExecutionPolicy,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    controller, store = _build_controller()
    session = store.ensure_session()
    received: list[dict | None] = []
    terminal_calls: list[str] = []
    run_id = f"run-base-exception-{failure_type.__name__}"
    call = MCPPendingCall(
        llm_name="watchlists_create_collection",
        server_key="local:__local__",
        tool_name="watchlists_create_collection",
        server_label="Local",
        arguments={"name": "Threat intel", "if_exists": "auto_suffix"},
        reason="ask",
        call_id="call-never-dispatched",
        execution_policy=ToolExecutionPolicy.DEFINITIVE_AFTER_START,
    )
    controller.app = _FakeApp()
    controller.mcp_approval_timeout_seconds = lambda: 2.0
    controller.set_pending_approval = received.append

    decision_box: dict[str, dict[str, str]] = {}

    def request_approval() -> None:
        with use_run_id(run_id):
            decision_box["decisions"] = controller.request_mcp_approvals(
                [call], session_id=session.id
            )

    approval_worker = threading.Thread(target=request_approval, daemon=True)
    approval_worker.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not received:
        time.sleep(0.01)
    assert received, "approved-but-undispatched row never mounted"
    payload = received[-1]
    assert payload is not None
    controller.resolve_pending_approval(
        {call.call_id: "approve_once"}, round_id=str(payload["round_id"])
    )
    approval_worker.join(2)
    assert not approval_worker.is_alive()
    assert decision_box["decisions"] == {call.call_id: "approve_once"}
    retained = received[-1]
    assert retained is not None
    assert retained["phase"] == "finishing"
    assert retained["run_id"] == run_id

    def fail_after_approval(*_args, **_kwargs):
        assert current_run_id() == run_id
        raise failure_type("loop terminal")

    monkeypatch.setattr(agent_service_module, "run_agent_loop", fail_after_approval)

    def observe_terminal(run_id: str) -> None:
        terminal_calls.append(run_id)
        controller.complete_definitive_run(run_id)

    runs = AgentRunsDB(tmp_path / "base-exception-runs.db", "test")
    runs.create_run(
        conversation_id="conversation",
        agent_kind="primary",
        run_id=run_id,
    )
    monkeypatch.setattr(runs, "create_run", lambda **_kwargs: run_id)
    service = AgentService(
        db=runs,
        registry=ToolCatalogRegistry(),
        chat_call=lambda **_kwargs: {"choices": [{"message": {"content": "x"}}]},
        on_run_terminal=observe_terminal,
    )
    with pytest.raises(failure_type):
        service.run_turn(
            conversation_id="conversation",
            messages=[{"role": "user", "content": "go"}],
            config=AgentConfig(
                model="test", system_prompt="s", allowed_tools=()
            ),
            api_endpoint="openai",
        )

    assert terminal_calls == [run_id]
    assert received[-1] is None
    assert controller._parked_approval_payloads == {}


def test_local_same_name_finishing_rows_complete_by_call_id_out_of_order(tmp_path):
    """Two approved local mutations remain independently addressable."""
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import build_local_review_hook

    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )
    controller, store = _build_controller()
    session = store.ensure_session()
    received: list[dict | None] = []
    result_box: dict[str, dict[str, str]] = {}
    run_id = "run-local-same-name"
    calls = [
        ToolCall(
            name="watchlists_create_collection",
            args={"name": "First", "if_exists": "conflict"},
            call_id="call-first",
        ),
        ToolCall(
            name="watchlists_create_collection",
            args={"name": "Second", "if_exists": "conflict"},
            call_id="call-second",
        ),
    ]
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 2.0
    hook = build_local_review_hook(
        provider,
        lambda pending: controller.request_mcp_approvals(
            pending, session_id=session.id
        ),
    )

    def review() -> None:
        with use_run_id(run_id):
            result_box["verdicts"] = hook(calls, run_id)

    worker = threading.Thread(target=review, daemon=True)
    worker.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not received:
        time.sleep(0.01)
    assert received, "same-name local approval batch never mounted"
    payload = received[-1]
    assert payload is not None
    payload_calls = list(payload["calls"])
    decisions = {
        str(row.get("call_id") or row["llm_name"]): "approve_once"
        for row in payload_calls
    }
    controller.resolve_pending_approval(
        decisions, round_id=str(payload["round_id"])
    )
    worker.join(2)
    assert not worker.is_alive()

    finishing = received[-1]
    assert finishing is not None
    assert finishing["phase"] == "finishing"
    assert [row["call_id"] for row in finishing["calls"]] == [
        "call-first",
        "call-second",
    ]

    controller.complete_definitive_tool(
        run_id, "call-second", "watchlists_create_collection"
    )
    remaining = received[-1]
    assert remaining is not None
    assert [row["call_id"] for row in remaining["calls"]] == ["call-first"]

    controller.complete_definitive_tool(
        run_id, "call-first", "watchlists_create_collection"
    )
    assert received[-1] is None


def test_definitive_tool_terminal_falls_back_to_name_for_empty_call_id():
    """A fence/legacy terminal removes one no-id row and keeps its sibling."""
    controller, store = _build_controller()
    session = store.ensure_session()
    run_id = "run-no-call-id"
    tool_name = "watchlists_create_collection"
    target = {"llm_name": tool_name, "call_id": ""}
    sibling = {"llm_name": tool_name, "call_id": "call-sibling"}
    controller._parked_approval_payloads["round-no-call-id"] = {
        "round_id": "round-no-call-id",
        "session_id": session.id,
        "run_id": run_id,
        "phase": "finishing",
        "calls": [target, sibling],
    }

    controller.complete_definitive_tool(run_id, tool_name, tool_name)

    retained = controller._parked_approval_payloads["round-no-call-id"]
    assert retained["calls"] == [sibling]


@pytest.mark.asyncio
async def test_finishing_card_is_not_counted_and_keyboard_focuses_the_card():
    """Finishing is status, not a pending decision or disabled focus target."""
    app = _build_test_app()
    with patch(
        "tldw_chatbook.app.get_cli_setting", side_effect=_settings_without_splash
    ):
        async with app.run_test(size=(200, 40)) as pilot:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                screen = app.screen
                if isinstance(screen, ChatScreen) and screen.is_mounted:
                    break
                await pilot.pause(0.05)
            else:
                raise AssertionError("Production Console did not finish mounting")

            screen.set_task_resume_state(
                TaskResumeState(
                    pending_approval={
                        "calls": _single_call(),
                        "timeout_seconds": 0.0,
                        "round_id": "round-finishing-focus",
                        "phase": "finishing",
                    }
                )
            )
            await pilot.pause()
            card = screen.query_one(ChatApprovalCard)

            assert card.display is True
            assert screen._console_pending_approval_count() == 0
            assert all(select.disabled for select in card.query(Select))

            card.focus_first_decision()
            await pilot.pause()

            assert card.can_focus is True
            assert app.focused is card


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
    stays compact (height <= 6 -- three stacked lines since TASK-1846 split
    header/args/controls, plus slack for a multi-arg collapsed row; a row
    that lost `height: auto` balloons to ~15), the `#approval-batch-rows`
    container doesn't balloon (height <= rows*3 + slack), and the
    `#approval-batch-actions` bar sits close after the rows (region.y within
    a few rows of the last row's bottom), matching the audit-mode geometry
    tests' discipline so all Horizontals/Verticals in the bundle stay compact."""
    app = _build_test_app()
    with patch(
        "tldw_chatbook.app.get_cli_setting", side_effect=_settings_without_splash
    ):
        async with app.run_test(size=(200, 40)) as pilot:
            card = await _show_production_approval_batch(app, pilot, _sample_calls())

            rows = list(card.query(".approval-row"))
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
                # TASK-1846: the row is three stacked lines now -- header,
                # arguments, then `.approval-row-controls` -- so neither text
                # widget shares a line with a fixed-width control. The
                # left-to-right ordering this used to assert (`args.x >=
                # header.right`) no longer describes the layout, so the
                # guarantee it was protecting -- nothing overlaps anything --
                # is asserted directly instead.
                assert select.region.right <= row.region.right
                assert args.region.y >= header.region.bottom, (
                    "the arguments did not drop below the header"
                )
                assert select.region.y >= args.region.bottom, (
                    "the controls did not drop below the arguments"
                )
                # The whole point of the split: arguments get the row, not a
                # leftover share of it after 54 cells of fixed-width controls.
                assert args.region.width >= row.region.width - 2, (
                    f"arguments got {args.region.width} of {row.region.width} "
                    "cells -- the controls are still eating the row"
                )
                for a, b in ((header, args), (select, args), (header, select)):
                    assert not (
                        a.region.x < b.region.right
                        and b.region.x < a.region.right
                        and a.region.y < b.region.bottom
                        and b.region.y < a.region.bottom
                    ), f"{a.classes} overlaps {b.classes}: {a.region} vs {b.region}"

                # T9: height bounds -- each row must stay compact (height: auto;
                # min-height: 1) instead of ballooning to 1fr (which would balloon
                # to fill the card height and push the actions bar far down).
                # Empirically measured before this fix: rows ballooning to height 9-10.
                # TASK-1846: 4 -> 6. The row gained a line when the
                # arguments moved to their own, and a collapsed `xN` row may
                # legitimately render several argument sets. A row that has
                # lost `height: auto` balloons to 15, so this still catches it.
                assert row.size.height <= 6, (
                    f"approval row ballooned to height {row.size.height} under "
                    "bundled CSS -- height: auto; min-height: 1; is not winning"
                )

            # T9: container height bound -- the Vertical wrapping all rows must
            # also stay compact (height: auto; min-height: 0) instead of balloning
            # to 1fr and claiming the full card height, which would push the
            # #approval-batch-actions bar far down. Empirically measured before
            # this fix: container ballooning to height 19, actions pushed to y=20.
            batch_rows = card.query_one("#approval-batch-rows")
            # TASK-1846: per-row budget 3 -> 6 (a row is two lines now and a
            # collapsed row may carry several argument sets). Still catches a
            # balloon: the container is capped at 15, so two ballooned rows
            # clamp to 15 and blow this bound.
            assert batch_rows.size.height <= len(rows) * 6 + 2, (
                f"approval-batch-rows container ballooned to height "
                f"{batch_rows.size.height} (with {len(rows)} rows) under bundled CSS "
                "-- height: auto; min-height: 0; is not winning"
            )

            # T9: action bar positioning -- must sit close after the rows,
            # not far below due to container ballooning. Within a few rows'
            # worth of lines from the last row's bottom edge.
            batch_actions = card.query_one("#approval-batch-actions")
            last_row = rows[-1]
            max_y_gap = 3  # generous slack: a few rows worth of lines
            assert batch_actions.region.y <= last_row.region.bottom + max_y_gap, (
                f"approval-batch-actions bar at y={batch_actions.region.y} is too far "
                f"below last row's bottom ({last_row.region.bottom}) -- should be "
                f"within {max_y_gap} lines"
            )


@pytest.mark.asyncio
async def test_single_row_fast_buttons_have_nonzero_geometry_and_do_not_overlap_under_bundled_css():
    """task-1234 review round 1: the sibling multi-row geometry test above
    only ever mounts a 2-row batch, so `.approval-row-fast-approve`/
    `.approval-row-fast-deny` (single-row-only, see `set_batch`'s
    `single_row` gate) never actually render under the REAL bundled
    stylesheet in any existing test. Asserts both fast buttons get real,
    non-overlapping geometry, sit after the decision Select, and stay
    inside the row's own bounds -- the same class of `Select { width:
    100%; }`-style cascade surprise the sibling test guards against, now
    for the two new buttons.

    The production Console is allowed to mount and settle before the pending
    approval state is delivered, matching the real worker-to-UI round trip."""
    app = _build_test_app()
    with patch(
        "tldw_chatbook.app.get_cli_setting", side_effect=_settings_without_splash
    ):
        async with app.run_test(size=(200, 40)) as pilot:
            card = await _show_production_approval_batch(app, pilot, _single_call())

            rows = list(card.query(".approval-row"))
            assert len(rows) == 1
            row = rows[0]
            select = row.query_one(".approval-row-decision", Select)
            fast_approve = row.query_one(".approval-row-fast-approve", Button)
            fast_deny = row.query_one(".approval-row-fast-deny", Button)

            for widget, label in (
                (fast_approve, "fast-approve"),
                (fast_deny, "fast-deny"),
            ):
                assert widget.size.width > 0 and widget.size.height > 0, (
                    f"approval row {label} button collapsed to zero size under "
                    "bundled CSS"
                )
                assert widget.size.width == 14, (
                    f"{label} button width {widget.size.width} != pinned 14"
                )

            # Left-to-right order, no overlap: Select, then fast-approve, then
            # fast-deny, each starting no earlier than the previous widget's
            # right edge, and both fast buttons stay inside the row.
            assert fast_approve.region.x >= select.region.right
            assert fast_deny.region.x >= fast_approve.region.right
            assert fast_approve.region.right <= row.region.right
            assert fast_deny.region.right <= row.region.right

            # Compact row (same discipline as the sibling test). TASK-1846
            # made it two lines -- headline + full-width arguments -- so the
            # bound moves 4 -> 6; a row that lost `height: auto` is 15.
            assert row.size.height <= 6, (
                f"single-row approval row ballooned to height {row.size.height} "
                "under bundled CSS"
            )


def test_approval_row_decision_select_width_rule_pinned_in_bundle_source_and_bundle() -> (
    None
):
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
    `MCPAuditMode.BUNDLED_CSS`'s Findings-view comment."""
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
    call_id: str = "",
) -> MCPPendingCall:
    return MCPPendingCall(
        llm_name=llm_name,
        server_key=server_key,
        tool_name=tool_name,
        server_label=server_label,
        arguments=arguments or {"a": 1},
        reason=reason,
        call_id=call_id,
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
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "approve_session"}, round_id=received[0]["round_id"]
        )

    decisions_task = asyncio.create_task(
        asyncio.to_thread(controller.request_mcp_approvals, pending)
    )
    await resolve_soon()
    decisions = await decisions_task

    assert decisions == {"mcp__srv__tool": "approve_session"}
    # The card is always cleared afterwards, regardless of resolution path.
    assert received[-1] is None


def test_request_mcp_approvals_routes_one_decision_to_duplicate_names():
    """Duplicate same-name rows resolve through ONE round trip and map.

    TASK-294 rename: this was called `..._collapses_duplicate_llm_names_in_
    payload`, but nothing here asserts payload collapsing -- visual
    collapsing lives in `_collapse_pending_calls` (and since TASK-1861 is
    keyed per call id). What this proves is the round trip: two pending rows
    sharing a name, one card decision, one decisions map back.
    """
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    pending = [_pending(), _pending()]

    def _resolve_soon() -> None:
        time.sleep(0.05)
        assert received and received[-1] is not None
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "always_allow"}, round_id=received[-1]["round_id"]
        )

    threading.Thread(target=_resolve_soon).start()
    decisions = controller.request_mcp_approvals(pending)

    assert decisions == {"mcp__srv__tool": "always_allow"}


def test_request_mcp_approvals_preserves_native_call_ids_as_verdict_keys():
    """Two same-tool native calls remain independently addressable."""
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0
    pending = [
        _pending(call_id="call-a", arguments={"path": "a.txt"}),
        _pending(call_id="call-b", arguments={"path": "b.txt"}),
    ]

    def _resolve_soon() -> None:
        time.sleep(0.05)
        payload = received[-1]
        assert payload is not None
        assert [call["call_id"] for call in payload["calls"]] == [
            "call-a",
            "call-b",
        ]
        controller.resolve_pending_approval(
            {"call-a": "approve_once", "call-b": "deny"},
            round_id=payload["round_id"],
        )

    threading.Thread(target=_resolve_soon).start()
    decisions = controller.request_mcp_approvals(pending)

    assert decisions == {"call-a": "approve_once", "call-b": "deny"}


def test_run_terminal_clears_approved_definitive_row_never_dispatched():
    from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy

    controller, store = _build_controller()
    session = store.ensure_session()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0
    call = MCPPendingCall(
        llm_name="watchlists_create_collection",
        server_key="local:__local__",
        tool_name="watchlists_create_collection",
        server_label="Local",
        arguments={"name": "Threat intel"},
        reason="ask",
        execution_policy=ToolExecutionPolicy.DEFINITIVE_AFTER_START,
    )
    result_box: dict[str, dict[str, str]] = {}

    def request() -> None:
        with use_run_id("run-never-dispatched"):
            result_box["decisions"] = controller.request_mcp_approvals(
                [call], session_id=session.id
            )

    worker = threading.Thread(target=request)
    worker.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not received:
        time.sleep(0.01)
    payload = received[-1]
    assert payload is not None
    controller.resolve_pending_approval(
        {call.llm_name: "approve_once"}, round_id=payload["round_id"]
    )
    worker.join(2)

    assert result_box["decisions"] == {call.llm_name: "approve_once"}
    assert received[-1] is not None
    assert received[-1]["phase"] == "finishing"

    controller.complete_definitive_run("run-never-dispatched")

    assert received[-1] is None
    assert controller._parked_approval_payloads == {}


@pytest.mark.parametrize("retained_phase", ["approval", "finishing"])
def test_close_session_discards_its_approved_definitive_row(retained_phase):
    from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy

    controller, store = _build_controller()
    session = store.ensure_session()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0
    call = MCPPendingCall(
        llm_name="watchlists_create_collection",
        server_key="local:__local__",
        tool_name="watchlists_create_collection",
        server_label="Local",
        arguments={"name": "Threat intel"},
        reason="ask",
        execution_policy=ToolExecutionPolicy.DEFINITIVE_AFTER_START,
    )

    def request() -> None:
        with use_run_id("run-session-close"):
            controller.request_mcp_approvals([call], session_id=session.id)

    worker = threading.Thread(target=request)
    worker.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not received:
        time.sleep(0.01)
    payload = received[-1]
    assert payload is not None
    controller.resolve_pending_approval(
        {call.llm_name: "approve_once"}, round_id=payload["round_id"]
    )
    worker.join(2)
    assert received[-1] is not None
    assert received[-1]["phase"] == "finishing"
    # ``approval`` represents close winning the lock immediately before the
    # request thread can publish its finishing transition.
    received[-1]["phase"] = retained_phase

    controller.close_session(session.id)

    assert received[-1] is None
    assert controller._parked_approval_payloads == {}


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
    """F5 fix (Qodo wave): this test's INTENT is "global stop denies every
    in-flight approval round" -- real process teardown, not a single
    session's Stop. It used to flip the bare, session-agnostic
    `_stop_requested` flag to exercise that; F5 removed `_stop_requested`
    from the bridge's poll (a single session's Stop must not cross-cancel
    an unrelated session's approval round any more), so the equivalent
    "global" signal is the production app-exit path `begin_shutdown()`, which
    sets the visit Event AND cancels headless-bound rounds -- a raw
    `_shutdown_requested.set()` poke stopped reaching never-visited
    controllers' rounds when the Qodo-S2 fix (PR #1799) made those bind the
    headless cancel signal. Drive the seam, not the flag."""
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    def _cancel_soon() -> None:
        time.sleep(0.05)
        controller.begin_shutdown()

    canceller = threading.Thread(target=_cancel_soon)
    canceller.start()
    decisions = controller.request_mcp_approvals([_pending()])
    canceller.join()

    assert decisions == {"mcp__srv__tool": "deny"}
    assert received[-1] is None


def test_request_mcp_approvals_active_cancel_event_denies_undecided():
    """The per-run cancel event (stop/close_session/shutdown's
    `_signal_stop`) is observed even when `_stop_requested` has already
    been reset by the coroutine side (task-227's own documented race).
    Task 3b: registered under the ACTIVE session's key -- `request_mcp_
    approvals` has no session id of its own to key by (see
    `_is_active_session_cancelled`'s docstring), so it falls back to
    whatever session is currently active, same as `stop_active_run`."""
    controller, _ = _build_controller()
    controller.mcp_approval_timeout_seconds = lambda: 30.0
    cancel_event = threading.Event()
    controller._active_cancel_events[controller.store.active_session_id or ""] = (
        cancel_event
    )

    def _cancel_soon() -> None:
        time.sleep(0.05)
        cancel_event.set()

    canceller = threading.Thread(target=_cancel_soon)
    canceller.start()
    decisions = controller.request_mcp_approvals([_pending()])
    canceller.join()

    assert decisions == {"mcp__srv__tool": "deny"}


def test_request_mcp_approvals_unrelated_session_stop_does_not_cross_cancel():
    """F5 fix (Qodo wave): stopping a DIFFERENT session must not deny THIS
    session's in-flight approval round. Before the fix, `_stop_requested`
    (set globally by `_signal_stop` for ANY session's Stop/Close) was OR'd
    into the bridge's own poll check, so any session's Stop denied every
    session's in-flight approval round -- a narrower, approval-round-only
    echo of the stream cross-cancellation Critical-1 already fixed for
    the stream itself. `_signal_stop` still flips the (now bridge-inert)
    `_stop_requested` flag here; only the UNRELATED session's own cancel
    event is registered, and the store has no active session at all
    (fresh `ConsoleChatStore()`), so `_is_active_session_cancelled` cannot
    match it either -- the round must resolve via the real, explicit
    decision below, not get denied out from under it."""
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    def _stop_unrelated_session_soon() -> None:
        time.sleep(0.05)
        controller._signal_stop(session_id="unrelated-session-id")

    def _resolve_soon() -> None:
        time.sleep(0.2)
        assert received and received[-1] is not None
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "approve_once"}, round_id=received[-1]["round_id"]
        )

    stopper = threading.Thread(target=_stop_unrelated_session_soon)
    resolver = threading.Thread(target=_resolve_soon)
    stopper.start()
    resolver.start()
    decisions = controller.request_mcp_approvals([_pending()])
    stopper.join()
    resolver.join()

    assert decisions == {"mcp__srv__tool": "approve_once"}


def test_request_mcp_approvals_cancellation_records_denied_decision_to_execution_log(
    tmp_path,
):
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
    the actual persistence path.

    F5 fix (Qodo wave): flips `_shutdown_requested` rather than the bare
    `_stop_requested` -- see `test_request_mcp_approvals_cancellation_
    denies_undecided`'s own docstring for why."""
    from types import SimpleNamespace

    from tldw_chatbook.MCP.execution_log import MCPExecutionLog
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    store = LocalMCPStore(tmp_path / "store.json")
    store.save_profile(
        LocalExternalMCPProfile(
            profile_id="docs", command="python", args=("-m", "demo")
        )
    )
    service = UnifiedMCPControlPlaneService(
        local_service=SimpleNamespace(store=store),
        server_service=None,
        target_store=None,
        context_store=None,
    )

    controller, _ = _build_controller()
    controller.app = SimpleNamespace(
        call_from_thread=lambda fn, *a, **kw: fn(*a, **kw),
        unified_mcp_service=service,
    )
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    def _cancel_soon() -> None:
        time.sleep(0.05)
        controller.begin_shutdown()

    canceller = threading.Thread(target=_cancel_soon)
    canceller.start()
    decisions = controller.request_mcp_approvals(
        [
            _pending(
                server_key="local:docs",
                tool_name="search",
                llm_name="mcp__docs__search",
            )
        ]
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
    assert records[0]["error_category"] == "approval_cancelled"
    assert "error" not in records[0]
    assert "run stopped while approval pending" not in str(records[0])


def test_switch_session_parks_rather_than_denies_a_pending_approval_round():
    """PA-T9 supersedes the old deny-on-switch contract this test used to
    assert (`test_switch_session_denies_a_pending_approval_round`, removed):
    pre-Task-9, only ONE approval round could ever be in flight controller-
    wide, so `switch_session` force-denied it unconditionally on ANY
    switch. Once a background session can carry its own live round (Task
    3's concurrent runs + this task's parking design), that assumption no
    longer holds -- switching away now PARKS the round (fleet badge, no
    denial) instead, and it only resolves once the owning session is
    revisited and a decision is actually submitted (or it independently
    times out/cancels).

    Final-review CRITICAL 1 strengthening: this test originally resolved
    via the no-token active-session fallback
    (`resolve_pending_approval(decisions)`, no `round_id`), which masked a
    real bug -- `_parked_approval_payloads` was populated ONLY for a
    PARKED round, so switching back to `owning_session` (which mounted
    immediately and was NEVER parked) found nothing to re-derive the card
    from. The fallback resolve "worked" anyway (it doesn't need the card
    to be mounted), so the test passed despite the card being permanently
    gone. Now asserts the card actually RE-MOUNTS on the switch back
    (mirroring what a real `ChatApprovalCard` would show) and resolves via
    the round_id THAT mount carried, exercising the real re-derive path
    rather than a resolve call that stands in for it.
    """
    controller, store = _build_controller()
    owning_session = store.create_session(title="Owning").id
    other_session = store.create_session(title="Other").id
    store.switch_session(owning_session)
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    result_holder: dict[str, dict[str, str]] = {}

    def _run_round() -> None:
        result_holder["decisions"] = controller.request_mcp_approvals(
            [_pending()], session_id=owning_session
        )

    worker = threading.Thread(target=_run_round)
    worker.start()
    time.sleep(0.1)
    # `owning_session` WAS the active session at round-start, so it mounted
    # immediately (no parking) -- same as every pre-Task-9 call site.
    assert mounted and mounted[-1] is not None

    controller.switch_session(other_session)
    time.sleep(0.05)
    assert "decisions" not in result_holder  # not denied by the switch
    assert mounted[-1] is None  # the departing session's card is cleared

    controller.switch_session(owning_session)
    # CRITICAL 1 fix: switching back re-mounts the SAME round's card --
    # pre-fix, `mounted[-1]` would still be `None` here (no retained
    # payload for a round that was never parked).
    assert mounted[-1] is not None
    round_id = mounted[-1]["round_id"]
    controller.resolve_pending_approval(
        {"mcp__srv__tool": "approve_once"}, round_id=round_id
    )
    worker.join(timeout=2.0)

    assert result_holder["decisions"] == {"mcp__srv__tool": "approve_once"}


def test_request_mcp_approvals_parks_for_a_non_active_session():
    """PA-T9: a round whose `session_id` differs from the store's ACTIVE
    session parks -- no card mount (`set_pending_approval` never called
    with a real payload), the run-marker pending flag flips, and
    `park_pending_approval` fires exactly once. Visiting (switching to)
    the owning session later mounts the SAME retained payload and lets it
    resolve normally."""
    controller, store = _build_controller()
    viewed = store.create_session(title="Viewed").id
    background = store.create_session(title="Background").id
    store.switch_session(viewed)  # keep viewing the first session
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append
    parked: list[str] = []
    controller.park_pending_approval = parked.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    result_holder: dict[str, dict[str, str]] = {}

    def _run_round() -> None:
        result_holder["decisions"] = controller.request_mcp_approvals(
            [_pending()], session_id=background
        )

    worker = threading.Thread(target=_run_round)
    worker.start()
    time.sleep(0.1)

    assert parked == [background]
    assert mounted == []  # never mounted -- the active session's card is untouched
    assert background in controller._pending_approvals
    assert controller.run_marker_for(background) is ConsoleRunMarker.NEEDS_APPROVAL

    # Visiting + deciding resolves it.
    controller.switch_session(background)
    assert mounted and mounted[-1] is not None
    round_id = mounted[-1]["round_id"]
    controller.resolve_pending_approval(
        {"mcp__srv__tool": "approve_once"}, round_id=round_id
    )
    worker.join(timeout=2.0)

    assert result_holder["decisions"] == {"mcp__srv__tool": "approve_once"}
    assert background not in controller._pending_approvals
    # Mounted once on visit, cleared once resolved.
    assert len(mounted) == 2
    assert mounted[0] is not None
    assert mounted[1] is None


def test_mcp_round_and_skill_install_round_for_the_same_session_both_keep_the_badge_up():
    """TASK-1050 (Defect A): the pending-approval badge used to be a single
    boolean per session shared by all three approval-like bridges (MCP
    approvals, skill-install confirms, skill-script confirms) --
    whichever bridge's round resolved first cleared the badge even if a
    SIBLING round from a DIFFERENT bridge was still outstanding for that
    same session. Exercises the exact scenario the task names: an MCP
    round and a skill-install round parked for the SAME background
    session. Resolving the MCP round first must not clear the badge (nor
    the skill-install round's own still-armed retained payload) while
    the skill-install round is still pending; only resolving BOTH clears
    it."""
    controller, store = _build_controller()
    viewed = store.create_session(title="Viewed").id
    background = store.create_session(title="Background").id
    store.switch_session(viewed)  # keep viewing the first session
    controller.app = _FakeApp()
    controller.set_pending_approval = lambda payload: None
    controller.set_pending_skill_install = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0
    controller.skill_install_confirm_timeout_seconds = lambda: 30.0

    mcp_result: dict[str, dict[str, str]] = {}
    install_result: dict[str, bool] = {}

    def _run_mcp() -> None:
        mcp_result["decisions"] = controller.request_mcp_approvals(
            [_pending()], session_id=background
        )

    def _run_install() -> None:
        install_result["allowed"] = controller.request_skill_install_confirm(
            "https://x/y", session_id=background
        )

    mcp_worker = threading.Thread(target=_run_mcp)
    mcp_worker.start()
    time.sleep(0.1)
    assert background in controller._pending_approvals
    assert controller.run_marker_for(background) is ConsoleRunMarker.NEEDS_APPROVAL
    mcp_round_id = controller._head_round_payload(
        controller._parked_approval_payloads, background
    )["round_id"]
    # (Minor, review) Data-level check: after just the MCP round has
    # parked, the round-keyed set for this session is EXACTLY the
    # singleton of its own round id -- locks in that arming never
    # double-counts (e.g. via both `add_pending_round` and a stray
    # legacy-shim call landing on the same session).
    assert controller._pending_approvals[background] == {mcp_round_id}

    install_worker = threading.Thread(target=_run_install)
    install_worker.start()
    time.sleep(0.1)
    install_request_id = controller._head_round_payload(
        controller._parked_skill_install_payloads, background
    )["request_id"]
    # Both rounds' ids are now tracked, still with no double-count.
    assert controller._pending_approvals[background] == {
        mcp_round_id,
        install_request_id,
    }

    # The MCP round resolves FIRST -- the skill-install round is still
    # outstanding for the same session, so the badge must stay up and the
    # install round's own retained payload must survive untouched.
    controller.resolve_pending_approval(
        {"mcp__srv__tool": "deny"}, round_id=mcp_round_id
    )
    mcp_worker.join(timeout=2.0)
    assert mcp_result["decisions"] == {"mcp__srv__tool": "deny"}
    assert controller.run_marker_for(background) is ConsoleRunMarker.NEEDS_APPROVAL
    assert background in controller._pending_approvals
    assert (
        controller._head_round_payload(
            controller._parked_skill_install_payloads, background
        )["request_id"]
        == install_request_id
    )
    # The MCP bridge's OWN payload map is cleared (it was the last MCP
    # round for this session).
    assert (
        controller._head_round_payload(controller._parked_approval_payloads, background)
        is None
    )

    # The skill-install round resolves LAST -- only now does the badge
    # fully clear.
    controller.resolve_pending_skill_install(True, request_id=install_request_id)
    install_worker.join(timeout=2.0)
    assert install_result["allowed"] is True
    assert controller.run_marker_for(background) is ConsoleRunMarker.NONE
    assert background not in controller._pending_approvals
    assert (
        controller._head_round_payload(
            controller._parked_skill_install_payloads, background
        )
        is None
    )


def test_request_mcp_approvals_other_sessions_cancel_event_does_not_deny_this_round():
    """PA-T9 finding #1: pre-Task-9, `request_mcp_approvals`'s cancel check
    fell back to the VIEWED session's cancel event regardless of which
    session's round was actually waiting -- so ANY session's Stop could
    spuriously deny a DIFFERENT session's in-flight approval batch. With
    `session_id` threaded through, session A's cancel event -- already set
    BEFORE this round even starts -- must never deny session B's round;
    only B's own cancel event (or a genuine deadline) may resolve it."""
    controller, store = _build_controller()
    session_a = store.create_session(title="A").id
    session_b = store.create_session(title="B").id
    # ADR-067: an app-less controller now fails closed before arming (see
    # `request_mcp_approvals`' no-app guard), so wire the usual fake
    # bridge -- this test's subject is cross-session cancel isolation, not
    # the no-app path.
    controller.app = _FakeApp()
    controller.set_pending_approval = lambda payload: None
    # A short deadline: if A's cancel event wrongly denied this round, the
    # cancellation branch would fire first (`_record_cancelled_approval_
    # decisions` aside) -- observing "timeout" instead of "deny" proves the
    # cancellation branch never triggered.
    controller.mcp_approval_timeout_seconds = lambda: 0.05

    a_cancel_event = threading.Event()
    a_cancel_event.set()
    controller._active_cancel_events[session_a] = a_cancel_event

    decisions = controller.request_mcp_approvals([_pending()], session_id=session_b)

    assert decisions == {"mcp__srv__tool": "timeout"}


def test_request_mcp_approvals_own_session_cancel_event_denies_the_round():
    """PA-T9 finding #1, positive case: session B's OWN cancel event (as
    `stop_active_run`/`close_session` would set via `_signal_stop` if B
    were the viewed/closing session) still correctly denies B's round when
    `session_id=B` is threaded through."""
    controller, store = _build_controller()
    session_b = store.ensure_session(title="B").id
    controller.mcp_approval_timeout_seconds = lambda: 30.0
    cancel_event = threading.Event()
    controller._active_cancel_events[session_b] = cancel_event

    def _cancel_soon() -> None:
        time.sleep(0.05)
        cancel_event.set()

    threading.Thread(target=_cancel_soon).start()
    decisions = controller.request_mcp_approvals([_pending()], session_id=session_b)

    assert decisions == {"mcp__srv__tool": "deny"}


def test_resolve_pending_approval_by_round_id_survives_a_mid_flight_session_switch():
    """CRITICAL (review round 1): `ApprovalDecided` travels as an async
    Textual message -- a `switch_session` landing in the gap between the
    user's click and the handler running must NOT let session A's decision
    resolve session B's completely different, unreviewed batch.
    `resolve_pending_approval` now resolves by the `round_id` stamped onto
    the card at mount time (exactly what `ChatApprovalCard.set_batch`/
    `ApprovalDecided` round-trip), never by "whichever session is active
    right now"."""
    controller, store = _build_controller()
    session_a = store.create_session(title="A").id
    session_b = store.create_session(title="B").id
    store.switch_session(session_a)
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    result_a: dict[str, dict[str, str]] = {}

    def _run_round_a() -> None:
        result_a["decisions"] = controller.request_mcp_approvals(
            [_pending(llm_name="mcp__a__tool")], session_id=session_a
        )

    worker_a = threading.Thread(target=_run_round_a)
    worker_a.start()
    time.sleep(0.1)
    # A is active at round-start, so it mounted immediately -- capture the
    # round_id the real ChatApprovalCard would have stashed via set_batch.
    assert mounted and mounted[-1] is not None
    round_id_a = mounted[-1]["round_id"]

    # Session B gets its own, completely independent pending round (parked,
    # since B isn't active).
    result_b: dict[str, dict[str, str]] = {}

    def _run_round_b() -> None:
        result_b["decisions"] = controller.request_mcp_approvals(
            [_pending(llm_name="mcp__b__tool")], session_id=session_b
        )

    worker_b = threading.Thread(target=_run_round_b)
    worker_b.start()
    time.sleep(0.1)
    round_id_b = controller._head_round_payload(
        controller._parked_approval_payloads, session_b
    )["round_id"]
    assert round_id_b != round_id_a

    # The user clicked "Submit" on A's card, but before the resulting
    # ApprovalDecided message is handled, a switch_session moved the
    # ACTIVE session to B -- exactly the async-message race under review.
    store.switch_session(session_b)

    controller.resolve_pending_approval(
        {"mcp__a__tool": "approve_once"}, round_id=round_id_a
    )
    worker_a.join(timeout=2.0)

    assert result_a["decisions"] == {"mcp__a__tool": "approve_once"}
    assert "decisions" not in result_b  # B's round is completely untouched

    # Clean up B's still-waiting round rather than leaving a live thread
    # blocked for the rest of its 30s timeout.
    controller.resolve_pending_approval({"mcp__b__tool": "deny"}, round_id=round_id_b)
    worker_b.join(timeout=2.0)
    assert result_b["decisions"] == {"mcp__b__tool": "deny"}


def _other_round_id(controller, session_id: str, not_this_one: str) -> str:
    """The session's OTHER retained round id (PR0: one key per round)."""
    return [
        payload["round_id"]
        for payload in controller._session_round_payloads(
            controller._parked_approval_payloads, session_id
        )
        if payload["round_id"] != not_this_one
    ][0]


def test_two_mcp_rounds_for_the_same_session_the_earlier_ones_teardown_does_not_evict_the_newer_ones_payload():
    """One round's teardown must never discard a sibling's retained payload.

    TASK-1050 (Defect B) originally: `_parked_approval_payloads` was keyed
    by session id ALONE, so arming a SECOND round for the SAME session
    overwrote the first's payload and an unconditional pop in either
    teardown discarded the survivor's only copy -- a switch-away/back
    remounted `None` and the survivor sat unresolvable until its timeout.
    That was patched with an order-dependent "only pop when this is the
    LAST armed round" guard.

    PR0 (task-15661) removes the shared slot entirely: the map is keyed by
    ROUND, so each teardown drops exactly its own key and the guard is
    gone. The CONTRACT this test pins is unchanged and now holds by
    construction rather than by a guard -- the earlier round resolving
    first leaves the later round's payload intact, and the session's card
    re-derives to it."""
    controller, store = _build_controller()
    session_a = store.create_session(title="A").id
    store.switch_session(session_a)
    controller.app = _FakeApp()
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    result_1: dict[str, dict[str, str]] = {}
    result_2: dict[str, dict[str, str]] = {}

    def _run_round_1() -> None:
        result_1["decisions"] = controller.request_mcp_approvals(
            [_pending(llm_name="mcp__one__tool")], session_id=session_a
        )

    worker_1 = threading.Thread(target=_run_round_1)
    worker_1.start()
    time.sleep(0.1)
    round_id_1 = controller._head_round_payload(
        controller._parked_approval_payloads, session_a
    )["round_id"]
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL

    def _run_round_2() -> None:
        result_2["decisions"] = controller.request_mcp_approvals(
            [_pending(llm_name="mcp__two__tool")], session_id=session_a
        )

    worker_2 = threading.Thread(target=_run_round_2)
    worker_2.start()
    time.sleep(0.1)
    # PR0: round 2 keeps its OWN key rather than overwriting round 1's --
    # round 1 is still the session's head and still owns the card.
    round_id_2 = _other_round_id(controller, session_a, round_id_1)
    assert round_id_2 != round_id_1
    assert (
        controller._head_round_payload(controller._parked_approval_payloads, session_a)[
            "round_id"
        ]
        == round_id_1
    ), "arming round 2 must not evict round 1's card"

    # Round 1 (the EARLIER, now-superseded round) resolves first -- its
    # teardown must not discard round 2's still-armed, newer payload, nor
    # clear the badge.
    controller.resolve_pending_approval({"mcp__one__tool": "deny"}, round_id=round_id_1)
    worker_1.join(timeout=2.0)
    assert result_1["decisions"] == {"mcp__one__tool": "deny"}
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL
    assert session_a in controller._pending_approvals
    assert (
        controller._head_round_payload(controller._parked_approval_payloads, session_a)[
            "round_id"
        ]
        == round_id_2
    ), "round 1 resolving must promote round 2, not discard it"

    # Round 2 (the LAST remaining round) resolves -- now everything clears.
    controller.resolve_pending_approval(
        {"mcp__two__tool": "approve_once"}, round_id=round_id_2
    )
    worker_2.join(timeout=2.0)
    assert result_2["decisions"] == {"mcp__two__tool": "approve_once"}
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE
    assert session_a not in controller._pending_approvals
    assert (
        controller._head_round_payload(controller._parked_approval_payloads, session_a)
        is None
    )


def test_two_mcp_rounds_for_the_same_session_resolving_the_newer_one_first_leaves_the_slot_populated():
    """Reverse ordering: the newer round resolving first must not strand
    the older one card-less with the badge still lit.

    This is the NATURAL live ordering, not an edge case -- pre-PR0, arming
    a round re-mounted its card, so the newest round was typically decided
    before an already-waiting sibling. Two successive TASK-1050 fix rounds
    were needed to stop the newer round's teardown from popping the SHARED
    per-session payload slot (fix round 1) and then from firing the
    card-clear seam (fix round 3) while the older round was still armed.

    PR0 (task-15661) makes both hazards structural non-events: each round
    owns its own key, and the card is always the session's FIFO HEAD.
    Round 1 is therefore the round that MOUNTS (round 2 queues silently
    behind it), and round 2 resolving re-derives the head -- which is
    still round 1's own payload, so the card the user is looking at does
    not change and is certainly never cleared. That last point is what the
    fix-round-3 assertion below now checks: it asserts on the card's
    CONTENT rather than on whether the seam was called, because under
    FIFO the seam legitimately fires (re-deriving to the same head) where
    pre-PR0 it had to stay silent. Recording every call through
    `mounted.append` (rather than a discarding no-op) is what makes that
    distinction observable at all."""
    controller, store = _build_controller()
    session_a = store.create_session(title="A").id
    store.switch_session(session_a)
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    result_1: dict[str, dict[str, str]] = {}
    result_2: dict[str, dict[str, str]] = {}

    def _run_round_1() -> None:
        result_1["decisions"] = controller.request_mcp_approvals(
            [_pending(llm_name="mcp__one__tool")], session_id=session_a
        )

    worker_1 = threading.Thread(target=_run_round_1)
    worker_1.start()
    time.sleep(0.1)
    assert mounted and mounted[-1] is not None
    round_id_1 = mounted[-1]["round_id"]

    def _run_round_2() -> None:
        result_2["decisions"] = controller.request_mcp_approvals(
            [_pending(llm_name="mcp__two__tool")], session_id=session_a
        )

    worker_2 = threading.Thread(target=_run_round_2)
    worker_2.start()
    time.sleep(0.1)
    # PR0: round 2 does NOT mount -- round 1 is the head and keeps the card.
    assert mounted[-1] is not None
    assert mounted[-1]["round_id"] == round_id_1, (
        "arming round 2 must not evict the head round's card"
    )
    round_id_2 = _other_round_id(controller, session_a, round_id_1)
    assert round_id_2 != round_id_1

    # Round 2 (the NEWER, queued round) resolves FIRST -- round 1 is still
    # outstanding, so the badge must stay up, round 1 must keep its own
    # retained payload, and the card the user is looking at must still be
    # round 1's own (PR0: the head re-derive resolves back to it).
    controller.resolve_pending_approval(
        {"mcp__two__tool": "approve_once"}, round_id=round_id_2
    )
    worker_2.join(timeout=2.0)
    assert result_2["decisions"] == {"mcp__two__tool": "approve_once"}
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL
    assert session_a in controller._pending_approvals
    assert round_id_1 in controller._parked_approval_payloads, (
        "round 1 must keep its own retained payload -- dropping it here "
        "would strand the still-armed older round unresolvable on the "
        "next switch-away/back"
    )
    assert mounted[-1] is not None and mounted[-1]["round_id"] == round_id_1, (
        "round 2 resolving must leave round 1's card on screen -- clearing "
        "it (or swapping in anything else) strands round 1 card-less with "
        f"the badge still lit; got {mounted[-1]}"
    )

    # Round 1 (the OLDER round) remains fully decidable through the UI
    # the whole time round 2 was resolving -- resolving it now by its OWN
    # `round_id` must still work correctly.
    controller.resolve_pending_approval({"mcp__one__tool": "deny"}, round_id=round_id_1)
    worker_1.join(timeout=2.0)
    assert result_1["decisions"] == {"mcp__one__tool": "deny"}
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE
    assert session_a not in controller._pending_approvals
    assert (
        controller._head_round_payload(controller._parked_approval_payloads, session_a)
        is None
    )
    # Round 1 (now the LAST remaining round) resolving DOES clear the card.
    assert mounted[-1] is None


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _DeferredClearApp:
    """`call_from_thread` stand-in that BLOCKS the round-identity-guarded
    clear closures until a test explicitly releases them, while every
    OTHER `call_from_thread` use (mount, park) still runs immediately.

    The clear closures built by `_clear_pending_approval_if_round_is_
    current` (and its skill-install/skill-script mirrors) are always
    invoked with zero positional/keyword args -- every other
    `call_from_thread` call in these bridges carries a positional
    payload/session_id -- so that shape is what identifies "this is a
    teardown clear" without needing any bridge-specific hook.
    """

    def __init__(self) -> None:
        self.clear_enqueued = threading.Event()
        self.release_clear = threading.Event()

    def call_from_thread(self, fn, *args, **kwargs):
        if not args and not kwargs:
            self.clear_enqueued.set()
            self.release_clear.wait(timeout=5)
            return fn()
        return fn(*args, **kwargs)


def test_teardown_clear_does_not_clobber_a_newer_same_session_round_arming_mid_teardown():
    """TASK-1050 fix round 2 (review, Qodo PR #1041, CRITICAL): `request_
    mcp_approvals`'s teardown used to decide whether to clear the mounted
    card via a boolean snapshot (`still_active`/`still_armed_same_
    session`) computed BEFORE the clear was enqueued to the UI thread via
    `call_from_thread`. A race window existed between that snapshot and
    the UI thread actually running the clear: a NEWER same-session round
    could arm -- and mount its own card -- in that window, and then get
    wiped by round 1's now-stale clear, stranding it until a manual
    remount or its own timeout. This test drives that EXACT interleaving
    deterministically via an event/gate-controlled fake app (never a
    sleep-timed guess): round 1 resolves and its teardown's clear call
    BLOCKS mid-flight (right where the race window used to be); round 2
    arms and mounts for the SAME session while round 1's clear is still
    blocked; only then is round 1's clear released to actually run.
    PR0 (task-15661) replaced that round-identity guard with a FIFO-head
    re-derive (`_remount_head`), which closes the same race by the same
    principle -- the decision is computed INSIDE the callable, on the UI
    thread, never from a worker-thread snapshot -- and is order-
    independent besides. Released here, round 1's deferred callable must
    re-derive to round 2 (the session's head by then) and leave round 2's
    freshly-mounted card intact."""
    controller, store = _build_controller()
    session_a = store.create_session(title="A").id
    store.switch_session(session_a)
    app = _DeferredClearApp()
    controller.app = app
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    result_1: dict[str, dict[str, str]] = {}

    def _run_round_1() -> None:
        result_1["decisions"] = controller.request_mcp_approvals(
            [_pending(llm_name="mcp__one__tool")], session_id=session_a
        )

    worker_1 = threading.Thread(target=_run_round_1)
    worker_1.start()
    assert _wait_until(lambda: len(mounted) == 1), "round 1 never mounted"
    assert mounted[-1] is not None
    round_id_1 = mounted[-1]["round_id"]

    # Resolve round 1 -- its teardown runs its accounting cleanup (badge
    # discard, own payload-map pop) synchronously, then reaches its clear
    # call and BLOCKS there, before the clear itself ever runs.
    controller.resolve_pending_approval({"mcp__one__tool": "deny"}, round_id=round_id_1)
    assert app.clear_enqueued.wait(timeout=5), (
        "round 1's teardown never reached its clear call"
    )
    # Round 1's own accounting is already fully torn down at this point --
    # entirely independent of whether its still-blocked UI-thread clear
    # has run yet.
    assert session_a not in controller._pending_approvals
    assert (
        controller._head_round_payload(controller._parked_approval_payloads, session_a)
        is None
    )

    # Round 2 arms for the SAME session WHILE round 1's clear is still
    # blocked -- it mounts its own card immediately (session_a is active).
    result_2: dict[str, dict[str, str]] = {}

    def _run_round_2() -> None:
        result_2["decisions"] = controller.request_mcp_approvals(
            [_pending(llm_name="mcp__two__tool")], session_id=session_a
        )

    worker_2 = threading.Thread(target=_run_round_2)
    worker_2.start()
    assert _wait_until(lambda: len(mounted) == 2), "round 2 never mounted"
    assert mounted[-1] is not None
    round_id_2 = mounted[-1]["round_id"]
    assert round_id_2 != round_id_1
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL

    # NOW release round 1's blocked clear. A snapshot-guarded clear would
    # unconditionally wipe round 2's just-mounted card here; `_remount_head`'s
    # FIFO head re-derive must instead see round 2 is now the session's head
    # and leave its card mounted.
    app.release_clear.set()
    worker_1.join(timeout=2.0)
    assert result_1["decisions"] == {"mcp__one__tool": "deny"}

    # Round 2's card survived round 1's (now-run) clear, and round 1's
    # teardown otherwise completed normally (accounting stays clean --
    # the badge is up only because of round 2, not a leaked round 1
    # entry).
    assert mounted[-1] is not None
    assert mounted[-1]["round_id"] == round_id_2
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL
    assert controller._pending_approvals[session_a] == {round_id_2}

    # Resolving round 2 normally still clears everything -- proves round
    # 1's no-op clear didn't leave the bridge in some broken state.
    controller.resolve_pending_approval(
        {"mcp__two__tool": "approve_once"}, round_id=round_id_2
    )
    worker_2.join(timeout=2.0)
    assert result_2["decisions"] == {"mcp__two__tool": "approve_once"}
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE
    assert session_a not in controller._pending_approvals
    assert (
        controller._head_round_payload(controller._parked_approval_payloads, session_a)
        is None
    )
    assert mounted[-1] is None


def test_resolve_pending_approval_ignores_a_stale_or_unknown_round_id():
    """A `round_id` that doesn't match any currently-armed round (a
    fabricated/unknown id, or one whose round already resolved and was
    popped) is a safe no-op -- the request just returns, no round is
    touched, nothing raises."""
    controller, _ = _build_controller()
    controller.resolve_pending_approval(
        {"mcp__srv__tool": "deny"}, round_id="not-a-real-round"
    )  # must not raise


def test_resolve_pending_approval_stale_round_id_never_resolves_a_newer_round_for_the_same_session():
    """Mirrors `resolve_pending_skill_script`'s identical defended scenario:
    round 1 for session A times out (its round_id is popped), round 2 arms
    for the SAME session immediately after -- a late decision carrying
    round 1's now-stale id must never resolve round 2."""
    controller, store = _build_controller()
    session_a = store.create_session(title="A").id
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append
    controller.mcp_approval_timeout_seconds = lambda: 0.05

    round_1_decisions = controller.request_mcp_approvals(
        [_pending(llm_name="mcp__srv__tool")], session_id=session_a
    )
    assert round_1_decisions == {"mcp__srv__tool": "timeout"}
    stale_round_id = mounted[0]["round_id"]

    controller.mcp_approval_timeout_seconds = lambda: 30.0
    result_2: dict[str, dict[str, str]] = {}

    def _run_round_2() -> None:
        result_2["decisions"] = controller.request_mcp_approvals(
            [_pending(llm_name="mcp__srv__tool")], session_id=session_a
        )

    worker = threading.Thread(target=_run_round_2)
    worker.start()
    time.sleep(0.1)
    round_2_id = mounted[-1]["round_id"]
    assert round_2_id != stale_round_id

    # The stale decision (round 1's id) must not touch round 2.
    controller.resolve_pending_approval(
        {"mcp__srv__tool": "deny"}, round_id=stale_round_id
    )
    time.sleep(0.1)
    assert "decisions" not in result_2

    # Round 2 still resolves normally via its OWN id.
    controller.resolve_pending_approval(
        {"mcp__srv__tool": "approve_once"}, round_id=round_2_id
    )
    worker.join(timeout=2.0)
    assert result_2["decisions"] == {"mcp__srv__tool": "approve_once"}


def test_resolve_pending_approval_without_active_round_is_a_noop():
    controller, _ = _build_controller()
    controller.resolve_pending_approval({"mcp__srv__tool": "deny"})  # must not raise


def test_resolve_pending_approval_without_round_id_fails_closed_and_leaves_round_pending():
    """TASK-913 AC#2: `round_id=None` no longer falls back to "whichever
    round belongs to the active session" -- it fails closed immediately,
    mirroring `resolve_pending_skill_script`'s/
    `resolve_pending_skill_install`'s identical `if request_id is None:
    return` contract. Even with a round genuinely armed for the (only)
    active session, a resolve carrying no round_id must be a pure no-op:
    the round stays pending, undecided, and its waiting worker thread
    stays blocked -- exactly like an unknown/stale round_id already does
    (see test_resolve_pending_approval_ignores_a_stale_or_unknown_round_id).
    Pre-fix, this used to resolve via the active-session fallback."""
    controller, _ = _build_controller()
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    result_holder: dict[str, dict[str, str]] = {}

    def _run_round() -> None:
        result_holder["decisions"] = controller.request_mcp_approvals([_pending()])

    worker = threading.Thread(target=_run_round)
    worker.start()
    time.sleep(0.1)
    assert mounted and mounted[-1] is not None
    round_id = mounted[-1]["round_id"]

    # A None round_id (the default) must NOT resolve the armed round.
    controller.resolve_pending_approval({"mcp__srv__tool": "deny"})
    time.sleep(0.1)
    assert "decisions" not in result_holder  # still pending, undecided

    # Clean up via the real round_id so the worker thread actually ends.
    controller.resolve_pending_approval(
        {"mcp__srv__tool": "approve_once"}, round_id=round_id
    )
    worker.join(timeout=2.0)
    assert result_holder["decisions"] == {"mcp__srv__tool": "approve_once"}


def test_request_mcp_approvals_with_no_pending_calls_returns_empty_and_never_surfaces_card():
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append

    assert controller.request_mcp_approvals([]) == {}
    assert received == []


def test_request_mcp_approvals_snapshot_covers_exactly_the_unique_names():
    """F4 (Gemini): the final snapshot is built by keyed lookup over
    `unique_names` rather than `dict(decisions)` -- must still return
    exactly one entry per unique llm_name in `pending`, matching whatever
    `resolve_pending_approval` supplied (extra keys are dropped, missing
    ones fail closed to "deny", both preserved by the setdefault pass
    before the snapshot)."""
    controller, _ = _build_controller()
    controller.app = _FakeApp()
    received: list[dict | None] = []
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    pending = [
        _pending(llm_name="mcp__srv__a", tool_name="a"),
        _pending(llm_name="mcp__srv__b", tool_name="b"),
    ]

    def _resolve_soon() -> None:
        time.sleep(0.05)
        assert received and received[-1] is not None
        # Only decides "a" explicitly, includes an unrelated stray key,
        # and leaves "b" undecided (backstopped to "deny" pre-snapshot).
        controller.resolve_pending_approval(
            {
                "mcp__srv__a": "approve_once",
                "mcp__unrelated__c": "deny",
            },
            round_id=received[-1]["round_id"],
        )

    threading.Thread(target=_resolve_soon).start()
    decisions = controller.request_mcp_approvals(pending)

    assert decisions == {"mcp__srv__a": "approve_once", "mcp__srv__b": "deny"}


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


def test_chat_screen_forwards_approval_decided_to_controller(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    controller = Mock()
    screen._console_chat_controller = controller

    # Task 9 fix round 1: the event's round_id must forward too --
    # resolve_pending_approval resolves BY that id, never by "whichever
    # session is active".
    event = ChatApprovalCard.ApprovalDecided(
        {"mcp__a__b": "deny"}, round_id="round-123"
    )
    screen.handle_console_approval_decided(event)

    controller.resolve_pending_approval.assert_called_once_with(
        {"mcp__a__b": "deny"}, round_id="round-123"
    )


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
        def __init__(self) -> None:
            self._call_count = 0

        def call_from_thread(self, fn, *args, **kwargs):
            self._call_count += 1
            if self._call_count == 1:
                # The initial mount call -- succeeds normally, capturing
                # the payload for the resolver thread to read the
                # round_id off of.
                calls.append(args[0] if args else None)
                return fn(*args, **kwargs)
            # TASK-1050 fix round 2 / PR0: the teardown-time card update
            # is a zero-arg wrapper closure (`_remount_head`'s `_apply`)
            # rather than a direct `call_from_thread(setter, None)` call,
            # so it can no longer be identified by inspecting
            # `args[0] is None`.
            # Simulate the app having stopped by the time this (second)
            # `call_from_thread` invocation happens, regardless of what
            # it was called with.
            raise RuntimeError("App is not running")

    controller.app = _TeardownApp()
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    def _resolve_soon() -> None:
        time.sleep(0.05)
        assert calls and calls[-1] is not None
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "approve_once"}, round_id=calls[-1]["round_id"]
        )

    resolver = threading.Thread(target=_resolve_soon)
    resolver.start()
    try:
        decisions = controller.request_mcp_approvals([_pending()])
    finally:
        resolver.join()

    assert decisions == {"mcp__srv__tool": "approve_once"}


@pytest.mark.unit
def test_call_id_keyed_decision_still_stamps_the_builtin_gate():
    """Per-call verdicts must not starve the NAME-keyed consumers.

    The approval card now keys verdicts by `call_id` so two reads of two
    files are two decisions. But `builtin_gate.stamp` records a grant against
    a tool NAME (a session/always grant is per tool, not per call), and
    `MCPToolProvider.apply_batch_decisions` also takes names. Without an
    explicit per-call-then-name resolution, a call-id-keyed decision reached
    NEITHER: MCP got {} and no grant was ever stamped, silently.

    This test exists because the whole approval suite passed with that break
    in place -- nothing covered the key mismatch.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    stamped: list[tuple[str, str]] = []

    class _Gate:
        def begin_turn(self, run_id):
            pass

        def resolve(self, tool):
            return SimpleNamespace(state="ask", risk_floored=False)

        def stamp(self, run_id, name, decision):
            stamped.append((name, decision))

        def is_session_approved(self, name):
            return False

        def options_for(self, tool):
            return ("approve_once", "approve_session", "deny")

    class _Provider:
        def tool_for(self, name):
            return SimpleNamespace(name=name)

    # The card answers with a CALL-ID key, exactly as it now does.
    def request_approvals(pending):
        assert pending, "precondition: a row was surfaced for review"
        return {pending[0].call_id: "approve_session"}

    hook = build_tool_review_hook(
        _Gate(), _Provider(), None, request_approvals, workspace_id=None
    )
    hook([ToolCall(name="read_file", args={"path": "spec.md"}, call_id="call-1")], RUN)

    assert stamped == [("read_file", "approve_session")], (
        f"the call-id-keyed decision never reached the gate: {stamped}"
    )


@pytest.mark.asyncio
async def test_collapsed_row_discloses_every_target_in_the_rendered_row():
    """TASK-1845 regression: the "xN" row must show every target ON SCREEN.

    The original fix aggregated `all_arguments` in `_collapse_pending_calls`
    and taught `_summarize_arguments` to render them -- but `set_batch` kept
    passing `entry["arguments"]` (the FIRST call's payload), so the branch
    never ran in production and the row still showed one target out of three.
    The test that "covered" it called the helper with a collapsed entry, a
    shape production never builds.

    This drives the real widget: three reads of three different files must
    all be legible in the mounted row, or the user approves what they cannot
    see.
    """
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            [
                {"llm_name": "read_file", "arguments": {"path": "~/notes/spec.md"}},
                {"llm_name": "read_file", "arguments": {"path": "~/notes/secrets.md"}},
                {"llm_name": "read_file", "arguments": {"path": "~/notes/todo.md"}},
            ],
            timeout_seconds=45.0,
        )
        await pilot.pause()

        rows = list(app.query(".approval-row-args"))
        assert len(rows) == 1, "same-name calls collapse to one row by contract"
        rendered = _text(rows[0])
        for path in ("spec.md", "secrets.md", "todo.md"):
            assert path in rendered, (
                f"{path} is hidden behind the x3 -- the mounted row shows: {rendered!r}"
            )


@pytest.mark.asyncio
async def test_armed_deadline_is_visible_on_the_mounted_card():
    """TASK-1844: the countdown must reach the SCREEN, not just a helper.

    `set_batch` updates `#approval-deadline` inside `except NoMatches: pass`,
    so if that Static ever stopped being composed the clock would silently
    vanish while `format_approval_deadline`'s unit tests stayed green. The
    controller arms a 120s auto-deny; a deadline the user cannot see is the
    machine deciding for them.
    """
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(_sample_calls(), timeout_seconds=120.0)
        await pilot.pause()

        deadline = app.query_one("#approval-deadline", Static)
        assert _text(deadline) == "Auto-denies in 2:00"
        assert deadline.display, "the countdown is composed but not displayed"

        # No deadline armed -> say nothing rather than invent a number.
        card.set_batch(_sample_calls(), timeout_seconds=0)
        await pilot.pause()
        assert not app.query_one("#approval-deadline", Static).display


@pytest.mark.unit
def test_refusing_one_call_does_not_get_overwritten_by_approving_another():
    """A per-call REFUSAL must reach the runtime, not be flattened away.

    TASK-1861. Verdicts are keyed per `call_id` so the user can allow
    `spec.md` and refuse `secrets.md` in one batch -- but both enforcement
    consumers are name-keyed (`builtin_gate.stamp` records a grant against a
    tool NAME; `apply_batch_decisions` takes llm_names), and the hook
    returned a flat `{name: "proceed"}`. So two rows of one tool disagreeing
    resolved LAST-WRITE-WINS on a single name:

        stamped: [("read_file", "deny"), ("read_file", "approve_once")]

    Refuse `secrets.md` first, approve `spec.md` second, and the surviving
    stamp is `approve_once` -- the file the user explicitly refused is read.
    This fails OPEN, which is why it is pinned rather than left to the
    round-trip tests: the card offers a decision the pipeline could not
    honour.

    The fix enforces refusals PER CALL at the runtime hook (the runtime
    resolves `call_id` before name, and a non-"proceed" verdict string
    becomes the call's result without dispatch), leaving the name-keyed
    stamps to carry only what was APPROVED.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    stamped: list[tuple[str, str]] = []

    class _Gate:
        def begin_turn(self, run_id):
            pass

        def resolve(self, tool):
            return SimpleNamespace(state="ask", risk_floored=False)

        def stamp(self, run_id, name, decision):
            stamped.append((name, decision))

        def is_session_approved(self, name):
            return False

        def options_for(self, tool):
            return ("approve_once", "approve_session", "deny")

    class _Provider:
        def tool_for(self, name):
            return SimpleNamespace(name=name)

    def request_approvals(pending):
        by_path = {row.call_id: (row.arguments or {}).get("path") for row in pending}
        # Refuse secrets.md; allow spec.md. Refusal FIRST is the fail-open
        # ordering -- the later approval used to overwrite it.
        return {
            call_id: ("deny" if path == "secrets.md" else "approve_once")
            for call_id, path in by_path.items()
        }

    hook = build_tool_review_hook(
        _Gate(), _Provider(), None, request_approvals, workspace_id=None
    )
    verdicts = hook(
        [
            ToolCall(name="read_file", args={"path": "secrets.md"}, call_id="call-1"),
            ToolCall(name="read_file", args={"path": "spec.md"}, call_id="call-2"),
        ],
        RUN,
    )

    refusal = verdicts.get("call-1")
    assert refusal and refusal != "proceed", (
        "the refusal of secrets.md never reached the runtime, so the "
        f"name-keyed approval of spec.md let it through: {verdicts}"
    )
    assert verdicts.get("call-2", "proceed") == "proceed", (
        f"approving spec.md must still let it run: {verdicts}"
    )
    assert ("read_file", "deny") not in stamped, (
        "a refusal must not be stamped against the NAME -- that would also "
        f"stop the call the user approved: {stamped}"
    )


@pytest.mark.unit
def test_mcp_rows_carry_their_call_id_so_two_targets_are_two_decisions():
    """TASK-1861: MCP rows dropped the call id, so `xN` still hid targets.

    `_collect_mcp_pending` walks the batch and calls
    `provider.pending_gate_for(call.name, call.args)` -- the call's
    `call_id` was discarded at that boundary, so every MCP pending row
    carried `call_id=""`. The card then collapsed all same-name MCP calls
    into ONE `xN` row with one verdict, which is the exact defect the
    per-call re-key fixed for built-in tools and left standing for MCP.
    """
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import _collect_mcp_pending

    class _Provider:
        def pending_gate_for(self, llm_name, args, call_id="", rationale=""):
            return MCPPendingCall(
                llm_name=llm_name,
                server_key="local:fs",
                tool_name="read_file",
                server_label="FS",
                arguments=dict(args or {}),
                call_id=call_id,
                rationale=rationale,
                reason="ask",
            )

    rows = _collect_mcp_pending(
        _Provider(),
        [
            ToolCall(name="mcp__fs__read", args={"path": "spec.md"}, call_id="c1"),
            ToolCall(name="mcp__fs__read", args={"path": "secrets.md"}, call_id="c2"),
        ],
    )
    assert [r.call_id for r in rows] == ["c1", "c2"], (
        f"the call ids never reached the MCP rows: {[r.call_id for r in rows]}"
    )


@pytest.mark.unit
def test_a_refusal_never_stamps_the_name_even_when_it_is_decided_last():
    """TASK-1861, the other ordering -- and the one that fails CLOSED.

    Approve `spec.md`, then refuse `secrets.md`. Both rows share the tool
    NAME, and the stamp is what `invoke()` peeks at, so stamping the refusal
    would block the call the user just approved.

    This case is separate because the sibling test cannot catch it: there
    the refusal is decided FIRST, so a bug that stamps refusals is masked by
    the later approval overwriting it. Mutation-checked -- stamping every
    decision regardless of verdict passes that test and fails this one.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    stamped: list[tuple[str, str]] = []

    class _Gate:
        def begin_turn(self, run_id):
            pass

        def resolve(self, tool):
            return SimpleNamespace(state="ask", risk_floored=False)

        def stamp(self, run_id, name, decision):
            stamped.append((name, decision))

        def is_session_approved(self, name):
            return False

        def options_for(self, tool):
            return ("approve_once", "approve_session", "deny")

    class _Provider:
        def tool_for(self, name):
            return SimpleNamespace(name=name)

    def request_approvals(pending):
        return {
            row.call_id: (
                "deny"
                if (row.arguments or {}).get("path") == "secrets.md"
                else "approve_session"
            )
            for row in pending
        }

    hook = build_tool_review_hook(
        _Gate(), _Provider(), None, request_approvals, workspace_id=None
    )
    verdicts = hook(
        [
            ToolCall(name="read_file", args={"path": "spec.md"}, call_id="c-ok"),
            ToolCall(name="read_file", args={"path": "secrets.md"}, call_id="c-no"),
        ],
        RUN,
    )

    assert stamped == [("read_file", "approve_session")], (
        "the refusal was stamped against the tool NAME, which also blocks "
        f"the call the user approved: {stamped}"
    )
    assert verdicts.get("c-no", "proceed") != "proceed", (
        f"secrets.md must still be refused per call: {verdicts}"
    )


@pytest.mark.unit
def test_the_broadest_approval_scope_for_a_tool_survives_collapsing():
    """TASK-1861: per-call rows can disagree on SCOPE, not just allow/refuse.

    A session or always grant belongs to the TOOL, so the stamp can hold only
    one scope per name. Collapsing them last-write-wins silently downgraded
    "Approve for session" to "approve once" whenever a later row of the same
    tool was approved once -- dropping the grant the user explicitly asked
    for and re-prompting for it on the next call.

    The user picking "for session" on any call of a tool IS choosing to grant
    that tool for the session (that is what the control means, and the label
    says so), so the broadest chosen scope wins.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    stamped: list[tuple[str, str]] = []

    class _Gate:
        def begin_turn(self, run_id):
            pass

        def resolve(self, tool):
            return SimpleNamespace(state="ask", risk_floored=False)

        def stamp(self, run_id, name, decision):
            stamped.append((name, decision))

        def is_session_approved(self, name):
            return False

        def options_for(self, tool):
            return ("approve_once", "approve_session", "deny")

    class _Provider:
        def tool_for(self, name):
            return SimpleNamespace(name=name)

    def request_approvals(pending):
        # Broad scope FIRST, narrow second -- the ordering that used to lose it.
        return {
            row.call_id: ("approve_session" if row.call_id == "c1" else "approve_once")
            for row in pending
        }

    hook = build_tool_review_hook(
        _Gate(), _Provider(), None, request_approvals, workspace_id=None
    )
    hook(
        [
            ToolCall(name="read_file", args={"path": "a.md"}, call_id="c1"),
            ToolCall(name="read_file", args={"path": "b.md"}, call_id="c2"),
        ],
        RUN,
    )

    assert stamped == [("read_file", "approve_session")], (
        "the session grant the user chose was downgraded to approve_once, so "
        f"the next call re-prompts: {stamped}"
    )


# -- PR2a Task 7: cancellation revokes a child's pending approval cards ----
#
# The hazard: the approval wait blocks inside `_call_with_timeout`'s per-call
# daemon thread. When the fleet cancels or abandons a child while its card is
# still on screen, the user can still press Approve -- and the tool would
# EXECUTE FOR REAL (file written, message sent) for a run that already reads
# `cancelled`. `revoke_approval_rounds_for_run` is the fail-closed answer.

#: Two concurrent children of one turn. They share the console SESSION (the
#: fleet's children all run under the parent's session) and differ only by
#: run id -- which is exactly why round ownership had to become run-keyed.
RUN_A = "run-child-a"
RUN_B = "run-child-b"


def _arm_round(controller, *, run_id, session_id, llm_name, results):
    """Arm one approval round on a worker thread, owned by ``run_id``.

    Mirrors production: `request_mcp_approvals` runs on the agent bridge's
    worker thread, and the dispatching run's id reaches it through the
    `run_context` ContextVar that `AgentService` binds around the review
    hook and around each tool invocation.
    """

    def _run() -> None:
        with use_run_id(run_id):
            results[run_id] = controller.request_mcp_approvals(
                [_pending(llm_name=llm_name)], session_id=session_id
            )

    thread = threading.Thread(target=_run)
    thread.start()
    return thread


def _round_id_for(controller, run_id: str) -> str:
    rounds = [
        rid
        for rid, state in controller._pending_approval_rounds.items()
        if state.get("run_id") == run_id
    ]
    assert len(rounds) == 1, f"expected exactly one armed round for {run_id}: {rounds}"
    return rounds[0]


def test_revoking_a_run_denies_its_rounds_and_leaves_another_runs_untouched():
    """Two concurrent children, one cancelled: only its own card is revoked.

    Both rounds belong to the SAME console session, so nothing session-keyed
    could tell them apart -- the round has to record which RUN armed it.
    """
    controller, store = _build_controller()
    session_id = store.create_session(title="Fleet").id
    store.switch_session(session_id)
    controller.app = _FakeApp()
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    results: dict[str, dict[str, str]] = {}
    worker_a = _arm_round(
        controller,
        run_id=RUN_A,
        session_id=session_id,
        llm_name="mcp__srv__a",
        results=results,
    )
    worker_b = _arm_round(
        controller,
        run_id=RUN_B,
        session_id=session_id,
        llm_name="mcp__srv__b",
        results=results,
    )
    time.sleep(0.15)

    round_a = _round_id_for(controller, RUN_A)
    round_b = _round_id_for(controller, RUN_B)
    assert controller._pending_approvals[session_id] == {round_a, round_b}

    assert controller.revoke_approval_rounds_for_run(RUN_A) == 1

    worker_a.join(timeout=3.0)
    assert not worker_a.is_alive(), "the revoked round never released its thread"
    assert results[RUN_A] == {"mcp__srv__a": "deny"}

    # The sibling child is mid-approval and must be completely undisturbed:
    # still armed, still counted for the badge, still blocking its thread.
    assert worker_b.is_alive()
    assert round_a not in controller._pending_approval_rounds
    assert round_b in controller._pending_approval_rounds
    assert controller._pending_approvals[session_id] == {round_b}
    assert controller.run_marker_for(session_id) is ConsoleRunMarker.NEEDS_APPROVAL

    controller.resolve_pending_approval(
        {"mcp__srv__b": "approve_once"}, round_id=round_b
    )
    worker_b.join(timeout=3.0)
    assert results[RUN_B] == {"mcp__srv__b": "approve_once"}
    assert session_id not in controller._pending_approvals


def test_revoking_a_run_unblocks_its_waiting_thread_and_clears_the_card():
    """Revocation resolves the round NOW -- not at the 120s auto-deny -- and
    takes the card down through the same clear a resolved card uses."""
    controller, store = _build_controller()
    session_id = store.create_session(title="Child").id
    store.switch_session(session_id)
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    results: dict[str, dict[str, str]] = {}
    worker = _arm_round(
        controller,
        run_id=RUN_A,
        session_id=session_id,
        llm_name="mcp__srv__tool",
        results=results,
    )
    time.sleep(0.15)
    assert mounted and mounted[-1] is not None, "precondition: the card is on screen"

    started = time.monotonic()
    assert controller.revoke_approval_rounds_for_run(RUN_A) == 1
    worker.join(timeout=3.0)
    elapsed = time.monotonic() - started

    assert not worker.is_alive()
    assert results[RUN_A] == {"mcp__srv__tool": "deny"}
    # Nowhere near the 30s deadline: the Event was set, not waited out.
    assert elapsed < 2.5
    assert mounted[-1] is None, "the revoked card was left on screen"
    assert session_id not in controller._pending_approvals
    assert (
        controller._head_round_payload(controller._parked_approval_payloads, session_id)
        is None
    )


def test_revoking_an_unknown_run_is_a_zero_return_noop():
    """Safe for a run that never armed a card (the overwhelmingly common
    case: every cancelled child with no approval outstanding)."""
    controller, store = _build_controller()
    session_id = store.create_session(title="Child").id
    store.switch_session(session_id)
    controller.app = _FakeApp()
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    # Nothing armed at all.
    assert controller.revoke_approval_rounds_for_run(RUN_A) == 0

    results: dict[str, dict[str, str]] = {}
    worker = _arm_round(
        controller,
        run_id=RUN_A,
        session_id=session_id,
        llm_name="mcp__srv__tool",
        results=results,
    )
    time.sleep(0.15)
    round_a = _round_id_for(controller, RUN_A)

    assert controller.revoke_approval_rounds_for_run("run-nobody") == 0
    # An empty/absent run id must never match the rounds armed outside any
    # run (a direct provider call) -- fail closed by refusing to sweep.
    assert controller.revoke_approval_rounds_for_run("") == 0

    assert round_a in controller._pending_approval_rounds
    assert controller._pending_approvals[session_id] == {round_a}
    assert worker.is_alive()

    controller.resolve_pending_approval(
        {"mcp__srv__tool": "approve_once"}, round_id=round_a
    )
    worker.join(timeout=3.0)
    assert results[RUN_A] == {"mcp__srv__tool": "approve_once"}


def test_a_decision_landing_after_a_revoke_cannot_reopen_the_round():
    """The in-flight-click race, pinned deterministically.

    `ApprovalDecided` travels as an async Textual message, so a user click
    can be delivered AFTER the fleet cancelled the child -- and
    `resolve_pending_approval` snapshots the round's shared decisions dict,
    so a write can even land in that dict after the round was torn down.
    The waiting thread must still return "deny".

    The worker is held inside its own mount callback (which runs on the
    worker thread, after the round is registered and before the wait loop),
    which makes the ordering exact rather than hopeful.
    """
    controller, store = _build_controller()
    session_id = store.create_session(title="Child").id
    store.switch_session(session_id)
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    release = threading.Event()

    def _mount(payload):
        mounted.append(payload)
        if payload is not None:
            # Park the worker just before it enters the wait loop.
            release.wait(5.0)

    controller.set_pending_approval = _mount
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    results: dict[str, dict[str, str]] = {}
    worker = _arm_round(
        controller,
        run_id=RUN_A,
        session_id=session_id,
        llm_name="mcp__srv__tool",
        results=results,
    )
    try:
        deadline = time.monotonic() + 3.0
        while not mounted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert mounted and mounted[0] is not None
        round_a = _round_id_for(controller, RUN_A)
        decisions_box = controller._pending_approval_rounds[round_a]["decisions"]

        assert controller.revoke_approval_rounds_for_run(RUN_A) == 1
        # The stale click is delivered now: by round id (a no-op, the round
        # is gone) AND straight into the shared box a pre-revoke snapshot
        # would still hold. Neither may resurrect the approval.
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "approve_once"}, round_id=round_a
        )
        decisions_box["mcp__srv__tool"] = "approve_once"
    finally:
        release.set()

    worker.join(timeout=3.0)
    assert not worker.is_alive()
    assert results[RUN_A] == {"mcp__srv__tool": "deny"}, (
        "a click that landed after cancellation re-opened a revoked round"
    )


@pytest.mark.unit
def test_a_revoked_childs_card_can_no_longer_execute_its_tool(tmp_path, monkeypatch):
    """End to end, with the real gate, the real provider and a real file.

    The whole point of the task: a child cancelled mid-approval must not be
    able to write to disk when the user presses Approve afterwards. Drives
    the production chain -- `build_tool_review_hook` -> `request_mcp_
    approvals` -> `BuiltinToolGate` stamp -> `BuiltinToolProvider.invoke`.
    """
    import tldw_chatbook.config as config_module
    import tldw_chatbook.Tools.file_operation_tools as fot
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    from Tests.Agents.test_builtin_tool_gate import _FakeService

    def _tools_setting(section, key=None, default=None):
        if section == "tools" and key == "write_file_enabled":
            return True
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", _tools_setting)
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path))

    controller, store = _build_controller()
    session_id = store.create_session(title="Child").id
    store.switch_session(session_id)
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_approval = mounted.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    gate = BuiltinToolGate(service=_FakeService())
    provider = BuiltinToolProvider(gate=gate)
    hook = build_tool_review_hook(
        gate,
        provider,
        None,
        lambda pending: controller.request_mcp_approvals(
            pending, session_id=session_id
        ),
        workspace_id=None,
    )
    args = {"file_path": "owned.txt", "content": "written by a cancelled child"}
    target = tmp_path / "owned.txt"

    verdicts: dict[str, str] = {}

    def _review() -> None:
        with use_run_id(RUN_A):
            verdicts.update(
                hook([ToolCall(name="write_file", args=args, call_id="call-1")], RUN_A)
            )

    worker = threading.Thread(target=_review)
    worker.start()
    time.sleep(0.2)
    assert mounted and mounted[-1] is not None, "precondition: the card is on screen"
    round_a = mounted[-1]["round_id"]

    # The fleet cancels/abandons this child while its card is still up.
    assert controller.revoke_approval_rounds_for_run(RUN_A) == 1
    worker.join(timeout=3.0)
    assert not worker.is_alive()

    # ... and the user presses Approve anyway.
    controller.resolve_pending_approval(
        {"call-1": "approve_once", "write_file": "approve_once"}, round_id=round_a
    )

    with use_run_id(RUN_A):
        result = provider.invoke("builtin:write_file", args)

    assert verdicts["call-1"] != "proceed", (
        f"the revoked call was cleared for dispatch: {verdicts}"
    )
    assert result.ok is False
    assert not target.exists(), (
        "a revoked approval executed the tool for real -- the file was written"
    )


# -- PR3a-1 Task 6b (audit F4): a survivor's round binds ITS OWN cancel signal
#
# `_is_session_cancelled` used to resolve `_active_cancel_events[session_id]`
# at POLL time, once a second, for the whole life of an approval round. That
# is correct only while the round's own turn is the turn that owns the
# session's entry. A fleet child that outlives its turn (PR3a-1's whole
# point) breaks the assumption in two directions, both silent:
#
#   * between turns the entry is gone, so `.get()` returns None; and
#   * DURING THE NEXT TURN the entry is the NEXT turn's Event, so pressing
#     Stop on turn 2 denies turn 1's survivor's still-open card and fails a
#     legitimate tool call closed with nothing saying why.
#
# The audit labelled the second direction INFERENCE FROM STRUCTURE, NOT
# EXECUTION. These tests execute it. The fix binds the Event BY VALUE at arm
# time, which is what `revoke_approval_rounds_for_run`'s run-keyed sweep
# already does for the push side.

#: The child of turn 1 that is still running when turn 2 begins.
RUN_SURVIVOR = "run-turn1-survivor"


def _begin_turn(controller, session_id: str) -> threading.Event:
    """Register one turn's own per-run cancel Event, as `_run_agent_reply` does."""
    cancel_event = threading.Event()
    controller._active_cancel_events[session_id] = cancel_event
    return cancel_event


def _end_turn(controller, session_id: str) -> None:
    """Drop the turn's per-session entry, as `_stream_assistant_response_inner`
    does once the turn (not its surviving children) has finished."""
    controller._active_cancel_events.pop(session_id, None)


def test_the_next_turns_stop_does_not_deny_an_earlier_turns_survivors_card():
    """Stop on turn 2 must not fail turn 1's survivor's tool call closed."""
    controller, store = _build_controller()
    session_id = store.create_session(title="Fleet").id
    store.switch_session(session_id)
    controller.app = _FakeApp()
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    turn_one_cancel = _begin_turn(controller, session_id)
    results: dict[str, dict[str, str]] = {}
    worker = _arm_round(
        controller,
        run_id=RUN_SURVIVOR,
        session_id=session_id,
        llm_name="mcp__srv__write",
        results=results,
    )
    time.sleep(0.15)
    survivor_round = _round_id_for(controller, RUN_SURVIVOR)

    # Turn 1 returns; its child (and the child's card) outlive it.
    _end_turn(controller, session_id)
    # Turn 2 starts with its OWN Event, and the user presses Stop on IT.
    _begin_turn(controller, session_id)
    controller._signal_stop(session_id=session_id)

    # More than one poll interval (`_MCP_APPROVAL_POLL_SECONDS` = 1.0s).
    worker.join(timeout=2.5)

    assert worker.is_alive(), (
        f"turn 2's Stop denied turn 1's survivor's still-open card: {results}"
    )
    assert survivor_round in controller._pending_approval_rounds
    assert results == {}
    assert not turn_one_cancel.is_set(), (
        "turn 2's Stop reached back into turn 1's own cancel Event"
    )

    # The survivor IS still stoppable -- through its own run-keyed revoke,
    # which is what the fleet panel's Cancel presses.
    assert controller.revoke_approval_rounds_for_run(RUN_SURVIVOR) == 1
    worker.join(timeout=3.0)
    assert not worker.is_alive()
    assert results[RUN_SURVIVOR] == {"mcp__srv__write": "deny"}


def test_a_round_armed_between_turns_is_not_deniable_by_a_later_turns_stop():
    """The other direction: a survivor arms a card with NO turn in flight.

    At poll time `_active_cancel_events.get(session_id)` was None -- and
    then became the NEXT turn's Event the moment the user sent again, so a
    Stop on that unrelated turn denied this card too.
    """
    controller, store = _build_controller()
    session_id = store.create_session(title="Fleet").id
    store.switch_session(session_id)
    controller.app = _FakeApp()
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    # No turn in flight: turn 1 has returned, turn 2 has not begun.
    assert session_id not in controller._active_cancel_events

    results: dict[str, dict[str, str]] = {}
    worker = _arm_round(
        controller,
        run_id=RUN_SURVIVOR,
        session_id=session_id,
        llm_name="mcp__srv__write",
        results=results,
    )
    time.sleep(0.15)
    survivor_round = _round_id_for(controller, RUN_SURVIVOR)

    _begin_turn(controller, session_id)
    controller._signal_stop(session_id=session_id)
    worker.join(timeout=2.5)

    assert worker.is_alive(), (
        f"a later turn's Stop denied a between-turns round: {results}"
    )
    assert survivor_round in controller._pending_approval_rounds

    assert controller.revoke_approval_rounds_for_run(RUN_SURVIVOR) == 1
    worker.join(timeout=3.0)
    assert not worker.is_alive()
    assert results[RUN_SURVIVOR] == {"mcp__srv__write": "deny"}


def test_the_owning_turns_own_stop_still_denies_its_round():
    """The control: binding by value must not make Stop stop working.

    A round armed by the turn that owns the session's Event is denied by
    that turn's Stop, exactly as before -- the fix narrows WHICH Event a
    round listens to, it does not remove the signal.
    """
    controller, store = _build_controller()
    session_id = store.create_session(title="Owner").id
    store.switch_session(session_id)
    controller.app = _FakeApp()
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    _begin_turn(controller, session_id)
    results: dict[str, dict[str, str]] = {}
    worker = _arm_round(
        controller,
        run_id=RUN_A,
        session_id=session_id,
        llm_name="mcp__srv__tool",
        results=results,
    )
    time.sleep(0.15)
    assert _round_id_for(controller, RUN_A)

    controller._signal_stop(session_id=session_id)
    worker.join(timeout=3.0)

    assert not worker.is_alive(), "the owning turn's Stop no longer reaches its round"
    assert results[RUN_A] == {"mcp__srv__tool": "deny"}


def test_shutdown_still_denies_a_survivors_round():
    """Teardown is the one signal that legitimately reaches every round.

    `shutdown()` sets `_shutdown_requested`, which is never reset for this
    controller instance -- a survivor whose bound Event is None must still
    fail closed when the Console screen it is attached to goes away.
    """
    controller, store = _build_controller()
    session_id = store.create_session(title="Fleet").id
    store.switch_session(session_id)
    controller.app = _FakeApp()
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    results: dict[str, dict[str, str]] = {}
    worker = _arm_round(
        controller,
        run_id=RUN_SURVIVOR,
        session_id=session_id,
        llm_name="mcp__srv__write",
        results=results,
    )
    time.sleep(0.15)

    controller.begin_shutdown()
    worker.join(timeout=3.0)

    assert not worker.is_alive(), "teardown left a survivor's round parked"
    assert results[RUN_SURVIVOR] == {"mcp__srv__write": "deny"}


# -- ADR-067: no-deadline approval rounds --------------------------------
#
# The shipped default is now 0 = "no auto-deny": a round stays armed until
# the user answers or the run is stopped. A positive timeout still denies
# undecided calls (the existing expiry tests above, via seam values like
# 0.05, pin that). The pre-ADR code computed `deadline = now + 0` for a
# zero timeout and stamped "timeout" at the first 1s poll -- every test
# below asserts the round survives past that point.


def test_request_mcp_approvals_zero_timeout_keeps_round_armed_for_late_decision():
    """Timeout 0 means NO deadline: the round survives the first 1s poll
    and resolves only on the user's decision."""
    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 0.0

    def _decide_late() -> None:
        time.sleep(1.6)  # beyond the first poll where the old code bailed
        assert received and received[0] is not None
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "approve_once"}, round_id=received[0]["round_id"]
        )

    decider = threading.Thread(target=_decide_late)
    decider.start()
    started = time.monotonic()
    decisions = controller.request_mcp_approvals([_pending()])
    elapsed = time.monotonic() - started
    decider.join()

    assert decisions == {"mcp__srv__tool": "approve_once"}
    # The round was still armed past the old first-poll bail point.
    assert elapsed >= 1.5
    # The card was told no deadline exists (countdown copy hidden).
    assert received[0]["timeout_seconds"] == 0.0
    assert received[-1] is None


def test_request_mcp_approvals_marks_human_input_wait_while_round_armed():
    """While a round waits, its OWNING run is marked in the human-input
    wait registry so `_call_with_timeout` pauses the per-call clock --
    keyed by the round's `use_run_id` binding, cleared when it resolves."""
    from tldw_chatbook.Agents.human_input_wait import human_input_wait_active

    controller, _ = _build_controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    result: dict[str, object] = {}

    def _run_round() -> None:
        with use_run_id("run-approval-wait"):
            result["decisions"] = controller.request_mcp_approvals([_pending()])

    def _sample_then_decide() -> None:
        # Wait for the round to arm, sample the mark from THIS thread (the
        # wrapper's vantage point -- the mark must be process state, not
        # worker-thread-local), then resolve the round.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received:
            time.sleep(0.01)
        assert received, "round never armed"
        result["marked_while_armed"] = human_input_wait_active("run-approval-wait")
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "deny"}, round_id=received[0]["round_id"]
        )

    worker = threading.Thread(target=_run_round)
    decider = threading.Thread(target=_sample_then_decide)
    worker.start()
    decider.start()
    worker.join(timeout=10.0)
    decider.join(timeout=10.0)

    assert result["decisions"] == {"mcp__srv__tool": "deny"}
    assert result["marked_while_armed"] is True
    assert human_input_wait_active("run-approval-wait") is False


def test_request_mcp_approvals_without_ui_fails_closed_immediately():
    """With no UI bridge wired nothing can EVER resolve the round, and the
    no-deadline default means the poll loop would never end -- so the
    bridge must fail closed on the spot, mirroring the skill confirms'
    own no-app guards."""
    controller, _ = _build_controller()
    assert controller.app is None
    controller.mcp_approval_timeout_seconds = lambda: 0.05

    started = time.monotonic()
    decisions = controller.request_mcp_approvals([_pending()])
    elapsed = time.monotonic() - started

    assert decisions == {"mcp__srv__tool": "deny"}
    assert elapsed < 2.5


def test_mcp_approval_timeout_default_is_no_deadline(monkeypatch):
    """ADR-067: the shipped default flips 120 -> 0 (armed until answered);
    auto-deny is opt-in via [mcp] approval_timeout_seconds."""
    import tldw_chatbook.Chat.console_chat_controller as cc_module

    controller, _ = _build_controller()
    assert controller.mcp_approval_timeout_seconds is None  # no seam wired
    monkeypatch.setattr(
        cc_module, "get_cli_setting", lambda section, key, default: default
    )
    assert controller._resolve_mcp_approval_timeout_seconds() == 0.0


def test_human_prompt_defaults_pin_no_deadline():
    """ADR-067 contract pin: every blocking human prompt defaults to 0
    (wait indefinitely). Flip any of these and this test must fail."""
    import tldw_chatbook.Chat.console_chat_controller as cc_module

    assert cc_module._DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS == 0.0
    assert cc_module._DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS == 0.0
    assert cc_module._DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS == 0.0
