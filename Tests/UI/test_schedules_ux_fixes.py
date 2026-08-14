"""Regression tests for the Schedules UX critique fixes (UX-001..UX-019).

Covers ADR-031 binding conventions, footer-hint truthfulness, plain-language
conflict labels, sync-bar copy, and reminder-form validation helper behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Input, Static

from tldw_chatbook.UI.Screens.scheduling.conflicts_tab import _conflict_type_label
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.scheduling.sync_status_widget import SyncStatusWidget

# ADR-031: keys screens must never bind for app actions.
_FORBIDDEN_KEYS = {
    "ctrl+c",
    "ctrl+v",
    "ctrl+x",
    "ctrl+s",
    "ctrl+d",
    "ctrl+z",
    "ctrl+a",
    "ctrl+r",
    "ctrl+w",
    "ctrl+p",
    "ctrl+q",
}


def test_schedules_bindings_avoid_terminal_conventions() -> None:
    bound = {binding.key for binding in SchedulesWorkbench.BINDINGS}
    assert bound.isdisjoint(_FORBIDDEN_KEYS), (
        f"Schedules binds terminal-convention keys: {bound & _FORBIDDEN_KEYS}"
    )


def test_schedules_bindings_use_single_letters() -> None:
    bound = {binding.key for binding in SchedulesWorkbench.BINDINGS}
    assert {"c", "d", "s"} <= bound


def test_every_binding_has_an_implemented_action() -> None:
    for binding in SchedulesWorkbench.BINDINGS:
        action = binding.action
        assert hasattr(SchedulesWorkbench, f"action_{action}"), (
            f"Binding '{binding.key}' advertises unimplemented action '{action}'"
        )


def test_footer_shortcuts_match_bindings_exactly() -> None:
    # Escape (clear marks) needs no advertising; everything else must be 1:1.
    binding_keys = {
        binding.key for binding in SchedulesWorkbench.BINDINGS if binding.key != "escape"
    }
    hint_keys = {key for key, _label in SchedulesWorkbench.SCHEDULES_SHORTCUTS}
    assert hint_keys == binding_keys, (
        f"Footer hints {hint_keys} are not 1:1 with BINDINGS {binding_keys}"
    )


def test_stub_actions_removed() -> None:
    assert not hasattr(SchedulesWorkbench, "action_run_now")
    assert not hasattr(SchedulesWorkbench, "action_pause_resume")


def test_conflict_type_labels_are_plain_language() -> None:
    assert _conflict_type_label({"server_state": {}}) == "Deleted on server"
    assert (
        _conflict_type_label({"server_state": {"updated_at": "2026-01-01"}})
        == "Changed on server"
    )


class _SyncBarHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield SyncStatusWidget(
            current_owner="local",
            active_server_id="http://127.0.0.1:8000",
            server_available=True,
        )


@pytest.mark.asyncio
async def test_sync_bar_labels_and_tooltips() -> None:
    app = _SyncBarHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        server = app.query_one("#scheduling-owner-server")
        clear = app.query_one("#scheduling-clear-error")
        assert str(server.label) == "Server (http://127.0.0.1:8000)"
        assert server.tooltip == "Use the connected server as the Schedules owner."
        assert str(clear.label) == "Clear"
        assert clear.tooltip == "Clear the latest scheduling sync error."


class _FormHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield Static("harness")


def _error_text(form: ReminderForm) -> str:
    return str(form.query_one("#reminder-errors", Static).render())


@pytest.mark.asyncio
async def test_past_one_time_run_is_rejected_for_new_tasks() -> None:
    app = _FormHarness()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = app.screen
        assert isinstance(form, ReminderForm)
        form.query_one("#reminder-title", Input).value = "Test task"
        form.query_one("#reminder-run-at", Input).value = "2020-01-01T00:00:00+00:00"
        form._save()
        await pilot.pause()
        assert "past" in _error_text(form)


@pytest.mark.asyncio
async def test_future_one_time_run_passes_validation() -> None:
    app = _FormHarness()
    results: list[dict] = []
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm(), callback=lambda r: results.append(r))
        await pilot.pause()
        form = app.screen
        assert isinstance(form, ReminderForm)
        future = datetime.now(timezone.utc) + timedelta(days=1)
        form.query_one("#reminder-title", Input).value = "Test task"
        form.query_one("#reminder-run-at", Input).value = future.isoformat()
        form._save()
        await pilot.pause()
        # Valid form dismisses with form_data; an invalid one stays on screen.
        assert results and results[0]["title"] == "Test task"
