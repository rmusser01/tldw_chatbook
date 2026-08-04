"""Regression tests for the round-3 UX batch (UX-052, 053, 055, 056, 058-063)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Evals.navigation.eval_nav_screen import (
    EVAL_NAV_CARDS,
    evals_workflows_chip_label,
)
from tldw_chatbook.UI.Evals.screens.quick_test import QuickTestScreen
from tldw_chatbook.UI.Logs_Window import LogsWindow
from tldw_chatbook.UI.Navigation.main_navigation import nav_button_label
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
from tldw_chatbook.UI.Screens.scheduling.conflicts_tab import ConflictsTab


# UX-053 -----------------------------------------------------------------
def test_nav_labels_carry_the_ctrl_modifier() -> None:
    assert nav_button_label(0, "Home") == "^1 Home"
    assert nav_button_label(9, "ACP") == "^0 ACP"
    # Unnumbered destinations carry their F-key route.
    assert nav_button_label(10, "Lab") == "F7 Lab"
    assert nav_button_label(12, "Settings") == "F9 Settings"


# UX-052 + UX-058 ---------------------------------------------------------
def test_quick_test_card_is_marked_demo() -> None:
    quick_test = next(c for c in EVAL_NAV_CARDS if c.id == "quick_test")
    assert quick_test.demo is True
    assert "simulated" in quick_test.description.lower()


def test_evals_chip_label_counts_live_demo_planned() -> None:
    assert evals_workflows_chip_label() == "2 live · 1 demo · 3 planned"


def test_evals_screen_binds_no_dead_digit_keys() -> None:
    keys = {binding.key for binding in EvalsScreen.BINDINGS}
    assert keys == {"escape", "1", "4", "5"}


class _QuickTestHost(App[None]):
    def __init__(self):
        super().__init__()
        self._screen = QuickTestScreen(app_instance=SimpleNamespace(notify=MagicMock()))

    def compose(self) -> ComposeResult:
        yield self._screen


@pytest.mark.asyncio
async def test_simulated_results_are_labeled_everywhere() -> None:
    with patch.object(QuickTestScreen, "_initialize_screen", lambda self: None):
        app = _QuickTestHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app._screen
            screen._handle_results(
                {
                    "task": "demo-task",
                    "model": "demo-model",
                    "samples": 1,
                    "accuracy": 0.87,
                    "duration": "1s",
                    "timestamp": "2026-08-04T00:00:00",
                }
            )
            await pilot.pause()
            summary = str(screen.query_one("#summary-text", Static).render())
            assert "SIMULATED RUN" in summary
            assert "no model was queried" in summary
            detail = screen.query_one("#results-detail").text
            assert "simulated" in detail.lower()


# UX-056 -----------------------------------------------------------------
def test_logs_bindings_follow_adr_and_have_actions() -> None:
    keys = {binding.key for binding in LogsWindow.BINDINGS}
    assert keys == {"/", "p", "1", "2", "3", "4", "y"}
    for binding in LogsWindow.BINDINGS:
        assert hasattr(LogsWindow, f"action_{binding.action.split('(')[0]}") or hasattr(
            LogsWindow, "action_level"
        )


def test_logs_footer_hints_match_bindings() -> None:
    hint_keys = {key for key, _label in LogsWindow.LOGS_SHORTCUTS}
    assert hint_keys == {"/", "1-4", "p", "y"}


# UX-055 -----------------------------------------------------------------
def test_conflict_version_summary_covers_deletion_and_normal() -> None:
    assert (
        ConflictsTab._version_summary({}, missing_text="(deleted on server)")
        == "(deleted on server)"
    )
    summary = ConflictsTab._version_summary(
        {
            "updated_at": "2026-08-01T10:00:00",
            "record": {"title": "Digest", "cron": "0 9 * * *", "body": "hello"},
        },
        missing_text="(missing)",
    )
    assert "'Digest'" in summary
    assert "2026-08-01T10:00:00" in summary
    assert "0 9 * * *" in summary
    assert "hello" in summary
