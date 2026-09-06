"""Regression tests for the round-3 UX batch (UX-053, 055, 056).

The UX-052/058/071 Evals card-hub tests were removed when dev retired the
hub (commit 46b4c61b5); the Quick Test real-eval path needs a home in
dev's bench/grid Evals architecture before it can be re-covered.
"""

from __future__ import annotations

from tldw_chatbook.UI.Logs_Window import LogsWindow
from tldw_chatbook.UI.Navigation.main_navigation import nav_button_label
from tldw_chatbook.UI.Screens.scheduling.conflicts_tab import ConflictsTab
from tldw_chatbook.UI.Screens.scheduling.unified_rows import _format_local_timestamp


# UX-053 -----------------------------------------------------------------
def test_nav_labels_carry_the_ctrl_modifier() -> None:
    assert nav_button_label("home", "Home") == "⌃1 Home"
    assert nav_button_label("acp", "ACP") == "⌃0 ACP"
    # Unnumbered destinations carry their F-key route.
    assert nav_button_label("lab", "Lab") == "F7 Lab"
    assert nav_button_label("settings", "Settings") == "F9 Settings"


# UX-056 -----------------------------------------------------------------
def test_logs_bindings_follow_adr_and_have_actions() -> None:
    keys = {binding.key for binding in LogsWindow.BINDINGS}
    assert keys == {"/", "p", "1", "2", "3", "4", "n", "N", "y"}
    for binding in LogsWindow.BINDINGS:
        assert hasattr(LogsWindow, f"action_{binding.action.split('(')[0]}") or hasattr(
            LogsWindow, "action_level"
        )


def test_logs_footer_hints_match_bindings() -> None:
    hint_keys = {key for key, _label in LogsWindow.LOGS_SHORTCUTS}
    assert hint_keys == {"/", "1-4", "p", "n", "y"}


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
    # task-31711 AC#3: a human-readable LOCAL timestamp, not the raw
    # ISO-8601 string -- pin the shared formatter's own output rather
    # than a literal clock value (which shifts with the test host's TZ).
    assert "2026-08-01T10:00:00" not in summary
    assert _format_local_timestamp("2026-08-01T10:00:00") in summary
    assert "0 9 * * *" in summary
    assert "hello" in summary
