"""AC27: incomplete pending facts never inherit a complete state's authority."""

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    ConsoleDisplayRow,
    ConsoleInspectorState,
)
from tldw_chatbook.Widgets.Console.console_inspector_ownership import (
    InspectorOwnershipPolicy,
    classify_inspector_content,
)
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    project_console_send_authority,
)


# Retain the profile admitted before collection-time model imports.
pytestmark = pytest.mark.bootstrap_profile

def test_pending_only_marker_survives_owned_projection_and_equality():
    state = ConsoleInspectorState.from_pending_facts(2, "Waiting for your answer")
    owned = classify_inspector_content(state, InspectorOwnershipPolicy.STRICT)
    assert state.pending_only and owned.pending_only and not owned.incomplete
    assert replace(state).pending_only
    assert state != replace(state, pending_only=False)
    assert state.with_pending_facts(0, "") is None
    assert state._pending_inputs is None
    assert state.pending_approval_count == 2 and state.has_pending_approval
    assert state.can_save_chatbook is False and state.scope_item_count is None
    assert {action.widget_id for action in owned.known_actions} == {
        CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID
    }
    assert owned.known_actions[0].enabled


@pytest.mark.parametrize(
    "count,copy,live,review",
    [
        (1, "Waiting for your answer", "Waiting for your approval", True),
        (0, "Waiting for your answer", "Waiting for your answer", True),
        (0, "Waiting for your confirmation", "Waiting for your confirmation", True),
        (0, "", "Refreshing…", False),
    ],
)
def test_partial_decision_rows_and_pinned_summary_stay_coherent(
    count, copy, live, review
):
    state = ConsoleInspectorState.from_pending_facts(count, copy)
    rows = {row.label: row.value for row in state.rows}
    assert rows["Live work"] == live
    assert rows["Approvals"] == f"{count} pending"
    assert f"approvals {count}" in rows["Run recipe"]
    assert all(
        value == "Refreshing…"
        for label, value in rows.items()
        if label not in {"Live work", "Approvals", "Run recipe"}
    )
    assert state.actions[0].enabled is review
    projection = project_console_send_authority(state)
    assert projection.where == projection.scope == projection.sources == "Refreshing…"
    assert projection.run == live
    assert projection.approvals == f"{count} pending" + (
        " · action required" if count else ""
    )


def test_pending_only_marker_keeps_strict_ownership_validation():
    state = ConsoleInspectorState.from_pending_facts(1, "")
    invalid = replace(
        state, rows=state.rows + (ConsoleDisplayRow("Unknown", "foreign"),)
    )
    with pytest.raises(ValueError, match="Unowned Inspector content"):
        project_console_send_authority(
            invalid, ownership_policy=InspectorOwnershipPolicy.STRICT
        )


def test_pending_only_constructor_rejects_raw_optional_objects_without_stringifying():
    class Raw:
        calls = 0

        def __str__(self):
            self.calls += 1
            return "Waiting for your approval"

    raw = Raw()
    with pytest.raises(ValueError, match="invalid_pending_display_facts"):
        ConsoleInspectorState.from_pending_facts(1, raw)
    assert raw.calls == 0
    with pytest.raises(ValueError, match="invalid_pending_display_copy"):
        ConsoleInspectorState.from_pending_facts(0, "Ready")
