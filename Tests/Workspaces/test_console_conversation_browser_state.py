"""Pure tests for the Default/unassigned Console Conversations projection."""

from __future__ import annotations

from datetime import datetime, timezone

from tldw_chatbook.Workspaces.conversation_browser_state import (
    CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT,
    CONSOLE_CONVERSATION_BROWSER_ROW_HEIGHT,
    CONSOLE_RAIL_SECTION_MIN_BUDGET_LINES,
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
    console_conversation_browser_group_row_limit,
    console_persisted_row_updated_sort,
    console_rail_section_height_budget,
    format_console_relative_age,
    overlay_console_conversation_markers,
)
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID


def _row(
    key: str,
    title: str,
    *,
    workspace_id: str | None = DEFAULT_WORKSPACE_ID,
    scope_type: str = "workspace",
    workspace_label: str = "Chats",
    starred: bool = False,
    selected: bool = False,
    source_kind: str = "persisted",
    updated_sort: str = "",
    updated_label: str = "",
    run_marker: str = "",
) -> ConsoleConversationBrowserInputRow:
    return ConsoleConversationBrowserInputRow(
        row_key=key,
        conversation_id=None if key.startswith("native:") else key,
        native_session_id=key.removeprefix("native:")
        if key.startswith("native:")
        else None,
        title=title,
        scope_type=scope_type,
        workspace_id=workspace_id,
        workspace_label=workspace_label,
        status="active" if selected else "workspace-thread",
        updated_label=updated_label,
        selected=selected,
        starred=starred,
        star_enabled=not key.startswith("native:"),
        source_kind=source_kind,
        updated_sort=updated_sort,
        run_marker=run_marker,
    )


def _chats(state):
    assert [section.section_id for section in state.sections] == ["chats"]
    return state.sections[0]


def test_section_budget_defaults_to_the_historical_ceiling_without_a_measurement():
    assert (
        console_rail_section_height_budget(None)
        == CONSOLE_RAIL_SECTION_MIN_BUDGET_LINES
    )
    assert (
        console_rail_section_height_budget(0)
        == CONSOLE_RAIL_SECTION_MIN_BUDGET_LINES
    )
    assert (
        console_rail_section_height_budget(-5)
        == CONSOLE_RAIL_SECTION_MIN_BUDGET_LINES
    )


def test_section_budget_splits_a_tall_rail_evenly_between_peer_sections():
    # 200 available lines: the Workspaces and Conversations sections each
    # get half the rail as their growth ceiling.
    assert console_rail_section_height_budget(200) == 100
    assert console_rail_section_height_budget(120) == 60
    # A 40-line rail halves below the historical ceiling: unchanged.
    assert console_rail_section_height_budget(40) == CONSOLE_RAIL_SECTION_MIN_BUDGET_LINES


def test_group_row_limit_defaults_to_the_historical_cap_without_a_measurement():
    assert (
        console_conversation_browser_group_row_limit(None)
        == CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT
    )
    assert (
        console_conversation_browser_group_row_limit(0)
        == CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT
    )
    assert (
        console_conversation_browser_group_row_limit(-5)
        == CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT
    )


def test_group_row_limit_keeps_the_default_on_short_rails():
    # A 40-line rail keeps the historical 20-line budget; after the tray's
    # fixed chrome that is 4 rows, below the 12-row floor: short terminals
    # are unchanged from the pre-adaptive behaviour.
    assert (
        console_conversation_browser_group_row_limit(40)
        == CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT
    )


def test_group_row_limit_scales_a_tall_rail_to_the_chats_share():
    # 200 available lines: the Chats budget is half the rail (100 lines),
    # and after the fixed chrome the visible-row cap is (100 - 8) // 3 = 30
    # rows -- the list fills its half of the rail alongside the Workspaces
    # section's half.
    budget = console_rail_section_height_budget(200)
    assert budget == 100
    assert (
        console_conversation_browser_group_row_limit(200)
        == (budget - 8) // CONSOLE_CONVERSATION_BROWSER_ROW_HEIGHT
        == 30
    )


def test_group_row_limit_grows_monotonically_with_available_height():
    limits = [
        console_conversation_browser_group_row_limit(height)
        for height in (None, 24, 48, 120, 200, 400)
    ]
    assert limits == sorted(limits)
    assert console_conversation_browser_group_row_limit(400) > (
        console_conversation_browser_group_row_limit(120)
    )


def test_flat_projection_excludes_named_workspaces_and_starred_aggregate() -> None:
    state = build_console_conversation_browser_state(
        rows=(
            _row(
                "named", "Named", workspace_id="ws-a", workspace_label="A", starred=True
            ),
            _row("default", "Default", starred=True),
            _row("global", "Global", workspace_id=None, scope_type="global"),
        ),
        active_workspace_id="ws-a",
    )

    chats = _chats(state)
    assert [row.conversation_id for row in chats.rows] == ["default", "global"]
    assert chats.count == 2


def test_starred_is_a_property_and_sorts_first_then_by_recency() -> None:
    state = build_console_conversation_browser_state(
        rows=(
            _row("new", "New", updated_sort="2026-08-22"),
            _row("star-old", "Star old", starred=True, updated_sort="2026-08-01"),
            _row("star-new", "Star new", starred=True, updated_sort="2026-08-20"),
            _row("old", "Old", updated_sort="2026-08-02"),
        ),
        active_workspace_id=None,
    )

    assert [row.conversation_id for row in _chats(state).rows] == [
        "star-new",
        "star-old",
        "new",
        "old",
    ]


def test_duplicate_stable_ids_are_materialized_once() -> None:
    state = build_console_conversation_browser_state(
        rows=(
            _row("same", "First", starred=True),
            _row("same", "Duplicate", starred=True),
        ),
        active_workspace_id=None,
    )

    assert [(row.conversation_id, row.title) for row in _chats(state).rows] == [
        ("same", "First")
    ]


def test_query_matches_only_flat_scope_and_reports_service_total() -> None:
    state = build_console_conversation_browser_state(
        rows=(
            _row("default", "Needle"),
            _row("named", "Needle named", workspace_id="ws-a", workspace_label="A"),
        ),
        active_workspace_id="ws-a",
        query="needle",
        result_total_count=7,
    )

    assert [row.conversation_id for row in _chats(state).rows] == ["default"]
    assert state.result_total_count == 7
    assert state.status_copy == "7 matches. Showing 1 of 7"


def test_literal_title_and_selected_run_marker_are_preserved() -> None:
    title = "[bold]你好[/bold] 🧭"
    state = build_console_conversation_browser_state(
        rows=(_row("selected", title, selected=True, run_marker="◆"),),
        active_workspace_id=None,
    )

    row = _chats(state).rows[0]
    assert row.title == title
    assert row.selected is True
    assert row.run_marker == "◆"
    assert state.selected_summary == title


def test_marker_overlay_reuses_unrelated_rows() -> None:
    target = _row("target", "Target")
    cleared = _row("cleared", "Cleared", run_marker="✗")
    unrelated = _row("unrelated", "Unrelated", run_marker="✓")

    rows = overlay_console_conversation_markers(
        (target, cleared, unrelated),
        starred_ids=("target",),
        selected_conversation_id="target",
        run_markers={"target": "◆", "cleared": ""},
    )

    assert (rows[0].starred, rows[0].selected, rows[0].run_marker) == (
        True,
        True,
        "◆",
    )
    assert rows[1].run_marker == ""
    assert rows[2] is unrelated


def test_flat_cap_reports_hidden_rows_and_marker_without_duplicates() -> None:
    state = build_console_conversation_browser_state(
        rows=(
            _row("one", "One", updated_sort="3"),
            _row("two", "Two", updated_sort="2"),
            _row("three", "Three", updated_sort="1", run_marker="◆"),
        ),
        active_workspace_id=None,
        group_row_limit=2,
    )

    chats = _chats(state)
    assert [row.conversation_id for row in chats.rows] == ["one", "two"]
    assert chats.hidden_count == 1
    assert chats.capped_run_marker == "◆"
    assert state.status_copy == "1 more conversation — search with Ctrl+K"


def test_empty_copy_points_named_workspace_users_to_workspaces() -> None:
    state = build_console_conversation_browser_state(
        rows=(_row("named", "Named", workspace_id="ws-a", workspace_label="A"),),
        active_workspace_id="ws-a",
    )

    chats = _chats(state)
    assert chats.count == 0
    assert chats.collapsed is True
    assert "Workspaces" in chats.empty_copy


def test_builder_fills_and_preserves_updated_labels() -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    generated = build_console_conversation_browser_state(
        rows=(_row("a", "A", updated_sort="2026-06-01T11:00:00+00:00"),),
        active_workspace_id=None,
        now=now,
    )
    supplied = build_console_conversation_browser_state(
        rows=(
            _row(
                "b",
                "B",
                updated_sort="2026-06-01T11:00:00+00:00",
                updated_label="recent",
            ),
        ),
        active_workspace_id=None,
        now=now,
    )

    assert _chats(generated).rows[0].updated_label == "1h"
    assert _chats(supplied).rows[0].updated_label == "recent"


def test_format_console_relative_age_buckets_and_bad_input() -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

    assert format_console_relative_age("2026-06-01T11:58:00Z", now=now) == "2m"
    assert format_console_relative_age("2026-06-01T10:00:00Z", now=now) == "2h"
    assert format_console_relative_age("2026-05-29T12:00:00Z", now=now) == "3d"
    assert format_console_relative_age("not-a-time", now=now) == ""


def test_persisted_updated_sort_uses_activity_fallback_chain() -> None:
    assert (
        console_persisted_row_updated_sort(
            {
                "updated_at": "explicit",
                "last_modified": "modified",
                "created_at": "created",
            }
        )
        == "explicit"
    )
    assert (
        console_persisted_row_updated_sort(
            {"last_modified": "modified", "created_at": "created"}
        )
        == "modified"
    )
    assert console_persisted_row_updated_sort({"created_at": "created"}) == "created"
    assert console_persisted_row_updated_sort({}) == ""
