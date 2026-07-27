from __future__ import annotations

from datetime import datetime, timezone

from tldw_chatbook.Workspaces.conversation_browser_state import (
    CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT,
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
    console_persisted_row_updated_sort,
    format_console_relative_age,
)
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID


def _row(
    key,
    title,
    *,
    scope_type="workspace",
    workspace_id="ws-a",
    workspace_label="Workspace A",
    starred=False,
    selected=False,
    source_kind="persisted",
    starred_sort="",
    updated_sort="",
    run_marker="",
):
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
        updated_label="1d",
        selected=selected,
        starred=starred,
        star_enabled=not key.startswith("native:"),
        source_kind=source_kind,
        starred_sort=starred_sort,
        updated_sort=updated_sort,
        run_marker=run_marker,
    )


def _section(state, section_id):
    return next(
        section for section in state.sections if section.section_id == section_id
    )


def _workspace_group(state, group_id):
    workspace_section = _section(state, "workspaces")
    return next(
        group for group in workspace_section.groups if group.group_id == group_id
    )


def test_browser_groups_starred_workspaces_and_chats():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Workspace chat", starred=True),
            _row(
                "conv-b",
                "Global chat",
                scope_type="global",
                workspace_id=None,
                workspace_label="Chats",
            ),
            _row(
                "conv-c",
                "Default chat",
                workspace_id=DEFAULT_WORKSPACE_ID,
                workspace_label="Default",
            ),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={},
        query="",
    )

    assert [section.section_id for section in state.sections] == [
        "starred",
        "workspaces",
        "chats",
    ]
    assert state.sections[0].rows[0].row_key == "conv-a"
    assert state.sections[1].groups[0].group_id == "workspace:ws-a"
    assert [row.row_key for row in state.sections[2].rows] == ["conv-c", "conv-b"]
    assert state.sections[2].rows[0].workspace_id == DEFAULT_WORKSPACE_ID
    assert state.sections[2].rows[1].scope_type == "global"


def test_search_exposes_matching_rows_from_collapsed_groups_without_changing_preference():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row(
                "conv-b", "Needle", workspace_id="ws-b", workspace_label="Workspace B"
            ),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={"workspace:ws-b": True},
        query="needle",
    )

    group = _workspace_group(state, "workspace:ws-b")
    assert group.collapsed is False
    assert group.preference_collapsed is True
    assert [row.title for row in group.rows] == ["Needle"]


def test_explicitly_expanded_inactive_workspace_group_is_remembered():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row("conv-b", "Beta", workspace_id="ws-b", workspace_label="Workspace B"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={"workspace:ws-b": False},
        query="",
    )

    group = _workspace_group(state, "workspace:ws-b")
    assert group.collapsed is False
    assert group.preference_collapsed is False
    assert [row.title for row in group.rows] == ["Beta"]


def test_dedupe_is_within_normal_group_not_across_starred_section():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Canonical", starred=True, source_kind="native"),
            _row("conv-a", "Duplicate", starred=True, source_kind="persisted"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={},
        query="",
    )

    starred = _section(state, "starred")
    workspaces = _section(state, "workspaces")
    assert [row.row_key for row in starred.rows] == ["conv-a"]
    assert [row.row_key for row in workspaces.groups[0].rows] == ["conv-a"]


def test_active_workspace_group_is_expanded_by_default():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row("conv-b", "Beta", workspace_id="ws-b", workspace_label="Workspace B"),
        ),
        active_workspace_id="ws-a",
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert group.collapsed is False
    assert group.preference_collapsed is False


def test_workspaces_section_is_expanded_by_default_when_it_has_rows():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
        ),
        active_workspace_id="ws-a",
    )

    workspaces = _section(state, "workspaces")
    assert workspaces.collapsed is False
    assert workspaces.count == 1
    assert [group.group_id for group in workspaces.groups] == ["workspace:ws-a"]


def test_workspaces_section_can_be_collapsed_by_preference():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row("conv-b", "Beta", workspace_id="ws-b", workspace_label="Workspace B"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={
            "section:workspaces": True,
            "workspace:ws-b": False,
        },
    )

    workspaces = _section(state, "workspaces")
    assert workspaces.collapsed is True
    assert workspaces.count == 2
    assert _workspace_group(state, "workspace:ws-b").collapsed is False


def test_search_exposes_workspaces_section_matches_when_section_preference_collapsed():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row(
                "conv-b", "Needle", workspace_id="ws-b", workspace_label="Workspace B"
            ),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={"section:workspaces": True},
        query="needle",
    )

    workspaces = _section(state, "workspaces")
    assert workspaces.collapsed is False
    assert [row.title for row in _workspace_group(state, "workspace:ws-b").rows] == [
        "Needle"
    ]


def test_inactive_workspace_groups_are_collapsed_by_default():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row("conv-b", "Beta", workspace_id="ws-b", workspace_label="Workspace B"),
        ),
        active_workspace_id="ws-a",
    )

    group = _workspace_group(state, "workspace:ws-b")
    assert group.collapsed is True
    assert group.preference_collapsed is True


def test_explicitly_expanded_inactive_workspace_groups_stay_expanded_after_refresh():
    preferences = {"workspace:ws-b": False}
    first = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row("conv-b", "Beta", workspace_id="ws-b", workspace_label="Workspace B"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences=preferences,
    )
    refreshed = build_console_conversation_browser_state(
        rows=(
            _row("conv-c", "Gamma", workspace_id="ws-a", workspace_label="Workspace A"),
            _row("conv-b", "Beta", workspace_id="ws-b", workspace_label="Workspace B"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences=preferences,
    )

    assert _workspace_group(first, "workspace:ws-b").collapsed is False
    assert _workspace_group(refreshed, "workspace:ws-b").collapsed is False


def test_starred_is_expanded_by_default_and_can_be_overridden():
    default_state = build_console_conversation_browser_state(
        rows=(_row("conv-a", "Alpha", starred=True),),
        active_workspace_id="ws-a",
    )
    collapsed_state = build_console_conversation_browser_state(
        rows=(_row("conv-a", "Alpha", starred=True),),
        active_workspace_id="ws-a",
        group_collapse_preferences={"section:starred": True},
    )

    assert _section(default_state, "starred").collapsed is False
    assert _section(collapsed_state, "starred").collapsed is True


def test_chats_is_expanded_when_it_has_rows():
    state = build_console_conversation_browser_state(
        rows=(
            _row(
                "conv-a",
                "Global chat",
                scope_type="global",
                workspace_id=None,
                workspace_label="Chats",
            ),
        ),
        active_workspace_id="ws-a",
    )

    chats = _section(state, "chats")
    assert chats.collapsed is False
    assert chats.count == 1


def test_native_rows_have_star_enabled_false():
    state = build_console_conversation_browser_state(
        rows=(_row("native:session-a", "Draft session", source_kind="native"),),
        active_workspace_id="ws-a",
    )

    row = _workspace_group(state, "workspace:ws-a").rows[0]
    assert row.native_session_id == "session-a"
    assert row.star_enabled is False


def test_persisted_native_rows_keep_star_enabled_true():
    state = build_console_conversation_browser_state(
        rows=(
            ConsoleConversationBrowserInputRow(
                row_key="conv-native-a",
                conversation_id="conv-native-a",
                native_session_id="session-a",
                title="Saved native session",
                scope_type="workspace",
                workspace_id="ws-a",
                workspace_label="Workspace A",
                status="active",
                star_enabled=True,
                source_kind="native",
            ),
        ),
        active_workspace_id="ws-a",
    )

    row = _workspace_group(state, "workspace:ws-a").rows[0]
    assert row.conversation_id == "conv-native-a"
    assert row.native_session_id == "session-a"
    assert row.source_kind == "native"
    assert row.star_enabled is True


def test_titles_are_plain_strings_and_do_not_render_markup_control_data():
    title = "[bold red]Do not style[/bold red]"
    state = build_console_conversation_browser_state(
        rows=(_row("conv-a", title),),
        active_workspace_id="ws-a",
        query="bold red",
    )

    row = _workspace_group(state, "workspace:ws-a").rows[0]
    assert row.title == title
    assert isinstance(row.title, str)


def test_capped_groups_expose_hidden_count_and_status_copy():
    state = build_console_conversation_browser_state(
        rows=tuple(
            _row(
                f"conv-{index}", f"Needle {index}", updated_sort=f"2026-06-{index:02d}"
            )
            for index in range(1, 5)
        ),
        active_workspace_id="ws-a",
        query="needle",
        group_row_limit=2,
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert [row.row_key for row in group.rows] == ["conv-4", "conv-3"]
    assert group.count == 4
    assert group.hidden_count == 2
    assert state.status_copy == "4 matches. Showing 2 of 4"


def test_result_total_count_is_used_for_capped_status_copy():
    state = build_console_conversation_browser_state(
        rows=tuple(
            _row(
                f"conv-{index}", f"Needle {index}", updated_sort=f"2026-06-{index:02d}"
            )
            for index in range(1, 4)
        ),
        active_workspace_id="ws-a",
        query="needle",
        result_total_count=10,
        result_limit=3,
        group_row_limit=10,
    )

    assert state.result_total_count == 10
    assert state.status_copy == "10 matches. Showing 3 of 10"


def test_status_copy_reports_actual_visible_rows_when_groups_exceed_result_limit():
    state = build_console_conversation_browser_state(
        rows=tuple(
            _row(
                f"conv-{workspace_index}-{row_index}",
                f"Needle {workspace_index}-{row_index}",
                workspace_id=f"ws-{workspace_index}",
                workspace_label=f"Workspace {workspace_index}",
                updated_sort=f"2026-06-{row_index:02d}",
            )
            for workspace_index in range(4)
            for row_index in range(2)
        ),
        active_workspace_id="ws-0",
        query="needle",
        result_total_count=20,
        result_limit=3,
        group_row_limit=2,
    )

    assert state.status_copy == "20 matches. Showing 8 of 20"


def test_query_matches_workspace_label_status_and_scope_copy():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Research"),
            _row(
                "conv-b",
                "Beta",
                scope_type="global",
                workspace_id=None,
                workspace_label="Chats",
            ),
            _row(
                "conv-c",
                "Gamma",
                selected=True,
                workspace_id="ws-c",
                workspace_label="Gamma WS",
            ),
        ),
        active_workspace_id="ws-a",
        query="global",
    )

    assert [row.row_key for row in _section(state, "chats").rows] == ["conv-b"]

    status_state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Research"),
            _row(
                "conv-c",
                "Gamma",
                selected=True,
                workspace_id="ws-c",
                workspace_label="Gamma WS",
            ),
        ),
        active_workspace_id="ws-a",
        query="active",
    )

    assert [
        row.row_key for row in _workspace_group(status_state, "workspace:ws-c").rows
    ] == ["conv-c"]


def test_selected_summary_prefers_title_and_workspace_label():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_label="Workspace A"),
            _row("conv-b", "Beta", workspace_label="Workspace B", selected=True),
        ),
        active_workspace_id="ws-a",
    )

    assert state.selected_summary == "Beta - Workspace B"


def test_all_empty_sort_fields_order_by_title_then_row_key_not_input_order():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-c", "Zulu"),
            _row("conv-b", "Alpha"),
            _row("conv-a", "Alpha"),
        ),
        active_workspace_id="ws-a",
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert [row.row_key for row in group.rows] == ["conv-a", "conv-b", "conv-c"]


def test_missing_sort_values_are_ordered_after_timestamped_rows():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-missing", "Missing", updated_sort=""),
            _row("conv-old", "Old", updated_sort="2026-06-01T00:00:00Z"),
            _row("conv-new", "New", updated_sort="2026-06-02T00:00:00Z"),
        ),
        active_workspace_id="ws-a",
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert [row.row_key for row in group.rows] == [
        "conv-new",
        "conv-old",
        "conv-missing",
    ]


def test_sort_keys_accept_supplementary_plane_text_without_surrogate_error():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-plane", "Plane", updated_sort="\U00102000"),
            _row("conv-normal", "Normal", updated_sort="2026-06-01T00:00:00Z"),
        ),
        active_workspace_id="ws-a",
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert {row.row_key for row in group.rows} == {"conv-plane", "conv-normal"}


def test_equal_title_order_by_row_key():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-c", "Same"),
            _row("conv-a", "Same"),
            _row("conv-b", "Same"),
        ),
        active_workspace_id="ws-a",
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert [row.row_key for row in group.rows] == ["conv-a", "conv-b", "conv-c"]


def test_workspace_group_label_tie_breaks_by_group_id():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-c", "Gamma", workspace_id="ws-c", workspace_label="Shared"),
            _row("conv-b", "Beta", workspace_id="ws-b", workspace_label="Shared"),
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Shared"),
        ),
        active_workspace_id=None,
    )

    workspaces = _section(state, "workspaces")
    assert [group.group_id for group in workspaces.groups] == [
        "workspace:ws-a",
        "workspace:ws-b",
        "workspace:ws-c",
    ]


def test_duplicate_filtered_rows_produce_status_copy_from_deduped_matches():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Needle", updated_sort="2026-06-02"),
            _row("conv-a", "Needle duplicate", updated_sort="2026-06-01"),
        ),
        active_workspace_id="ws-a",
        query="needle",
        group_row_limit=1,
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert group.count == 1
    assert group.hidden_count == 0
    assert state.result_total_count == 1
    assert state.status_copy == "1 match"


_NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


def test_format_console_relative_age_buckets():
    assert format_console_relative_age("2026-07-02T11:59:40+00:00", now=_NOW) == "now"
    assert format_console_relative_age("2026-07-02T11:58:00+00:00", now=_NOW) == "2m"
    assert format_console_relative_age("2026-07-02T10:59:00+00:00", now=_NOW) == "1h"
    assert format_console_relative_age("2026-06-29T12:00:00+00:00", now=_NOW) == "3d"
    assert format_console_relative_age("2026-06-10T12:00:00+00:00", now=_NOW) == "3w"
    assert format_console_relative_age("2024-06-10T12:00:00+00:00", now=_NOW) == "2y"


def test_format_console_relative_age_tolerates_bad_input():
    assert format_console_relative_age("", now=_NOW) == ""
    assert format_console_relative_age("not a timestamp", now=_NOW) == ""
    # SQLite space-separated naive timestamps are treated as UTC.
    assert format_console_relative_age("2026-07-02 11:58:00", now=_NOW) == "2m"
    # Future timestamps clamp to "now".
    assert format_console_relative_age("2026-07-02T13:00:00+00:00", now=_NOW) == "now"


def test_format_console_relative_age_tolerates_naive_now():
    naive_now = datetime(2026, 7, 2, 12, 0, 0)
    assert (
        format_console_relative_age("2026-07-02T11:58:00+00:00", now=naive_now) == "2m"
    )


def _input_row(**overrides):
    defaults = dict(
        row_key="conv-1",
        conversation_id="conv-1",
        native_session_id=None,
        title="Example",
        scope_type="workspace",
        workspace_id="ws-1",
        workspace_label="Workspace 1",
        updated_sort="2026-07-02T11:58:00+00:00",
    )
    defaults.update(overrides)
    return ConsoleConversationBrowserInputRow(**defaults)


def test_builder_fills_updated_label_from_updated_sort():
    state = build_console_conversation_browser_state(
        rows=[_input_row()],
        active_workspace_id="ws-1",
        now=_NOW,
    )
    workspaces = next(s for s in state.sections if s.section_id == "workspaces")
    row = workspaces.groups[0].rows[0]
    assert row.updated_label == "2m"


def test_builder_keeps_caller_supplied_updated_label():
    state = build_console_conversation_browser_state(
        rows=[_input_row(updated_label="today")],
        active_workspace_id="ws-1",
        now=_NOW,
    )
    workspaces = next(s for s in state.sections if s.section_id == "workspaces")
    assert workspaces.groups[0].rows[0].updated_label == "today"


def test_non_active_workspace_groups_default_collapsed_regression():
    state = build_console_conversation_browser_state(
        rows=[
            _input_row(),
            _input_row(
                row_key="conv-2",
                conversation_id="conv-2",
                workspace_id="ws-2",
                workspace_label="Workspace 2",
            ),
        ],
        active_workspace_id="ws-1",
        now=_NOW,
    )
    workspaces = next(s for s in state.sections if s.section_id == "workspaces")
    by_id = {group.group_id: group for group in workspaces.groups}
    assert not by_id["workspace:ws-1"].collapsed
    assert by_id["workspace:ws-2"].collapsed


def test_rows_sorted_recent_first_regression():
    state = build_console_conversation_browser_state(
        rows=[
            _input_row(
                row_key="old",
                conversation_id="old",
                title="Old",
                updated_sort="2026-06-01T00:00:00+00:00",
            ),
            _input_row(
                row_key="new",
                conversation_id="new",
                title="New",
                updated_sort="2026-07-01T00:00:00+00:00",
            ),
        ],
        active_workspace_id="ws-1",
        now=_NOW,
    )
    workspaces = next(s for s in state.sections if s.section_id == "workspaces")
    titles = [row.title for row in workspaces.groups[0].rows]
    assert titles == ["New", "Old"]


def test_persisted_updated_sort_prefers_last_modified_over_created_at():
    """TASK-355: normalize_conversation_row exposes last_modified/created_at but
    NO updated_at key, so the rail's recency ordering + age labels must derive
    from last_modified. Falling through to created_at orders the rail by
    creation time, so a just-used conversation looks stale and sorts wrong."""
    row = {
        "last_modified": "2026-07-21T13:19:00+00:00",
        "created_at": "2026-07-21T13:04:00+00:00",
    }
    assert console_persisted_row_updated_sort(row) == "2026-07-21T13:19:00+00:00"


def test_persisted_updated_sort_fallback_chain():
    # explicit updated_at still wins when present (back-compat).
    assert (
        console_persisted_row_updated_sort(
            {"updated_at": "U", "last_modified": "M", "created_at": "C"}
        )
        == "U"
    )
    # last_modified beats created_at (the recency fix).
    assert (
        console_persisted_row_updated_sort({"last_modified": "M", "created_at": "C"})
        == "M"
    )
    # created_at / last_updated remain the fallbacks when no recency field.
    assert console_persisted_row_updated_sort({"created_at": "C"}) == "C"
    assert console_persisted_row_updated_sort({"last_updated": "L"}) == "L"
    # empty and None-bearing rows never raise and collapse to "".
    assert console_persisted_row_updated_sort({}) == ""
    assert (
        console_persisted_row_updated_sort({"last_modified": None, "created_at": "C"})
        == "C"
    )


def test_normalized_conversation_row_recency_flows_through_helper():
    """Integration for TASK-355: persisted rail rows are built from
    normalize_conversation_row's output, which emits last_modified but NO
    updated_at — so the helper must source recency from last_modified end-to-end,
    reproducing the exact data shape the rail sees."""
    from tldw_chatbook.Chat.chat_conversation_service import normalize_conversation_row

    normalized = normalize_conversation_row(
        {
            "id": "conv-1",
            "title": "Long conversation about embeddings",
            "last_modified": "2026-07-21T13:19:00+00:00",
            "created_at": "2026-07-21T13:04:00+00:00",
        }
    )
    # The root cause: the normalized payload carries no updated_at key.
    assert "updated_at" not in normalized
    # The fix: recency is sourced from last_modified, not creation time.
    assert (
        console_persisted_row_updated_sort(normalized) == "2026-07-21T13:19:00+00:00"
    )


def _chat_rows(n):
    return tuple(
        _row(
            f"c{i}",
            f"Chat {i}",
            scope_type="global",
            workspace_id=None,
            workspace_label="Chats",
        )
        for i in range(n)
    )


def test_no_query_view_discloses_conversations_hidden_by_the_cap():
    """TASK-354: the rail silently caps each group at
    CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT; with no search active the
    overflow was dropped with zero disclosure, so the oldest conversations
    looked deleted. The no-query view must announce the hidden rows and how to
    reach them."""
    rows = _chat_rows(CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 3)
    state = build_console_conversation_browser_state(
        rows=rows, active_workspace_id="ws-a"
    )
    assert "3 more" in state.status_copy
    assert "Ctrl+K" in state.status_copy


def test_no_query_view_has_no_disclosure_when_nothing_is_capped():
    state = build_console_conversation_browser_state(
        rows=_chat_rows(3), active_workspace_id="ws-a"
    )
    assert state.status_copy == ""


def test_cap_disclosure_excludes_user_collapsed_sections():
    """A collapsed section shows its own count in the header — its rows are not
    'silently' hidden, so they must not inflate the cap disclosure."""
    rows = _chat_rows(CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 5)
    state = build_console_conversation_browser_state(
        rows=rows,
        active_workspace_id="ws-a",
        group_collapse_preferences={"section:chats": True},
    )
    assert state.status_copy == ""


# PA-T8 review fix round 1 (IMPORTANT 2): a collapsed workspace group's
# `rows` tuple is empty (`_visible_rows` returns `()` for a collapsed
# group), so any row-level `run_marker` glyph is otherwise invisible for
# exactly the user who collapsed the group. `group.run_marker` is computed
# from the FULL row set (before collapsing empties it) and is the single
# most-urgent glyph among them.


def test_collapsed_group_exposes_most_urgent_row_marker_on_the_group():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-b", workspace_label="Workspace B"),
            _row(
                "conv-b",
                "Beta",
                workspace_id="ws-b",
                workspace_label="Workspace B",
                run_marker="●",
            ),
        ),
        # ws-a is active, so ws-b defaults to collapsed (see
        # `test_inactive_workspace_groups_are_collapsed_by_default`).
        active_workspace_id="ws-a",
    )

    group = _workspace_group(state, "workspace:ws-b")
    assert group.collapsed is True
    assert group.rows == ()  # the row carrying the marker is indeed hidden
    assert group.run_marker == "●"


def test_collapsed_group_marker_picks_the_most_urgent_of_several():
    state = build_console_conversation_browser_state(
        rows=(
            _row(
                "conv-a",
                "Alpha",
                workspace_id="ws-b",
                workspace_label="Workspace B",
                run_marker="✓",
            ),
            _row(
                "conv-b",
                "Beta",
                workspace_id="ws-b",
                workspace_label="Workspace B",
                run_marker="◆",
            ),
            _row(
                "conv-c",
                "Gamma",
                workspace_id="ws-b",
                workspace_label="Workspace B",
                run_marker="●",
            ),
        ),
        active_workspace_id="ws-a",
    )

    # Urgency: NEEDS_APPROVAL ("◆") outranks RUNNING ("●") outranks
    # FINISHED_OK ("✓").
    assert _workspace_group(state, "workspace:ws-b").run_marker == "◆"


def test_expanded_group_still_exposes_run_marker_field_but_rows_show_their_own():
    """`group.run_marker` is computed unconditionally (cheap); it is the
    RENDERING layer's job (not this pure-state layer's) to only borrow it
    onto the header when collapsed. Expanded groups keep their per-row
    markers visible in `group.rows`, which this asserts stays populated."""
    state = build_console_conversation_browser_state(
        rows=(
            _row(
                "conv-a",
                "Alpha",
                workspace_id="ws-a",
                workspace_label="Workspace A",
                run_marker="●",
            ),
        ),
        active_workspace_id="ws-a",  # ws-a is active -> expanded by default
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert group.collapsed is False
    assert group.run_marker == "●"
    assert group.rows[0].run_marker == "●"


def test_collapsed_group_with_no_marked_rows_exposes_empty_marker():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-b", workspace_label="Workspace B"),
        ),
        active_workspace_id="ws-a",
    )

    assert _workspace_group(state, "workspace:ws-b").run_marker == ""


# TASK-912 AC#1: top-level sections (Starred/Workspaces/Chats) had no
# `run_marker` aggregate, so collapsing a whole section hid every marker
# beneath it. `section.run_marker` is the most-urgent glyph among ALL of a
# section's contents (its own rows, or every workspace group's full pre-cap
# rows), computed unconditionally -- same "the RENDERING layer decides when
# to borrow it onto the header" split as `group.run_marker`.


def test_collapsed_workspaces_section_exposes_most_urgent_marker_among_groups():
    state = build_console_conversation_browser_state(
        rows=(
            _row(
                "conv-a",
                "Alpha",
                workspace_id="ws-a",
                workspace_label="Workspace A",
                run_marker="✓",
            ),
            _row(
                "conv-b",
                "Beta",
                workspace_id="ws-b",
                workspace_label="Workspace B",
                run_marker="◆",
            ),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={"section:workspaces": True},
    )

    workspaces = _section(state, "workspaces")
    assert workspaces.collapsed is True
    # Urgency: NEEDS_APPROVAL ("◆") outranks FINISHED_OK ("✓") -- same table
    # as the group-header aggregation (AC#3).
    assert workspaces.run_marker == "◆"


def test_expanded_workspaces_section_still_exposes_run_marker_field():
    """Computed unconditionally regardless of collapse state -- it is the
    rendering layer's job to only borrow it onto the header when
    collapsed, same contract as `group.run_marker`."""
    state = build_console_conversation_browser_state(
        rows=(
            _row(
                "conv-a",
                "Alpha",
                workspace_id="ws-a",
                workspace_label="Workspace A",
                run_marker="●",
            ),
        ),
        active_workspace_id="ws-a",
    )

    workspaces = _section(state, "workspaces")
    assert workspaces.collapsed is False
    assert workspaces.run_marker == "●"


def test_collapsed_chats_section_exposes_most_urgent_row_marker():
    """Flat (non-grouped) sections aggregate over their own rows directly."""
    rows = _chat_rows(3) + (
        _row(
            "c-marked",
            "Marked chat",
            scope_type="global",
            workspace_id=None,
            workspace_label="Chats",
            run_marker="✗",
        ),
    )
    state = build_console_conversation_browser_state(
        rows=rows,
        active_workspace_id="ws-a",
        group_collapse_preferences={"section:chats": True},
    )

    chats = _section(state, "chats")
    assert chats.collapsed is True
    assert chats.run_marker == "✗"


def test_section_with_no_marked_contents_exposes_empty_marker():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={"section:workspaces": True},
    )

    assert _section(state, "workspaces").run_marker == ""


# TASK-912 AC#2: an expanded workspace group with more rows than
# `CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT` shows no header marker
# today -- a marked row pushed past the cap is invisible even though the
# group is expanded. `group.capped_run_marker` is the most-urgent glyph
# among ONLY the rows beyond the cap; a marker on a still-visible row must
# not surface it (that row already shows its own glyph).


def _workspace_rows_with_markers(markers_by_index):
    """`n` rows in the active workspace `ws-a`, newest-first by construction
    (descending `updated_sort`) so display order matches `range(n)` exactly.
    ``markers_by_index`` maps a row index to the `run_marker` it carries."""
    n = CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 3
    return tuple(
        _row(
            f"ws-a-{i}",
            f"Chat {i}",
            workspace_id="ws-a",
            workspace_label="Workspace A",
            updated_sort=f"2026-07-{31 - i:02d}T00:00:00",
            run_marker=markers_by_index.get(i, ""),
        )
        for i in range(n)
    )


def test_expanded_group_capped_row_marker_surfaces_on_header():
    marked_index = CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 1  # beyond the cap
    rows = _workspace_rows_with_markers({marked_index: "●"})
    state = build_console_conversation_browser_state(
        rows=rows, active_workspace_id="ws-a"
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert group.collapsed is False
    assert len(group.rows) == CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT
    assert group.capped_run_marker == "●"


def test_expanded_group_visible_row_marker_does_not_surface_capped_marker():
    marked_index = 2  # well within the cap -- already visible in group.rows
    rows = _workspace_rows_with_markers({marked_index: "●"})
    state = build_console_conversation_browser_state(
        rows=rows, active_workspace_id="ws-a"
    )

    group = _workspace_group(state, "workspace:ws-a")
    assert group.collapsed is False
    assert group.rows[marked_index].run_marker == "●"
    assert group.capped_run_marker == ""
    # `run_marker` (the full-row aggregate) still reflects it -- it is the
    # rendering layer, not the pure-state layer, that decides which of the
    # two fields applies.
    assert group.run_marker == "●"


def test_expanded_group_capped_marker_picks_most_urgent_of_hidden_rows_only():
    cap = CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT
    rows = _workspace_rows_with_markers(
        {
            0: "◆",  # visible (index 0 < cap) -- must not affect capped_run_marker
            cap: "✓",  # hidden, least urgent
            cap + 1: "●",  # hidden, more urgent than "✓"
        }
    )
    state = build_console_conversation_browser_state(
        rows=rows, active_workspace_id="ws-a"
    )

    group = _workspace_group(state, "workspace:ws-a")
    # Urgency: RUNNING ("●") outranks FINISHED_OK ("✓") -- same table as
    # `_most_urgent_run_marker` uses everywhere else (AC#3). The visible
    # "◆" (most urgent of all three) is excluded from this aggregate.
    assert group.capped_run_marker == "●"


def test_collapsed_group_capped_marker_field_stays_a_pure_computation():
    """`capped_run_marker` is computed unconditionally like `run_marker` --
    collapsed groups simply never have it rendered (the header already
    shows `run_marker`, the full aggregate, in that case)."""
    marked_index = CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 1
    rows = tuple(
        _row(
            f"ws-b-{i}",
            f"Chat {i}",
            workspace_id="ws-b",
            workspace_label="Workspace B",
            updated_sort=f"2026-07-{31 - i:02d}T00:00:00",
            run_marker="●" if i == marked_index else "",
        )
        for i in range(CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 3)
    )
    state = build_console_conversation_browser_state(
        rows=rows, active_workspace_id="ws-a",  # ws-b defaults to collapsed
    )

    group = _workspace_group(state, "workspace:ws-b")
    assert group.collapsed is True
    assert group.capped_run_marker == "●"
    assert group.run_marker == "●"
