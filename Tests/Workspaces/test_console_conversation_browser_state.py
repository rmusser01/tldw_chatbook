from __future__ import annotations

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
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
):
    return ConsoleConversationBrowserInputRow(
        row_key=key,
        conversation_id=None if key.startswith("native:") else key,
        native_session_id=key.removeprefix("native:") if key.startswith("native:") else None,
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
    )


def _section(state, section_id):
    return next(section for section in state.sections if section.section_id == section_id)


def _workspace_group(state, group_id):
    workspace_section = _section(state, "workspaces")
    return next(group for group in workspace_section.groups if group.group_id == group_id)


def test_browser_groups_starred_workspaces_and_chats():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Workspace chat", starred=True),
            _row("conv-b", "Global chat", scope_type="global", workspace_id=None, workspace_label="Chats"),
            _row("conv-c", "Default chat", workspace_id=DEFAULT_WORKSPACE_ID, workspace_label="Default"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={},
        query="",
    )

    assert [section.section_id for section in state.sections] == ["starred", "workspaces", "chats"]
    assert state.sections[0].rows[0].row_key == "conv-a"
    assert state.sections[1].groups[0].group_id == "workspace:ws-a"
    assert [row.row_key for row in state.sections[2].rows] == ["conv-c", "conv-b"]
    assert state.sections[2].rows[0].workspace_id == DEFAULT_WORKSPACE_ID
    assert state.sections[2].rows[1].scope_type == "global"


def test_search_exposes_matching_rows_from_collapsed_groups_without_changing_preference():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row("conv-b", "Needle", workspace_id="ws-b", workspace_label="Workspace B"),
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
            _row("conv-b", "Needle", workspace_id="ws-b", workspace_label="Workspace B"),
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
            _row("conv-a", "Global chat", scope_type="global", workspace_id=None, workspace_label="Chats"),
        ),
        active_workspace_id="ws-a",
    )

    chats = _section(state, "chats")
    assert chats.collapsed is False
    assert chats.count == 1


def test_native_rows_have_star_enabled_false():
    state = build_console_conversation_browser_state(
        rows=(
            _row("native:session-a", "Draft session", source_kind="native"),
        ),
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
            _row(f"conv-{index}", f"Needle {index}", updated_sort=f"2026-06-{index:02d}")
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
            _row(f"conv-{index}", f"Needle {index}", updated_sort=f"2026-06-{index:02d}")
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
            _row("conv-b", "Beta", scope_type="global", workspace_id=None, workspace_label="Chats"),
            _row("conv-c", "Gamma", selected=True, workspace_id="ws-c", workspace_label="Gamma WS"),
        ),
        active_workspace_id="ws-a",
        query="global",
    )

    assert [row.row_key for row in _section(state, "chats").rows] == ["conv-b"]

    status_state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Research"),
            _row("conv-c", "Gamma", selected=True, workspace_id="ws-c", workspace_label="Gamma WS"),
        ),
        active_workspace_id="ws-a",
        query="active",
    )

    assert [row.row_key for row in _workspace_group(status_state, "workspace:ws-c").rows] == ["conv-c"]


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
