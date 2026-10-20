"""Workspace active-context eligibility tests."""

from __future__ import annotations

from tldw_chatbook.Workspaces import (
    WorkspaceOperation,
    evaluate_workspace_eligibility,
)


def test_cross_workspace_note_is_visible_but_not_context_eligible() -> None:
    result = evaluate_workspace_eligibility(
        active_workspace_id="ws-a",
        item_workspace_ids=("ws-b",),
        item_type="note",
        operation=WorkspaceOperation.STAGE_IN_CONSOLE,
    )

    assert result.visible is True
    assert result.active_context_eligible is False
    assert result.reason_code == "cross_workspace"
    assert "Copy or link" in result.recovery_copy
    # TASK-34000.24: the copy names the workspace by display name, never its
    # raw id (this pin used to assert the id was present).
    assert "ws-a" not in result.recovery_copy


def test_global_browse_and_search_remain_visible_without_workspace() -> None:
    for operation in (WorkspaceOperation.BROWSE, WorkspaceOperation.SEARCH):
        result = evaluate_workspace_eligibility(
            active_workspace_id=None,
            item_workspace_ids=(),
            item_type="media",
            operation=operation,
        )

        assert result.visible is True
        assert result.active_context_eligible is True
        assert result.reason_code == "visible"


def test_active_context_operation_requires_active_workspace_membership() -> None:
    result = evaluate_workspace_eligibility(
        active_workspace_id="ws-a",
        item_workspace_ids=("ws-a", "ws-b"),
        item_type="artifact",
        operation=WorkspaceOperation.RAG_GROUND,
    )

    assert result.visible is True
    assert result.active_context_eligible is True
    assert result.reason_code == "active_workspace_match"


def test_active_context_operation_without_workspace_has_recovery_copy() -> None:
    result = evaluate_workspace_eligibility(
        active_workspace_id=None,
        item_workspace_ids=("ws-a",),
        item_type="conversation",
        operation=WorkspaceOperation.AGENT_MANIPULATE,
    )

    assert result.visible is True
    assert result.active_context_eligible is False
    assert result.reason_code == "no_active_workspace"
    assert "Select an active workspace" in result.recovery_copy


def test_global_item_is_not_silently_converted_to_active_context() -> None:
    result = evaluate_workspace_eligibility(
        active_workspace_id="ws-a",
        item_workspace_ids=(),
        item_type="conversation",
        operation=WorkspaceOperation.STAGE_IN_CONSOLE,
    )

    assert result.visible is True
    assert result.active_context_eligible is False
    assert result.reason_code == "not_in_active_workspace"
    assert "Copy or link" in result.recovery_copy


# --- TASK-34000.24: the shared copy names the workspace, never its raw id ---


def test_recovery_copy_names_the_workspace_by_display_name() -> None:
    result = evaluate_workspace_eligibility(
        active_workspace_id="workspace-default",
        item_workspace_ids=(),
        item_type="media",
        operation=WorkspaceOperation.STAGE_IN_CONSOLE,
        active_workspace_name="Local Default",
    )

    assert result.reason_code == "not_in_active_workspace"
    assert result.recovery_copy == (
        "Copy or link this media into the Local Default workspace before "
        "using it in Console."
    )
    assert "workspace-default" not in result.recovery_copy


def test_recovery_copy_for_a_cross_workspace_item_never_shows_the_id() -> None:
    result = evaluate_workspace_eligibility(
        active_workspace_id="ws-thesis-9f2c",
        item_workspace_ids=("ws-other",),
        item_type="conversation",
        operation=WorkspaceOperation.STAGE_IN_CONSOLE,
        active_workspace_name="Thesis",
    )

    assert result.reason_code == "cross_workspace"
    assert result.recovery_copy == (
        "Copy or link this conversation into the Thesis workspace before "
        "using it in Console."
    )
    assert "ws-thesis-9f2c" not in result.recovery_copy


def test_recovery_copy_without_a_name_says_the_active_workspace() -> None:
    result = evaluate_workspace_eligibility(
        active_workspace_id="ws-a",
        item_workspace_ids=(),
        item_type="note",
        operation=WorkspaceOperation.STAGE_IN_CONSOLE,
    )

    assert result.recovery_copy == (
        "Copy or link this note into the active workspace before using it "
        "in Console."
    )
    assert "ws-a" not in result.recovery_copy
