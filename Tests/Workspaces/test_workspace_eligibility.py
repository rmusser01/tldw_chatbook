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
    assert "ws-a" in result.recovery_copy


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
