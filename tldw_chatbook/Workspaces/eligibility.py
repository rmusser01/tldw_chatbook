"""Pure workspace active-context eligibility rules."""

from __future__ import annotations

from collections.abc import Iterable

from .models import WorkspaceEligibility, WorkspaceOperation


_ACTIVE_CONTEXT_OPERATIONS = {
    WorkspaceOperation.STAGE_IN_CONSOLE,
    WorkspaceOperation.RAG_GROUND,
    WorkspaceOperation.AGENT_MANIPULATE,
    WorkspaceOperation.TOOL_USE,
}


def evaluate_workspace_eligibility(
    *,
    active_workspace_id: str | None,
    item_workspace_ids: Iterable[str],
    item_type: str,
    operation: WorkspaceOperation | str,
) -> WorkspaceEligibility:
    """Evaluate visibility and active-context eligibility for one item operation.

    Workspace switching must never hide user-owned Library/Notes/Artifact records.
    The gating applies only when the item would be staged into the active Console
    context or manipulated by an agent/runtime.
    """

    normalized_operation = _normalize_operation(operation)
    workspace_ids = tuple(_normalize_workspace_ids(item_workspace_ids))

    if normalized_operation not in _ACTIVE_CONTEXT_OPERATIONS:
        return WorkspaceEligibility(
            visible=True,
            active_context_eligible=True,
            reason_code="visible",
        )

    active_id = active_workspace_id.strip() if active_workspace_id else ""
    if not active_id:
        return WorkspaceEligibility(
            visible=True,
            active_context_eligible=False,
            reason_code="no_active_workspace",
            recovery_copy=(
                "Select an active workspace before using this item in Console."
            ),
        )

    if active_id in workspace_ids:
        return WorkspaceEligibility(
            visible=True,
            active_context_eligible=True,
            reason_code="active_workspace_match",
        )

    reason_code = "cross_workspace" if workspace_ids else "not_in_active_workspace"
    return WorkspaceEligibility(
        visible=True,
        active_context_eligible=False,
        reason_code=reason_code,
        recovery_copy=(
            f"Copy or link this {item_type} into workspace {active_id} before "
            "using it in Console."
        ),
    )


def _normalize_operation(operation: WorkspaceOperation | str) -> WorkspaceOperation:
    try:
        return operation if isinstance(operation, WorkspaceOperation) else WorkspaceOperation(operation)
    except ValueError as exc:
        raise ValueError("operation is invalid") from exc


def _normalize_workspace_ids(workspace_ids: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for workspace_id in workspace_ids:
        if not isinstance(workspace_id, str):
            continue
        value = workspace_id.strip()
        if value:
            normalized.append(value)
    return tuple(dict.fromkeys(normalized))
