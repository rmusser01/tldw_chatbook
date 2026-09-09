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

    Args:
        active_workspace_id: Currently selected workspace id, if any.
        item_workspace_ids: Workspace ids associated with the visible item.
        item_type: Human-readable item type used in recovery copy.
        operation: Operation being attempted against the item.

    Returns:
        A `WorkspaceEligibility` decision preserving visibility and describing
        whether the active-context operation may proceed.

    Raises:
        ValueError: If `operation` is not a supported `WorkspaceOperation`.
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


#: (task-32056) Short, inline-safe labels for the reason codes a workspace
#: LINK can resolve. Rendered on the blocked control itself ("○ Open in
#: Console · not in this workspace"), beside the action that fixes it --
#: ``recovery_copy`` is a whole sentence and belongs in a tooltip, not on a
#: button. Codes absent from this map (``no_active_workspace``) are not
#: fixable by linking, so no link affordance is offered for them.
_LINKABLE_REASON_LABELS = {
    "not_in_active_workspace": "not in this workspace",
    "cross_workspace": "in another workspace",
}


#: (task-32056, fix round 1) Short label for a block that is real but whose
#: reason code linking cannot resolve -- an item missing from the row model
#: falls back to the aggregate handoff gate, which refuses. The control must
#: still disable and say so rather than claiming eligibility.
LIBRARY_GENERIC_WORKSPACE_BLOCK = "blocked for this workspace"


def linkable_ineligibility_label(reason_code: str) -> str:
    """Return the short inline label for a link-resolvable block.

    Args:
        reason_code: A ``WorkspaceEligibility.reason_code``.

    Returns:
        A short phrase for a blocked control's own label, or an empty
        string when the item is eligible or linking would not resolve it.
    """
    return _LINKABLE_REASON_LABELS.get(str(reason_code or "").strip(), "")


def _normalize_operation(operation: WorkspaceOperation | str) -> WorkspaceOperation:
    try:
        return (
            operation
            if isinstance(operation, WorkspaceOperation)
            else WorkspaceOperation(operation)
        )
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
