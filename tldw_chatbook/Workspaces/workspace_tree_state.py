"""Pure immutable projection values for the Console workspace Tree."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    console_conversation_starred_recency_sort_key,
)
from .models import DEFAULT_WORKSPACE_ID


@dataclass(frozen=True, slots=True)
class WorkspaceTreeConversation:
    conversation_id: str
    title: str
    starred: bool
    updated_sort: str
    selected: bool
    run_marker: str


@dataclass(frozen=True, slots=True)
class WorkspaceTreeWorkspace:
    workspace_id: str
    label: str
    conversations: tuple[WorkspaceTreeConversation, ...]
    next_cursor: int | None
    loading: bool = False
    error: str = ""
    retry_cursor: int | None = None
    membership_unknown: bool = False
    active: bool = False


def build_workspace_tree_state(
    *,
    workspaces: Iterable[tuple[str, str]],
    rows: Iterable[ConsoleConversationBrowserInputRow],
    next_cursors: Mapping[str, int | None] | None = None,
    loading: Mapping[str, bool] | None = None,
    errors: Mapping[str, str] | None = None,
    retry_cursors: Mapping[str, int | None] | None = None,
    membership_unknown: Mapping[str, bool] | None = None,
    active_workspace_id: str | None = None,
    query: str = "",
) -> tuple[WorkspaceTreeWorkspace, ...]:
    """Build the named-workspace projection without I/O or UI dependencies."""
    cursors = dict(next_cursors or {})
    loading_by_workspace = dict(loading or {})
    errors_by_workspace = dict(errors or {})
    retries_by_workspace = dict(retry_cursors or {})
    unknown_by_workspace = dict(membership_unknown or {})
    normalized_query = str(query or "").strip().casefold()
    labels: dict[str, str] = {}
    for raw_workspace_id, raw_label in workspaces:
        workspace_id = str(raw_workspace_id or "").strip()
        if not workspace_id or workspace_id == DEFAULT_WORKSPACE_ID:
            continue
        label = str(raw_label or workspace_id)
        existing = labels.get(workspace_id)
        if existing is None or (label.casefold(), label) < (
            existing.casefold(),
            existing,
        ):
            labels[workspace_id] = label

    owned_rows: dict[str, list[ConsoleConversationBrowserInputRow]] = {
        workspace_id: [] for workspace_id in labels
    }
    seen_conversation_ids: set[str] = set()
    for row in rows:
        conversation_id = str(row.conversation_id or row.row_key or "").strip()
        if not conversation_id or conversation_id in seen_conversation_ids:
            continue
        seen_conversation_ids.add(conversation_id)
        workspace_id = str(row.workspace_id or "").strip()
        if row.scope_type == "global" or workspace_id not in owned_rows:
            continue
        owned_rows[workspace_id].append(row)

    projected: list[WorkspaceTreeWorkspace] = []
    for workspace_id, label in sorted(
        labels.items(), key=lambda item: (item[1].casefold(), item[0])
    ):
        workspace_rows = owned_rows[workspace_id]
        label_matches = normalized_query in label.casefold()
        if normalized_query and not label_matches:
            workspace_rows = [
                row
                for row in workspace_rows
                if normalized_query in str(row.title or "").casefold()
            ]
            if not workspace_rows:
                continue
        projected.append(
            WorkspaceTreeWorkspace(
                workspace_id=workspace_id,
                label=label,
                conversations=tuple(
                    WorkspaceTreeConversation(
                        conversation_id=str(row.conversation_id or row.row_key),
                        title=str(row.title or ""),
                        starred=bool(row.starred),
                        updated_sort=str(row.updated_sort or ""),
                        selected=bool(row.selected),
                        run_marker=str(row.run_marker or ""),
                    )
                    for row in sorted(
                        workspace_rows,
                        key=console_conversation_starred_recency_sort_key,
                    )
                ),
                next_cursor=cursors.get(workspace_id),
                loading=bool(loading_by_workspace.get(workspace_id, False)),
                error=str(errors_by_workspace.get(workspace_id, "") or ""),
                retry_cursor=retries_by_workspace.get(workspace_id),
                membership_unknown=bool(unknown_by_workspace.get(workspace_id, False)),
                active=workspace_id == str(active_workspace_id or ""),
            )
        )
    return tuple(projected)


def update_workspace_tree_conversation(
    workspaces: tuple[WorkspaceTreeWorkspace, ...],
    *,
    workspace_id: str,
    conversation: WorkspaceTreeConversation,
) -> tuple[WorkspaceTreeWorkspace, ...]:
    """Replace one existing child while retaining every unrelated owner object."""
    target_workspace_id = str(workspace_id or "").strip()
    updated: list[WorkspaceTreeWorkspace] = []
    changed = False
    for workspace in workspaces:
        if workspace.workspace_id != target_workspace_id:
            updated.append(workspace)
            continue
        conversations = tuple(
            conversation if row.conversation_id == conversation.conversation_id else row
            for row in workspace.conversations
        )
        if conversations == workspace.conversations:
            updated.append(workspace)
            continue
        changed = True
        updated.append(
            replace(
                workspace,
                conversations=tuple(
                    sorted(
                        conversations,
                        key=console_conversation_starred_recency_sort_key,
                    )
                ),
            )
        )
    return tuple(updated) if changed else workspaces
