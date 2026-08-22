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


def build_workspace_tree_state(
    *,
    workspaces: Iterable[tuple[str, str]],
    rows: Iterable[ConsoleConversationBrowserInputRow],
    next_cursors: Mapping[str, int | None] | None = None,
) -> tuple[WorkspaceTreeWorkspace, ...]:
    """Build the named-workspace projection without I/O or UI dependencies."""
    cursors = dict(next_cursors or {})
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
        conversation_id = str(row.conversation_id or "").strip()
        if not conversation_id or conversation_id in seen_conversation_ids:
            continue
        seen_conversation_ids.add(conversation_id)
        workspace_id = str(row.workspace_id or "").strip()
        if row.scope_type == "global" or workspace_id not in owned_rows:
            continue
        owned_rows[workspace_id].append(row)

    return tuple(
        WorkspaceTreeWorkspace(
            workspace_id=workspace_id,
            label=label,
            conversations=tuple(
                WorkspaceTreeConversation(
                    conversation_id=str(row.conversation_id),
                    title=str(row.title or ""),
                    starred=bool(row.starred),
                    updated_sort=str(row.updated_sort or ""),
                    selected=bool(row.selected),
                    run_marker=str(row.run_marker or ""),
                )
                for row in sorted(
                    owned_rows[workspace_id],
                    key=console_conversation_starred_recency_sort_key,
                )
            ),
            next_cursor=cursors.get(workspace_id),
        )
        for workspace_id, label in sorted(
            labels.items(), key=lambda item: (item[1].casefold(), item[0])
        )
    )


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
