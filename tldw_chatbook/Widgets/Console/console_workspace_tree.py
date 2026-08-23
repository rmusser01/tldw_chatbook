"""Native Textual workspace/conversation tree for the Console left rail."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import textual
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Tree
from textual.widgets._tree import TreeNode

from ...Workspaces.workspace_tree_state import WorkspaceTreeWorkspace
from ..glyph_fallback import ascii_glyph_mode, resolve_glyph


_TEXTUAL_PRIVATE_MOVE_VERSION = "8.2.8"
_NodeKind = Literal["workspace", "conversation", "status", "load-more", "retry"]


def _single_physical_row(raw: str) -> str:
    """Return one literal physical row; the full raw value remains in data."""

    rows = str(raw).splitlines()
    return rows[0] if rows else ""


@dataclass(frozen=True, slots=True)
class WorkspaceTreeNodeData:
    """Small immutable payload attached to every native Tree node."""

    kind: _NodeKind
    key: str
    raw_label: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    starred: bool = False
    selected: bool = False
    selectable: bool = False

    @classmethod
    def workspace(cls, workspace_id: str, label: str) -> "WorkspaceTreeNodeData":
        return cls(
            "workspace",
            f"workspace:{workspace_id}",
            label,
            workspace_id=workspace_id,
            selectable=True,
        )

    @classmethod
    def conversation(
        cls,
        workspace_id: str,
        conversation_id: str,
        label: str,
        *,
        starred: bool,
        selected: bool,
    ) -> "WorkspaceTreeNodeData":
        return cls(
            "conversation",
            f"conversation:{conversation_id}",
            label,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            starred=starred,
            selected=selected,
            selectable=True,
        )

    @classmethod
    def auxiliary(
        cls,
        kind: Literal["status", "load-more", "retry"],
        workspace_id: str,
        key: str,
        label: str,
    ) -> "WorkspaceTreeNodeData":
        return cls(
            kind,
            key,
            label,
            workspace_id=workspace_id,
            selectable=kind in {"load-more", "retry"},
        )


class WorkspaceTreeWorkspaceSelected(Message):
    """The user selected a workspace row."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        super().__init__()


class WorkspaceTreeConversationSelected(Message):
    """The user selected a saved conversation row."""

    def __init__(self, workspace_id: str, conversation_id: str) -> None:
        self.workspace_id = workspace_id
        self.conversation_id = conversation_id
        super().__init__()


class WorkspaceTreeStarRequested(Message):
    """The user requested the contextual Star/Unstar action."""

    def __init__(
        self,
        workspace_id: str,
        conversation_id: str,
        *,
        starred: bool,
    ) -> None:
        self.workspace_id = workspace_id
        self.conversation_id = conversation_id
        self.starred = starred
        super().__init__()


class WorkspaceTreeLoadMoreRequested(Message):
    """The user requested the next page for one workspace."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        super().__init__()


class WorkspaceTreeRetryRequested(Message):
    """The user requested retry for one workspace page."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        super().__init__()


class WorkspaceTreeExpansionChanged(Message):
    """A non-search workspace disclosure gesture changed."""

    def __init__(self, workspace_id: str, *, expanded: bool) -> None:
        self.workspace_id = workspace_id
        self.expanded = expanded
        super().__init__()


class ConsoleWorkspaceTree(Tree[WorkspaceTreeNodeData]):
    """Incrementally synchronize the Console Workspace projection into a Tree.

    The hidden synthetic root is structural only. Business node objects survive
    labels, markers, ordering, and membership changes so Textual's cursor and
    local scroll state remain attached to the same objects.
    """

    BINDINGS = [
        *Tree.BINDINGS,
        Binding("left", "workspace_left", "Collapse", show=False),
        Binding("right", "workspace_right", "Expand", show=False),
        Binding("s", "workspace_star", "Star/Unstar", show=False),
    ]

    if ascii_glyph_mode():
        ICON_NODE = "> "
        ICON_NODE_EXPANDED = "v "
        LINES = {
            "default": ("  ", "| ", "`-", "|-"),
            "bold": ("  ", "| ", "`-", "|-"),
            "double": ("  ", "| ", "`-", "|-"),
        }
    else:
        ICON_NODE = f"{resolve_glyph('▸')} "
        ICON_NODE_EXPANDED = f"{resolve_glyph('▾')} "
        LINES = {
            "default": ("  ", "│ ", "└─", "├─"),
            "bold": ("  ", "┃ ", "┗━", "┣━"),
            "double": ("  ", "║ ", "╚═", "╠═"),
        }

    def __init__(self, **kwargs) -> None:
        if ascii_glyph_mode():
            self.ICON_NODE = "> "
            self.ICON_NODE_EXPANDED = "v "
            self.LINES = {
                "default": ("  ", "| ", "`-", "|-"),
                "bold": ("  ", "| ", "`-", "|-"),
                "double": ("  ", "| ", "`-", "|-"),
            }
        super().__init__(
            Text("Workspace tree"),
            data=None,
            id=kwargs.pop("id", "console-workspace-tree"),
            **kwargs,
        )
        self.show_root = False
        self.auto_expand = False
        self.guide_depth = 2
        self.root.expand()
        self.root.allow_expand = False
        self.workspace_nodes: dict[str, TreeNode[WorkspaceTreeNodeData]] = {}
        self.conversation_nodes: dict[str, TreeNode[WorkspaceTreeNodeData]] = {}
        self.auxiliary_nodes: dict[str, TreeNode[WorkspaceTreeNodeData]] = {}
        self._syncing = False
        self._search_active = False
        self._search_expansion_snapshot: frozenset[str] | None = None
        self._search_workspace_ids_snapshot: frozenset[str] | None = None
        self._preferred_expanded_workspace_ids: set[str] = set()
        self.expansion_preferences_changed: Callable[[frozenset[str]], None] | None = (
            None
        )
        self.star_enabled = True
        self.can_focus = False

    @staticmethod
    def _workspace_label(workspace: WorkspaceTreeWorkspace) -> Text:
        marker = "● " if getattr(workspace, "active", False) else ""
        return Text(f"{marker}{_single_physical_row(workspace.label)}")

    @staticmethod
    def _conversation_label(conversation) -> Text:
        parts = []
        if conversation.starred:
            parts.append("★")
        if conversation.selected:
            parts.append("›")
        if conversation.run_marker:
            parts.append(resolve_glyph(conversation.run_marker))
        prefix = f"{' '.join(parts)} " if parts else ""
        return Text(f"{prefix}{_single_physical_row(conversation.title)}")

    @staticmethod
    def _status_specs(
        workspace: WorkspaceTreeWorkspace,
    ) -> tuple[WorkspaceTreeNodeData, ...]:
        specs: list[WorkspaceTreeNodeData] = []
        workspace_id = workspace.workspace_id
        if workspace.loading:
            specs.append(
                WorkspaceTreeNodeData.auxiliary(
                    "status",
                    workspace_id,
                    f"status:{workspace_id}:loading",
                    "Loading…",
                )
            )
        if workspace.error:
            specs.append(
                WorkspaceTreeNodeData.auxiliary(
                    "status",
                    workspace_id,
                    f"status:{workspace_id}:error",
                    workspace.error,
                )
            )
            specs.append(
                WorkspaceTreeNodeData.auxiliary(
                    "retry",
                    workspace_id,
                    f"action:{workspace_id}:retry",
                    "Retry",
                )
            )
        if (
            not workspace.conversations
            and not workspace.loading
            and not workspace.error
        ):
            specs.append(
                WorkspaceTreeNodeData.auxiliary(
                    "status",
                    workspace_id,
                    f"status:{workspace_id}:empty",
                    "No conversations",
                )
            )
        if workspace.next_cursor is not None and not workspace.loading:
            specs.append(
                WorkspaceTreeNodeData.auxiliary(
                    "load-more",
                    workspace_id,
                    f"action:{workspace_id}:load-more",
                    "Load more…",
                )
            )
        return tuple(specs)

    def sync_projection(
        self,
        workspaces: Iterable[WorkspaceTreeWorkspace],
        *,
        expanded_workspace_ids: set[str] | frozenset[str],
    ) -> None:
        """Apply a keyed immutable projection without clearing the native Tree."""

        projection = tuple(workspaces)
        self._preferred_expanded_workspace_ids = set(expanded_workspace_ids)
        self._syncing = True
        try:
            wanted_workspace_ids = {workspace.workspace_id for workspace in projection}
            for workspace_id in tuple(self.workspace_nodes):
                if workspace_id not in wanted_workspace_ids:
                    self._remove_workspace(workspace_id)

            wanted_conversations = {
                conversation.conversation_id
                for workspace in projection
                for conversation in workspace.conversations
            }
            for conversation_id in tuple(self.conversation_nodes):
                if conversation_id not in wanted_conversations:
                    node = self.conversation_nodes.pop(conversation_id)
                    node.remove()

            wanted_auxiliary: set[str] = set()
            for workspace_index, workspace in enumerate(projection):
                workspace_node = self.workspace_nodes.get(workspace.workspace_id)
                workspace_data = WorkspaceTreeNodeData.workspace(
                    workspace.workspace_id, workspace.label
                )
                if workspace_node is None:
                    workspace_node = self.root.add(
                        self._workspace_label(workspace),
                        workspace_data,
                        before=workspace_index,
                        expand=workspace.workspace_id in expanded_workspace_ids,
                    )
                    self.workspace_nodes[workspace.workspace_id] = workspace_node
                else:
                    workspace_node.data = workspace_data
                    workspace_node.set_label(self._workspace_label(workspace))
                    self._move_node_preserving_identity(
                        workspace_node, self.root, workspace_index
                    )

                if not self._search_active:
                    should_expand = workspace.workspace_id in expanded_workspace_ids
                    if should_expand and workspace_node.is_collapsed:
                        workspace_node.expand()
                    elif not should_expand and workspace_node.is_expanded:
                        workspace_node.collapse()

                for conversation_index, conversation in enumerate(
                    workspace.conversations
                ):
                    data = WorkspaceTreeNodeData.conversation(
                        workspace.workspace_id,
                        conversation.conversation_id,
                        conversation.title,
                        starred=conversation.starred,
                        selected=conversation.selected,
                    )
                    node = self.conversation_nodes.get(conversation.conversation_id)
                    if node is None:
                        node = workspace_node.add_leaf(
                            self._conversation_label(conversation),
                            data,
                            before=conversation_index,
                        )
                        self.conversation_nodes[conversation.conversation_id] = node
                    else:
                        node.data = data
                        node.set_label(self._conversation_label(conversation))
                        self._move_node_preserving_identity(
                            node, workspace_node, conversation_index
                        )

                specs = self._status_specs(workspace)
                for offset, data in enumerate(specs, len(workspace.conversations)):
                    wanted_auxiliary.add(data.key)
                    node = self.auxiliary_nodes.get(data.key)
                    if node is None:
                        node = workspace_node.add_leaf(
                            Text(_single_physical_row(data.raw_label)),
                            data,
                            before=offset,
                        )
                        self.auxiliary_nodes[data.key] = node
                    else:
                        node.data = data
                        node.set_label(Text(_single_physical_row(data.raw_label)))
                        self._move_node_preserving_identity(
                            node, workspace_node, offset
                        )

            for key in tuple(self.auxiliary_nodes):
                if key not in wanted_auxiliary:
                    self.auxiliary_nodes.pop(key).remove()
        finally:
            self._syncing = False
        self.can_focus = any(
            node.data is not None and node.data.selectable
            for node in (
                *self.workspace_nodes.values(),
                *self.conversation_nodes.values(),
                *self.auxiliary_nodes.values(),
            )
        )
        self._update_tooltip()

    def render_label(
        self,
        node: TreeNode[WorkspaceTreeNodeData],
        base_style: Style,
        style: Style,
    ) -> Text:
        """Render one literal row with an end ellipsis inside the Tree width."""

        label = super().render_label(node, base_style, style)
        guide_levels = 0
        ancestor: TreeNode[WorkspaceTreeNodeData] | None = node
        while ancestor is not None and ancestor is not self.root:
            guide_levels += 1
            ancestor = ancestor.parent
        label.truncate(
            max(1, self.size.width - (guide_levels * 3)),
            overflow="ellipsis",
        )
        return label

    @property
    def preferred_expanded_workspace_ids(self) -> frozenset[str]:
        """Return current non-search disclosure preferences."""

        return frozenset(self._preferred_expanded_workspace_ids)

    @property
    def search_active(self) -> bool:
        """Whether disclosure gestures are currently transient search state."""

        return self._search_active

    def on_unmount(self) -> None:
        """Drop owner maps when Textual retires this Tree instance."""

        self.workspace_nodes.clear()
        self.conversation_nodes.clear()
        self.auxiliary_nodes.clear()

    def _remove_workspace(self, workspace_id: str) -> None:
        node = self.workspace_nodes.pop(workspace_id)
        for conversation_id, conversation_node in tuple(
            self.conversation_nodes.items()
        ):
            if conversation_node.parent is node:
                self.conversation_nodes.pop(conversation_id)
        for key, auxiliary_node in tuple(self.auxiliary_nodes.items()):
            if auxiliary_node.parent is node:
                self.auxiliary_nodes.pop(key)
        node.remove()

    def _move_node_preserving_identity(
        self,
        node: TreeNode[WorkspaceTreeNodeData],
        parent: TreeNode[WorkspaceTreeNodeData],
        index: int,
    ) -> None:
        """Version-pinned Textual 8.2.8 reparent/reorder primitive.

        Textual has no public move API. Public remove/add unregisters the old
        object and loses cursor identity, so ADR-081 permits this one isolated
        private-shape operation. Any version or shape mismatch fails closed.
        """

        required = (
            textual.__version__ == _TEXTUAL_PRIVATE_MOVE_VERSION
            and hasattr(self, "_tree_nodes")
            and hasattr(self, "_invalidate")
            and hasattr(node, "_parent")
            and hasattr(node, "_children")
            and hasattr(parent, "_children")
            and node.id in self._tree_nodes
            and self._tree_nodes[node.id] is node
        )
        if not required:
            raise RuntimeError(
                "Console workspace Tree move requires exact Textual 8.2.8 private shape"
            )
        old_parent = node._parent
        if old_parent is None or not hasattr(old_parent, "_children"):
            raise RuntimeError(
                "Console workspace Tree move requires exact Textual 8.2.8 private shape"
            )
        old_index = old_parent._children.index(node)
        bounded_index = max(0, min(index, len(parent._children)))
        if old_parent is parent and old_index == bounded_index:
            return
        old_parent._children.pop(old_index)
        if old_parent is parent and old_index < bounded_index:
            bounded_index -= 1
        parent._children.insert(bounded_index, node)
        node._parent = parent
        self._invalidate()

    def set_search_active(
        self,
        active: bool,
        *,
        forced_workspace_ids: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        """Enter/leave transient search disclosure without writing preferences."""

        if active and not self._search_active:
            self._search_expansion_snapshot = frozenset(
                workspace_id
                for workspace_id, node in self.workspace_nodes.items()
                if node.is_expanded
            )
            self._search_workspace_ids_snapshot = frozenset(self.workspace_nodes)
        if active:
            self._search_active = True
            for workspace_id in forced_workspace_ids:
                node = self.workspace_nodes.get(workspace_id)
                if node is not None and node.is_collapsed:
                    node.expand()
            return
        if not self._search_active:
            return
        snapshot = self._search_expansion_snapshot or frozenset()
        original_workspace_ids = self._search_workspace_ids_snapshot or frozenset()
        self._search_active = False
        self._syncing = True
        try:
            for workspace_id, node in self.workspace_nodes.items():
                is_new_workspace = workspace_id not in original_workspace_ids
                should_expand = workspace_id in snapshot or is_new_workspace
                if is_new_workspace:
                    self._preferred_expanded_workspace_ids.add(workspace_id)
                if should_expand and node.is_collapsed:
                    node.expand()
                elif not should_expand and node.is_expanded:
                    node.collapse()
        finally:
            self._syncing = False
        self._search_expansion_snapshot = None
        self._search_workspace_ids_snapshot = None

    def on_tree_node_selected(
        self, event: Tree.NodeSelected[WorkspaceTreeNodeData]
    ) -> None:
        data = event.node.data
        if data is None or not data.selectable:
            event.stop()
            return
        if data.kind == "workspace" and data.workspace_id is not None:
            self.post_message(WorkspaceTreeWorkspaceSelected(data.workspace_id))
        elif (
            data.kind == "conversation"
            and data.workspace_id is not None
            and data.conversation_id is not None
        ):
            self.post_message(
                WorkspaceTreeConversationSelected(
                    data.workspace_id, data.conversation_id
                )
            )
        elif data.kind == "load-more" and data.workspace_id is not None:
            self.post_message(WorkspaceTreeLoadMoreRequested(data.workspace_id))
        elif data.kind == "retry" and data.workspace_id is not None:
            self.post_message(WorkspaceTreeRetryRequested(data.workspace_id))
        event.stop()

    def on_tree_node_expanded(
        self, event: Tree.NodeExpanded[WorkspaceTreeNodeData]
    ) -> None:
        self._record_expansion_gesture(event.node)

    def on_tree_node_collapsed(
        self, event: Tree.NodeCollapsed[WorkspaceTreeNodeData]
    ) -> None:
        node = event.node
        cursor = self.cursor_node
        ancestor = cursor.parent if cursor is not None else None
        cursor_was_descendant = False
        while ancestor is not None:
            if ancestor is node:
                cursor_was_descendant = True
                break
            ancestor = ancestor.parent
        if cursor_was_descendant:
            self.move_cursor(node)
        self._record_expansion_gesture(node)

    def _record_expansion_gesture(self, node: TreeNode[WorkspaceTreeNodeData]) -> None:
        data = node.data
        if (
            self._syncing
            or self._search_active
            or data is None
            or data.kind != "workspace"
            or data.workspace_id is None
        ):
            return
        if node.is_expanded:
            self._preferred_expanded_workspace_ids.add(data.workspace_id)
        else:
            self._preferred_expanded_workspace_ids.discard(data.workspace_id)
        callback = self.expansion_preferences_changed
        if callback is not None:
            callback(frozenset(self._preferred_expanded_workspace_ids))
        self.post_message(
            WorkspaceTreeExpansionChanged(
                data.workspace_id,
                expanded=node.is_expanded,
            )
        )

    def action_workspace_left(self) -> None:
        node = self.cursor_node
        if node is None or node is self.root:
            return
        data = node.data
        if data is not None and data.kind == "workspace":
            if node.is_expanded:
                node.collapse()
            return
        parent = node.parent
        if parent is not None and parent is not self.root:
            self.move_cursor(parent)

    def action_workspace_right(self) -> None:
        node = self.cursor_node
        if node is None or node is self.root or not node.allow_expand:
            return
        if node.is_collapsed:
            node.expand()

    def action_workspace_star(self) -> None:
        node = self.cursor_node
        data = node.data if node is not None else None
        if (
            data is None
            or data.kind != "conversation"
            or not self.star_enabled
            or data.workspace_id is None
            or data.conversation_id is None
        ):
            return
        self.post_message(
            WorkspaceTreeStarRequested(
                data.workspace_id,
                data.conversation_id,
                starred=data.starred,
            )
        )

    def action_cursor_parent(self) -> None:
        node = self.cursor_node
        if node is None or node.parent is None or node.parent is self.root:
            return
        super().action_cursor_parent()

    def action_cursor_parent_next_sibling(self) -> None:
        node = self.cursor_node
        if node is None or node.parent is None:
            return
        target = node.parent.next_sibling
        if target is not None and target is not self.root:
            self.move_cursor(target, animate=True)

    def action_cursor_previous_sibling(self) -> None:
        node = self.cursor_node
        if node is None or node is self.root:
            return
        target = node.previous_sibling
        if target is None:
            parent = node.parent
            if parent is None or parent is self.root:
                return
            target = parent
        self.move_cursor(target, animate=True)

    def action_cursor_next_sibling(self) -> None:
        node = self.cursor_node
        if node is None or node is self.root:
            return
        target = node.next_sibling
        if target is None and node.parent is not None:
            target = node.parent.next_sibling
        if target is not None and target is not self.root:
            self.move_cursor(target, animate=True)

    def watch_cursor_line(self, old_value: int, new_value: int) -> None:
        super().watch_cursor_line(old_value, new_value)
        self._update_tooltip()

    def watch_hover_line(self, old_value: int, new_value: int) -> None:
        super().watch_hover_line(old_value, new_value)
        self._update_tooltip()

    def _update_tooltip(self) -> None:
        line = self.hover_line if self.hover_line >= 0 else self.cursor_line
        node = self.get_node_at_line(line)
        self.tooltip = node.data.raw_label if node is not None and node.data else None


__all__ = [
    "ConsoleWorkspaceTree",
    "WorkspaceTreeConversationSelected",
    "WorkspaceTreeExpansionChanged",
    "WorkspaceTreeLoadMoreRequested",
    "WorkspaceTreeNodeData",
    "WorkspaceTreeRetryRequested",
    "WorkspaceTreeStarRequested",
    "WorkspaceTreeWorkspaceSelected",
]
