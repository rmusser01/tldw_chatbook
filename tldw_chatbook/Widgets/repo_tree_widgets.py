# tldw_chatbook/Widgets/repo_tree_widgets.py
# Description: Custom tree widgets for GitHub repository file selection
#
# This module provides tree widgets for displaying and selecting files
# from GitHub repositories with expand/collapse and checkbox functionality.

from __future__ import annotations
from typing import Optional, Dict, List, Set, Callable
import os
from pathlib import Path

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Static

logger = logger.bind(module="repo_tree_widgets")


class TreeNodeSelected(Message):
    """Message sent when a tree node is selected."""
    def __init__(self, path: str, selected: bool) -> None:
        self.path = path
        self.selected = selected
        super().__init__()


class TreeNodeExpanded(Message):
    """Message sent when a tree node is expanded."""
    def __init__(self, path: str, expanded: bool) -> None:
        self.path = path
        self.expanded = expanded
        super().__init__()


class TreeNode(Widget):
    """A single node in the repository tree view."""
    
    DEFAULT_CSS = """
    TreeNode {
        height: 3;
        width: 100%;
    }
    
    .tree-node-row {
        layout: grid;
        grid-size: 4 1;
        grid-columns: auto 3 3 1fr;
        height: 3;
        align: left middle;
        width: 100%;
    }
    
    .tree-indent {
        width: auto;
        content-align: left middle;
    }
    
    .tree-expand-btn {
        width: 3;
        height: 3;
        min-width: 3;
        background: transparent;
        border: none;
        padding: 0;
    }
    
    .tree-expand-btn:hover {
        background: $primary 20%;
    }
    
    .tree-expand-spacer {
        width: 3;
        height: 3;
    }
    
    .tree-checkbox {
        width: 3;
        height: 3;
    }
    
    .tree-content {
        padding-left: 1;
        width: 100%;
        content-align: left middle;
    }
    
    .tree-node-row:hover {
        background: $panel-lighten-1;
    }
    
    .tree-node-selected {
        background: $accent 20%;
    }
    """
    
    def __init__(
        self,
        path: str,
        name: str,
        is_directory: bool,
        level: int = 0,
        size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.path = path
        self.node_name = name  # Changed from 'name' to avoid conflict with Widget.name property
        self.is_directory = is_directory
        self.level = level
        self.file_size = size  # Changed from 'size' to avoid conflict with Widget.size property
        self.expanded = reactive(False)
        self.selected = reactive(False)
        self.children_loaded = False
        
    def compose(self) -> ComposeResult:
        """Compose the tree node UI."""
        with Container(classes="tree-node-row"):
            # Column 1: Indentation
            yield Static(" " * (self.level * 2), classes="tree-indent")
            
            # Column 2: Expand/collapse button
            if self.is_directory:
                yield Button(
                    "▶",
                    classes="tree-expand-btn",
                    id=f"expand-{self.path}"
                )
            else:
                yield Static("", classes="tree-expand-spacer")
            
            # Column 3: Checkbox
            yield Checkbox(
                value=self.selected,
                id=f"select-{self.path}",
                classes="tree-checkbox"
            )
            
            # Column 4: Icon and name
            icon = self._get_icon()
            size_text = f" ({self._format_size(self.file_size)})" if self.file_size else ""
            yield Static(
                f"{icon} {self.node_name}{size_text}",
                classes="tree-content"
            )
    
    def _get_icon(self) -> str:
        """Get the appropriate icon for the file/folder."""
        if self.is_directory:
            return "📁" if not self.expanded else "📂"
        
        # Special case for dotfiles
        if self.node_name == '.gitignore':
            return '🚫'
        elif self.node_name == '.env':
            return '🔐'
        
        # File type based icons
        ext = os.path.splitext(self.node_name)[1].lower()
        icon_map = {
            '.py': '🐍',
            '.js': '📜',
            '.jsx': '⚛️',
            '.ts': '📘',
            '.tsx': '⚛️',
            '.md': '📝',
            '.json': '📊',
            '.yaml': '⚙️',
            '.yml': '⚙️',
            '.txt': '📄',
            '.html': '🌐',
            '.css': '🎨',
            '.png': '🖼️',
            '.jpg': '🖼️',
            '.jpeg': '🖼️',
            '.gif': '🖼️',
            '.svg': '🖼️',
            '.pdf': '📑',
            '.zip': '📦',
        }
        
        return icon_map.get(ext, '📄')
    
    def _format_size(self, size: Optional[int]) -> str:
        """Format file size in human-readable format."""
        if size is None:
            return ""
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    @on(Button.Pressed, ".tree-expand-btn")
    def handle_expand(self, event: Button.Pressed) -> None:
        """Handle expand/collapse button press."""
        self.expanded = not self.expanded
        button = self.query_one(".tree-expand-btn", Button)
        button.label = "▼" if self.expanded else "▶"
        
        # Update icon for directories
        content = self.query_one(".tree-content", Static)
        icon = self._get_icon()
        size_text = f" ({self._format_size(self.size)})" if self.size else ""
        content.update(f"{icon} {self.node_name}{size_text}")
        
        # Post message for parent to handle
        self.post_message(TreeNodeExpanded(self.path, self.expanded))
    
    @on(Checkbox.Changed, ".tree-checkbox")
    def handle_selection(self, event: Checkbox.Changed) -> None:
        """Handle checkbox selection change."""
        self.selected = event.value
        self.post_message(TreeNodeSelected(self.path, self.selected))
        
        # Update visual state
        container = self.query_one(".tree-node-row", Container)
        if self.selected:
            container.add_class("tree-node-selected")
        else:
            container.remove_class("tree-node-selected")


class TreeView(VerticalScroll):
    """Scrollable tree view for repository files."""
    
    DEFAULT_CSS = """
    TreeView {
        height: 100%;
        width: 100%;
        overflow-y: auto;
        overflow-x: hidden;
        border: solid $primary;
        padding: 1;
    }
    
    .tree-container {
        width: 100%;
        height: auto;
    }
    
    .tree-loading {
        width: 100%;
        text-align: center;
        padding: 2;
        color: $text-muted;
    }
    
    .tree-empty {
        width: 100%;
        text-align: center;
        padding: 4;
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        on_selection_change: Optional[Callable[[str, bool], None]] = None,
        on_node_expanded: Optional[Callable[[str, bool], None]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nodes: Dict[str, TreeNode] = {}
        self.selection: Set[str] = set()
        self.on_selection_change = on_selection_change
        self.on_node_expanded = on_node_expanded
        self._tree_data: Optional[List[Dict]] = None
        
    def compose(self) -> ComposeResult:
        """Compose the tree view."""
        with Container(classes="tree-container", id="tree-container"):
            yield Static("No repository loaded", classes="tree-empty")
    
    async def load_tree(self, tree_data: List[Dict]) -> None:
        """Load tree data and build the view."""
        self._tree_data = tree_data
        container = self.query_one("#tree-container", Container)
        
        # Clear existing content
        await container.remove_children()
        self.nodes.clear()
        self.selection.clear()
        
        if not tree_data:
            await container.mount(Static("No files found", classes="tree-empty"))
            return
        
        # Show loading indicator
        loading = Static("Loading repository structure...", classes="tree-loading")
        await container.mount(loading)
        
        # Build tree nodes
        await self._build_tree_nodes(tree_data, container, level=0)
        
        # Remove loading indicator
        await loading.remove()
    
    async def _build_tree_nodes(
        self,
        items: List[Dict],
        parent_container: Container,
        level: int = 0,
        parent_path: str = ""
    ) -> None:
        """Recursively build tree nodes."""
        # Sort items: directories first, then by name
        sorted_items = sorted(
            items,
            key=lambda x: (not x.get('type') == 'tree', x.get('name', '').lower())
        )
        
        for item in sorted_items:
            name = item.get('name', '')
            path = item.get('path', name)
            is_dir = item.get('type') == 'tree'
            size = item.get('size')
            
            # Create tree node
            node = TreeNode(
                path=path,
                name=name,
                is_directory=is_dir,
                level=level,
                size=size,
                id=f"node-{path}"
            )
            
            self.nodes[path] = node
            await parent_container.mount(node)
            
            # If item has children and is expanded, add them
            if is_dir and item.get('children'):
                # We'll handle expansion dynamically
                pass
    
    async def expand_node(self, path: str, children: List[Dict]) -> None:
        """Expand a node and add its children."""
        node = self.nodes.get(path)
        if not node or not node.is_directory:
            return
        
        # Find the position to insert children
        container = self.query_one("#tree-container", Container)
        node_index = container.children.index(node)
        
        # Build child nodes
        child_nodes = []
        for child in children:
            child_name = child.get('name', '')
            child_path = child.get('path', child_name)
            is_dir = child.get('type') == 'tree'
            size = child.get('size')
            
            child_node = TreeNode(
                path=child_path,
                name=child_name,
                is_directory=is_dir,
                level=node.level + 1,
                size=size,
                id=f"node-{child_path}"
            )
            
            self.nodes[child_path] = child_node
            child_nodes.append(child_node)
        
        # Insert children after the parent node
        for i, child_node in enumerate(child_nodes):
            await container.mount(child_node, after=node_index + i)
        
        node.children_loaded = True
    
    async def collapse_node(self, path: str) -> None:
        """Collapse a node and remove its children."""
        node = self.nodes.get(path)
        if not node or not node.is_directory:
            return
        
        # Find all child nodes to remove
        container = self.query_one("#tree-container", Container)
        nodes_to_remove = []
        
        for child_path, child_node in self.nodes.items():
            if child_path.startswith(path + '/'):
                nodes_to_remove.append(child_node)
        
        # Remove child nodes
        for child_node in nodes_to_remove:
            await child_node.remove()
            del self.nodes[child_node.path]
            self.selection.discard(child_node.path)
    
    def select_node(self, path: str, selected: bool) -> None:
        """Update node selection."""
        if selected:
            self.selection.add(path)
        else:
            self.selection.discard(path)
        
        # Update the node's selected state
        node = self.nodes.get(path)
        if node:
            node.selected = selected
        
        # If it's a directory, cascade to children
        if node and node.is_directory:
            for child_path, child_node in self.nodes.items():
                if child_path.startswith(path + '/'):
                    child_node.selected = selected
                    if selected:
                        self.selection.add(child_path)
                    else:
                        self.selection.discard(child_path)
    
    def get_selected_files(self) -> List[str]:
        """Get list of selected file paths."""
        return [
            path for path in self.selection
            if path in self.nodes and not self.nodes[path].is_directory
        ]
    
    def get_selection_stats(self) -> Dict[str, int]:
        """Get statistics about current selection."""
        total_files = 0
        total_size = 0
        
        for path in self.selection:
            node = self.nodes.get(path)
            if node and not node.is_directory:
                total_files += 1
                if node.file_size:
                    total_size += node.file_size
        
        return {
            'files': total_files,
            'size': total_size
        }
    
    @on(TreeNodeSelected)
    def handle_node_selection(self, event: TreeNodeSelected) -> None:
        """Handle node selection events."""
        self.select_node(event.path, event.selected)
        
        if self.on_selection_change:
            self.on_selection_change(event.path, event.selected)
    
    @on(TreeNodeExpanded)
    def handle_node_expansion(self, event: TreeNodeExpanded) -> None:
        """Handle node expansion events."""
        if self.on_node_expanded:
            self.on_node_expanded(event.path, event.expanded)