# branch_tree_view.py
# Description: Widget for visualizing conversation branches in a tree structure
#
# Imports
from typing import Dict, List, Any, Optional, Callable
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, Static, Tree
from textual.tree import TreeNode
from textual.message import Message
from rich.text import Text
#
# Local Imports
from ..Utils.Emoji_Handling import get_char

# Configure logger with context
logger = logger.bind(module="branch_tree_view")

#
#######################################################################################################################
#
# Classes:

class BranchSelectedMessage(Message):
    """Message emitted when a branch is selected in the tree."""
    def __init__(self, branch_id: str, branch_info: Dict[str, Any]) -> None:
        super().__init__()
        self.branch_id = branch_id
        self.branch_info = branch_info


class BranchTreeView(Widget):
    """
    A tree view widget for displaying conversation branches.
    Shows the hierarchical structure of conversation branches.
    """
    
    DEFAULT_CSS = """
    BranchTreeView {
        width: 100%;
        height: 100%;
        border: round $surface;
        background: $panel;
    }
    
    BranchTreeView > Vertical {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    
    .branch-tree-header {
        width: 100%;
        height: 3;
        padding: 0 1;
        background: $surface-darken-1;
        text-style: bold;
        align: center middle;
    }
    
    .branch-tree-container {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
    }
    
    .branch-item {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    
    .branch-item-current {
        background: $accent;
        color: $text;
        text-style: bold;
    }
    
    .branch-item:hover {
        background: $surface-lighten-1;
    }
    
    .branch-controls {
        width: 100%;
        height: 3;
        layout: horizontal;
        align: center middle;
        margin-top: 1;
    }
    
    .branch-controls Button {
        min-width: 10;
        margin: 0 1;
    }
    """
    
    # Reactive properties
    current_branch_id = reactive(None)
    branches = reactive({})
    
    def __init__(
        self,
        branch_data: Optional[Dict[str, Any]] = None,
        on_branch_selected: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.branch_data = branch_data or {}
        self.on_branch_selected_callback = on_branch_selected
        self._tree_widget = None
        logger.debug("BranchTreeView initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the branch tree view UI."""
        with Vertical():
            yield Static("🌳 Conversation Branches", classes="branch-tree-header")
            
            with VerticalScroll(classes="branch-tree-container"):
                tree = Tree("Conversation Tree", id="branch-tree")
                tree.show_root = False
                self._tree_widget = tree
                yield tree
            
            with Horizontal(classes="branch-controls"):
                yield Button("↻ Refresh", id="refresh-branches", variant="primary")
                yield Button("🔀 New Branch", id="create-branch", variant="success")
                yield Button("📊 Compare", id="compare-branches", variant="default")
    
    def on_mount(self) -> None:
        """Called when widget is mounted."""
        if self.branch_data:
            self.update_branch_tree(self.branch_data)
    
    def update_branch_tree(self, branch_info: Dict[str, Any]) -> None:
        """
        Update the tree view with branch information.
        
        Args:
            branch_info: Dictionary containing branch structure information
        """
        self.branch_data = branch_info
        
        if not self._tree_widget:
            logger.warning("Tree widget not initialized")
            return
            
        # Clear existing tree
        self._tree_widget.clear()
        
        # Get root conversation info
        root_id = branch_info.get('root_id')
        current_id = branch_info.get('conversation_id')
        
        if not root_id:
            logger.warning("No root_id in branch info")
            return
        
        # Build tree structure
        self._build_tree_from_branches(branch_info)
        
        # Highlight current branch
        self.current_branch_id = current_id
    
    def _build_tree_from_branches(self, branch_info: Dict[str, Any]) -> None:
        """Build the tree structure from branch information."""
        all_branches = branch_info.get('all_branches', [])
        if not all_branches:
            # If no all_branches, create a simple structure
            self._create_simple_tree(branch_info)
            return
        
        # Create a map of conversations by ID
        conv_map = {conv['id']: conv for conv in all_branches}
        
        # Find root conversation
        root_id = branch_info.get('root_id')
        root_conv = conv_map.get(root_id)
        
        if not root_conv:
            logger.warning(f"Root conversation {root_id} not found in branches")
            return
        
        # Add root node
        root_label = self._create_branch_label(root_conv, is_root=True)
        root_node = self._tree_widget.root.add(root_label)
        root_node.data = root_conv
        
        # Build tree recursively
        self._add_children_to_node(root_node, root_id, conv_map)
        
        # Expand important nodes
        root_node.expand()
        self._expand_to_current(root_node, branch_info.get('conversation_id'))
    
    def _create_simple_tree(self, branch_info: Dict[str, Any]) -> None:
        """Create a simple tree when full branch data isn't available."""
        current_id = branch_info.get('conversation_id')
        
        # Add current conversation
        current_label = Text("📍 Current Conversation", style="bold cyan")
        current_node = self._tree_widget.root.add(current_label)
        current_node.data = {'id': current_id}
        
        # Add siblings if any
        siblings = branch_info.get('siblings', [])
        if siblings:
            siblings_label = Text("📁 Sibling Branches", style="dim")
            siblings_node = self._tree_widget.root.add(siblings_label)
            
            for sibling in siblings:
                sib_label = self._create_branch_label(sibling)
                sib_node = siblings_node.add(sib_label)
                sib_node.data = sibling
            
            siblings_node.expand()
        
        # Add children if any
        children = branch_info.get('children', [])
        if children:
            children_label = Text("📂 Child Branches", style="dim")
            children_node = self._tree_widget.root.add(children_label)
            
            for child in children:
                child_label = self._create_branch_label(child)
                child_node = children_node.add(child_label)
                child_node.data = child
            
            children_node.expand()
    
    def _create_branch_label(
        self, 
        conv: Dict[str, Any], 
        is_root: bool = False,
        is_current: bool = False
    ) -> Text:
        """Create a formatted label for a branch node."""
        title = conv.get('title', 'Untitled')
        conv_id = conv.get('id', '')
        
        # Truncate long titles
        if len(title) > 40:
            title = title[:37] + "..."
        
        # Add indicators
        indicators = []
        if is_root:
            indicators.append("🌱")
        if is_current or conv_id == self.current_branch_id:
            indicators.append("📍")
        if conv.get('forked_from_message_id'):
            indicators.append("🔀")
        
        # Format label
        indicator_str = " ".join(indicators)
        label = f"{indicator_str} {title}".strip()
        
        # Apply styling
        style = "bold cyan" if is_current or conv_id == self.current_branch_id else ""
        
        return Text(label, style=style)
    
    def _add_children_to_node(
        self, 
        parent_node: TreeNode,
        parent_id: str,
        conv_map: Dict[str, Dict[str, Any]]
    ) -> None:
        """Recursively add child branches to a node."""
        # Find all conversations that have this parent
        children = [
            conv for conv in conv_map.values()
            if conv.get('parent_conversation_id') == parent_id
        ]
        
        # Sort by creation date
        children.sort(key=lambda x: x.get('created_at', ''))
        
        for child in children:
            child_label = self._create_branch_label(
                child,
                is_current=child['id'] == self.current_branch_id
            )
            child_node = parent_node.add(child_label)
            child_node.data = child
            
            # Recursively add children
            self._add_children_to_node(child_node, child['id'], conv_map)
    
    def _expand_to_current(self, node: TreeNode, current_id: str) -> bool:
        """Expand tree nodes to show the current conversation."""
        if not node.data:
            return False
            
        # Check if this is the current node
        if node.data.get('id') == current_id:
            node.expand()
            return True
        
        # Check children
        for child in node.children:
            if self._expand_to_current(child, current_id):
                node.expand()
                return True
        
        return False
    
    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        node = event.node
        
        if not node.data or not isinstance(node.data, dict):
            return
        
        branch_id = node.data.get('id')
        if not branch_id:
            return
        
        logger.debug(f"Branch selected: {branch_id}")
        
        # Emit custom message
        self.post_message(BranchSelectedMessage(branch_id, node.data))
        
        # Call callback if provided
        if self.on_branch_selected_callback:
            await self.on_branch_selected_callback(branch_id, node.data)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in the branch controls."""
        button_id = event.button.id
        
        if button_id == "refresh-branches":
            # Emit message to refresh branch data
            logger.debug("Refresh branches requested")
            # Parent widget should handle this
            
        elif button_id == "create-branch":
            # Emit message to create new branch
            logger.debug("Create branch requested")
            # Parent widget should handle this
            
        elif button_id == "compare-branches":
            # Emit message to compare branches
            logger.debug("Compare branches requested")
            # Parent widget should handle this
    
    def highlight_branch(self, branch_id: str) -> None:
        """Highlight a specific branch in the tree."""
        self.current_branch_id = branch_id
        
        # Update tree labels
        if self._tree_widget and self.branch_data:
            self.update_branch_tree(self.branch_data)
    
    def get_selected_branch(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected branch data."""
        if not self._tree_widget:
            return None
            
        # Get highlighted node
        highlighted = self._tree_widget.cursor_node
        if highlighted and highlighted.data:
            return highlighted.data
        
        return None


class CompactBranchIndicator(Widget):
    """A compact widget that shows branch status inline."""
    
    DEFAULT_CSS = """
    CompactBranchIndicator {
        width: auto;
        height: 1;
        margin: 0 1;
    }
    
    .branch-indicator {
        color: $text-muted;
    }
    
    .has-branches {
        color: $accent;
        text-style: bold;
    }
    """
    
    def __init__(
        self,
        branch_count: int = 0,
        is_branch_point: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.branch_count = branch_count
        self.is_branch_point = is_branch_point
    
    def compose(self) -> ComposeResult:
        """Compose the branch indicator."""
        if self.is_branch_point and self.branch_count > 1:
            text = f"🔀 {self.branch_count} branches"
            classes = "branch-indicator has-branches"
        elif self.is_branch_point:
            text = "🔀 Branch point"
            classes = "branch-indicator has-branches"
        else:
            text = ""
            classes = "branch-indicator"
        
        yield Static(text, classes=classes)


#
# End of branch_tree_view.py
#######################################################################################################################