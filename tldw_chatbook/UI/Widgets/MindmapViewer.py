# MindmapViewer.py
# Description: Interactive mindmap viewer widget for Textual
#
"""
Mindmap Viewer Widget
--------------------

Interactive mindmap viewer with:
- Multiple view modes (tree, outline, ASCII)
- Keyboard navigation with vim bindings
- Search and filter functionality
- Export capabilities
- Integration with tldw_chatbook content
"""

from typing import Optional, Dict, List, Any
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Input, Button, Checkbox, Label
from textual.reactive import reactive
from textual.message import Message
from textual.worker import Worker, WorkerState
from loguru import logger
from anytree import Node

from ...Tools.Mind_Map.mindmap_model import MindmapModel
from ...Tools.Mind_Map.mindmap_renderer import ThemedMindmapRenderer
from ...Tools.Mind_Map.mermaid_parser import ExtendedMermaidParser


class MindmapNodeSelected(Message):
    """Message sent when a mindmap node is selected"""
    def __init__(self, node: Node) -> None:
        super().__init__()
        self.node = node


class MindmapViewer(Container):
    """Interactive mindmap viewer widget"""
    
    DEFAULT_CSS = """
    MindmapViewer {
        layout: vertical;
        height: 100%;
    }
    
    .mindmap-header {
        height: auto;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
        margin-bottom: 1;
    }
    
    .mindmap-controls {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    .mindmap-button {
        margin-right: 1;
        min-width: 12;
    }
    
    .view-mode-selector {
        layout: horizontal;
        height: 3;
        align: right top;
    }
    
    .view-mode-button {
        margin-left: 1;
        width: 10;
    }
    
    #search-container {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
        display: none;
    }
    
    #search-container.visible {
        display: block;
    }
    
    #search-input {
        width: 1fr;
        margin-right: 1;
    }
    
    #mindmap-display {
        height: 1fr;
        border: round $background-darken-1;
        background: $boost;
        padding: 1;
        overflow-y: scroll;
    }
    
    #mindmap-content {
        width: 100%;
        height: auto;
    }
    
    #status-bar {
        height: 2;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }
    
    #aria-live {
        display: none;
    }
    
    .search-nav-button {
        width: 8;
        margin-left: 1;
    }
    """
    
    BINDINGS = [
        # Standard navigation
        Binding("up", "move_up", "Move up"),
        Binding("down", "move_down", "Move down"),
        Binding("left", "collapse", "Collapse"),
        Binding("right", "expand", "Expand"),
        # Vim-style navigation
        Binding("h", "collapse", "Collapse", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("l", "expand", "Expand", show=False),
        # Node operations
        Binding("enter", "toggle", "Toggle"),
        Binding("space", "select", "Select", show=False),
        # Jump navigation
        Binding("g", "jump_top", "Jump to top"),
        Binding("G", "jump_bottom", "Jump to bottom"),
        Binding("p", "jump_parent", "Jump to parent"),
        Binding("f", "jump_first_child", "First child"),
        # View controls
        Binding("1", "view_tree", "Tree view"),
        Binding("2", "view_outline", "Outline view"),
        Binding("3", "view_ascii", "ASCII view"),
        # Search
        Binding("/", "search", "Search"),
        Binding("n", "next_match", "Next match"),
        Binding("N", "prev_match", "Previous match"),
        Binding("escape", "clear_search", "Clear search"),
        # Expand/Collapse all
        Binding("E", "expand_all", "Expand all"),
        Binding("C", "collapse_all", "Collapse all"),
        # Accessibility
        Binding("ctrl+a", "announce_position", "Announce position"),
        Binding("?", "show_help", "Show help"),
    ]
    
    # Reactive properties
    view_mode = reactive("tree", recompose=False)
    search_active = reactive(False, recompose=False)
    selected_node_text = reactive("", recompose=False)
    total_nodes = reactive(0, recompose=False)
    visible_nodes = reactive(0, recompose=False)
    
    def __init__(
        self,
        mermaid_code: Optional[str] = None,
        theme: str = "default",
        show_controls: bool = True,
        **kwargs
    ):
        """Initialize the mindmap viewer
        
        Args:
            mermaid_code: Optional Mermaid mindmap code to load
            theme: Color theme to use
            show_controls: Whether to show control buttons
        """
        super().__init__(**kwargs)
        self.model = MindmapModel()
        self.renderer = ThemedMindmapRenderer(self.model, theme)
        self.show_controls = show_controls
        self.search_index = 0
        self.announcement_enabled = True
        
        # Load initial content if provided
        if mermaid_code:
            try:
                self.model.load_from_mermaid(mermaid_code)
            except Exception as e:
                logger.error(f"Failed to load initial mindmap: {e}")
    
    def compose(self) -> ComposeResult:
        """Compose the mindmap viewer UI"""
        with Container(classes="mindmap-header"):
            if self.show_controls:
                # Main controls
                with Horizontal(classes="mindmap-controls"):
                    yield Button("◀ Collapse All", id="collapse-all", classes="mindmap-button")
                    yield Button("▶ Expand All", id="expand-all", classes="mindmap-button")
                    yield Button("🔍 Search", id="search-btn", classes="mindmap-button")
                    yield Button("↻ Refresh", id="refresh-btn", classes="mindmap-button")
                
                # View mode selector
                with Horizontal(classes="view-mode-selector"):
                    yield Label("View: ")
                    yield Button("Tree", id="view-tree", classes="view-mode-button", variant="primary")
                    yield Button("Outline", id="view-outline", classes="view-mode-button")
                    yield Button("ASCII", id="view-ascii", classes="view-mode-button")
            
            # Search bar (hidden by default)
            with Horizontal(id="search-container"):
                yield Input(
                    placeholder="Search nodes...",
                    id="search-input"
                )
                yield Button("Next", id="search-next", classes="search-nav-button")
                yield Button("Prev", id="search-prev", classes="search-nav-button")
                yield Button("✕", id="search-close", classes="search-nav-button", variant="error")
        
        # Main display area
        yield ScrollableContainer(
            Static(id="mindmap-content"),
            id="mindmap-display"
        )
        
        # Status bar
        yield Static("No mindmap loaded", id="status-bar")
        
        # Hidden ARIA live region for accessibility
        yield Static("", id="aria-live")
    
    def on_mount(self) -> None:
        """Initialize the display when mounted"""
        self.refresh_display()
        self._update_status_bar()
    
    def refresh_display(self) -> None:
        """Update the mindmap display"""
        content_widget = self.query_one("#mindmap-content", Static)
        
        if not self.model.root:
            content_widget.update("No mindmap loaded")
            return
        
        # Render based on current view mode
        if self.view_mode == "tree":
            rendered = self.renderer.render_tree_view()
        elif self.view_mode == "outline":
            rendered = self.renderer.render_outline_view()
        elif self.view_mode == "ascii":
            rendered = self.renderer.render_ascii_art()
        else:
            rendered = self.renderer.render_tree_view()
        
        content_widget.update(rendered)
        
        # Update reactive properties
        stats = self.model.get_statistics()
        self.total_nodes = stats['total_nodes']
        self.visible_nodes = stats['visible_nodes']
        if self.model.selected_node:
            self.selected_node_text = getattr(self.model.selected_node, 'text', self.model.selected_node.name)
    
    def _update_status_bar(self) -> None:
        """Update the status bar with current information"""
        if not self.model.root:
            status = "No mindmap loaded"
        else:
            status = f"Nodes: {self.visible_nodes}/{self.total_nodes}"
            
            if self.model.selected_node:
                status += f" | Selected: {self.selected_node_text[:30]}..."
                
            if self.model.search_results:
                status += f" | Search: {len(self.model.search_results)} matches"
        
        self.query_one("#status-bar", Static).update(status)
    
    def watch_view_mode(self, old_mode: str, new_mode: str) -> None:
        """React to view mode changes"""
        # Update button variants
        for mode in ["tree", "outline", "ascii"]:
            btn = self.query_one(f"#view-{mode}", Button)
            btn.variant = "primary" if mode == new_mode else "default"
        
        self.refresh_display()
    
    def watch_search_active(self, was_active: bool, is_active: bool) -> None:
        """React to search state changes"""
        search_container = self.query_one("#search-container")
        if is_active:
            search_container.add_class("visible")
            self.query_one("#search-input", Input).focus()
        else:
            search_container.remove_class("visible")
            self.focus()
    
    # Action handlers
    def action_move_up(self) -> None:
        """Move selection up"""
        if self.model.move_selection_up():
            self.refresh_display()
            self._update_status_bar()
            if self.announcement_enabled:
                self.action_announce_position()
    
    def action_move_down(self) -> None:
        """Move selection down"""
        if self.model.move_selection_down():
            self.refresh_display()
            self._update_status_bar()
            if self.announcement_enabled:
                self.action_announce_position()
    
    def action_expand(self) -> None:
        """Expand current node"""
        if self.model.selected_node:
            self.model.expand_node(self.model.selected_node)
            self.refresh_display()
            self._update_status_bar()
    
    def action_collapse(self) -> None:
        """Collapse current node"""
        if self.model.selected_node:
            self.model.collapse_node(self.model.selected_node)
            self.refresh_display()
            self._update_status_bar()
    
    def action_toggle(self) -> None:
        """Toggle current node expansion"""
        if self.model.selected_node:
            self.model.toggle_node(self.model.selected_node)
            self.refresh_display()
            self._update_status_bar()
    
    def action_select(self) -> None:
        """Select current node (emit message)"""
        if self.model.selected_node:
            self.post_message(MindmapNodeSelected(self.model.selected_node))
    
    def action_jump_top(self) -> None:
        """Jump to root node"""
        if self.model.root:
            self.model.selected_node = self.model.root
            self.refresh_display()
            self._update_status_bar()
    
    def action_jump_bottom(self) -> None:
        """Jump to last visible node"""
        visible = self.model.get_visible_nodes()
        if visible:
            self.model.selected_node = visible[-1]
            self.refresh_display()
            self._update_status_bar()
    
    def action_jump_parent(self) -> None:
        """Jump to parent node"""
        if self.model.jump_to_parent():
            self.refresh_display()
            self._update_status_bar()
    
    def action_jump_first_child(self) -> None:
        """Jump to first child"""
        if self.model.jump_to_first_child():
            self.refresh_display()
            self._update_status_bar()
    
    def action_expand_all(self) -> None:
        """Expand all nodes"""
        self.model.expand_all()
        self.refresh_display()
        self._update_status_bar()
    
    def action_collapse_all(self) -> None:
        """Collapse all nodes"""
        self.model.collapse_all()
        self.refresh_display()
        self._update_status_bar()
    
    def action_search(self) -> None:
        """Toggle search mode"""
        self.search_active = not self.search_active
    
    def action_clear_search(self) -> None:
        """Clear search and results"""
        self.search_active = False
        self.model.search("")
        self.search_index = 0
        self.refresh_display()
        self._update_status_bar()
    
    def action_next_match(self) -> None:
        """Jump to next search match"""
        if self.model.search_results:
            self.search_index = (self.search_index + 1) % len(self.model.search_results)
            self.model.selected_node = self.model.search_results[self.search_index]
            self.refresh_display()
            self._update_status_bar()
    
    def action_prev_match(self) -> None:
        """Jump to previous search match"""
        if self.model.search_results:
            self.search_index = (self.search_index - 1) % len(self.model.search_results)
            self.model.selected_node = self.model.search_results[self.search_index]
            self.refresh_display()
            self._update_status_bar()
    
    def action_view_tree(self) -> None:
        """Switch to tree view"""
        self.view_mode = "tree"
    
    def action_view_outline(self) -> None:
        """Switch to outline view"""
        self.view_mode = "outline"
    
    def action_view_ascii(self) -> None:
        """Switch to ASCII view"""
        self.view_mode = "ascii"
    
    def action_announce_position(self) -> None:
        """Announce current position for accessibility"""
        if not self.model.selected_node:
            self.notify("No node selected")
            return
        
        # Build path
        path = []
        node = self.model.selected_node
        while node:
            path.append(getattr(node, 'text', node.name))
            node = node.parent
        
        path.reverse()
        position = " → ".join(path)
        
        # Add sibling info
        if self.model.selected_node.parent:
            siblings = self.model.selected_node.parent.children
            index = siblings.index(self.model.selected_node) + 1
            total = len(siblings)
            position += f" (item {index} of {total})"
        
        self.notify(position)
        self._update_aria_live(position)
    
    def _update_aria_live(self, message: str) -> None:
        """Update ARIA live region for screen readers"""
        self.query_one("#aria-live", Static).update(message)
    
    # Event handlers
    @on(Button.Pressed)
    def handle_button_press(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "collapse-all":
            self.action_collapse_all()
        elif button_id == "expand-all":
            self.action_expand_all()
        elif button_id == "search-btn":
            self.action_search()
        elif button_id == "refresh-btn":
            self.refresh_display()
        elif button_id == "view-tree":
            self.action_view_tree()
        elif button_id == "view-outline":
            self.action_view_outline()
        elif button_id == "view-ascii":
            self.action_view_ascii()
        elif button_id == "search-next":
            self.action_next_match()
        elif button_id == "search-prev":
            self.action_prev_match()
        elif button_id == "search-close":
            self.action_clear_search()
    
    @on(Input.Submitted, "#search-input")
    def handle_search_submit(self, event: Input.Submitted) -> None:
        """Handle search input submission"""
        query = event.value.strip()
        if query:
            results = self.model.search(query)
            if results:
                self.search_index = 0
                self.model.selected_node = results[0]
                self.refresh_display()
                self._update_status_bar()
                self.notify(f"Found {len(results)} matches")
            else:
                self.notify("No matches found")
    
    @on(Input.Changed, "#search-input")
    def handle_search_change(self, event: Input.Changed) -> None:
        """Handle real-time search as user types"""
        query = event.value.strip()
        if query:
            results = self.model.search(query)
            self.refresh_display()
            self._update_status_bar()
    
    # Public methods
    def load_mermaid(self, mermaid_code: str) -> None:
        """Load a new mindmap from Mermaid code
        
        Args:
            mermaid_code: Mermaid mindmap syntax
        """
        try:
            self.model.load_from_mermaid(mermaid_code)
            self.refresh_display()
            self._update_status_bar()
            self.notify("Mindmap loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load mindmap: {e}")
            self.notify(f"Error loading mindmap: {str(e)}", severity="error")
    
    def get_selected_node(self) -> Optional[Node]:
        """Get the currently selected node
        
        Returns:
            Selected node or None
        """
        return self.model.selected_node
    
    def set_theme(self, theme_name: str) -> None:
        """Change the color theme
        
        Args:
            theme_name: Name of the theme to apply
        """
        self.renderer.set_theme(theme_name)
        self.refresh_display()