# Building a Mermaid.js Mindmap Viewer in Textual

## Executive Overview

Building a Mermaid.js mindmap viewer using Rich/Textual is not only feasible but offers unique advantages for terminal-based workflows. While we cannot achieve the same visual richness as browser-based Mermaid.js, we can create a highly functional, keyboard-driven mindmap viewer that integrates seamlessly with the tldw_chatbook ecosystem.

### What's Possible

1. **Full Mermaid Mindmap Syntax Support**
   - Parse all Mermaid mindmap node types: `((circle))`, `[square]`, `(rounded)`, `{hexagon}`, `{{cloud}}`
   - Support for node IDs and hierarchical relationships
   - Multi-line text support within nodes

2. **Interactive Navigation**
   - Keyboard-driven navigation (arrow keys, vim bindings)
   - Expand/collapse branches
   - Search and filter functionality
   - Jump to node by ID or text

3. **Multiple View Modes**
   - Tree view (vertical hierarchy)
   - Outline view (indented text)
   - Compact view (minimal spacing)
   - ASCII art view (box drawing)

4. **Integration Features**
   - Import from existing Notes, Conversations, or Media
   - Export to multiple formats
   - Real-time updates from database
   - Collaborative editing support

### Limitations

1. **Visual Constraints**
   - No curved lines or arbitrary positioning
   - Limited to terminal color palette
   - Fixed-width character grid
   - No images or custom shapes

2. **Layout Constraints**
   - Radial layouts are impractical
   - Limited to tree-like structures
   - No overlapping elements
   - Fixed character sizes

## Technical Architecture

### Core Components

```python
# Core architecture overview
class MermaidMindmapSystem:
    """
    Main components:
    1. Parser: Mermaid syntax → Tree structure
    2. Model: Tree data structure (using anytree)
    3. Renderer: Tree structure → Terminal display
    4. Controller: Handle user interactions
    """
    
    def __init__(self):
        self.parser = MermaidMindmapParser()
        self.model = MindmapModel()
        self.renderer = MindmapRenderer()
        self.controller = MindmapController()
```

### Parser Design

```python
from typing import List, Dict, Optional, Tuple
import re
from anytree import Node, RenderTree
from dataclasses import dataclass
from enum import Enum

class NodeShape(Enum):
    """Mermaid node shapes mapped to Unicode symbols"""
    CIRCLE = "○"
    DOUBLE_CIRCLE = "◉"
    SQUARE = "□"
    ROUNDED = "▢"
    HEXAGON = "⬡"
    CLOUD = "☁"
    DEFAULT = "•"

@dataclass
class MindmapNode:
    """Represents a parsed mindmap node"""
    id: str
    text: str
    shape: NodeShape
    level: int
    children: List['MindmapNode'] = None
    metadata: Dict[str, any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}

class MermaidMindmapParser:
    """Parse Mermaid mindmap syntax into tree structure"""
    
    # Regex patterns for different node types
    PATTERNS = {
        'double_circle': r'(\w+)\(\((.+?)\)\)',  # id((text))
        'square': r'(\w+)\[(.+?)\]',              # id[text]
        'rounded': r'(\w+)\((.+?)\)',             # id(text)
        'hexagon': r'(\w+)\{(.+?)\}',             # id{text}
        'cloud': r'(\w+)\{\{(.+?)\}\}',           # id{{text}}
    }
    
    SHAPE_MAP = {
        'double_circle': NodeShape.DOUBLE_CIRCLE,
        'square': NodeShape.SQUARE,
        'rounded': NodeShape.ROUNDED,
        'hexagon': NodeShape.HEXAGON,
        'cloud': NodeShape.CLOUD,
    }
    
    def parse(self, mermaid_code: str) -> Node:
        """Parse Mermaid mindmap code into anytree structure"""
        lines = mermaid_code.strip().split('\n')
        root_node = None
        node_stack = []  # Stack of (node, indent_level)
        
        for line in lines:
            # Skip empty lines and the 'mindmap' declaration
            if not line.strip() or line.strip() == 'mindmap':
                continue
            
            # Calculate indentation level
            indent_level = self._get_indent_level(line)
            
            # Parse node
            node_info = self._parse_node_line(line.strip())
            if not node_info:
                continue
            
            # Create anytree node
            if root_node is None:
                root_node = Node(
                    node_info['id'],
                    text=node_info['text'],
                    shape=node_info['shape'],
                    level=0
                )
                node_stack = [(root_node, indent_level)]
            else:
                # Find parent based on indentation
                parent_node = self._find_parent_node(node_stack, indent_level)
                
                new_node = Node(
                    node_info['id'],
                    parent=parent_node,
                    text=node_info['text'],
                    shape=node_info['shape'],
                    level=indent_level
                )
                
                # Update stack
                node_stack = [(n, l) for n, l in node_stack if l < indent_level]
                node_stack.append((new_node, indent_level))
        
        return root_node
    
    def _get_indent_level(self, line: str) -> int:
        """Calculate indentation level (2 spaces = 1 level)"""
        return (len(line) - len(line.lstrip())) // 2
    
    def _parse_node_line(self, line: str) -> Optional[Dict]:
        """Parse a single node line"""
        for shape_type, pattern in self.PATTERNS.items():
            match = re.match(pattern, line)
            if match:
                return {
                    'id': match.group(1),
                    'text': match.group(2),
                    'shape': self.SHAPE_MAP[shape_type]
                }
        
        # Default format: just text
        if line:
            return {
                'id': line.replace(' ', '_'),
                'text': line,
                'shape': NodeShape.DEFAULT
            }
        
        return None
    
    def _find_parent_node(self, node_stack: List[Tuple[Node, int]], 
                          indent_level: int) -> Node:
        """Find the appropriate parent node based on indentation"""
        for node, level in reversed(node_stack):
            if level < indent_level:
                return node
        return node_stack[0][0]  # Default to root
```

### Tree Model

```python
from anytree import Node, PreOrderIter, find_by_attr
from typing import List, Optional, Callable

class MindmapModel:
    """Model for mindmap data using anytree"""
    
    def __init__(self):
        self.root: Optional[Node] = None
        self.selected_node: Optional[Node] = None
        self.expanded_nodes: set = set()
        self.search_results: List[Node] = []
        
    def load_from_mermaid(self, mermaid_code: str):
        """Load mindmap from Mermaid syntax"""
        parser = MermaidMindmapParser()
        self.root = parser.parse(mermaid_code)
        self.selected_node = self.root
        self.expanded_nodes = {self.root}
        
    def load_from_database(self, mindmap_id: str, db):
        """Load mindmap from database"""
        # Implementation depends on database schema
        pass
    
    def get_visible_nodes(self) -> List[Node]:
        """Get all nodes that should be visible (respecting collapse state)"""
        if not self.root:
            return []
        
        visible = []
        for node in PreOrderIter(self.root):
            visible.append(node)
            if node not in self.expanded_nodes and node.children:
                # Skip children of collapsed nodes
                for child in PreOrderIter(node):
                    if child != node:
                        child._visible = False
        
        return [n for n in visible if getattr(n, '_visible', True)]
    
    def toggle_node(self, node: Node):
        """Toggle expand/collapse state of a node"""
        if node in self.expanded_nodes:
            self.expanded_nodes.remove(node)
        else:
            self.expanded_nodes.add(node)
    
    def search(self, query: str) -> List[Node]:
        """Search for nodes containing query text"""
        if not self.root:
            return []
        
        self.search_results = []
        query_lower = query.lower()
        
        for node in PreOrderIter(self.root):
            if query_lower in node.text.lower():
                self.search_results.append(node)
                # Ensure path to result is expanded
                self._expand_path_to_node(node)
        
        return self.search_results
    
    def _expand_path_to_node(self, node: Node):
        """Expand all ancestors of a node"""
        current = node.parent
        while current:
            self.expanded_nodes.add(current)
            current = current.parent
```

### Renderer Implementation

```python
from textual.app import App, ComposeResult
from textual.widgets import Tree, Static, Input, Button
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual import events
from rich.text import Text
from rich.panel import Panel

class MindmapRenderer(Static):
    """Render mindmap in various formats"""
    
    def __init__(self, model: MindmapModel, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.view_mode = "tree"  # tree, outline, ascii
        
    def render_tree_view(self) -> str:
        """Render as indented tree with Unicode box drawing"""
        if not self.model.root:
            return "No mindmap loaded"
        
        lines = []
        self._render_node_tree(self.model.root, lines, "", True)
        return "\n".join(lines)
    
    def _render_node_tree(self, node: Node, lines: List[str], 
                          prefix: str, is_last: bool):
        """Recursively render tree structure"""
        # Determine if node is expanded
        is_expanded = node in self.model.expanded_nodes
        has_children = bool(node.children)
        
        # Select appropriate symbols
        if node == self.model.root:
            connector = ""
            extension = ""
        else:
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
        
        # Node symbol based on shape
        symbol = node.shape.value if hasattr(node, 'shape') else "•"
        
        # Expansion indicator
        if has_children:
            expand_symbol = "▼ " if is_expanded else "▶ "
        else:
            expand_symbol = "  "
        
        # Highlight selected node
        text = node.text
        if node == self.model.selected_node:
            text = f"[bold yellow]{text}[/bold yellow]"
        
        # Build line
        line = f"{prefix}{connector}{expand_symbol}{symbol} {text}"
        lines.append(line)
        
        # Render children if expanded
        if is_expanded and has_children:
            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                child_prefix = prefix + extension
                self._render_node_tree(child, lines, child_prefix, is_last_child)
    
    def render_outline_view(self) -> str:
        """Render as hierarchical outline"""
        if not self.model.root:
            return "No mindmap loaded"
        
        lines = []
        self._render_node_outline(self.model.root, lines, 0)
        return "\n".join(lines)
    
    def _render_node_outline(self, node: Node, lines: List[str], level: int):
        """Render as indented outline"""
        indent = "  " * level
        bullet = "•" if level > 0 else "▣"
        
        text = node.text
        if node == self.model.selected_node:
            text = f"[bold yellow]{text}[/bold yellow]"
        
        lines.append(f"{indent}{bullet} {text}")
        
        if node in self.model.expanded_nodes:
            for child in node.children:
                self._render_node_outline(child, lines, level + 1)
    
    def render_ascii_art(self) -> str:
        """Render as ASCII art with boxes"""
        if not self.model.root:
            return "No mindmap loaded"
        
        # This is a simplified version - full implementation would
        # calculate positions and draw connections
        lines = []
        self._render_node_ascii(self.model.root, lines, 0, 0)
        return "\n".join(lines)
    
    def _render_node_ascii(self, node: Node, lines: List[str], 
                           x: int, y: int) -> Tuple[int, int]:
        """Render node as ASCII box"""
        # Calculate box dimensions
        text_lines = node.text.split('\n')
        width = max(len(line) for line in text_lines) + 4
        height = len(text_lines) + 2
        
        # Draw box
        # Top border
        lines.append("┌" + "─" * (width - 2) + "┐")
        
        # Content
        for line in text_lines:
            padded = line.center(width - 2)
            lines.append("│" + padded + "│")
        
        # Bottom border
        lines.append("└" + "─" * (width - 2) + "┘")
        
        return width, height
```

### Interactive Widget

```python
from textual import on
from textual.binding import Binding

class MindmapViewer(Container):
    """Interactive mindmap viewer widget"""
    
    BINDINGS = [
        Binding("up", "move_up", "Move up"),
        Binding("down", "move_down", "Move down"),
        Binding("left", "collapse", "Collapse"),
        Binding("right", "expand", "Expand"),
        Binding("enter", "toggle", "Toggle"),
        Binding("/", "search", "Search"),
        Binding("e", "export", "Export"),
        Binding("v", "cycle_view", "Change view"),
    ]
    
    def __init__(self, mermaid_code: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model = MindmapModel()
        self.renderer = MindmapRenderer(self.model)
        self.search_mode = False
        
        if mermaid_code:
            self.model.load_from_mermaid(mermaid_code)
    
    def compose(self) -> ComposeResult:
        """Compose the mindmap viewer UI"""
        with Vertical():
            # Header with controls
            with Horizontal(classes="mindmap-header"):
                yield Button("◀ Collapse All", id="collapse-all")
                yield Button("▶ Expand All", id="expand-all")
                yield Button("🔍 Search", id="search-btn")
                yield Button("💾 Export", id="export-btn")
                yield Static("View: Tree", id="view-mode")
            
            # Search bar (hidden by default)
            yield Input(
                placeholder="Search nodes...",
                id="search-input",
                classes="hidden"
            )
            
            # Main display area
            yield ScrollableContainer(
                self.renderer,
                id="mindmap-display"
            )
            
            # Status bar
            yield Static("", id="status-bar")
    
    def on_mount(self):
        """Initialize the display"""
        self.refresh_display()
    
    def refresh_display(self):
        """Update the mindmap display"""
        content = ""
        
        if self.renderer.view_mode == "tree":
            content = self.renderer.render_tree_view()
        elif self.renderer.view_mode == "outline":
            content = self.renderer.render_outline_view()
        elif self.renderer.view_mode == "ascii":
            content = self.renderer.render_ascii_art()
        
        self.renderer.update(content)
        self.update_status_bar()
    
    def update_status_bar(self):
        """Update status bar with current info"""
        if not self.model.root:
            status = "No mindmap loaded"
        else:
            total_nodes = len(list(PreOrderIter(self.model.root)))
            visible_nodes = len(self.model.get_visible_nodes())
            selected = self.model.selected_node.text if self.model.selected_node else "None"
            
            status = f"Nodes: {visible_nodes}/{total_nodes} | Selected: {selected}"
        
        self.query_one("#status-bar").update(status)
    
    def action_move_up(self):
        """Move selection up"""
        if not self.model.selected_node:
            return
        
        visible_nodes = self.model.get_visible_nodes()
        current_idx = visible_nodes.index(self.model.selected_node)
        
        if current_idx > 0:
            self.model.selected_node = visible_nodes[current_idx - 1]
            self.refresh_display()
    
    def action_move_down(self):
        """Move selection down"""
        if not self.model.selected_node:
            return
        
        visible_nodes = self.model.get_visible_nodes()
        current_idx = visible_nodes.index(self.model.selected_node)
        
        if current_idx < len(visible_nodes) - 1:
            self.model.selected_node = visible_nodes[current_idx + 1]
            self.refresh_display()
    
    def action_expand(self):
        """Expand current node"""
        if self.model.selected_node and self.model.selected_node.children:
            self.model.expanded_nodes.add(self.model.selected_node)
            self.refresh_display()
    
    def action_collapse(self):
        """Collapse current node"""
        if self.model.selected_node:
            self.model.expanded_nodes.discard(self.model.selected_node)
            self.refresh_display()
    
    def action_toggle(self):
        """Toggle expand/collapse"""
        if self.model.selected_node:
            self.model.toggle_node(self.model.selected_node)
            self.refresh_display()
    
    def action_search(self):
        """Toggle search mode"""
        search_input = self.query_one("#search-input")
        if search_input.has_class("hidden"):
            search_input.remove_class("hidden")
            search_input.focus()
        else:
            search_input.add_class("hidden")
            self.search_mode = False
    
    @on(Input.Submitted, "#search-input")
    def handle_search(self, event: Input.Submitted):
        """Handle search submission"""
        query = event.value
        if query:
            results = self.model.search(query)
            if results:
                self.model.selected_node = results[0]
                self.refresh_display()
                self.notify(f"Found {len(results)} matches")
            else:
                self.notify("No matches found")
    
    def action_cycle_view(self):
        """Cycle through view modes"""
        modes = ["tree", "outline", "ascii"]
        current_idx = modes.index(self.renderer.view_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.renderer.view_mode = modes[next_idx]
        
        self.query_one("#view-mode").update(f"View: {modes[next_idx].title()}")
        self.refresh_display()
```

## Advanced Features

### Export Capabilities

```python
class MindmapExporter:
    """Export mindmap to various formats"""
    
    @staticmethod
    def to_markdown(root: Node) -> str:
        """Export to Markdown outline"""
        lines = []
        
        def render_node(node: Node, level: int):
            indent = "  " * level
            prefix = "#" * min(level + 1, 6) if level < 6 else "-"
            lines.append(f"{prefix} {node.text}")
            
            for child in node.children:
                render_node(child, level + 1)
        
        render_node(root, 0)
        return "\n".join(lines)
    
    @staticmethod
    def to_graphviz(root: Node) -> str:
        """Export to GraphViz DOT format"""
        lines = ["digraph mindmap {", "  rankdir=LR;", "  node [shape=box];"]
        
        def add_node(node: Node):
            # Escape quotes in text
            text = node.text.replace('"', '\\"')
            lines.append(f'  "{node.name}" [label="{text}"];')
            
            for child in node.children:
                lines.append(f'  "{node.name}" -> "{child.name}";')
                add_node(child)
        
        add_node(root)
        lines.append("}")
        return "\n".join(lines)
    
    @staticmethod
    def to_ascii_art(root: Node, max_width: int = 80) -> str:
        """Export to ASCII art representation"""
        # Implementation would calculate positions and draw connections
        # This is a simplified version
        return AsciiTreeRenderer().render(root, max_width)
```

### Integration with tldw_chatbook

```python
class MindmapIntegration:
    """Integrate mindmap with existing tldw_chatbook features"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        self.db = app_instance.chachanotes_db
    
    def create_from_conversation(self, conversation_id: str) -> Node:
        """Create mindmap from conversation messages"""
        messages = self.db.get_messages_for_conversation(conversation_id)
        
        root = Node("conversation", text=f"Conversation {conversation_id}")
        
        for msg in messages:
            # Extract key points using LLM
            key_points = self._extract_key_points(msg['content'])
            
            msg_node = Node(
                f"msg_{msg['id']}", 
                parent=root,
                text=f"{msg['sender']}: {msg['summary']}"
            )
            
            for point in key_points:
                Node(f"point_{point['id']}", parent=msg_node, text=point['text'])
        
        return root
    
    def create_from_notes(self, note_ids: List[str]) -> Node:
        """Create mindmap from notes"""
        root = Node("notes", text="Notes Overview")
        
        for note_id in note_ids:
            note = self.db.get_note_by_id(note_id)
            if note:
                note_node = Node(
                    f"note_{note_id}",
                    parent=root,
                    text=note['title']
                )
                
                # Parse note content for headers
                headers = self._extract_headers(note['content'])
                for header in headers:
                    Node(
                        f"header_{header['id']}",
                        parent=note_node,
                        text=header['text']
                    )
        
        return root
```

### Performance Optimization

```python
class VirtualMindmapTree(Tree):
    """Virtual scrolling for large mindmaps"""
    
    def __init__(self, model: MindmapModel, **kwargs):
        super().__init__("Mindmap", **kwargs)
        self.model = model
        self.visible_window = 50  # Number of visible nodes
        self.scroll_offset = 0
        
    def on_mount(self):
        """Load only visible portion"""
        self.refresh_visible_nodes()
    
    def refresh_visible_nodes(self):
        """Update only the visible portion of the tree"""
        self.clear()
        
        all_visible = self.model.get_visible_nodes()
        start_idx = self.scroll_offset
        end_idx = min(start_idx + self.visible_window, len(all_visible))
        
        # Build tree structure for visible portion
        node_map = {}
        
        for node in all_visible[start_idx:end_idx]:
            # Find parent in our map or root
            if node.parent and node.parent in node_map:
                tree_parent = node_map[node.parent]
            else:
                tree_parent = self.root
            
            # Add to tree
            tree_node = tree_parent.add(node.text)
            node_map[node] = tree_node
            
            # Mark as selected if needed
            if node == self.model.selected_node:
                self.cursor_node = tree_node
```

## Testing Strategy

```python
import pytest
from io import StringIO

class TestMermaidParser:
    """Test Mermaid mindmap parser"""
    
    def test_parse_simple_mindmap(self):
        """Test parsing a simple mindmap"""
        mermaid_code = """
        mindmap
          root((Root))
            A[Node A]
            B[Node B]
        """
        
        parser = MermaidMindmapParser()
        root = parser.parse(mermaid_code)
        
        assert root.text == "Root"
        assert len(root.children) == 2
        assert root.children[0].text == "Node A"
        assert root.children[1].text == "Node B"
    
    def test_parse_nested_mindmap(self):
        """Test parsing nested structures"""
        mermaid_code = """
        mindmap
          root((Root))
            A[Node A]
              A1(Child A1)
              A2(Child A2)
            B[Node B]
              B1{Child B1}
        """
        
        parser = MermaidMindmapParser()
        root = parser.parse(mermaid_code)
        
        assert len(root.children) == 2
        assert len(root.children[0].children) == 2
        assert len(root.children[1].children) == 1
    
    def test_parse_all_node_types(self):
        """Test all Mermaid node shapes"""
        mermaid_code = """
        mindmap
          root((Double Circle))
            A[Square]
            B(Rounded)
            C{Hexagon}
            D{{Cloud}}
        """
        
        parser = MermaidMindmapParser()
        root = parser.parse(mermaid_code)
        
        assert root.shape == NodeShape.DOUBLE_CIRCLE
        assert root.children[0].shape == NodeShape.SQUARE
        assert root.children[1].shape == NodeShape.ROUNDED
        assert root.children[2].shape == NodeShape.HEXAGON
        assert root.children[3].shape == NodeShape.CLOUD

class TestMindmapModel:
    """Test mindmap model operations"""
    
    def test_expand_collapse(self):
        """Test expand/collapse functionality"""
        model = MindmapModel()
        model.load_from_mermaid("""
        mindmap
          root((Root))
            A[Node A]
              A1(Child A1)
        """)
        
        # Initially only root is expanded
        assert model.root in model.expanded_nodes
        assert len(model.get_visible_nodes()) == 2  # root + A
        
        # Expand Node A
        node_a = model.root.children[0]
        model.toggle_node(node_a)
        assert len(model.get_visible_nodes()) == 3  # root + A + A1
    
    def test_search(self):
        """Test search functionality"""
        model = MindmapModel()
        model.load_from_mermaid("""
        mindmap
          root((Root))
            A[Important Node]
              A1(Child Node)
            B[Another Node]
              B1(Important Child)
        """)
        
        results = model.search("Important")
        assert len(results) == 2
        assert "Important" in results[0].text
        assert "Important" in results[1].text
```

## Example Applications

### Standalone Mindmap Viewer

```python
class MindmapViewerApp(App):
    """Standalone mindmap viewer application"""
    
    CSS = """
    .mindmap-header {
        dock: top;
        height: 3;
        background: $boost;
    }
    
    #mindmap-display {
        background: $surface;
        border: solid $primary;
    }
    
    #status-bar {
        dock: bottom;
        height: 1;
        background: $panel;
    }
    
    .hidden {
        display: none;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the application"""
        sample_mermaid = """
        mindmap
          root((tldw_chatbook))
            Features[Core Features]
              Chat(Conversations)
                AI[AI Integration]
                History[Chat History]
              Notes(Notes System)
                Sync[File Sync]
                Templates[Templates]
              Media(Media Processing)
                Import[Import]
                Analysis[Analysis]
            Technical[Technical]
              UI{Textual UI}
                Widgets[Custom Widgets]
                Themes[Theming]
              Storage{Data Storage}
                SQLite[SQLite]
                FTS[Full Text Search]
        """
        
        yield MindmapViewer(sample_mermaid)
```

### Integration with Study Window

```python
class StudyMindmapWidget(MindmapViewer):
    """Mindmap widget for Study Window integration"""
    
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.db = app_instance.chachanotes_db
    
    def compose(self) -> ComposeResult:
        """Compose with study-specific controls"""
        yield from super().compose()
        
        # Add study-specific controls
        with Horizontal(classes="study-controls"):
            yield Button("Import from Notes", id="import-notes")
            yield Button("Create Flashcards", id="create-flashcards")
            yield Button("Save to Study Plan", id="save-plan")
    
    @on(Button.Pressed, "#import-notes")
    async def import_from_notes(self):
        """Import notes into mindmap"""
        # Show note selector dialog
        notes = await self.show_note_selector()
        
        if notes:
            integration = MindmapIntegration(self.app_instance)
            root = integration.create_from_notes([n['id'] for n in notes])
            self.model.root = root
            self.model.selected_node = root
            self.model.expanded_nodes = {root}
            self.refresh_display()
```

## Performance Considerations

### Memory Management

```python
class LazyLoadMindmap:
    """Lazy loading for very large mindmaps"""
    
    def __init__(self, db, mindmap_id: str):
        self.db = db
        self.mindmap_id = mindmap_id
        self.loaded_nodes = {}
        self.root = None
        
    def get_node(self, node_id: str) -> Node:
        """Load node on demand"""
        if node_id in self.loaded_nodes:
            return self.loaded_nodes[node_id]
        
        # Load from database
        node_data = self.db.get_mindmap_node(node_id)
        
        # Create node
        parent = None
        if node_data['parent_id']:
            parent = self.get_node(node_data['parent_id'])
        
        node = Node(
            node_id,
            parent=parent,
            text=node_data['text'],
            shape=NodeShape[node_data['shape']]
        )
        
        self.loaded_nodes[node_id] = node
        return node
    
    def load_visible_portion(self, center_node_id: str, radius: int = 2):
        """Load nodes within radius of center node"""
        # Implementation would load nodes within specified
        # graph distance from center node
        pass
```

### Rendering Optimization

```python
class CachedRenderer:
    """Cache rendered output for performance"""
    
    def __init__(self, renderer: MindmapRenderer):
        self.renderer = renderer
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def render(self, view_mode: str) -> str:
        """Render with caching"""
        # Generate cache key
        visible_nodes = self.renderer.model.get_visible_nodes()
        cache_key = (
            view_mode,
            tuple(n.name for n in visible_nodes),
            self.renderer.model.selected_node.name if self.renderer.model.selected_node else None
        )
        
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # Render and cache
        self.cache_misses += 1
        
        if view_mode == "tree":
            result = self.renderer.render_tree_view()
        elif view_mode == "outline":
            result = self.renderer.render_outline_view()
        else:
            result = self.renderer.render_ascii_art()
        
        self.cache[cache_key] = result
        
        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entries
            oldest_keys = list(self.cache.keys())[:20]
            for key in oldest_keys:
                del self.cache[key]
        
        return result
```

## Known Issues and Solutions

### Unicode Compatibility

```python
class SafeUnicodeRenderer:
    """Renderer with fallback for limited Unicode support"""
    
    UNICODE_SYMBOLS = {
        'tree_vertical': '│',
        'tree_horizontal': '─',
        'tree_corner': '└',
        'tree_branch': '├',
        'expand': '▼',
        'collapse': '▶',
    }
    
    ASCII_FALLBACK = {
        'tree_vertical': '|',
        'tree_horizontal': '-',
        'tree_corner': '\\',
        'tree_branch': '+',
        'expand': 'v',
        'collapse': '>',
    }
    
    def __init__(self, use_unicode: bool = True):
        self.symbols = self.UNICODE_SYMBOLS if use_unicode else self.ASCII_FALLBACK
    
    def render_tree_line(self, prefix: str, is_last: bool, 
                         has_children: bool, is_expanded: bool) -> str:
        """Render tree line with appropriate symbols"""
        if is_last:
            connector = self.symbols['tree_corner'] + self.symbols['tree_horizontal']
        else:
            connector = self.symbols['tree_branch'] + self.symbols['tree_horizontal']
        
        if has_children:
            expand = self.symbols['expand'] if is_expanded else self.symbols['collapse']
        else:
            expand = " "
        
        return f"{prefix}{connector} {expand}"
```

### Cross-Platform Considerations

```python
import platform
import os

class PlatformAdapter:
    """Adapt rendering for different platforms"""
    
    @staticmethod
    def get_terminal_size() -> Tuple[int, int]:
        """Get terminal size safely across platforms"""
        try:
            if platform.system() == 'Windows':
                # Windows-specific handling
                import shutil
                return shutil.get_terminal_size()
            else:
                # Unix-like systems
                rows, cols = os.popen('stty size', 'r').read().split()
                return int(cols), int(rows)
        except:
            # Fallback
            return 80, 24
    
    @staticmethod
    def supports_unicode() -> bool:
        """Check if terminal supports Unicode"""
        if platform.system() == 'Windows':
            # Check Windows version and console
            import sys
            return sys.version_info >= (3, 6)
        else:
            # Check locale
            import locale
            return 'UTF' in locale.getpreferredencoding()
```

## Conclusion and Best Practices

Building a Mermaid mindmap viewer in Textual is highly feasible and offers unique advantages:

1. **Keyboard-Driven Efficiency**: Terminal users can navigate complex mindmaps without leaving their keyboard-centric workflow.

2. **Integration Potential**: Deep integration with existing tldw_chatbook features creates powerful workflows.

3. **Performance**: Lazy loading and virtual scrolling enable handling of very large mindmaps.

4. **Accessibility**: Text-based interface is inherently screen-reader friendly.

### Best Practices

1. **Start Simple**: Begin with basic tree rendering before adding advanced features.

2. **Use Existing Libraries**: Leverage anytree for tree operations and Rich for formatting.

3. **Cache Aggressively**: Terminal rendering can be expensive for large trees.

4. **Provide Multiple Views**: Different users prefer different representations.

5. **Test Cross-Platform**: Ensure Unicode fallbacks and platform-specific handling.

6. **Document Shortcuts**: Terminal users expect comprehensive keyboard shortcuts.

### Future Enhancements

1. **Collaborative Editing**: Real-time sync across multiple terminals
2. **AI Integration**: Auto-generate mindmaps from text using LLMs
3. **Advanced Layouts**: Implement more sophisticated layout algorithms
4. **Animation**: Smooth transitions for expand/collapse operations
5. **Plugin System**: Allow custom node types and renderers

The terminal-based mindmap viewer, while visually constrained compared to graphical versions, offers a powerful and efficient tool for users who prefer keyboard-driven interfaces and seamless integration with text-based workflows.