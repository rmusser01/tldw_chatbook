# mindmap_renderer.py
# Description: Renderers for mindmap visualization
#
"""
Mindmap Renderer
---------------

Renders mindmap trees in various formats:
- Tree view with Unicode box drawing
- Outline view with indentation
- ASCII art view with boxes
- Screen reader friendly format
"""

from typing import List, Dict, Tuple, Optional
from anytree import Node, PreOrderIter
from rich.text import Text
from rich.console import Console
from rich.panel import Panel
from loguru import logger

from .mindmap_model import MindmapModel
from .mermaid_parser import NodeShape


class MindmapRenderer:
    """Base renderer for mindmap visualization"""
    
    def __init__(self, model: MindmapModel):
        self.model = model
        self.console = Console()
        
        # Unicode symbols for tree drawing
        self.symbols = {
            'tree_vertical': '│',
            'tree_horizontal': '─',
            'tree_corner': '└',
            'tree_branch': '├',
            'expand': '▼',
            'collapse': '▶',
            'space': ' ',
        }
        
        # ASCII fallback symbols
        self.ascii_symbols = {
            'tree_vertical': '|',
            'tree_horizontal': '-',
            'tree_corner': '\\',
            'tree_branch': '+',
            'expand': 'v',
            'collapse': '>',
            'space': ' ',
        }
        
        self.use_unicode = True
        
    def render_tree_view(self) -> Text:
        """Render as indented tree with Unicode box drawing
        
        Returns:
            Rich Text object with formatted tree
        """
        if not self.model.root:
            return Text("No mindmap loaded", style="dim italic")
        
        text = Text()
        self._render_node_tree(self.model.root, text, "", True, [])
        return text
    
    def _render_node_tree(self, node: Node, text: Text, prefix: str, 
                          is_last: bool, parent_continues: List[bool]) -> None:
        """Recursively render tree structure
        
        Args:
            node: Current node to render
            text: Text object to append to
            prefix: Current line prefix
            is_last: Whether this is the last child
            parent_continues: Stack of parent continuation states
        """
        symbols = self.symbols if self.use_unicode else self.ascii_symbols
        
        # Determine if node is expanded
        is_expanded = node in self.model.expanded_nodes
        has_children = bool(node.children)
        
        # Build the connector line
        if node == self.model.root:
            connector = ""
        else:
            connector = symbols['tree_corner'] if is_last else symbols['tree_branch']
            connector += symbols['tree_horizontal'] * 2 + symbols['space']
        
        # Node symbol based on shape
        shape_symbol = node.shape.value if hasattr(node, 'shape') else NodeShape.DEFAULT.value
        
        # Expansion indicator
        if has_children:
            expand_symbol = symbols['expand'] if is_expanded else symbols['collapse']
            expand_symbol += symbols['space']
        else:
            expand_symbol = symbols['space'] * 2
        
        # Build the line
        line_text = f"{prefix}{connector}{expand_symbol}{shape_symbol} "
        text.append(line_text)
        
        # Add node text with styling
        node_text = node.text if hasattr(node, 'text') else node.name
        
        # Apply search highlighting
        if self.model.search_results and node in self.model.search_results:
            text.append(node_text, style="bold yellow on dark_blue")
        # Apply selection highlighting
        elif node == self.model.selected_node:
            text.append(node_text, style="bold cyan")
        else:
            text.append(node_text)
        
        text.append("\n")
        
        # Render children if expanded
        if is_expanded and has_children:
            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                
                # Build child prefix
                if node == self.model.root:
                    child_prefix = ""
                else:
                    if is_last:
                        child_prefix = prefix + symbols['space'] * 4
                    else:
                        child_prefix = prefix + symbols['tree_vertical'] + symbols['space'] * 3
                
                self._render_node_tree(child, text, child_prefix, is_last_child, 
                                     parent_continues + [not is_last])
    
    def render_outline_view(self) -> Text:
        """Render as hierarchical outline
        
        Returns:
            Rich Text object with formatted outline
        """
        if not self.model.root:
            return Text("No mindmap loaded", style="dim italic")
        
        text = Text()
        self._render_node_outline(self.model.root, text, 0)
        return text
    
    def _render_node_outline(self, node: Node, text: Text, level: int) -> None:
        """Render as indented outline
        
        Args:
            node: Current node to render
            text: Text object to append to
            level: Current indentation level
        """
        indent = "  " * level
        
        # Choose bullet based on level
        if level == 0:
            bullet = "▣"
        elif level == 1:
            bullet = "▪"
        elif level == 2:
            bullet = "▫"
        else:
            bullet = "•"
        
        # Build line
        text.append(f"{indent}{bullet} ")
        
        # Add node text with styling
        node_text = node.text if hasattr(node, 'text') else node.name
        
        if self.model.search_results and node in self.model.search_results:
            text.append(node_text, style="bold yellow on dark_blue")
        elif node == self.model.selected_node:
            text.append(node_text, style="bold cyan")
        else:
            text.append(node_text)
        
        text.append("\n")
        
        # Render children if expanded
        if node in self.model.expanded_nodes:
            for child in node.children:
                self._render_node_outline(child, text, level + 1)
    
    def render_ascii_art(self) -> Text:
        """Render as ASCII art with boxes
        
        Returns:
            Rich Text object with ASCII art
        """
        if not self.model.root:
            return Text("No mindmap loaded", style="dim italic")
        
        text = Text()
        text.append("ASCII Art View (Simplified)\n", style="bold")
        text.append("=" * 40 + "\n\n")
        
        # Simplified ASCII rendering
        self._render_node_ascii_simple(self.model.root, text, 0)
        
        return text
    
    def _render_node_ascii_simple(self, node: Node, text: Text, level: int) -> None:
        """Simple ASCII box rendering
        
        Args:
            node: Current node to render
            text: Text object to append to
            level: Current indentation level
        """
        indent = "  " * level
        node_text = node.text if hasattr(node, 'text') else node.name
        
        # Calculate box dimensions
        width = len(node_text) + 4
        
        # Top border
        text.append(f"{indent}┌" + "─" * (width - 2) + "┐\n")
        
        # Content
        text.append(f"{indent}│ {node_text} │")
        
        # Add selection indicator
        if node == self.model.selected_node:
            text.append(" ←", style="bold cyan")
        
        text.append("\n")
        
        # Bottom border
        text.append(f"{indent}└" + "─" * (width - 2) + "┘\n")
        
        # Connection to children
        if node in self.model.expanded_nodes and node.children:
            text.append(f"{indent}    │\n")
            
            for i, child in enumerate(node.children):
                if i > 0:
                    text.append(f"{indent}    │\n")
                self._render_node_ascii_simple(child, text, level + 2)
    
    def render_for_screen_reader(self) -> str:
        """Generate screen reader friendly representation
        
        Returns:
            Plain text suitable for screen readers
        """
        if not self.model.root:
            return "Empty mindmap"
        
        lines = []
        self._render_node_sr(self.model.root, lines, 0, [1])
        return "\n".join(lines)
    
    def _render_node_sr(self, node: Node, lines: List[str], 
                        level: int, position: List[int]) -> None:
        """Render node for screen reader
        
        Args:
            node: Current node to render
            lines: List of output lines
            level: Current depth level
            position: Current position in tree (e.g., [1, 2, 3])
        """
        # Build position string (e.g., "1.2.3")
        pos_str = ".".join(str(p) for p in position)
        
        # Build level indicator
        level_str = f"Level {level}"
        
        # Node status
        if node.children:
            if node in self.model.expanded_nodes:
                status = "expanded"
            else:
                status = "collapsed"
            child_count = f", {len(node.children)} children"
        else:
            status = "leaf"
            child_count = ""
        
        # Selection status
        selected = ", selected" if node == self.model.selected_node else ""
        
        # Search match status
        search_match = ", search match" if self.model.search_results and node in self.model.search_results else ""
        
        # Node text
        node_text = node.text if hasattr(node, 'text') else node.name
        
        # Full description
        line = f"{pos_str}. {node_text} ({level_str}, {status}{child_count}{selected}{search_match})"
        lines.append(line)
        
        # Render children if expanded
        if node in self.model.expanded_nodes:
            for i, child in enumerate(node.children, 1):
                child_position = position + [i]
                self._render_node_sr(child, lines, level + 1, child_position)
    
    def get_node_at_position(self, x: int, y: int) -> Optional[Node]:
        """Get node at terminal position (for future mouse support)
        
        Args:
            x: Column position
            y: Row position
            
        Returns:
            Node at position or None
        """
        # This would require tracking rendered positions
        # Placeholder for future implementation
        return None
    
    def set_unicode_mode(self, use_unicode: bool) -> None:
        """Toggle between Unicode and ASCII symbols
        
        Args:
            use_unicode: Whether to use Unicode symbols
        """
        self.use_unicode = use_unicode
        logger.info(f"Renderer unicode mode: {use_unicode}")


class ThemedMindmapRenderer(MindmapRenderer):
    """Renderer with theme support"""
    
    THEMES = {
        "default": {
            "node_color": "white",
            "selected_color": "bold cyan",
            "search_color": "bold yellow on dark_blue",
            "branch_color": "blue",
            "expand_color": "green",
            "collapse_color": "yellow"
        },
        "high_contrast": {
            "node_color": "bright_white",
            "selected_color": "bright_yellow on black",
            "search_color": "black on bright_yellow",
            "branch_color": "bright_white",
            "expand_color": "bright_green",
            "collapse_color": "bright_red"
        },
        "dark": {
            "node_color": "grey70",
            "selected_color": "cyan",
            "search_color": "yellow",
            "branch_color": "grey50",
            "expand_color": "green",
            "collapse_color": "red"
        },
        "solarized": {
            "node_color": "#839496",
            "selected_color": "#268bd2",
            "search_color": "#b58900",
            "branch_color": "#586e75",
            "expand_color": "#859900",
            "collapse_color": "#cb4b16"
        }
    }
    
    def __init__(self, model: MindmapModel, theme: str = "default"):
        super().__init__(model)
        self.theme_name = theme
        self.theme = self.THEMES.get(theme, self.THEMES["default"])
    
    def set_theme(self, theme_name: str) -> None:
        """Change the current theme
        
        Args:
            theme_name: Name of the theme to apply
        """
        if theme_name in self.THEMES:
            self.theme_name = theme_name
            self.theme = self.THEMES[theme_name]
            logger.info(f"Applied theme: {theme_name}")