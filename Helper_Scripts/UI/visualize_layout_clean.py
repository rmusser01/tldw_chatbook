#!/usr/bin/env python3
"""
Clean Textual UI Layout Visualizer

Creates a clean, simplified view of Textual UI layouts focusing on structure.
"""

import ast
import sys
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class Widget:
    """Simple widget representation."""
    type: str
    id: Optional[str] = None
    text: Optional[str] = None
    children: List['Widget'] = field(default_factory=list)
    is_container: bool = False
    layout: str = "vertical"  # vertical or horizontal


class CleanLayoutVisualizer:
    """Creates clean layout visualizations."""
    
    CONTAINERS = {
        'Vertical': 'vertical',
        'VerticalScroll': 'vertical',
        'Container': 'vertical',
        'Horizontal': 'horizontal',
        'HorizontalScroll': 'horizontal',
        'Collapsible': 'vertical',
        'Grid': 'grid'
    }
    
    # Widget display symbols
    SYMBOLS = {
        'Button': '🔘',
        'Input': '📝',
        'TextArea': '📄',
        'Select': '📋',
        'Checkbox': '☑️',
        'Label': '📍',
        'Static': '📍',
        'ListView': '📑',
        'LoadingIndicator': '⏳',
        'ProgressBar': '📊',
        'default': '▫️'
    }
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.source_code = self.file_path.read_text()
        
    def visualize(self) -> str:
        """Create a clean visualization."""
        tree = ast.parse(self.source_code)
        widgets = self._extract_widgets(tree)
        
        output = []
        output.append("=" * 60)
        output.append(f"Layout: {self.file_path.name}")
        output.append("=" * 60)
        output.append("")
        
        for widget in widgets:
            self._render_widget(widget, output, 0)
        
        output.append("")
        output.append("Legend:")
        output.append("  📦 = Container (Vertical/Horizontal/Scroll)")
        output.append("  🔘 = Button")
        output.append("  📝 = Input Field")
        output.append("  📄 = Text Area")
        output.append("  📋 = Select/Dropdown")
        output.append("  ☑️  = Checkbox")
        output.append("  📍 = Label/Static Text")
        output.append("  📑 = List View")
        output.append("  ➡️  = Horizontal Layout")
        output.append("  ⬇️  = Vertical Layout")
        
        return '\n'.join(output)
    
    def _extract_widgets(self, tree) -> List[Widget]:
        """Extract widgets from AST."""
        widgets = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'compose':
                for stmt in node.body:
                    widget = self._process_statement(stmt)
                    if widget:
                        widgets.append(widget)
        
        return widgets
    
    def _process_statement(self, stmt) -> Optional[Widget]:
        """Process a statement to extract widget."""
        if isinstance(stmt, ast.With):
            # Container widget
            for item in stmt.items:
                widget = self._extract_widget_from_call(item.context_expr)
                if widget:
                    # Process children
                    for child_stmt in stmt.body:
                        child = self._process_statement(child_stmt)
                        if child:
                            widget.children.append(child)
                    return widget
                    
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
            # Regular widget
            if stmt.value.value:
                return self._extract_widget_from_call(stmt.value.value)
        
        return None
    
    def _extract_widget_from_call(self, node) -> Optional[Widget]:
        """Extract widget info from a call node."""
        if not isinstance(node, ast.Call):
            return None
        
        # Get widget type
        widget_type = None
        if isinstance(node.func, ast.Name):
            widget_type = node.func.id
        elif isinstance(node.func, ast.Attribute):
            widget_type = node.func.attr
        
        if not widget_type:
            return None
        
        widget = Widget(type=widget_type)
        
        # Check if container
        for container_type, layout in self.CONTAINERS.items():
            if widget_type == container_type:
                widget.is_container = True
                widget.layout = layout
                break
        
        # Extract properties
        for i, arg in enumerate(node.args):
            if i == 0 and isinstance(arg, ast.Constant):
                widget.text = str(arg.value)
        
        for kw in node.keywords:
            if kw.arg == 'id' and isinstance(kw.value, ast.Constant):
                widget.id = str(kw.value.value)
        
        return widget
    
    def _render_widget(self, widget: Widget, output: List[str], depth: int):
        """Render a widget with proper indentation."""
        indent = "  " * depth
        
        # Get symbol
        if widget.is_container:
            symbol = "📦"
            layout_indicator = "➡️" if widget.layout == "horizontal" else "⬇️"
        else:
            symbol = self.SYMBOLS.get(widget.type, self.SYMBOLS['default'])
        
        # Build label
        label_parts = [symbol, widget.type]
        
        if widget.id:
            # Shorten long IDs
            id_display = widget.id
            if len(id_display) > 20:
                parts = id_display.split('-')
                if len(parts) > 2:
                    id_display = f"{parts[0]}-...-{parts[-1]}"
            label_parts.append(f"#{id_display}")
        
        if widget.text and not widget.is_container:
            text_preview = widget.text[:25]
            if len(widget.text) > 25:
                text_preview += "..."
            label_parts.append(f'"{text_preview}"')
        
        label = " ".join(label_parts)
        
        # Add layout indicator for containers
        if widget.is_container:
            label += f" {layout_indicator}"
        
        output.append(f"{indent}{label}")
        
        # Render children
        if widget.children:
            if widget.layout == "horizontal":
                # Show horizontal children in a compact way
                self._render_horizontal_children(widget, output, depth + 1)
            else:
                # Vertical children
                for child in widget.children:
                    self._render_widget(child, output, depth + 1)
    
    def _render_horizontal_children(self, widget: Widget, output: List[str], depth: int):
        """Render horizontal layout children in a compact format."""
        indent = "  " * depth
        
        # Group children by row (assuming max 3 per row for readability)
        rows = []
        current_row = []
        
        for child in widget.children:
            current_row.append(child)
            if len(current_row) >= 3:
                rows.append(current_row)
                current_row = []
        
        if current_row:
            rows.append(current_row)
        
        # Render each row
        for row in rows:
            row_items = []
            for child in row:
                if child.is_container:
                    item = f"[{child.type}...]"
                else:
                    symbol = self.SYMBOLS.get(child.type, self.SYMBOLS['default'])
                    if child.id:
                        item = f"{symbol}{child.id.split('-')[-1]}"
                    elif child.text:
                        item = f"{symbol}{child.text[:10]}"
                    else:
                        item = f"{symbol}{child.type}"
                row_items.append(item)
            
            output.append(f"{indent}├─ {' | '.join(row_items)}")
        
        # If any children are containers, show them expanded below
        for child in widget.children:
            if child.is_container and child.children:
                output.append(f"{indent}└─ {child.type} details:")
                self._render_widget(child, output, depth + 1)


def create_markdown_visualization(file_path: str) -> str:
    """Create a markdown visualization with mermaid diagram."""
    visualizer = CleanLayoutVisualizer(file_path)
    tree = ast.parse(visualizer.source_code)
    widgets = visualizer._extract_widgets(tree)
    
    output = [f"# UI Layout: {Path(file_path).name}\n"]
    
    # Add text visualization
    output.append("## Layout Structure\n")
    output.append("```")
    output.append(visualizer.visualize())
    output.append("```\n")
    
    # Add mermaid diagram
    output.append("## Flow Diagram\n")
    output.append("```mermaid")
    output.append("graph TD")
    output.append("    A[App Window] --> B[Main Container]")
    
    node_counter = 0
    
    def add_mermaid_nodes(widget, parent_id, counter):
        node_id = f"N{counter}"
        counter += 1
        
        # Create node label
        if widget.is_container:
            label = f"{widget.type}"
            shape_start, shape_end = "[", "]"
        else:
            label = widget.type
            if widget.id:
                label += f"\\n#{widget.id.split('-')[-1]}"
            shape_start, shape_end = "(", ")"
        
        output.append(f"    {node_id}{shape_start}{label}{shape_end}")
        output.append(f"    {parent_id} --> {node_id}")
        
        # Add children
        for child in widget.children:
            counter = add_mermaid_nodes(child, node_id, counter)
        
        return counter
    
    parent = "B"
    for widget in widgets:
        node_counter = add_mermaid_nodes(widget, parent, node_counter)
    
    output.append("```\n")
    
    return '\n'.join(output)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_layout_clean.py <ui_file> [format]")
        print("\nFormats: text (default), markdown")
        print("\nExample:")
        print("  python visualize_layout_clean.py tldw_chatbook/Widgets/IngestLocalVideoWindow.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    format = sys.argv[2] if len(sys.argv) > 2 else "text"
    
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    try:
        if format == "markdown":
            output = create_markdown_visualization(file_path)
            output_path = Path(file_path).with_suffix('.layout.md')
            output_path.write_text(output)
            print(f"Markdown visualization saved to: {output_path}")
        else:
            visualizer = CleanLayoutVisualizer(file_path)
            output = visualizer.visualize()
            print(output)
            
            # Save to file
            output_path = Path(file_path).with_suffix('.clean_layout.txt')
            output_path.write_text(output)
            print(f"\nVisualization also saved to: {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()