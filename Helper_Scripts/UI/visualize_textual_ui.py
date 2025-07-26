#!/usr/bin/env python3
"""
Textual UI Layout Visualizer

This script parses Textual UI files and generates visual representations
of the widget hierarchy and layout.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import re
from dataclasses import dataclass, field
import html


@dataclass
class Widget:
    """Represents a Textual widget in the UI hierarchy."""
    type: str
    id: Optional[str] = None
    classes: Optional[str] = None
    text: Optional[str] = None
    children: List['Widget'] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    parent_type: Optional[str] = None
    indent_level: int = 0


class TextualUIParser:
    """Parses Textual UI files to extract widget hierarchy."""
    
    # Common Textual containers
    CONTAINER_WIDGETS = {
        'Vertical', 'Horizontal', 'Container', 'VerticalScroll', 
        'HorizontalScroll', 'ScrollableContainer', 'Grid', 
        'Collapsible', 'TabbedContent', 'TabPane'
    }
    
    # Widgets that typically don't have children
    LEAF_WIDGETS = {
        'Static', 'Label', 'Button', 'Input', 'TextArea', 'Select',
        'Checkbox', 'RadioButton', 'LoadingIndicator', 'ProgressBar',
        'ListView', 'ListItem', 'DataTable', 'Tree', 'Switch'
    }
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.source_code = self.file_path.read_text()
        self.tree = ast.parse(self.source_code)
        self.widgets: List[Widget] = []
        self.current_context_stack: List[str] = []
        
    def parse(self) -> List[Widget]:
        """Parse the file and extract widget hierarchy."""
        # Find compose methods
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'compose':
                self.widgets = self._parse_compose_method(node)
                break
        return self.widgets
    
    def _parse_compose_method(self, node: ast.FunctionDef) -> List[Widget]:
        """Parse the compose method to extract widget hierarchy."""
        widgets = []
        for stmt in node.body:
            if isinstance(stmt, ast.With):
                widgets.extend(self._parse_with_statement(stmt, 0))
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                widget = self._parse_yield_statement(stmt.value, 0)
                if widget:
                    widgets.append(widget)
        return widgets
    
    def _parse_with_statement(self, node: ast.With, indent: int) -> List[Widget]:
        """Parse a with statement (context manager)."""
        widgets = []
        
        for item in node.items:
            context_widget = self._parse_expression(item.context_expr)
            if context_widget:
                context_widget.indent_level = indent
                # Parse children in the with block
                for stmt in node.body:
                    if isinstance(stmt, ast.With):
                        context_widget.children.extend(self._parse_with_statement(stmt, indent + 1))
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                        child = self._parse_yield_statement(stmt.value, indent + 1)
                        if child:
                            child.parent_type = context_widget.type
                            context_widget.children.append(child)
                widgets.append(context_widget)
        
        return widgets
    
    def _parse_yield_statement(self, node: ast.Yield, indent: int) -> Optional[Widget]:
        """Parse a yield statement."""
        if node.value:
            widget = self._parse_expression(node.value)
            if widget:
                widget.indent_level = indent
            return widget
        return None
    
    def _parse_expression(self, node: ast.expr) -> Optional[Widget]:
        """Parse an expression that creates a widget."""
        if isinstance(node, ast.Call):
            widget_type = self._get_widget_type(node.func)
            if widget_type:
                widget = Widget(type=widget_type)
                
                # Parse arguments
                for i, arg in enumerate(node.args):
                    if i == 0 and isinstance(arg, ast.Constant):
                        # First positional argument is often text content
                        widget.text = str(arg.value)
                    elif isinstance(arg, ast.Constant):
                        widget.attributes[f'arg{i}'] = str(arg.value)
                
                # Parse keyword arguments
                for keyword in node.keywords:
                    if keyword.arg == 'id' and isinstance(keyword.value, ast.Constant):
                        widget.id = str(keyword.value.value)
                    elif keyword.arg == 'classes' and isinstance(keyword.value, ast.Constant):
                        widget.classes = str(keyword.value.value)
                    elif keyword.arg and isinstance(keyword.value, ast.Constant):
                        widget.attributes[keyword.arg] = str(keyword.value.value)
                
                return widget
        
        return None
    
    def _get_widget_type(self, node: ast.expr) -> Optional[str]:
        """Extract widget type from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None


class SVGRenderer:
    """Renders widget hierarchy as SVG."""
    
    # Widget dimensions
    WIDGET_HEIGHT = 40
    WIDGET_MIN_WIDTH = 150
    WIDGET_PADDING = 10
    INDENT_WIDTH = 30
    VERTICAL_SPACING = 10
    
    # Colors
    COLORS = {
        'Vertical': '#e8f5e9',
        'Horizontal': '#e3f2fd',
        'Container': '#f3e5f5',
        'VerticalScroll': '#e0f2f1',
        'HorizontalScroll': '#fce4ec',
        'Collapsible': '#fff3e0',
        'Static': '#f5f5f5',
        'Label': '#eeeeee',
        'Button': '#90caf9',
        'Input': '#a5d6a7',
        'TextArea': '#81c784',
        'Select': '#4fc3f7',
        'Checkbox': '#ba68c8',
        'ListView': '#ffb74d',
        'LoadingIndicator': '#ff8a65',
        'default': '#e0e0e0'
    }
    
    def __init__(self, widgets: List[Widget]):
        self.widgets = widgets
        self.svg_content = []
        self.current_y = 20
        self.max_width = 0
        
    def render(self) -> str:
        """Render widgets as SVG."""
        # Start SVG
        self.svg_content.append('<?xml version="1.0" encoding="UTF-8"?>')
        
        # Calculate dimensions first
        for widget in self.widgets:
            self._calculate_dimensions(widget)
        
        # Add some padding to max_width
        self.max_width += 40
        total_height = self.current_y + 40
        
        # SVG header with calculated dimensions
        self.svg_content.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.max_width}" height="{total_height}" '
            f'viewBox="0 0 {self.max_width} {total_height}">'
        )
        
        # Add styles
        self._add_styles()
        
        # Reset for actual rendering
        self.current_y = 20
        
        # Render widgets
        for widget in self.widgets:
            self._render_widget(widget)
        
        # Close SVG
        self.svg_content.append('</svg>')
        
        return '\n'.join(self.svg_content)
    
    def _add_styles(self):
        """Add CSS styles to SVG."""
        styles = """
        <style>
            .widget-rect { stroke: #333; stroke-width: 1; }
            .widget-text { font-family: Arial, sans-serif; font-size: 12px; fill: #333; }
            .widget-id { font-size: 10px; fill: #666; }
            .widget-class { font-size: 10px; fill: #999; font-style: italic; }
            .container-rect { stroke-width: 2; stroke-dasharray: 5,5; }
        </style>
        """
        self.svg_content.append(styles)
    
    def _calculate_dimensions(self, widget: Widget):
        """Calculate dimensions needed for the widget and its children."""
        x = 20 + (widget.indent_level * self.INDENT_WIDTH)
        width = self._get_widget_width(widget)
        
        # Update max width
        if x + width > self.max_width:
            self.max_width = x + width
        
        # Account for this widget's height
        self.current_y += self.WIDGET_HEIGHT + self.VERTICAL_SPACING
        
        # Recursively calculate for children
        for child in widget.children:
            self._calculate_dimensions(child)
    
    def _render_widget(self, widget: Widget):
        """Render a single widget and its children."""
        x = 20 + (widget.indent_level * self.INDENT_WIDTH)
        y = self.current_y
        width = self._get_widget_width(widget)
        height = self.WIDGET_HEIGHT
        
        # Get color
        color = self.COLORS.get(widget.type, self.COLORS['default'])
        
        # Determine if this is a container
        is_container = widget.type in TextualUIParser.CONTAINER_WIDGETS or widget.children
        
        # Draw rectangle
        rect_class = "widget-rect container-rect" if is_container else "widget-rect"
        self.svg_content.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'fill="{color}" class="{rect_class}" rx="5" />'
        )
        
        # Add widget type text
        text_y = y + 20
        self.svg_content.append(
            f'<text x="{x + 10}" y="{text_y}" class="widget-text">{html.escape(widget.type)}</text>'
        )
        
        # Add ID if present
        if widget.id:
            self.svg_content.append(
                f'<text x="{x + 10}" y="{text_y + 15}" class="widget-id">#{html.escape(widget.id)}</text>'
            )
        
        # Add classes if present
        if widget.classes:
            class_x = x + width - 10
            self.svg_content.append(
                f'<text x="{class_x}" y="{text_y}" class="widget-class" text-anchor="end">.{html.escape(widget.classes)}</text>'
            )
        
        # Add text content if present
        if widget.text:
            text_preview = widget.text[:30] + '...' if len(widget.text) > 30 else widget.text
            text_x = x + width / 2
            self.svg_content.append(
                f'<text x="{text_x}" y="{text_y + 15}" class="widget-id" text-anchor="middle">'
                f'"{html.escape(text_preview)}"</text>'
            )
        
        # Update current Y position
        self.current_y += height + self.VERTICAL_SPACING
        
        # Render children
        if widget.children:
            # Draw a connecting line for containers
            line_x = x + 10
            line_y1 = y + height
            line_y2 = self.current_y - self.VERTICAL_SPACING
            
            for child in widget.children:
                self._render_widget(child)
            
            # Draw vertical line connecting container to its children
            if len(widget.children) > 0:
                self.svg_content.append(
                    f'<line x1="{line_x}" y1="{line_y1}" x2="{line_x}" y2="{line_y2}" '
                    f'stroke="#999" stroke-width="1" stroke-dasharray="2,2" />'
                )
    
    def _get_widget_width(self, widget: Widget) -> int:
        """Calculate widget width based on content."""
        base_width = self.WIDGET_MIN_WIDTH
        
        # Add width for ID
        if widget.id:
            base_width = max(base_width, len(widget.id) * 7 + 40)
        
        # Add width for text
        if widget.text:
            text_preview = widget.text[:30]
            base_width = max(base_width, len(text_preview) * 7 + 40)
        
        # Containers get extra width
        if widget.type in TextualUIParser.CONTAINER_WIDGETS:
            base_width += 30
        
        return base_width


def create_html_wrapper(svg_content: str, file_path: str) -> str:
    """Create an HTML wrapper for the SVG with metadata."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Textual UI Visualization - {Path(file_path).name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .metadata {{
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .legend h3 {{
            margin-top: 0;
            color: #555;
        }}
        .legend-item {{
            display: inline-block;
            margin: 5px 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 1px solid #333;
            vertical-align: middle;
            margin-right: 5px;
        }}
        svg {{
            border: 1px solid #ddd;
            background-color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Textual UI Layout Visualization</h1>
        <div class="metadata">
            <strong>File:</strong> {Path(file_path).name}<br>
            <strong>Path:</strong> {file_path}
        </div>
        
        {svg_content}
        
        <div class="legend">
            <h3>Widget Types</h3>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #e8f5e9;"></span>
                Vertical
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #e3f2fd;"></span>
                Horizontal
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #90caf9;"></span>
                Button
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #a5d6a7;"></span>
                Input
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #4fc3f7;"></span>
                Select
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #ba68c8;"></span>
                Checkbox
            </div>
        </div>
    </div>
</body>
</html>"""


def visualize_textual_ui(file_path: str, output_path: Optional[str] = None):
    """Main function to visualize a Textual UI file."""
    try:
        # Parse the file
        parser = TextualUIParser(file_path)
        widgets = parser.parse()
        
        if not widgets:
            print(f"No widgets found in {file_path}")
            return
        
        # Render as SVG
        renderer = SVGRenderer(widgets)
        svg_content = renderer.render()
        
        # Determine output path
        if output_path is None:
            input_path = Path(file_path)
            output_path = input_path.parent / f"{input_path.stem}_layout.html"
        
        # Create HTML with SVG
        html_content = create_html_wrapper(svg_content, file_path)
        
        # Write output
        Path(output_path).write_text(html_content)
        print(f"Visualization saved to: {output_path}")
        
        # Also save standalone SVG
        svg_path = Path(output_path).with_suffix('.svg')
        Path(svg_path).write_text(svg_content)
        print(f"SVG saved to: {svg_path}")
        
    except Exception as e:
        print(f"Error visualizing {file_path}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_textual_ui.py <path_to_ui_file> [output_path]")
        print("\nExample:")
        print("  python visualize_textual_ui.py tldw_chatbook/Widgets/IngestLocalVideoWindow.py")
        print("  python visualize_textual_ui.py tldw_chatbook/UI/Chat_Window_Enhanced.py chat_layout.html")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    visualize_textual_ui(file_path, output_path)


if __name__ == "__main__":
    main()