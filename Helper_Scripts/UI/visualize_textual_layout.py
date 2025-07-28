#!/usr/bin/env python3
"""
Textual UI Layout Visualizer with Box Layout

Creates visual representations showing how widgets will be laid out in the UI,
with proper spacing and positioning to understand the visual structure.
"""

import ast
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import html


@dataclass
class LayoutWidget:
    """Represents a widget with layout information."""
    type: str
    id: Optional[str] = None
    classes: Optional[str] = None
    text: Optional[str] = None
    children: List['LayoutWidget'] = field(default_factory=list)
    layout_type: str = "vertical"  # vertical, horizontal, grid
    width_hint: str = "auto"  # auto, full, half, third
    height_hint: str = "auto"  # auto, full, compact
    collapsible: bool = False
    collapsed: bool = False


class LayoutAnalyzer:
    """Analyzes Textual UI files to understand layout structure."""
    
    # Layout containers
    VERTICAL_CONTAINERS = {'Vertical', 'VerticalScroll', 'Container'}
    HORIZONTAL_CONTAINERS = {'Horizontal', 'HorizontalScroll'}
    
    # Widget size hints based on type
    WIDGET_SIZES = {
        'Button': ('auto', 'compact'),
        'Input': ('full', 'compact'),
        'TextArea': ('full', 'medium'),
        'Select': ('full', 'compact'),
        'Label': ('auto', 'compact'),
        'Static': ('auto', 'compact'),
        'Checkbox': ('auto', 'compact'),
        'ListView': ('full', 'large'),
        'LoadingIndicator': ('auto', 'compact'),
        'Collapsible': ('full', 'auto'),
    }
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.source_code = self.file_path.read_text()
        self.tree = ast.parse(self.source_code)
        
    def analyze(self) -> List[LayoutWidget]:
        """Analyze the file and extract layout structure."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'compose':
                return self._analyze_compose(node)
        return []
    
    def _analyze_compose(self, node: ast.FunctionDef) -> List[LayoutWidget]:
        """Analyze the compose method."""
        widgets = []
        for stmt in node.body:
            if isinstance(stmt, ast.With):
                widgets.extend(self._analyze_with(stmt))
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                widget = self._analyze_yield(stmt.value)
                if widget:
                    widgets.append(widget)
        return widgets
    
    def _analyze_with(self, node: ast.With) -> List[LayoutWidget]:
        """Analyze a with statement (container)."""
        widgets = []
        
        for item in node.items:
            widget = self._extract_widget(item.context_expr)
            if widget:
                # Determine layout type
                if widget.type in self.HORIZONTAL_CONTAINERS:
                    widget.layout_type = "horizontal"
                elif widget.type in self.VERTICAL_CONTAINERS:
                    widget.layout_type = "vertical"
                
                # Process children
                for stmt in node.body:
                    if isinstance(stmt, ast.With):
                        widget.children.extend(self._analyze_with(stmt))
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                        child = self._analyze_yield(stmt.value)
                        if child:
                            widget.children.append(child)
                
                widgets.append(widget)
        
        return widgets
    
    def _analyze_yield(self, node: ast.Yield) -> Optional[LayoutWidget]:
        """Analyze a yield statement."""
        if node.value:
            return self._extract_widget(node.value)
        return None
    
    def _extract_widget(self, node: ast.expr) -> Optional[LayoutWidget]:
        """Extract widget information with layout hints."""
        if not isinstance(node, ast.Call):
            return None
        
        # Get widget type
        widget_type = self._get_widget_type(node.func)
        if not widget_type:
            return None
        
        widget = LayoutWidget(type=widget_type)
        
        # Extract properties
        for i, arg in enumerate(node.args):
            if i == 0 and isinstance(arg, ast.Constant):
                widget.text = str(arg.value)
        
        for kw in node.keywords:
            if kw.arg == 'id' and isinstance(kw.value, ast.Constant):
                widget.id = str(kw.value.value)
            elif kw.arg == 'classes' and isinstance(kw.value, ast.Constant):
                widget.classes = str(kw.value.value)
            elif kw.arg == 'collapsed' and isinstance(kw.value, ast.Constant):
                widget.collapsed = bool(kw.value.value)
        
        # Set size hints
        if widget_type in self.WIDGET_SIZES:
            widget.width_hint, widget.height_hint = self.WIDGET_SIZES[widget_type]
        
        # Special handling for Collapsible
        if widget_type == 'Collapsible':
            widget.collapsible = True
        
        return widget
    
    def _get_widget_type(self, node: ast.expr) -> Optional[str]:
        """Extract widget type from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None


class BoxLayoutRenderer:
    """Renders widget layout as ASCII boxes showing actual layout structure."""
    
    def __init__(self, widgets: List[LayoutWidget]):
        self.widgets = widgets
        self.output = []
        
    def render(self) -> str:
        """Render the layout as ASCII boxes."""
        self.output = []
        self._add_header()
        
        # Render each top-level widget
        for widget in self.widgets:
            self._render_widget(widget, 0, full_width=True)
            self.output.append("")  # Space between top-level widgets
        
        return '\n'.join(self.output)
    
    def _add_header(self):
        """Add header to output."""
        self.output.append("=" * 80)
        self.output.append("TEXTUAL UI LAYOUT VISUALIZATION")
        self.output.append("=" * 80)
        self.output.append("")
        self.output.append("Legend:")
        self.output.append("  [====] Full width widgets")
        self.output.append("  [==]   Auto/compact width widgets")
        self.output.append("  |---|  Horizontal layout containers")
        self.output.append("  [...]  Collapsed sections")
        self.output.append("")
        self.output.append("-" * 80)
        self.output.append("")
    
    def _render_widget(self, widget: LayoutWidget, indent: int, full_width: bool = False):
        """Render a single widget and its children."""
        prefix = "  " * indent
        
        # Handle collapsed widgets
        if widget.collapsible and widget.collapsed:
            self._render_collapsed_widget(widget, indent)
            return
        
        # Determine box width
        if full_width or widget.width_hint == "full":
            box_width = 76 - (indent * 2)
        elif widget.width_hint == "half":
            box_width = 35 - (indent * 2)
        elif widget.width_hint == "third":
            box_width = 23 - (indent * 2)
        else:  # auto
            box_width = max(20, len(self._get_widget_label(widget)) + 4)
        
        # Ensure minimum width
        box_width = max(box_width, 10)
        
        # Render based on widget type
        if widget.children:
            self._render_container(widget, indent, box_width)
        else:
            self._render_leaf_widget(widget, indent, box_width)
    
    def _render_container(self, widget: LayoutWidget, indent: int, box_width: int):
        """Render a container widget with children."""
        prefix = "  " * indent
        label = self._get_widget_label(widget)
        
        # Top border
        self.output.append(f"{prefix}┌{'─' * (box_width - 2)}┐")
        
        # Container label
        label_line = f"│ {label:<{box_width - 4}} │"
        self.output.append(f"{prefix}{label_line}")
        
        # Separator
        self.output.append(f"{prefix}├{'─' * (box_width - 2)}┤")
        
        # Render children based on layout type
        if widget.layout_type == "horizontal":
            self._render_horizontal_children(widget, indent, box_width)
        else:
            self._render_vertical_children(widget, indent, box_width)
        
        # Bottom border
        self.output.append(f"{prefix}└{'─' * (box_width - 2)}┘")
    
    def _render_horizontal_children(self, widget: LayoutWidget, indent: int, box_width: int):
        """Render children in horizontal layout."""
        if not widget.children:
            return
        
        prefix = "  " * indent
        
        # For horizontal layout, show children side by side
        # Simplified: show them in a row representation
        children_repr = []
        for child in widget.children:
            child_label = self._get_widget_label(child, compact=True)
            if child.children:
                children_repr.append(f"[{child_label}...]")
            else:
                children_repr.append(f"[{child_label}]")
        
        # Split into lines if too long
        line = " ".join(children_repr)
        if len(line) > box_width - 4:
            # Split into multiple lines
            current_line = ""
            for repr in children_repr:
                if len(current_line) + len(repr) + 1 > box_width - 4:
                    self.output.append(f"{prefix}│ {current_line:<{box_width - 4}} │")
                    current_line = repr
                else:
                    current_line = f"{current_line} {repr}" if current_line else repr
            if current_line:
                self.output.append(f"{prefix}│ {current_line:<{box_width - 4}} │")
        else:
            self.output.append(f"{prefix}│ {line:<{box_width - 4}} │")
    
    def _render_vertical_children(self, widget: LayoutWidget, indent: int, box_width: int):
        """Render children in vertical layout."""
        prefix = "  " * indent
        
        for i, child in enumerate(widget.children):
            # Add spacing between children
            if i > 0:
                self.output.append(f"{prefix}│{' ' * (box_width - 2)}│")
            
            # Render child inline if it's simple
            if not child.children and child.height_hint == "compact":
                child_label = self._get_widget_label(child)
                if child.width_hint == "full":
                    # Full width widgets
                    label_width = box_width - 6
                    if len(child_label) > label_width:
                        child_label = child_label[:label_width-3] + "..."
                    self.output.append(f"{prefix}│ [{child_label:=^{label_width}}] │")
                else:
                    # Auto width widgets
                    max_label_width = box_width - 5
                    if len(child_label) > max_label_width:
                        child_label = child_label[:max_label_width-3] + "..."
                    padding = box_width - len(child_label) - 5
                    self.output.append(f"{prefix}│ [{child_label}]{' ' * padding} │")
            else:
                # For nested containers, render them inline with simplified view
                self._render_nested_container(child, indent, box_width)
    
    def _render_nested_container(self, widget: LayoutWidget, indent: int, parent_width: int):
        """Render a nested container in a simplified inline format."""
        prefix = "  " * indent
        
        # Create a simplified representation
        label = self._get_widget_label(widget, compact=True)
        
        # For deeply nested containers, show a summary
        if widget.layout_type == "horizontal":
            # Show horizontal containers with their children in a row
            children_summary = []
            for child in widget.children[:3]:  # Show first 3 children
                child_label = self._get_widget_label(child, compact=True)
                if len(child_label) > 15:
                    child_label = child_label[:12] + "..."
                children_summary.append(child_label)
            if len(widget.children) > 3:
                children_summary.append("...")
            
            content = f"[{label}] → {' | '.join(children_summary)}"
        else:
            # For vertical containers, show count of children
            content = f"[{label}] ({len(widget.children)} items)"
        
        # Ensure content fits
        max_width = parent_width - 4
        if len(content) > max_width:
            content = content[:max_width-3] + "..."
        
        self.output.append(f"{prefix}│ {content:<{parent_width - 4}} │")
        
        # If it's a small container with few children, show them indented
        if len(widget.children) <= 3 and widget.layout_type == "vertical":
            for child in widget.children:
                child_label = self._get_widget_label(child, compact=True)
                child_content = f"  └─ {child_label}"
                if len(child_content) > max_width:
                    child_content = child_content[:max_width-3] + "..."
                self.output.append(f"{prefix}│ {child_content:<{parent_width - 4}} │")
    
    def _render_leaf_widget(self, widget: LayoutWidget, indent: int, box_width: int):
        """Render a leaf widget (no children)."""
        prefix = "  " * indent
        label = self._get_widget_label(widget)
        
        if widget.height_hint == "large":
            # Larger widgets (like ListView, TextArea)
            self.output.append(f"{prefix}┌{'─' * (box_width - 2)}┐")
            self.output.append(f"{prefix}│ {label:<{box_width - 4}} │")
            self.output.append(f"{prefix}│{' ' * (box_width - 2)}│")
            self.output.append(f"{prefix}│{'.' * (box_width - 2)}│")
            self.output.append(f"{prefix}│{'.' * (box_width - 2)}│")
            self.output.append(f"{prefix}│{'.' * (box_width - 2)}│")
            self.output.append(f"{prefix}└{'─' * (box_width - 2)}┘")
        elif widget.height_hint == "medium":
            # Medium widgets (like TextArea)
            self.output.append(f"{prefix}┌{'─' * (box_width - 2)}┐")
            self.output.append(f"{prefix}│ {label:<{box_width - 4}} │")
            self.output.append(f"{prefix}│{'.' * (box_width - 2)}│")
            self.output.append(f"{prefix}│{'.' * (box_width - 2)}│")
            self.output.append(f"{prefix}└{'─' * (box_width - 2)}┘")
        else:
            # Compact widgets (buttons, inputs, labels)
            if widget.type in ['Button', 'Input', 'Select']:
                self.output.append(f"{prefix}[{label:^{box_width - 2}}]")
            elif widget.type in ['Checkbox']:
                self.output.append(f"{prefix}☐ {label}")
            elif widget.type in ['Label', 'Static']:
                self.output.append(f"{prefix}{label}")
            else:
                self.output.append(f"{prefix}[{label}]")
    
    def _render_collapsed_widget(self, widget: LayoutWidget, indent: int):
        """Render a collapsed widget."""
        prefix = "  " * indent
        label = self._get_widget_label(widget)
        self.output.append(f"{prefix}[▶ {label} ...]")
    
    def _get_widget_label(self, widget: LayoutWidget, compact: bool = False) -> str:
        """Get a label for the widget."""
        parts = []
        
        # Always include type for compact mode, but shortened
        if compact:
            # Shorten common widget types
            type_map = {
                'Horizontal': 'H',
                'Vertical': 'V',
                'Container': 'C',
                'VerticalScroll': 'VS',
                'HorizontalScroll': 'HS',
                'Button': 'Btn',
                'Checkbox': 'Check',
                'TextArea': 'Text',
                'Static': 'S',
                'Label': 'L',
                'LoadingIndicator': 'Load',
                'Collapsible': 'Coll'
            }
            parts.append(type_map.get(widget.type, widget.type[:4]))
        else:
            parts.append(widget.type)
        
        if widget.id:
            if compact:
                # Shorten IDs for compact view
                id_parts = widget.id.split('-')
                if len(id_parts) > 2:
                    # Take first and last part
                    short_id = f"{id_parts[0][:3]}-{id_parts[-1][:4]}"
                else:
                    short_id = widget.id[:8]
                parts.append(f"#{short_id}")
            else:
                parts.append(f"#{widget.id}")
        elif widget.text:
            if compact:
                text_preview = widget.text[:15]
            else:
                text_preview = widget.text[:30]
            if len(widget.text) > len(text_preview):
                text_preview += "..."
            parts.append(f'"{text_preview}"')
        
        return " ".join(parts) if parts else widget.type


class HTMLLayoutRenderer:
    """Renders widget layout as HTML with CSS for better visualization."""
    
    def __init__(self, widgets: List[LayoutWidget]):
        self.widgets = widgets
        
    def render(self) -> str:
        """Render the layout as HTML."""
        html_parts = [self._get_html_header()]
        
        html_parts.append('<div class="app-container">')
        for widget in self.widgets:
            html_parts.append(self._render_widget_html(widget))
        html_parts.append('</div>')
        
        html_parts.append(self._get_html_footer())
        return '\n'.join(html_parts)
    
    def _get_html_header(self) -> str:
        """Get HTML header with styles."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Textual UI Layout Visualization</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1e1e1e;
            color: #d4d4d4;
        }
        
        .app-container {
            max-width: 1200px;
            margin: 0 auto;
            background: #252526;
            border: 1px solid #3e3e42;
            border-radius: 8px;
            padding: 20px;
        }
        
        .widget {
            margin: 5px 0;
            padding: 10px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background: #2d2d30;
        }
        
        .widget-container {
            border: 2px dashed #007acc;
            background: #1e1e1e;
        }
        
        .widget-horizontal {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .widget-horizontal > .widget {
            flex: 1;
            min-width: 200px;
        }
        
        .widget-label {
            font-size: 12px;
            color: #608b4e;
            margin-bottom: 5px;
        }
        
        .widget-id {
            color: #9cdcfe;
            font-size: 11px;
        }
        
        .widget-text {
            color: #ce9178;
            font-style: italic;
            font-size: 11px;
        }
        
        .widget-button {
            background: #0e639c;
            color: white;
            text-align: center;
            padding: 8px 16px;
            cursor: pointer;
        }
        
        .widget-input, .widget-select {
            background: #3c3c3c;
            border: 1px solid #464647;
            padding: 6px 10px;
        }
        
        .widget-textarea {
            background: #3c3c3c;
            border: 1px solid #464647;
            min-height: 60px;
            padding: 6px 10px;
        }
        
        .widget-checkbox {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .widget-collapsed {
            background: #2d2d30;
            padding: 8px 12px;
            cursor: pointer;
        }
        
        .widget-collapsed::before {
            content: "▶ ";
        }
        
        .widget-listview {
            background: #1e1e1e;
            border: 1px solid #464647;
            min-height: 100px;
            padding: 10px;
        }
        
        h1 {
            color: #d4d4d4;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .legend {
            margin-top: 30px;
            padding: 15px;
            background: #1e1e1e;
            border-radius: 4px;
        }
        
        .legend h3 {
            margin-top: 0;
            color: #d4d4d4;
        }
        
        .legend-item {
            display: inline-block;
            margin-right: 20px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <h1>Textual UI Layout Visualization</h1>
"""
    
    def _get_html_footer(self) -> str:
        """Get HTML footer."""
        return """
    <div class="legend">
        <h3>Widget Types</h3>
        <div class="legend-item">🔲 Containers (Vertical/Horizontal)</div>
        <div class="legend-item">🔘 Buttons</div>
        <div class="legend-item">📝 Input Fields</div>
        <div class="legend-item">📄 Text Areas</div>
        <div class="legend-item">☑️ Checkboxes</div>
        <div class="legend-item">📋 Lists</div>
    </div>
</body>
</html>"""
    
    def _render_widget_html(self, widget: LayoutWidget) -> str:
        """Render a widget as HTML."""
        classes = ['widget']
        
        # Add type-specific classes
        if widget.children:
            classes.append('widget-container')
            if widget.layout_type == 'horizontal':
                classes.append('widget-horizontal')
        else:
            widget_type_class = f'widget-{widget.type.lower()}'
            classes.append(widget_type_class)
        
        # Handle collapsed state
        if widget.collapsible and widget.collapsed:
            classes.append('widget-collapsed')
            label = self._get_widget_label_html(widget)
            return f'<div class="{" ".join(classes)}">{label} (collapsed)</div>'
        
        # Build HTML
        html_parts = [f'<div class="{" ".join(classes)}">']
        
        # Add widget info
        label = self._get_widget_label_html(widget)
        html_parts.append(f'<div class="widget-label">{label}</div>')
        
        # Render content based on type
        if widget.children:
            for child in widget.children:
                html_parts.append(self._render_widget_html(child))
        else:
            html_parts.append(self._render_widget_content(widget))
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def _get_widget_label_html(self, widget: LayoutWidget) -> str:
        """Get HTML label for widget."""
        parts = [widget.type]
        if widget.id:
            parts.append(f'<span class="widget-id">#{widget.id}</span>')
        if widget.text:
            text = html.escape(widget.text[:40])
            if len(widget.text) > 40:
                text += '...'
            parts.append(f'<span class="widget-text">"{text}"</span>')
        return ' '.join(parts)
    
    def _render_widget_content(self, widget: LayoutWidget) -> str:
        """Render widget-specific content."""
        if widget.type == 'Button':
            text = widget.text or 'Button'
            return f'<div style="text-align: center">{html.escape(text)}</div>'
        elif widget.type == 'Checkbox':
            text = widget.text or 'Checkbox'
            return f'☐ {html.escape(text)}'
        elif widget.type in ['Input', 'TextArea', 'Select']:
            return '...'
        elif widget.type == 'ListView':
            return '<div style="color: #666">List items...</div>'
        else:
            return ''


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_textual_layout.py <ui_file> [format]")
        print("\nFormats: ascii (default), html")
        print("\nExamples:")
        print("  python visualize_textual_layout.py tldw_chatbook/Widgets/IngestLocalVideoWindow.py")
        print("  python visualize_textual_layout.py tldw_chatbook/UI/Chat_Window.py html")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "ascii"
    
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    try:
        # Analyze the file
        analyzer = LayoutAnalyzer(file_path)
        widgets = analyzer.analyze()
        
        if not widgets:
            print(f"No widgets found in {file_path}")
            return
        
        if output_format == "html":
            # Render as HTML
            renderer = HTMLLayoutRenderer(widgets)
            html_content = renderer.render()
            
            output_path = Path(file_path).with_suffix('.layout.html')
            output_path.write_text(html_content)
            print(f"HTML layout visualization saved to: {output_path}")
        else:
            # Render as ASCII
            renderer = BoxLayoutRenderer(widgets)
            ascii_content = renderer.render()
            
            print(ascii_content)
            
            # Also save to file
            output_path = Path(file_path).with_suffix('.layout.txt')
            output_path.write_text(ascii_content)
            print(f"\nLayout visualization also saved to: {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()