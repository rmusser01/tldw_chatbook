#!/usr/bin/env python3
"""
Simple Textual UI Layout Visualizer

Creates clean ASCII-art style visualizations of Textual UI layouts.
"""

import ast
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import re


class SimpleUIVisualizer:
    """Simple visualizer for Textual UI layouts."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.source_code = self.file_path.read_text()
        self.output_lines = []
        self.indent_stack = []
        
    def visualize(self) -> str:
        """Create a simple text visualization of the UI."""
        # Parse the compose method
        tree = ast.parse(self.source_code)
        
        # Find compose method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'compose':
                self._process_compose(node)
                break
        
        return '\n'.join(self.output_lines)
    
    def _process_compose(self, node: ast.FunctionDef):
        """Process the compose method."""
        self.output_lines.append("=" * 80)
        self.output_lines.append(f"TEXTUAL UI LAYOUT: {self.file_path.name}")
        self.output_lines.append("=" * 80)
        self.output_lines.append("")
        
        for stmt in node.body:
            self._process_statement(stmt, 0)
    
    def _process_statement(self, stmt, indent: int):
        """Process a statement in the compose method."""
        if isinstance(stmt, ast.With):
            self._process_with(stmt, indent)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
            self._process_yield(stmt.value, indent)
    
    def _process_with(self, node: ast.With, indent: int):
        """Process a with statement (container)."""
        for item in node.items:
            widget_info = self._extract_widget_info(item.context_expr)
            if widget_info:
                self._add_widget_line(widget_info, indent, is_container=True)
                
                # Process children
                for child_stmt in node.body:
                    self._process_statement(child_stmt, indent + 1)
                
                # Close container
                self._add_line("└" + "─" * 20, indent)
    
    def _process_yield(self, node: ast.Yield, indent: int):
        """Process a yield statement."""
        if node.value:
            widget_info = self._extract_widget_info(node.value)
            if widget_info:
                self._add_widget_line(widget_info, indent, is_container=False)
    
    def _extract_widget_info(self, node) -> Optional[Dict]:
        """Extract widget information from AST node."""
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
        
        info = {
            'type': widget_type,
            'text': None,
            'id': None,
            'classes': None,
            'attrs': []
        }
        
        # Extract arguments
        for i, arg in enumerate(node.args):
            if i == 0 and isinstance(arg, ast.Constant):
                info['text'] = str(arg.value)
        
        # Extract keyword arguments
        for kw in node.keywords:
            if kw.arg == 'id' and isinstance(kw.value, ast.Constant):
                info['id'] = str(kw.value.value)
            elif kw.arg == 'classes' and isinstance(kw.value, ast.Constant):
                info['classes'] = str(kw.value.value)
            elif kw.arg and isinstance(kw.value, ast.Constant):
                info['attrs'].append(f"{kw.arg}={kw.value.value}")
        
        return info
    
    def _add_widget_line(self, info: Dict, indent: int, is_container: bool):
        """Add a widget line to the output."""
        # Build the widget representation
        parts = []
        
        # Widget type
        if is_container:
            parts.append(f"[{info['type']}]")
        else:
            parts.append(f"{info['type']}")
        
        # ID
        if info['id']:
            parts.append(f"#{info['id']}")
        
        # Classes
        if info['classes']:
            parts.append(f".{info['classes']}")
        
        # Text content
        if info['text']:
            text_preview = info['text'][:40] + '...' if len(info['text']) > 40 else info['text']
            parts.append(f'"{text_preview}"')
        
        # Additional attributes
        if info['attrs']:
            parts.append(f"({', '.join(info['attrs'][:3])})")
        
        line = " ".join(parts)
        
        # Add tree structure
        if indent > 0:
            prefix = "│   " * (indent - 1) + "├── "
        else:
            prefix = ""
        
        self._add_line(prefix + line, 0)
    
    def _add_line(self, text: str, indent: int):
        """Add a line to the output."""
        self.output_lines.append("  " * indent + text)


def create_detailed_visualization(file_path: str) -> str:
    """Create a more detailed markdown visualization."""
    visualizer = SimpleUIVisualizer(file_path)
    tree_view = visualizer.visualize()
    
    # Create markdown output
    output = f"""# Textual UI Layout Visualization

**File:** `{Path(file_path).name}`  
**Path:** `{file_path}`

## Widget Tree Structure

```
{tree_view}
```

## Layout Description

This visualization shows the hierarchical structure of widgets in the Textual UI:

- **Containers** are shown in `[brackets]` - these can contain other widgets
- **Widget IDs** are prefixed with `#` 
- **CSS Classes** are prefixed with `.`
- **Text content** is shown in quotes
- **Additional attributes** are shown in parentheses

### Common Widget Types:

- `Vertical` / `Horizontal` - Layout containers
- `VerticalScroll` / `HorizontalScroll` - Scrollable containers
- `Static` / `Label` - Text display widgets
- `Input` / `TextArea` - Text input widgets
- `Button` - Interactive buttons
- `Select` - Dropdown selection
- `Checkbox` - Toggle options
- `ListView` / `ListItem` - List displays
- `Collapsible` - Expandable sections

"""
    
    return output


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_ui_simple.py <path_to_ui_file> [output_format]")
        print("\nOutput formats: text (default), markdown")
        print("\nExample:")
        print("  python visualize_ui_simple.py tldw_chatbook/Widgets/IngestLocalVideoWindow.py")
        print("  python visualize_ui_simple.py tldw_chatbook/UI/Chat_Window_Enhanced.py markdown")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "text"
    
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    try:
        if output_format == "markdown":
            output = create_detailed_visualization(file_path)
            output_path = Path(file_path).with_suffix('.layout.md')
            output_path.write_text(output)
            print(f"Markdown visualization saved to: {output_path}")
        else:
            visualizer = SimpleUIVisualizer(file_path)
            output = visualizer.visualize()
            print(output)
            
            # Also save to file
            output_path = Path(file_path).with_suffix('.layout.txt')
            output_path.write_text(output)
            print(f"\nVisualization also saved to: {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()