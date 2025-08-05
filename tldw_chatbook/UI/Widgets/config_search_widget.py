# config_search_widget.py
# Widget for searching through UI form elements in configuration settings
#
from typing import Any, Dict, List, Tuple, Optional, Union
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, Input, ListItem, Label, Checkbox, Select, TextArea
from textual.widget import Widget
from textual.message import Message
from textual.dom import DOMNode
from loguru import logger


class ConfigSearchResult(ListItem):
    """A single UI element search result item."""
    
    class Selected(Message):
        """Message sent when a result is selected."""
        def __init__(self, widget: Widget, label: str, element_type: str) -> None:
            super().__init__()
            self.widget = widget
            self.label = label
            self.element_type = element_type
    
    def __init__(
        self,
        widget: Widget,
        label: str,
        element_type: str,
        current_value: Any = None,
        widget_id: str = "",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.widget = widget
        self.label = label
        self.element_type = element_type
        self.current_value = current_value
        self.widget_id = widget_id
    
    def compose(self) -> ComposeResult:
        """Compose the search result item."""
        with Horizontal(classes="config-search-result-item"):
            with Vertical(classes="result-info"):
                # Label of the setting
                yield Label(f"[b cyan]{self.label}[/b cyan]", classes="result-path")
                
                # Element type and current value
                type_str = f"[dim]{self.element_type}[/dim]"
                if self.current_value is not None:
                    value_str = self._format_value(self.current_value)
                    yield Label(f"{type_str} • Value: [yellow]{value_str}[/yellow]", classes="result-value")
                else:
                    yield Label(type_str, classes="result-value")
                
                # Widget ID if available
                if self.widget_id:
                    yield Label(f"[dim]ID: {self.widget_id}[/dim]", classes="result-context")
            
            # Action button
            with Horizontal(classes="result-actions"):
                yield Button("Go to", id=f"goto_{id(self.widget)}", classes="result-goto-btn", variant="primary")
    
    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, str):
            # Truncate long strings
            if len(value) > 50:
                return f'"{value[:47]}..."'
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (list, dict)):
            # Show type and length for complex types
            if isinstance(value, list):
                return f"[list with {len(value)} items]"
            else:
                return f"[dict with {len(value)} keys]"
        else:
            return str(value)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id and event.button.id.startswith("goto_"):
            self.post_message(self.Selected(self.widget, self.label, self.element_type))


class UIElementSearchEngine:
    """Engine for searching through UI form elements."""
    
    def __init__(self, container: Widget):
        self.container = container
        self.elements_index = []
        self._build_index()
    
    def _build_index(self) -> None:
        """Build an index of all searchable UI elements in the container."""
        self.elements_index = []
        self._scan_widget(self.container)
    
    def _scan_widget(self, widget: Widget, depth: int = 0) -> None:
        """Recursively scan widgets for form elements."""
        # Skip if widget is the search container itself
        if widget.id == "config-search-container" or widget.id == "config-search-results-list":
            return
            
        # Check if this is a form element
        if isinstance(widget, (Input, Select, Checkbox, TextArea)):
            # Try to find associated label
            label_text = self._find_label_for_widget(widget)
            
            # Get current value
            current_value = None
            element_type = widget.__class__.__name__
            
            if isinstance(widget, Input):
                current_value = widget.value
            elif isinstance(widget, Select):
                current_value = widget.value
                if hasattr(widget, '_options') and widget.value is not None:
                    # Try to get the display text for the selected value
                    for option in widget._options:
                        if option[1] == widget.value:
                            current_value = option[0]
                            break
            elif isinstance(widget, Checkbox):
                current_value = "Enabled" if widget.value else "Disabled"
            elif isinstance(widget, TextArea):
                current_value = widget.text[:50] + "..." if len(widget.text) > 50 else widget.text
            
            # Add to index
            self.elements_index.append({
                "widget": widget,
                "label": label_text or widget.id or "Unnamed",
                "element_type": element_type,
                "current_value": current_value,
                "widget_id": widget.id or "",
                "depth": depth
            })
        
        # Recursively scan children
        for child in widget.children:
            self._scan_widget(child, depth + 1)
    
    def _find_label_for_widget(self, widget: Widget) -> Optional[str]:
        """Try to find a label associated with a widget."""
        # Look for a Label widget immediately before this widget in the parent
        if widget.parent:
            siblings = list(widget.parent.children)
            widget_index = siblings.index(widget)
            
            # Check previous siblings for a Label
            for i in range(widget_index - 1, -1, -1):
                sibling = siblings[i]
                if isinstance(sibling, Label):
                    # Extract text content from the label
                    label_text = sibling.renderable
                    if isinstance(label_text, str):
                        # Remove markup
                        import re
                        clean_text = re.sub(r'\[.*?\]', '', label_text)
                        return clean_text.strip()
                elif hasattr(sibling, 'plain'):
                    return sibling.plain.strip()
                # If we hit another form element, stop looking
                elif isinstance(sibling, (Input, Select, Checkbox, TextArea)):
                    break
        
        return None
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for UI elements matching the query."""
        if not query:
            return []
        
        # Rebuild index to get current state
        self._build_index()
        
        query_lower = query.lower()
        results = []
        
        for element in self.elements_index:
            score = 0
            
            # Search in label
            if query_lower in element["label"].lower():
                score += 3
                if element["label"].lower().startswith(query_lower):
                    score += 2
            
            # Search in widget ID
            if element["widget_id"] and query_lower in element["widget_id"].lower():
                score += 2
            
            # Search in current value
            if element["current_value"] and query_lower in str(element["current_value"]).lower():
                score += 1
            
            # Search in element type
            if query_lower in element["element_type"].lower():
                score += 1
            
            if score > 0:
                results.append((score, element))
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda x: (-x[0], x[1]["depth"]))
        return [r[1] for r in results[:20]]  # Return top 20 results