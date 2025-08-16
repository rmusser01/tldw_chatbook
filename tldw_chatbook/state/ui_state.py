"""
UI state management.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class UIState:
    """Manages UI-related state."""
    
    # Theme
    theme: str = "default"
    dark_mode: bool = True
    
    # Layout
    sidebars: Dict[str, bool] = field(default_factory=lambda: {
        "chat_left": False,
        "chat_right": False,
        "notes_left": False,
        "notes_right": False,
        "conv_char_left": False,
        "conv_char_right": False,
        "evals": False,
        "media": False,
        "search": False,
    })
    
    sidebar_widths: Dict[str, int] = field(default_factory=lambda: {
        "default": 30,
        "chat_right": 35,
        "notes_left": 25,
    })
    
    # Modals and dialogs
    modal_open: bool = False
    current_modal: Optional[str] = None
    
    # Loading states
    loading_states: Dict[str, bool] = field(default_factory=dict)
    
    # Error states
    last_error: Optional[str] = None
    error_count: int = 0
    
    # User preferences
    show_tooltips: bool = True
    show_animations: bool = True
    compact_mode: bool = False
    
    def toggle_sidebar(self, sidebar_id: str) -> bool:
        """Toggle a sidebar's visibility."""
        current = self.sidebars.get(sidebar_id, False)
        self.sidebars[sidebar_id] = not current
        return self.sidebars[sidebar_id]
    
    def set_sidebar_width(self, sidebar_id: str, width: int) -> None:
        """Set a sidebar's width."""
        self.sidebar_widths[sidebar_id] = max(10, min(50, width))
    
    def set_loading(self, component: str, is_loading: bool) -> None:
        """Set loading state for a component."""
        self.loading_states[component] = is_loading
    
    def is_loading(self, component: str) -> bool:
        """Check if a component is loading."""
        return self.loading_states.get(component, False)
    
    def set_error(self, error: str) -> None:
        """Set the last error."""
        self.last_error = error
        self.error_count += 1
    
    def clear_error(self) -> None:
        """Clear the last error."""
        self.last_error = None
    
    def open_modal(self, modal_id: str) -> None:
        """Open a modal dialog."""
        self.modal_open = True
        self.current_modal = modal_id
    
    def close_modal(self) -> None:
        """Close the current modal."""
        self.modal_open = False
        self.current_modal = None
    
    def toggle_dark_mode(self) -> bool:
        """Toggle dark mode."""
        self.dark_mode = not self.dark_mode
        self.theme = "dark" if self.dark_mode else "light"
        return self.dark_mode
    
    def set_theme(self, theme: str) -> None:
        """Set the UI theme."""
        self.theme = theme
        self.dark_mode = theme in ["dark", "monokai", "dracula"]