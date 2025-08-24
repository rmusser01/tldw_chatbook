"""
Root application state container.
"""

from dataclasses import dataclass, field
from typing import Optional
from textual.reactive import reactive

from .navigation_state import NavigationState
from .chat_state import ChatState
from .notes_state import NotesState
from .ui_state import UIState


@dataclass
class AppState:
    """
    Root state container for the entire application.
    This is the single source of truth for all application state.
    """
    
    # Sub-states
    navigation: NavigationState = field(default_factory=NavigationState)
    chat: ChatState = field(default_factory=ChatState)
    notes: NotesState = field(default_factory=NotesState)
    ui: UIState = field(default_factory=UIState)
    
    # App-level state
    version: str = "1.0.0"
    is_ready: bool = False
    encryption_enabled: bool = False
    encryption_password: Optional[str] = None
    
    # Configuration
    config_path: Optional[str] = None
    data_path: Optional[str] = None
    
    def reset(self) -> None:
        """Reset all state to defaults."""
        self.navigation = NavigationState()
        self.chat = ChatState()
        self.notes = NotesState()
        self.ui = UIState()
        self.is_ready = False
    
    def to_dict(self) -> dict:
        """Convert state to dictionary for serialization."""
        return {
            "version": self.version,
            "navigation": {
                "current_screen": self.navigation.current_screen,
                "history": self.navigation.history[-10:],  # Last 10 items
            },
            "chat": {
                "provider": self.chat.provider,
                "model": self.chat.model,
                "sidebar_collapsed": self.chat.sidebar_collapsed,
                "right_sidebar_collapsed": self.chat.right_sidebar_collapsed,
            },
            "notes": {
                "selected_note_id": self.notes.selected_note_id,
                "sort_by": self.notes.sort_by,
                "sort_ascending": self.notes.sort_ascending,
                "preview_mode": self.notes.preview_mode,
                "auto_save_enabled": self.notes.auto_save_enabled,
            },
            "ui": {
                "theme": self.ui.theme,
                "dark_mode": self.ui.dark_mode,
                "sidebars": self.ui.sidebars,
                "sidebar_widths": self.ui.sidebar_widths,
                "show_tooltips": self.ui.show_tooltips,
                "show_animations": self.ui.show_animations,
                "compact_mode": self.ui.compact_mode,
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AppState':
        """Create state from dictionary."""
        state = cls()
        
        # Navigation state
        if "navigation" in data:
            nav = data["navigation"]
            state.navigation.current_screen = nav.get("current_screen", "chat")
            state.navigation.history = nav.get("history", [])
        
        # Chat state
        if "chat" in data:
            chat = data["chat"]
            state.chat.provider = chat.get("provider", "openai")
            state.chat.model = chat.get("model", "gpt-4")
            state.chat.sidebar_collapsed = chat.get("sidebar_collapsed", False)
            state.chat.right_sidebar_collapsed = chat.get("right_sidebar_collapsed", False)
        
        # Notes state
        if "notes" in data:
            notes = data["notes"]
            state.notes.selected_note_id = notes.get("selected_note_id")
            state.notes.sort_by = notes.get("sort_by", "date_created")
            state.notes.sort_ascending = notes.get("sort_ascending", False)
            state.notes.preview_mode = notes.get("preview_mode", False)
            state.notes.auto_save_enabled = notes.get("auto_save_enabled", True)
        
        # UI state
        if "ui" in data:
            ui = data["ui"]
            state.ui.theme = ui.get("theme", "default")
            state.ui.dark_mode = ui.get("dark_mode", True)
            state.ui.sidebars = ui.get("sidebars", state.ui.sidebars)
            state.ui.sidebar_widths = ui.get("sidebar_widths", state.ui.sidebar_widths)
            state.ui.show_tooltips = ui.get("show_tooltips", True)
            state.ui.show_animations = ui.get("show_animations", True)
            state.ui.compact_mode = ui.get("compact_mode", False)
        
        return state