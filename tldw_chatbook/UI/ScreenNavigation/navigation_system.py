"""Screen-based navigation system with state management."""

from typing import Dict, Any, Optional, Type, TYPE_CHECKING
from dataclasses import dataclass, field
from loguru import logger

from textual.app import App
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static, Button
from textual.containers import Container, Horizontal
from textual.app import ComposeResult
from textual import on

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


@dataclass
class ScreenState:
    """Stores the state of a screen for restoration."""
    screen_name: str
    form_data: Dict[str, Any] = field(default_factory=dict)
    scroll_position: tuple[int, int] = (0, 0)
    focused_widget: Optional[str] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def save_from_screen(self, screen: Screen) -> None:
        """Save state from a screen instance."""
        # Save scroll positions
        scrollables = screen.query("VerticalScroll, ScrollableContainer")
        if scrollables:
            main_scroll = scrollables.first()
            self.scroll_position = (main_scroll.scroll_x, main_scroll.scroll_y)
        
        # Save focused widget ID
        if screen.focused:
            self.focused_widget = screen.focused.id
        
        # Save form data (inputs, textareas, checkboxes, etc.)
        self.form_data = {}
        
        # Input fields
        for input_widget in screen.query("Input"):
            if input_widget.id:
                self.form_data[f"input_{input_widget.id}"] = input_widget.value
        
        # TextAreas
        for textarea in screen.query("TextArea"):
            if textarea.id:
                self.form_data[f"textarea_{textarea.id}"] = textarea.text
        
        # Checkboxes
        for checkbox in screen.query("Checkbox"):
            if checkbox.id:
                self.form_data[f"checkbox_{checkbox.id}"] = checkbox.value
        
        # Selects
        for select in screen.query("Select"):
            if select.id:
                self.form_data[f"select_{select.id}"] = select.value
        
        # RadioSets
        for radioset in screen.query("RadioSet"):
            if radioset.id and radioset.pressed_button:
                self.form_data[f"radioset_{radioset.id}"] = radioset.pressed_button.id
    
    def restore_to_screen(self, screen: Screen) -> None:
        """Restore state to a screen instance."""
        # Restore form data
        for key, value in self.form_data.items():
            widget_type, widget_id = key.split("_", 1)
            
            try:
                if widget_type == "input":
                    widget = screen.query_one(f"#{widget_id}", expect_type="Input")
                    widget.value = value
                elif widget_type == "textarea":
                    widget = screen.query_one(f"#{widget_id}", expect_type="TextArea")
                    widget.text = value
                elif widget_type == "checkbox":
                    widget = screen.query_one(f"#{widget_id}", expect_type="Checkbox")
                    widget.value = value
                elif widget_type == "select":
                    widget = screen.query_one(f"#{widget_id}", expect_type="Select")
                    widget.value = value
                elif widget_type == "radioset":
                    radioset = screen.query_one(f"#{widget_id}", expect_type="RadioSet")
                    button = radioset.query_one(f"#{value}")
                    if button:
                        radioset.pressed_button = button
            except Exception as e:
                logger.debug(f"Could not restore {key}: {e}")
        
        # Restore scroll position
        scrollables = screen.query("VerticalScroll, ScrollableContainer")
        if scrollables:
            main_scroll = scrollables.first()
            main_scroll.scroll_to(self.scroll_position[0], self.scroll_position[1], animate=False)
        
        # Restore focus
        if self.focused_widget:
            try:
                widget = screen.query_one(f"#{self.focused_widget}")
                widget.focus()
            except Exception:
                pass


class ScreenManager:
    """Manages screen navigation and state."""
    
    def __init__(self, app: App):
        self.app = app
        self.screen_states: Dict[str, ScreenState] = {}
        self.screen_stack: list[str] = []
        self.current_screen: Optional[str] = None
        
        logger.info("ScreenManager initialized")
    
    def register_screen(self, name: str, screen_class: Type[Screen]) -> None:
        """Register a screen class with the manager."""
        # This would be used to track available screens
        pass
    
    def save_current_state(self) -> None:
        """Save the state of the current screen."""
        if self.current_screen and self.app.screen:
            state = self.screen_states.get(self.current_screen, ScreenState(self.current_screen))
            state.save_from_screen(self.app.screen)
            self.screen_states[self.current_screen] = state
            logger.debug(f"Saved state for screen: {self.current_screen}")
    
    def navigate_to(self, screen_name: str, screen: Screen) -> None:
        """Navigate to a new screen with state management."""
        # Save current screen state
        self.save_current_state()
        
        # Track navigation
        if self.current_screen:
            self.screen_stack.append(self.current_screen)
        
        # Switch to new screen
        self.app.push_screen(screen)
        self.current_screen = screen_name
        
        # Restore state if it exists
        if screen_name in self.screen_states:
            # Schedule restoration after screen is mounted
            self.app.call_after_refresh(
                lambda: self.screen_states[screen_name].restore_to_screen(screen)
            )
            logger.debug(f"Restored state for screen: {screen_name}")
    
    def go_back(self) -> None:
        """Navigate back to the previous screen."""
        if self.screen_stack:
            # Save current state
            self.save_current_state()
            
            # Pop the screen
            self.app.pop_screen()
            
            # Update current screen
            self.current_screen = self.screen_stack.pop() if self.screen_stack else None


class NavigationBar(Container):
    """Navigation bar with links to different screens."""
    
    DEFAULT_CSS = """
    NavigationBar {
        height: 3;
        width: 100%;
        background: $primary;
        dock: top;
        padding: 0 1;
    }
    
    .nav-links {
        height: 100%;
        width: 100%;
        layout: horizontal;
        align: left middle;
    }
    
    .nav-link {
        margin: 0 1;
        background: transparent;
        border: none;
        color: $text;
        text-style: none;
    }
    
    .nav-link:hover {
        text-style: underline;
        color: $accent;
    }
    
    .nav-link.active {
        text-style: bold;
        color: $warning;
    }
    
    .nav-separator {
        margin: 0 1;
        color: $text-muted;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.current_screen = "chat"
    
    def compose(self) -> ComposeResult:
        """Compose the navigation bar."""
        with Horizontal(classes="nav-links"):
            yield Button("Chat", id="nav-chat", classes="nav-link active")
            yield Static("|", classes="nav-separator")
            yield Button("Conversations", id="nav-ccp", classes="nav-link")
            yield Static("|", classes="nav-separator")
            yield Button("Notes", id="nav-notes", classes="nav-link")
            yield Static("|", classes="nav-separator")
            yield Button("Media", id="nav-media", classes="nav-link")
            yield Static("|", classes="nav-separator")
            yield Button("Ingest", id="nav-ingest", classes="nav-link")
            yield Static("|", classes="nav-separator")
            yield Button("Search", id="nav-search", classes="nav-link")
            yield Static("|", classes="nav-separator")
            yield Button("Settings", id="nav-settings", classes="nav-link")
    
    @on(Button.Pressed, ".nav-link")
    def handle_navigation(self, event: Button.Pressed) -> None:
        """Handle navigation link clicks."""
        button_id = event.button.id
        if not button_id:
            return
        
        # Extract screen name from button ID
        screen_name = button_id.replace("nav-", "")
        
        # Update active state
        for button in self.query(".nav-link"):
            button.remove_class("active")
        event.button.add_class("active")
        
        # Notify app to switch screens
        self.post_message(NavigateToScreen(screen_name))
        
        logger.info(f"Navigation requested to: {screen_name}")


class NavigateToScreen(Message):
    """Message to request screen navigation."""
    
    def __init__(self, screen_name: str):
        super().__init__()
        self.screen_name = screen_name