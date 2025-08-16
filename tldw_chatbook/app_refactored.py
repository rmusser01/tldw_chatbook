"""
Refactored main application following Textual best practices.
This is a clean implementation that should replace the monolithic app.py.
"""

import os
import time
from typing import Optional
from pathlib import Path

from loguru import logger
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Button

# State management
from .state import AppState
from .navigation import NavigationManager

# UI Components
from .UI.titlebar import TitleBar
from .UI.Tab_Links import TabLinks
from .UI.Navigation.main_navigation import NavigateToScreen
from .Widgets.AppFooterStatus import AppFooterStatus
from .Widgets.splash_screen import SplashScreen

# Configuration
from .config import get_cli_setting, load_cli_config_and_ensure_existence
from .Constants import ALL_TABS

# Disable progress bars for TUI
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class TldwCliRefactored(App):
    """
    Main application class following Textual best practices.
    Clean, maintainable, and properly structured.
    """
    
    # CSS
    CSS_PATH = "css/tldw_cli_modular.tcss"
    
    # Bindings
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+b", "toggle_sidebar", "Toggle Sidebar"),
        ("ctrl+n", "new_note", "New Note"),
        ("ctrl+s", "save", "Save"),
        ("escape", "go_back", "Go Back"),
    ]
    
    # Single reactive state
    state = reactive(AppState())
    
    def __init__(self):
        """Initialize the application."""
        super().__init__()
        
        # Load configuration
        load_cli_config_and_ensure_existence()
        
        # Initialize managers
        self.nav_manager = NavigationManager(self, self.state.navigation)
        
        # Track initialization
        self._start_time = time.perf_counter()
        self._splash_widget: Optional[SplashScreen] = None
        
        logger.info("Application initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the application UI."""
        # Check if splash screen is enabled
        if get_cli_setting("splash_screen", "enabled", True):
            # Show splash screen first
            self._splash_widget = SplashScreen(
                duration=get_cli_setting("splash_screen", "duration", 1.5),
                skip_on_keypress=get_cli_setting("splash_screen", "skip_on_keypress", True),
                show_progress=get_cli_setting("splash_screen", "show_progress", True),
                id="app-splash-screen"
            )
            yield self._splash_widget
            return
        
        # Compose main UI
        yield from self._compose_main_ui()
    
    def _compose_main_ui(self) -> ComposeResult:
        """Compose the main UI components."""
        # Title bar
        yield TitleBar()
        
        # Navigation (using links as requested)
        initial_tab = get_cli_setting("general", "initial_tab", "chat")
        yield TabLinks(tab_ids=ALL_TABS, initial_active_tab=initial_tab)
        
        # Screen container
        yield Container(id="screen-container")
        
        # Footer
        yield AppFooterStatus(id="app-footer-status")
    
    async def on_mount(self) -> None:
        """Handle application mount."""
        logger.info("Application mounting")
        
        # If splash screen is active, wait for it to close
        if self._splash_widget:
            return
        
        # Navigate to initial screen
        await self._mount_initial_screen()
    
    async def _mount_initial_screen(self) -> None:
        """Mount the initial screen."""
        initial_screen = get_cli_setting("general", "initial_tab", "chat")
        
        # Navigate to initial screen
        success = await self.nav_manager.navigate_to(initial_screen)
        if success:
            logger.info(f"Initial screen mounted: {initial_screen}")
            self.state.is_ready = True
        else:
            logger.error(f"Failed to mount initial screen: {initial_screen}")
    
    # Event Handlers
    
    @on(SplashScreen.Closed)
    async def on_splash_closed(self, event: SplashScreen.Closed) -> None:
        """Handle splash screen closing."""
        logger.debug("Splash screen closed")
        
        # Remove splash screen
        if self._splash_widget:
            await self._splash_widget.remove()
            self._splash_widget = None
        
        # Mount main UI
        await self.mount(*self._compose_main_ui())
        
        # Navigate to initial screen
        await self._mount_initial_screen()
    
    @on(NavigateToScreen)
    async def handle_navigation(self, message: NavigateToScreen) -> None:
        """Handle screen navigation requests."""
        await self.nav_manager.navigate_to(message.screen_name)
    
    @on(Button.Pressed)
    async def handle_button_press(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if not button_id:
            return
        
        # Handle common buttons
        if button_id == "quit-button":
            self.exit()
        elif button_id == "save-button":
            await self.action_save()
        elif button_id == "back-button":
            await self.action_go_back()
    
    # Actions
    
    async def action_quit(self) -> None:
        """Quit the application."""
        # Save state before quitting
        await self._save_state()
        self.exit()
    
    async def action_toggle_sidebar(self) -> None:
        """Toggle the current screen's sidebar."""
        # Determine which sidebar based on current screen
        screen = self.nav_manager.get_current_screen()
        
        if screen == "chat":
            self.state.chat.toggle_sidebar()
        elif screen == "notes":
            self.state.notes.left_sidebar_collapsed = not self.state.notes.left_sidebar_collapsed
        else:
            self.state.ui.toggle_sidebar(f"{screen}_left")
        
        # Notify screen of state change
        self.refresh()
    
    async def action_save(self) -> None:
        """Save current work."""
        screen = self.nav_manager.get_current_screen()
        
        if screen == "notes":
            # Save current note
            if self.state.notes.unsaved_changes:
                self.state.notes.mark_saved()
                self.notify("Note saved")
        elif screen == "chat":
            # Save chat session
            session = self.state.chat.get_active_session()
            if session and session.is_ephemeral:
                # Convert to persistent
                # (would call database here)
                self.notify("Chat saved")
    
    async def action_go_back(self) -> None:
        """Go back to previous screen."""
        await self.nav_manager.go_back()
    
    async def action_new_note(self) -> None:
        """Create a new note."""
        # Navigate to notes screen
        await self.nav_manager.navigate_to("notes")
        
        # Create new note
        note = self.state.notes.create_note("Untitled Note")
        self.notify(f"Created note: {note.title}")
    
    # State Management
    
    def watch_state(self, old_state: AppState, new_state: AppState) -> None:
        """React to state changes."""
        # This is called whenever the state reactive changes
        # Can be used for auto-save, logging, etc.
        if old_state.ui.theme != new_state.ui.theme:
            self._apply_theme(new_state.ui.theme)
    
    def _apply_theme(self, theme: str) -> None:
        """Apply a UI theme."""
        # Would update CSS variables here
        logger.info(f"Applied theme: {theme}")
    
    async def _save_state(self) -> None:
        """Save application state to disk."""
        try:
            state_file = Path.home() / ".config" / "tldw_cli" / "state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            state_dict = self.state.to_dict()
            state_file.write_text(json.dumps(state_dict, indent=2))
            
            logger.debug("State saved")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _load_state(self) -> None:
        """Load application state from disk."""
        try:
            state_file = Path.home() / ".config" / "tldw_cli" / "state.json"
            if state_file.exists():
                import json
                state_dict = json.loads(state_file.read_text())
                self.state = AppState.from_dict(state_dict)
                logger.debug("State loaded")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    # Lifecycle
    
    def on_shutdown(self) -> None:
        """Handle application shutdown."""
        # Save state synchronously on shutdown
        import asyncio
        try:
            asyncio.run(self._save_state())
        except:
            pass
        
        logger.info("Application shutdown")


def run():
    """Run the refactored application."""
    app = TldwCliRefactored()
    app.run()


if __name__ == "__main__":
    run()