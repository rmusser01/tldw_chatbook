"""
Refactored main application following Textual best practices - v2.0
Corrected implementation with proper reactive state and error handling.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import inspect

from loguru import logger
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Button
from textual.screen import Screen

# Disable progress bars for TUI
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class TldwCliRefactored(App):
    """
    Refactored application with proper reactive state management.
    Follows Textual best practices with comprehensive error handling.
    """
    
    # Proper CSS path using absolute reference
    CSS_PATH = Path(__file__).parent / "css" / "tldw_cli_modular.tcss"
    
    # Key bindings
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+s", "save", "Save"),
        ("ctrl+b", "toggle_sidebar", "Sidebar"),
        ("escape", "back", "Back"),
    ]
    
    # Screen registry will be populated in __init__
    SCREENS = {}
    
    # Simple reactive attributes (Textual best practice)
    current_screen: reactive[str] = reactive("chat")
    is_loading: reactive[bool] = reactive(False)
    error_message: reactive[Optional[str]] = reactive(None)
    # Note: 'theme' is a built-in Textual attribute, don't override it
    
    # Reactive dictionaries for complex state
    chat_state: reactive[Dict[str, Any]] = reactive({
        "provider": "openai",
        "model": "gpt-4",
        "is_streaming": False,
        "sidebar_collapsed": False,
        "active_session_id": None
    })
    
    notes_state: reactive[Dict[str, Any]] = reactive({
        "selected_note_id": None,
        "unsaved_changes": False,
        "preview_mode": False,
        "auto_save": True
    })
    
    ui_state: reactive[Dict[str, Any]] = reactive({
        "sidebars": {
            "chat_left": False,
            "chat_right": False,
            "notes_left": False,
            "notes_right": False
        },
        "modal_open": False,
        "dark_mode": True
    })
    
    def __init__(self):
        """Initialize with proper error handling."""
        super().__init__()
        
        # Track initialization
        self._initialized = False
        self._splash_widget = None
        self._screen_registry = {}
        self._screen_cache = {}
        
        # Load configuration safely
        self._load_configuration()
        
        # Build screen registry
        self._build_screen_registry()
        
        logger.info("Application initialized")
    
    def _load_configuration(self):
        """Load configuration with error handling."""
        try:
            from .config import load_cli_config_and_ensure_existence
            load_cli_config_and_ensure_existence()
            logger.debug("Configuration loaded")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Continue with defaults
    
    def _build_screen_registry(self):
        """Build screen registry with fallbacks and install screens."""
        # Try to import screens from new locations, fallback to old
        screen_mappings = [
            ("chat", "UI.Screens.chat_screen", "ChatScreen", 
             "UI.Chat_Window_Enhanced", "ChatWindowEnhanced"),
            ("notes", "UI.Screens.notes_screen", "NotesScreen",
             "UI.Notes_Window", "NotesWindow"),
            ("media", "UI.Screens.media_screen", "MediaScreen",
             "UI.MediaWindow_v2", "MediaWindow"),
            ("search", "UI.Screens.search_screen", "SearchScreen",
             "UI.SearchWindow", "SearchWindow"),
            ("coding", "UI.Screens.coding_screen", "CodingScreen",
             "UI.Coding_Window", "CodingWindow"),
            ("ccp", "UI.Screens.conversation_screen", "ConversationScreen",
             "UI.Conv_Char_Window", "CCPWindow"),
            ("ingest", "UI.Screens.media_ingest_screen", "MediaIngestScreen",
             "UI.MediaIngestWindowRebuilt", "MediaIngestWindow"),
            ("evals", "UI.Screens.evals_screen", "EvalsScreen",
             "UI.Evals.evals_window_v3", "EvalsWindow"),
            ("tools_settings", "UI.Screens.tools_settings_screen", "ToolsSettingsScreen",
             "UI.Tools_Settings_Window", "ToolsSettingsWindow"),
            ("llm", "UI.Screens.llm_screen", "LLMScreen",
             "UI.LLM_Management_Window", "LLMManagementWindow"),
            ("customize", "UI.Screens.customize_screen", "CustomizeScreen",
             "UI.Customize_Window", "CustomizeWindow"),
            ("logs", "UI.Screens.logs_screen", "LogsScreen",
             "UI.Logs_Window", "LogsWindow"),
            ("stats", "UI.Screens.stats_screen", "StatsScreen",
             "UI.Stats_Window", "StatsWindow"),
            ("stts", "UI.Screens.stts_screen", "STTSScreen",
             "UI.STTS_Window", "STTSWindow"),
            ("study", "UI.Screens.study_screen", "StudyScreen",
             "UI.Study_Window", "StudyWindow"),
            ("chatbooks", "UI.Screens.chatbooks_screen", "ChatbooksScreen",
             "UI.Chatbooks_Window", "ChatbooksWindow"),
            ("subscription", "UI.Screens.subscription_screen", "SubscriptionScreen",
             "UI.SubscriptionWindow", "SubscriptionWindow"),
        ]
        
        for screen_name, new_module, new_class, old_module, old_class in screen_mappings:
            screen_class = self._try_import_screen(
                screen_name, new_module, new_class, old_module, old_class
            )
            if screen_class:
                self._screen_registry[screen_name] = screen_class
                # Install screen in the app's SCREENS dict for Textual
                self.SCREENS[screen_name] = screen_class
        
        # Add aliases
        self._screen_registry["subscriptions"] = self._screen_registry.get("subscription")
        self._screen_registry["conversation"] = self._screen_registry.get("ccp")
        
        # Install aliases too
        if "subscription" in self._screen_registry:
            self.SCREENS["subscriptions"] = self._screen_registry["subscription"]
        if "ccp" in self._screen_registry:
            self.SCREENS["conversation"] = self._screen_registry["ccp"]
        
        logger.info(f"Registered and installed {len(self._screen_registry)} screens")
    
    def _try_import_screen(self, name, new_module, new_class, old_module, old_class):
        """Try to import a screen from new or old location."""
        # Try new location first
        try:
            module = __import__(f"tldw_chatbook.{new_module}", fromlist=[new_class])
            screen_class = getattr(module, new_class)
            logger.debug(f"Loaded {name} from new location")
            return screen_class
        except (ImportError, AttributeError):
            pass
        
        # Try old location
        try:
            module = __import__(f"tldw_chatbook.{old_module}", fromlist=[old_class])
            screen_class = getattr(module, old_class)
            logger.debug(f"Loaded {name} from legacy location")
            return screen_class
        except (ImportError, AttributeError):
            logger.warning(f"Failed to load screen: {name}")
            return None
    
    def compose(self) -> ComposeResult:
        """Compose the application UI."""
        # Check for splash screen
        try:
            from .config import get_cli_setting
            if get_cli_setting("splash_screen", "enabled", False):
                from .Widgets.splash_screen import SplashScreen
                self._splash_widget = SplashScreen(
                    duration=get_cli_setting("splash_screen", "duration", 1.5),
                    id="splash-screen"
                )
                yield self._splash_widget
                return
        except Exception as e:
            logger.debug(f"Splash screen not available: {e}")
        
        # Compose main UI immediately if no splash screen
        yield from self._compose_main_ui()
    
    def _compose_main_ui(self) -> ComposeResult:
        """Compose main UI components with fallbacks."""
        # Title bar
        try:
            from .UI.titlebar import TitleBar
            yield TitleBar()
        except ImportError:
            logger.warning("TitleBar not available")
            yield Container(id="titlebar-placeholder")
        
        # Navigation
        try:
            from .UI.Tab_Links import TabLinks
            from .Constants import ALL_TABS
            initial_tab = self.current_screen
            yield TabLinks(tab_ids=ALL_TABS, initial_active_tab=initial_tab)
        except ImportError:
            logger.warning("TabLinks not available")
            yield Container(id="navigation-placeholder")
        
        # Screen container
        yield Container(id="screen-container")
        
        # Footer
        try:
            from .Widgets.AppFooterStatus import AppFooterStatus
            yield AppFooterStatus(id="app-footer")
        except ImportError:
            logger.warning("AppFooterStatus not available")
            yield Container(id="footer-placeholder")
    
    async def on_mount(self):
        """Handle application mount."""
        logger.info("Application mounting")
        
        # If splash screen is active, wait for it
        if self._splash_widget:
            return
        
        # Navigate to initial screen
        await self._mount_initial_screen()
        
        # Load saved state
        await self._load_state()
        
        # Mark as initialized
        self._initialized = True
    
    async def _mount_initial_screen(self):
        """Mount the initial screen with error handling."""
        try:
            await self.navigate_to_screen(self.current_screen)
        except Exception as e:
            logger.error(f"Failed to mount initial screen: {e}")
            # Try fallback to chat
            if self.current_screen != "chat":
                try:
                    await self.navigate_to_screen("chat")
                except:
                    self.notify("Failed to load initial screen", severity="error")
    
    async def navigate_to_screen(self, screen_name: str) -> bool:
        """Navigate to a screen with proper error handling."""
        try:
            # Check if already on this screen
            if self.current_screen == screen_name:
                logger.debug(f"Already on screen: {screen_name}")
                return True
            
            # Check if screen is installed
            if screen_name not in self.SCREENS:
                logger.error(f"Unknown screen: {screen_name}")
                self.notify(f"Screen '{screen_name}' not found", severity="error")
                return False
            
            # Set loading state
            self.is_loading = True
            
            # Use Textual's built-in screen pushing
            # push_screen can take either a screen instance or a screen name
            try:
                # First, check if we have any screens on the stack
                try:
                    current = self.screen
                    # If we get here, we have screens, use switch_screen
                    await self.switch_screen(screen_name)
                except:
                    # No screens yet, push the first one
                    await self.push_screen(screen_name)
                
                # Update state
                old_screen = self.current_screen
                self.current_screen = screen_name
                
                # Clear loading state  
                self.is_loading = False
                
                logger.info(f"Navigated from {old_screen} to {screen_name}")
                return True
                
            except Exception as nav_error:
                logger.error(f"Screen navigation error: {nav_error}")
                # Try creating and pushing screen instance as fallback
                screen_class = self.SCREENS[screen_name]
                screen = self._create_screen_instance(screen_class)
                if screen:
                    await self.push_screen(screen)
                    self.current_screen = screen_name
                    self.is_loading = False
                    return True
                raise
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}", exc_info=True)
            self.is_loading = False
            self.notify("Navigation failed", severity="error")
            return False
    
    def _create_screen_instance(self, screen_class: type) -> Optional[Screen]:
        """Create screen instance with proper parameter handling."""
        try:
            # Check what parameters the screen expects
            sig = inspect.signature(screen_class.__init__)
            params = list(sig.parameters.keys())
            
            # Remove 'self' from parameters
            if 'self' in params:
                params.remove('self')
            
            # Determine construction method
            if not params:
                # No parameters needed
                return screen_class()
            elif 'app' in params:
                # Expects app parameter
                return screen_class(app=self)
            elif 'app_instance' in params:
                # Legacy parameter name
                return screen_class(app_instance=self)
            else:
                # Try with self as first parameter (common pattern)
                return screen_class(self)
                
        except Exception as e:
            logger.error(f"Failed to create screen instance: {e}")
            # Last resort: try no parameters
            try:
                return screen_class()
            except:
                return None
    
    # Event Handlers
    
    @on(Button.Pressed)
    async def handle_button_press(self, event: Button.Pressed):
        """Handle button presses with compatibility layer."""
        button_id = event.button.id
        
        if not button_id:
            return
        
        # Compatibility for old tab buttons
        if button_id.startswith("tab-"):
            screen_name = button_id[4:]
            await self.navigate_to_screen(screen_name)
        
        # Handle navigation from TabLinks
        elif button_id.startswith("tab-link-"):
            screen_name = button_id[9:]
            await self.navigate_to_screen(screen_name)
    
    # Import NavigateToScreen only if available
    try:
        from .UI.Navigation.main_navigation import NavigateToScreen
        
        @on(NavigateToScreen)
        async def handle_navigation_message(self, message: NavigateToScreen):
            """Handle navigation messages."""
            await self.navigate_to_screen(message.screen_name)
    except ImportError:
        logger.debug("NavigateToScreen message not available")
    
    # Handle splash screen if available
    try:
        from .Widgets.splash_screen import SplashScreen
        
        @on(SplashScreen.Closed)
        async def on_splash_closed(self, event):
            """Handle splash screen closing."""
            logger.debug("Splash screen closed")
            
            if self._splash_widget:
                await self._splash_widget.remove()
                self._splash_widget = None
            
            # Mount main UI components after splash
            await self.mount(*self._compose_main_ui())
            
            # Navigate to initial screen
            await self._mount_initial_screen()
            
            # Load state
            await self._load_state()
            
            self._initialized = True
    except ImportError:
        pass
    
    # Reactive Watchers
    
    def watch_current_screen(self, old_screen: str, new_screen: str):
        """React to screen changes."""
        if old_screen != new_screen:
            logger.debug(f"Screen changed: {old_screen} -> {new_screen}")
    
    def watch_is_loading(self, was_loading: bool, is_loading: bool):
        """React to loading state changes."""
        if is_loading:
            logger.debug("Loading started")
        else:
            logger.debug("Loading finished")
    
    def watch_error_message(self, old_error: Optional[str], new_error: Optional[str]):
        """React to error messages."""
        if new_error:
            logger.error(f"Error: {new_error}")
            self.notify(new_error, severity="error")
    
    # Actions
    
    async def action_quit(self):
        """Quit the application."""
        await self._save_state()
        self.exit()
    
    async def action_save(self):
        """Save current state."""
        try:
            await self._save_state()
            self.notify("Saved")
        except Exception as e:
            logger.error(f"Save failed: {e}")
            self.notify("Save failed", severity="error")
    
    async def action_toggle_sidebar(self):
        """Toggle sidebar for current screen."""
        screen = self.current_screen
        
        # Update appropriate sidebar state
        if screen == "chat":
            current = self.chat_state.get("sidebar_collapsed", False)
            self.chat_state = {**self.chat_state, "sidebar_collapsed": not current}
        elif screen == "notes":
            sidebars = self.ui_state.get("sidebars", {})
            current = sidebars.get("notes_left", False)
            sidebars["notes_left"] = not current
            self.ui_state = {**self.ui_state, "sidebars": sidebars}
    
    async def action_back(self):
        """Go back to previous screen."""
        # Simple implementation - go to chat
        if self.current_screen != "chat":
            await self.navigate_to_screen("chat")
    
    # State Persistence
    
    async def _save_state(self):
        """Save application state."""
        try:
            state_path = Path.home() / ".config" / "tldw_cli" / "state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create state dictionary
            state = {
                "current_screen": self.current_screen,
                "chat_state": dict(self.chat_state),
                "notes_state": dict(self.notes_state),
                "ui_state": dict(self.ui_state),
                "timestamp": datetime.now().isoformat()
            }
            
            # Only save theme if it's available (after Textual initialization)
            try:
                if hasattr(self, 'theme') and self.theme:
                    state["theme"] = self.theme
            except AttributeError:
                # Theme not yet initialized
                pass
            
            # Save with proper JSON encoding
            state_path.write_text(json.dumps(state, indent=2, default=str))
            logger.debug("State saved")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise
    
    async def _load_state(self):
        """Load application state."""
        try:
            state_path = Path.home() / ".config" / "tldw_cli" / "state.json"
            if not state_path.exists():
                logger.debug("No saved state found")
                return
            
            state = json.loads(state_path.read_text())
            
            # Restore state with validation
            # Only restore theme if Textual has initialized it
            if "theme" in state and isinstance(state["theme"], str):
                try:
                    if hasattr(self, 'theme'):
                        self.theme = state["theme"]
                except AttributeError:
                    # Theme system not ready yet
                    pass
            
            if "chat_state" in state and isinstance(state["chat_state"], dict):
                self.chat_state = state["chat_state"]
            
            if "notes_state" in state and isinstance(state["notes_state"], dict):
                self.notes_state = state["notes_state"]
            
            if "ui_state" in state and isinstance(state["ui_state"], dict):
                self.ui_state = state["ui_state"]
            
            logger.debug("State loaded")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            # Continue with defaults


def run():
    """Run the refactored application."""
    app = TldwCliRefactored()
    app.run()


if __name__ == "__main__":
    run()