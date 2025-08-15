# Customize_Window.py
# Description: Window for customizing the application's appearance (themes and splash screens)
#
# Imports
from typing import TYPE_CHECKING, Optional
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Static, Button, ContentSwitcher
from textual import on
from loguru import logger
#
# Local Imports
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class CustomizeWindow(Container):
    """
    Container for the Customize Tab's UI - Theme Editor and Splash Screen Gallery.
    Both views are lazily loaded when first accessed.
    """
    
    DEFAULT_CSS = """
    CustomizeWindow {
        layout: horizontal;
        height: 100%;
        width: 100%;
    }
    
    .customize-nav-pane {
        width: 25;
        min-width: 20;
        max-width: 35;
        background: $boost;
        padding: 1;
        border-right: thick $background;
    }
    
    .customize-content-pane {
        width: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    .customize-view-area {
        width: 100%;
        height: 100%;
    }
    
    .customize-nav-button {
        width: 100%;
        margin-bottom: 1;
    }
    
    .customize-nav-button.active-nav {
        background: $primary;
    }
    
    .sidebar-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
        color: $primary;
    }
    
    .section-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
        color: $primary;
    }
    
    .section-description {
        text-align: center;
        margin-bottom: 2;
        color: $text-muted;
    }
    
    .embedded-splash-viewer {
        width: 100%;
        height: 100%;
    }
    
    .embedded-theme-editor {
        width: 100%;
        height: 100%;
    }
    
    .loading-placeholder {
        text-align: center;
        color: $text-muted;
        margin-top: 5;
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.theme_editor_loaded = False
        self.splash_viewer_loaded = False
        self.current_view = "customize-view-theme"
        
    def compose(self) -> ComposeResult:
        """Compose the Customize Window UI."""
        # Navigation pane
        with Container(id="customize-nav-pane", classes="customize-nav-pane"):
            yield Static("🎨 Customize", classes="sidebar-title")
            yield Button("Theme Editor", id="customize-nav-theme", classes="customize-nav-button active-nav")
            yield Button("Splash Screens", id="customize-nav-splash", classes="customize-nav-button")
        
        # Content pane - simple container that we'll manage manually
        with Container(id="customize-content-pane", classes="customize-content-pane"):
            # Theme Editor view (initially visible but not loaded)
            with Container(id="customize-view-theme", classes="customize-view-area"):
                yield Static("🎨 Theme Editor", classes="section-title")
                yield Static("Customize the application's color theme", classes="section-description")
                yield Container(
                    Static("Loading theme editor...", classes="loading-placeholder"),
                    id="theme-editor-container",
                    classes="embedded-theme-editor"
                )
            
            # Splash Screen Gallery view (initially hidden)
            with Container(id="customize-view-splash", classes="customize-view-area") as splash_container:
                splash_container.display = False  # Initially hidden
                yield Static("🎨 Splash Screen Gallery", classes="section-title")
                yield Static("Browse and preview all available splash screen animations", classes="section-description")
                yield Container(
                    Static("Loading splash screen gallery...", classes="loading-placeholder"),
                    id="splash-viewer-container",
                    classes="embedded-splash-viewer"
                )
    
    async def on_mount(self) -> None:
        """Called when the widget is mounted. Load the default view."""
        # Load theme editor by default since it's the initial view
        await self._load_theme_editor()
    
    @on(Button.Pressed)
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle navigation button presses."""
        button_id = event.button.id
        
        if button_id == "customize-nav-theme":
            await self._show_view("customize-view-theme")
            # Load theme editor on first access
            if not self.theme_editor_loaded:
                await self._load_theme_editor()
            event.stop()  # Stop event propagation
        elif button_id == "customize-nav-splash":
            await self._show_view("customize-view-splash")
            # Load splash viewer on first access
            if not self.splash_viewer_loaded:
                await self._load_splash_viewer()
            event.stop()  # Stop event propagation
    
    async def _show_view(self, view_id: str) -> None:
        """Switch to a specific view by hiding/showing containers."""
        try:
            # Hide all views first
            theme_view = self.query_one("#customize-view-theme")
            splash_view = self.query_one("#customize-view-splash")
            
            if view_id == "customize-view-theme":
                theme_view.display = True
                splash_view.display = False
                self.current_view = "customize-view-theme"
            elif view_id == "customize-view-splash":
                theme_view.display = False
                splash_view.display = True
                self.current_view = "customize-view-splash"
            
            # Update navigation button states
            nav_buttons = {
                "customize-view-theme": "customize-nav-theme",
                "customize-view-splash": "customize-nav-splash"
            }
            
            for v_id, btn_id in nav_buttons.items():
                try:
                    button = self.query_one(f"#{btn_id}")
                    if v_id == view_id:
                        button.add_class("active-nav")
                    else:
                        button.remove_class("active-nav")
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"Error switching to view {view_id}: {e}")
    
    async def _load_theme_editor(self) -> None:
        """Load the theme editor when first accessed."""
        if self.theme_editor_loaded:
            return
            
        try:
            container = self.query_one("#theme-editor-container")
            # Clear the placeholder
            await container.remove_children()
            
            # Import and mount the actual theme editor
            from .Theme_Editor_Window import ThemeEditorView
            theme_editor = ThemeEditorView()
            await container.mount(theme_editor)
            self.theme_editor_loaded = True
            
            logger.info("Theme editor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading theme editor: {e}")
            # Show error message instead of placeholder
            try:
                container = self.query_one("#theme-editor-container")
                await container.remove_children()
                await container.mount(Static(f"Error loading theme editor: {str(e)}", classes="error-message"))
            except Exception:
                pass
    
    async def _load_splash_viewer(self) -> None:
        """Load the splash screen viewer when first accessed."""
        if self.splash_viewer_loaded:
            return
            
        try:
            container = self.query_one("#splash-viewer-container")
            # Clear the placeholder
            await container.remove_children()
            
            # Import and mount the actual splash viewer
            from ..Widgets.splash_screen_viewer import SplashScreenViewer
            splash_viewer = SplashScreenViewer()
            await container.mount(splash_viewer)
            self.splash_viewer_loaded = True
            
            logger.info("Splash viewer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading splash viewer: {e}")
            # Show error message instead of placeholder
            try:
                container = self.query_one("#splash-viewer-container")
                await container.remove_children()
                await container.mount(Static(f"Error loading splash viewer: {str(e)}", classes="error-message"))
            except Exception:
                pass

#
# End of Customize_Window.py
#######################################################################################################################