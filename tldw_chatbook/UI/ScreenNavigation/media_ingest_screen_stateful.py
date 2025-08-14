"""Media Ingest Screen with state management support."""

from typing import TYPE_CHECKING, Dict, Any
from loguru import logger

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static
from textual.containers import Container

from ..MediaIngest.video import VideoIngestTab
from .navigation_system import ScreenState

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class PlaceholderTab(Container):
    """Placeholder for tabs not yet implemented."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message
        self.add_class("placeholder-tab")
    
    def compose(self) -> ComposeResult:
        yield Static(self.message)


class MediaIngestScreenStateful(Screen):
    """Media Ingestion Screen with full screen features and state management."""
    
    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+s", "save_state", "Save State"),
    ]
    
    DEFAULT_CSS = """
    MediaIngestScreenStateful {
        background: $surface;
    }
    
    #ingest-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    
    #ingest-header {
        height: 3;
        width: 100%;
        padding: 1;
        background: $primary;
        content-align: center middle;
        margin-bottom: 1;
    }
    
    .header-title {
        text-style: bold;
        color: $text;
    }
    
    TabbedContent {
        width: 100%;
        height: 1fr;
    }
    
    TabPane {
        padding: 0;
        width: 100%;
        height: 100%;
    }
    
    .placeholder-tab {
        width: 100%;
        height: 100%;
        content-align: center middle;
        padding: 2;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._saved_state: Optional[ScreenState] = None
        logger.debug("MediaIngestScreenStateful initialized")
    
    def compose(self) -> ComposeResult:
        """Build the media ingestion interface."""
        yield Header()
        
        with Container(id="ingest-container"):
            # Header
            with Container(id="ingest-header"):
                yield Static("Media Ingestion", classes="header-title")
            
            # Tabbed content for different media types
            with TabbedContent(initial="video", id="media-tabs"):
                # Video tab
                with TabPane("Video", id="video"):
                    yield VideoIngestTab(self.app_instance, "video")
                
                # Audio tab
                with TabPane("Audio", id="audio"):
                    yield PlaceholderTab("Audio ingestion - Coming soon")
                
                # PDF tab
                with TabPane("PDF", id="pdf"):
                    yield PlaceholderTab("PDF ingestion - Coming soon")
                
                # Documents tab
                with TabPane("Documents", id="documents"):
                    yield PlaceholderTab("Document ingestion - Coming soon")
                
                # Ebooks tab
                with TabPane("Ebooks", id="ebooks"):
                    yield PlaceholderTab("Ebook ingestion - Coming soon")
                
                # Web tab
                with TabPane("Web", id="web"):
                    yield PlaceholderTab("Web article ingestion - Coming soon")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Handle mount event."""
        logger.info("MediaIngestScreenStateful mounted")
        
        # Restore state if available
        if self._saved_state:
            self.restore_state(self._saved_state)
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        # Save state before leaving
        self.save_state()
        
        # Pop this screen
        self.app.pop_screen()
    
    def action_save_state(self) -> None:
        """Manually save the current state."""
        self.save_state()
        self.notify("State saved", severity="information")
    
    def save_state(self) -> ScreenState:
        """Save the current state of the screen."""
        state = ScreenState("media_ingest")
        state.save_from_screen(self)
        
        # Save additional custom data
        tabbed_content = self.query_one("#media-tabs", TabbedContent)
        state.custom_data["active_tab"] = tabbed_content.active
        
        self._saved_state = state
        logger.debug(f"Saved MediaIngest state: {len(state.form_data)} form fields")
        
        return state
    
    def restore_state(self, state: ScreenState) -> None:
        """Restore a saved state."""
        state.restore_to_screen(self)
        
        # Restore custom data
        if "active_tab" in state.custom_data:
            tabbed_content = self.query_one("#media-tabs", TabbedContent)
            tabbed_content.active = state.custom_data["active_tab"]
        
        logger.debug(f"Restored MediaIngest state: {len(state.form_data)} form fields")
    
    def on_screen_suspend(self) -> None:
        """Called when screen is about to be suspended (navigated away from)."""
        self.save_state()
        logger.debug("MediaIngestScreenStateful suspended, state saved")
    
    def on_screen_resume(self) -> None:
        """Called when screen is resumed (navigated back to)."""
        if self._saved_state:
            self.restore_state(self._saved_state)
        logger.debug("MediaIngestScreenStateful resumed, state restored")