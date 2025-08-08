# tldw_chatbook/Widgets/Media_Ingest/IngestUIFactory.py
# Factory pattern for selecting the appropriate ingestion UI based on configuration

from typing import TYPE_CHECKING, Optional
from loguru import logger
from textual.containers import Container

from tldw_chatbook.config import get_ingest_ui_style

# Import all UI variants
from .IngestLocalVideoWindowSimplified import IngestLocalVideoWindowSimplified
from .IngestGridWindow import IngestGridWindow
from .IngestWizardWindow import IngestWizardWindow
from .IngestSplitPaneWindow import IngestSplitPaneWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

logger = logger.bind(module="IngestUIFactory")


class IngestUIFactory:
    """Factory class for creating the appropriate ingestion UI based on configuration."""
    
    @staticmethod
    def create_ui(app_instance: 'TldwCli', media_type: str = "video") -> Container:
        """
        Create and return the appropriate ingestion UI based on configuration.
        
        Args:
            app_instance: The main application instance
            media_type: Type of media to ingest (video, audio, pdf, etc.)
            
        Returns:
            Container widget for the selected UI style
        """
        # Get configured UI style
        ui_style = get_ingest_ui_style()
        
        logger.info(f"Creating ingestion UI with style: {ui_style} for media type: {media_type}")
        
        # Create and return the appropriate UI
        if ui_style == "grid":
            return IngestGridWindow(app_instance, media_type)
        elif ui_style == "wizard":
            return IngestWizardWindow(app_instance, media_type)
        elif ui_style == "split":
            return IngestSplitPaneWindow(app_instance, media_type)
        else:
            # Default to simplified UI
            if media_type == "video":
                return IngestLocalVideoWindowSimplified(app_instance)
            else:
                # For other media types, fall back to grid as it's more generic
                logger.warning(f"Simplified UI not available for {media_type}, using grid layout")
                return IngestGridWindow(app_instance, media_type)
    
    @staticmethod
    def get_available_styles() -> list[str]:
        """
        Get list of available UI styles.
        
        Returns:
            List of UI style names
        """
        return ["simplified", "grid", "wizard", "split"]
    
    @staticmethod
    def get_style_description(style: str) -> str:
        """
        Get a description of a UI style.
        
        Args:
            style: UI style name
            
        Returns:
            Human-readable description of the style
        """
        descriptions = {
            "simplified": "Simple, vertical layout with progressive disclosure",
            "grid": "Compact 3-column grid layout for efficient space usage",
            "wizard": "Step-by-step wizard interface for guided ingestion",
            "split": "Split-pane interface with live preview on the right"
        }
        return descriptions.get(style, "Unknown UI style")


# Convenience function for direct import
def create_ingest_ui(app_instance: 'TldwCli', media_type: str = "video") -> Container:
    """
    Convenience function to create ingestion UI.
    
    Args:
        app_instance: The main application instance
        media_type: Type of media to ingest
        
    Returns:
        Container widget for the selected UI style
    """
    return IngestUIFactory.create_ui(app_instance, media_type)


# Test function for development
def test_factory():
    """Test the factory with different configurations."""
    from textual.app import App
    
    class TestFactoryApp(App):
        def __init__(self, ui_style: str = "simplified"):
            super().__init__()
            self.app_config = {
                "api_settings": {
                    "openai": {},
                    "anthropic": {}
                }
            }
            self.ui_style = ui_style
            
            # Mock the config to return our test style
            import tldw_chatbook.config as config
            original_get_style = config.get_ingest_ui_style
            config.get_ingest_ui_style = lambda: self.ui_style
        
        def compose(self):
            yield IngestUIFactory.create_ui(self, "video")
        
        def notify(self, message: str, severity: str = "information"):
            print(f"[{severity.upper()}] {message}")
    
    import sys
    ui_style = sys.argv[1] if len(sys.argv) > 1 else "simplified"
    print(f"Testing with UI style: {ui_style}")
    
    app = TestFactoryApp(ui_style)
    app.run()


if __name__ == "__main__":
    test_factory()