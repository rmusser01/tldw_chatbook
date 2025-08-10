# tldw_chatbook/Widgets/Media_Ingest/IngestUIFactory.py
# Factory pattern for selecting the appropriate ingestion UI based on configuration

from typing import TYPE_CHECKING
from loguru import logger
from textual.containers import Container

# Import standard UI windows - one per media type
from .Ingest_Local_Video_Window import VideoIngestWindowRedesigned
from .Ingest_Local_Audio_Window import AudioIngestWindowRedesigned
from .IngestLocalVideoWindow import IngestLocalVideoWindow
from .IngestLocalDocumentWindow import IngestLocalDocumentWindow
from .IngestLocalPdfWindow import IngestLocalPdfWindow
from .IngestLocalEbookWindow import IngestLocalEbookWindow
from .IngestLocalPlaintextWindow import IngestLocalPlaintextWindow
from .IngestWizardWindow import IngestWizardWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

logger = logger.bind(module="IngestUIFactory")


class IngestUIFactory:
    """Factory class for creating the appropriate ingestion UI based on configuration."""
    
    @staticmethod
    def create_ui(app_instance: 'TldwCli', media_type: str = "video") -> Container:
        """
        Create and return the standard ingestion UI for the specified media type.
        
        Args:
            app_instance: The main application instance
            media_type: Type of media to ingest (video, audio, pdf, etc.)
            
        Returns:
            Container widget for the media type
        """
        logger.info(f"Creating ingestion UI for media type: {media_type}")
        
        # Create the standard UI for each media type
        if media_type == "video":
            return VideoIngestWindowRedesigned(app_instance)
        elif media_type == "audio":
            return AudioIngestWindowRedesigned(app_instance)
        elif media_type == "document":
            return IngestLocalDocumentWindow(app_instance)
        elif media_type == "pdf":
            return IngestLocalPdfWindow(app_instance)
        elif media_type == "ebook":
            return IngestLocalEbookWindow(app_instance)
        elif media_type == "plaintext":
            return IngestLocalPlaintextWindow(app_instance)
        else:
            # For unknown media types, use video as fallback
            logger.warning(f"Unknown media type {media_type}, using video ingestion window")
            return VideoIngestWindowRedesigned(app_instance)
    
    @staticmethod
    def get_available_media_types() -> list[str]:
        """
        Get list of supported media types.
        
        Returns:
            List of media type names
        """
        return ["video", "audio", "document", "pdf", "ebook", "plaintext"]


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