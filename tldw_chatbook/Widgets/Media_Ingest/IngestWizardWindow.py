# tldw_chatbook/Widgets/Media_Ingest/IngestWizardWindow.py
# Wizard-based media ingestion using BaseWizard framework

from typing import TYPE_CHECKING, Optional, Dict, Any
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen

from tldw_chatbook.UI.Wizards.BaseWizard import WizardContainer, WizardScreen
from .IngestWizardSteps import (
    SourceSelectionStep,
    ConfigurationStep,
    EnhancementStep,
    ReviewStep,
    WIZARD_STEPS_CSS
)

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

logger = logger.bind(module="IngestWizardWindow")


class IngestWizardScreen(WizardScreen):
    """Modal screen for the ingestion wizard."""
    
    DEFAULT_CSS = WIZARD_STEPS_CSS
    
    def compose(self) -> ComposeResult:
        """Compose the wizard screen."""
        yield IngestWizardContainer(
            self.app_instance,
            media_type=self.wizard_kwargs.get("media_type", "video")
        )


class IngestWizardContainer(WizardContainer):
    """Main wizard container for media ingestion."""
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "video"):
        # Create steps
        steps = [
            SourceSelectionStep(media_type),
            ConfigurationStep(media_type),
            EnhancementStep(app_instance),
            ReviewStep()
        ]
        
        super().__init__(
            app_instance=app_instance,
            steps=steps,
            title=f"Media Ingestion Wizard - {media_type.title()}",
            on_complete=self.handle_completion,
            on_cancel=self.handle_cancellation
        )
        
        self.media_type = media_type
        logger.info(f"IngestWizardContainer initialized for {media_type}")
    
    def handle_completion(self, data: Dict[str, Any]) -> None:
        """Handle wizard completion."""
        logger.info(f"Wizard completed with data: {data}")
        
        # Process the media based on collected data
        self.process_media(data)
        
        # Notify user of completion
        if hasattr(self.app_instance, 'notify'):
            self.app_instance.notify("Media processing started", severity="information")
    
    def handle_cancellation(self) -> None:
        """Handle wizard cancellation."""
        logger.info("Wizard cancelled")
        
        # Notify user of cancellation
        if hasattr(self.app_instance, 'notify'):
            self.app_instance.notify("Media ingestion cancelled", severity="warning")
    
    def process_media(self, data: Dict[str, Any]) -> None:
        """Process media based on wizard data."""
        # This would connect to the actual processing logic
        # For now, just log the intent
        logger.info(f"Processing {self.media_type} with settings: {data}")
        
        # Import and call the appropriate handler
        if self.media_type == "video":
            from tldw_chatbook.Event_Handlers.ingest_events import handle_local_video_process
            # Would need to format data appropriately for the handler
        elif self.media_type == "audio":
            from tldw_chatbook.Event_Handlers.ingest_events import handle_local_audio_process
            # Would need to format data appropriately for the handler
        # etc.


class IngestWizardWindow(Container):
    """Simplified wizard container for media ingestion."""
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "video", **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_type = media_type
        logger.debug(f"[Wizard] IngestWizardWindow initialized for {media_type}")
    
    def compose(self) -> ComposeResult:
        """Compose the wizard content directly."""
        # Create the wizard container directly instead of launching a modal
        yield IngestWizardContainer(self.app_instance, self.media_type)


# Standalone test function
def test_wizard():
    """Test the wizard independently."""
    from textual.app import App
    
    class TestWizardApp(App):
        def __init__(self):
            super().__init__()
            self.app_config = {
                "api_settings": {
                    "openai": {},
                    "anthropic": {}
                }
            }
        
        def compose(self):
            yield IngestWizardContainer(
                app_instance=self,
                media_type="video"
            )
        
        def notify(self, message: str, severity: str = "information"):
            print(f"[{severity.upper()}] {message}")
    
    app = TestWizardApp()
    app.run()


if __name__ == "__main__":
    test_wizard()