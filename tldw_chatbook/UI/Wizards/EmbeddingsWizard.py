# tldw_chatbook/UI/Wizards/EmbeddingsWizard.py
# Description: Main embeddings creation wizard that brings together all steps
#
# Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Any
import asyncio
from datetime import datetime

# 3rd-Party Imports
from loguru import logger
from textual import work
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label, Button

# Local Imports
from .BaseWizard import WizardContainer
from .EmbeddingSteps import (
    ContentSelectionStep,
    SpecificContentStep,
    QuickSettingsStep,
    ProcessingStep
)
from ...Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Configure logger
logger = logger.bind(module="EmbeddingsWizard")

if TYPE_CHECKING:
    from textual.app import App

# Check if embeddings dependencies are available
embeddings_available = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)

if embeddings_available:
    try:
        from ...Embeddings.Embeddings_Lib import EmbeddingFactory
        from ...Embeddings.Chroma_Lib import ChromaDBManager
        from ...Chunking.Chunk_Lib import chunk_for_embedding
    except ImportError:
        EmbeddingFactory = None
        ChromaDBManager = None
        chunk_for_embedding = None
        embeddings_available = False
else:
    EmbeddingFactory = None
    ChromaDBManager = None
    chunk_for_embedding = None

########################################################################################################################
#
# Main Wizard Implementation
#
########################################################################################################################

class EmbeddingsCreationWizard(WizardContainer):
    """Main embeddings creation wizard with improved UX."""
    
    def __init__(self, on_complete: Optional[callable] = None, on_cancel: Optional[callable] = None):
        # Initialize steps
        self.content_type_step = ContentSelectionStep()
        self.content_selection_step = None  # Created dynamically based on content type
        self.settings_step = QuickSettingsStep()
        self.processing_step = ProcessingStep()
        
        # Start with just the first step
        steps = [self.content_type_step]
        
        super().__init__(
            steps=steps,
            title="Create Search Collection",
            on_complete=on_complete or self.handle_completion,
            on_cancel=on_cancel or self.handle_cancellation
        )
        
    def validate_current_step(self, old_value: int) -> None:
        """Override to handle Textual's reactive system expecting this signature."""
        # Just call the parent's method without the argument
        super().validate_current_step()
        
    def show_step(self, step_index: int) -> None:
        """Override to handle dynamic step creation."""
        # If moving from step 1 to step 2, create content selection step
        if step_index == 1 and self.content_selection_step is None:
            content_type = self.content_type_step.selected_content_type
            if content_type:
                # Create content selection step
                self.content_selection_step = SpecificContentStep(content_type)
                self.content_selection_step.step_number = 2
                
                # Insert it into steps list
                self.steps.insert(1, self.content_selection_step)
                
                # Add remaining steps if not already added
                if len(self.steps) == 2:
                    self.steps.append(self.settings_step)
                    self.steps.append(self.processing_step)
                    self.total_steps = len(self.steps)
                
                # Mount the new step
                container = self.query_one(".wizard-steps-container", Container)
                container.mount(self.content_selection_step)
                
                # Update progress bar with new step count
                self.update_progress()
                
        # Call parent implementation
        super().show_step(step_index)
        
        # If we're showing the processing step, start processing
        if step_index == 3 and isinstance(self.steps[step_index], ProcessingStep):
            self.start_processing()
            
    def start_processing(self) -> None:
        """Start the embedding creation process."""
        # Collect all configuration
        config = self.get_all_data()
        
        # Start processing
        processing_step = self.steps[3]
        if isinstance(processing_step, ProcessingStep):
            processing_step.start_processing(config)
            
            # Start the actual processing work
            self.process_embeddings(config)
            
    @work(thread=True)
    def process_embeddings(self, config: Dict[str, Any]) -> None:
        """Process embeddings in background thread."""
        try:
            logger.info(f"Starting embedding creation with config: {config}")
            
            # Simulate processing with progress updates
            # In real implementation, this would call the actual embedding logic
            total_items = config.get("item_count", 10)
            
            for i in range(total_items):
                # Simulate processing time
                asyncio.run(asyncio.sleep(0.5))
                
                # Update progress
                percent = int((i + 1) / total_items * 100)
                current_item = f"Item {i + 1} of {total_items}"
                
                self.call_from_thread(
                    self.update_processing_progress,
                    percent,
                    current_item,
                    i + 1
                )
                
            # Mark as complete
            self.call_from_thread(self.processing_complete)
            
        except Exception as e:
            logger.error(f"Error processing embeddings: {e}")
            self.call_from_thread(self.processing_error, str(e))
            
    def update_processing_progress(self, percent: int, current_item: str, items_done: int) -> None:
        """Update processing progress in UI thread."""
        processing_step = self.steps[3]
        if isinstance(processing_step, ProcessingStep):
            processing_step.update_progress(percent, current_item, items_done)
            
    def processing_complete(self) -> None:
        """Handle processing completion."""
        processing_step = self.steps[3]
        if isinstance(processing_step, ProcessingStep):
            processing_step.progress_percent = 100
            processing_step.is_processing = False
            
        # Enable finish button
        self.can_proceed = True
        self.update_progress()
        
    def processing_error(self, error_msg: str) -> None:
        """Handle processing error."""
        logger.error(f"Processing error: {error_msg}")
        # TODO: Show error to user
        
    def handle_completion(self, data: Dict[str, Any]) -> None:
        """Handle wizard completion."""
        logger.info(f"Wizard completed with data: {data}")
        
        # Dismiss the wizard
        if hasattr(self.app, 'pop_screen'):
            self.app.pop_screen()
            
    def handle_cancellation(self) -> None:
        """Handle wizard cancellation."""
        logger.info("Wizard cancelled")
        
        # Dismiss the wizard
        if hasattr(self.app, 'pop_screen'):
            self.app.pop_screen()


class EmbeddingsWizardScreen(ModalScreen):
    """Modal screen wrapper for the embeddings wizard."""
    
    BINDINGS = [
        ("escape", "dismiss", "Cancel"),
    ]
    
    def __init__(self, on_complete: Optional[callable] = None):
        super().__init__()
        self.on_complete = on_complete
        
    def compose(self) -> ComposeResult:
        """Compose the wizard screen."""
        with Container(id="wizard-modal-container"):
            yield EmbeddingsCreationWizard(
                on_complete=self.handle_wizard_complete,
                on_cancel=self.action_dismiss
            )
            
    def handle_wizard_complete(self, data: Dict[str, Any]) -> None:
        """Handle wizard completion."""
        if self.on_complete:
            self.on_complete(data)
        self.dismiss()
        
    def action_dismiss(self) -> None:
        """Dismiss the wizard."""
        self.app.pop_screen()


class SimpleEmbeddingsWizard(Container):
    """Simplified embeddings wizard for integration into existing windows."""
    
    def __init__(self):
        super().__init__()
        self.wizard = None
        
    def compose(self) -> ComposeResult:
        """Compose the simplified wizard."""
        if not embeddings_available:
            # Show error message if dependencies not available
            with Container(classes="wizard-error-container"):
                yield Label(
                    "⚠️ Embeddings Dependencies Not Available",
                    classes="wizard-error-title"
                )
                yield Label(
                    "To use embeddings, install with: pip install tldw_chatbook[embeddings_rag]",
                    classes="wizard-error-message"
                )
                yield Button("Dismiss", id="dismiss-error", variant="error")
        else:
            # Create the wizard
            self.wizard = EmbeddingsCreationWizard(
                on_complete=self.handle_completion,
                on_cancel=self.handle_cancellation
            )
            yield self.wizard
            
    def handle_completion(self, data: Dict[str, Any]) -> None:
        """Handle wizard completion."""
        logger.info(f"Embeddings created: {data}")
        
        # Notify parent window
        self.post_message(EmbeddingsCreatedMessage(data))
        
    def handle_cancellation(self) -> None:
        """Handle wizard cancellation."""
        logger.info("Wizard cancelled")
        
        # Notify parent window
        self.post_message(EmbeddingsCancelledMessage())


########################################################################################################################
#
# Messages
#
########################################################################################################################

class EmbeddingsCreatedMessage:
    """Message sent when embeddings are created."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        

class EmbeddingsCancelledMessage:
    """Message sent when wizard is cancelled."""
    pass