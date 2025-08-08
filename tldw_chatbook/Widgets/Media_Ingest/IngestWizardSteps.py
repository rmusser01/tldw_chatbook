# tldw_chatbook/Widgets/Media_Ingest/IngestWizardSteps.py
# Wizard steps for media ingestion using BaseWizard framework

from typing import TYPE_CHECKING, Optional, List, Dict, Any
from pathlib import Path
from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea,
    Label, ListView, ListItem, ProgressBar, LoadingIndicator, DataTable
)
from textual.reactive import reactive

from tldw_chatbook.UI.Wizards.BaseWizard import WizardStep, WizardStepConfig
from tldw_chatbook.config import get_media_ingestion_defaults
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

logger = logger.bind(module="IngestWizardSteps")

class SourceSelectionStep(WizardStep):
    """Step 1: Select media source (files or URLs)."""
    
    selected_files = reactive([])
    selected_urls = reactive([])
    
    def __init__(self, media_type: str = "video"):
        config = WizardStepConfig(
            id="source",
            title="Select Source",
            description="Choose files or enter URLs",
            icon="📁"
        )
        super().__init__(
            config=config,
            step_number=1,
            step_title="Source Selection",
            step_description="Select media files or enter URLs"
        )
        self.media_type = media_type
        self.add_class("source-selection-step")
    
    def compose(self) -> ComposeResult:
        """Compose the source selection UI."""
        with Container(classes="step-content"):
            yield Static("Select your media source", classes="step-header")
            
            with Horizontal(classes="source-selector"):
                # File drop zone
                with Container(classes="drop-zone", id="file-drop"):
                    icon = "🎬" if self.media_type == "video" else "🎵" if self.media_type == "audio" else "📄"
                    yield Static(icon, classes="drop-icon")
                    yield Static(f"Drop {self.media_type} files here", classes="drop-text")
                    yield Static("or", classes="drop-or")
                    yield Button("Browse Files", id="browse", variant="primary")
                
                # OR divider
                yield Static("OR", classes="or-divider")
                
                # URL input zone
                with Container(classes="url-zone"):
                    yield Static("🔗", classes="url-icon")
                    yield Label(f"Paste {self.media_type} URLs:")
                    yield TextArea(
                        "",
                        id="url-input",
                        classes="url-input-large"
                    )
                    yield Button("Add URLs", id="add-url", variant="primary")
            
            # Selected items list
            yield Label("Selected items:", classes="items-label")
            yield ListView(
                id="selected-items",
                classes="selected-items-list"
            )
    
    @on(Button.Pressed, "#browse")
    async def handle_browse(self, event: Button.Pressed) -> None:
        """Handle file browse button."""
        # Define filters based on media type
        if self.media_type == "video":
            filters = Filters(
                ("Video Files", lambda p: p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov")),
                ("All Files", lambda _: True)
            )
        elif self.media_type == "audio":
            filters = Filters(
                ("Audio Files", lambda p: p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")),
                ("All Files", lambda _: True)
            )
        else:
            filters = Filters(("All Files", lambda _: True))
        
        await self.app.push_screen(
            FileOpen(
                title=f"Select {self.media_type.title()} Files",
                filters=filters
            ),
            callback=self.add_file
        )
    
    async def add_file(self, path: Path | None) -> None:
        """Add a file to the selection."""
        if path:
            self.selected_files.append(path)
            list_view = self.query_one("#selected-items", ListView)
            list_view.append(ListItem(Static(f"📁 {path.name}")))
            
            # Trigger validation
            if self.wizard:
                self.wizard.validate_step()
    
    @on(Button.Pressed, "#add-url")
    def handle_add_urls(self, event: Button.Pressed) -> None:
        """Handle URL addition."""
        url_text = self.query_one("#url-input", TextArea).text
        if url_text.strip():
            urls = [url.strip() for url in url_text.split("\n") if url.strip()]
            self.selected_urls.extend(urls)
            
            list_view = self.query_one("#selected-items", ListView)
            for url in urls:
                list_view.append(ListItem(Static(f"🔗 {url[:50]}...")))
            
            # Clear input
            self.query_one("#url-input", TextArea).clear()
            
            # Trigger validation
            if self.wizard:
                self.wizard.validate_step()
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate that at least one source is selected."""
        errors = []
        if not self.selected_files and not self.selected_urls:
            errors.append("Please select at least one file or URL")
        
        return len(errors) == 0, errors
    
    def get_data(self) -> Dict[str, Any]:
        """Get the selected sources."""
        return {
            "files": self.selected_files,
            "urls": self.selected_urls,
            "media_type": self.media_type
        }


class ConfigurationStep(WizardStep):
    """Step 2: Configure processing options."""
    
    def __init__(self, media_type: str = "video"):
        config = WizardStepConfig(
            id="config",
            title="Configure",
            description="Set processing options",
            icon="⚙️"
        )
        super().__init__(
            config=config,
            step_number=2,
            step_title="Configuration",
            step_description="Configure processing options"
        )
        self.media_type = media_type
        self.transcription_service = TranscriptionService()
    
    def compose(self) -> ComposeResult:
        """Compose the configuration UI."""
        media_defaults = get_media_ingestion_defaults(self.media_type)
        
        with Container(classes="step-content"):
            yield Static("Configure processing options", classes="step-header")
            
            with Grid(classes="config-grid"):
                # Title and metadata
                yield Label("Title (optional):")
                yield Input(id="title", placeholder="Auto-detect from file")
                
                yield Label("Keywords:")
                yield Input(id="keywords", placeholder="Comma-separated tags")
                
                # Media-specific options
                if self.media_type in ["video", "audio"]:
                    yield Label("Language:")
                    yield Select(
                        [("Auto", "auto"), ("English", "en"), ("Spanish", "es")],
                        id="language",
                        value="auto"
                    )
                    
                    yield Label("Model:")
                    yield Select(
                        [("Fast", "base"), ("Accurate", "large")],
                        id="model",
                        value="base"
                    )
                    
                    # Checkboxes
                    yield Checkbox("Extract audio only", True, id="audio-only")
                    yield Checkbox("Include timestamps", True, id="timestamps")
                    yield Checkbox("Generate summary", True, id="summary")
                    yield Checkbox("Speaker diarization", False, id="diarize")
                else:
                    # Document options
                    yield Label("Chunk size:")
                    yield Input("500", id="chunk-size")
                    
                    yield Label("Chunk overlap:")
                    yield Input("200", id="chunk-overlap")
                    
                    yield Checkbox("Generate summary", True, id="summary")
                    yield Checkbox("Extract keywords", True, id="keywords-extract")
                    yield Checkbox("Enable OCR", False, id="ocr")
                    yield Checkbox("Adaptive chunking", False, id="adaptive")
    
    def validate(self) -> tuple[bool, List[str]]:
        """Configuration is always valid (uses defaults)."""
        return True, []
    
    def get_data(self) -> Dict[str, Any]:
        """Get configuration data."""
        data = {
            "title": self.query_one("#title", Input).value,
            "keywords": self.query_one("#keywords", Input).value,
        }
        
        if self.media_type in ["video", "audio"]:
            data.update({
                "language": self.query_one("#language", Select).value,
                "model": self.query_one("#model", Select).value,
                "audio_only": self.query_one("#audio-only", Checkbox).value,
                "timestamps": self.query_one("#timestamps", Checkbox).value,
                "summary": self.query_one("#summary", Checkbox).value,
                "diarize": self.query_one("#diarize", Checkbox).value,
            })
        else:
            data.update({
                "chunk_size": int(self.query_one("#chunk-size", Input).value or 500),
                "chunk_overlap": int(self.query_one("#chunk-overlap", Input).value or 200),
                "summary": self.query_one("#summary", Checkbox).value,
                "keywords_extract": self.query_one("#keywords-extract", Checkbox).value,
                "ocr": self.query_one("#ocr", Checkbox).value,
                "adaptive": self.query_one("#adaptive", Checkbox).value,
            })
        
        return data


class EnhancementStep(WizardStep):
    """Step 3: Optional enhancements and analysis."""
    
    def __init__(self, app_instance):
        config = WizardStepConfig(
            id="enhance",
            title="Enhance",
            description="Additional processing",
            icon="✨",
            can_skip=True
        )
        super().__init__(
            config=config,
            step_number=3,
            step_title="Enhancements",
            step_description="Optional enhancements"
        )
        self.app_instance = app_instance
    
    def compose(self) -> ComposeResult:
        """Compose enhancement options."""
        with Container(classes="step-content"):
            yield Static("Optional enhancements", classes="step-header")
            
            # Custom prompt
            yield Label("Custom analysis prompt (optional):")
            yield TextArea(
                "",
                id="custom-prompt",
                classes="prompt-area"
            )
            
            # API provider for analysis
            yield Label("Analysis provider:")
            api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
            api_options = [(name, name) for name in api_providers if name]
            if not api_options:
                api_options = [("Default", "default")]
            
            yield Select(
                api_options,
                id="api-provider",
                value=api_options[0][1] if api_options else None
            )
            
            # Additional options
            yield Checkbox("Advanced RAG indexing", False, id="rag-index")
            yield Checkbox("Generate Q&A pairs", False, id="qa-pairs")
            yield Checkbox("Extract entities", False, id="entities")
    
    def validate(self) -> tuple[bool, List[str]]:
        """Enhancement step is always valid (optional)."""
        return True, []
    
    def get_data(self) -> Dict[str, Any]:
        """Get enhancement options."""
        return {
            "custom_prompt": self.query_one("#custom-prompt", TextArea).text,
            "api_provider": self.query_one("#api-provider", Select).value,
            "rag_index": self.query_one("#rag-index", Checkbox).value,
            "qa_pairs": self.query_one("#qa-pairs", Checkbox).value,
            "entities": self.query_one("#entities", Checkbox).value,
        }


class ReviewStep(WizardStep):
    """Step 4: Review and confirm settings."""
    
    def __init__(self):
        config = WizardStepConfig(
            id="review",
            title="Review",
            description="Confirm settings",
            icon="👀"
        )
        super().__init__(
            config=config,
            step_number=4,
            step_title="Review & Process",
            step_description="Review settings and start processing"
        )
        self.settings_data = {}
    
    def compose(self) -> ComposeResult:
        """Compose review UI."""
        with Container(classes="step-content"):
            yield Static("Review your settings", classes="step-header")
            
            # Settings table
            yield DataTable(
                id="settings-table",
                show_header=False,
                classes="review-table"
            )
            
            # Process button
            yield Button(
                "Start Processing",
                id="start-process",
                variant="success",
                classes="process-button"
            )
            
            # Progress area (hidden initially)
            with Container(id="progress-area", classes="progress-area hidden"):
                yield ProgressBar(id="progress")
                yield Static("", id="progress-text")
                yield LoadingIndicator(id="loading")
    
    def on_show(self) -> None:
        """Update review when step is shown."""
        super().on_show()
        # Collect all data from previous steps
        if self.wizard:
            self.settings_data = self.wizard.get_all_data()
            self.update_review_table()
    
    def update_review_table(self) -> None:
        """Update the review table with collected data."""
        table = self.query_one("#settings-table", DataTable)
        table.clear()
        
        # Add columns if not present
        if not table.columns:
            table.add_column("Setting", key="setting")
            table.add_column("Value", key="value")
        
        # Add rows for each setting
        if "files" in self.settings_data:
            files = self.settings_data.get("files", [])
            if files:
                table.add_row("Files", f"{len(files)} selected")
        
        if "urls" in self.settings_data:
            urls = self.settings_data.get("urls", [])
            if urls:
                table.add_row("URLs", f"{len(urls)} entered")
        
        for key, value in self.settings_data.items():
            if key not in ["files", "urls"] and value:
                # Format the key nicely
                display_key = key.replace("_", " ").title()
                # Format boolean values
                if isinstance(value, bool):
                    display_value = "✓" if value else "✗"
                else:
                    display_value = str(value)
                table.add_row(display_key, display_value)
    
    @on(Button.Pressed, "#start-process")
    async def handle_start_process(self, event: Button.Pressed) -> None:
        """Start the processing."""
        # Show progress area
        progress_area = self.query_one("#progress-area")
        progress_area.remove_class("hidden")
        
        # Hide button
        event.button.add_class("hidden")
        
        # Start processing (would connect to actual processing logic)
        self.simulate_processing()
    
    @work(thread=True)
    def simulate_processing(self) -> None:
        """Simulate processing (replace with actual processing)."""
        import time
        for i in range(101):
            time.sleep(0.05)  # Simulate work
            self.call_from_thread(self.update_progress, i)
        
        self.call_from_thread(self.processing_complete)
    
    def update_progress(self, percent: int) -> None:
        """Update progress bar."""
        progress = self.query_one("#progress", ProgressBar)
        progress.update(total=100, progress=percent)
        
        progress_text = self.query_one("#progress-text", Static)
        progress_text.update(f"Processing... {percent}%")
    
    def processing_complete(self) -> None:
        """Mark processing as complete."""
        progress_text = self.query_one("#progress-text", Static)
        progress_text.update("✓ Processing complete!")
        
        loading = self.query_one("#loading", LoadingIndicator)
        loading.add_class("hidden")
        
        # Mark step as complete
        self.is_complete = True
        if self.wizard:
            self.wizard.can_proceed = True
    
    def validate(self) -> tuple[bool, List[str]]:
        """Review is valid when processing is complete."""
        return self.is_complete, [] if self.is_complete else ["Processing not complete"]
    
    def get_data(self) -> Dict[str, Any]:
        """Return all collected data."""
        return self.settings_data


# CSS for wizard steps
WIZARD_STEPS_CSS = """
/* Source Selection Step */
.source-selector {
    height: 20;
    align: center middle;
    margin: 2 0;
}

.drop-zone {
    width: 40%;
    height: 18;
    border: dashed $primary 2;
    align: center middle;
    background: $surface-lighten-1;
    padding: 2;
}

.drop-zone:hover {
    background: $surface-lighten-2;
    border-color: $accent;
}

.drop-icon {
    text-align: center;
    margin-bottom: 1;
}

.drop-text {
    margin: 1 0;
    text-align: center;
}

.or-divider {
    width: 10%;
    text-align: center;
    color: $text-muted;
    text-style: bold;
}

.url-zone {
    width: 40%;
    height: 18;
    border: solid $primary;
    padding: 2;
}

.url-icon {
    text-align: center;
    margin-bottom: 1;
}

.url-input-large {
    height: 8;
    margin: 1 0;
}

.selected-items-list {
    height: 10;
    border: round $surface;
    background: $surface-darken-1;
    margin-top: 1;
}

/* Configuration Step */
.config-grid {
    grid-size: 2 8;
    grid-columns: auto 1fr;
    grid-gutter: 1;
    margin: 2 0;
}

/* Enhancement Step */
.prompt-area {
    height: 10;
    margin: 1 0;
}

/* Review Step */
.review-table {
    height: 15;
    margin: 2 0;
}

.process-button {
    width: 100%;
    height: 3;
    text-style: bold;
    margin: 2 0;
}

.progress-area {
    margin: 2 0;
}

.progress-area.hidden {
    display: none;
}

/* Common step styles */
.step-header {
    text-style: bold;
    color: $primary;
    margin-bottom: 2;
    text-align: center;
}

.step-content {
    padding: 2;
}

.items-label {
    margin-top: 2;
    text-style: bold;
}
"""