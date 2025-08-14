"""Container-based media ingestion panels for embedding in screens."""

from typing import TYPE_CHECKING, Optional
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Label, Input, Select, TextArea, Static

# from .models import MediaIngestModel, IngestType, IngestSource
# from ...Utils.Emoji_Handling import get_char  # Not needed for now

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class BaseMediaIngestPanel(Container):
    """Base container for media ingestion panels."""
    
    DEFAULT_CSS = """
    .media-nav {
        height: 3;
        width: 100%;
        background: $surface;
        border-bottom: solid $primary;
        padding: 0 1;
    }
    
    .media-nav-button {
        margin: 0 1;
        height: 3;
    }
    
    .media-nav-button.active {
        background: $primary;
        text-style: bold;
    }
    
    .media-form {
        padding: 1;
        height: 100%;
        overflow-y: auto;
    }
    
    .form-group {
        margin-bottom: 1;
    }
    
    .form-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .small-textarea {
        height: 5;
    }
    
    .form-buttons {
        margin-top: 2;
        height: 3;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', media_type: str, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_type = media_type
    
    def compose(self) -> ComposeResult:
        """Compose the panel UI."""
        with Vertical(id=f"{self.media_type}-panel-container"):
            # Navigation bar for media types
            yield from self.compose_navigation()
            
            # Form content
            with Container(id=f"{self.media_type}-form-container", classes="media-form"):
                yield from self.compose_form()
    
    def compose_navigation(self) -> ComposeResult:
        """Compose the media type navigation bar."""
        with Horizontal(id="media-type-nav", classes="media-nav"):
            media_types = ["video", "audio", "pdf", "document", "ebook", "web"]
            for mtype in media_types:
                btn_classes = "media-nav-button"
                if mtype == self.media_type:
                    btn_classes += " active"
                yield Button(
                    mtype.title(),
                    id=f"nav-{mtype}",
                    classes=btn_classes
                )
    
    def compose_form(self) -> ComposeResult:
        """Override in subclasses to provide specific form fields."""
        yield Label(f"{self.media_type.title()} Ingestion Form")
        
        # Common fields
        with Vertical(classes="form-group"):
            yield Label("Ingest Type:")
            yield Select(
                [("Local", "local"), ("TLDW API", "tldw_api")],
                id=f"{self.media_type}-ingest-type"
            )
        
        with Vertical(classes="form-group"):
            yield Label("URL/Path:")
            yield Input(placeholder="Enter URL or file path", id=f"{self.media_type}-input")
        
        with Horizontal(classes="form-buttons"):
            yield Button("Process", variant="primary", id=f"{self.media_type}-process")
            yield Button("Clear", variant="default", id=f"{self.media_type}-clear")


class VideoIngestPanel(BaseMediaIngestPanel):
    """Video ingestion panel."""
    
    def compose_form(self) -> ComposeResult:
        """Compose video-specific form."""
        yield Label("🎬 Video Ingestion", classes="form-title")
        
        with Vertical(classes="form-group"):
            yield Label("Ingest Type:")
            yield Select(
                [("Local File", "local"), ("TLDW API", "tldw"), ("YouTube URL", "youtube")],
                id="video-ingest-type"
            )
        
        with Vertical(classes="form-group"):
            yield Label("Video Source:")
            yield Input(
                placeholder="Enter file path or URL",
                id="video-source"
            )
        
        with Vertical(classes="form-group"):
            yield Label("Processing Options:")
            yield TextArea(
                "transcribe=true\nsummarize=true\nchunking=true",
                id="video-options",
                classes="small-textarea"
            )
        
        with Horizontal(classes="form-buttons"):
            yield Button("🎬 Process Video", variant="primary", id="video-process")
            yield Button("Clear", variant="default", id="video-clear")


class AudioIngestPanel(BaseMediaIngestPanel):
    """Audio ingestion panel."""
    
    def compose_form(self) -> ComposeResult:
        """Compose audio-specific form."""
        yield Label("🎵 Audio Ingestion", classes="form-title")
        
        with Vertical(classes="form-group"):
            yield Label("Ingest Type:")
            yield Select(
                [("Local File", "local"), ("TLDW API", "tldw")],
                id="audio-ingest-type"
            )
        
        with Vertical(classes="form-group"):
            yield Label("Audio Source:")
            yield Input(
                placeholder="Enter file path",
                id="audio-source"
            )
        
        with Vertical(classes="form-group"):
            yield Label("Transcription Settings:")
            yield TextArea(
                "language=auto\nspeaker_diarization=false",
                id="audio-options",
                classes="small-textarea"
            )
        
        with Horizontal(classes="form-buttons"):
            yield Button("🎵 Process Audio", variant="primary", id="audio-process")
            yield Button("Clear", variant="default", id="audio-clear")


class PDFIngestPanel(BaseMediaIngestPanel):
    """PDF ingestion panel."""
    
    def compose_form(self) -> ComposeResult:
        """Compose PDF-specific form."""
        yield Label("📄 PDF Ingestion", classes="form-title")
        
        with Vertical(classes="form-group"):
            yield Label("PDF Source:")
            yield Input(
                placeholder="Enter PDF file path",
                id="pdf-source"
            )
        
        with Vertical(classes="form-group"):
            yield Label("Extraction Options:")
            yield TextArea(
                "extract_images=true\nocr_enabled=false\npreserve_formatting=true",
                id="pdf-options",
                classes="small-textarea"
            )
        
        with Horizontal(classes="form-buttons"):
            yield Button("📄 Process PDF", variant="primary", id="pdf-process")
            yield Button("Clear", variant="default", id="pdf-clear")


class DocumentIngestPanel(BaseMediaIngestPanel):
    """Document ingestion panel."""
    
    def compose_form(self) -> ComposeResult:
        """Compose document-specific form."""
        yield Label("📝 Document Ingestion", classes="form-title")
        
        with Vertical(classes="form-group"):
            yield Label("Document Source:")
            yield Input(
                placeholder="Enter document file path",
                id="document-source"
            )
        
        with Vertical(classes="form-group"):
            yield Label("Document Type:")
            yield Select(
                [("Word", "docx"), ("Text", "txt"), ("Markdown", "md"), ("RTF", "rtf")],
                id="document-type"
            )
        
        with Horizontal(classes="form-buttons"):
            yield Button("📝 Process Document", variant="primary", id="document-process")
            yield Button("Clear", variant="default", id="document-clear")


class EbookIngestPanel(BaseMediaIngestPanel):
    """Ebook ingestion panel."""
    
    def compose_form(self) -> ComposeResult:
        """Compose ebook-specific form."""
        yield Label("📚 Ebook Ingestion", classes="form-title")
        
        with Vertical(classes="form-group"):
            yield Label("Ebook Source:")
            yield Input(
                placeholder="Enter ebook file path",
                id="ebook-source"
            )
        
        with Vertical(classes="form-group"):
            yield Label("Ebook Format:")
            yield Select(
                [("EPUB", "epub"), ("MOBI", "mobi"), ("AZW3", "azw3")],
                id="ebook-format"
            )
        
        with Horizontal(classes="form-buttons"):
            yield Button("📚 Process Ebook", variant="primary", id="ebook-process")
            yield Button("Clear", variant="default", id="ebook-clear")


class WebIngestPanel(BaseMediaIngestPanel):
    """Web ingestion panel."""
    
    def compose_form(self) -> ComposeResult:
        """Compose web-specific form."""
        yield Label("🌐 Web Ingestion", classes="form-title")
        
        with Vertical(classes="form-group"):
            yield Label("Web URL:")
            yield Input(
                placeholder="Enter website URL",
                id="web-source"
            )
        
        with Vertical(classes="form-group"):
            yield Label("Scraping Options:")
            yield TextArea(
                "follow_links=false\nmax_depth=1\ninclude_images=false",
                id="web-options",
                classes="small-textarea"
            )
        
        with Horizontal(classes="form-buttons"):
            yield Button("🌐 Process Web", variant="primary", id="web-process")
            yield Button("Clear", variant="default", id="web-clear")