# tldw_chatbook/Widgets/IngestLocalPlaintextWindow.py

from typing import TYPE_CHECKING
from pathlib import Path
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    ListView, ListItem, LoadingIndicator, Collapsible
)

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalPlaintextWindow(Vertical):
    """Window for ingesting local plaintext files."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.debug("IngestLocalPlaintextWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the plaintext ingestion form."""
        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("Text File Selection", classes="sidebar-title")
            
            # File selection buttons
            with Horizontal(classes="ingest-controls-row"):
                yield Button("Select Text Files", id="ingest-local-plaintext-select-files")
                yield Button("Clear Selection", id="ingest-local-plaintext-clear-files")
            yield Label("Selected Files:", classes="ingest-label")
            yield ListView(id="ingest-local-plaintext-files-list", classes="ingest-selected-files-list")
            
            # Metadata section
            yield Static("Metadata", classes="sidebar-title")
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title Override:")
                    yield Input(id="ingest-local-plaintext-title", placeholder="Use filename")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author:")
                    yield Input(id="ingest-local-plaintext-author", placeholder="Optional")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="ingest-local-plaintext-keywords", classes="ingest-textarea-small")
            
            # Plaintext Processing Options
            yield Static("Text Processing Options", classes="sidebar-title")
            
            yield Label("Text Encoding:")
            yield Select(
                [
                    ("UTF-8", "utf-8"), 
                    ("ASCII", "ascii"), 
                    ("Latin-1", "latin-1"), 
                    ("Auto-detect", "auto")
                ],
                id="ingest-local-plaintext-encoding",
                value="utf-8",
                prompt="Select encoding..."
            )
            
            yield Label("Line Ending:")
            yield Select(
                [
                    ("Auto", "auto"), 
                    ("Unix (LF)", "lf"), 
                    ("Windows (CRLF)", "crlf")
                ],
                id="ingest-local-plaintext-line-ending",
                value="auto",
                prompt="Select line ending..."
            )
            
            yield Checkbox("Remove Extra Whitespace", True, id="ingest-local-plaintext-remove-whitespace")
            yield Checkbox("Convert to Paragraphs", False, id="ingest-local-plaintext-paragraphs")
            
            yield Label("Split Pattern (Regex, optional):")
            yield Input(
                id="ingest-local-plaintext-split-pattern", 
                placeholder="e.g., \\n\\n+ for double newlines",
                tooltip="Regular expression pattern for custom text splitting"
            )
            
            # Database Options
            yield Static("Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if exists in database", False, id="ingest-local-plaintext-overwrite-existing")
            
            # Action section
            yield Button("Process Text Files", id="ingest-local-plaintext-process", variant="primary", classes="ingest-submit-button")
            yield LoadingIndicator(id="ingest-local-plaintext-loading", classes="hidden")
            yield TextArea(
                "",
                id="ingest-local-plaintext-status",
                read_only=True,
                classes="ingest-status-area"
            )