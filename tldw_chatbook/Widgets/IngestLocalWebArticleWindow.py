# tldw_chatbook/Widgets/IngestLocalWebArticleWindow.py

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

class IngestLocalWebArticleWindow(Vertical):
    """Window for ingesting web articles."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.debug("IngestLocalWebArticleWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the web article ingestion form."""
        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("Web Article URLs", classes="sidebar-title")
            
            # URL Input Section
            yield Label("Enter URLs (one per line):")
            yield TextArea(id="ingest-local-web-urls", classes="ingest-textarea-medium")
            
            with Horizontal(classes="ingest-controls-row"):
                yield Button("Clear URLs", id="ingest-local-web-clear-urls")
                yield Button("Import from File", id="ingest-local-web-import-urls")
                yield Button("Remove Duplicates", id="ingest-local-web-remove-duplicates")
            
            yield Label("URL Count: 0 valid, 0 invalid", id="ingest-local-web-url-count", classes="ingest-label")
            
            # Metadata Section
            yield Static("Metadata", classes="sidebar-title")
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title Override:")
                    yield Input(id="ingest-local-web-title", placeholder="Use page title")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author Override:")
                    yield Input(id="ingest-local-web-author", placeholder="Extract from page")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="ingest-local-web-keywords", classes="ingest-textarea-small")
            
            # Web Scraping Options
            yield Static("Web Scraping Options", classes="sidebar-title")
            yield Checkbox("Extract Main Content Only", True, id="ingest-local-web-main-content")
            yield Checkbox("Include Images", False, id="ingest-local-web-include-images")
            yield Checkbox("Follow Redirects", True, id="ingest-local-web-follow-redirects")
            
            with Collapsible(title="Authentication Options", collapsed=True):
                yield Label("Cookie String (optional):")
                yield Input(id="ingest-local-web-cookies", placeholder="name=value; name2=value2")
                yield Label("User Agent:")
                yield Input(id="ingest-local-web-user-agent", placeholder="Default browser agent")
            
            with Collapsible(title="Advanced Options", collapsed=True):
                yield Label("CSS Selector for Content:")
                yield Input(id="ingest-local-web-css-selector", placeholder="Auto-detect")
                yield Checkbox("JavaScript Rendering", False, id="ingest-local-web-js-render")
                yield Label("Wait Time (seconds):")
                yield Input("3", id="ingest-local-web-wait-time", type="integer")
            
            # Action Section
            yield Button("Scrape Articles", id="ingest-local-web-process", variant="primary", classes="ingest-submit-button")
            yield LoadingIndicator(id="ingest-local-web-loading", classes="hidden")
            
            # Progress section
            with Container(id="ingest-local-web-progress", classes="hidden"):
                yield Static("Progress: 0/0", id="ingest-local-web-progress-text")
                yield Static("✅ 0  ❌ 0  ⏳ 0", id="ingest-local-web-counters")
            
            yield TextArea(
                "",
                id="ingest-local-web-status",
                read_only=True,
                classes="ingest-status-area"
            )