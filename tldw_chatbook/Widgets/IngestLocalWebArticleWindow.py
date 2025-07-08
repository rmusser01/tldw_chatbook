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
from textual.reactive import reactive
from textual.widget import Widget
from ..config import get_media_ingestion_defaults

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalWebArticleWindow(Vertical):
    """Window for ingesting web articles."""
    
    # Reactive attributes
    scraping_mode = reactive("single")
    extract_cookies = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.debug("IngestLocalWebArticleWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the web article ingestion form."""
        # Get web article-specific default chunking settings from config
        web_article_defaults = get_media_ingestion_defaults("web_article")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            # Scraping Mode Selection
            yield Static("Scraping Mode", classes="sidebar-title")
            yield Label("Select scraping mode:")
            yield Select(
                [
                    ("Single Page URLs", "single"),
                    ("Website Crawl", "crawl"),
                    ("Sitemap Import", "sitemap")
                ],
                value="single",
                id="ingest-local-web-mode",
                allow_blank=False
            )
            
            # URL Input Section
            yield Static("URL Input", classes="sidebar-title")
            yield Label("Enter URLs (one per line):", id="ingest-local-web-url-label")
            yield TextArea(id="ingest-local-web-urls", classes="ingest-textarea-medium")
            
            yield Button("Clear URLs", id="ingest-local-web-clear-urls", classes="ingest-url-button")
            yield Button("Import from File", id="ingest-local-web-import-urls", classes="ingest-url-button")
            yield Button("Remove Duplicates", id="ingest-local-web-remove-duplicates", classes="ingest-url-button")
            
            yield Label("URL Count: 0 valid, 0 invalid", id="ingest-local-web-url-count", classes="ingest-label")
            
            # Crawling Configuration (shown when mode is crawl/sitemap)
            with Collapsible(title="Crawling Configuration", collapsed=True, id="ingest-local-web-crawl-config"):
                with Horizontal(classes="ingest-controls-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Max Depth:")
                        yield Input("3", id="ingest-local-web-max-depth", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Max Pages:")
                        yield Input("100", id="ingest-local-web-max-pages", type="integer")
                
                yield Label("Include URL Patterns (one per line):")
                yield TextArea(id="ingest-local-web-include-patterns", classes="ingest-textarea-small")
                
                yield Label("Exclude URL Patterns (one per line):")
                yield TextArea(id="ingest-local-web-exclude-patterns", classes="ingest-textarea-small")
                
                yield Checkbox("Same Domain Only", True, id="ingest-local-web-same-domain")
            
            # Import Options
            with Collapsible(title="Import Options", collapsed=True):
                yield Static("Supported file types:", classes="ingest-label")
                yield Static("• Browser bookmarks (HTML export)", classes="ingest-label")
                yield Static("• CSV files (with 'url' column)", classes="ingest-label")
                yield Static("• Text files (one URL per line)", classes="ingest-label")
                yield Static("• Chrome/Firefox bookmark databases", classes="ingest-label")
            
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
            
            # Extraction Options
            yield Static("Extraction Options", classes="sidebar-title")
            yield Checkbox("Extract Main Content Only", True, id="ingest-local-web-main-content")
            yield Checkbox("Include Images", False, id="ingest-local-web-include-images")
            yield Checkbox("Include Tables", False, id="ingest-local-web-include-tables")
            yield Checkbox("Include Comments", False, id="ingest-local-web-include-comments")
            yield Checkbox("Follow Redirects", True, id="ingest-local-web-follow-redirects")
            yield Checkbox("Stealth Mode (avoid bot detection)", True, id="ingest-local-web-stealth-mode")
            
            # Authentication Options
            with Collapsible(title="Authentication Options", collapsed=True):
                yield Label("Cookie String (optional):")
                yield Input(id="ingest-local-web-cookies", placeholder="name=value; name2=value2")
                yield Label("User Agent:")
                yield Input(id="ingest-local-web-user-agent", placeholder="Default browser agent")
                yield Checkbox("Extract Cookies from Browser", False, id="ingest-local-web-extract-cookies")
                yield Select(
                    [
                        ("All Browsers", "all"),
                        ("Chrome", "chrome"),
                        ("Firefox", "firefox"),
                        ("Edge", "edge"),
                        ("Safari", "safari")
                    ],
                    value="all",
                    id="ingest-local-web-browser",
                    allow_blank=False
                )
            
            # Advanced Options
            with Collapsible(title="Advanced Options", collapsed=True):
                yield Label("CSS Selector for Content:")
                yield Input(id="ingest-local-web-css-selector", placeholder="Auto-detect")
                yield Checkbox("JavaScript Rendering", False, id="ingest-local-web-js-render")
                yield Label("Wait Time (seconds):")
                yield Input("3", id="ingest-local-web-wait-time", type="integer")
            
            # Performance Configuration
            with Collapsible(title="Performance Configuration", collapsed=True):
                with Horizontal(classes="ingest-controls-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Retry Attempts:")
                        yield Input("3", id="ingest-local-web-retries", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Timeout (seconds):")
                        yield Input("60", id="ingest-local-web-timeout", type="integer")
                
                with Horizontal(classes="ingest-controls-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Rate Limit Delay (seconds):")
                        yield Input("0.5", id="ingest-local-web-rate-limit", type="number")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Concurrent Requests:")
                        yield Input("5", id="ingest-local-web-concurrent", type="integer")
            
            # Processing Options
            with Collapsible(title="Content Processing (Optional)", collapsed=True):
                yield Checkbox("Summarize Content with LLM", False, id="ingest-local-web-summarize")
                yield Label("LLM Provider:")
                yield Select(
                    [
                        ("None", "none"),
                        ("OpenAI", "openai"),
                        ("Anthropic", "anthropic"),
                        ("Local LLM", "local")
                    ],
                    value="none",
                    id="ingest-local-web-llm-provider",
                    allow_blank=False
                )
                yield Label("Custom Analysis Prompt (optional):")
                yield TextArea(id="ingest-local-web-custom-prompt", classes="ingest-textarea-small")
            
            # Chunking Options
            with Collapsible(title="Chunking Options", collapsed=True, id="ingest-local-web-chunking-collapsible"):
                yield Checkbox("Perform Chunking", True, id="ingest-local-web-perform-chunking")
                yield Label("Chunking Method:")
                chunk_method_options = [
                    ("paragraphs", "paragraphs"),
                    ("sentences", "sentences"),
                    ("tokens", "tokens"),
                    ("words", "words"),
                    ("sliding_window", "sliding_window")
                ]
                yield Select(chunk_method_options, id="ingest-local-web-chunk-method", 
                            value=web_article_defaults.get("chunk_method", "paragraphs"),
                            prompt="Select chunking method...")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input(str(web_article_defaults.get("chunk_size", 500)), 
                                   id="ingest-local-web-chunk-size", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input(str(web_article_defaults.get("chunk_overlap", 200)), 
                                   id="ingest-local-web-chunk-overlap", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(web_article_defaults.get("chunk_language", ""), 
                           id="ingest-local-web-chunk-lang", 
                           placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", 
                              web_article_defaults.get("use_adaptive_chunking", False), 
                              id="ingest-local-web-adaptive-chunking")
                yield Checkbox("Use Multi-level Chunking", 
                              web_article_defaults.get("use_multi_level_chunking", False), 
                              id="ingest-local-web-multi-level-chunking")
            
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
    
    def on_mount(self) -> None:
        """Set up event handlers when the widget is mounted."""
        # Watch for mode changes
        mode_select = self.query_one("#ingest-local-web-mode", Select)
        mode_select.watch(self, "value", self._on_mode_change)
        
        # Watch for cookie extraction toggle
        cookie_checkbox = self.query_one("#ingest-local-web-extract-cookies", Checkbox)
        cookie_checkbox.watch(self, "value", self._on_cookie_extract_change)
    
    def _on_mode_change(self, event) -> None:
        """Handle scraping mode changes."""
        mode = self.query_one("#ingest-local-web-mode", Select).value
        self.scraping_mode = mode
        
        # Update URL label based on mode
        url_label = self.query_one("#ingest-local-web-url-label", Label)
        if mode == "single":
            url_label.update("Enter URLs (one per line):")
        elif mode == "crawl":
            url_label.update("Enter starting URL for crawl:")
        elif mode == "sitemap":
            url_label.update("Enter sitemap URL:")
        
        # Show/hide crawling configuration
        crawl_config = self.query_one("#ingest-local-web-crawl-config", Collapsible)
        if mode in ["crawl", "sitemap"]:
            crawl_config.collapsed = False
        else:
            crawl_config.collapsed = True
        
        logger.debug(f"Scraping mode changed to: {mode}")
    
    def _on_cookie_extract_change(self, event) -> None:
        """Handle cookie extraction toggle."""
        extract = self.query_one("#ingest-local-web-extract-cookies", Checkbox).value
        self.extract_cookies = extract
        
        # Show/hide browser selection
        browser_select = self.query_one("#ingest-local-web-browser", Select)
        browser_select.disabled = not extract
        
        # Disable manual cookie input if extracting
        cookie_input = self.query_one("#ingest-local-web-cookies", Input)
        if extract:
            cookie_input.disabled = True
            cookie_input.placeholder = "Will extract from selected browser"
        else:
            cookie_input.disabled = False
            cookie_input.placeholder = "name=value; name2=value2"
        
        logger.debug(f"Cookie extraction toggled: {extract}")
    
    def watch_scraping_mode(self, old_value: str, new_value: str) -> None:
        """React to scraping mode changes."""
        logger.debug(f"Scraping mode changed from {old_value} to {new_value}")
    
    def watch_extract_cookies(self, old_value: bool, new_value: bool) -> None:
        """React to cookie extraction toggle."""
        logger.debug(f"Cookie extraction changed from {old_value} to {new_value}")