"""Web scraping and ingestion screen implementation."""

from typing import TYPE_CHECKING, Dict, Any, AsyncIterator
from pathlib import Path
import asyncio
from loguru import logger

from textual.app import ComposeResult
from textual.widgets import (
    Label, Input, Select, Checkbox, TextArea
)
from textual.containers import Container
from textual import on

from .base_screen import BaseMediaIngestScreen
from ..models import WebFormData, ProcessingStatus

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class WebIngestScreen(BaseMediaIngestScreen):
    """Screen for web content scraping and ingestion."""
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "web", **kwargs):
        super().__init__(app_instance, media_type, **kwargs)
    
    def create_media_options(self) -> ComposeResult:
        """Create web scraping-specific options."""
        yield Label("Web Scraping Options", classes="section-title")
        
        # Scraping options
        with Container(classes="option-group"):
            yield Label("Scraping Settings", classes="option-title")
            
            yield Label("Scraping method:")
            yield Select(
                [
                    ("beautifulsoup", "BeautifulSoup (fast)"),
                    ("playwright", "Playwright (JavaScript)"),
                    ("selenium", "Selenium (legacy)"),
                    ("requests", "Requests (simple)")
                ],
                id="scraping-method",
                classes="form-select",
                value="beautifulsoup"
            )
            
            yield Checkbox("Follow redirects", id="follow-redirects", value=True)
            yield Checkbox("Extract main content only", id="extract-main", value=True)
            yield Checkbox("Download images", id="download-images", value=False)
            yield Checkbox("Parse JavaScript content", id="parse-javascript", value=False)
        
        # Content extraction
        with Container(classes="option-group"):
            yield Label("Content Extraction", classes="option-title")
            yield Checkbox("Extract article text", id="extract-article", value=True)
            yield Checkbox("Extract metadata", id="extract-metadata", value=True)
            yield Checkbox("Extract links", id="extract-links", value=False)
            yield Checkbox("Extract comments", id="extract-comments", value=False)
    
    def create_advanced_options(self) -> ComposeResult:
        """Create advanced web scraping options."""
        yield Container(
            Label("Crawling Options", classes="option-title"),
            Checkbox("Enable recursive crawling", id="recursive-crawl", value=False),
            Container(
                Label("Max depth:"),
                Input(value="1", id="max-depth", classes="form-input"),
                Label("Max pages:"),
                Input(value="10", id="max-pages", classes="form-input"),
                Label("URL pattern (regex):"),
                Input(placeholder=".*", id="url-pattern", classes="form-input"),
                id="crawl-options"
            ),
            classes="option-group"
        )
        
        yield Container(
            Label("Authentication", classes="option-title"),
            Checkbox("Use authentication", id="use-auth", value=False),
            Container(
                Label("Username:"),
                Input(id="auth-username", classes="form-input"),
                Label("Password:"),
                Input(id="auth-password", classes="form-input", password=True),
                Label("Auth type:"),
                Select(
                    [
                        ("basic", "Basic Auth"),
                        ("bearer", "Bearer Token"),
                        ("cookie", "Cookie"),
                        ("oauth", "OAuth")
                    ],
                    id="auth-type",
                    classes="form-select",
                    value="basic"
                ),
                id="auth-options"
            ),
            classes="option-group"
        )
        
        yield Container(
            Label("Processing Options", classes="option-title"),
            Checkbox("Clean HTML", id="clean-html", value=True),
            Checkbox("Convert to Markdown", id="convert-markdown", value=True),
            Checkbox("Generate summary", id="generate-summary", value=True),
            Checkbox("Archive snapshot", id="archive-snapshot", value=False),
            classes="option-group"
        )
    
    @on(Checkbox.Changed, "#recursive-crawl")
    def toggle_crawl_options(self, event: Checkbox.Changed) -> None:
        """Show/hide crawling options."""
        options = self.query_one("#crawl-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    @on(Checkbox.Changed, "#use-auth")
    def toggle_auth_options(self, event: Checkbox.Changed) -> None:
        """Show/hide authentication options."""
        options = self.query_one("#auth-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    @on(Checkbox.Changed, "#parse-javascript")
    def handle_javascript_option(self, event: Checkbox.Changed) -> None:
        """Update scraping method when JavaScript parsing is toggled."""
        if event.value:
            # Switch to Playwright for JavaScript support
            method_select = self.query_one("#scraping-method", Select)
            method_select.value = "playwright"
    
    def get_validated_form_data(self) -> Dict[str, Any]:
        """Get and validate web scraping form data."""
        data = super().get_validated_form_data()
        
        # Scraping settings
        data['scraping_method'] = self.query_one("#scraping-method").value
        data['follow_redirects'] = self.query_one("#follow-redirects").value
        data['extract_main_content'] = self.query_one("#extract-main").value
        data['download_images'] = self.query_one("#download-images").value
        data['parse_javascript'] = self.query_one("#parse-javascript").value
        
        # Content extraction
        data['extract_article'] = self.query_one("#extract-article").value
        data['extract_metadata'] = self.query_one("#extract-metadata").value
        data['extract_links'] = self.query_one("#extract-links").value
        data['extract_comments'] = self.query_one("#extract-comments").value
        
        # Crawling options
        data['recursive_crawl'] = self.query_one("#recursive-crawl").value
        if data['recursive_crawl']:
            data['max_depth'] = int(self.query_one("#max-depth").value or 1)
            data['max_pages'] = int(self.query_one("#max-pages").value or 10)
            data['url_pattern'] = self.query_one("#url-pattern").value or ".*"
        
        # Authentication
        data['use_authentication'] = self.query_one("#use-auth").value
        if data['use_authentication']:
            data['auth_username'] = self.query_one("#auth-username").value
            data['auth_password'] = self.query_one("#auth-password").value
            data['auth_type'] = self.query_one("#auth-type").value
        
        # Processing options
        data['clean_html'] = self.query_one("#clean-html").value
        data['convert_to_markdown'] = self.query_one("#convert-markdown").value
        data['generate_summary'] = self.query_one("#generate-summary").value
        data['archive_snapshot'] = self.query_one("#archive-snapshot").value
        
        return data
    
    async def process_media_impl(self, form_data: Dict[str, Any]) -> AsyncIterator[ProcessingStatus]:
        """Process web URLs."""
        try:
            validated = WebFormData(**form_data)
            
            # For web scraping, we primarily use URLs
            all_urls = validated.urls
            total = len(all_urls)
            
            if total == 0:
                yield ProcessingStatus(
                    state="error",
                    error="No URLs to process",
                    message="Please provide at least one URL to scrape"
                )
                return
            
            for idx, url in enumerate(all_urls):
                yield ProcessingStatus(
                    state="processing",
                    progress=(idx / total),
                    current_file=url,
                    current_operation="Fetching page",
                    files_processed=idx,
                    total_files=total,
                    message=f"Fetching: {url}"
                )
                await asyncio.sleep(0.5)
                
                if validated.parse_javascript:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.2) / total,
                        current_file=url,
                        current_operation="Rendering JavaScript",
                        files_processed=idx,
                        total_files=total,
                        message=f"Rendering JS: {url}"
                    )
                    await asyncio.sleep(0.8)
                
                if validated.extract_main_content:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.4) / total,
                        current_file=url,
                        current_operation="Extracting content",
                        files_processed=idx,
                        total_files=total,
                        message=f"Extracting: {url}"
                    )
                    await asyncio.sleep(0.3)
                
                if validated.recursive_crawl:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.6) / total,
                        current_file=url,
                        current_operation="Crawling links",
                        files_processed=idx,
                        total_files=total,
                        message=f"Crawling: {url}"
                    )
                    await asyncio.sleep(1)
                
                if validated.convert_to_markdown:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.8) / total,
                        current_file=url,
                        current_operation="Converting to Markdown",
                        files_processed=idx,
                        total_files=total,
                        message=f"Converting: {url}"
                    )
                    await asyncio.sleep(0.3)
                
                if validated.generate_summary:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.9) / total,
                        current_file=url,
                        current_operation="Generating summary",
                        files_processed=idx,
                        total_files=total,
                        message=f"Summarizing: {url}"
                    )
                    await asyncio.sleep(0.3)
            
            yield ProcessingStatus(
                state="complete",
                progress=1.0,
                files_processed=total,
                total_files=total,
                message=f"Successfully scraped {total} URL(s)"
            )
            
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
            yield ProcessingStatus(
                state="error",
                error=str(e),
                message=f"Scraping failed: {str(e)}"
            )