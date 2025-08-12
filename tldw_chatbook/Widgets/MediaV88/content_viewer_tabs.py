"""
Content Viewer Tabs for Media UI V88.

Provides tabbed interface for viewing media content and analysis.
"""

from typing import TYPE_CHECKING, Dict, Any, Optional, List
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    TabbedContent, TabPane, Markdown, Button, Label, 
    Input, Select, TextArea, Static
)
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ...app import TldwCli


class MediaAnalysisRequestEvent(Message):
    """Event fired when analysis generation is requested."""
    
    def __init__(
        self,
        media_id: int,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        **params
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.params = params


class ContentViewerTabs(Container):
    """
    Tabbed viewer for media content and analysis.
    
    Features:
    - Content tab with markdown rendering and search
    - Analysis tab with generation and viewing
    - Version history for analyses
    """
    
    DEFAULT_CSS = """
    ContentViewerTabs {
        height: 1fr;
        min-height: 0;
        layout: vertical;
    }
    
    ContentViewerTabs TabbedContent {
        height: 100%;
        min-height: 0;
    }
    
    ContentViewerTabs TabPane {
        height: 100%;
        padding: 1;
        overflow-y: auto;
    }
    
    .content-controls {
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: solid $primary-lighten-3;
        margin-bottom: 1;
    }
    
    .content-search {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    #content-search-input {
        width: 1fr;
        margin-right: 1;
    }
    
    .search-nav {
        layout: horizontal;
        height: 3;
    }
    
    .search-nav Button {
        width: auto;
        min-width: 8;
        margin-right: 1;
    }
    
    .search-status {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }
    
    #content-display {
        height: 1fr;
        min-height: 0;
        padding: 1;
        background: $primary-background;
        border: solid $primary-lighten-3;
        overflow-y: auto;
    }
    
    .analysis-controls {
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: solid $primary-lighten-3;
        margin-bottom: 1;
    }
    
    .provider-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    .provider-label {
        width: 12;
        text-align: right;
        padding-right: 1;
        content-align: center middle;
    }
    
    #provider-select {
        width: 20;
        margin-right: 1;
    }
    
    #model-select {
        width: 1fr;
    }
    
    .prompt-section {
        layout: vertical;
        margin-bottom: 1;
    }
    
    .prompt-label {
        text-style: bold;
        margin-bottom: 0;
    }
    
    .prompt-input {
        width: 100%;
        min-height: 3;
        max-height: 10;
        margin-top: 0;
    }
    
    .generate-button {
        width: auto;
        min-width: 15;
        align-horizontal: center;
        background: $success;
    }
    
    #analysis-display {
        height: 1fr;
        min-height: 0;
        padding: 1;
        background: $primary-background;
        border: solid $primary-lighten-3;
        overflow-y: auto;
    }
    
    .analysis-nav {
        layout: horizontal;
        height: 3;
        margin-top: 1;
        align-horizontal: center;
    }
    
    .analysis-nav Button {
        width: auto;
        min-width: 10;
        margin: 0 1;
    }
    
    .version-info {
        width: auto;
        padding: 0 2;
        color: $text-muted;
    }
    
    .empty-state {
        text-align: center;
        color: $text-muted;
        padding: 4;
    }
    
    .highlight {
        background: $warning 30%;
        color: $text;
    }
    """
    
    # Reactive properties
    current_media: reactive[Optional[Dict[str, Any]]] = reactive(None)
    content_search_term: reactive[str] = reactive("")
    current_analysis: reactive[Optional[str]] = reactive(None)
    analysis_versions: reactive[List[Dict[str, Any]]] = reactive([])
    current_version_index: reactive[int] = reactive(0)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the content viewer tabs."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.search_results: List[tuple] = []
        self.current_search_index = 0
        self.available_providers = self._get_available_providers()
        self.provider_models = self._get_provider_models()
    
    def _get_available_providers(self) -> List[str]:
        """Get available LLM providers from config."""
        # This should come from app config
        return [
            "openai",
            "anthropic", 
            "google",
            "local-llm",
            "groq",
            "cohere",
            "mistralai"
        ]
    
    def _get_provider_models(self) -> Dict[str, List[str]]:
        """Get available models for each provider."""
        return {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "google": ["gemini-pro", "gemini-pro-vision"],
            "local-llm": ["llama2", "mistral", "mixtral"],
            "groq": ["llama2-70b", "mixtral-8x7b"],
            "cohere": ["command", "command-light"],
            "mistralai": ["mistral-large", "mistral-medium", "mistral-small"]
        }
    
    def compose(self) -> ComposeResult:
        """Compose the content viewer tabs UI."""
        with TabbedContent(id="media-tabs"):
            # Content Tab
            with TabPane("Content", id="content-tab"):
                # Search controls
                with Container(classes="content-controls"):
                    with Horizontal(classes="content-search"):
                        yield Input(
                            placeholder="Search in content...",
                            id="content-search-input"
                        )
                        yield Button("Find", id="find-button", variant="primary")
                        yield Button("Clear", id="clear-search-button")
                    
                    with Horizontal(classes="search-nav", id="search-nav"):
                        yield Button("◀ Prev", id="prev-match", disabled=True)
                        yield Button("Next ▶", id="next-match", disabled=True)
                        yield Static("No matches", id="search-status", classes="search-status")
                
                # Content display
                yield Markdown(
                    "# No Content\n\nSelect a media item to view its content.",
                    id="content-display"
                )
            
            # Analysis Tab
            with TabPane("Analysis", id="analysis-tab"):
                # Analysis generation controls
                with Container(classes="analysis-controls"):
                    # Provider and model selection
                    with Horizontal(classes="provider-row"):
                        yield Label("Provider:", classes="provider-label")
                        yield Select(
                            options=[(p, p) for p in self.available_providers],
                            value=self.available_providers[0] if self.available_providers else None,
                            id="provider-select"
                        )
                        yield Select(
                            options=[],
                            id="model-select",
                            prompt="Select model..."
                        )
                    
                    # System prompt
                    with Container(classes="prompt-section"):
                        yield Label("System Prompt:", classes="prompt-label")
                        yield TextArea(
                            "You are an expert analyst. Provide a comprehensive analysis of the provided content.",
                            id="system-prompt-input",
                            classes="prompt-input"
                        )
                    
                    # User prompt
                    with Container(classes="prompt-section"):
                        yield Label("User Prompt:", classes="prompt-label")
                        yield TextArea(
                            "Please analyze the following content and provide key insights, summary, and recommendations.",
                            id="user-prompt-input",
                            classes="prompt-input"
                        )
                    
                    # Generate button
                    yield Button(
                        "Generate Analysis",
                        id="generate-analysis-button",
                        classes="generate-button",
                        variant="success"
                    )
                
                # Analysis display
                yield Markdown(
                    "# No Analysis\n\nGenerate or select an analysis to view.",
                    id="analysis-display"
                )
                
                # Version navigation
                with Horizontal(classes="analysis-nav", id="analysis-nav"):
                    yield Button("◀ Previous", id="prev-version", disabled=True)
                    yield Static("No versions", id="version-info", classes="version-info")
                    yield Button("Next ▶", id="next-version", disabled=True)
                    yield Button("Save", id="save-analysis", disabled=True)
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        logger.info("ContentViewerTabs mounted")
        
        # Hide search navigation initially
        try:
            search_nav = self.query_one("#search-nav", Horizontal)
            search_nav.styles.display = "none"
        except Exception:
            pass
        
        # Update model options for default provider
        if self.available_providers:
            self._update_model_options(self.available_providers[0])
    
    @on(Select.Changed, "#provider-select")
    def handle_provider_change(self, event: Select.Changed) -> None:
        """Handle provider selection change."""
        if event.value:
            self._update_model_options(event.value)
    
    def _update_model_options(self, provider: str) -> None:
        """Update model options based on selected provider."""
        try:
            model_select = self.query_one("#model-select", Select)
            models = self.provider_models.get(provider, [])
            
            if models:
                model_select.set_options([(m, m) for m in models])
                model_select.value = models[0]
            else:
                model_select.set_options([])
                model_select.value = None
                
        except Exception as e:
            logger.debug(f"Could not update model options: {e}")
    
    @on(Button.Pressed, "#find-button")
    def handle_content_search(self, event: Button.Pressed) -> None:
        """Execute content search."""
        try:
            search_input = self.query_one("#content-search-input", Input)
            search_term = search_input.value
            
            if search_term:
                self.content_search_term = search_term
                self._search_content(search_term)
        except Exception as e:
            logger.error(f"Error searching content: {e}")
    
    @on(Button.Pressed, "#clear-search-button")
    def handle_clear_search(self, event: Button.Pressed) -> None:
        """Clear content search."""
        try:
            # Clear input
            search_input = self.query_one("#content-search-input", Input)
            search_input.value = ""
            
            # Clear search state
            self.content_search_term = ""
            self.search_results = []
            self.current_search_index = 0
            
            # Hide navigation
            search_nav = self.query_one("#search-nav", Horizontal)
            search_nav.styles.display = "none"
            
            # Clear highlights
            self._clear_highlights()
            
        except Exception as e:
            logger.debug(f"Error clearing search: {e}")
    
    @on(Button.Pressed, "#prev-match")
    def handle_prev_match(self, event: Button.Pressed) -> None:
        """Navigate to previous search match."""
        if self.search_results and self.current_search_index > 0:
            self.current_search_index -= 1
            self._highlight_current_match()
    
    @on(Button.Pressed, "#next-match")
    def handle_next_match(self, event: Button.Pressed) -> None:
        """Navigate to next search match."""
        if self.search_results and self.current_search_index < len(self.search_results) - 1:
            self.current_search_index += 1
            self._highlight_current_match()
    
    @on(Button.Pressed, "#generate-analysis-button")
    def handle_generate_analysis(self, event: Button.Pressed) -> None:
        """Generate analysis for current media."""
        if not self.current_media:
            self.app_instance.notify("No media selected", severity="warning")
            return
        
        try:
            # Get parameters
            provider = self.query_one("#provider-select", Select).value
            model = self.query_one("#model-select", Select).value
            system_prompt = self.query_one("#system-prompt-input", TextArea).text
            user_prompt = self.query_one("#user-prompt-input", TextArea).text
            
            if not provider or not model:
                self.app_instance.notify("Please select provider and model", severity="warning")
                return
            
            media_id = self.current_media.get('id')
            if not media_id:
                self.app_instance.notify("Invalid media item", severity="error")
                return
            
            # Post analysis request event
            self.post_message(MediaAnalysisRequestEvent(
                media_id=media_id,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=2000
            ))
            
            # Update UI to show generating state
            analysis_display = self.query_one("#analysis-display", Markdown)
            analysis_display.update("# Generating Analysis...\n\nPlease wait while the analysis is being generated.")
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}", exc_info=True)
            self.app_instance.notify(f"Generation failed: {str(e)[:100]}", severity="error")
    
    @on(Button.Pressed, "#prev-version")
    def handle_prev_version(self, event: Button.Pressed) -> None:
        """Navigate to previous analysis version."""
        if self.analysis_versions and self.current_version_index > 0:
            self.current_version_index -= 1
            self._display_analysis_version(self.current_version_index)
    
    @on(Button.Pressed, "#next-version")
    def handle_next_version(self, event: Button.Pressed) -> None:
        """Navigate to next analysis version."""
        if self.analysis_versions and self.current_version_index < len(self.analysis_versions) - 1:
            self.current_version_index += 1
            self._display_analysis_version(self.current_version_index)
    
    @on(Button.Pressed, "#save-analysis")
    def handle_save_analysis(self, event: Button.Pressed) -> None:
        """Save current analysis."""
        # This would trigger save event to parent
        self.app_instance.notify("Analysis saved", severity="information")
    
    def load_media(self, media_data: Dict[str, Any]) -> None:
        """Load media data into the viewer."""
        logger.info(f"Loading media content: {media_data.get('id')} - {media_data.get('title', 'Untitled')}")
        
        self.current_media = media_data
        
        # Update content tab
        self._load_content(media_data)
        
        # Update analysis tab
        self._load_analysis(media_data)
    
    def _load_content(self, media_data: Dict[str, Any]) -> None:
        """Load content into the content tab."""
        try:
            content_display = self.query_one("#content-display", Markdown)
            
            content = media_data.get('content', '')
            if content:
                # Format content as markdown
                title = media_data.get('title', 'Untitled')
                author = media_data.get('author', 'Unknown')
                
                markdown_content = f"# {title}\n\n"
                if author and author != 'Unknown':
                    markdown_content += f"*By {author}*\n\n---\n\n"
                
                # Add the actual content
                markdown_content += content
                
                content_display.update(markdown_content)
            else:
                content_display.update("# No Content Available\n\nThis media item has no content to display.")
                
        except Exception as e:
            logger.error(f"Error loading content: {e}")
    
    def _load_analysis(self, media_data: Dict[str, Any]) -> None:
        """Load analysis into the analysis tab."""
        try:
            analysis_display = self.query_one("#analysis-display", Markdown)
            
            # Check for existing analysis
            analysis = media_data.get('analysis')
            if analysis:
                self.current_analysis = analysis
                analysis_display.update(analysis)
                
                # Enable save button
                try:
                    save_btn = self.query_one("#save-analysis", Button)
                    save_btn.disabled = False
                except Exception:
                    pass
            else:
                analysis_display.update("# No Analysis\n\nNo analysis has been generated for this media item yet.")
                self.current_analysis = None
            
            # Load version history if available
            self._load_analysis_versions(media_data)
            
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
    
    def _load_analysis_versions(self, media_data: Dict[str, Any]) -> None:
        """Load analysis version history."""
        # This would load from database
        # For now, just use placeholder
        self.analysis_versions = []
        self.current_version_index = 0
        
        # Update navigation
        self._update_version_navigation()
    
    def _update_version_navigation(self) -> None:
        """Update version navigation controls."""
        try:
            prev_btn = self.query_one("#prev-version", Button)
            next_btn = self.query_one("#next-version", Button)
            version_info = self.query_one("#version-info", Static)
            
            if self.analysis_versions:
                total = len(self.analysis_versions)
                current = self.current_version_index + 1
                version_info.update(f"Version {current} / {total}")
                
                prev_btn.disabled = self.current_version_index <= 0
                next_btn.disabled = self.current_version_index >= total - 1
            else:
                version_info.update("No versions")
                prev_btn.disabled = True
                next_btn.disabled = True
                
        except Exception as e:
            logger.debug(f"Could not update version navigation: {e}")
    
    def _display_analysis_version(self, index: int) -> None:
        """Display a specific analysis version."""
        if 0 <= index < len(self.analysis_versions):
            version = self.analysis_versions[index]
            analysis_display = self.query_one("#analysis-display", Markdown)
            analysis_display.update(version.get('content', 'No content'))
            
            self._update_version_navigation()
    
    def _search_content(self, search_term: str) -> None:
        """Search for term in content."""
        try:
            content_display = self.query_one("#content-display", Markdown)
            content_text = content_display.markdown
            
            # Find all matches (case-insensitive)
            import re
            pattern = re.compile(re.escape(search_term), re.IGNORECASE)
            self.search_results = [(m.start(), m.end()) for m in pattern.finditer(content_text)]
            
            if self.search_results:
                self.current_search_index = 0
                self._highlight_current_match()
                
                # Show navigation
                search_nav = self.query_one("#search-nav", Horizontal)
                search_nav.styles.display = "block"
                
                # Update status
                self._update_search_status()
            else:
                # No matches
                search_status = self.query_one("#search-status", Static)
                search_status.update("No matches")
                
                # Hide navigation
                search_nav = self.query_one("#search-nav", Horizontal)
                search_nav.styles.display = "none"
                
        except Exception as e:
            logger.error(f"Error searching content: {e}")
    
    def _highlight_current_match(self) -> None:
        """Highlight the current search match."""
        # This is simplified - in real implementation would highlight in markdown
        self._update_search_status()
        
        # Update navigation buttons
        try:
            prev_btn = self.query_one("#prev-match", Button)
            next_btn = self.query_one("#next-match", Button)
            
            prev_btn.disabled = self.current_search_index <= 0
            next_btn.disabled = self.current_search_index >= len(self.search_results) - 1
        except Exception:
            pass
    
    def _update_search_status(self) -> None:
        """Update search status display."""
        try:
            search_status = self.query_one("#search-status", Static)
            if self.search_results:
                current = self.current_search_index + 1
                total = len(self.search_results)
                search_status.update(f"Match {current} of {total}")
            else:
                search_status.update("No matches")
        except Exception:
            pass
    
    def _clear_highlights(self) -> None:
        """Clear search highlights from content."""
        # This would clear highlighting in the markdown display
        pass
    
    def clear_display(self) -> None:
        """Clear all displays."""
        self.current_media = None
        self.current_analysis = None
        self.analysis_versions = []
        self.search_results = []
        
        try:
            # Clear content
            content_display = self.query_one("#content-display", Markdown)
            content_display.update("# No Content\n\nSelect a media item to view its content.")
            
            # Clear analysis
            analysis_display = self.query_one("#analysis-display", Markdown)
            analysis_display.update("# No Analysis\n\nGenerate or select an analysis to view.")
            
            # Reset controls
            self.handle_clear_search(None)
            
        except Exception as e:
            logger.debug(f"Could not clear display: {e}")