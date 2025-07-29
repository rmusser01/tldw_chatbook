# tldw_chatbook/Widgets/HuggingFace/model_card_viewer.py
"""
Model card viewer for displaying HuggingFace model details and available files.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import Label, Button, Markdown, Static, LoadingIndicator, TabbedContent, TabPane, ListView, ListItem
from textual.message import Message
from textual.reactive import reactive
from textual import work
from loguru import logger


class DownloadRequestEvent(Message):
    """Event fired when user requests to download files."""
    
    def __init__(self, repo_id: str, files: List[Dict[str, Any]]) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.files = files


class ModelCardViewer(Container):
    """Widget for viewing model details and selecting files to download."""
    
    DEFAULT_CSS = """
    ModelCardViewer {
        height: 1fr;
        layout: vertical;
        background: $surface;
        border: solid $primary;
    }
    
    ModelCardViewer .header {
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: solid $primary;
    }
    
    ModelCardViewer .model-title {
        text-style: bold;
        color: $primary;
    }
    
    ModelCardViewer .model-meta {
        color: $text-muted;
        margin-top: 0;
    }
    
    ModelCardViewer .content-area {
        height: 1fr;
        overflow: hidden;
    }
    
    ModelCardViewer TabbedContent {
        height: 1fr;
    }
    
    ModelCardViewer TabPane {
        padding: 0;
        height: 1fr;
    }
    
    ModelCardViewer .tab-content {
        height: 1fr;
        padding: 1;
    }
    
    ModelCardViewer #files-list {
        height: 1fr;
        overflow-y: auto;
        border: none;
        background: $background;
        padding: 0;
    }
    
    ModelCardViewer .file-item {
        padding: 1;
        margin: 0;
        background: $surface;
        border-bottom: solid $primary-background-darken-1;
        border-left: thick transparent;
    }
    
    ModelCardViewer .file-item:hover {
        background: $surface-lighten-1;
    }
    
    ModelCardViewer .file-item.selected {
        background: $primary 10%;
        border-left: thick $primary;
    }
    
    ModelCardViewer .file-item.selected:hover {
        background: $primary 20%;
    }
    
    ModelCardViewer .file-item-content {
        width: 100%;
        padding: 0;
    }
    
    ModelCardViewer .readme-content {
        height: auto;
        padding: 1;
        background: $background;
    }
    
    ModelCardViewer #readme-display {
        height: auto;
        padding: 1;
    }
    
    ModelCardViewer .model-card-container {
        height: auto;
        padding: 1;
    }
    
    ModelCardViewer .model-detail-row {
        layout: horizontal;
        margin-bottom: 1;
        padding: 0 1;
    }
    
    ModelCardViewer .model-detail-label {
        text-style: bold;
        width: 10;
        color: $secondary;
    }
    
    ModelCardViewer .model-detail-value {
        width: 1fr;
        color: $text;
    }
    
    ModelCardViewer .action-bar {
        height: auto;
        padding: 1;
        background: $boost;
        border-top: solid $primary;
        layout: horizontal;
        align: center middle;
    }
    
    ModelCardViewer .download-info {
        width: 1fr;
        color: $text-muted;
    }
    
    ModelCardViewer .loading {
        text-align: center;
        padding: 4;
        color: $text-muted;
    }
    
    ModelCardViewer .error {
        text-align: center;
        padding: 4;
        color: $error;
    }
    
    ModelCardViewer .placeholder {
        text-align: center;
        padding: 4;
        color: $text-muted;
    }
    """
    
    # Reactive properties
    model_info: reactive[Optional[Dict[str, Any]]] = reactive(None)
    model_files: reactive[List[Dict[str, Any]]] = reactive([])
    readme_content: reactive[Optional[str]] = reactive(None)
    selected_file: reactive[Optional[str]] = reactive(None)
    is_loading: reactive[bool] = reactive(False)
    error_message: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._files_initialized = False
    
            
    def compose(self) -> ComposeResult:
        """Compose the model card viewer UI."""
        # Header
        with Container(classes="header"):
            yield Label("", id="model-title", classes="model-title")
            yield Static("", id="model-meta", classes="model-meta")
        
        # Content area with tabs
        with Container(classes="content-area"):
            with TabbedContent(id="model-tabs"):
                # Available Files tab
                with TabPane("Available Files", id="files-tab"):
                    yield ListView(id="files-list", classes="tab-content")
                
                # README tab
                with TabPane("README", id="readme-tab"):
                    with VerticalScroll(classes="tab-content"):
                        yield Markdown("", id="readme-display", classes="readme-content")
                
                # Model Card tab
                with TabPane("Model Card", id="model-card-tab"):
                    with VerticalScroll(classes="tab-content"):
                        yield Container(id="model-card-content", classes="model-card-container")
        
        # Action bar
        with Container(classes="action-bar"):
            yield Static("", id="download-info", classes="download-info")
            yield Button(
                "Download Selected",
                id="download-button",
                variant="primary",
                disabled=True
            )
            yield Button(
                "View on HuggingFace",
                id="view-button",
                variant="default"
            )
    
    def watch_model_info(self, model_info: Optional[Dict[str, Any]]) -> None:
        """Update display when model info changes."""
        if not model_info:
            self.clear_display()
            return
        
        # Update header
        title = self.query_one("#model-title", Label)
        title.update(model_info.get("id", "Unknown Model"))
        
        meta = self.query_one("#model-meta", Static)
        author = model_info.get("author", "Unknown")
        downloads = model_info.get("downloads", 0)
        likes = model_info.get("likes", 0)
        meta.update(f"by {author} • Downloads: {downloads:,} • Likes: {likes:,}")
        
        # Load model details
        repo_id = model_info.get("id")
        if repo_id:
            self.load_model_details(repo_id)
        
        # Populate Model Card tab after a small delay to ensure mounting
        from textual import work
        self.call_after_refresh(self._populate_model_card, model_info)
    
    def watch_model_files(self, files: List[Dict[str, Any]]) -> None:
        """Update files list when model files change."""
        logger.info(f"watch_model_files called with {len(files)} files")
        # Schedule async update
        self.call_later(self._update_files_list, files)
    
    def watch_readme_content(self, content: Optional[str]) -> None:
        """Update README display."""
        try:
            readme_display = self.query_one("#readme-display", Markdown)
            if content:
                # Ensure content is a string and not too large
                content_str = str(content)[:100000]  # Limit to 100k chars
                readme_display.update(content_str)
            else:
                readme_display.update("*No README available*")
        except Exception as e:
            logger.error(f"Error updating README display: {e}")
    
    def watch_selected_file(self, selected: Optional[str]) -> None:
        """Update UI when file selection changes."""
        # Update download button state
        download_btn = self.query_one("#download-button", Button)
        download_btn.disabled = selected is None
        
        # Update download info
        info = self.query_one("#download-info", Static)
        if selected:
            # Find the selected file's size
            total_size = 0
            for file in self.model_files:
                if file.get("path") == selected:
                    total_size = file.get("size", 0)
                    break
            
            size_str = self._format_bytes(total_size)
            info.update(f"1 file selected • Size: {size_str}")
        else:
            info.update("No file selected")
    
    def watch_is_loading(self, is_loading: bool) -> None:
        """Show loading state."""
        if is_loading:
            # Schedule async loading message
            self.call_later(self._show_loading_message)
            
            # Also show loading in README tab
            try:
                readme_display = self.query_one("#readme-display", Markdown)
                readme_display.update("*Loading README...*")
            except Exception as e:
                logger.debug(f"Could not update README loading state: {e}")
    
    async def _show_loading_message(self) -> None:
        """Show loading message in the ListView."""
        try:
            files_list = self.query_one("#files-list", ListView)
            await files_list.clear()
            await files_list.append(
                ListItem(Static("Loading model files...", classes="loading"))
            )
        except Exception as e:
            logger.debug(f"Could not show loading message: {e}")
    
    def watch_error_message(self, error: Optional[str]) -> None:
        """Display error message."""
        if error:
            self.call_later(self._show_error_message, error)
    
    async def _show_error_message(self, error: str) -> None:
        """Show error message in the ListView."""
        try:
            files_list = self.query_one("#files-list", ListView)
            await files_list.clear()
            await files_list.append(
                ListItem(Static(f"Error: {error}", classes="error"))
            )
        except Exception as e:
            logger.debug(f"Could not show error message: {e}")
    
    def load_model_details(self, repo_id: str) -> None:
        """Trigger loading of model files and README in background."""
        # Use run_worker with an async coroutine
        async def _load_details():
            self.is_loading = True
            self.error_message = None
            self.selected_file = None
            
            try:
                # Import here to avoid circular imports
                from ...LLM_Calls.huggingface_api import HuggingFaceAPI
                
                api = HuggingFaceAPI()
                
                # Load files
                logger.info(f"Loading files for repo: {repo_id}")
                files = await api.list_model_files(repo_id)
                logger.info(f"Received {len(files)} files from API")
                self.model_files = files
                
                # Load README
                logger.info(f"Loading README for repo: {repo_id}")
                readme = await api.get_model_readme(repo_id)
                self.readme_content = readme
                
            except Exception as e:
                import traceback
                logger.error(f"Error loading model details: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.error_message = str(e)
            finally:
                self.is_loading = False
        
        # Run the worker
        self.run_worker(_load_details(), exclusive=True)
    
    def clear_display(self) -> None:
        """Clear all displayed content."""
        try:
            self.query_one("#model-title", Label).update("No model selected")
            self.query_one("#model-meta", Static).update("")
            # Schedule async clear for ListView
            self.call_later(self._clear_files_list)
            self.query_one("#readme-display", Markdown).update("")
            self.query_one("#model-card-content", Container).remove_children()
            self.query_one("#download-info", Static).update("")
            self.query_one("#download-button", Button).disabled = True
            self.selected_file = None
        except Exception as e:
            logger.debug(f"Error clearing display: {e}")
    
    async def _clear_files_list(self) -> None:
        """Clear the files ListView."""
        try:
            files_list = self.query_one("#files-list", ListView)
            await files_list.clear()
        except Exception as e:
            logger.debug(f"Could not clear files list: {e}")
    
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle file selection from ListView."""
        if hasattr(event.item, "file_path"):
            file_path = event.item.file_path
            files_list = self.query_one("#files-list", ListView)
            
            # Store previous selection
            previous_selected = self.selected_file
            
            # For single selection, just set the selected file
            # If clicking the same file, deselect it
            if self.selected_file == file_path:
                self.selected_file = None
            else:
                self.selected_file = file_path
            
            # Update only the affected items instead of refreshing entire list
            for item in files_list.children:
                if isinstance(item, ListItem) and hasattr(item, "file_path"):
                    # Update previously selected item
                    if item.file_path == previous_selected:
                        item.remove_class("selected")
                        static_widget = item.query_one(Static)
                        if hasattr(item, "file_data"):
                            file_name = Path(item.file_path).name
                            file_size = item.file_data.get("size_human", "Unknown size")
                            static_widget.update(f"○ [bold]{file_name}[/bold] [dim]{file_size}[/dim]")
                    
                    # Update newly selected item
                    if item.file_path == self.selected_file:
                        item.add_class("selected")
                        static_widget = item.query_one(Static)
                        if hasattr(item, "file_data"):
                            file_name = Path(item.file_path).name
                            file_size = item.file_data.get("size_human", "Unknown size")
                            static_widget.update(f"◉ [bold]{file_name}[/bold] [dim]{file_size}[/dim]")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "download-button":
            if self.model_info and self.selected_file:
                # Get selected file info
                selected_file_info = [
                    f for f in self.model_files 
                    if f.get("path") == self.selected_file
                ]
                self.post_message(
                    DownloadRequestEvent(
                        self.model_info.get("id", ""),
                        selected_file_info
                    )
                )
        
        elif event.button.id == "view-button":
            if self.model_info:
                # Open in browser (would need to implement platform-specific logic)
                repo_id = self.model_info.get("id", "")
                url = f"https://huggingface.co/{repo_id}"
                logger.info(f"View model at: {url}")
                # TODO: Implement browser opening
    
    async def _update_files_list(self, files: List[Dict[str, Any]]) -> None:
        """Update the ListView with file items."""
        logger.info(f"_update_files_list called with {len(files)} files")
        
        try:
            files_list = self.query_one("#files-list", ListView)
            await files_list.clear()
            
            if not files:
                logger.warning("No files to display")
                await files_list.append(
                    ListItem(Static("No GGUF files found", classes="placeholder"))
                )
                return
            
            # Create ListItem for each file - using a single Static widget
            for i, file in enumerate(files):
                file_path = file.get("path", "")
                file_size = file.get("size_human", "Unknown size")
                file_name = Path(file_path).name
                
                logger.debug(f"Adding file {i}: {file_path} ({file_size})")
                
                # Check if file is selected and use appropriate symbol (radio button for single selection)
                is_selected = file_path == self.selected_file
                radio_symbol = "◉" if is_selected else "○"
                
                # Create a formatted string with radio button symbol
                content = f"{radio_symbol} [bold]{file_name}[/bold] [dim]{file_size}[/dim]"
                
                # Create ListItem with a single Static widget
                item = ListItem(
                    Static(content, classes="file-item-content"),
                    classes="file-item selected" if is_selected else "file-item"
                )
                
                # Store file data on the item for selection tracking
                item.file_data = file
                item.file_path = file_path
                
                await files_list.append(item)
            
            logger.info(f"Successfully added {len(files)} files to the UI")
            
        except Exception as e:
            logger.error(f"Error updating files list: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _populate_model_card(self, model_info: Dict[str, Any]) -> None:
        """Populate the Model Card tab with formatted model information."""
        try:
            # Check if widget is mounted
            if not self.is_mounted:
                logger.warning("Widget not mounted yet, deferring model card update")
                self.call_after_refresh(self._populate_model_card, model_info)
                return
            
            try:
                model_card_container = self.query_one("#model-card-content", Container)
            except Exception:
                logger.warning("Model card container not found, deferring update")
                self.call_after_refresh(self._populate_model_card, model_info)
                return
                
            # Check if the container itself is mounted
            if not model_card_container.is_mounted:
                logger.warning("Model card container not mounted yet, deferring update")
                self.call_after_refresh(self._populate_model_card, model_info)
                return
                
            model_card_container.remove_children()
            
            # Model details
            details = [
                ("Model ID", model_info.get("id", "Unknown")),
                ("Author", model_info.get("author", "Unknown")),
                ("Downloads", f"{model_info.get('downloads', 0):,}"),
                ("Likes", f"{model_info.get('likes', 0):,}"),
                ("Last Modified", model_info.get("lastModified", "Unknown")),
                ("Created", model_info.get("createdAt", "Unknown")),
                ("Pipeline Tag", model_info.get("pipeline_tag", "N/A")),
                ("Library", model_info.get("library_name", "N/A")),
            ]
            
            # Add tags if available
            tags = model_info.get("tags", [])
            if tags:
                tags_str = ", ".join(tags[:10])  # Limit to first 10 tags
                if len(tags) > 10:
                    tags_str += f" (+{len(tags) - 10} more)"
                details.append(("Tags", tags_str))
            
            # Create detail rows
            for label, value in details:
                row = Container(classes="model-detail-row")
                row.mount(Static(f"{label}:", classes="model-detail-label"))
                row.mount(Static(str(value), classes="model-detail-value"))
                model_card_container.mount(row)
                
        except Exception as e:
            logger.error(f"Error populating model card: {e}")
            model_card_container = self.query_one("#model-card-content", Container)
            model_card_container.remove_children()
            model_card_container.mount(Static("Error loading model card information", classes="error"))
    
    @staticmethod
    def _format_bytes(bytes: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.1f} PB"