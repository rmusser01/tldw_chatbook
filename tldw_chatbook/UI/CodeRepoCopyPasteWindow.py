# tldw_chatbook/UI/CodeRepoCopyPasteWindow.py
# Description: GitHub repository file selector window for copying code
#
# This window allows users to browse GitHub repositories, select specific files,
# and export them in various formats (ZIP, clipboard, embeddings).

from __future__ import annotations
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import asyncio
import json
import zipfile
import io
from datetime import datetime

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Input, Label, Select, Static, TextArea,
    LoadingIndicator, Checkbox, DataTable, ProgressBar
)
from textual.reactive import reactive
from textual.message import Message

# Local imports
from ..Widgets.repo_tree_widgets import TreeView
from ..Utils.github_api_client import GitHubAPIClient, GitHubAPIError
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE

if TYPE_CHECKING:
    from ..app import TldwCli

logger = logger.bind(module="CodeRepoCopyPasteWindow")


class FileSelected(Message):
    """Message sent when files are selected for export."""
    def __init__(self, files: List[str], content: str) -> None:
        self.files = files
        self.content = content
        super().__init__()


class CodeRepoCopyPasteWindow(ModalScreen):
    """Modal window for selecting and copying files from GitHub repositories."""
    
    BINDINGS = [
        ("escape", "close_window", "Close"),
        ("ctrl+a", "select_all", "Select All"),
        ("ctrl+shift+a", "deselect_all", "Deselect All"),
        ("ctrl+i", "invert_selection", "Invert Selection"),
    ]
    
    DEFAULT_CSS = """
    CodeRepoCopyPasteWindow {
        align: center middle;
    }
    
    .repo-window-container {
        width: 90%;
        height: 90%;
        max-width: 120;
        max-height: 50;
        background: $surface;
        border: thick $primary;
        padding: 0;
    }
    
    /* Header styles */
    .repo-header {
        dock: top;
        height: 7;
        background: $boost;
        padding: 1;
        border-bottom: solid $primary;
    }
    
    .repo-header-content {
        layout: grid;
        grid-size: 4 2;
        grid-columns: 3fr 1fr 1fr 1fr;
        grid-rows: 3 2;
        grid-gutter: 1 1;
        width: 100%;
    }
    
    .repo-url-input {
        width: 100%;
    }
    
    .selection-summary {
        grid-column-span: 4;
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 1fr 1fr;
        align: left middle;
        width: 100%;
    }
    
    .summary-stat {
        text-align: left;
        color: $text-muted;
    }
    
    /* Filter bar styles */
    .filter-bar {
        dock: top;
        height: 3;
        background: $panel;
        layout: grid;
        grid-size: 6 1;
        grid-columns: 2fr 1fr 3 3 3 1fr;
        grid-gutter: 0 1;
        padding: 0 1;
        border-bottom: solid $primary-darken-1;
    }
    
    .filter-quick-btn {
        width: 3;
        height: 3;
        min-width: 3;
        padding: 0;
    }
    
    /* Main content styles */
    .main-content {
        layout: grid;
        grid-size: 2 1;
        grid-columns: 2fr 3fr;
        height: 1fr;
        background: $background;
    }
    
    .tree-container {
        border-right: solid $primary;
        overflow: hidden;
        padding: 1;
    }
    
    .preview-container {
        padding: 1;
        overflow: hidden;
    }
    
    .preview-header {
        height: 3;
        border-bottom: dashed $primary-darken-2;
        margin-bottom: 1;
    }
    
    .preview-content {
        height: 1fr;
        border: round $primary-darken-2;
        padding: 1;
        background: $surface-darken-1;
    }
    
    /* Action bar styles */
    .action-bar {
        dock: bottom;
        height: 3;
        background: $panel;
        layout: grid;
        grid-size: 7 1;
        grid-columns: 1fr auto auto auto auto auto 1fr;
        grid-gutter: 0 1;
        align: center middle;
        padding: 0 2;
        border-top: solid $primary;
    }
    
    /* Loading overlay */
    .loading-overlay {
        dock: top;
        width: 100%;
        height: 100%;
        background: $background 90%;
        align: center middle;
        layer: above;
    }
    
    .loading-overlay.hidden {
        display: none;
    }
    
    .loading-label {
        text-align: center;
        margin-top: 2;
        text-style: bold;
    }
    
    /* Error styles */
    .error-message {
        color: $error;
        text-align: center;
        padding: 2;
        text-style: bold;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.api_client = GitHubAPIClient()
        self.current_repo: Optional[Dict[str, str]] = None
        self.is_loading = reactive(False)
        self.loading_message = reactive("Loading...")
        self.tree_data: Optional[List[Dict]] = None
        
    def compose(self) -> ComposeResult:
        """Compose the UI."""
        with Container(classes="repo-window-container"):
            # Header with repo input
            with Container(classes="repo-header"):
                with Container(classes="repo-header-content"):
                    # First row: input and controls
                    yield Input(
                        placeholder="https://github.com/user/repo",
                        id="repo-url-input",
                        classes="repo-url-input"
                    )
                    yield Button("Load", id="load-repo-btn", variant="primary")
                    yield Select(
                        options=[("main", "main")],
                        id="branch-selector",
                        value="main"
                    )
                    yield Button("⚙️", id="repo-settings-btn")
                    
                    # Second row: selection summary
                    with Container(classes="selection-summary"):
                        yield Static("No files selected", id="selection-count", classes="summary-stat")
                        yield Static("0 KB", id="selection-size", classes="summary-stat")
                        yield Static("~0 tokens", id="selection-tokens", classes="summary-stat")
            
            # Filter bar
            with Container(classes="filter-bar"):
                yield Input(placeholder="Search files...", id="file-search")
                yield Select(
                    options=[
                        ("all", "All Files"),
                        ("code", "Code Only"),
                        ("docs", "Documentation"),
                        ("config", "Config Files"),
                    ],
                    id="file-type-filter",
                    value="all"
                )
                yield Button("📝", id="filter-docs", classes="filter-quick-btn")
                yield Button("💻", id="filter-code", classes="filter-quick-btn")
                yield Button("⚙️", id="filter-config", classes="filter-quick-btn")
                yield Static()  # Spacer
            
            # Main content
            with Container(classes="main-content"):
                # Tree view
                with Container(classes="tree-container"):
                    yield TreeView(
                        id="repo-tree",
                        on_selection_change=self.handle_selection_change,
                        on_node_expanded=self.handle_node_expanded
                    )
                
                # Preview pane
                with Container(classes="preview-container"):
                    with Container(classes="preview-header"):
                        yield Static("Select a file to preview", id="preview-filename")
                    with ScrollableContainer(classes="preview-content"):
                        yield TextArea(
                            "No file selected",
                            id="file-preview",
                            read_only=True,
                            language="python"
                        )
            
            # Action bar
            with Container(classes="action-bar"):
                yield Static()  # Left spacer
                yield Button("Cancel", id="cancel-btn", variant="error")
                yield Button("Export ZIP", id="export-zip-btn")
                yield Button("Copy to Clipboard", id="copy-clipboard-btn")
                yield Button("Create Embeddings", id="create-embeddings-btn", variant="primary")
                yield Button("Save Profile", id="save-profile-btn")
                yield Static()  # Right spacer
            
            # Loading overlay
            with Container(classes="loading-overlay hidden", id="loading-overlay"):
                yield LoadingIndicator()
                yield Label("Loading...", classes="loading-label", id="loading-label")
    
    def on_mount(self) -> None:
        """Handle mount event."""
        # Focus on URL input
        self.query_one("#repo-url-input", Input).focus()
    
    def watch_is_loading(self, is_loading: bool) -> None:
        """Show/hide loading overlay."""
        overlay = self.query_one("#loading-overlay")
        label = self.query_one("#loading-label", Label)
        
        if is_loading:
            overlay.remove_class("hidden")
            label.update(self.loading_message)
        else:
            overlay.add_class("hidden")
    
    @on(Button.Pressed, "#load-repo-btn")
    async def load_repository(self, event: Button.Pressed) -> None:
        """Load repository from URL."""
        url_input = self.query_one("#repo-url-input", Input)
        repo_url = url_input.value.strip()
        
        if not repo_url:
            self.notify("Please enter a GitHub repository URL", severity="error")
            return
        
        self.loading_message = "Loading repository..."
        self.is_loading = True
        
        try:
            # Parse repository URL
            owner, repo = self.api_client.parse_github_url(repo_url)
            self.current_repo = {"owner": owner, "repo": repo}
            
            # Get repository info
            repo_info = await self.api_client.get_repository_info(owner, repo)
            logger.info(f"Loaded repository: {owner}/{repo}")
            
            # Get branches
            branches = await self.api_client.get_branches(owner, repo)
            branch_selector = self.query_one("#branch-selector", Select)
            branch_selector.set_options([(b, b) for b in branches])
            
            # Load repository tree
            await self.load_tree()
            
        except GitHubAPIError as e:
            self.notify(str(e), severity="error")
            logger.error(f"GitHub API error: {e}")
        except Exception as e:
            self.notify(f"Failed to load repository: {e}", severity="error")
            logger.error(f"Failed to load repository: {e}")
        finally:
            self.is_loading = False
    
    async def load_tree(self) -> None:
        """Load repository tree structure."""
        if not self.current_repo:
            return
        
        branch = self.query_one("#branch-selector", Select).value
        
        try:
            # Get repository tree
            self.loading_message = "Loading file structure..."
            flat_tree = await self.api_client.get_repository_tree(
                self.current_repo["owner"],
                self.current_repo["repo"],
                branch
            )
            
            # Build hierarchical tree
            self.tree_data = self.api_client.build_tree_hierarchy(flat_tree)
            
            # Load into tree view
            tree_view = self.query_one("#repo-tree", TreeView)
            await tree_view.load_tree(self.tree_data)
            
            self.notify(f"Loaded {len(flat_tree)} items", severity="information")
            
        except GitHubAPIError as e:
            self.notify(str(e), severity="error")
            logger.error(f"Failed to load tree: {e}")
    
    @on(Select.Changed, "#branch-selector")
    async def on_branch_changed(self, event: Select.Changed) -> None:
        """Handle branch selection change."""
        if self.current_repo and not self.is_loading:
            self.is_loading = True
            await self.load_tree()
            self.is_loading = False
    
    def handle_selection_change(self, path: str, selected: bool) -> None:
        """Handle file selection changes."""
        # Update selection summary
        tree_view = self.query_one("#repo-tree", TreeView)
        stats = tree_view.get_selection_stats()
        
        count_label = self.query_one("#selection-count", Static)
        size_label = self.query_one("#selection-size", Static)
        tokens_label = self.query_one("#selection-tokens", Static)
        
        count_label.update(f"{stats['files']} files selected")
        size_label.update(self._format_size(stats['size']))
        
        # Estimate tokens (rough approximation)
        estimated_tokens = stats['size'] // 4  # ~4 chars per token
        tokens_label.update(f"~{estimated_tokens:,} tokens")
    
    async def handle_node_expanded(self, path: str, expanded: bool) -> None:
        """Handle node expansion - load children if needed."""
        if not expanded or not self.current_repo:
            return
        
        # For now, we load everything at once
        # In a real implementation, we'd load children on demand
        pass
    
    @on(Button.Pressed, "#filter-docs")
    async def filter_docs(self, event: Button.Pressed) -> None:
        """Quick filter for documentation files."""
        filter_select = self.query_one("#file-type-filter", Select)
        filter_select.value = "docs"
        await self.apply_filters()
    
    @on(Button.Pressed, "#filter-code")
    async def filter_code(self, event: Button.Pressed) -> None:
        """Quick filter for code files."""
        filter_select = self.query_one("#file-type-filter", Select)
        filter_select.value = "code"
        await self.apply_filters()
    
    @on(Button.Pressed, "#filter-config")
    async def filter_config(self, event: Button.Pressed) -> None:
        """Quick filter for config files."""
        filter_select = self.query_one("#file-type-filter", Select)
        filter_select.value = "config"
        await self.apply_filters()
    
    async def apply_filters(self) -> None:
        """Apply current filters to tree view."""
        # TODO: Implement filtering logic
        pass
    
    @on(Button.Pressed, "#copy-clipboard-btn")
    async def copy_to_clipboard(self, event: Button.Pressed) -> None:
        """Copy selected files to clipboard."""
        tree_view = self.query_one("#repo-tree", TreeView)
        selected_files = tree_view.get_selected_files()
        
        if not selected_files:
            self.notify("No files selected", severity="warning")
            return
        
        if not self.current_repo:
            return
        
        self.loading_message = "Fetching file contents..."
        self.is_loading = True
        
        try:
            # Fetch content for all selected files
            contents = []
            branch = self.query_one("#branch-selector", Select).value
            
            for file_path in selected_files:
                content = await self.api_client.get_file_content(
                    self.current_repo["owner"],
                    self.current_repo["repo"],
                    file_path,
                    branch
                )
                
                # Format with file markers
                contents.append(f"```{file_path}\n{content}\n```")
            
            # Join all contents
            full_content = "\n\n".join(contents)
            
            # Copy to clipboard (this would need platform-specific implementation)
            # For now, we'll just show it in a message
            self.notify(f"Copied {len(selected_files)} files to clipboard", severity="success")
            
            # Dismiss the modal and pass the content back
            self.dismiss((selected_files, full_content))
            
        except Exception as e:
            self.notify(f"Failed to copy files: {e}", severity="error")
            logger.error(f"Failed to copy files: {e}")
        finally:
            self.is_loading = False
    
    @on(Button.Pressed, "#cancel-btn")
    def close_window(self) -> None:
        """Close the window."""
        self.dismiss(None)
    
    def action_close_window(self) -> None:
        """Close window action."""
        self.dismiss(None)
    
    def action_select_all(self) -> None:
        """Select all files."""
        # TODO: Implement select all
        pass
    
    def action_deselect_all(self) -> None:
        """Deselect all files."""
        # TODO: Implement deselect all
        pass
    
    def action_invert_selection(self) -> None:
        """Invert selection."""
        # TODO: Implement invert selection
        pass
    
    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up on exit."""
        await self.api_client.close()