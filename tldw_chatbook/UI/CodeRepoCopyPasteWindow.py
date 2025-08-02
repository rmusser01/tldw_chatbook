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
import os
import subprocess
from pathlib import Path
from datetime import datetime
import aiofiles

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Input, Label, Select, Static, TextArea,
    LoadingIndicator, Checkbox, DataTable, ProgressBar
)
from textual.screen import Screen
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
    
    # CSS is handled by external file: css/features/_code_repo.tcss
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.api_client = GitHubAPIClient()
        self.current_repo: Optional[Dict[str, str]] = None
        self.is_loading = reactive(False)
        self.loading_message = reactive("Loading...")
        self.tree_data: Optional[List[Dict]] = None
        self.has_loaded_repo = False
        self.selected_files: set = set()
        self.compiled_text = reactive("")  # Store the compiled text
        self.is_local_repo = False  # Track if it's a local repo
        self._token_configured = bool(self.api_client.token)
        
    def compose(self) -> ComposeResult:
        """Compose the UI."""
        logger.info("CodeRepoCopyPasteWindow.compose() called - NEW VERSION WITH COMPILATION")
        with Container(classes="repo-window-container"):
            # Header
            with Container(classes="repo-header"):
                yield Static("📦 Code Repository Browser", classes="repo-title")
                
                # Repository input (hidden when showing empty state)
                with Container(id="repo-input-container", classes="repo-input-container hidden"):
                    yield Input(
                        placeholder="Enter GitHub URL or local Git repo path",
                        id="repo-url-input",
                        classes="repo-url-input"
                    )
                    yield Button("Load Repository", id="load-repo-btn", variant="primary", classes="load-button")
                    yield Button("🔑 Token", id="token-config-btn", classes="token-button")
                
                # Controls (hidden until repo loaded)
                with Container(id="repo-controls-container", classes="repo-controls hidden"):
                    yield Select(
                        options=[("main", "main")],
                        id="branch-selector",
                        value="main",
                        classes="branch-selector"
                    )
                    with Container(classes="selection-summary"):
                        yield Static("0 files selected", id="selection-count", classes="summary-stat")
                        yield Static("0 KB", id="selection-size", classes="summary-stat")
                        yield Static("~0 tokens", id="selection-tokens", classes="summary-stat")
            
            # Empty state (shown by default)
            with Container(id="empty-state", classes="empty-state-container"):
                with Container(classes="empty-state-content"):
                    yield Static(
                        "🗂️\n\nNo Repository Loaded",
                        classes="empty-state-icon"
                    )
                    yield Static(
                        "Load a GitHub Repository",
                        classes="empty-state-title"
                    )
                    yield Static(
                        "Enter a GitHub repository URL above to browse and select files.\n"
                        "You can then export selected files to ZIP, copy to clipboard,\n"
                        "or create embeddings for RAG applications.",
                        classes="empty-state-description"
                    )
                    
                    with Container(classes="empty-state-actions"):
                        yield Input(
                            placeholder="https://github.com/user/repo",
                            id="empty-state-input",
                            classes="empty-state-input"
                        )
                        yield Button(
                            "Load Repository",
                            id="empty-load-btn",
                            variant="primary",
                            classes="empty-state-button"
                        )
            
            # Filter bar (hidden until repo loaded)
            with Container(id="filter-bar", classes="filter-bar hidden"):
                yield Input(
                    placeholder="Search files...",
                    id="file-search",
                    classes="file-search"
                )
                yield Select(
                    options=[
                        ("All Files", "all"),
                        ("Code Only", "code"),
                        ("Documentation", "docs"),
                        ("Config Files", "config"),
                    ],
                    id="file-type-filter",
                    value="all",
                    classes="file-type-filter"
                )
                with Container(classes="filter-buttons"):
                    yield Button("📝 Docs", id="filter-docs", classes="filter-quick-btn")
                    yield Button("💻 Code", id="filter-code", classes="filter-quick-btn")
                    yield Button("⚙️ Config", id="filter-config", classes="filter-quick-btn")
            
            # Main content (hidden until repo loaded)
            with Container(id="main-content", classes="main-content hidden"):
                with Container(classes="content-split"):
                    # Tree panel
                    with Container(classes="tree-panel"):
                        with Container(classes="tree-header"):
                            yield Static("Repository Files", classes="tree-title")
                        with Container(classes="tree-container", id="tree-container"):
                            yield TreeView(
                                id="repo-tree",
                                on_selection_change=self.handle_selection_change,
                                on_node_expanded=self.handle_node_expanded,
                                on_node_selected=self.handle_node_selected
                            )
                    
                    # Right panel with aggregated text and preview
                    with Container(classes="preview-panel"):
                        # Aggregated text display (top half)
                        with Container(classes="aggregated-text-container"):
                            with Container(classes="preview-header"):
                                yield Static("Generated Compilation", id="compilation-title", classes="preview-title")
                            with ScrollableContainer(classes="preview-content"):
                                yield TextArea(
                                    "Click 'Generate Compilation' to aggregate selected files",
                                    id="aggregated-text",
                                    read_only=True,
                                    language="markdown"
                                )
                        
                        # File preview (bottom half)
                        with Container(classes="file-preview-container"):
                            with Container(classes="preview-header"):
                                yield Static("File Preview", id="preview-title", classes="preview-title")
                            with ScrollableContainer(classes="preview-content"):
                                yield TextArea(
                                    "Select a file to preview its contents",
                                    id="file-preview",
                                    read_only=True,
                                    language="python"
                                )
            
            # Action bar
            with Container(id="action-bar", classes="action-bar"):
                with Container(classes="action-buttons"):
                    yield Static(classes="action-spacer")
                    yield Button("Reset", id="reset-btn", classes="action-button")
                    yield Button("Export ZIP", id="export-zip-btn", classes="action-button")
                    yield Button("Copy to Clipboard", id="copy-clipboard-btn", classes="action-button")
                    yield Button("Generate Compilation", id="generate-compilation-btn", variant="primary", classes="action-button")
                    yield Static(classes="action-spacer")
            
            # Loading overlay
            with Container(classes="loading-overlay hidden", id="loading-overlay"):
                with Container(classes="loading-content"):
                    yield LoadingIndicator()
                    yield Label("Loading repository...", classes="loading-label", id="loading-label")
    
    def on_mount(self) -> None:
        """Handle mount event."""
        # Focus on URL input
        self.query_one("#repo-url-input", Input).focus()
        
        # Update token button style if token is configured
        if self._token_configured:
            token_btn = self.query_one("#token-config-btn", Button)
            token_btn.label = "🔓 Token"
            token_btn.add_class("token-configured")
    
    def watch_is_loading(self, is_loading: bool) -> None:
        """Show/hide loading overlay."""
        overlay = self.query_one("#loading-overlay")
        label = self.query_one("#loading-label", Label)
        
        if is_loading:
            overlay.remove_class("hidden")
            label.update(self.loading_message)
        else:
            overlay.add_class("hidden")
    
    def _show_loaded_state(self) -> None:
        """Show the loaded repository state."""
        # Hide empty state
        self.query_one("#empty-state").add_class("hidden")
        
        # Show header input
        self.query_one("#repo-input-container").remove_class("hidden")
        
        # Show repo controls
        self.query_one("#repo-controls-container").remove_class("hidden")
        
        # Show filter bar
        self.query_one("#filter-bar").remove_class("hidden")
        
        # Show main content
        self.query_one("#main-content").remove_class("hidden")
        
        # Action bar is always visible
        
        self.has_loaded_repo = True
    
    def _show_empty_state(self) -> None:
        """Show the empty repository state."""
        # Show empty state
        self.query_one("#empty-state").remove_class("hidden")
        
        # Hide everything else
        self.query_one("#repo-controls-container").add_class("hidden")
        self.query_one("#filter-bar").add_class("hidden")
        self.query_one("#main-content").add_class("hidden")
        # Action bar is always visible
        
        self.has_loaded_repo = False
    
    async def load_local_repository(self, repo_path: str) -> None:
        """Load a local Git repository."""
        try:
            repo_path = Path(repo_path).resolve()
            
            if not repo_path.exists():
                raise Exception(f"Path does not exist: {repo_path}")
            
            if not repo_path.is_dir():
                raise Exception(f"Path is not a directory: {repo_path}")
            
            # Check if it's a git repository
            git_dir = repo_path / ".git"
            if not git_dir.exists():
                raise Exception(f"Not a Git repository: {repo_path}")
            
            self.current_repo = {"path": str(repo_path), "type": "local"}
            
            # Get branch info
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "branch", "--all"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True
                )
                
                branches = []
                current_branch = None
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("*"):
                        current_branch = line[2:]
                        branches.append(current_branch)
                    elif line and not line.startswith("remotes/"):
                        branches.append(line)
                
                branch_selector = self.query_one("#branch-selector", Select)
                if branches:
                    branch_selector.set_options([(b, b) for b in branches])
                    if current_branch:
                        branch_selector.value = current_branch
                else:
                    branch_selector.set_options([("main", "main")])
            except Exception as e:
                logger.warning(f"Could not get branch info: {e}")
                branch_selector = self.query_one("#branch-selector", Select)
                branch_selector.set_options([("main", "main")])
            
            # Load file tree
            await self.load_local_tree(repo_path)
            
        except Exception as e:
            raise Exception(f"Failed to load local repository: {e}")
    
    async def load_local_tree(self, repo_path: Path) -> None:
        """Load file tree from local repository."""
        try:
            # Build tree structure from local files
            tree_data = []
            
            # Walk through the directory
            for root, dirs, files in os.walk(repo_path):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in [
                    'node_modules', '__pycache__', 'venv', '.venv', 'env', '.env',
                    'build', 'dist', 'target', 'out'
                ]]
                
                rel_root = Path(root).relative_to(repo_path)
                
                # Add directories
                for dir_name in sorted(dirs):
                    rel_path = str(rel_root / dir_name) if str(rel_root) != '.' else dir_name
                    tree_data.append({
                        "path": rel_path,
                        "type": "tree",
                        "name": dir_name
                    })
                
                # Add files
                for file_name in sorted(files):
                    # Skip hidden files and common ignore patterns
                    if file_name.startswith('.') or file_name.endswith(('.pyc', '.pyo')):
                        continue
                    
                    rel_path = str(rel_root / file_name) if str(rel_root) != '.' else file_name
                    file_path = repo_path / rel_path
                    
                    tree_data.append({
                        "path": rel_path,
                        "type": "blob",
                        "name": file_name,
                        "size": file_path.stat().st_size
                    })
            
            # Build hierarchical tree
            self.tree_data = self.api_client.build_tree_hierarchy(tree_data)
            
            # Load into tree view
            tree_view = self.query_one("#repo-tree", TreeView)
            await tree_view.load_tree(self.tree_data)
            
            # Show the loaded state UI
            self._show_loaded_state()
            
            self.notify(f"Loaded {len(tree_data)} items from local repository", severity="information")
            
        except Exception as e:
            raise Exception(f"Failed to load local tree: {e}")
    
    @on(Button.Pressed, "#load-repo-btn")
    async def load_repository(self, event: Button.Pressed) -> None:
        """Load repository from URL or local path."""
        url_input = self.query_one("#repo-url-input", Input)
        repo_input = url_input.value.strip()
        
        if not repo_input:
            self.notify("Please enter a GitHub repository URL or local path", severity="error")
            return
        
        self.loading_message = "Loading repository..."
        self.is_loading = True
        
        try:
            # Check if it's a local path
            if os.path.exists(repo_input) and os.path.isdir(repo_input):
                # Handle local repository
                self.is_local_repo = True
                await self.load_local_repository(repo_input)
            else:
                # Handle GitHub repository
                self.is_local_repo = False
                # Parse repository URL
                owner, repo = self.api_client.parse_github_url(repo_input)
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
    
    @on(Button.Pressed, "#empty-load-btn")
    async def load_from_empty_state(self, event: Button.Pressed) -> None:
        """Load repository from empty state."""
        # Get URL from empty state input
        empty_input = self.query_one("#empty-state-input", Input)
        repo_url = empty_input.value.strip()
        
        if not repo_url:
            self.notify("Please enter a GitHub repository URL", severity="error")
            return
        
        # Copy URL to main input
        main_input = self.query_one("#repo-url-input", Input)
        main_input.value = repo_url
        
        # Trigger load
        await self.load_repository(event)
    
    @on(Button.Pressed, "#token-config-btn")
    async def configure_token(self, event: Button.Pressed) -> None:
        """Open token configuration dialog."""
        from ..config import get_cli_setting, save_setting_to_cli_config
        
        # Create a simple input dialog using Textual's built-in functionality
        current_token = get_cli_setting("github", "api_token", "")
        has_token = bool(current_token and not current_token.startswith("<"))
        
        if has_token:
            # Mask the current token
            masked_token = current_token[:8] + "*" * (len(current_token) - 12) + current_token[-4:]
            message = f"Current token: {masked_token}\n\nEnter new GitHub personal access token (or leave empty to keep current):"
        else:
            message = "Enter GitHub personal access token:\n\nCreate a token at: https://github.com/settings/tokens\nRequired scopes: repo (private), public_repo (public only)"
        
        # Use a simple approach with Textual's Input dialog
        # For now, we'll use a notification and ask them to add it to the config file
        # In a real implementation, we'd create a proper modal dialog
        
        # Simplified approach: Direct the user to update config
        config_path = "~/.config/tldw_cli/config.toml"
        if has_token:
            self.notify(
                f"To update your GitHub token, edit:\n{config_path}\n\nLook for [github] section, api_token field",
                severity="information",
                timeout=10
            )
        else:
            self.notify(
                f"To add a GitHub token for private repos:\n1. Create token at: https://github.com/settings/tokens\n2. Edit: {config_path}\n3. Add token to [github] section, api_token field",
                severity="information", 
                timeout=15
            )
    
    async def load_tree(self) -> None:
        """Load repository tree structure."""
        if not self.current_repo:
            return
        
        branch = self.query_one("#branch-selector", Select).value
        
        try:
            # Check if lazy loading is enabled
            from ..config import get_cli_setting
            lazy_load = get_cli_setting("github", "lazy_load_tree", True)
            
            # Get repository tree
            self.loading_message = "Loading file structure..."
            
            if lazy_load:
                # Load only root level initially
                flat_tree = await self.api_client.get_repository_tree(
                    self.current_repo["owner"],
                    self.current_repo["repo"],
                    branch,
                    recursive=False  # Only load root level
                )
            else:
                # Load entire tree (old behavior)
                flat_tree = await self.api_client.get_repository_tree(
                    self.current_repo["owner"],
                    self.current_repo["repo"],
                    branch,
                    recursive=True
                )
            
            # Build hierarchical tree
            self.tree_data = self.api_client.build_tree_hierarchy(flat_tree)
            
            # Load into tree view
            tree_view = self.query_one("#repo-tree", TreeView)
            await tree_view.load_tree(self.tree_data)
            
            # Show the loaded state UI
            self._show_loaded_state()
            
            self.notify(f"Loaded {len(flat_tree)} items", severity="information")
            
        except GitHubAPIError as e:
            self.notify(str(e), severity="error")
            logger.error(f"Failed to load tree: {e}")
    
    @on(Select.Changed, "#branch-selector")
    async def on_branch_changed(self, event: Select.Changed) -> None:
        """Handle branch selection change."""
        if self.current_repo and not self.is_loading:
            self.is_loading = True
            if self.is_local_repo:
                # For local repos, we might want to checkout the branch
                # For now, just reload the tree
                await self.load_local_tree(Path(self.current_repo["path"]))
            else:
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
        
        # Update selected files set
        if selected:
            self.selected_files.add(path)
        else:
            self.selected_files.discard(path)
        
        # Action bar is always visible
    
    async def handle_node_expanded(self, path: str, expanded: bool) -> None:
        """Handle node expansion - load children if needed."""
        if not expanded or not self.current_repo:
            return
        
        # Check if lazy loading is enabled
        from ..config import get_cli_setting
        lazy_load = get_cli_setting("github", "lazy_load_tree", True)
        
        if not lazy_load or self.is_local_repo:
            # Already loaded everything or using local repo
            return
        
        # Check if we've already loaded children for this path
        tree_view = self.query_one("#repo-tree", TreeView)
        node = tree_view.nodes.get(path)
        
        if node and node.children_loaded:
            return
        
        # Load children for this directory
        self.run_worker(self.load_node_children(path), exclusive=False)
    
    async def handle_node_selected(self, path: str) -> None:
        """Handle node selection - show file preview."""
        if not self.current_repo:
            return
        
        # Update preview title
        preview_title = self.query_one("#preview-title", Static)
        preview_title.update(f"Preview: {path}")
        
        # Check if it's a file or directory
        tree_view = self.query_one("#repo-tree", TreeView)
        node = tree_view.get_node_by_path(path)
        
        if not node or node.get("type") == "tree":
            # It's a directory, show directory info
            self.query_one("#file-preview", TextArea).text = "Select a file to preview its contents"
            return
        
        # It's a file, fetch and display content
        self.loading_message = "Loading file preview..."
        self.is_loading = True
        
        try:
            if self.is_local_repo:
                # Load from local file
                full_path = Path(self.current_repo["path"]) / path
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # Load from GitHub
                branch = self.query_one("#branch-selector", Select).value
                content = await self.api_client.get_file_content(
                    self.current_repo["owner"],
                    self.current_repo["repo"],
                    path,
                    branch
                )
            
            # Display in preview
            preview_area = self.query_one("#file-preview", TextArea)
            preview_area.text = content
            
            # Try to set language based on file extension
            ext = path.split('.')[-1] if '.' in path else ''
            lang_map = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'jsx': 'javascript',
                'tsx': 'typescript',
                'java': 'java',
                'cpp': 'cpp',
                'c': 'c',
                'h': 'c',
                'hpp': 'cpp',
                'cs': 'csharp',
                'rb': 'ruby',
                'go': 'go',
                'rs': 'rust',
                'php': 'php',
                'swift': 'swift',
                'kt': 'kotlin',
                'scala': 'scala',
                'r': 'r',
                'sql': 'sql',
                'sh': 'bash',
                'bash': 'bash',
                'zsh': 'bash',
                'yml': 'yaml',
                'yaml': 'yaml',
                'json': 'json',
                'xml': 'xml',
                'html': 'html',
                'htm': 'html',
                'css': 'css',
                'scss': 'scss',
                'sass': 'sass',
                'less': 'less',
                'md': 'markdown',
                'markdown': 'markdown',
                'rst': 'restructuredtext',
                'tex': 'latex',
                'dockerfile': 'dockerfile',
                'makefile': 'makefile',
                'cmake': 'cmake',
                'gradle': 'gradle',
                'maven': 'xml'
            }
            
            if ext in lang_map:
                preview_area.language = lang_map[ext]
            else:
                preview_area.language = None
                
        except Exception as e:
            self.query_one("#file-preview", TextArea).text = f"Error loading file: {str(e)}"
            logger.error(f"Failed to load file preview: {e}")
        finally:
            self.is_loading = False
    
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
    
    @on(Select.Changed, "#file-type-filter")
    async def on_filter_type_changed(self, event: Select.Changed) -> None:
        """Handle file type filter changes."""
        await self.apply_filters()
    
    async def apply_filters(self) -> None:
        """Apply current filters to tree view."""
        tree_view = self.query_one("#repo-tree", TreeView)
        search_input = self.query_one("#file-search", Input)
        type_filter = self.query_one("#file-type-filter", Select)
        
        # Apply search filter
        tree_view.set_search_filter(search_input.value)
        
        # Apply type filter
        tree_view.set_file_type_filter(type_filter.value)
    
    @on(Input.Changed, "#file-search")
    async def on_search_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        await self.apply_filters()
    
    @on(Button.Pressed, "#copy-clipboard-btn")
    async def copy_to_clipboard(self, event: Button.Pressed) -> None:
        """Copy compiled text to clipboard."""
        aggregated_text = self.query_one("#aggregated-text", TextArea).text
        
        if not aggregated_text or aggregated_text == "Click 'Generate Compilation' to aggregate selected files":
            self.notify("No compilation to copy. Generate compilation first.", severity="warning")
            return
        
        try:
            # Copy to clipboard (this would need platform-specific implementation)
            # For now, we'll dismiss with the content
            self.notify("Copied compilation to clipboard", severity="success")
            self.dismiss((self.selected_files, aggregated_text))
            
        except Exception as e:
            self.notify(f"Failed to copy to clipboard: {e}", severity="error")
            logger.error(f"Failed to copy to clipboard: {e}")
    
    @on(Button.Pressed, "#export-zip-btn")
    async def export_to_zip(self, event: Button.Pressed) -> None:
        """Export selected files to ZIP."""
        tree_view = self.query_one("#repo-tree", TreeView)
        selected_files = tree_view.get_selected_files()
        
        if not selected_files:
            self.notify("No files selected for export", severity="warning")
            return
        
        if not self.current_repo:
            self.notify("No repository loaded", severity="error")
            return
        
        # Run export in a worker to avoid blocking UI
        self.run_worker(self._export_to_zip_worker(selected_files), exclusive=True)
    
    @on(Button.Pressed, "#generate-compilation-btn")
    async def generate_compilation(self, event: Button.Pressed) -> None:
        """Generate compilation from selected files."""
        tree_view = self.query_one("#repo-tree", TreeView)
        selected_files = tree_view.get_selected_files()
        
        if not selected_files:
            self.notify("No files selected", severity="warning")
            return
        
        if not self.current_repo and not self.is_local_repo:
            return
        
        self.loading_message = "Generating compilation..."
        self.is_loading = True
        
        try:
            # Fetch content for all selected files
            contents = []
            
            if self.is_local_repo:
                # Load from local files
                for file_path in selected_files:
                    try:
                        full_path = Path(self.current_repo["path"]) / file_path
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # Format with file markers
                        contents.append(f"```{file_path}\n{content}\n```")
                    except Exception as e:
                        logger.error(f"Failed to read file {file_path}: {e}")
                        contents.append(f"```{file_path}\n# Error reading file: {e}\n```")
            else:
                # Load from GitHub
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
            self.compiled_text = full_content
            
            # Update the aggregated text display
            aggregated_text_area = self.query_one("#aggregated-text", TextArea)
            aggregated_text_area.text = full_content
            
            self.notify(f"Generated compilation with {len(selected_files)} files", severity="success")
            
        except Exception as e:
            self.notify(f"Failed to generate compilation: {e}", severity="error")
            logger.error(f"Failed to generate compilation: {e}")
        finally:
            self.is_loading = False
    
    @on(Button.Pressed, "#reset-btn")
    def reset_selection(self) -> None:
        """Reset all selections and compiled text."""
        # Clear selections
        tree_view = self.query_one("#repo-tree", TreeView)
        tree_view.deselect_all()
        self.selected_files.clear()
        
        # Clear compiled text
        self.compiled_text = ""
        aggregated_text_area = self.query_one("#aggregated-text", TextArea)
        aggregated_text_area.text = "Click 'Generate Compilation' to aggregate selected files"
        
        # Update selection stats
        count_label = self.query_one("#selection-count", Static)
        size_label = self.query_one("#selection-size", Static)
        tokens_label = self.query_one("#selection-tokens", Static)
        
        count_label.update("0 files selected")
        size_label.update("0 KB")
        tokens_label.update("~0 tokens")
        
        self.notify("Reset selection and compilation", severity="info")
    
    def action_close_window(self) -> None:
        """Close window action."""
        self.dismiss(None)
    
    def action_select_all(self) -> None:
        """Select all files."""
        if self.has_loaded_repo:
            tree_view = self.query_one("#repo-tree", TreeView)
            tree_view.select_all()
            self.notify("Selected all files", severity="info")
    
    def action_deselect_all(self) -> None:
        """Deselect all files."""
        if self.has_loaded_repo:
            tree_view = self.query_one("#repo-tree", TreeView)
            tree_view.deselect_all()
            self.selected_files.clear()
            self.notify("Deselected all files", severity="info")
    
    def action_invert_selection(self) -> None:
        """Invert selection."""
        if self.has_loaded_repo:
            tree_view = self.query_one("#repo-tree", TreeView)
            tree_view.invert_selection()
            self.notify("Inverted selection", severity="info")
    
    @work(thread=True)
    async def _export_to_zip_worker(self, selected_files: List[str]) -> None:
        """Worker to export selected files to ZIP."""
        try:
            # Show loading state
            self.loading_message = "Preparing ZIP export..."
            self.is_loading = True
            
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.is_local_repo:
                repo_name = Path(self.current_repo["path"]).name
            else:
                repo_name = self.current_repo["repo"]
            
            default_filename = f"{repo_name}_{timestamp}.zip"
            
            # Get save location from user
            # For now, save to downloads folder
            downloads_path = Path.home() / "Downloads"
            zip_path = downloads_path / default_filename
            
            # Ensure downloads directory exists
            downloads_path.mkdir(exist_ok=True)
            
            # Create ZIP file
            self.loading_message = f"Creating ZIP file with {len(selected_files)} files..."
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                total_files = len(selected_files)
                
                if self.is_local_repo:
                    # Handle local files one by one
                    for idx, file_path in enumerate(selected_files):
                        # Update progress
                        progress = (idx + 1) / total_files
                        self.loading_message = f"Adding to ZIP: {file_path} ({idx + 1}/{total_files})"
                        
                        try:
                            # Read from local file
                            full_path = Path(self.current_repo["path"]) / file_path
                            if full_path.exists() and full_path.is_file():
                                with open(full_path, 'rb') as f:
                                    content = f.read()
                                zip_file.writestr(file_path, content)
                        except Exception as e:
                            logger.error(f"Failed to add {file_path} to ZIP: {e}")
                            # Add error file
                            error_content = f"Error reading file: {str(e)}"
                            zip_file.writestr(f"{file_path}.error", error_content)
                else:
                    # Batch fetch from GitHub
                    branch = self.query_one("#branch-selector", Select).value
                    
                    def update_progress(completed, total, current_file):
                        self.loading_message = f"Fetching files: {completed}/{total} - {current_file}"
                    
                    # Fetch all files concurrently
                    self.loading_message = f"Fetching {total_files} files from GitHub..."
                    file_contents = await self.api_client.get_files_content_batch(
                        self.current_repo["owner"],
                        self.current_repo["repo"],
                        selected_files,
                        branch,
                        progress_callback=update_progress
                    )
                    
                    # Add fetched files to ZIP
                    self.loading_message = "Creating ZIP archive..."
                    for file_path in selected_files:
                        if file_path in file_contents:
                            # Write as UTF-8 encoded bytes
                            zip_file.writestr(file_path, file_contents[file_path].encode('utf-8'))
                        else:
                            # Add error file for failed fetches
                            error_content = f"Error fetching file from GitHub"
                            zip_file.writestr(f"{file_path}.error", error_content)
                
                # Add a manifest file
                manifest = {
                    "repository": repo_name,
                    "export_date": datetime.now().isoformat(),
                    "total_files": len(selected_files),
                    "files": selected_files
                }
                if not self.is_local_repo:
                    manifest["github_url"] = f"https://github.com/{self.current_repo['owner']}/{self.current_repo['repo']}"
                    manifest["branch"] = self.query_one("#branch-selector", Select).value
                
                zip_file.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
            
            # Get file size
            file_size = zip_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            self.call_from_thread(
                self.notify,
                f"ZIP exported successfully!\nSaved to: {zip_path}\nSize: {size_mb:.1f} MB",
                severity="success",
                timeout=10
            )
            
            # Open the containing folder (platform-specific)
            if os.name == 'nt':  # Windows
                subprocess.Popen(['explorer', '/select,', str(zip_path)])
            elif os.name == 'posix':  # macOS and Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.Popen(['open', '-R', str(zip_path)])
                else:  # Linux
                    subprocess.Popen(['xdg-open', str(downloads_path)])
            
        except Exception as e:
            logger.error(f"Failed to export ZIP: {e}")
            self.call_from_thread(
                self.notify,
                f"Failed to export ZIP: {str(e)}",
                severity="error"
            )
        finally:
            self.is_loading = False
    
    @work(thread=True)
    async def load_node_children(self, node_path: str) -> None:
        """Load children for a specific directory node."""
        try:
            branch = self.query_one("#branch-selector", Select).value
            
            # Fetch directory contents
            children = await self.api_client.get_directory_contents(
                self.current_repo["owner"],
                self.current_repo["repo"],
                node_path,
                branch
            )
            
            # Transform children to include full paths
            tree_view = self.query_one("#repo-tree", TreeView)
            child_nodes = []
            
            for child in children:
                child_nodes.append({
                    'path': child['path'],
                    'name': child['name'],
                    'type': child['type'],
                    'size': child.get('size', 0),
                    'children': [] if child['type'] == 'tree' else None
                })
            
            # Update tree view with new children
            await self.call_from_thread(
                tree_view.expand_node,
                node_path,
                child_nodes
            )
            
        except Exception as e:
            logger.error(f"Failed to load children for {node_path}: {e}")
            self.call_from_thread(
                self.notify,
                f"Failed to load directory contents: {str(e)}",
                severity="error"
            )
    
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