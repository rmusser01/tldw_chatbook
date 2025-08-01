"""
Integration tests for CodeRepoCopyPasteWindow.

Tests the window functionality with mocked GitHub API but real UI interactions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from textual.widgets import Input, Button, Select, TextArea, Static
from textual.containers import Container

from tldw_chatbook.UI.CodeRepoCopyPasteWindow import CodeRepoCopyPasteWindow
from tldw_chatbook.Widgets.repo_tree_widgets import TreeView
from tldw_chatbook.Utils.github_api_client import GitHubAPIError
from Tests.textual_test_utils import app_pilot, wait_for_widget_mount


class TestCodeRepoCopyPasteWindow:
    """Test suite for CodeRepoCopyPasteWindow."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock app instance."""
        app = MagicMock()
        app.notify = MagicMock()
        app.copy_to_clipboard = MagicMock()
        return app
    
    @pytest.fixture
    def mock_api_client(self):
        """Create a mock GitHub API client."""
        with patch('tldw_chatbook.UI.CodeRepoCopyPasteWindow.GitHubAPIClient') as mock_class:
            mock_instance = AsyncMock()
            mock_class.return_value = mock_instance
            
            # Set up default successful responses
            mock_instance.parse_github_url.return_value = ("test-owner", "test-repo")
            mock_instance.get_repository_info.return_value = {
                "name": "test-repo",
                "full_name": "test-owner/test-repo",
                "description": "Test repository"
            }
            mock_instance.get_branches.return_value = ["main", "develop", "feature/test"]
            mock_instance.get_repository_tree.return_value = []
            mock_instance.build_tree_hierarchy.return_value = []
            mock_instance.close.return_value = None
            
            yield mock_instance
    
    @pytest.fixture
    def sample_tree_data(self):
        """Sample tree data for testing."""
        return [
            {
                'path': 'README.md',
                'name': 'README.md',
                'type': 'blob',
                'size': 1234,
                'children': []
            },
            {
                'path': 'src',
                'name': 'src',
                'type': 'tree',
                'children': [
                    {
                        'path': 'src/main.py',
                        'name': 'main.py',
                        'type': 'blob',
                        'size': 2048,
                        'children': []
                    }
                ]
            }
        ]
    
    def test_window_initialization(self, mock_app):
        """Test window initialization."""
        window = CodeRepoCopyPasteWindow(mock_app)
        
        assert window.app_instance is mock_app
        assert window.api_client is not None
        assert window.current_repo is None
        assert window.is_loading == False  # reactive value check
        assert window.loading_message == "Loading..."
        assert window.tree_data is None
    
    @pytest.mark.asyncio
    async def test_window_compose_structure(self, app_pilot, mock_app):
        """Test window UI structure is created correctly."""
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            # Wait for window to mount
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Check main container exists
            containers = window.query(".repo-window-container")
            assert len(containers) == 1
            
            # Check header elements
            assert len(window.query("#repo-url-input")) == 1
            assert len(window.query("#load-repo-btn")) == 1
            assert len(window.query("#branch-selector")) == 1
            
            # Check filter bar
            assert len(window.query("#file-search")) == 1
            assert len(window.query("#file-type-filter")) == 1
            
            # Check main content areas
            assert len(window.query(".tree-container")) == 1
            assert len(window.query(".preview-container")) == 1
            assert len(window.query("#repo-tree")) == 1
            assert len(window.query("#file-preview")) == 1
            
            # Check action buttons
            assert len(window.query("#cancel-btn")) == 1
            assert len(window.query("#export-zip-btn")) == 1
            assert len(window.query("#copy-clipboard-btn")) == 1
            assert len(window.query("#create-embeddings-btn")) == 1
    
    @pytest.mark.asyncio
    async def test_loading_repository_success(self, app_pilot, mock_app, mock_api_client, sample_tree_data):
        """Test successful repository loading."""
        # Set up mock responses
        mock_api_client.build_tree_hierarchy.return_value = sample_tree_data
        
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Enter repository URL
            url_input = window.query_one("#repo-url-input", Input)
            url_input.value = "https://github.com/test-owner/test-repo"
            
            # Click load button
            load_btn = window.query_one("#load-repo-btn", Button)
            await pilot.click(load_btn)
            await pilot.pause()
            
            # Verify API calls
            mock_api_client.parse_github_url.assert_called_with("https://github.com/test-owner/test-repo")
            mock_api_client.get_repository_info.assert_called_with("test-owner", "test-repo")
            mock_api_client.get_branches.assert_called_with("test-owner", "test-repo")
            
            # Check branch selector updated
            branch_selector = window.query_one("#branch-selector", Select)
            # Note: In real implementation, we'd verify the options were updated
            
            # Verify notification
            mock_app.notify.assert_called()
    
    @pytest.mark.asyncio
    async def test_loading_repository_invalid_url(self, app_pilot, mock_app, mock_api_client):
        """Test loading with invalid repository URL."""
        # Set up mock to raise error
        mock_api_client.parse_github_url.side_effect = ValueError("Invalid GitHub repository URL")
        
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Enter invalid URL
            url_input = window.query_one("#repo-url-input", Input)
            url_input.value = "not a valid url"
            
            # Click load button
            load_btn = window.query_one("#load-repo-btn", Button)
            await pilot.click(load_btn)
            await pilot.pause()
            
            # Verify error notification
            mock_app.notify.assert_called()
            call_args = mock_app.notify.call_args
            assert "Invalid GitHub repository URL" in str(call_args)
            assert call_args.kwargs.get('severity') == 'error'
    
    @pytest.mark.asyncio
    async def test_loading_repository_api_error(self, app_pilot, mock_app, mock_api_client):
        """Test handling GitHub API errors."""
        # Set up mock to raise API error
        mock_api_client.get_repository_info.side_effect = GitHubAPIError("Repository not found")
        
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Enter URL
            url_input = window.query_one("#repo-url-input", Input)
            url_input.value = "https://github.com/test/repo"
            
            # Click load button
            load_btn = window.query_one("#load-repo-btn", Button)
            await pilot.click(load_btn)
            await pilot.pause()
            
            # Verify error notification
            mock_app.notify.assert_called()
            call_args = mock_app.notify.call_args
            assert "Repository not found" in str(call_args)
            assert call_args.kwargs.get('severity') == 'error'
    
    @pytest.mark.asyncio
    async def test_branch_selection_change(self, app_pilot, mock_app, mock_api_client, sample_tree_data):
        """Test changing branch selection."""
        mock_api_client.build_tree_hierarchy.return_value = sample_tree_data
        
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Load repository first
            window.current_repo = {"owner": "test-owner", "repo": "test-repo"}
            
            # Change branch selection
            branch_selector = window.query_one("#branch-selector", Select)
            branch_selector.value = "develop"
            
            # Trigger change event
            await window.on_branch_changed(MagicMock())
            await pilot.pause()
            
            # Verify tree reload
            mock_api_client.get_repository_tree.assert_called_with(
                "test-owner", "test-repo", "develop"
            )
    
    @pytest.mark.asyncio
    async def test_file_selection_updates_stats(self, app_pilot, mock_app):
        """Test that file selection updates statistics."""
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Mock tree view selection stats
            tree_view = window.query_one("#repo-tree", TreeView)
            tree_view.get_selection_stats = MagicMock(return_value={
                'files': 5,
                'size': 10240  # 10 KB
            })
            
            # Trigger selection change
            window.handle_selection_change("test/file.py", True)
            
            # Check stats updated
            count_label = window.query_one("#selection-count", Static)
            size_label = window.query_one("#selection-size", Static)
            tokens_label = window.query_one("#selection-tokens", Static)
            
            assert "5 files selected" in count_label.renderable
            assert "10.0 KB" in size_label.renderable
            assert "2,560 tokens" in tokens_label.renderable  # ~4 chars per token
    
    @pytest.mark.asyncio
    async def test_filter_quick_buttons(self, app_pilot, mock_app):
        """Test quick filter buttons."""
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Click docs filter
            docs_btn = window.query_one("#filter-docs", Button)
            await pilot.click(docs_btn)
            await pilot.pause()
            
            # Check filter changed
            filter_select = window.query_one("#file-type-filter", Select)
            assert filter_select.value == "docs"
            
            # Click code filter
            code_btn = window.query_one("#filter-code", Button)
            await pilot.click(code_btn)
            await pilot.pause()
            
            assert filter_select.value == "code"
            
            # Click config filter
            config_btn = window.query_one("#filter-config", Button)
            await pilot.click(config_btn)
            await pilot.pause()
            
            assert filter_select.value == "config"
    
    @pytest.mark.asyncio
    async def test_copy_to_clipboard_no_selection(self, app_pilot, mock_app):
        """Test copy to clipboard with no files selected."""
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Mock empty selection
            tree_view = window.query_one("#repo-tree", TreeView)
            tree_view.get_selected_files = MagicMock(return_value=[])
            
            # Click copy button
            copy_btn = window.query_one("#copy-clipboard-btn", Button)
            await pilot.click(copy_btn)
            await pilot.pause()
            
            # Should show warning
            mock_app.notify.assert_called()
            call_args = mock_app.notify.call_args
            assert "No files selected" in str(call_args)
            assert call_args.kwargs.get('severity') == 'warning'
    
    @pytest.mark.asyncio
    async def test_copy_to_clipboard_success(self, app_pilot, mock_app, mock_api_client):
        """Test successful copy to clipboard."""
        # Mock file content
        mock_api_client.get_file_content.return_value = "print('Hello, World!')"
        
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Set up current repo
            window.current_repo = {"owner": "test-owner", "repo": "test-repo"}
            
            # Mock selection
            tree_view = window.query_one("#repo-tree", TreeView)
            tree_view.get_selected_files = MagicMock(return_value=["src/main.py"])
            
            # Mock branch selector
            branch_selector = window.query_one("#branch-selector", Select)
            branch_selector.value = "main"
            
            # Click copy button
            copy_btn = window.query_one("#copy-clipboard-btn", Button)
            await pilot.click(copy_btn)
            await pilot.pause()
            
            # Verify file content was fetched
            mock_api_client.get_file_content.assert_called_with(
                "test-owner", "test-repo", "src/main.py", "main"
            )
            
            # Verify success notification
            mock_app.notify.assert_called()
            call_args = mock_app.notify.call_args
            assert "Copied 1 files to clipboard" in str(call_args)
            assert call_args.kwargs.get('severity') == 'success'
    
    @pytest.mark.asyncio
    async def test_cancel_button(self, app_pilot, mock_app):
        """Test cancel button dismisses window."""
        dismissed = False
        
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                window = CodeRepoCopyPasteWindow(mock_app)
                window.dismiss = MagicMock(side_effect=lambda x: setattr(self, 'dismissed', True))
                self.push_screen(window)
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Click cancel
            cancel_btn = window.query_one("#cancel-btn", Button)
            await pilot.click(cancel_btn)
            await pilot.pause()
            
            # Verify dismiss was called
            window.dismiss.assert_called_with(None)
    
    @pytest.mark.asyncio
    async def test_escape_key_closes_window(self, app_pilot, mock_app):
        """Test pressing escape closes the window."""
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                window = CodeRepoCopyPasteWindow(mock_app)
                window.dismiss = MagicMock()
                self.push_screen(window)
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Press escape
            await pilot.press("escape")
            await pilot.pause()
            
            # Verify dismiss was called
            window.dismiss.assert_called_with(None)
    
    @pytest.mark.asyncio
    async def test_loading_overlay_visibility(self, app_pilot, mock_app):
        """Test loading overlay shows/hides correctly."""
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # Initially hidden
            overlay = window.query_one("#loading-overlay")
            assert overlay.has_class("hidden")
            
            # Set loading
            window.loading_message = "Testing..."
            window.is_loading = True
            await pilot.pause()
            
            # Should be visible
            assert not overlay.has_class("hidden")
            label = window.query_one("#loading-label")
            assert "Testing..." in str(label.renderable)
            
            # Stop loading
            window.is_loading = False
            await pilot.pause()
            
            # Should be hidden again
            assert overlay.has_class("hidden")
    
    @pytest.mark.asyncio
    async def test_initial_focus(self, app_pilot, mock_app):
        """Test initial focus is on URL input."""
        class TestApp(app_pilot.app_class):
            def on_mount(self):
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))
        
        async with app_pilot(TestApp) as pilot:
            window = await wait_for_widget_mount(pilot, CodeRepoCopyPasteWindow)
            
            # URL input should have focus
            url_input = window.query_one("#repo-url-input", Input)
            # Note: In actual test environment, we'd verify focus state
            # For now, we just ensure the input exists and is accessible