"""
Full integration tests for Code Repo Copy/Paste feature.

Tests the complete workflow through to file export. The standalone Coding
screen/tab that used to host this window has been retired (folded into
Console); the window itself is exercised directly here.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.CodeRepoCopyPasteWindow import CodeRepoCopyPasteWindow
from tldw_chatbook.Utils.github_api_client import GitHubAPIError
from Tests.textual_test_utils import app_pilot


class CodeRepoTestApp(TldwCli):
    """Minimal app shell for code-repo integration screens."""

    def compose(self):
        yield from []

    def on_mount(self):
        pass


class TestCodeRepoIntegration:
    """Integration tests for complete Code Repo Copy/Paste workflow."""
    
    @pytest.fixture
    def mock_github_api(self):
        """Mock GitHub API for integration tests."""
        with patch('tldw_chatbook.UI.CodeRepoCopyPasteWindow.GitHubAPIClient') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            # Default successful responses
            mock_instance.parse_github_url.return_value = ("test-owner", "test-repo")
            mock_instance.get_repository_info = AsyncMock(return_value={
                "name": "test-repo",
                "full_name": "test-owner/test-repo"
            })
            mock_instance.get_branches = AsyncMock(return_value=["main", "develop"])
            mock_instance.get_repository_tree = AsyncMock(return_value=[
                {'path': 'README.md', 'name': 'README.md', 'type': 'blob', 'size': 1000},
                {'path': 'LICENSE', 'name': 'LICENSE', 'type': 'blob', 'size': 500},
                {'path': 'src', 'name': 'src', 'type': 'tree'},
                {'path': 'src/main.py', 'name': 'main.py', 'type': 'blob', 'size': 2000},
            ])
            mock_instance.build_tree_hierarchy.return_value = [
                {
                    'path': 'README.md',
                    'name': 'README.md',
                    'type': 'blob',
                    'size': 1000,
                    'children': []
                },
                {
                    'path': 'LICENSE',
                    'name': 'LICENSE',
                    'type': 'blob',
                    'size': 500,
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
                            'size': 2000,
                            'children': []
                        }
                    ]
                }
            ]
            mock_instance.get_file_content = AsyncMock(side_effect=lambda o, r, p, b: f"Content of {p}")
            mock_instance.close = AsyncMock(return_value=None)
            
            yield mock_instance
    
    @pytest.fixture
    def mock_clipboard(self):
        """Mock clipboard operations."""
        with patch('pyperclip.copy') as mock_copy:
            yield mock_copy
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, app_pilot, mock_github_api):
        """Test error handling throughout the workflow."""
        # Set up API to fail
        mock_github_api.get_repository_info.side_effect = GitHubAPIError("Repository not found")
        
        errors_captured = []
        
        class TestApp(CodeRepoTestApp):
            def notify(self, message, severity="information", **kwargs):
                if severity == "error":
                    errors_captured.append(message)
                super().notify(message, severity=severity, **kwargs)
        
        async with app_pilot(TestApp) as pilot:
            # Open repo window directly
            repo_window = CodeRepoCopyPasteWindow(pilot.app)
            await pilot.app.push_screen(repo_window)
            await pilot.pause()
            
            # Try to load non-existent repository
            url_input = repo_window.query_one("#repo-url-input")
            url_input.value = "https://github.com/nonexistent/repo"
            
            load_button = repo_window.query_one("#load-repo-btn")
            load_button.press()
            await pilot.pause()
            
            # Should show error
            assert len(errors_captured) > 0
            assert "Repository not found" in errors_captured[0]
            
            # Window should still be functional
            assert repo_window.is_loading is False
            assert repo_window.current_repo is None
    
    @pytest.mark.asyncio
    async def test_large_repository_handling(self, app_pilot, mock_github_api):
        """Test handling of large repositories."""
        # Create large tree structure
        large_tree = []
        for i in range(100):
            large_tree.append({
                'path': f'file_{i}.py',
                'name': f'file_{i}.py',
                'type': 'blob',
                'size': 1000 * (i + 1),
                'children': []
            })
        
        mock_github_api.build_tree_hierarchy.return_value = large_tree
        
        class TestApp(CodeRepoTestApp):
            pass
        
        async with app_pilot(TestApp) as pilot:
            repo_window = CodeRepoCopyPasteWindow(pilot.app)
            await pilot.app.push_screen(repo_window)
            await pilot.pause()
            
            # Set up repo
            repo_window.current_repo = {"owner": "test", "repo": "large-repo"}
            
            # Load tree
            await repo_window.load_tree()
            await pilot.pause()
            
            # Verify all nodes loaded
            tree_view = repo_window.query_one("#repo-tree")
            assert len(tree_view.nodes) == 100
            
            # Select all files
            for path in tree_view.nodes:
                tree_view.select_node(path, True)
            
            # Check statistics
            stats = tree_view.get_selection_stats()
            assert stats['files'] == 100
            assert stats['size'] == sum(1000 * (i + 1) for i in range(100))
    
    @pytest.mark.asyncio 
    async def test_branch_switching_workflow(self, app_pilot, mock_github_api):
        """Test switching between branches."""
        # Different content for different branches
        main_tree = [{'path': 'main.txt', 'name': 'main.txt', 'type': 'blob', 'children': []}]
        develop_tree = [{'path': 'develop.txt', 'name': 'develop.txt', 'type': 'blob', 'children': []}]
        
        mock_github_api.build_tree_hierarchy.return_value = main_tree
        
        class TestApp(CodeRepoTestApp):
            pass
        
        async with app_pilot(TestApp) as pilot:
            repo_window = CodeRepoCopyPasteWindow(pilot.app)
            await pilot.app.push_screen(repo_window)
            await pilot.pause()
            
            # Load repository
            url_input = repo_window.query_one("#repo-url-input")
            url_input.value = "https://github.com/test/repo"
            
            load_button = repo_window.query_one("#load-repo-btn")
            load_button.press()
            await pilot.pause()
            
            # Should load main branch by default
            tree_view = repo_window.query_one("#repo-tree")
            assert 'main.txt' in tree_view.nodes
            
            # Switch to develop branch
            mock_github_api.build_tree_hierarchy.return_value = develop_tree
            branch_selector = repo_window.query_one("#branch-selector")
            branch_selector.value = "develop"
            await repo_window.on_branch_changed(MagicMock())
            await pilot.pause()
            
            # Should now show develop content
            assert 'develop.txt' in tree_view.nodes
            assert 'main.txt' not in tree_view.nodes
    
    @pytest.mark.asyncio
    async def test_cancellation_and_cleanup(self, app_pilot, mock_github_api):
        """Test cancellation and resource cleanup."""
        cleanup_called = False
        
        async def mock_close():
            nonlocal cleanup_called
            cleanup_called = True
        
        mock_github_api.close = mock_close
        
        class TestApp(CodeRepoTestApp):
            pass
        
        async with app_pilot(TestApp) as pilot:
            repo_window = CodeRepoCopyPasteWindow(pilot.app)
            await pilot.app.push_screen(repo_window)
            await pilot.pause()
            
            # Cleanup should close the API client when window exits.
            await repo_window.__aexit__(None, None, None)
            assert cleanup_called
    
    @pytest.mark.asyncio
    async def test_file_filtering_workflow(self, app_pilot, mock_github_api):
        """Test file filtering functionality."""
        # Mixed file types
        mixed_tree = [
            {'path': 'README.md', 'name': 'README.md', 'type': 'blob', 'children': []},
            {'path': 'main.py', 'name': 'main.py', 'type': 'blob', 'children': []},
            {'path': 'config.json', 'name': 'config.json', 'type': 'blob', 'children': []},
            {'path': 'styles.css', 'name': 'styles.css', 'type': 'blob', 'children': []},
        ]
        mock_github_api.build_tree_hierarchy.return_value = mixed_tree
        
        class TestApp(CodeRepoTestApp):
            pass
        
        async with app_pilot(TestApp) as pilot:
            repo_window = CodeRepoCopyPasteWindow(pilot.app)
            await pilot.app.push_screen(repo_window)
            await pilot.pause()
            
            # Load repository
            repo_window.current_repo = {"owner": "test", "repo": "mixed"}
            await repo_window.load_tree()
            await pilot.pause()
            
            # Apply different filters
            filter_docs = repo_window.query_one("#filter-docs")
            filter_docs.press()
            await pilot.pause()
            
            filter_select = repo_window.query_one("#file-type-filter")
            assert filter_select.value == "docs"
            
            # Try other filters
            filter_code = repo_window.query_one("#filter-code")
            filter_code.press()
            await pilot.pause()
            
            assert filter_select.value == "code"
            
            filter_config = repo_window.query_one("#filter-config")
            filter_config.press()
            await pilot.pause()
            
            assert filter_select.value == "config"
    
    @pytest.mark.asyncio
    async def test_keyboard_shortcuts(self, app_pilot, mock_github_api):
        """Test keyboard shortcuts in the window."""
        class TestApp(CodeRepoTestApp):
            pass
        
        async with app_pilot(TestApp) as pilot:
            repo_window = CodeRepoCopyPasteWindow(pilot.app)
            await pilot.app.push_screen(repo_window)
            await pilot.pause()
            
            # Test Ctrl+A (select all)
            await pilot.press("ctrl+a")
            await pilot.pause()
            
            # Test Ctrl+Shift+A (deselect all)  
            await pilot.press("ctrl+shift+a")
            await pilot.pause()
            
            # Test Ctrl+I (invert selection)
            await pilot.press("ctrl+i")
            await pilot.pause()
            
            # Test Escape (close window)
            repo_window.dismiss = MagicMock()
            await pilot.press("escape")
            await pilot.pause()
            
            repo_window.dismiss.assert_called_once()
