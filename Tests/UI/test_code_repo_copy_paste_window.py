"""Mounted-screen tests for CodeRepoCopyPasteWindow."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select, Static, TextArea

from tldw_chatbook.UI.CodeRepoCopyPasteWindow import CodeRepoCopyPasteWindow
from Tests.textual_test_utils import app_pilot


async def _active_window(pilot) -> CodeRepoCopyPasteWindow:
    await pilot.pause(0.05)
    window = pilot.app.screen
    assert isinstance(window, CodeRepoCopyPasteWindow)
    return window


class TestCodeRepoCopyPasteWindow:
    @pytest.fixture
    def mock_app(self):
        app = MagicMock()
        app.notify = MagicMock()
        app.copy_to_clipboard = MagicMock()
        return app

    @pytest.fixture
    def mock_api_client(self):
        with patch("tldw_chatbook.UI.CodeRepoCopyPasteWindow.GitHubAPIClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            mock_instance.parse_github_url = Mock(return_value=("test-owner", "test-repo"))
            mock_instance.get_repository_info = AsyncMock(return_value={
                "name": "test-repo",
                "full_name": "test-owner/test-repo",
                "description": "Test repository",
            })
            mock_instance.get_branches = AsyncMock(return_value=["main", "develop"])
            mock_instance.get_repository_tree = AsyncMock(return_value=[])
            mock_instance.get_file_content = AsyncMock(return_value="print('Hello')")
            mock_instance.build_tree_hierarchy = Mock(return_value=[])
            mock_instance.close = AsyncMock(return_value=None)
            yield mock_instance

    @pytest.fixture
    def sample_tree_data(self):
        return [
            {
                "path": "README.md",
                "name": "README.md",
                "type": "blob",
                "size": 1024,
                "children": [],
            },
            {
                "path": "src",
                "name": "src",
                "type": "tree",
                "children": [
                    {
                        "path": "src/main.py",
                        "name": "main.py",
                        "type": "blob",
                        "size": 2048,
                        "children": [],
                    }
                ],
            },
        ]

    def test_window_initialization(self, mock_app, mock_api_client):
        window = CodeRepoCopyPasteWindow(mock_app)

        assert window.app_instance is mock_app
        assert window.api_client is not None
        assert window.current_repo is None
        assert window.is_loading is False
        assert window.loading_message == "Loading..."
        assert window.tree_data is None
        assert window.compiled_text == ""

    @pytest.mark.asyncio
    async def test_window_compose_structure(self, app_pilot, mock_app, mock_api_client):
        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)

            assert len(window.query(".repo-window-container")) == 1
            assert len(window.query("#repo-url-input")) == 1
            assert len(window.query("#load-repo-btn")) == 1
            assert len(window.query("#branch-selector")) == 1
            assert len(window.query("#file-search")) == 1
            assert len(window.query("#file-type-filter")) == 1
            assert len(window.query("#repo-tree")) == 1
            assert len(window.query("#file-preview")) == 1
            assert len(window.query("#aggregated-text")) == 1
            assert len(window.query("#reset-btn")) == 1
            assert len(window.query("#export-zip-btn")) == 1
            assert len(window.query("#copy-clipboard-btn")) == 1
            assert len(window.query("#generate-compilation-btn")) == 1

    @pytest.mark.asyncio
    async def test_loading_repository_success(
        self,
        app_pilot,
        mock_app,
        mock_api_client,
        sample_tree_data,
    ):
        mock_api_client.build_tree_hierarchy.return_value = sample_tree_data

        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)
            window.notify = Mock()

            url_input = window.query_one("#repo-url-input", Input)
            url_input.value = "https://github.com/test-owner/test-repo"

            await window.load_repository(Mock())
            await pilot.pause()

            assert window.current_repo == {"owner": "test-owner", "repo": "test-repo"}
            assert window.has_loaded_repo is True
            assert not window.query_one("#repo-controls-container").has_class("hidden")
            assert not window.query_one("#filter-bar").has_class("hidden")
            assert not window.query_one("#main-content").has_class("hidden")
            mock_api_client.parse_github_url.assert_called_with(
                "https://github.com/test-owner/test-repo"
            )
            mock_api_client.get_repository_info.assert_called_with("test-owner", "test-repo")
            mock_api_client.get_branches.assert_called_with("test-owner", "test-repo")
            window.notify.assert_called()

    @pytest.mark.asyncio
    async def test_loading_repository_invalid_url(self, app_pilot, mock_app, mock_api_client):
        mock_api_client.parse_github_url.side_effect = ValueError("Invalid GitHub repository URL")

        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)
            window.notify = Mock()

            url_input = window.query_one("#repo-url-input", Input)
            url_input.value = "not a valid url"

            await window.load_repository(Mock())
            await pilot.pause()

            window.notify.assert_called()
            call_args = window.notify.call_args
            assert "Failed to load repository: Invalid GitHub repository URL" in str(call_args)
            assert call_args.kwargs.get("severity") == "error"

    @pytest.mark.asyncio
    async def test_quick_filter_buttons_update_select(self, app_pilot, mock_app, mock_api_client):
        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)
            filter_select = window.query_one("#file-type-filter", Select)

            await window.filter_docs(Mock())
            await pilot.pause()
            assert filter_select.value == "docs"

            await window.filter_code(Mock())
            await pilot.pause()
            assert filter_select.value == "code"

            await window.filter_config(Mock())
            await pilot.pause()
            assert filter_select.value == "config"

    @pytest.mark.asyncio
    async def test_copy_to_clipboard_requires_compilation(self, app_pilot, mock_app, mock_api_client):
        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)
            window.notify = Mock()

            await window.copy_to_clipboard(Mock())
            await pilot.pause()

            window.notify.assert_called_with(
                "No compilation to copy. Generate compilation first.",
                severity="warning",
            )

    @pytest.mark.asyncio
    async def test_generate_compilation_requires_selected_files(
        self,
        app_pilot,
        mock_app,
        mock_api_client,
    ):
        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)
            window.notify = Mock()

            await window.generate_compilation(Mock())
            await pilot.pause()

            window.notify.assert_called_with("No files selected", severity="warning")

    @pytest.mark.asyncio
    async def test_reset_clears_selection_and_compilation(self, app_pilot, mock_app, mock_api_client):
        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)
            window.notify = Mock()

            tree_view = window.query_one("#repo-tree")
            tree_view.deselect_all = Mock()
            window.selected_files = {"src/main.py"}
            window.compiled_text = "compiled output"
            window.query_one("#aggregated-text", TextArea).text = "compiled output"

            window.reset_selection()

            tree_view.deselect_all.assert_called_once()
            assert window.selected_files == set()
            assert window.compiled_text == ""
            assert (
                window.query_one("#aggregated-text", TextArea).text
                == "Click 'Generate Compilation' to aggregate selected files"
            )
            window.notify.assert_called_with("Reset selection and compilation", severity="info")

    @pytest.mark.asyncio
    async def test_loading_overlay_visibility(self, app_pilot, mock_app, mock_api_client):
        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)
            overlay = window.query_one("#loading-overlay")
            label = window.query_one("#loading-label", Static)

            assert overlay.has_class("hidden")

            window.loading_message = "Testing..."
            window.is_loading = True
            await pilot.pause()

            assert not overlay.has_class("hidden")
            assert "Testing..." in str(label.render())

            window.is_loading = False
            await pilot.pause()

            assert overlay.has_class("hidden")

    @pytest.mark.asyncio
    async def test_escape_key_closes_window(self, app_pilot, mock_app, mock_api_client):
        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)
            window.dismiss = Mock()

            await pilot.press("escape")
            await pilot.pause()

            window.dismiss.assert_called_with(None)

    @pytest.mark.asyncio
    async def test_initial_focus_is_repo_input(self, app_pilot, mock_app, mock_api_client):
        class TestApp(App):
            def on_mount(self) -> None:
                self.push_screen(CodeRepoCopyPasteWindow(mock_app))

        async with await app_pilot(TestApp) as pilot:
            window = await _active_window(pilot)
            await pilot.pause()

            url_input = window.query_one("#repo-url-input", Input)
            assert url_input.has_focus
