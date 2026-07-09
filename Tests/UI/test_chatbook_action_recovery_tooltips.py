from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.ChatbookExportManagementWindow import ChatbookExportManagementWindow
from tldw_chatbook.UI.ChatbookTemplatesWindow import ChatbookTemplatesWindow
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


def _assert_button_tooltips(root, expected_tooltips: dict[str, str]) -> None:
    for button_id, expected_tooltip in expected_tooltips.items():
        button = root.query_one(f"#{button_id}", Button)
        assert str(button.tooltip) == expected_tooltip


@pytest.mark.asyncio
async def test_chatbook_template_use_action_explains_selection_requirement():
    class TemplatesApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbookTemplatesWindow(SimpleNamespace())

    app = TemplatesApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(ChatbookTemplatesWindow)

        _assert_button_tooltips(
            window,
            {"use-template": "Select a Chatbook template before using it."},
        )

        await window._select_template("research_project")

        _assert_button_tooltips(
            window,
            {"use-template": "Create a Chatbook from the selected template."},
        )


@pytest.mark.asyncio
async def test_chatbook_export_toolbar_actions_explain_selected_pack_requirement(monkeypatch, tmp_path):
    async def no_refresh(self):
        return None

    async def no_load_details(self, index):
        return None

    monkeypatch.setattr(ChatbookExportManagementWindow, "refresh_chatbook_list", no_refresh)
    monkeypatch.setattr(ChatbookExportManagementWindow, "refresh_server_job_list", no_refresh)
    monkeypatch.setattr(ChatbookExportManagementWindow, "_load_chatbook_details", no_load_details)

    class ManagementApp(App):
        def __init__(self):
            super().__init__()
            self.config_data = {}
            self.notify = Mock()

        def compose(self) -> ComposeResult:
            yield ChatbookExportManagementWindow(self)

    app = ManagementApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(ChatbookExportManagementWindow)

        _assert_button_tooltips(
            window,
            {
                "delete-selected": "Select an exported Chatbook before deleting it.",
                "re-export": "Select an exported Chatbook before re-exporting it.",
                "share-selected": "Select an exported Chatbook before sharing it.",
                "open-location": "Select an exported Chatbook before opening its folder.",
            },
        )

        window.chatbook_files = [
            {
                "name": "Research Pack",
                "path": tmp_path / "Research Pack.chatbook.zip",
                "size": 1024,
                "created": datetime(2026, 4, 20, 9, 0),
                "modified": datetime(2026, 4, 20, 9, 0),
            }
        ]
        await window.on_option_list_option_selected(SimpleNamespace(option_id="0"))

        _assert_button_tooltips(
            window,
            {
                "delete-selected": "Delete the selected exported Chatbook.",
                "re-export": "Re-export the selected Chatbook.",
                "share-selected": "Share the selected Chatbook.",
                "open-location": "Open the selected Chatbook location.",
            },
        )


@pytest.mark.asyncio
async def test_chatbook_delete_selected_opens_confirmation_with_delete_labels(tmp_path):
    class ManagementApp:
        def __init__(self):
            self.config_data = {}
            self.notify = Mock()
            self.pushed_screen = None
            self.wait_for_dismiss = None

        async def push_screen(self, screen, wait_for_dismiss=False):
            self.pushed_screen = screen
            self.wait_for_dismiss = wait_for_dismiss
            return False

    app = ManagementApp()
    window = ChatbookExportManagementWindow(app)
    window.chatbook_files = [
        {
            "name": "Research Pack",
            "path": tmp_path / "Research Pack.chatbook.zip",
            "size": 1024,
            "created": datetime(2026, 4, 20, 9, 0),
            "modified": datetime(2026, 4, 20, 9, 0),
        }
    ]
    window.selected_chatbook = 0

    await window._delete_selected()

    assert isinstance(app.pushed_screen, ConfirmationDialog)
    assert app.wait_for_dismiss is True
    assert app.pushed_screen.confirm_label == "Delete"
    assert app.pushed_screen.cancel_label == "Cancel"


@pytest.mark.asyncio
async def test_chatbook_server_job_actions_explain_job_state(monkeypatch):
    async def no_refresh(self):
        return None

    monkeypatch.setattr(ChatbookExportManagementWindow, "refresh_chatbook_list", no_refresh)
    monkeypatch.setattr(ChatbookExportManagementWindow, "refresh_server_job_list", no_refresh)

    class ManagementApp(App):
        def __init__(self):
            super().__init__()
            self.config_data = {}
            self.notify = Mock()

        def compose(self) -> ComposeResult:
            yield ChatbookExportManagementWindow(self)

    app = ManagementApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(ChatbookExportManagementWindow)

        _assert_button_tooltips(
            window,
            {
                "cancel-server-job": "Select a running server Chatbook job before cancelling it.",
                "download-server-job": "Select a completed server export job before downloading it.",
                "remove-server-job": "Select a completed, failed, or cancelled server job before removing it.",
            },
        )

        window._select_server_job_record(
            {
                "source": "server",
                "job_type": "import",
                "job_id": "import-1",
                "status": "running",
            }
        )

        _assert_button_tooltips(
            window,
            {
                "cancel-server-job": "Cancel the selected running server Chatbook job.",
                "download-server-job": "Only completed server export jobs can be downloaded.",
                "remove-server-job": "Only completed, failed, or cancelled server jobs can be removed.",
            },
        )

        window._select_server_job_record(
            {
                "source": "server",
                "job_type": "export",
                "job_id": "export-1",
                "status": "completed",
            }
        )

        _assert_button_tooltips(
            window,
            {
                "cancel-server-job": "Completed server jobs cannot be cancelled.",
                "download-server-job": "Download the selected server Chatbook export.",
                "remove-server-job": "Remove the selected completed server job record.",
            },
        )
