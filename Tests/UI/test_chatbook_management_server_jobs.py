import pytest

from textual.app import App, ComposeResult
from textual.widgets import DataTable

from tldw_chatbook.UI.ChatbookExportManagementWindow import ChatbookExportManagementWindow
from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved


@pytest.mark.asyncio
async def test_management_window_lists_server_jobs_from_app_state(monkeypatch):
    async def no_refresh(self):
        self.chatbook_files = []
        self.chatbook_count = 0
        self.total_size = 0
        self._update_list_count()
        self._update_status()

    monkeypatch.setattr(ChatbookExportManagementWindow, "refresh_chatbook_list", no_refresh)

    class ManagementApp(App):
        def __init__(self):
            super().__init__()
            self._chatbook_server_jobs = [
                {
                    "job_type": "export",
                    "job_id": "job-1",
                    "status": "completed",
                    "progress_percentage": 100,
                    "chatbook_name": "Pack One",
                    "recorded_at": "2026-04-19T12:00:00Z",
                },
                {
                    "job_type": "import",
                    "job_id": "job-2",
                    "status": "running",
                    "progress_percentage": 65,
                    "chatbook_name": "Pack Two",
                    "recorded_at": "2026-04-19T12:05:00Z",
                },
            ]

        def compose(self) -> ComposeResult:
            yield ChatbookExportManagementWindow(self)

        def notify(self, *args, **kwargs):
            return None

    app = ManagementApp()
    async with app.run_test() as pilot:
        table = app.query_one("#server-job-table", DataTable)
        assert table.row_count == 2
        first_row = table.get_row_at(0)
        assert first_row[0] == "import"
        assert first_row[1] == "running"
        assert first_row[2] == "65%"


@pytest.mark.asyncio
async def test_manage_exports_action_pushes_management_window(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def __init__(self):
            super().__init__()
            self.pushed = None

        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

        async def push_screen(self, screen, wait_for_dismiss=False):
            self.pushed = screen
            return None

        def notify(self, *args, **kwargs):
            return None

    app = ChatbooksWindowApp()
    async with app.run_test() as pilot:
        window = app.query_one(ChatbooksWindowImproved)
        await window.action_manage_exports()
        assert isinstance(app.pushed, ChatbookExportManagementWindow)
