import pytest

from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable

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
async def test_management_window_lists_live_remote_server_jobs(monkeypatch):
    async def no_refresh(self):
        self.chatbook_files = []
        self.chatbook_count = 0
        self.total_size = 0
        self._update_list_count()
        self._update_status()

    class FakeClient:
        async def close(self):
            return None

    class FakeServerChatbookService:
        def __init__(self, client):
            self.client = client

        async def list_export_jobs(self, limit=100, offset=0):
            return {
                "jobs": [
                    {
                        "job_id": "remote-export-1",
                        "status": "completed",
                        "progress_percentage": 100,
                        "chatbook_name": "Remote Export",
                        "created_at": "2026-04-22T12:00:00Z",
                    }
                ],
                "total": 1,
            }

        async def list_import_jobs(self, limit=100, offset=0):
            return {
                "jobs": [
                    {
                        "job_id": "remote-import-1",
                        "status": "in_progress",
                        "progress_percentage": 40,
                        "chatbook_path": "/tmp/remote-import.chatbook.zip",
                        "created_at": "2026-04-22T12:05:00Z",
                    }
                ],
                "total": 1,
            }

    def fake_client_factory(config):
        assert config["tldw_api"]["base_url"] == "http://server.test"
        client = FakeClient()
        return FakeServerChatbookService(client), client

    monkeypatch.setattr(ChatbookExportManagementWindow, "refresh_chatbook_list", no_refresh)
    monkeypatch.setattr(
        "tldw_chatbook.UI.ChatbookExportManagementWindow.build_server_chatbook_service_from_config",
        fake_client_factory,
    )

    class ManagementApp(App):
        def __init__(self):
            super().__init__()
            self.config_data = {
                "tldw_api": {
                    "base_url": "http://server.test",
                    "api_key": "token",
                }
            }
            self._chatbook_server_jobs = []

        def compose(self) -> ComposeResult:
            yield ChatbookExportManagementWindow(self)

        def notify(self, *args, **kwargs):
            return None

    app = ManagementApp()
    async with app.run_test() as pilot:
        table = app.query_one("#server-job-table", DataTable)
        rows = [table.get_row_at(index) for index in range(table.row_count)]

        assert table.row_count == 2
        assert ["export", "completed", "100%", "Remote Export", "server"] in rows
        assert ["import", "in_progress", "40%", "remote-import.chatbook.zip", "server"] in rows


@pytest.mark.asyncio
async def test_management_window_remote_job_actions_call_server(monkeypatch, tmp_path):
    async def no_refresh(self):
        self.chatbook_files = []
        self.chatbook_count = 0
        self.total_size = 0
        self._update_list_count()
        self._update_status()

    actions = []

    class FakeClient:
        async def close(self):
            return None

    class FakeServerChatbookService:
        def __init__(self, client):
            self.client = client

        async def list_export_jobs(self, limit=100, offset=0):
            return {
                "jobs": [
                    {
                        "job_id": "remote-export-1",
                        "status": "completed",
                        "progress_percentage": 100,
                        "chatbook_name": "Remote Export",
                        "created_at": "2026-04-22T12:00:00Z",
                    }
                ],
                "total": 1,
            }

        async def list_import_jobs(self, limit=100, offset=0):
            return {
                "jobs": [
                    {
                        "job_id": "remote-import-1",
                        "status": "in_progress",
                        "progress_percentage": 40,
                        "chatbook_path": "/tmp/remote-import.chatbook.zip",
                        "created_at": "2026-04-22T12:05:00Z",
                    }
                ],
                "total": 1,
            }

        async def cancel_import_job(self, job_id):
            actions.append(("cancel_import", job_id))
            return {"success": True, "job_id": job_id, "message": "cancelled"}

        async def remove_export_job(self, job_id):
            actions.append(("remove_export", job_id))
            return {"success": True, "job_id": job_id, "message": "removed"}

        async def download_export_job(self, job_id, destination_path):
            actions.append(("download_export", job_id, Path(destination_path).name))
            Path(destination_path).write_bytes(b"zip")
            return Path(destination_path)

    monkeypatch.setattr(ChatbookExportManagementWindow, "refresh_chatbook_list", no_refresh)
    monkeypatch.setattr(
        "tldw_chatbook.UI.ChatbookExportManagementWindow.build_server_chatbook_service_from_config",
        lambda config: (FakeServerChatbookService(FakeClient()), FakeClient()),
    )

    class ManagementApp(App):
        def __init__(self):
            super().__init__()
            self.config_data = {"tldw_api": {"base_url": "http://server.test", "api_key": "token"}}
            self._chatbook_server_jobs = []
            self.notifications = []

        def compose(self) -> ComposeResult:
            yield ChatbookExportManagementWindow(self)

        def notify(self, message, **kwargs):
            self.notifications.append((message, kwargs))

    app = ManagementApp()
    async with app.run_test() as pilot:
        window = app.query_one(ChatbookExportManagementWindow)
        cancel_button = app.query_one("#cancel-server-job", Button)
        download_button = app.query_one("#download-server-job", Button)
        remove_button = app.query_one("#remove-server-job", Button)

        assert cancel_button.disabled is True
        assert download_button.disabled is True
        assert remove_button.disabled is True

        window._select_server_job_record(window.server_job_records[0])
        assert window.selected_server_job_record["job_id"] == "remote-import-1"
        assert cancel_button.disabled is False
        assert download_button.disabled is True
        assert remove_button.disabled is True

        await window._cancel_selected_server_job()
        assert actions == [("cancel_import", "remote-import-1")]

        window._select_server_job_record(window.server_job_records[1])
        assert window.selected_server_job_record["job_id"] == "remote-export-1"
        assert cancel_button.disabled is True
        assert download_button.disabled is False
        assert remove_button.disabled is False

        window.chatbooks_dir = tmp_path
        await window._download_selected_server_export()
        assert actions[-1] == ("download_export", "remote-export-1", "Remote Export.chatbook.zip")
        assert (tmp_path / "Remote Export.chatbook.zip").read_bytes() == b"zip"

        await window._remove_selected_server_job()
        assert actions[-1] == ("remove_export", "remote-export-1")

        window._select_server_job_record({
            "source": "server",
            "job_type": "export",
            "job_id": "failed-export",
            "status": "failed",
        })
        assert cancel_button.disabled is True
        assert download_button.disabled is True
        assert remove_button.disabled is False

        window._select_server_job_record({
            "source": "local",
            "job_type": "export",
            "job_id": "local-job",
            "status": "in_progress",
        })
        assert cancel_button.disabled is True
        assert download_button.disabled is True
        assert remove_button.disabled is True


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
