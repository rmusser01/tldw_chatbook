from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Widgets.chunking_templates_widget import ChunkingTemplatesWidget


def _normalized_template(backend: str, name: str, *, is_builtin: bool = False) -> dict:
    return {
        "record_id": f"{backend}:chunking_template:{name}",
        "backend": backend,
        "backing_template_name": name,
        "name": name,
        "description": f"{name} description",
        "template_json": '{"chunking": {"method": "words", "config": {"max_size": 256}}}',
        "template": {"chunking": {"method": "words", "config": {"max_size": 256}}},
        "is_builtin": is_builtin,
        "tags": [],
        "created_at": "2026-04-20T00:00:00",
        "updated_at": "2026-04-20T00:00:00",
        "version": 1,
        "backing_id": name,
    }


class FakeScopeService:
    def __init__(self, *, local_templates=None, server_templates=None):
        self.local_templates = list(local_templates or [])
        self.server_templates = list(server_templates or [])
        self.calls = []

    async def list_templates(self, *, mode, **kwargs):
        self.calls.append(("list_templates", mode, kwargs))
        return self.server_templates if mode == "server" else self.local_templates


class _ChunkingTemplatesTestApp(App):
    def __init__(self, *, runtime_backend: str = "local", scope_service=None):
        super().__init__()
        self.current_runtime_backend = runtime_backend
        self.rag_admin_scope_service = scope_service
        self.notify = Mock()
        self.push_screen = Mock()
        self.push_screen_wait = Mock()
        self.media_db = None
        self.widget = ChunkingTemplatesWidget(app_instance=self)

    def compose(self) -> ComposeResult:
        yield self.widget


@pytest.mark.asyncio
async def test_chunking_templates_widget_reads_server_templates_when_runtime_backend_is_server():
    scope = FakeScopeService(server_templates=[_normalized_template("server", "server-demo")])
    app = _ChunkingTemplatesTestApp(runtime_backend="server", scope_service=scope)

    async with app.run_test() as pilot:
        await pilot.pause()
        widget = app.widget

        assert "server-demo" in widget.templates_cache_by_name
        assert widget.templates_cache_by_name["server-demo"]["backend"] == "server"
        assert scope.calls == [("list_templates", "server", {"include_builtin": True, "include_custom": True})]


@pytest.mark.asyncio
async def test_chunking_templates_widget_disables_builtin_edit_actions():
    scope = FakeScopeService(local_templates=[_normalized_template("local", "general", is_builtin=True)])
    app = _ChunkingTemplatesTestApp(runtime_backend="local", scope_service=scope)

    async with app.run_test() as pilot:
        await pilot.pause()
        widget = app.widget
        widget.selected_template_record_id = "local:chunking_template:general"
        await pilot.pause()

        assert widget.query_one("#edit-template-btn", Button).disabled is True
        assert widget.query_one("#delete-template-btn", Button).disabled is True
