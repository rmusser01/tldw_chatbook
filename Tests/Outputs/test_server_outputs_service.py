from __future__ import annotations

import importlib
from typing import Any

import pytest


class FakePolicyEnforcer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def list_output_templates(self, *, q=None, limit=50, offset=0):
        self.calls.append(("list_output_templates", q, limit, offset))
        return {
            "items": [
                {
                    "id": 7,
                    "user_id": "3",
                    "name": "Weekly Briefing",
                    "type": "briefing_markdown",
                    "format": "md",
                    "body": "# {{ job.name }}",
                    "description": "Render a weekly markdown briefing",
                    "is_default": True,
                    "created_at": "2026-04-23T20:00:00Z",
                    "updated_at": "2026-04-23T20:30:00Z",
                    "metadata": {"category": "briefing"},
                }
            ],
            "total": 1,
        }

    async def create_output_template(self, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("create_output_template", payload))
        return {
            "id": 7,
            "user_id": "3",
            **payload,
            "created_at": "2026-04-23T20:00:00Z",
            "updated_at": "2026-04-23T20:30:00Z",
        }

    async def get_output_template(self, template_id):
        self.calls.append(("get_output_template", template_id))
        return {
            "id": template_id,
            "user_id": "3",
            "name": "Weekly Briefing",
            "type": "briefing_markdown",
            "format": "md",
            "body": "# {{ job.name }}",
            "description": "Render a weekly markdown briefing",
            "is_default": True,
            "created_at": "2026-04-23T20:00:00Z",
            "updated_at": "2026-04-23T20:30:00Z",
            "metadata": {"category": "briefing"},
        }

    async def update_output_template(self, template_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("update_output_template", template_id, payload))
        return {
            "id": template_id,
            "user_id": "3",
            "name": payload.get("name", "Weekly Briefing"),
            "type": "briefing_markdown",
            "format": "md",
            "body": "# {{ job.name }}",
            "description": "Render a weekly markdown briefing",
            "is_default": True,
            "created_at": "2026-04-23T20:00:00Z",
            "updated_at": "2026-04-23T20:45:00Z",
            "metadata": {"category": "briefing"},
        }

    async def delete_output_template(self, template_id):
        self.calls.append(("delete_output_template", template_id))
        return {"success": True}

    async def preview_output_template(self, template_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("preview_output_template", template_id, payload))
        return {"rendered": "# Preview", "format": "md"}

    async def list_outputs(self, **kwargs):
        self.calls.append(("list_outputs", kwargs))
        return {
            "items": [
                {
                    "id": 11,
                    "title": "Weekly Briefing",
                    "type": "briefing_markdown",
                    "format": "md",
                    "storage_path": "weekly-briefing_20260423_200000.md",
                    "media_item_id": 42,
                    "created_at": "2026-04-23T20:00:00Z",
                    "workspace_tag": "workspace:demo",
                }
            ],
            "total": 1,
            "page": kwargs.get("page", 1),
            "size": kwargs.get("size", 50),
        }

    async def list_deleted_outputs(self, *, page=1, size=50):
        self.calls.append(("list_deleted_outputs", page, size))
        return {
            "items": [
                {
                    "id": 12,
                    "title": "Deleted Briefing",
                    "type": "briefing_markdown",
                    "format": "md",
                    "storage_path": "deleted-briefing_20260423_200001.md",
                    "media_item_id": None,
                    "created_at": "2026-04-23T20:01:00Z",
                    "workspace_tag": "workspace:demo",
                }
            ],
            "total": 1,
            "page": page,
            "size": size,
        }

    async def create_output(self, request_data):
        payload = request_data.model_dump(exclude_none=True, exclude_defaults=True, mode="json")
        self.calls.append(("create_output", payload))
        return {
            "id": 11,
            "title": payload.get("title", "Weekly Briefing"),
            "type": "briefing_markdown",
            "format": "md",
            "storage_path": "weekly-briefing_20260423_200000.md",
            "media_item_id": 42,
            "created_at": "2026-04-23T20:00:00Z",
            "workspace_tag": payload.get("workspace_tag"),
        }

    async def get_output(self, output_id):
        self.calls.append(("get_output", output_id))
        return {
            "id": output_id,
            "title": "Weekly Briefing",
            "type": "briefing_markdown",
            "format": "md",
            "storage_path": "weekly-briefing_20260423_200000.md",
            "media_item_id": 42,
            "created_at": "2026-04-23T20:00:00Z",
            "workspace_tag": "workspace:demo",
        }

    async def update_output(self, output_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("update_output", output_id, payload))
        return {
            "id": output_id,
            "title": payload.get("title", "Weekly Briefing"),
            "type": "briefing_markdown",
            "format": payload.get("format", "md"),
            "storage_path": "weekly-briefing_20260423_200000.md",
            "media_item_id": 42,
            "created_at": "2026-04-23T20:00:00Z",
            "workspace_tag": "workspace:demo",
        }

    async def delete_output(self, output_id, *, hard=False, delete_file=False):
        self.calls.append(("delete_output", output_id, hard, delete_file))
        return {"success": True, "file_deleted": bool(delete_file)}

    async def download_output(self, output_id):
        self.calls.append(("download_output", output_id))
        return b"# briefing"

    async def download_output_by_name(self, title, *, format=None):
        self.calls.append(("download_output_by_name", title, format))
        return b"<h1>Briefing</h1>"

    async def purge_outputs(self, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("purge_outputs", payload))
        return {"removed": 2, "files_deleted": 1}


@pytest.mark.asyncio
async def test_server_outputs_service_routes_typed_client_calls():
    outputs_module = importlib.import_module("tldw_chatbook.Outputs")
    client = FakeClient()
    service = outputs_module.ServerOutputsService(client=client)

    templates = await service.list_output_templates(q="brief", limit=25, offset=5)
    created_template = await service.create_output_template(
        name="Weekly Briefing",
        type="briefing_markdown",
        format="md",
        body="# {{ job.name }}",
        description="Render a weekly markdown briefing",
        is_default=True,
        metadata={"category": "briefing"},
    )
    preview = await service.preview_output_template(7, item_ids=[1, 2], limit=10)
    outputs = await service.list_outputs(page=2, size=10, run_id=77, workspace_tag="workspace:demo")
    deleted_outputs = await service.list_deleted_outputs(page=1, size=5)
    created_output = await service.create_output(
        template_id=7,
        item_ids=[1, 2],
        title="Weekly Briefing",
        workspace_tag="workspace:demo",
        ingest_to_media_db=True,
    )
    fetched_output = await service.get_output(11)
    updated_output = await service.update_output(11, title="Renamed Briefing")
    deleted_output = await service.delete_output(11, hard=True, delete_file=True)
    downloaded_output = await service.download_output(11)
    downloaded_by_name = await service.download_output_by_name("Weekly Briefing", format="html")
    purged_outputs = await service.purge_outputs(
        delete_files=True,
        soft_deleted_grace_days=7,
        include_retention=False,
    )

    assert templates["total"] == 1
    assert created_template["id"] == 7
    assert preview["rendered"] == "# Preview"
    assert outputs["items"][0]["id"] == 11
    assert deleted_outputs["items"][0]["id"] == 12
    assert created_output["workspace_tag"] == "workspace:demo"
    assert fetched_output["id"] == 11
    assert updated_output["title"] == "Renamed Briefing"
    assert deleted_output["file_deleted"] is True
    assert downloaded_output == b"# briefing"
    assert downloaded_by_name == b"<h1>Briefing</h1>"
    assert purged_outputs == {"removed": 2, "files_deleted": 1}
    assert client.calls == [
        ("list_output_templates", "brief", 25, 5),
        ("create_output_template", {"name": "Weekly Briefing", "type": "briefing_markdown", "format": "md", "body": "# {{ job.name }}", "description": "Render a weekly markdown briefing", "is_default": True, "metadata": {"category": "briefing"}}),
        ("preview_output_template", 7, {"template_id": 7, "item_ids": [1, 2], "limit": 10}),
        ("list_outputs", {"page": 2, "size": 10, "run_id": 77, "workspace_tag": "workspace:demo"}),
        ("list_deleted_outputs", 1, 5),
        ("create_output", {"template_id": 7, "item_ids": [1, 2], "title": "Weekly Briefing", "workspace_tag": "workspace:demo", "ingest_to_media_db": True}),
        ("get_output", 11),
        ("update_output", 11, {"title": "Renamed Briefing"}),
        ("delete_output", 11, True, True),
        ("download_output", 11),
        ("download_output_by_name", "Weekly Briefing", "html"),
        (
            "purge_outputs",
            {"delete_files": True, "soft_deleted_grace_days": 7, "include_retention": False},
        ),
    ]


@pytest.mark.asyncio
async def test_outputs_scope_service_enforces_policy_and_normalizes_server_ids():
    outputs_module = importlib.import_module("tldw_chatbook.Outputs")
    policy = FakePolicyEnforcer()
    scope = outputs_module.ServerOutputsScopeService(
        server_service=outputs_module.ServerOutputsService(client=FakeClient()),
        policy_enforcer=policy,
    )

    templates = await scope.list_output_templates(mode="server", q="brief", limit=25, offset=5)
    created_template = await scope.create_output_template(
        mode="server",
        name="Weekly Briefing",
        type="briefing_markdown",
        format="md",
        body="# {{ job.name }}",
        description="Render a weekly markdown briefing",
        is_default=True,
        metadata={"category": "briefing"},
    )
    preview = await scope.preview_output_template(mode="server", template_id=7, item_ids=[1, 2], limit=10)
    outputs = await scope.list_outputs(mode="server", page=2, size=10, run_id=77, workspace_tag="workspace:demo")
    deleted_outputs = await scope.list_deleted_outputs(mode="server", page=1, size=5)
    created_output = await scope.create_output(
        mode="server",
        template_id=7,
        item_ids=[1, 2],
        title="Weekly Briefing",
        workspace_tag="workspace:demo",
        ingest_to_media_db=True,
    )
    fetched_output = await scope.get_output(mode="server", output_id=11)
    updated_output = await scope.update_output(mode="server", output_id=11, title="Renamed Briefing")
    deleted_output = await scope.delete_output(mode="server", output_id=11, hard=True, delete_file=True)
    downloaded_output = await scope.download_output(mode="server", output_id=11)
    downloaded_by_name = await scope.download_output_by_name(mode="server", title="Weekly Briefing", format="html")
    purged_outputs = await scope.purge_outputs(
        mode="server",
        delete_files=True,
        soft_deleted_grace_days=7,
        include_retention=False,
    )

    assert templates["items"][0]["id"] == "server:output_template:7"
    assert templates["entity_kind"] == "output_template_list"
    assert created_template["id"] == "server:output_template:7"
    assert preview["entity_kind"] == "output_template_preview"
    assert outputs["items"][0]["id"] == "server:output:11"
    assert deleted_outputs["items"][0]["id"] == "server:output:12"
    assert created_output["id"] == "server:output:11"
    assert created_output["entity_kind"] == "output_render_result"
    assert fetched_output["id"] == "server:output:11"
    assert updated_output["id"] == "server:output:11"
    assert deleted_output["entity_kind"] == "output_delete"
    assert downloaded_output == b"# briefing"
    assert downloaded_by_name == b"<h1>Briefing</h1>"
    assert purged_outputs["entity_kind"] == "output_purge"
    assert purged_outputs["removed"] == 2
    assert purged_outputs["files_deleted"] == 1
    assert policy.calls == [
        "outputs.templates.list.server",
        "outputs.templates.create.server",
        "outputs.templates.detail.server",
        "outputs.artifacts.list.server",
        "outputs.artifacts.list.server",
        "outputs.render_jobs.launch.server",
        "outputs.artifacts.detail.server",
        "outputs.artifacts.update.server",
        "outputs.artifacts.delete.server",
        "outputs.artifacts.detail.server",
        "outputs.artifacts.detail.server",
        "outputs.artifacts.delete.server",
    ]


@pytest.mark.asyncio
async def test_outputs_scope_service_rejects_local_mode_before_policy_dispatch():
    outputs_module = importlib.import_module("tldw_chatbook.Outputs")
    policy = FakePolicyEnforcer()
    scope = outputs_module.ServerOutputsScopeService(
        server_service=object(),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Server outputs require server mode"):
        await scope.list_output_templates(mode="local")

    assert policy.calls == []
