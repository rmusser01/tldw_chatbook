from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook import tldw_api as api


def _template_payload(template_id: int = 7) -> dict:
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


def _output_payload(output_id: int = 11) -> dict:
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


@pytest.mark.asyncio
async def test_client_routes_output_template_calls():
    client = api.TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(
        side_effect=[
            {"items": [_template_payload()], "total": 1},
            _template_payload(),
            _template_payload(),
            {**_template_payload(), "name": "Renamed Template"},
            {"success": True},
            {"rendered": "# Preview", "format": "md"},
        ]
    )

    listed = await client.list_output_templates(q="brief", limit=25, offset=5)
    created = await client.create_output_template(
        api.OutputTemplateCreateRequest(
            name="Weekly Briefing",
            type="briefing_markdown",
            format="md",
            body="# {{ job.name }}",
            description="Render a weekly markdown briefing",
            is_default=True,
            metadata={"category": "briefing"},
        )
    )
    fetched = await client.get_output_template(7)
    updated = await client.update_output_template(
        7,
        api.OutputTemplateUpdateRequest(name="Renamed Template"),
    )
    deleted = await client.delete_output_template(7)
    preview = await client.preview_output_template(
        7,
        api.OutputTemplatePreviewRequest(
            template_id=7,
            item_ids=[1, 2],
            limit=10,
        ),
    )

    assert listed.total == 1
    assert listed.items[0].id == 7
    assert created.name == "Weekly Briefing"
    assert fetched.id == 7
    assert updated.name == "Renamed Template"
    assert deleted == {"success": True}
    assert preview.rendered == "# Preview"
    assert [call.args for call in client._request.await_args_list] == [
        ("GET", "/api/v1/outputs/templates"),
        ("POST", "/api/v1/outputs/templates"),
        ("GET", "/api/v1/outputs/templates/7"),
        ("PATCH", "/api/v1/outputs/templates/7"),
        ("DELETE", "/api/v1/outputs/templates/7"),
        ("POST", "/api/v1/outputs/templates/7/preview"),
    ]
    assert client._request.await_args_list[0].kwargs["params"] == {"q": "brief", "limit": 25, "offset": 5}
    assert client._request.await_args_list[1].kwargs["json_data"] == {
        "name": "Weekly Briefing",
        "type": "briefing_markdown",
        "format": "md",
        "body": "# {{ job.name }}",
        "description": "Render a weekly markdown briefing",
        "is_default": True,
        "metadata": {"category": "briefing"},
    }
    assert client._request.await_args_list[3].kwargs["json_data"] == {"name": "Renamed Template"}
    assert client._request.await_args_list[5].kwargs["json_data"] == {
        "template_id": 7,
        "item_ids": [1, 2],
        "limit": 10,
    }


@pytest.mark.asyncio
async def test_client_routes_output_artifact_calls():
    client = api.TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(
        side_effect=[
            {"items": [_output_payload()], "total": 1, "page": 2, "size": 10},
            {"items": [_output_payload(12)], "total": 1, "page": 1, "size": 5},
            _output_payload(),
            _output_payload(),
            {**_output_payload(), "title": "Renamed Briefing"},
            {"success": True, "file_deleted": True},
        ]
    )

    listed = await client.list_outputs(
        page=2,
        size=10,
        run_id=77,
        type="briefing_markdown",
        workspace_tag="workspace:demo",
        include_deleted=True,
    )
    deleted_list = await client.list_deleted_outputs(page=1, size=5)
    created = await client.create_output(
        api.OutputCreateRequest(
            template_id=7,
            item_ids=[1, 2],
            title="Weekly Briefing",
            workspace_tag="workspace:demo",
            ingest_to_media_db=True,
        )
    )
    fetched = await client.get_output(11)
    updated = await client.update_output(
        11,
        api.OutputUpdateRequest(title="Renamed Briefing"),
    )
    deleted = await client.delete_output(11, hard=True, delete_file=True)

    assert listed.total == 1
    assert listed.items[0].id == 11
    assert deleted_list.items[0].id == 12
    assert created.id == 11
    assert fetched.title == "Weekly Briefing"
    assert updated.title == "Renamed Briefing"
    assert deleted == {"success": True, "file_deleted": True}
    assert [call.args for call in client._request.await_args_list] == [
        ("GET", "/api/v1/outputs"),
        ("GET", "/api/v1/outputs/deleted"),
        ("POST", "/api/v1/outputs"),
        ("GET", "/api/v1/outputs/11"),
        ("PATCH", "/api/v1/outputs/11"),
        ("DELETE", "/api/v1/outputs/11"),
    ]
    assert client._request.await_args_list[0].kwargs["params"] == {
        "page": 2,
        "size": 10,
        "run_id": 77,
        "type": "briefing_markdown",
        "workspace_tag": "workspace:demo",
        "include_deleted": True,
    }
    assert client._request.await_args_list[1].kwargs["params"] == {"page": 1, "size": 5}
    assert client._request.await_args_list[2].kwargs["json_data"] == {
        "template_id": 7,
        "item_ids": [1, 2],
        "title": "Weekly Briefing",
        "workspace_tag": "workspace:demo",
        "ingest_to_media_db": True,
    }
    assert client._request.await_args_list[4].kwargs["json_data"] == {"title": "Renamed Briefing"}
    assert client._request.await_args_list[5].kwargs["params"] == {"hard": True, "delete_file": True}


@pytest.mark.asyncio
async def test_client_routes_output_download_and_purge_calls():
    client = api.TLDWAPIClient("http://example.test", "token")
    client._request_bytes = AsyncMock(side_effect=[b"# briefing", b"<h1>Briefing</h1>"])
    client._request = AsyncMock(return_value={"removed": 2, "files_deleted": 1})

    downloaded = await client.download_output(11)
    by_name = await client.download_output_by_name("Weekly Briefing", format="html")
    purged = await client.purge_outputs(
        api.OutputsPurgeRequest(
            delete_files=True,
            soft_deleted_grace_days=7,
            include_retention=False,
        )
    )

    assert downloaded == b"# briefing"
    assert by_name == b"<h1>Briefing</h1>"
    assert purged == {"removed": 2, "files_deleted": 1}
    assert client._request_bytes.await_args_list[0].args[:2] == ("GET", "/api/v1/outputs/11/download")
    assert client._request_bytes.await_args_list[1].args[:2] == ("GET", "/api/v1/outputs/download/by-name")
    assert client._request_bytes.await_args_list[1].kwargs["params"] == {
        "title": "Weekly Briefing",
        "format": "html",
    }
    assert client._request.await_args.args[:2] == ("POST", "/api/v1/outputs/purge")
    assert client._request.await_args.kwargs["json_data"] == {
        "delete_files": True,
        "soft_deleted_grace_days": 7,
        "include_retention": False,
    }
