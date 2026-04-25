from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    OutputCreateRequest,
    OutputTemplateCreate,
    OutputTemplateUpdate,
    OutputUpdateRequest,
    TemplatePreviewRequest,
    TLDWAPIClient,
)


def _template_payload(template_id: int = 5) -> dict:
    return {
        "id": template_id,
        "user_id": "user-1",
        "name": "Newsletter",
        "type": "newsletter_markdown",
        "format": "md",
        "body": "# {{ title }}",
        "description": "Daily notes",
        "is_default": False,
        "created_at": "2026-04-24T12:00:00Z",
        "updated_at": "2026-04-24T12:00:00Z",
        "metadata": {"audience": "team"},
    }


def _artifact_payload(output_id: int = 9) -> dict:
    return {
        "id": output_id,
        "title": "Digest",
        "type": "newsletter_markdown",
        "format": "md",
        "storage_path": "digest.md",
        "media_item_id": None,
        "created_at": "2026-04-24T12:00:00Z",
        "workspace_tag": "workspace:alpha",
    }


@pytest.mark.asyncio
async def test_outputs_templates_client_routes_crud_and_preview(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"items": [_template_payload()], "total": 1},
            _template_payload(),
            _template_payload(6),
            {**_template_payload(6), "name": "Briefing"},
            {"success": True},
            {"rendered": "# Digest", "format": "md"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_output_templates(q="news", limit=25, offset=5)
    fetched = await client.get_output_template(5)
    created = await client.create_output_template(
        OutputTemplateCreate(
            name="Newsletter",
            type="newsletter_markdown",
            format="md",
            body="# {{ title }}",
        )
    )
    updated = await client.update_output_template(6, OutputTemplateUpdate(name="Briefing"))
    deleted = await client.delete_output_template(6)
    preview = await client.preview_output_template(
        5,
        TemplatePreviewRequest(template_id=5, data={"title": "Digest"}),
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/outputs/templates")
    assert mocked.await_args_list[0].kwargs["params"] == {"q": "news", "limit": 25, "offset": 5}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/outputs/templates/5")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/outputs/templates")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "name": "Newsletter",
        "type": "newsletter_markdown",
        "format": "md",
        "body": "# {{ title }}",
        "is_default": False,
    }
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/outputs/templates/6")
    assert mocked.await_args_list[3].kwargs["json_data"] == {"name": "Briefing"}
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/outputs/templates/6")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/outputs/templates/5/preview")
    assert mocked.await_args_list[5].kwargs["json_data"] == {"template_id": 5, "limit": 50, "data": {"title": "Digest"}}

    assert listed.total == 1
    assert fetched.id == 5
    assert created.id == 6
    assert updated.name == "Briefing"
    assert deleted["success"] is True
    assert preview.rendered == "# Digest"


@pytest.mark.asyncio
async def test_outputs_artifacts_client_routes_crud_and_purge(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"items": [_artifact_payload()], "total": 1, "page": 2, "size": 25},
            {"items": [_artifact_payload()], "total": 1, "page": 1, "size": 25},
            _artifact_payload(),
            _artifact_payload(10),
            {**_artifact_payload(), "title": "Digest 2"},
            {"success": True, "file_deleted": False},
            {"removed": 3, "files_deleted": 2},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_outputs(page=2, size=25, type="newsletter_markdown", workspace_tag="workspace:alpha")
    deleted_rows = await client.list_deleted_outputs(size=25)
    fetched = await client.get_output(9)
    created = await client.create_output(OutputCreateRequest(template_id=5, data={"title": "Digest"}))
    updated = await client.update_output(9, OutputUpdateRequest(title="Digest 2"))
    deleted = await client.delete_output(9, hard=True, delete_file=False)
    purged = await client.purge_outputs(delete_files=True, soft_deleted_grace_days=7)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/outputs")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "page": 2,
        "size": 25,
        "type": "newsletter_markdown",
        "workspace_tag": "workspace:alpha",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/outputs/deleted")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/outputs/9")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/outputs")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "template_id": 5,
        "data": {"title": "Digest"},
        "generate_mece": False,
        "generate_tts": False,
        "ingest_to_media_db": False,
    }
    assert mocked.await_args_list[4].args[:2] == ("PATCH", "/api/v1/outputs/9")
    assert mocked.await_args_list[4].kwargs["json_data"] == {"title": "Digest 2"}
    assert mocked.await_args_list[5].args[:2] == ("DELETE", "/api/v1/outputs/9")
    assert mocked.await_args_list[5].kwargs["params"] == {"hard": True, "delete_file": False}
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/outputs/purge")
    assert mocked.await_args_list[6].kwargs["json_data"] == {
        "delete_files": True,
        "soft_deleted_grace_days": 7,
        "include_retention": True,
    }

    assert listed.items[0].workspace_tag == "workspace:alpha"
    assert deleted_rows.total == 1
    assert fetched.id == 9
    assert created.id == 10
    assert updated.title == "Digest 2"
    assert deleted.file_deleted is False
    assert purged.removed == 3
