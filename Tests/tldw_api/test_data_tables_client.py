from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    DataTableColumnInput,
    DataTableContentUpdateRequest,
    DataTableGenerateRequest,
    DataTableGenerateResponse,
    DataTableRegenerateRequest,
    DataTableSourceInput,
    DataTableUpdateRequest,
    DataTablesListResponse,
    TLDWAPIClient,
)


def _summary(description: str | None = None) -> dict:
    return {
        "uuid": "table-1",
        "name": "Extracted entities",
        "description": description,
        "prompt": "extract entities",
        "status": "ready",
        "row_count": 1,
        "column_count": 1,
    }


def _detail() -> dict:
    return {
        "table": _summary(),
        "columns": [
            {
                "column_id": "col-name",
                "name": "Name",
                "type": "text",
                "position": 0,
            }
        ],
        "rows": [
            {
                "row_id": "row-1",
                "row_index": 0,
                "data": {"col-name": "Ada"},
            }
        ],
        "sources": [
            {
                "source_type": "document",
                "source_id": "doc-1",
                "title": "Source Doc",
            }
        ],
        "rows_limit": 25,
        "rows_offset": 0,
    }


@pytest.mark.asyncio
async def test_data_table_routes_wire_and_serialize_payloads(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"job_id": 101, "job_uuid": "job-101", "status": "queued", "table": _summary()},
            {"tables": [_summary()], "count": 1, "limit": 5, "offset": 10, "total": 1},
            _detail(),
            {
                "table_uuid": "table-1",
                "file_id": 77,
                "export": {"status": "ready", "format": "csv", "url": "/download/77", "bytes": 123},
            },
            _detail(),
            _summary(description="Updated"),
            {"job_id": 102, "job_uuid": "job-102", "status": "queued", "table": _summary()},
            {
                "id": 102,
                "uuid": "job-102",
                "status": "queued",
                "job_type": "data_table_generate",
                "owner_user_id": "1",
                "created_at": "2026-04-25T00:00:00Z",
                "started_at": None,
                "completed_at": None,
                "cancelled_at": None,
                "cancellation_reason": None,
                "progress_percent": 0,
                "progress_message": "queued",
                "result": None,
                "error_message": None,
                "table_uuid": "table-1",
            },
            {"success": True, "job_id": 102, "status": "cancelled", "message": "Job cancellation requested"},
            {"success": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    generated = await client.generate_data_table(
        DataTableGenerateRequest(
            name="Extracted entities",
            prompt="extract entities",
            sources=[DataTableSourceInput(source_type="document", source_id="doc-1")],
            max_rows=25,
        ),
        wait_for_completion=True,
        wait_timeout_seconds=30,
    )
    listed = await client.list_data_tables(status_filter="ready", search="entities", workspace_tag="workspace:one", limit=5, offset=10)
    detail = await client.get_data_table("table-1", rows_limit=25, rows_offset=5, include_rows=False, include_sources=True)
    exported = await client.export_data_table("table-1", format="csv", async_mode="sync", mode="url", download=False)
    await client.update_data_table_content(
        "table-1",
        DataTableContentUpdateRequest(
            columns=[DataTableColumnInput(name="Name", type="text")],
            rows=[{"Name": "Ada"}],
        ),
    )
    patched = await client.update_data_table("table-1", DataTableUpdateRequest(description="Updated"))
    regenerated = await client.regenerate_data_table("table-1", DataTableRegenerateRequest(prompt="refresh"))
    job = await client.get_data_table_job(102)
    cancelled = await client.cancel_data_table_job(102, reason="no longer needed")
    deleted = await client.delete_data_table("table-1")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/data-tables/generate")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "wait_for_completion": "true",
        "wait_timeout_seconds": 30,
    }
    assert mocked.await_args_list[0].kwargs["json_data"]["sources"] == [{"source_type": "document", "source_id": "doc-1"}]
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/data-tables")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "status": "ready",
        "search": "entities",
        "workspace_tag": "workspace:one",
        "limit": 5,
        "offset": 10,
    }
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/data-tables/table-1")
    assert mocked.await_args_list[2].kwargs["params"] == {
        "rows_limit": 25,
        "rows_offset": 5,
        "include_rows": "false",
        "include_sources": "true",
    }
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/data-tables/table-1/export")
    assert mocked.await_args_list[3].kwargs["params"] == {
        "format": "csv",
        "async_mode": "sync",
        "mode": "url",
        "download": "false",
    }
    assert mocked.await_args_list[4].args[:2] == ("PUT", "/api/v1/data-tables/table-1/content")
    assert mocked.await_args_list[4].kwargs["json_data"] == {
        "columns": [{"name": "Name", "type": "text"}],
        "rows": [{"Name": "Ada"}],
    }
    assert mocked.await_args_list[5].args[:2] == ("PATCH", "/api/v1/data-tables/table-1")
    assert mocked.await_args_list[5].kwargs["json_data"] == {"description": "Updated"}
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/data-tables/table-1/regenerate")
    assert mocked.await_args_list[6].kwargs["json_data"] == {"prompt": "refresh"}
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/data-tables/jobs/102")
    assert mocked.await_args_list[8].args[:2] == ("DELETE", "/api/v1/data-tables/jobs/102")
    assert mocked.await_args_list[8].kwargs["params"] == {"reason": "no longer needed"}
    assert mocked.await_args_list[9].args[:2] == ("DELETE", "/api/v1/data-tables/table-1")
    assert isinstance(generated, DataTableGenerateResponse)
    assert isinstance(listed, DataTablesListResponse)
    assert detail.table.uuid == "table-1"
    assert exported.file_id == 77
    assert patched.description == "Updated"
    assert regenerated.job_id == 102
    assert job.table_uuid == "table-1"
    assert cancelled.success is True
    assert deleted.success is True
