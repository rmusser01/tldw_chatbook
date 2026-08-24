from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.notes_workspace_schemas import (
    WorkspaceSourceCreateRequest,
    WorkspaceSourceReorderRequest,
    WorkspaceSourceSelectionRequest,
    WorkspaceSourceUpdateRequest,
)


def source_row(**overrides):
    row = {
        "id": "source-1",
        "workspace_id": "workspace-1",
        "media_id": 12,
        "title": "Paper",
        "source_type": "pdf",
        "url": None,
        "position": 0,
        "selected": True,
        "added_at": "2026-08-24T00:00:00Z",
        "version": 3,
    }
    row.update(overrides)
    return row


def status_payload(**source_overrides):
    source = {
        "id": "source-1",
        "workspace_id": "workspace-1",
        "media_id": 12,
        "title": "Paper",
        "source_type": "pdf",
        "url": None,
        "selected": True,
        "state": "partially_queryable",
        "status_reason": "vector_index_pending",
        "readiness": {
            "metadata_ready": True,
            "text_extracted": True,
            "fts_ready": True,
            "vector_ready": False,
            "citation_ready": True,
            "summary_ready": False,
            "tool_accessible": False,
        },
        "progress_percent": 75.0,
        "job": {
            "id": 7,
            "uuid": "job-1",
            "status": "processing",
            "job_type": "media_ingest",
            "progress_percent": 75.0,
            "progress_message": "Indexing",
            "error_message": None,
        },
        "next_action": "wait_for_vector_index",
        "retry_eligible": False,
        "stale": False,
        "updated_at": "2026-08-24T00:00:00Z",
    }
    source.update(source_overrides)
    return {
        "workspace_id": "workspace-1",
        "sources": [source],
        "summary": {
            "total": 1,
            "selected": 1,
            "queryable": 0,
            "partially_queryable": 1,
            "processing": 0,
            "failed": 0,
            "missing": 0,
        },
    }


def preview_payload(**overrides):
    row = {
        "workspace_id": "workspace-1",
        "source_id": "source-1",
        "media_id": 12,
        "title": "Paper",
        "source_type": "pdf",
        "url": None,
        "state": "queryable",
        "status_reason": "source_queryable",
        "readiness": status_payload()["sources"][0]["readiness"]
        | {"vector_ready": True},
        "content_available": True,
        "preview_mode": "available",
        "unavailable_reason": None,
        "text_preview": "Bounded text",
        "text_total_chars": 12,
        "text_truncated": False,
        "snippets": [
            {
                "id": "snippet-1",
                "kind": "content_excerpt",
                "source_id": "source-1",
                "media_id": 12,
                "text": "Bounded text",
                "start_char": 0,
                "end_char": 12,
                "chunk_index": None,
                "chunk_uuid": None,
                "chunk_type": None,
            }
        ],
        "generated_at": "2026-08-24T00:00:00Z",
    }
    row.update(overrides)
    return row


def capabilities_payload(**overrides):
    row = {
        "workspace_id": "workspace-1",
        "workspace_profile": "research",
        "workspace_kind": "research_workspace",
        "access_level": "owner",
        "resolution": {"status": "complete", "partial_errors": []},
        "project_root": {"state": "not_configured"},
        "source_summary": status_payload()["summary"],
        "workspace_services": {
            "sources": {
                "state": "available",
                "reason_code": "available",
                "management_surface": None,
            }
        },
        "allowed_actions": {
            "source_list": {
                "allowed": True,
                "reason_code": "available",
            }
        },
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_source_paths_quote_every_opaque_segment_and_validate_responses(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(return_value=source_row(id="source/雪", workspace_id="ws/雪"))
    monkeypatch.setattr(client, "_request", request)

    result = await client.update_workspace_source(
        "ws/雪",
        "source/雪",
        WorkspaceSourceUpdateRequest(title="Renamed", version=3),
    )

    assert result["id"] == "source/雪"
    request.assert_awaited_once_with(
        "PUT",
        "/api/v1/workspaces/ws%2F%E9%9B%AA/sources/source%2F%E9%9B%AA",
        json_data={"title": "Renamed", "version": 3},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_id", ["", "   ", "x" * 1025])
async def test_source_paths_reject_invalid_workspace_id_before_dispatch(
    monkeypatch, workspace_id
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock()
    monkeypatch.setattr(client, "_request", request)

    with pytest.raises(ValueError, match="workspace_id"):
        await client.list_workspace_sources(workspace_id)

    request.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_list_rejects_oversized_owner_response_without_truncating(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(
        return_value=[source_row(id=f"source-{index}") for index in range(101)]
    )
    monkeypatch.setattr(client, "_request", request)

    with pytest.raises(ValidationError):
        await client.list_workspace_sources("workspace-1")


@pytest.mark.asyncio
async def test_source_delete_accepts_only_the_actual_empty_204_projection(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(return_value={})
    monkeypatch.setattr(client, "_request", request)

    assert await client.delete_workspace_source("workspace-1", "source-1") == {}

    request.return_value = {"deleted": True}
    with pytest.raises(ValidationError):
        await client.delete_workspace_source("workspace-1", "source-1")


def test_source_requests_reject_bool_as_int_and_oversized_or_duplicate_ids() -> None:
    with pytest.raises(ValidationError):
        WorkspaceSourceCreateRequest(
            id="source-1",
            media_id=True,
            title="Paper",
            source_type="pdf",
        )
    with pytest.raises(ValidationError):
        WorkspaceSourceSelectionRequest(selected_ids=["same", "same"])
    with pytest.raises(ValidationError):
        WorkspaceSourceReorderRequest(
            ordered_ids=[f"source-{index}" for index in range(101)]
        )


@pytest.mark.asyncio
async def test_selection_put_validates_ok_then_returns_refetched_rows(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(side_effect=[{"ok": True}, [source_row(version=5)]])
    monkeypatch.setattr(client, "_request", request)

    rows = await client.set_workspace_source_selection(
        "workspace-1", WorkspaceSourceSelectionRequest(selected_ids=["source-1"])
    )

    assert rows == [source_row(version=5)]
    assert [call.args[:2] for call in request.await_args_list] == [
        (
            "PUT",
            "/api/v1/workspaces/workspace-1/sources/selection",
        ),
        ("GET", "/api/v1/workspaces/workspace-1/sources"),
    ]
    assert request.await_args_list[0].kwargs == {
        "json_data": {"selected_ids": ["source-1"]}
    }


@pytest.mark.asyncio
async def test_selection_ok_false_stops_before_refetch(monkeypatch) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(return_value={"ok": False})
    monkeypatch.setattr(client, "_request", request)

    with pytest.raises(ValidationError):
        await client.set_workspace_source_selection(
            "workspace-1", WorkspaceSourceSelectionRequest(selected_ids=[])
        )

    assert request.await_count == 1


@pytest.mark.asyncio
async def test_reorder_put_then_refetch_reconciles_positions_and_versions(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    refetched = [
        source_row(id="source-2", position=0, version=8),
        source_row(id="source-1", position=1, version=7),
    ]
    request = AsyncMock(side_effect=[{"ok": True}, refetched])
    monkeypatch.setattr(client, "_request", request)

    result = await client.reorder_workspace_sources(
        "workspace-1",
        WorkspaceSourceReorderRequest(ordered_ids=["source-2", "source-1"]),
    )

    assert result == refetched
    assert [call.args[0] for call in request.await_args_list] == ["PUT", "GET"]


@pytest.mark.asyncio
async def test_post_write_refetch_failure_is_not_replaced_with_stale_rows(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(side_effect=[{"ok": True}, RuntimeError("offline")])
    monkeypatch.setattr(client, "_request", request)

    with pytest.raises(RuntimeError, match="offline"):
        await client.reorder_workspace_sources(
            "workspace-1",
            WorkspaceSourceReorderRequest(ordered_ids=["source-1"]),
        )

    assert request.await_count == 2


@pytest.mark.asyncio
async def test_preview_is_bounded_and_quotes_source_id(monkeypatch) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(return_value=preview_payload(source_id="source/1"))
    monkeypatch.setattr(client, "_request", request)

    result = await client.get_workspace_source_preview(
        "workspace-1", "source/1", max_chars=12000, chunk_limit=10
    )

    assert result["text_preview"] == "Bounded text"
    request.assert_awaited_once_with(
        "GET",
        "/api/v1/workspaces/workspace-1/sources/source%2F1/preview",
        params={"max_chars": 12000, "chunk_limit": 10},
    )


@pytest.mark.asyncio
async def test_preview_rejects_oversized_text_and_bool_bounds(monkeypatch) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(return_value=preview_payload(text_preview="x" * 12001))
    monkeypatch.setattr(client, "_request", request)

    with pytest.raises(ValidationError):
        await client.get_workspace_source_preview("workspace-1", "source-1")
    with pytest.raises(ValueError, match="max_chars"):
        await client.get_workspace_source_preview(
            "workspace-1", "source-1", max_chars=True
        )


@pytest.mark.asyncio
async def test_status_accepts_exact_lifecycle_and_rejects_unknown_or_oversized(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(return_value=status_payload())
    monkeypatch.setattr(client, "_request", request)

    assert (await client.get_workspace_source_status("workspace-1"))["sources"][0][
        "state"
    ] == "partially_queryable"

    request.return_value = status_payload(state="unknown_state")
    with pytest.raises(ValidationError):
        await client.get_workspace_source_status("workspace-1")
    oversized = status_payload()
    oversized["sources"] = [
        oversized["sources"][0] | {"id": f"source-{index}"} for index in range(101)
    ]
    request.return_value = oversized
    with pytest.raises(ValidationError):
        await client.get_workspace_source_status("workspace-1")


@pytest.mark.asyncio
async def test_status_rejects_oversized_job_text_and_malformed_summary(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(return_value=status_payload())
    monkeypatch.setattr(client, "_request", request)
    request.return_value["sources"][0]["job"]["progress_message"] = "x" * 1001

    with pytest.raises(ValidationError):
        await client.get_workspace_source_status("workspace-1")

    request.return_value = status_payload()
    request.return_value["summary"]["total"] = True
    with pytest.raises(ValidationError):
        await client.get_workspace_source_status("workspace-1")


@pytest.mark.asyncio
async def test_capabilities_are_bounded_and_malformed_projection_fails_closed(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    request = AsyncMock(return_value=capabilities_payload())
    monkeypatch.setattr(client, "_request", request)

    result = await client.get_workspace_capabilities("workspace-1")
    assert result["allowed_actions"]["source_list"]["allowed"] is True

    request.return_value = capabilities_payload(
        allowed_actions={"x" * 257: {"allowed": True, "reason_code": "available"}}
    )
    with pytest.raises(ValidationError):
        await client.get_workspace_capabilities("workspace-1")
