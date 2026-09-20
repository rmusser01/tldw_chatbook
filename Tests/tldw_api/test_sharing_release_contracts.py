"""Exercise current server contracts through real HTTP serialization and services."""

import json
from contextlib import asynccontextmanager

import httpx
import pytest

from tldw_chatbook.tldw_api import CloneWorkspaceRequest, TLDWAPIClient
from tldw_chatbook.tldw_api.exceptions import APIConnectionError, APIResponseError

OPERATION_ID = "898b8eb7-3b72-4d64-bc9e-4b2ac697353d"
KEY = "clone-request-2026-09-20"


def operation(status="queued"):
    """Use fields from the server SharedWorkspaceCloneOperationResponse contract."""
    return {
        "schema_version": 1,
        "operation_id": OPERATION_ID,
        "workspace_id": "copy-1",
        "command": "shared_workspace_clone",
        "status": status,
        "share_id": 7,
        "started_at": "2026-09-20T12:00:00Z",
        "updated_at": "2026-09-20T12:00:00Z",
        "retryable": False,
        "diagnostics": {},
        "poll_href": f"/api/v1/sharing/shared-with-me/7/clone/{OPERATION_ID}",
        "progress": {"phase": "queued", "percent": 0, "message_code": "clone_queued"}
        if status == "queued"
        else None,
        "result": None,
        "error": {
            "code": "clone_failed",
            "message_key": "clone.failed",
            "message": "Copy failed",
            "cleanup_state": "complete",
        }
        if status == "failed"
        else None,
    }


def source_page(offset=0, limit=1):
    return {
        "items": [
            {
                "source_id": f"source-{offset}",
                "title": "Source",
                "source_type": "media",
                "origin_url": "https://example.com/source",
                "origin_host": "example.com",
                "state": "ready",
                "reason_code": None,
                "citation_ready": True,
                "retrieval_ready": True,
                "position": offset,
                "added_at": None,
            }
        ],
        "pagination": {
            "offset": offset,
            "limit": limit,
            "total": 2,
            "has_more": offset == 0,
        },
        "summary": {"total": 2, "queryable": 2, "processing": 0, "failed": 0},
        "partial_errors": [
            {
                "area": "status",
                "code": "status_partial",
                "message": "Partial",
                "retryable": True,
            }
        ],
    }


@asynccontextmanager
async def http_client(handler):
    client = TLDWAPIClient("https://server.example")
    async with httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
        headers={"X-API-KEY": "test-credential"},
    ) as transport:
        client._client = transport
        yield client


@pytest.mark.asyncio
async def test_clone_request_replays_same_header_and_name_after_lost_response():
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadTimeout("lost accepted response", request=request)
        return httpx.Response(202, json=operation())

    request = CloneWorkspaceRequest(new_name="Copy")
    async with http_client(handler) as client:
        with pytest.raises(APIConnectionError):
            await client.clone_shared_workspace(7, request)
        receipt = await client.clone_shared_workspace(7, request)
    assert json.loads(requests[0].content) == {"name": "Copy"}
    assert requests[0].headers["Idempotency-Key"] == request.idempotency_key
    assert requests[1].headers["Idempotency-Key"] == request.idempotency_key
    assert "idempotency_key" not in json.loads(requests[0].content)
    assert receipt.operation_id == OPERATION_ID
    assert receipt.progress["phase"] == "queued"


@pytest.mark.asyncio
async def test_clone_operation_poll_uses_owner_scoped_route_and_preserves_failure():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=operation("failed"))

    async with http_client(handler) as client:
        receipt = await client.get_shared_workspace_clone_operation(7, OPERATION_ID)
    assert (
        requests[0].url.path == f"/api/v1/sharing/shared-with-me/7/clone/{OPERATION_ID}"
    )
    assert receipt.status == "failed"
    assert receipt.error["cleanup_state"] == "complete"
    assert receipt.poll_href.endswith(OPERATION_ID)


@pytest.mark.asyncio
async def test_source_page_retains_metadata_and_supports_filters():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=source_page(offset=1))

    async with http_client(handler) as client:
        page = await client.list_shared_workspace_source_page(
            7, offset=1, limit=1, q="Source", state="ready"
        )
    assert dict(requests[0].url.params) == {
        "offset": "1",
        "limit": "1",
        "q": "Source",
        "state": "ready",
    }
    assert page.items[0].source_id == page.items[0].id == "source-1"
    assert page.items[0].origin_url == page.items[0].url == "https://example.com/source"
    assert page.pagination.total == 2
    assert page.summary["queryable"] == 2
    assert page.partial_errors[0]["code"] == "status_partial"


@pytest.mark.asyncio
async def test_legacy_list_convenience_reads_every_server_page():
    offsets = []

    def handler(request):
        offset = int(request.url.params.get("offset", "0"))
        offsets.append(offset)
        return httpx.Response(200, json=source_page(offset))

    async with http_client(handler) as client:
        sources = await client.list_shared_workspace_sources(7)
    assert [source.id for source in sources] == ["source-0", "source-1"]
    assert offsets == [0, 1]


@pytest.mark.asyncio
async def test_notes_selected_link_preconditions_reach_delete_without_refetch():
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
    from tldw_chatbook.Notes.server_notes_workspace_service import (
        ServerNotesWorkspaceService,
    )

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            409, json={"detail": {"error_code": "notes_link_version_conflict"}}
        )

    async with http_client(handler) as client:
        scope = NotesScopeService(
            local_notes_service=None, server_service=ServerNotesWorkspaceService(client)
        )
        with pytest.raises(APIResponseError) as error:
            await scope.delete_note_link(
                scope="server_note",
                edge_id="e:edge-1",
                dataset_id="dataset-1",
                expected_version=4,
                idempotency_key="delete-selected-v4",
                reason="User removed link",
            )
    assert error.value.status_code == 409
    assert len(requests) == 1
    assert requests[0].method == "DELETE"
    assert dict(requests[0].url.params) == {
        "dataset_id": "dataset-1",
        "expected_version": "4",
        "idempotency_key": "delete-selected-v4",
        "reason": "User removed link",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["Sharing", "Sharing_Interop"])
async def test_sharing_service_families_preserve_clone_receipts_and_source_pages(
    family,
):
    from importlib import import_module

    module = import_module(f"tldw_chatbook.{family}.server_sharing_service")
    scope_module = import_module(
        f"tldw_chatbook.{family}."
        + (
            "server_sharing_scope_service"
            if family == "Sharing"
            else "sharing_scope_service"
        )
    )
    scope_class = getattr(
        scope_module,
        "ServerSharingScopeService" if family == "Sharing" else "SharingScopeService",
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=source_page(offset=1)
            if request.url.path.endswith("/sources")
            else operation(),
        )

    async with http_client(handler) as client:
        scope = scope_class(server_service=module.ServerSharingService(client))
        receipt = await scope.clone_shared_workspace(
            mode="server", share_id=7, new_name="Copy", idempotency_key=KEY
        )
        polled = await scope.get_shared_workspace_clone_operation(
            mode="server", share_id=7, operation_id=OPERATION_ID
        )
        page = await scope.list_shared_workspace_source_page(
            mode="server", share_id=7, offset=1, limit=1
        )
    assert requests[0].headers["Idempotency-Key"] == KEY
    assert json.loads(requests[0].content) == {"name": "Copy"}
    assert receipt["operation_id"] == polled["operation_id"] == OPERATION_ID
    assert page["items"][0]["source_id"] == "source-1"
    assert page["pagination"]["total"] == 2
    assert page["partial_errors"][0]["code"] == "status_partial"


@pytest.mark.parametrize(
    "key", ["short", "has spaces in this key", "key-with-ünicode-characters"]
)
def test_clone_rejects_invalid_idempotency_keys(key):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CloneWorkspaceRequest(name="Copy", idempotency_key=key)


@pytest.mark.parametrize("name", ["", " "])
def test_clone_rejects_blank_names(name):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CloneWorkspaceRequest(name=name)


def test_clone_success_retains_publication_and_readiness_details():
    from tldw_chatbook.tldw_api import CloneWorkspaceResponse

    payload = operation("succeeded")
    payload["result"] = {
        "schema_version": 1,
        "outcome": "partial",
        "workspace_id": "copy-1",
        "name": "Copy",
        "publication_confirmed": True,
        "counts": {
            **{
                f"{kind}_{count}": 0
                for kind in ("sources", "notes", "artifacts", "media")
                for count in ("attempted", "copied", "failed")
            },
            "sources_attempted": 2,
            "sources_copied": 1,
            "sources_failed": 1,
            "operation_owned_media_count": 0,
        },
        "readiness": {
            "text_search": "ready",
            "citations": "ready",
            "vector_search": "needs_indexing",
        },
        "warnings": [{"code": "source_skipped", "count": 1}],
    }
    assert CloneWorkspaceResponse.model_validate(payload).result == payload["result"]


@pytest.mark.asyncio
async def test_source_list_stops_if_server_pagination_cannot_advance():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json=source_page(0))

    async with http_client(handler) as client:
        with pytest.raises(ValueError, match="did not advance"):
            await client.list_shared_workspace_sources(7)
    assert count == 2


@pytest.mark.asyncio
async def test_legacy_source_list_and_url_remain_available():
    async with http_client(
        lambda request: httpx.Response(
            200,
            json=[
                {
                    "id": "legacy-1",
                    "workspace_id": "ws-1",
                    "url": "https://example.com/old",
                }
            ],
        )
    ) as client:
        sources = await client.list_shared_workspace_sources(7)
    assert sources[0].id == sources[0].source_id == "legacy-1"
    assert sources[0].url == sources[0].origin_url == "https://example.com/old"


@pytest.mark.asyncio
async def test_notes_missing_sync_version_remains_a_precondition_error():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            428, json={"detail": {"error_code": "notes_link_expected_version_required"}}
        )

    async with http_client(handler) as client:
        with pytest.raises(APIResponseError) as error:
            await client.delete_note_link("edge-1")
    assert error.value.status_code == 428
    assert len(requests) == 1
    assert not requests[0].url.query
