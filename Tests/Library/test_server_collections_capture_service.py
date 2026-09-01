"""Server Reading API adapter for the Collections capture reader."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest

from tldw_chatbook.Library.collections_capture_models import (
    CapabilityState,
    CaptureIdentity,
    CapturePageRequest,
    CaptureSaveRequest,
    CollectionsCaptureError,
    ExternalNoteReference,
)
from tldw_chatbook.Library.collections_capture_service import (
    build_server_capture_authority,
)
from tldw_chatbook.Library.server_collections_capture_service import (
    SERVER_SORT,
    ServerCollectionsCaptureService,
)
from tldw_chatbook.tldw_api.exceptions import APIConnectionError, APIResponseError


def _item(index: int, **changes: Any) -> dict[str, Any]:
    item = {
        "id": index,
        "media_id": 1000 + index,
        "title": f"Server Capture {index:03d}",
        "url": f"https://example.test/{index:03d}",
        "canonical_url": f"https://example.test/{index:03d}",
        "domain": "example.test",
        "summary": f"Summary {index:03d}",
        "notes": f"Note {index:03d}",
        "published_at": "2026-08-01T00:00:00Z",
        "status": "saved",
        "processing_status": "ready",
        "favorite": False,
        "tags": ["Research"],
        "created_at": f"2026-08-01T00:{index % 60:02d}:00Z",
        "updated_at": f"2026-08-02T00:{index % 60:02d}:00Z",
        "revision": index + 1,
        "text": f"Body {index:03d}",
        "clean_html": None,
    }
    item.update(changes)
    return item


class FakeReadingClient:
    def __init__(self) -> None:
        self.items = [_item(index) for index in range(1, 46)]
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failures: dict[str, BaseException] = {}
        self.note_links: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def _fail(self, method: str) -> None:
        failure = self.failures.get(method)
        if failure is not None:
            raise failure

    async def list_reading_items(self, **params: Any) -> dict[str, Any]:
        self._fail("list_reading_items")
        self.calls.append(("list_reading_items", dict(params)))
        page = int(params["page"])
        size = int(params["size"])
        offset = (page - 1) * size
        return {
            "items": self.items[offset : offset + size],
            "total": len(self.items),
            "page": page,
            "size": size,
            "source_revision": "server-snapshot-1",
        }

    async def get_reading_item(self, item_id: int) -> dict[str, Any]:
        self._fail("get_reading_item")
        return next(item for item in self.items if item["id"] == item_id)

    async def save_reading_item(self, request_data: Any) -> dict[str, Any]:
        self._fail("save_reading_item")
        payload = request_data.model_dump(mode="json")
        self.calls.append(("save_reading_item", payload))
        saved = _item(
            99, title=payload.get("title") or "Saved", url=str(payload["url"])
        )
        self.items.append(saved)
        return saved

    async def update_reading_item(
        self, item_id: int, request_data: Any
    ) -> dict[str, Any]:
        self._fail("update_reading_item")
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("update_reading_item", payload))
        item = next(item for item in self.items if item["id"] == item_id)
        item.update(payload)
        item["revision"] += 1
        return dict(item)

    async def list_reading_saved_searches(self, **_params: Any) -> dict[str, Any]:
        return {"items": [], "total": 0, "limit": 20, "offset": 0}

    async def list_reading_highlights(self, _item_id: int) -> list[Any]:
        self._fail("list_reading_highlights")
        return []

    async def list_reading_item_note_links(self, item_id: int) -> dict[str, Any]:
        return {"item_id": item_id, "links": list(self.note_links[item_id])}

    async def link_note_to_reading_item(
        self,
        item_id: int,
        note_id: str,
    ) -> dict[str, Any]:
        link = {
            "item_id": item_id,
            "note_id": note_id,
            "created_at": "2026-09-01T12:00:00Z",
        }
        self.note_links[item_id].append(link)
        return link


def _service(
    client: FakeReadingClient,
    docs: dict[str, Any],
) -> tuple[Any, ServerCollectionsCaptureService]:
    authority = build_server_capture_authority("server-a", "user-a")
    service = ServerCollectionsCaptureService(
        authority,
        client,
        docs_info_provider=lambda: docs,
        credential_fingerprint="credential-a",
    )
    return authority, service


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attestation", "state"),
    [
        (True, CapabilityState.SUPPORTED),
        (False, CapabilityState.UNSUPPORTED),
        (None, CapabilityState.UNSUPPORTED),
        ("true", CapabilityState.UNSUPPORTED),
        (1, CapabilityState.UNSUPPORTED),
    ],
)
async def test_server_browse_requires_exact_snapshot_attestation(
    attestation: Any,
    state: CapabilityState,
) -> None:
    capabilities = {}
    if attestation is not None:
        capabilities["hasReadingSnapshotPagesV1"] = attestation
    _authority, service = _service(
        FakeReadingClient(),
        {"api_version": "1", "capabilities": capabilities},
    )

    observed = await service.capabilities()

    browse = observed.for_action("browse")
    assert browse.state is state
    if state is CapabilityState.UNSUPPORTED:
        assert browse.reason == "server_page_snapshot_unavailable"


@pytest.mark.asyncio
async def test_server_maps_fixed_pages_and_source_neutral_sorts() -> None:
    client = FakeReadingClient()
    authority, service = _service(
        client,
        {
            "api_version": "1",
            "capabilities": {"hasReadingSnapshotPagesV1": True},
        },
    )

    page_two = await service.list_page(
        CapturePageRequest(
            authority.key,
            search="research",
            statuses=("saved",),
            favorite=False,
            tags=("research",),
            sort="saved_desc",
            page=2,
        )
    )
    page_three = await service.list_page(CapturePageRequest(authority.key, page=3))

    assert page_two.total == 45
    assert len(page_two.items) == 20
    assert len(page_three.items) == 5
    assert page_two.source_revision == "server-snapshot-1"
    call = client.calls[0][1]
    assert call == {
        "status": ["saved"],
        "tags": ["research"],
        "q": "research",
        "favorite": False,
        "page": 2,
        "size": 20,
        "sort": SERVER_SORT["saved_desc"],
    }
    assert all(item.identity.authority_key == authority.key for item in page_two.items)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed",
    [
        {"id": None},
        {"favorite": "yes"},
        {"status": "private-server-state"},
        {"processing_status": "private-server-state"},
    ],
)
async def test_server_rows_fail_closed_instead_of_coercing_malformed_data(
    malformed: dict[str, Any],
) -> None:
    client = FakeReadingClient()
    client.items = [_item(1, **malformed)]
    authority, service = _service(
        client,
        {
            "api_version": "1",
            "capabilities": {"hasReadingSnapshotPagesV1": True},
        },
    )

    with pytest.raises(CollectionsCaptureError) as caught:
        await service.list_page(CapturePageRequest(authority.key))

    assert caught.value.reason == "invalid_server_response"
    assert "private" not in str(caught.value)


@pytest.mark.asyncio
async def test_server_save_distinguishes_unknown_transport_from_response_failure() -> (
    None
):
    client = FakeReadingClient()
    authority, service = _service(
        client,
        {
            "api_version": "1",
            "capabilities": {"hasReadingSnapshotPagesV1": True},
        },
    )
    request = CaptureSaveRequest(authority.key, "https://example.test/new")
    client.failures["save_reading_item"] = APIConnectionError("private transport")

    unknown = await service.save_capture(request)

    assert unknown.outcome_unknown is True
    assert unknown.capture is None
    client.failures["save_reading_item"] = APIResponseError(
        422,
        "private response body",
    )
    with pytest.raises(CollectionsCaptureError) as caught:
        await service.save_capture(request)
    assert caught.value.reason == "server_save_rejected"
    assert "private" not in str(caught.value)


@pytest.mark.asyncio
async def test_server_update_and_note_links_preserve_authority() -> None:
    client = FakeReadingClient()
    authority, service = _service(
        client,
        {
            "api_version": "1",
            "capabilities": {"hasReadingSnapshotPagesV1": True},
        },
    )
    identity = CaptureIdentity(authority.key, "1")
    detail = await service.get_detail(identity)

    changed = await service.update_capture(
        identity,
        detail.revision,
        {"favorite": True, "status": "reading", "tags": ("AI",)},
    )
    link = await service.link_note(
        identity,
        ExternalNoteReference(authority.key, "note-1"),
    )

    assert changed.favorite is True
    assert changed.status == "reading"
    assert changed.tags == ("AI",)
    assert link.note_reference.authority_key == authority.key
    other = build_server_capture_authority("server-b", "user-b")
    with pytest.raises(CollectionsCaptureError) as caught:
        await service.link_note(
            identity,
            ExternalNoteReference(other.key, "note-2"),
        )
    assert caught.value.reason == "server_note_authority_mismatch"


@pytest.mark.asyncio
async def test_supported_update_does_not_depend_on_snapshot_browse_attestation() -> (
    None
):
    client = FakeReadingClient()
    authority, service = _service(
        client,
        {"api_version": "1", "capabilities": {}},
    )
    identity = CaptureIdentity(authority.key, "1")

    changed = await service.update_capture(identity, 2, {"favorite": True})

    assert changed.favorite is True
    assert (await service.capabilities()).for_action(
        "browse"
    ).state is CapabilityState.UNSUPPORTED
    assert (await service.capabilities()).for_action(
        "update"
    ).state is CapabilityState.SUPPORTED


@pytest.mark.asyncio
async def test_server_saved_search_accepts_api_string_filters() -> None:
    client = FakeReadingClient()

    async def saved_searches(**_params: Any) -> dict[str, Any]:
        return {
            "items": [
                {
                    "id": 7,
                    "name": "Saved research",
                    "query": {"status": "saved", "tags": "research"},
                    "sort": "created_desc",
                    "created_at": "2026-09-01T12:00:00Z",
                    "updated_at": "2026-09-01T12:00:00Z",
                }
            ],
            "total": 1,
            "limit": 20,
            "offset": 0,
        }

    client.list_reading_saved_searches = saved_searches  # type: ignore[method-assign]
    authority, service = _service(
        client,
        {
            "api_version": "1",
            "capabilities": {"hasReadingSnapshotPagesV1": True},
        },
    )

    page = await service.list_saved_searches(page=1)

    assert page.items[0].request.statuses == ("saved",)
    assert page.items[0].request.tags == ("research",)
    assert page.items[0].request.sort == "saved_desc"


@pytest.mark.asyncio
async def test_feature_route_404_downgrades_only_probed_capability() -> None:
    client = FakeReadingClient()
    client.failures["list_reading_highlights"] = APIResponseError(404, "missing")
    authority, service = _service(
        client,
        {
            "api_version": "1",
            "capabilities": {"hasReadingSnapshotPagesV1": True},
        },
    )

    downgraded = await service.probe_capability("highlights")
    capabilities = await service.capabilities()

    assert downgraded.state is CapabilityState.UNSUPPORTED
    assert downgraded.reason == "server_feature_unavailable"
    assert capabilities.for_action("highlights") == downgraded
    assert capabilities.for_action("browse").state is CapabilityState.SUPPORTED
    assert capabilities.for_action("capture").state is CapabilityState.SUPPORTED


@pytest.mark.asyncio
async def test_capability_snapshot_change_invalidates_feature_probe_downgrade() -> None:
    client = FakeReadingClient()
    client.failures["list_reading_highlights"] = APIResponseError(404, "missing")
    docs = {
        "api_version": "1",
        "capabilities": {"hasReadingSnapshotPagesV1": True},
    }
    _authority, service = _service(client, docs)
    assert (
        await service.probe_capability("highlights")
    ).state is CapabilityState.UNSUPPORTED

    client.failures.clear()
    docs["capabilities"]["readingCapabilityGeneration"] = 2

    assert (await service.capabilities()).for_action(
        "highlights"
    ).state is CapabilityState.SUPPORTED


@pytest.mark.asyncio
async def test_capability_discovery_failure_is_unknown_and_content_free() -> None:
    authority = build_server_capture_authority("server-a", "user-a")

    def unavailable_docs():
        raise APIConnectionError("private credential and endpoint")

    service = ServerCollectionsCaptureService(
        authority,
        FakeReadingClient(),
        docs_info_provider=unavailable_docs,
        credential_fingerprint="credential-a",
    )

    capability = (await service.capabilities()).for_action("capture")

    assert capability.state is CapabilityState.UNKNOWN
    assert capability.reason == "server_capability_discovery_failed"
    assert "private" not in repr(capability)
