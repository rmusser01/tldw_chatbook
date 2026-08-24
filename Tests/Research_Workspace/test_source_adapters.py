from __future__ import annotations

import asyncio
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingBackend
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry
from tldw_chatbook.Research_Workspace.contracts import (
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    SourceIdentityMismatchError,
    SourceReadinessState,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.local_adapter import (
    LocalResearchWorkspaceAdapter,
)
from tldw_chatbook.Research_Workspace.server_adapter import (
    ServerResearchWorkspaceAdapter,
)
from tldw_chatbook.Research_Workspace.source_association import (
    ResearchSourceAssociationCoordinator,
    ResearchSourceAssociationScheduler,
)
from tldw_chatbook.Research_Workspace.source_operation_store import (
    ResearchSourceOperationStore,
)
from tldw_chatbook.Research_Workspace.source_operations import SourceOperationStatus
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)


class RecordingMediaScope:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.details = {
            "7": {
                "source_id": "7",
                "backing_media_id": 7,
                "title": "Paper",
                "media_type": "pdf",
                "updated_at": "2026-08-24T00:00:00Z",
                "version": 4,
                "content": "Paper body",
                "has_transcript": True,
                "has_chunks": True,
                "chunking_status": "completed",
                "vector_processing": False,
            },
            "8": {
                "source_id": "8",
                "backing_media_id": 8,
                "title": "Book",
                "media_type": "ebook",
                "updated_at": "2026-08-23T00:00:00Z",
                "version": 2,
                "content": "Book body",
                "has_transcript": True,
                "has_chunks": True,
                "chunking_status": "completed",
                "vector_processing": "completed",
            },
        }

    async def search_media(self, **kwargs):
        self.calls.append(("search_media", kwargs))
        items = list(self.details.values())
        return {"items": items, "total": 2, "offset": 0, "limit": 25}

    async def search_backing_media_items(self, **kwargs):
        self.calls.append(("search_backing_media_items", kwargs))
        return {
            "items": [
                {
                    "id": 31,
                    "title": "Server paper",
                    "type": "pdf",
                    "last_modified": "2026-08-24T00:00:00Z",
                    "version": 6,
                }
            ],
            "pagination": {
                "page": kwargs["page"],
                "results_per_page": kwargs["results_per_page"],
                "total_items": 1,
                "total_pages": 1,
            },
        }

    async def get_media_detail(self, **kwargs):
        self.calls.append(("get_media_detail", kwargs))
        return dict(self.details[str(kwargs["media_id"])])


def local_registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="source-adapter")
    )
    registry.create_workspace(workspace_id="workspace-1", name="Research")
    return registry


@pytest.mark.asyncio
async def test_local_attached_rows_keep_membership_and_media_identities_distinct(
    tmp_path,
) -> None:
    registry = local_registry(tmp_path)
    membership = registry.link_membership(
        "workspace-1", item_type="media", item_id="7", role="source"
    )
    registry.set_workspace_scope(
        "workspace-1",
        RagScope(
            items=(ScopeItem("media", "7"),),
            updated_at="t1",
            empty_is_scoped=True,
        ),
    )
    scope = RecordingMediaScope()
    adapter = LocalResearchWorkspaceAdapter(registry, media_scope_service=scope)
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")

    page = await adapter.list_sources(ref)

    assert len(page.items) == 1
    row = page.items[0]
    assert row.workspace_source_id == membership.membership_id
    assert row.catalog_item_id == "7"
    assert row.workspace_source_version is None
    assert row.catalog_item_version == 4
    assert row.selected is True
    assert scope.calls == [
        (
            "get_media_detail",
            {"mode": MediaReadingBackend.LOCAL, "media_id": "7"},
        )
    ]


@pytest.mark.asyncio
async def test_local_catalog_search_is_bounded_deterministic_and_explicit_mode(
    tmp_path,
) -> None:
    scope = RecordingMediaScope()
    adapter = LocalResearchWorkspaceAdapter(
        local_registry(tmp_path), media_scope_service=scope
    )
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")

    page = await adapter.search_catalog(
        ref,
        query="paper",
        source_types=("pdf",),
        sort_by="updated_desc",
        limit=25,
        offset=0,
    )

    assert [item.catalog_item_id for item in page.items] == ["7", "8"]
    assert scope.calls == [
        (
            "search_media",
            {
                "mode": MediaReadingBackend.LOCAL,
                "query": "paper",
                "limit": 25,
                "offset": 0,
                "media_types": ["pdf"],
                "sort_by": "last_modified_desc",
            },
        )
    ]


@pytest.mark.asyncio
async def test_local_catalog_updated_sort_uses_real_owner_vocabulary_and_order(
    tmp_path,
) -> None:
    media_db = MediaDatabase(
        db_path=tmp_path / "media.sqlite", client_id="research-sort"
    )
    older_id, _, _ = media_db.add_media_with_keywords(
        title="Older", content="older", media_type="article", keywords=[]
    )
    newer_id, _, _ = media_db.add_media_with_keywords(
        title="Newer", content="newer", media_type="article", keywords=[]
    )
    assert older_id is not None and newer_id is not None
    media_db.execute_query(
        "UPDATE Media SET last_modified = ?, version = version + 1 WHERE id = ?",
        ("2026-01-01 00:00:00", older_id),
    )
    media_db.execute_query(
        "UPDATE Media SET last_modified = ?, version = version + 1 WHERE id = ?",
        ("2026-02-01 00:00:00", newer_id),
    )
    media_scope = MediaReadingScopeService(
        LocalMediaReadingService(media_db), SimpleNamespace()
    )
    adapter = LocalResearchWorkspaceAdapter(
        local_registry(tmp_path), media_scope_service=media_scope
    )

    page = await adapter.search_catalog(
        QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1"),
        sort_by="updated_asc",
        limit=25,
    )

    assert [item.catalog_item_id for item in page.items] == [
        str(older_id),
        str(newer_id),
    ]


@pytest.mark.asyncio
async def test_local_selection_uses_canonical_media_ids_and_persists_empty(
    tmp_path,
) -> None:
    registry = local_registry(tmp_path)
    registry.link_membership(
        "workspace-1", item_type="media", item_id="7", role="source"
    )
    adapter = LocalResearchWorkspaceAdapter(
        registry, media_scope_service=RecordingMediaScope()
    )
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")

    await adapter.set_selected_scope(ref, ())

    assert registry.get_workspace_scope("workspace-1") == RagScope(
        items=(),
        updated_at=registry.get_workspace_scope("workspace-1").updated_at,
        empty_is_scoped=True,
    )

    with pytest.raises(ValueError, match="canonical Media"):
        await adapter.set_selected_scope(
            ref,
            (registry.list_workspace_memberships("workspace-1")[0].membership_id,),
        )


@pytest.mark.asyncio
async def test_local_selection_reconciles_101_canonical_ids_without_page_one_loss(
    tmp_path,
) -> None:
    registry = local_registry(tmp_path)
    for media_id in range(1, 102):
        registry.link_membership(
            "workspace-1",
            item_type="media",
            item_id=str(media_id),
            role="source",
        )

    class ManyMediaScope:
        async def get_media_detail(self, **kwargs):
            media_id = str(kwargs["media_id"])
            return {
                "source_id": media_id,
                "title": f"Source {media_id}",
                "media_type": "document",
            }

    adapter = LocalResearchWorkspaceAdapter(
        registry, media_scope_service=ManyMediaScope()
    )
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    requested = tuple(str(media_id) for media_id in range(1, 102))

    result = await adapter.set_selected_scope(ref, requested)

    assert result.ref == ref
    assert result.desired_source_ids == requested
    assert len(result.sources) <= 100
    assert tuple(
        item.source_id for item in registry.get_workspace_scope("workspace-1").items
    ) == requested


@pytest.mark.asyncio
async def test_local_page_projects_exact_restarted_selection_beyond_page_one(
    tmp_path,
) -> None:
    registry = local_registry(tmp_path)
    for media_id in range(1, 102):
        registry.link_membership(
            "workspace-1", item_type="media", item_id=str(media_id), role="source"
        )
    registry.set_workspace_scope(
        "workspace-1",
        RagScope(
            items=(ScopeItem("media", "101"),),
            updated_at="restart",
            empty_is_scoped=True,
        ),
    )

    class ManyMediaScope:
        async def get_media_detail(self, **kwargs):
            media_id = str(kwargs["media_id"])
            return {
                "source_id": media_id,
                "backing_media_id": int(media_id),
                "title": f"Source {media_id}",
                "media_type": "document",
            }

    adapter = LocalResearchWorkspaceAdapter(
        registry, media_scope_service=ManyMediaScope()
    )
    page = await adapter.list_sources(
        QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1"),
        limit=100,
        offset=0,
    )

    assert page.desired_source_ids == ("101",)
    assert all(not row.selected for row in page.items)


@pytest.mark.asyncio
async def test_local_remove_unlinks_only_and_local_update_reorder_are_typed(
    tmp_path,
) -> None:
    registry = local_registry(tmp_path)
    membership = registry.link_membership(
        "workspace-1", item_type="media", item_id="7", role="source"
    )
    adapter = LocalResearchWorkspaceAdapter(
        registry, media_scope_service=RecordingMediaScope()
    )
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")

    assert await adapter.remove_source(ref, membership.membership_id)
    assert registry.list_workspace_memberships("workspace-1") == ()
    with pytest.raises(CapabilityUnavailableError) as update_error:
        await adapter.update_source(ref, membership.membership_id, title="Other")
    with pytest.raises(CapabilityUnavailableError) as reorder_error:
        await adapter.reorder_sources(ref, (membership.membership_id,))

    assert update_error.value.capability.reason_code == "canonical_owner_required"
    assert reorder_error.value.capability.reason_code == "local_order_unavailable"


@pytest.mark.asyncio
async def test_local_remove_refuses_unenforceable_version_before_storage_call(
    tmp_path, monkeypatch
) -> None:
    registry = local_registry(tmp_path)
    membership = registry.link_membership(
        "workspace-1", item_type="media", item_id="7", role="source"
    )
    calls: list[object] = []
    monkeypatch.setattr(
        registry,
        "unlink_membership",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    adapter = LocalResearchWorkspaceAdapter(
        registry, media_scope_service=RecordingMediaScope()
    )

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.remove_source(
            QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1"),
            membership.membership_id,
            expected_version=3,
        )

    assert exc_info.value.capability.reason_code == "version_precondition_unavailable"
    assert calls == []


@pytest.mark.asyncio
async def test_local_requested_readiness_finds_source_after_public_page_100(
    tmp_path,
) -> None:
    registry = local_registry(tmp_path)
    for media_id in range(1, 101):
        registry.link_membership(
            "workspace-1", item_type="media", item_id=str(media_id), role="source"
        )
    target = registry.link_membership(
        "workspace-1", item_type="media", item_id="9999", role="source"
    )
    scope = RecordingMediaScope()
    scope.details.update(
        {
            str(media_id): {
                "source_id": str(media_id),
                "backing_media_id": media_id,
                "title": f"Source {media_id}",
                "media_type": "pdf",
                "content": "ready",
                "has_transcript": True,
                "has_chunks": True,
                "chunking_status": "completed",
                "vector_processing": False,
            }
            for media_id in range(1, 101)
        }
    )
    scope.details["9999"] = {
        "source_id": "9999",
        "backing_media_id": 9999,
        "title": "Late source",
        "media_type": "pdf",
        "content": "ready",
        "has_transcript": True,
        "has_chunks": True,
        "chunking_status": "completed",
        "vector_processing": False,
    }
    adapter = LocalResearchWorkspaceAdapter(registry, media_scope_service=scope)

    rows = await adapter.get_readiness(
        QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1"),
        source_ids=(target.membership_id,),
    )

    assert len(rows) == 1
    assert rows[0].source_id == target.membership_id
    assert rows[0].catalog_item_id == "9999"


@pytest.mark.asyncio
async def test_local_attach_existing_creates_intent_then_converges_duplicate_retry(
    tmp_path,
) -> None:
    registry = local_registry(tmp_path)
    store = ResearchSourceOperationStore(registry.db)
    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=LibraryIngestJobRegistry(),
        local_registry=registry,
    )
    scheduler = ResearchSourceAssociationScheduler(
        coordinator=coordinator,
        operation_store=store,
    )
    operation_ids = iter(("operation-1", "operation-2"))
    adapter = LocalResearchWorkspaceAdapter(
        registry,
        media_scope_service=RecordingMediaScope(),
        operation_store=store,
        association_scheduler=scheduler,
        operation_id_factory=lambda: next(operation_ids),
        now_factory=lambda: "2026-08-24T12:00:00Z",
    )
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")

    first = await adapter.attach_existing(
        ref,
        catalog_item_id="7",
        desired_selected=True,
        idempotency_key="local:workspace-1:catalog-7",
    )
    replay = await adapter.attach_existing(
        ref,
        catalog_item_id="7",
        desired_selected=True,
        idempotency_key="local:workspace-1:catalog-7",
    )

    assert replay == first
    assert first.catalog_status is SourceOperationStatus.SUCCEEDED
    assert first.association_status is SourceOperationStatus.SUCCEEDED
    assert len(registry.get_item_memberships("media", "7")) == 1


@pytest.mark.asyncio
async def test_local_source_membership_query_does_not_block_event_loop(
    tmp_path,
    monkeypatch,
) -> None:
    registry = local_registry(tmp_path)
    release = threading.Event()
    original = registry.list_workspace_source_memberships

    def blocking_query(*args, **kwargs):
        release.wait(timeout=1)
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, "list_workspace_source_memberships", blocking_query)
    adapter = LocalResearchWorkspaceAdapter(
        registry, media_scope_service=RecordingMediaScope()
    )
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    timer = threading.Timer(0.15, release.set)
    timer.start()
    started = time.monotonic()
    try:
        task = asyncio.create_task(adapter.list_sources(ref))
        await asyncio.sleep(0.02)
        assert time.monotonic() - started < 0.1
        await task
    finally:
        release.set()
        timer.cancel()


def server_context(profile_id: str = "profile-1") -> object:
    return SimpleNamespace(
        active_server_id=profile_id,
        auth_token="test-token",
        credential_source="test",
        capabilities={
            "server_configured": True,
            "reachability": "reachable",
            "auth_state": "authenticated",
        },
    )


class ContextProvider:
    def __init__(self) -> None:
        self.context = server_context()
        self.calls = 0

    def get_active_context(self):
        self.calls += 1
        return self.context


def server_ref(provider: ContextProvider) -> QualifiedWorkspaceRef:
    return QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-1",
        server_profile_id="profile-1",
        principal_id=event_principal_id_from_active_context(provider.context) or "",
    )


class RecordingServerSourceService:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.rows = [
            {
                "id": "source-1",
                "workspace_id": "workspace-1",
                "media_id": 31,
                "title": "Server paper",
                "source_type": "pdf",
                "position": 0,
                "selected": True,
                "version": 5,
            }
        ]
        self.capability = {
            "workspace_id": "workspace-1",
            "workspace_profile": "research",
            "workspace_kind": "research_workspace",
            "access_level": "owner",
            "workspace_services": {},
            "allowed_actions": {
                "add_sources": {"allowed": True, "reason_code": None},
                "inspect_sources": {"allowed": True, "reason_code": None},
            },
        }

    async def get_workspace_capabilities(self, workspace_id):
        self.calls.append(("capabilities", workspace_id))
        return dict(self.capability)

    async def list_workspace_sources(self, workspace_id):
        self.calls.append(("list", workspace_id))
        return list(self.rows)

    async def set_workspace_source_selection(self, workspace_id, selected_ids):
        self.calls.append(("selection", workspace_id, list(selected_ids)))
        selected = set(selected_ids)
        return [
            row | {"selected": row["id"] in selected, "version": 7} for row in self.rows
        ]

    async def reorder_workspace_sources(self, workspace_id, ordered_ids):
        self.calls.append(("reorder", workspace_id, list(ordered_ids)))
        by_id = {row["id"]: row for row in self.rows}
        return [
            by_id[source_id] | {"position": index, "version": 8}
            for index, source_id in enumerate(ordered_ids)
        ]

    async def get_workspace_source_status(self, workspace_id):
        self.calls.append(("status", workspace_id))
        return {
            "workspace_id": workspace_id,
            "sources": [
                {
                    "id": "source-1",
                    "workspace_id": workspace_id,
                    "media_id": 31,
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
                    "retry_eligible": False,
                    "stale": False,
                }
            ],
            "summary": {},
        }

    async def delete_workspace_source(self, workspace_id, source_id):
        self.calls.append(("delete", workspace_id, source_id))
        return None


@pytest.mark.asyncio
async def test_server_source_list_preserves_two_identity_spaces_and_refetch_versions() -> (
    None
):
    provider = ContextProvider()
    service = RecordingServerSourceService()
    adapter = ServerResearchWorkspaceAdapter(
        service, provider, media_scope_service=RecordingMediaScope()
    )
    ref = server_ref(provider)

    page = await adapter.list_sources(ref)
    selected = await adapter.set_selected_scope(ref, ("source-1",))

    assert page.items[0].source_id == "source-1"
    assert page.items[0].catalog_item_id == "31"
    assert page.items[0].workspace_source_version == 5
    assert selected.sources[0].workspace_source_version == 7
    assert service.calls == [
        ("capabilities", "workspace-1"),
        ("list", "workspace-1"),
        ("capabilities", "workspace-1"),
        ("selection", "workspace-1", ["source-1"]),
    ]


@pytest.mark.asyncio
async def test_server_selection_reconciles_101_owner_ids_with_a_bounded_row_subset() -> (
    None
):
    provider = ContextProvider()
    service = RecordingServerSourceService()
    service.rows = [
        service.rows[0]
        | {"id": f"source-{index}", "media_id": index + 1, "position": index}
        for index in range(101)
    ]
    adapter = ServerResearchWorkspaceAdapter(service, provider)
    requested = tuple(f"source-{index}" for index in range(101))

    result = await adapter.set_selected_scope(server_ref(provider), requested)

    assert result.desired_source_ids == requested
    assert len(result.sources) == 100
    assert all(source.selected for source in result.sources)


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["top", "row"])
async def test_server_readiness_rejects_every_mismatched_workspace_identity(
    mismatch,
) -> None:
    provider = ContextProvider()

    class MismatchedStatusService(RecordingServerSourceService):
        async def get_workspace_source_status(self, workspace_id):
            payload = await super().get_workspace_source_status(workspace_id)
            if mismatch == "top":
                payload["workspace_id"] = "workspace-other"
            else:
                payload["sources"][0]["workspace_id"] = "workspace-other"
            return payload

    service = MismatchedStatusService()
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    with pytest.raises(SourceIdentityMismatchError, match="mismatched workspace"):
        await adapter.get_readiness(server_ref(provider))

    assert service.calls == [
        ("capabilities", "workspace-1"),
        ("status", "workspace-1"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["missing", "not-a-list", "non-mapping-row", "oversized"],
)
async def test_server_readiness_malformed_projection_is_not_identity_mismatch(
    case,
) -> None:
    provider = ContextProvider()

    class MalformedStatusService(RecordingServerSourceService):
        async def get_workspace_source_status(self, workspace_id):
            self.calls.append(("status", workspace_id))
            sources = {
                "missing": None,
                "not-a-list": "not-a-list",
                "non-mapping-row": [None],
                "oversized": [None] * 10_101,
            }[case]
            return {
                "workspace_id": workspace_id,
                "sources": sources,
                "summary": {},
            }

    adapter = ServerResearchWorkspaceAdapter(MalformedStatusService(), provider)

    with pytest.raises(ValueError) as exc_info:
        await adapter.get_readiness(server_ref(provider))

    assert type(exc_info.value) is ValueError


@pytest.mark.asyncio
async def test_server_missing_media_preview_keeps_association_without_catalog_id() -> (
    None
):
    provider = ContextProvider()

    class MissingMediaPreviewService(RecordingServerSourceService):
        async def preview_workspace_source(
            self, workspace_id, source_id, *, max_chars, chunk_limit
        ):
            self.calls.append(("preview", workspace_id, source_id))
            return {
                "workspace_id": workspace_id,
                "source_id": source_id,
                "media_id": 0,
                "title": "Missing source",
                "source_type": "document",
                "url": None,
                "state": "missing_media",
                "status_reason": "media_id_missing",
                "readiness": {
                    "metadata_ready": False,
                    "text_extracted": False,
                    "fts_ready": False,
                    "vector_ready": False,
                    "citation_ready": False,
                    "summary_ready": False,
                    "tool_accessible": False,
                },
                "content_available": False,
                "preview_mode": "missing_media",
                "unavailable_reason": "media_id_missing",
                "text_preview": None,
                "text_total_chars": None,
                "text_truncated": False,
                "snippets": [],
                "generated_at": "2026-08-24T00:00:00Z",
            }

    service = MissingMediaPreviewService()
    adapter = ServerResearchWorkspaceAdapter(service, provider)
    ref = server_ref(provider)

    preview = await adapter.preview_source(ref, "source-1")

    assert preview.ref == ref
    assert preview.source_id == "source-1"
    assert preview.catalog_item_id is None
    assert preview.preview_mode == "missing_media"
    assert preview.text == ""


@pytest.mark.asyncio
async def test_server_reorder_preflights_the_exact_owner_before_mutation() -> None:
    provider = ContextProvider()
    service = RecordingServerSourceService()
    service.rows = [
        service.rows[0]
        | {"id": "source-1", "media_id": 31, "position": 0},
        service.rows[0]
        | {"id": "source-2", "media_id": 32, "position": 1},
    ]
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    rows = await adapter.reorder_sources(
        server_ref(provider), ("source-2", "source-1")
    )

    assert [row.source_id for row in rows] == ["source-2", "source-1"]
    assert service.calls == [
        ("capabilities", "workspace-1"),
        ("list", "workspace-1"),
        ("reorder", "workspace-1", ["source-2", "source-1"]),
    ]


@pytest.mark.asyncio
async def test_server_reorder_refuses_owner_over_request_bound_before_put() -> None:
    provider = ContextProvider()
    service = RecordingServerSourceService()
    service.rows = [
        service.rows[0]
        | {"id": f"source-{index}", "media_id": index + 1, "position": index}
        for index in range(101)
    ]
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.reorder_sources(
            server_ref(provider), tuple(row["id"] for row in service.rows)
        )

    assert exc_info.value.capability.reason_code == "reorder_precondition_unavailable"
    assert service.calls == [
        ("capabilities", "workspace-1"),
        ("list", "workspace-1"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ordered_ids",
    [
        ("source-1",),
        ("source-1", "source-stale"),
        ("source-1", "source-1"),
    ],
)
async def test_server_reorder_refuses_nonexact_or_duplicate_owner_set_before_put(
    ordered_ids,
) -> None:
    provider = ContextProvider()
    service = RecordingServerSourceService()
    service.rows = [
        service.rows[0]
        | {"id": "source-1", "media_id": 31, "position": 0},
        service.rows[0]
        | {"id": "source-2", "media_id": 32, "position": 1},
    ]
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.reorder_sources(server_ref(provider), ordered_ids)

    assert exc_info.value.capability.reason_code == "reorder_precondition_unavailable"
    assert not any(call[0] == "reorder" for call in service.calls)


@pytest.mark.asyncio
async def test_server_catalog_uses_media_scope_server_mode_without_local_call() -> None:
    provider = ContextProvider()
    media_scope = RecordingMediaScope()
    service = RecordingServerSourceService()
    adapter = ServerResearchWorkspaceAdapter(
        service, provider, media_scope_service=media_scope
    )

    page = await adapter.search_catalog(
        server_ref(provider), query="paper", limit=25, offset=0
    )

    assert page.items[0].catalog_item_id == "31"
    assert media_scope.calls == [
        (
            "search_backing_media_items",
            {
                "mode": MediaReadingBackend.SERVER,
                "page": 1,
                "results_per_page": 100,
                "query": "paper",
                "sort_by": "updated_desc",
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_catalog_stitches_crossing_backing_pages_without_gaps() -> None:
    provider = ContextProvider()

    class PagedMediaScope(RecordingMediaScope):
        async def search_backing_media_items(self, **kwargs):
            self.calls.append(("search_backing_media_items", kwargs))
            page = kwargs["page"]
            start = (page - 1) * 100
            return {
                "items": [
                    {
                        "id": index,
                        "title": f"Item {index}",
                        "type": "pdf",
                        "last_modified": "2026-08-24T00:00:00Z",
                    }
                    for index in range(start, min(start + 100, 150))
                ],
                "pagination": {
                    "page": page,
                    "results_per_page": 100,
                    "total_items": 150,
                    "total_pages": 2,
                },
            }

    media_scope = PagedMediaScope()
    adapter = ServerResearchWorkspaceAdapter(
        RecordingServerSourceService(), provider, media_scope_service=media_scope
    )

    page = await adapter.search_catalog(
        server_ref(provider), limit=25, offset=90
    )

    assert [item.catalog_item_id for item in page.items] == [
        str(index) for index in range(90, 115)
    ]
    assert page.offset + len(page.items) == 115
    assert page.has_more is True
    assert [call[1]["page"] for call in media_scope.calls] == [1, 2]


@pytest.mark.asyncio
async def test_server_owner_rows_over_100_are_valid_but_public_page_stays_bounded() -> None:
    provider = ContextProvider()
    service = RecordingServerSourceService()
    service.rows = [
        service.rows[0]
        | {"id": f"source-{index}", "media_id": index + 1, "position": index}
        for index in range(101)
    ]
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    page = await adapter.list_sources(server_ref(provider), limit=100)

    assert len(page.items) == 100
    assert page.total == 101
    assert page.has_more is True


@pytest.mark.asyncio
async def test_server_pages_retain_exact_owner_selection_across_navigation() -> None:
    provider = ContextProvider()
    service = RecordingServerSourceService()
    service.rows = [
        service.rows[0]
        | {
            "id": f"source-{index}",
            "media_id": index + 1,
            "position": index,
            "selected": index == 100,
        }
        for index in range(101)
    ]
    adapter = ServerResearchWorkspaceAdapter(service, provider)
    ref = server_ref(provider)

    first = await adapter.list_sources(ref, limit=100, offset=0)
    second = await adapter.list_sources(ref, limit=100, offset=100)

    assert first.desired_source_ids == ("source-100",)
    assert second.desired_source_ids == first.desired_source_ids
    assert [row.source_id for row in second.items] == ["source-100"]


@pytest.mark.asyncio
async def test_server_readiness_owner_projection_over_100_is_valid() -> None:
    provider = ContextProvider()

    class ManyStatusService(RecordingServerSourceService):
        async def get_workspace_source_status(self, workspace_id):
            self.calls.append(("status", workspace_id))
            sources = []
            for index in range(101):
                sources.append(
                    {
                        "id": f"source-{index}",
                        "workspace_id": workspace_id,
                        "media_id": index + 1,
                        "state": "partially_queryable",
                        "status_reason": "vector_index_pending",
                        "readiness": {
                            "metadata_ready": True,
                            "text_extracted": True,
                            "fts_ready": True,
                            "vector_ready": False,
                            "citation_ready": True,
                            "summary_ready": False,
                            "tool_accessible": True,
                        },
                        "retry_eligible": False,
                        "stale": False,
                    }
                )
            return {
                "workspace_id": workspace_id,
                "sources": sources,
                "summary": {"total": 101},
            }

    adapter = ServerResearchWorkspaceAdapter(ManyStatusService(), provider)

    rows = await adapter.get_readiness(server_ref(provider))

    assert len(rows) == 101
    assert rows[-1].source_id == "source-100"


@pytest.mark.asyncio
async def test_server_remove_refuses_unenforceable_version_before_dispatch() -> None:
    provider = ContextProvider()
    service = RecordingServerSourceService()
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.remove_source(
            server_ref(provider), "source-1", expected_version=5
        )

    assert exc_info.value.capability.reason_code == "version_precondition_unavailable"
    assert service.calls == []


@pytest.mark.asyncio
async def test_server_readiness_and_missing_capability_are_typed() -> None:
    provider = ContextProvider()
    service = RecordingServerSourceService()
    adapter = ServerResearchWorkspaceAdapter(
        service, provider, media_scope_service=RecordingMediaScope()
    )
    ref = server_ref(provider)

    readiness = await adapter.get_readiness(ref)
    assert readiness[0].state is SourceReadinessState.FTS_READY
    service.capability["allowed_actions"] = {}
    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.list_sources(ref)

    assert exc_info.value.capability.reason_code == "unknown_capability"


@pytest.mark.asyncio
async def test_server_context_switch_is_checked_before_source_dispatch() -> None:
    provider = ContextProvider()
    service = RecordingServerSourceService()
    adapter = ServerResearchWorkspaceAdapter(
        service, provider, media_scope_service=RecordingMediaScope()
    )
    ref = server_ref(provider)
    provider.context = server_context("profile-2")

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.list_sources(ref)

    assert exc_info.value.capability.reason_code == "server_context_changed"
    assert service.calls == []


@pytest.mark.asyncio
async def test_server_rechecks_identity_after_capability_projection() -> None:
    provider = ContextProvider()

    class SwitchingService(RecordingServerSourceService):
        async def get_workspace_capabilities(self, workspace_id):
            projection = await super().get_workspace_capabilities(workspace_id)
            provider.context = server_context("profile-2")
            return projection

    service = SwitchingService()
    adapter = ServerResearchWorkspaceAdapter(
        service, provider, media_scope_service=RecordingMediaScope()
    )

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.list_sources(server_ref(ContextProvider()))

    assert exc_info.value.capability.reason_code == "server_context_changed"
    assert service.calls == [("capabilities", "workspace-1")]


@pytest.mark.asyncio
async def test_missing_server_capability_projection_is_typed_and_discoverable() -> None:
    provider = ContextProvider()
    service = SimpleNamespace()
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    capabilities = await adapter.capabilities(server_ref(provider))

    for action in (
        "list_sources",
        "search_catalog",
        "attach_existing",
        "remove_source",
        "update_source",
        "preview_source",
        "get_readiness",
        "set_selected_scope",
        "reorder_sources",
    ):
        assert capabilities[action].available is False
        assert capabilities[action].reason_code == "server_capability_unavailable"


@pytest.mark.asyncio
async def test_server_selection_rejects_nonassociation_ids_before_dispatch() -> None:
    provider = ContextProvider()
    service = RecordingServerSourceService()
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    with pytest.raises(ValueError, match="association IDs"):
        await adapter.set_selected_scope(server_ref(provider), (31,))

    assert service.calls == []
