"""Local Research Workspace adapter over the existing workspace registry."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingBackend
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.models import WorkspaceRecord
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    next_local_workspace_identity,
)

from .contracts import (
    BoundedPageResult,
    CapabilityUnavailableError,
    MAX_RESEARCH_SELECTION_IDS,
    QualifiedWorkspaceRef,
    ResearchCatalogItem,
    ResearchCapability,
    ResearchSourcePreview,
    ResearchSourceSummary,
    ResearchWorkspaceSummary,
    SourceSelectionResult,
    SourceReadiness,
    WorkspaceDataSource,
    require_capability,
)
from .source_operations import (
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
)
from .source_operation_store import SourceOperationConflictError
from .source_readiness import normalize_local_readiness


_LOCAL_AVAILABLE = ResearchCapability(
    available=True,
    reason_code="available",
    user_message="Available in Local workspaces.",
    owner="local",
)
_LOCAL_DELETE = ResearchCapability(
    available=False,
    reason_code="settings_owned",
    user_message="Delete local workspaces from Settings.",
    owner="settings",
    recovery_action="Open Settings > Workspaces.",
)
_LOCAL_CAPABILITIES: Mapping[str, ResearchCapability] = {
    "list": _LOCAL_AVAILABLE,
    "get": _LOCAL_AVAILABLE,
    "create": _LOCAL_AVAILABLE,
    "update": _LOCAL_AVAILABLE,
    "duplicate": _LOCAL_AVAILABLE,
    "archive": _LOCAL_AVAILABLE,
    "restore": _LOCAL_AVAILABLE,
    "delete": _LOCAL_DELETE,
    "list_sources": _LOCAL_AVAILABLE,
    "search_catalog": _LOCAL_AVAILABLE,
    "attach_existing": _LOCAL_AVAILABLE,
    "remove_source": _LOCAL_AVAILABLE,
    "preview_source": _LOCAL_AVAILABLE,
    "get_readiness": _LOCAL_AVAILABLE,
    "set_selected_scope": _LOCAL_AVAILABLE,
}


def _page_bounds(limit: object, offset: object) -> tuple[int, int]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if type(offset) is not int or not 0 <= offset <= 10_000:
        raise ValueError("offset must be between 0 and 10000")
    return limit, offset


def _bounded_query(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("query must be text")
    normalized = value.strip()
    if len(normalized) > 1000:
        raise ValueError("query is too long")
    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LocalResearchWorkspaceAdapter:
    """Expose Research notebook lifecycle without changing local ownership."""

    def __init__(
        self,
        service: LocalWorkspaceRegistryService,
        *,
        id_factory: Callable[[], str] | None = None,
        media_scope_service: Any | None = None,
        operation_store: Any | None = None,
        association_scheduler: Any | None = None,
        operation_id_factory: Callable[[], str] | None = None,
        now_factory: Callable[[], str] | None = None,
    ) -> None:
        self._service = service
        self._id_factory = id_factory
        self._media_scope = media_scope_service
        self._operation_store = operation_store
        self._association_scheduler = association_scheduler
        self._operation_id_factory = operation_id_factory or (
            lambda: f"source-operation-{uuid4().hex}"
        )
        self._now_factory = now_factory or _utc_now

    async def list_workspaces(
        self, *, include_archived: bool = False
    ) -> tuple[ResearchWorkspaceSummary, ...]:
        require_capability(_LOCAL_CAPABILITIES, "list")
        records = await asyncio.to_thread(
            self._service.list_workspaces, include_archived=include_archived
        )
        return tuple(
            self._summary(record)
            for record in records
            if record.workspace_id != DEFAULT_WORKSPACE_ID
        )

    async def get_workspace(
        self, ref: QualifiedWorkspaceRef
    ) -> ResearchWorkspaceSummary | None:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "get")
        record = await asyncio.to_thread(self._service.get_workspace, ref.workspace_id)
        return self._matching_summary(ref, record) if record is not None else None

    async def create_workspace(
        self, *, name: str, description: str = "", template_id: str = ""
    ) -> ResearchWorkspaceSummary:
        require_capability(_LOCAL_CAPABILITIES, "create")
        if template_id.strip():
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="template_unavailable",
                    user_message="Local workspace templates are not available.",
                    owner="local",
                    recovery_action="Create a blank workspace.",
                )
            )
        record = await asyncio.to_thread(
            self._service.create_workspace,
            workspace_id=await self._next_workspace_id(),
            name=name,
            description=description,
        )
        return self._summary(record)

    async def update_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        name: str | None = None,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "update")
        if name is None:
            current = await self.get_workspace(ref)
            if current is None:
                raise ValueError(f"Workspace not found: {ref.workspace_id}")
            return current
        record = await asyncio.to_thread(
            self._service.rename_workspace, ref.workspace_id, name
        )
        return self._matching_summary(ref, record)

    async def duplicate_workspace(
        self, ref: QualifiedWorkspaceRef, *, name: str
    ) -> ResearchWorkspaceSummary:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "duplicate")
        source = await asyncio.to_thread(self._service.get_workspace, ref.workspace_id)
        if source is None:
            raise ValueError(f"Workspace not found: {ref.workspace_id}")
        record = await asyncio.to_thread(
            self._service.create_workspace,
            workspace_id=await self._next_workspace_id(),
            name=name,
            description=source.description,
        )
        return self._summary(record)

    async def archive_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "archive")
        record = await asyncio.to_thread(
            self._service.archive_workspace, ref.workspace_id
        )
        return self._matching_summary(ref, record)

    async def restore_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "restore")
        record = await asyncio.to_thread(
            self._service.unarchive_workspace, ref.workspace_id
        )
        return self._matching_summary(ref, record)

    async def delete_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> bool:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "delete")
        raise AssertionError("unreachable")

    async def capabilities(
        self, ref: QualifiedWorkspaceRef
    ) -> Mapping[str, ResearchCapability]:
        self._require_local_ref(ref)
        return _LOCAL_CAPABILITIES

    async def list_sources(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> BoundedPageResult[ResearchSourceSummary]:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "list_sources")
        page_limit, page_offset = _page_bounds(limit, offset)
        media_scope = self._require_media_scope()
        memberships, total = await asyncio.to_thread(
            self._service.list_workspace_source_memberships,
            ref.workspace_id,
            limit=page_limit,
            offset=page_offset,
        )
        desired_scope = await asyncio.to_thread(
            self._service.get_workspace_scope, ref.workspace_id
        )
        desired_ids = (
            None
            if desired_scope is None
            else {
                item.source_id
                for item in desired_scope.items
                if item.source_type == "media"
            }
        )
        details = await asyncio.gather(
            *(
                media_scope.get_media_detail(
                    mode=MediaReadingBackend.LOCAL,
                    media_id=membership.item_id,
                )
                for membership in memberships
            )
        )
        rows = tuple(
            self._source_summary(
                ref,
                membership,
                detail,
                selected=(
                    True
                    if desired_ids is None
                    else membership.item_id in desired_ids
                ),
                position=page_offset + index,
            )
            for index, (membership, detail) in enumerate(zip(memberships, details))
        )
        return BoundedPageResult(
            items=rows,
            limit=page_limit,
            offset=page_offset,
            total=total,
            has_more=page_offset + len(rows) < total,
        )

    async def search_catalog(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        query: str = "",
        source_types: tuple[str, ...] = (),
        sort_by: str = "updated_desc",
        limit: int = 25,
        offset: int = 0,
    ) -> BoundedPageResult[ResearchCatalogItem]:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "search_catalog")
        page_limit, page_offset = _page_bounds(limit, offset)
        normalized_query = _bounded_query(query)
        allowed_sorts = {
            "relevance",
            "title_asc",
            "title_desc",
            "updated_asc",
            "updated_desc",
        }
        if sort_by not in allowed_sorts:
            raise ValueError("sort_by is unsupported")
        owner_sort = {
            "updated_asc": "last_modified_asc",
            "updated_desc": "last_modified_desc",
        }.get(sort_by, sort_by)
        if len(source_types) > 25 or any(
            not isinstance(value, str) or not value.strip() or len(value) > 128
            for value in source_types
        ):
            raise ValueError("source_types is invalid")
        result = await self._require_media_scope().search_media(
            mode=MediaReadingBackend.LOCAL,
            query=normalized_query or None,
            limit=page_limit,
            offset=page_offset,
            media_types=[value.strip() for value in source_types],
            sort_by=owner_sort,
        )
        raw_items = result.get("items") if isinstance(result, Mapping) else None
        if not isinstance(raw_items, list) or len(raw_items) > page_limit:
            raise ValueError("Local catalog returned an invalid bounded page")
        items = tuple(self._catalog_item(ref, item) for item in raw_items)
        total = result.get("total")
        if type(total) is not int or total < len(items):
            raise ValueError("Local catalog returned an invalid total")
        return BoundedPageResult(
            items=items,
            limit=page_limit,
            offset=page_offset,
            total=total,
            has_more=page_offset + len(items) < total,
        )

    async def attach_existing(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        catalog_item_id: str,
        desired_selected: bool = True,
        idempotency_key: str,
    ) -> ResearchSourceOperation:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "attach_existing")
        if type(desired_selected) is not bool:
            raise ValueError("desired_selected must be bool")
        canonical_id = self._canonical_media_id(catalog_item_id)
        await self._require_media_scope().get_media_detail(
            mode=MediaReadingBackend.LOCAL, media_id=canonical_id
        )
        if self._operation_store is None or self._association_scheduler is None:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="source_operation_unavailable",
                    user_message="Durable source attachment is unavailable.",
                    owner="local",
                    recovery_action="Restart and retry.",
                )
            )
        now = self._now_factory()
        operation = ResearchSourceOperation(
            operation_id=self._operation_id_factory(),
            idempotency_key=idempotency_key,
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id=ref.workspace_id,
            canonical_item_type=CanonicalItemType.LOCAL_LIBRARY,
            desired_selected=desired_selected,
            created_at=now,
            updated_at=now,
        )
        try:
            operation = await asyncio.to_thread(
                self._operation_store.create, operation
            )
        except SourceOperationConflictError:
            existing = await asyncio.to_thread(
                self._operation_store.get_by_idempotency_key, idempotency_key
            )
            if not self._matching_attach_intent(
                existing,
                ref=ref,
                canonical_id=canonical_id,
                desired_selected=desired_selected,
            ):
                raise
            operation = existing
        if operation.catalog_status is SourceOperationStatus.PENDING:
            operation = await asyncio.to_thread(
                self._operation_store.advance_stage,
                operation.operation_id,
                stage=SourceOperationStage.CATALOG,
                status=SourceOperationStatus.SUCCEEDED,
                expected_revision=operation.revision,
                canonical_item_id=canonical_id,
            )
        settled = await self._association_scheduler.resume(operation.operation_id)
        if settled is None:
            raise RuntimeError("Durable source attachment did not settle")
        return settled

    async def remove_source(
        self,
        ref: QualifiedWorkspaceRef,
        source_id: str,
        *,
        expected_version: int | None = None,
    ) -> bool:
        self._require_local_ref(ref)
        if expected_version is not None:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="version_precondition_unavailable",
                    user_message=(
                        "Version-checked source removal is unavailable for Local workspaces."
                    ),
                    owner="local",
                    recovery_action="Refresh sources, then remove without a version check.",
                )
            )
        require_capability(_LOCAL_CAPABILITIES, "remove_source")
        membership = await self._membership_for_source(ref, source_id)
        return await asyncio.to_thread(
            self._service.unlink_membership,
            ref.workspace_id,
            item_type="media",
            item_id=membership.item_id,
            role="source",
        )

    async def update_source(
        self,
        ref: QualifiedWorkspaceRef,
        source_id: str,
        *,
        title: str | None = None,
        expected_version: int | None = None,
    ) -> ResearchSourceSummary:
        self._require_local_ref(ref)
        raise CapabilityUnavailableError(
            ResearchCapability(
                available=False,
                reason_code="canonical_owner_required",
                user_message="Edit this source from the Library.",
                owner="local_library",
                recovery_action="Open the Library item.",
            )
        )

    async def preview_source(
        self,
        ref: QualifiedWorkspaceRef,
        source_id: str,
        *,
        max_chars: int = 3000,
        snippet_limit: int = 3,
    ) -> ResearchSourcePreview:
        self._require_local_ref(ref)
        if type(max_chars) is not int or not 1 <= max_chars <= 12_000:
            raise ValueError("max_chars must be between 1 and 12000")
        if type(snippet_limit) is not int or not 0 <= snippet_limit <= 10:
            raise ValueError("snippet_limit must be between 0 and 10")
        membership = await self._membership_for_source(ref, source_id)
        detail = await self._require_media_scope().get_media_detail(
            mode=MediaReadingBackend.LOCAL,
            media_id=membership.item_id,
        )
        text = str(
            detail.get("content") or detail.get("transcription") or ""
        )[:max_chars]
        return ResearchSourcePreview(
            ref=ref,
            source_id=source_id,
            catalog_item_id=membership.item_id,
            preview_mode="available" if text else "empty",
            text=text,
            snippets=(text,) if text and snippet_limit else (),
        )

    async def get_readiness(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[SourceReadiness, ...]:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "get_readiness")
        if not isinstance(source_ids, tuple) or len(source_ids) > 100:
            raise ValueError("source_ids must be a bounded tuple")
        normalized_ids = tuple(self._membership_id(item) for item in source_ids)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("source_ids must be unique")
        if normalized_ids:
            memberships = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self._service.get_workspace_source_membership,
                        ref.workspace_id,
                        source_id,
                    )
                    for source_id in normalized_ids
                )
            )
            if any(membership is None for membership in memberships):
                raise ValueError("source_ids contains an unattached source")
            rows = tuple(
                (membership.membership_id, membership.item_id)
                for membership in memberships
                if membership is not None
            )
        else:
            page = await self.list_sources(ref, limit=100, offset=0)
            rows = tuple(
                (row.source_id, row.catalog_item_id) for row in page.items
            )
        readiness: list[SourceReadiness] = []
        for source_id, catalog_item_id in rows:
            detail = await self._require_media_scope().get_media_detail(
                mode=MediaReadingBackend.LOCAL,
                media_id=catalog_item_id,
            )
            readiness.append(
                normalize_local_readiness(
                    ref=ref,
                    source_id=source_id,
                    catalog_item_id=catalog_item_id,
                    detail=detail,
                )
            )
        return tuple(readiness)

    async def set_selected_scope(
        self,
        ref: QualifiedWorkspaceRef,
        source_ids: tuple[str, ...],
    ) -> SourceSelectionResult:
        self._require_local_ref(ref)
        if (
            not isinstance(source_ids, tuple)
            or len(source_ids) > MAX_RESEARCH_SELECTION_IDS
        ):
            raise ValueError("source_ids must be a unique bounded list")
        desired = tuple(self._canonical_media_id(item) for item in source_ids)
        if len(desired) != len(set(desired)):
            raise ValueError("source_ids must be a unique bounded list")
        membership_groups = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self._service.get_item_memberships, "media", item_id
                )
                for item_id in desired
            )
        )
        if any(
            not any(
                membership.workspace_id == ref.workspace_id
                and membership.role == "source"
                for membership in memberships
            )
            for memberships in membership_groups
        ):
            raise ValueError("source_ids must use attached canonical Media ids")
        await asyncio.to_thread(
            self._service.set_workspace_scope,
            ref.workspace_id,
            RagScope(
                items=tuple(
                    ScopeItem("media", item_id) for item_id in desired
                ),
                updated_at=self._now_factory(),
                empty_is_scoped=True,
            ),
        )
        stored_scope = await asyncio.to_thread(
            self._service.get_workspace_scope, ref.workspace_id
        )
        stored_ids = (
            ()
            if stored_scope is None
            else tuple(
                item.source_id
                for item in stored_scope.items
                if item.source_type == "media"
            )
        )
        if stored_ids != desired or stored_scope is None:
            raise ValueError("Local source selection reconciliation did not match")
        return SourceSelectionResult(ref=ref, desired_source_ids=stored_ids)

    async def reorder_sources(
        self,
        ref: QualifiedWorkspaceRef,
        ordered_source_ids: tuple[str, ...],
    ) -> tuple[ResearchSourceSummary, ...]:
        self._require_local_ref(ref)
        raise CapabilityUnavailableError(
            ResearchCapability(
                available=False,
                reason_code="local_order_unavailable",
                user_message="Local source order is unavailable.",
                owner="local",
                recovery_action="Sort the source list instead.",
            )
        )

    def _require_media_scope(self) -> Any:
        if self._media_scope is None:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="local_media_unavailable",
                    user_message="The Local Library is unavailable.",
                    owner="local_library",
                    recovery_action="Restart after Local Library storage is available.",
                )
            )
        return self._media_scope

    @staticmethod
    def _matching_attach_intent(
        operation: ResearchSourceOperation | None,
        *,
        ref: QualifiedWorkspaceRef,
        canonical_id: str,
        desired_selected: bool,
    ) -> bool:
        return bool(
            operation is not None
            and operation.data_source is WorkspaceDataSource.LOCAL
            and operation.workspace_id == ref.workspace_id
            and operation.desired_selected is desired_selected
            and operation.catalog_status is SourceOperationStatus.SUCCEEDED
            and operation.canonical_item_id == canonical_id
        )

    async def _membership_for_source(self, ref, source_id):
        membership = await asyncio.to_thread(
            self._service.get_workspace_source_membership,
            ref.workspace_id,
            self._membership_id(source_id),
        )
        if membership is not None:
            return membership
        raise ValueError("Workspace source does not exist")

    @staticmethod
    def _membership_id(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source_ids must contain nonblank membership IDs")
        normalized = value.strip()
        if len(normalized) > 1024 or len(normalized.encode("utf-8")) > 4096:
            raise ValueError("source_ids contain an oversized membership ID")
        return normalized

    @staticmethod
    def _canonical_media_id(value: object) -> str:
        if isinstance(value, bool):
            raise ValueError("source_ids must contain positive canonical Media ids")
        normalized = str(value).strip()
        if not normalized.isdigit() or int(normalized) < 1:
            raise ValueError("source_ids must contain positive canonical Media ids")
        return str(int(normalized))

    @staticmethod
    def _catalog_item(
        ref: QualifiedWorkspaceRef, row: Mapping[str, Any]
    ) -> ResearchCatalogItem:
        catalog_id = row.get("backing_media_id", row.get("source_id"))
        version = row.get("version")
        return ResearchCatalogItem(
            ref=ref,
            catalog_item_id=str(catalog_id),
            title=str(row.get("title") or "Untitled"),
            source_type=str(row.get("media_type") or row.get("type") or "media"),
            catalog_item_version=version if type(version) is int else None,
            updated_at=str(row.get("updated_at") or ""),
        )

    @staticmethod
    def _source_summary(ref, membership, detail, *, selected, position):
        version = detail.get("version")
        readiness = normalize_local_readiness(
            ref=ref,
            source_id=membership.membership_id,
            catalog_item_id=membership.item_id,
            detail=detail,
        )
        return ResearchSourceSummary(
            ref=ref,
            source_id=membership.membership_id,
            catalog_item_id=membership.item_id,
            title=str(detail.get("title") or membership.title or "Untitled"),
            source_type=str(detail.get("media_type") or detail.get("type") or "media"),
            ready=readiness.fts_ready,
            version=None,
            catalog_item_version=version if type(version) is int else None,
            selected=selected,
            position=position,
        )

    async def _next_workspace_id(self) -> str:
        if self._id_factory is not None:
            return self._id_factory()
        workspace_id, _ = await asyncio.to_thread(
            next_local_workspace_identity, self._service
        )
        return workspace_id

    @staticmethod
    def _require_local_ref(ref: QualifiedWorkspaceRef) -> None:
        if ref.data_source is not WorkspaceDataSource.LOCAL:
            raise ValueError("Local adapter requires a Local workspace ref")

    @staticmethod
    def _summary(record: WorkspaceRecord) -> ResearchWorkspaceSummary:
        return ResearchWorkspaceSummary(
            ref=QualifiedWorkspaceRef(
                WorkspaceDataSource.LOCAL, record.workspace_id
            ),
            name=record.name,
            description=record.description,
            archived=record.archived,
            updated_at=record.updated_at,
        )

    def _matching_summary(
        self, expected_ref: QualifiedWorkspaceRef, record: WorkspaceRecord
    ) -> ResearchWorkspaceSummary:
        summary = self._summary(record)
        if summary.ref != expected_ref:
            raise ValueError("Adapter returned a mismatched workspace ref")
        return summary
