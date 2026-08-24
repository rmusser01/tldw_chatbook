"""Fail-closed Server adapter for Research Workspace lifecycle operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

from tldw_chatbook.Media.media_reading_scope_service import MediaReadingBackend
from tldw_chatbook.Notes.server_notes_workspace_service import (
    ServerNotesWorkspaceService,
)
from tldw_chatbook.runtime_policy.server_context import ServerContextError
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.tldw_api.exceptions import (
    APIConnectionError,
    APIResponseError,
    AuthenticationError,
)
from tldw_chatbook.tldw_api.notes_workspace_schemas import (
    MAX_WORKSPACE_SOURCE_OWNER_ROWS,
)

from .contracts import (
    BoundedPageResult,
    CapabilityUnavailableError,
    MAX_RESEARCH_SELECTION_IDS,
    MAX_RESEARCH_SELECTION_ROWS,
    QualifiedWorkspaceRef,
    ResearchCatalogItem,
    ResearchCapability,
    ResearchSourcePreview,
    ResearchSourcePage,
    ResearchSourceSummary,
    ResearchWorkspaceSummary,
    SourceSelectionResult,
    SourceReadiness,
    SourceIdentityMismatchError,
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
from .source_readiness import normalize_server_readiness


_AUDITED_SERVICE_METHODS = {
    "list": "list_workspaces",
    "get": "list_workspaces",
    "create": "save_workspace",
    "update": "save_workspace",
    "duplicate": "save_workspace",
    "archive": "save_workspace",
    "restore": "save_workspace",
    "delete": "delete_workspace",
}
_AUDITED_CAPABILITY_REVISION = "server-notes-workspace-service-v1"
_SOURCE_CAPABILITY_NAMES = (
    "list_sources",
    "search_catalog",
    "attach_existing",
    "remove_source",
    "update_source",
    "preview_source",
    "get_readiness",
    "set_selected_scope",
    "reorder_sources",
)
_RECOVERY_BY_REASON = {
    "server_not_configured": "Configure a server.",
    "server_profile_missing": "Choose or configure a server profile.",
    "server_unavailable": "Retry or change the selected server.",
    "auth_required": "Reauthenticate with the selected server.",
    "stale_authorization": "Reauthenticate with the selected server.",
    "server_credentials_unavailable": "Restore server credentials and retry.",
    "credential_store_unavailable": "Restore secure credential storage and retry.",
}
_ServerResult = TypeVar("_ServerResult")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _page_bounds(limit: object, offset: object) -> tuple[int, int]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if type(offset) is not int or not 0 <= offset <= 10_000:
        raise ValueError("offset must be between 0 and 10000")
    return limit, offset


def _catalog_backing_page(
    result: object, *, expected_page: int
) -> tuple[list[Mapping[str, Any]], int]:
    raw_items = result.get("items") if isinstance(result, Mapping) else None
    pagination = result.get("pagination") if isinstance(result, Mapping) else None
    if (
        not isinstance(raw_items, list)
        or len(raw_items) > 100
        or any(not isinstance(item, Mapping) for item in raw_items)
        or not isinstance(pagination, Mapping)
        or type(pagination.get("page")) is not int
        or pagination["page"] != expected_page
        or type(pagination.get("results_per_page")) is not int
        or pagination["results_per_page"] != 100
        or type(pagination.get("total_items")) is not int
        or pagination["total_items"] < 0
    ):
        raise ValueError("Server catalog returned an invalid bounded page")
    total = pagination["total_items"]
    expected_count = min(100, max(total - ((expected_page - 1) * 100), 0))
    if len(raw_items) != expected_count:
        raise ValueError("Server catalog returned an inconsistent bounded page")
    return raw_items, total


class ServerResearchWorkspaceAdapter:
    """Use only the selected server context and server workspace service."""

    def __init__(
        self,
        service: ServerNotesWorkspaceService,
        server_context_provider: Any,
        *,
        id_factory: Callable[[], str] | None = None,
        media_scope_service: Any | None = None,
        operation_store: Any | None = None,
        association_scheduler: Any | None = None,
        operation_id_factory: Callable[[], str] | None = None,
        now_factory: Callable[[], str] | None = None,
    ) -> None:
        self._service = service
        self._context_provider = server_context_provider
        self._id_factory = id_factory or (lambda: f"workspace-{uuid4().hex}")
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
        context, profile_id, principal_id = self._active_identity()
        require_capability(self._capabilities_for_context(context), "list")
        rows = await self._server_call(
            self._service.list_workspaces(), context=context
        )
        summaries = tuple(
            self._summary(row, profile_id=profile_id, principal_id=principal_id)
            for row in rows
        )
        if include_archived:
            return summaries
        return tuple(summary for summary in summaries if not summary.archived)

    async def get_workspace(
        self, ref: QualifiedWorkspaceRef
    ) -> ResearchWorkspaceSummary | None:
        self._require_server_ref(ref)
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), "get")
        rows = await self._server_call(
            self._service.list_workspaces(), context=context
        )
        for row in rows:
            if str(row.get("id") or "").strip() == ref.workspace_id:
                return self._matching_summary(ref, row)
        return None

    async def create_workspace(
        self, *, name: str, description: str = "", template_id: str = ""
    ) -> ResearchWorkspaceSummary:
        context, profile_id, principal_id = self._active_identity()
        require_capability(self._capabilities_for_context(context), "create")
        if description.strip() or template_id.strip():
            capability = ResearchCapability(
                available=False,
                reason_code="server_field_unavailable",
                user_message=(
                    "The selected server cannot create workspace descriptions or templates."
                ),
                owner="server",
                recovery_action="Create a workspace with a name only.",
                capability_revision=self._capability_revision(context),
            )
            raise CapabilityUnavailableError(capability)
        row = await self._server_call(
            self._service.save_workspace(
                workspace_id=self._id_factory(), name=name
            ),
            context=context,
        )
        return self._summary(row, profile_id=profile_id, principal_id=principal_id)

    async def update_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        name: str | None = None,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), "update")
        version = self._require_version(expected_version, context)
        row = await self._server_call(
            self._service.save_workspace(
                workspace_id=ref.workspace_id, name=name, version=version
            ),
            context=context,
        )
        return self._matching_summary(ref, row)

    async def duplicate_workspace(
        self, ref: QualifiedWorkspaceRef, *, name: str
    ) -> ResearchWorkspaceSummary:
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), "duplicate")
        row = await self._server_call(
            self._service.save_workspace(
                workspace_id=self._id_factory(), name=name
            ),
            context=context,
        )
        return self._summary(
            row,
            profile_id=ref.server_profile_id,
            principal_id=ref.principal_id,
        )

    async def archive_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        return await self._set_archived(
            ref, archived=True, expected_version=expected_version
        )

    async def restore_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        return await self._set_archived(
            ref, archived=False, expected_version=expected_version
        )

    async def delete_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> bool:
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), "delete")
        if expected_version is not None:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="version_precondition_unavailable",
                    user_message=(
                        "The selected server cannot enforce a delete version precondition."
                    ),
                    owner="server",
                    recovery_action="Reload the workspace and delete without a version.",
                    capability_revision=self._capability_revision(context),
                )
            )
        result = await self._server_call(
            self._service.delete_workspace(ref.workspace_id), context=context
        )
        if isinstance(result, Mapping):
            return bool(result.get("deleted", True))
        return True

    async def capabilities(
        self, ref: QualifiedWorkspaceRef
    ) -> Mapping[str, ResearchCapability]:
        context = self._context_for_ref(ref)
        lifecycle = dict(self._capabilities_for_context(context))
        try:
            projection = await self._server_call(
                self._service.get_workspace_capabilities(ref.workspace_id),
                context=context,
            )
        except AttributeError:
            lifecycle.update(
                self._unavailable_source_capabilities(
                    context,
                    reason_code="server_capability_unavailable",
                    message=(
                        "The selected server does not expose source capabilities."
                    ),
                )
            )
            return lifecycle
        lifecycle.update(self._project_source_capabilities(ref, projection, context))
        return lifecycle

    async def list_sources(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> ResearchSourcePage:
        page_limit, page_offset = _page_bounds(limit, offset)
        await self._require_source_action(ref, "inspect_sources", allow_empty=True)
        context = self._context_for_ref(ref)
        rows = await self._server_call(
            self._service.list_workspace_sources(ref.workspace_id), context=context
        )
        if (
            not isinstance(rows, list)
            or len(rows) > MAX_WORKSPACE_SOURCE_OWNER_ROWS
        ):
            raise ValueError("Server source list is not a bounded page")
        normalized = tuple(self._source_summary(ref, row) for row in rows)
        page = normalized[page_offset : page_offset + page_limit]
        return ResearchSourcePage(
            items=page,
            limit=page_limit,
            offset=page_offset,
            total=len(normalized),
            has_more=page_offset + len(page) < len(normalized),
            desired_source_ids=tuple(
                source.source_id for source in normalized if source.selected
            ),
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
        page_limit, page_offset = _page_bounds(limit, offset)
        if not isinstance(query, str) or len(query.strip()) > 1000:
            raise ValueError("query is invalid")
        if sort_by not in {
            "relevance",
            "title_asc",
            "title_desc",
            "updated_asc",
            "updated_desc",
        }:
            raise ValueError("sort_by is unsupported")
        if len(source_types) > 25 or any(
            not isinstance(item, str) or not item.strip() or len(item) > 128
            for item in source_types
        ):
            raise ValueError("source_types is invalid")
        await self._require_source_action(ref, "add_sources")
        context = self._context_for_ref(ref)
        media_scope = self._require_media_scope(context)
        server_page = page_offset // 100 + 1
        page_inner_offset = page_offset % 100
        filters: dict[str, Any] = {
            "query": query.strip() or None,
            "sort_by": sort_by,
        }
        if source_types:
            filters["media_types"] = [item.strip() for item in source_types]
        result = await self._server_call(
            media_scope.search_backing_media_items(
                mode=MediaReadingBackend.SERVER,
                page=server_page,
                results_per_page=100,
                **filters,
            ),
            context=context,
        )
        raw_items, total = _catalog_backing_page(
            result, expected_page=server_page
        )
        combined = list(raw_items[page_inner_offset:])
        if len(combined) < page_limit and page_offset + len(combined) < total:
            next_result = await self._server_call(
                media_scope.search_backing_media_items(
                    mode=MediaReadingBackend.SERVER,
                    page=server_page + 1,
                    results_per_page=100,
                    **filters,
                ),
                context=context,
            )
            next_items, next_total = _catalog_backing_page(
                next_result, expected_page=server_page + 1
            )
            if next_total != total:
                raise ValueError("Server catalog total changed between pages")
            combined.extend(next_items)
        selected = combined[:page_limit]
        items = tuple(self._catalog_item(ref, row) for row in selected)
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
        if type(desired_selected) is not bool:
            raise ValueError("desired_selected must be bool")
        canonical_id = self._canonical_media_id(catalog_item_id)
        await self._require_source_action(ref, "add_sources")
        context = self._context_for_ref(ref)
        media_scope = self._require_media_scope(context)
        await self._server_call(
            media_scope.get_backing_media_by_identifier(
                mode=MediaReadingBackend.SERVER, media_id=int(canonical_id)
            ),
            context=context,
        )
        if self._operation_store is None or self._association_scheduler is None:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="source_operation_unavailable",
                    user_message="Durable source attachment is unavailable.",
                    owner="server",
                    recovery_action="Restart and retry.",
                )
            )
        now = self._now_factory()
        operation = ResearchSourceOperation(
            operation_id=self._operation_id_factory(),
            idempotency_key=idempotency_key,
            data_source=WorkspaceDataSource.SERVER,
            server_profile_id=ref.server_profile_id,
            principal_id=ref.principal_id,
            workspace_id=ref.workspace_id,
            canonical_item_type=CanonicalItemType.SERVER_MEDIA,
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
        self._context_for_ref(ref)
        if expected_version is not None:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="version_precondition_unavailable",
                    user_message=(
                        "The server cannot enforce a version check when removing a source."
                    ),
                    owner="server",
                    recovery_action="Refresh sources, then remove without a version check.",
                )
            )
        source_id = self._association_id(source_id)
        await self._require_source_action(ref, "add_sources")
        context = self._context_for_ref(ref)
        await self._server_call(
            self._service.delete_workspace_source(ref.workspace_id, source_id),
            context=context,
        )
        return True

    async def update_source(
        self,
        ref: QualifiedWorkspaceRef,
        source_id: str,
        *,
        title: str | None = None,
        expected_version: int | None = None,
    ) -> ResearchSourceSummary:
        source_id = self._association_id(source_id)
        await self._require_source_action(ref, "add_sources")
        version = self._require_version(expected_version, self._context_for_ref(ref))
        context = self._context_for_ref(ref)
        row = await self._server_call(
            self._service.save_workspace_source(
                workspace_id=ref.workspace_id,
                source_id=source_id,
                title=title,
                version=version,
            ),
            context=context,
        )
        return self._source_summary(ref, row)

    async def preview_source(
        self,
        ref: QualifiedWorkspaceRef,
        source_id: str,
        *,
        max_chars: int = 3000,
        snippet_limit: int = 3,
    ) -> ResearchSourcePreview:
        source_id = self._association_id(source_id)
        await self._require_source_action(ref, "inspect_sources")
        context = self._context_for_ref(ref)
        row = await self._server_call(
            self._service.preview_workspace_source(
                ref.workspace_id,
                source_id,
                max_chars=max_chars,
                chunk_limit=snippet_limit,
            ),
            context=context,
        )
        if str(row.get("workspace_id") or "") != ref.workspace_id or str(
            row.get("source_id") or ""
        ) != source_id:
            raise ValueError("Server preview returned mismatched source identity")
        snippets = row.get("snippets") or []
        if not isinstance(snippets, list):
            raise ValueError("Server preview snippets are invalid")
        media_id = row.get("media_id")
        if media_id is None or (type(media_id) is int and media_id == 0):
            catalog_item_id = None
        elif type(media_id) is not int or media_id < 1:
            raise ValueError("Server preview returned an invalid canonical Media id")
        else:
            catalog_item_id = str(media_id)
        return ResearchSourcePreview(
            ref=ref,
            source_id=source_id,
            catalog_item_id=catalog_item_id,
            preview_mode=str(row.get("preview_mode") or "unavailable"),
            text=str(row.get("text_preview") or ""),
            snippets=tuple(
                str(snippet.get("text") or "")
                for snippet in snippets
                if isinstance(snippet, Mapping)
            ),
        )

    async def get_readiness(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[SourceReadiness, ...]:
        source_ids = self._association_ids(source_ids, allow_empty=True)
        await self._require_source_action(ref, "inspect_sources", allow_empty=True)
        context = self._context_for_ref(ref)
        payload = await self._server_call(
            self._service.get_workspace_source_status(ref.workspace_id),
            context=context,
        )
        rows = payload.get("sources") if isinstance(payload, Mapping) else None
        if not isinstance(payload, Mapping):
            raise ValueError("Server readiness projection must be an object")
        if str(payload.get("workspace_id") or "") != ref.workspace_id:
            raise SourceIdentityMismatchError(
                "Server readiness returned a mismatched workspace"
            )
        if not isinstance(rows, list) or len(rows) > MAX_WORKSPACE_SOURCE_OWNER_ROWS:
            raise ValueError("Server readiness sources projection is invalid")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Server readiness source must be an object")
            if str(row.get("workspace_id") or "") != ref.workspace_id:
                raise SourceIdentityMismatchError(
                    "Server readiness returned a mismatched workspace"
                )
        requested = set(source_ids)
        normalized = tuple(normalize_server_readiness(ref=ref, status=row) for row in rows)
        if requested and not requested.issubset({row.source_id for row in normalized}):
            raise ValueError("source_ids contains an unattached source")
        return tuple(
            row for row in normalized if not requested or row.source_id in requested
        )

    async def set_selected_scope(
        self,
        ref: QualifiedWorkspaceRef,
        source_ids: tuple[str, ...],
    ) -> SourceSelectionResult:
        source_ids = self._association_ids(
            source_ids,
            allow_empty=True,
            maximum=MAX_RESEARCH_SELECTION_IDS,
        )
        await self._require_source_action(ref, "add_sources")
        context = self._context_for_ref(ref)
        rows = await self._server_call(
            self._service.set_workspace_source_selection(
                ref.workspace_id, list(source_ids)
            ),
            context=context,
        )
        if not isinstance(rows, list) or len(rows) > MAX_WORKSPACE_SOURCE_OWNER_ROWS:
            raise ValueError("Server source selection reconciliation is invalid")
        owner_ids: list[str] = []
        selected_sources: list[ResearchSourceSummary] = []
        seen_ids: set[str] = set()
        for row in rows:
            if (
                not isinstance(row, Mapping)
                or str(row.get("workspace_id") or "") != ref.workspace_id
                or type(row.get("selected")) is not bool
            ):
                raise ValueError("Server source selection returned mismatched identity")
            source_id = self._association_id(row.get("id"))
            if source_id in seen_ids:
                raise ValueError("Server source selection returned duplicate identity")
            seen_ids.add(source_id)
            if not row["selected"]:
                continue
            owner_ids.append(source_id)
            if len(selected_sources) < MAX_RESEARCH_SELECTION_ROWS:
                selected_sources.append(self._source_summary(ref, row))
        if frozenset(owner_ids) != frozenset(source_ids):
            raise ValueError("Server source selection reconciliation did not match")
        return SourceSelectionResult(
            ref=ref,
            desired_source_ids=tuple(owner_ids),
            sources=tuple(selected_sources),
        )

    async def reorder_sources(
        self,
        ref: QualifiedWorkspaceRef,
        ordered_source_ids: tuple[str, ...],
    ) -> tuple[ResearchSourceSummary, ...]:
        context = self._context_for_ref(ref)
        try:
            ordered_source_ids = self._association_ids(
                ordered_source_ids,
                allow_empty=False,
                maximum=MAX_WORKSPACE_SOURCE_OWNER_ROWS,
            )
        except ValueError as exc:
            raise self._reorder_precondition_error(context) from exc
        await self._require_source_action(ref, "add_sources")
        context = self._context_for_ref(ref)
        owner_rows = await self._server_call(
            self._service.list_workspace_sources(ref.workspace_id), context=context
        )
        if (
            not isinstance(owner_rows, list)
            or len(owner_rows) > MAX_WORKSPACE_SOURCE_OWNER_ROWS
        ):
            raise ValueError("Server source reorder owner projection is invalid")
        owner_ids = tuple(
            self._source_summary(ref, row).source_id for row in owner_rows
        )
        if (
            len(owner_ids) > 100
            or len(owner_ids) != len(set(owner_ids))
            or frozenset(owner_ids) != frozenset(ordered_source_ids)
        ):
            raise self._reorder_precondition_error(context)
        rows = await self._server_call(
            self._service.reorder_workspace_sources(
                ref.workspace_id, list(ordered_source_ids)
            ),
            context=context,
        )
        return tuple(self._source_summary(ref, row) for row in rows)

    @staticmethod
    def _reorder_precondition_error(context: Any) -> CapabilityUnavailableError:
        return CapabilityUnavailableError(
            ResearchCapability(
                available=False,
                reason_code="reorder_precondition_unavailable",
                user_message=(
                    "Source order cannot be changed without the exact bounded owner list."
                ),
                owner="server",
                recovery_action="Refresh sources and retry.",
                capability_revision=ServerResearchWorkspaceAdapter._capability_revision(
                    context
                ),
            )
        )

    async def _require_source_action(
        self,
        ref: QualifiedWorkspaceRef,
        action: str,
        *,
        allow_empty: bool = False,
    ) -> Mapping[str, Any]:
        context = self._context_for_ref(ref)
        projection = await self._server_call(
            self._service.get_workspace_capabilities(ref.workspace_id),
            context=context,
        )
        if (
            not isinstance(projection, Mapping)
            or str(projection.get("workspace_id") or "") != ref.workspace_id
        ):
            raise self._source_capability_error(
                "malformed_capability",
                "The selected server returned an invalid capability projection.",
                context,
            )
        actions = projection.get("allowed_actions")
        row = actions.get(action) if isinstance(actions, Mapping) else None
        if not isinstance(row, Mapping) or type(row.get("allowed")) is not bool:
            raise self._source_capability_error(
                "unknown_capability",
                "The selected server did not report this source capability.",
                context,
            )
        if not row["allowed"] and not (
            allow_empty and row.get("reason_code") == "no_sources"
        ):
            raise self._source_capability_error(
                str(row.get("reason_code") or "server_capability_unavailable"),
                "The selected server blocked this source action.",
                context,
            )
        if action == "add_sources" and projection.get("access_level") == "viewer":
            raise self._source_capability_error(
                "server_permission_denied",
                "The selected server workspace is read-only.",
                context,
            )
        return projection

    def _project_source_capabilities(
        self,
        ref: QualifiedWorkspaceRef,
        projection: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, ResearchCapability]:
        if str(projection.get("workspace_id") or "") != ref.workspace_id:
            unavailable = self._source_capability_error(
                "malformed_capability",
                "The selected server returned an invalid capability projection.",
                context,
            ).capability
            return {
                "list_sources": unavailable,
                "search_catalog": unavailable,
                "attach_existing": unavailable,
                "remove_source": unavailable,
                "update_source": unavailable,
                "preview_source": unavailable,
                "get_readiness": unavailable,
                "set_selected_scope": unavailable,
                "reorder_sources": unavailable,
            }
        actions = projection.get("allowed_actions")
        if not isinstance(actions, Mapping):
            actions = {}
        revision = self._capability_revision(context)

        def project(action: str, *, allow_empty: bool = False):
            row = actions.get(action)
            if not isinstance(row, Mapping) or type(row.get("allowed")) is not bool:
                return ResearchCapability(
                    available=False,
                    reason_code="unknown_capability",
                    user_message="The selected server did not report this source capability.",
                    owner="server",
                    recovery_action="Refresh capabilities or update the server.",
                    capability_revision=revision,
                )
            available = bool(row["allowed"]) or (
                allow_empty and row.get("reason_code") == "no_sources"
            )
            return ResearchCapability(
                available=available,
                reason_code=(
                    "available"
                    if available
                    else str(row.get("reason_code") or "server_capability_unavailable")
                ),
                user_message=(
                    "Available on the selected server."
                    if available
                    else "The selected server blocked this source action."
                ),
                owner="server",
                recovery_action="Refresh capabilities or review server permissions.",
                capability_revision=revision,
            )

        inspect = project("inspect_sources", allow_empty=True)
        mutate = project("add_sources")
        return {
            "list_sources": inspect,
            "search_catalog": mutate,
            "attach_existing": mutate,
            "remove_source": mutate,
            "update_source": mutate,
            "preview_source": project("inspect_sources"),
            "get_readiness": inspect,
            "set_selected_scope": mutate,
            "reorder_sources": mutate,
        }

    def _source_capability_error(
        self, reason_code: str, message: str, context: Any
    ) -> CapabilityUnavailableError:
        return CapabilityUnavailableError(
            ResearchCapability(
                available=False,
                reason_code=reason_code,
                user_message=message,
                owner="server",
                recovery_action="Refresh capabilities or review server permissions.",
                capability_revision=self._capability_revision(context),
            )
        )

    def _unavailable_source_capabilities(
        self,
        context: Any,
        *,
        reason_code: str,
        message: str,
    ) -> Mapping[str, ResearchCapability]:
        capability = ResearchCapability(
            available=False,
            reason_code=reason_code,
            user_message=message,
            owner="server",
            recovery_action="Refresh capabilities or update the server.",
            capability_revision=self._capability_revision(context),
        )
        return {name: capability for name in _SOURCE_CAPABILITY_NAMES}

    def _require_media_scope(self, context: Any) -> Any:
        if self._media_scope is None:
            raise self._source_capability_error(
                "server_media_unavailable",
                "The selected server Media catalog is unavailable.",
                context,
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
            and operation.data_source is WorkspaceDataSource.SERVER
            and operation.workspace_id == ref.workspace_id
            and operation.server_profile_id == ref.server_profile_id
            and operation.principal_id == ref.principal_id
            and operation.desired_selected is desired_selected
            and operation.catalog_status is SourceOperationStatus.SUCCEEDED
            and operation.canonical_item_id == canonical_id
        )

    @staticmethod
    def _canonical_media_id(value: object) -> str:
        if isinstance(value, bool):
            raise ValueError("catalog_item_id must be a positive Media id")
        normalized = str(value).strip()
        if not normalized.isdigit() or int(normalized) < 1:
            raise ValueError("catalog_item_id must be a positive Media id")
        return str(int(normalized))

    @staticmethod
    def _association_id(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source_ids must contain nonblank association IDs")
        normalized = value.strip()
        if len(normalized) > 1024 or len(normalized.encode("utf-8")) > 4096:
            raise ValueError("source_ids contain an oversized association ID")
        return normalized

    @classmethod
    def _association_ids(
        cls, values: object, *, allow_empty: bool, maximum: int = 100
    ) -> tuple[str, ...]:
        if not isinstance(values, tuple) or len(values) > maximum:
            raise ValueError("source_ids must be a bounded tuple of association IDs")
        normalized = tuple(cls._association_id(value) for value in values)
        if (not allow_empty and not normalized) or len(set(normalized)) != len(
            normalized
        ):
            raise ValueError("source_ids must contain unique association IDs")
        return normalized

    @staticmethod
    def _catalog_item(
        ref: QualifiedWorkspaceRef, row: Mapping[str, Any]
    ) -> ResearchCatalogItem:
        catalog_id = row.get("id", row.get("media_id"))
        version = row.get("version")
        return ResearchCatalogItem(
            ref=ref,
            catalog_item_id=str(catalog_id),
            title=str(row.get("title") or "Untitled"),
            source_type=str(row.get("type") or row.get("media_type") or "media"),
            catalog_item_version=version if type(version) is int else None,
            updated_at=str(row.get("last_modified") or row.get("updated_at") or ""),
        )

    @staticmethod
    def _source_summary(
        ref: QualifiedWorkspaceRef, row: Mapping[str, Any]
    ) -> ResearchSourceSummary:
        if str(row.get("workspace_id") or "") != ref.workspace_id:
            raise ValueError("Server returned a mismatched workspace source")
        source_id = str(row.get("id") or "")
        media_id = row.get("media_id")
        version = row.get("version")
        if type(media_id) is not int or media_id < 1:
            raise ValueError("Server returned an invalid canonical Media id")
        return ResearchSourceSummary(
            ref=ref,
            source_id=source_id,
            catalog_item_id=str(media_id),
            title=str(row.get("title") or "Untitled"),
            source_type=str(row.get("source_type") or "media"),
            ready=False,
            version=version if type(version) is int else None,
            catalog_item_version=None,
            selected=bool(row.get("selected", True)),
            position=int(row.get("position") or 0),
        )

    async def _set_archived(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        archived: bool,
        expected_version: int | None,
    ) -> ResearchWorkspaceSummary:
        operation = "archive" if archived else "restore"
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), operation)
        version = self._require_version(expected_version, context)
        row = await self._server_call(
            self._service.save_workspace(
                workspace_id=ref.workspace_id,
                archived=archived,
                version=version,
            ),
            context=context,
        )
        return self._matching_summary(ref, row)

    def _active_identity(self) -> tuple[Any, str, str]:
        try:
            context = self._context_provider.get_active_context()
        except ServerContextError as exc:
            raise CapabilityUnavailableError(
                self._context_failure_capability(exc)
            ) from exc
        profile_id = str(getattr(context, "active_server_id", "") or "").strip()
        if not profile_id:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="server_profile_missing",
                    user_message="Active server profile is unavailable.",
                    owner="server",
                    recovery_action="Choose or configure a server profile.",
                )
            )
        principal_id = event_principal_id_from_active_context(context) or ""
        return context, profile_id, principal_id

    def _context_for_ref(self, ref: QualifiedWorkspaceRef) -> Any:
        self._require_server_ref(ref)
        context, profile_id, principal_id = self._active_identity()
        if profile_id != ref.server_profile_id or principal_id != ref.principal_id:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="server_context_changed",
                    user_message="The selected server identity changed.",
                    owner="server",
                    recovery_action="Reload the selected server workspace.",
                    capability_revision=self._capability_revision(context),
                )
            )
        return context

    def _capabilities_for_context(
        self, context: Any
    ) -> Mapping[str, ResearchCapability]:
        revision = self._capability_revision(context)
        unavailable = self._context_health_unavailable(context, revision=revision)
        result: dict[str, ResearchCapability] = {}
        for operation, service_method in _AUDITED_SERVICE_METHODS.items():
            if unavailable is not None:
                result[operation] = unavailable
            elif callable(getattr(self._service, service_method, None)):
                result[operation] = ResearchCapability(
                    available=True,
                    reason_code="available",
                    user_message="Available on the selected server.",
                    owner="server",
                    capability_revision=revision,
                )
            else:
                result[operation] = ResearchCapability(
                    available=False,
                    reason_code="server_capability_unavailable",
                    user_message=(
                        f"The selected server service cannot perform workspace {operation}."
                    ),
                    owner="server",
                    recovery_action="Choose another action or update the server service.",
                    capability_revision=revision,
                )
        return result

    @staticmethod
    def _context_health_unavailable(
        context: Any, *, revision: str
    ) -> ResearchCapability | None:
        capabilities = getattr(context, "capabilities", {})
        if not isinstance(capabilities, Mapping):
            capabilities = {}
        if capabilities.get("server_configured") is False:
            return ResearchCapability(
                available=False,
                reason_code="server_not_configured",
                user_message="A server is not configured.",
                owner="server",
                recovery_action="Configure a server.",
                capability_revision=revision,
            )
        if capabilities.get("reachability") == "unreachable":
            return ResearchCapability(
                available=False,
                reason_code="server_unavailable",
                user_message="The selected server is unavailable.",
                owner="server",
                recovery_action="Retry or change the selected server.",
                capability_revision=revision,
            )
        auth_state = capabilities.get("auth_state")
        if auth_state in {"auth_required", "session_invalid"}:
            stale = auth_state == "session_invalid"
            return ResearchCapability(
                available=False,
                reason_code="stale_authorization" if stale else "auth_required",
                user_message=(
                    "Authorization with the selected server is stale."
                    if stale
                    else "Authentication with the selected server is required."
                ),
                owner="server",
                recovery_action="Reauthenticate with the selected server.",
                capability_revision=revision,
            )
        return None

    async def _server_call(
        self, operation: Awaitable[_ServerResult], *, context: Any
    ) -> _ServerResult:
        try:
            return await operation
        except ServerContextError as exc:
            raise CapabilityUnavailableError(
                self._context_failure_capability(exc)
            ) from exc
        except AuthenticationError as exc:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="auth_required",
                    user_message="Authentication with the selected server is required.",
                    owner="server",
                    recovery_action="Reauthenticate with the selected server.",
                    capability_revision=self._capability_revision(context),
                )
            ) from exc
        except PolicyDeniedError as exc:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code=exc.reason_code,
                    user_message=exc.user_message,
                    owner=exc.authority_owner,
                    recovery_action="Review server permissions and retry.",
                    capability_revision=self._capability_revision(context),
                )
            ) from exc
        except APIResponseError as exc:
            permission_denied = exc.status_code == 403
            capability_missing = exc.status_code in {404, 405, 501}
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code=(
                        "server_permission_denied"
                        if permission_denied
                        else (
                            "server_capability_unavailable"
                            if capability_missing
                            else "server_request_failed"
                        )
                    ),
                    user_message=(
                        "The selected server denied this workspace action."
                        if permission_denied
                        else (
                            "The selected server does not expose this workspace action."
                            if capability_missing
                            else "The selected server could not complete this action."
                        )
                    ),
                    owner="server",
                    recovery_action=(
                        "Review server permissions and retry."
                        if permission_denied
                        else (
                            "Update the selected server or choose another action."
                            if capability_missing
                            else "Retry or review server diagnostics."
                        )
                    ),
                    capability_revision=self._capability_revision(context),
                )
            ) from exc
        except (APIConnectionError, ConnectionError, OSError, TimeoutError) as exc:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="server_unavailable",
                    user_message="The selected server is unavailable.",
                    owner="server",
                    recovery_action="Retry or change the selected server.",
                    capability_revision=self._capability_revision(context),
                )
            ) from exc

    @staticmethod
    def _capability_revision(context: Any) -> str:
        capabilities = getattr(context, "capabilities", {})
        if not isinstance(capabilities, Mapping):
            capabilities = {}
        reachability = str(capabilities.get("reachability") or "unknown").strip()
        auth_state = str(capabilities.get("auth_state") or "unknown").strip()
        return f"{_AUDITED_CAPABILITY_REVISION}:{reachability}:{auth_state}"

    @staticmethod
    def _context_failure_capability(exc: ServerContextError) -> ResearchCapability:
        reason_code = str(getattr(exc, "reason_code", "server_unavailable"))
        return ResearchCapability(
            available=False,
            reason_code=reason_code,
            user_message=str(exc),
            owner="server",
            recovery_action=_RECOVERY_BY_REASON.get(
                reason_code, "Retry or change the selected server."
            ),
        )

    @staticmethod
    def _require_server_ref(ref: QualifiedWorkspaceRef) -> None:
        if ref.data_source is not WorkspaceDataSource.SERVER:
            raise ValueError("Server adapter requires a Server workspace ref")

    @staticmethod
    def _require_version(expected_version: int | None, context: Any) -> int:
        if type(expected_version) is int and expected_version >= 0:
            return expected_version
        raise CapabilityUnavailableError(
            ResearchCapability(
                available=False,
                reason_code="version_required",
                user_message="Reload this server workspace before changing it.",
                owner="server",
                recovery_action="Reload the workspace and retry.",
                capability_revision=ServerResearchWorkspaceAdapter._capability_revision(
                    context
                ),
            )
        )

    @staticmethod
    def _summary(
        row: Mapping[str, Any], *, profile_id: str, principal_id: str
    ) -> ResearchWorkspaceSummary:
        workspace_id = str(row.get("id") or "").strip()
        ref = QualifiedWorkspaceRef(
            WorkspaceDataSource.SERVER,
            workspace_id,
            server_profile_id=profile_id,
            principal_id=principal_id,
        )
        version_value = row.get("version")
        version = (
            int(version_value)
            if type(version_value) in {int, str}
            and str(version_value).strip().isdigit()
            else None
        )
        return ResearchWorkspaceSummary(
            ref=ref,
            name=str(row.get("name") or "").strip(),
            description=str(row.get("description") or "").strip(),
            archived=bool(row.get("archived", False)),
            version=version,
            updated_at=str(row.get("updated_at") or "").strip(),
        )

    def _matching_summary(
        self, expected_ref: QualifiedWorkspaceRef, row: Mapping[str, Any]
    ) -> ResearchWorkspaceSummary:
        summary = self._summary(
            row,
            profile_id=expected_ref.server_profile_id,
            principal_id=expected_ref.principal_id,
        )
        if summary.ref != expected_ref:
            raise ValueError("Adapter returned a mismatched workspace ref")
        return summary
