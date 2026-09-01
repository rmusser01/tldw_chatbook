"""Headless, generation-fenced state for the Collections capture reader."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Literal

from tldw_chatbook.Library.collections_capture_models import (
    CAPTURE_PAGE_SIZE,
    CaptureAuthority,
    CaptureConflict,
    CaptureConflictError,
    CaptureDetail,
    CaptureIdentity,
    CapturePage,
    CapturePageRequest,
    CaptureSummary,
    CollectionsCaptureError,
    ResolvedCaptureDetail,
)
from tldw_chatbook.Library.collections_capture_service import (
    CollectionsCaptureBackend,
    CollectionsCaptureScopeService,
)


@dataclass(frozen=True)
class CaptureRequestFence:
    """Complete identity of one in-flight controller request."""

    destination: Literal["collections"]
    authority_key: str
    scope_key: str
    item_id: str | None
    revision: str | int | None
    generation: int


@dataclass(frozen=True)
class CaptureArchiveReceipt:
    """Session-only Undo receipt retained under its originating authority."""

    identity: CaptureIdentity
    previous_status: str
    archived_revision: int
    created_at: float


@dataclass(frozen=True)
class CollectionsCaptureControllerState:
    """Immutable render state owned by the Collections reader controller."""

    mounted: bool = True
    authority_key: str | None = None
    requested_scope: CapturePageRequest | None = None
    applied_scope: CapturePageRequest | None = None
    page: CapturePage | None = None
    page_loading: bool = False
    page_stale: bool = False
    page_error: str | None = None
    selected_identity: CaptureIdentity | None = None
    loaded_detail: ResolvedCaptureDetail | None = None
    detail_loading: bool = False
    detail_error: str | None = None
    mutation_loading: bool = False
    mutation_error: str | None = None
    conflict: CaptureConflict | None = None
    conflict_draft: Mapping[str, Any] | None = None
    extraction_loading: bool = False
    extraction_error: str | None = None
    visible_archive_receipts: tuple[CaptureArchiveReceipt, ...] = ()

    @property
    def exact_total(self) -> int | None:
        if self.page is None or self.page_stale or self.page_loading:
            return None
        return self.page.total

    @property
    def paging_enabled(self) -> bool:
        return (
            self.mounted
            and self.page is not None
            and not self.page_stale
            and not self.page_loading
        )

    @property
    def identity_actions_enabled(self) -> bool:
        loaded_identity = (
            self.loaded_detail.capture.identity
            if self.loaded_detail is not None
            else None
        )
        return (
            self.mounted
            and self.selected_identity is not None
            and self.selected_identity == loaded_identity
            and not self.detail_loading
            and not self.mutation_loading
            and not self.extraction_loading
            and not self.page_stale
        )

    @property
    def retained_reader_copy(self) -> str | None:
        if (
            not self.detail_loading
            or self.selected_identity is None
            or self.loaded_detail is None
            or self.selected_identity == self.loaded_detail.capture.identity
        ):
            return None
        selected_title = self.selected_identity.capture_id
        if self.page is not None:
            selected = next(
                (
                    item
                    for item in self.page.items
                    if item.identity == self.selected_identity
                ),
                None,
            )
            if selected is not None and selected.title:
                selected_title = selected.title
        loaded = self.loaded_detail.capture
        loaded_title = loaded.title or loaded.identity.capture_id
        return f"Loading “{selected_title}”… showing “{loaded_title}” until ready."


Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


class LibraryCollectionsCaptureController:
    """Single orchestration owner for one mounted Collections reader session."""

    _OPERATIONS = ("list", "detail", "mutation", "extraction")

    def __init__(
        self,
        scope_service: CollectionsCaptureScopeService,
        *,
        detail_settle_seconds: float = 0.18,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        if detail_settle_seconds < 0:
            raise CollectionsCaptureError("invalid_detail_settle_seconds")
        self.scope_service = scope_service
        self.detail_settle_seconds = detail_settle_seconds
        self._sleep = sleep
        self._clock = clock
        self.state = CollectionsCaptureControllerState()
        self._generations = {operation: 0 for operation in self._OPERATIONS}
        self._fences: dict[str, CaptureRequestFence] = {}
        self._archive_receipts: dict[tuple[str, str], CaptureArchiveReceipt] = {}

    def activate(
        self,
        authority: CaptureAuthority,
        backend: CollectionsCaptureBackend,
    ) -> None:
        """Replace the active authority and invalidate every pending result."""
        self.scope_service.activate(authority, backend)
        self._invalidate_all()
        self.state = CollectionsCaptureControllerState(
            mounted=True,
            authority_key=authority.key,
            visible_archive_receipts=self._visible_receipts(authority.key),
        )

    def adopt_active_authority(self) -> bool:
        """Adopt the authority already activated by the app-owned scope.

        Returns:
            ``True`` when visible authority state changed and pending work was
            invalidated, otherwise ``False``.
        """
        authority = self.scope_service.active_authority
        authority_key = authority.key if authority is not None else None
        unavailable = authority is None
        if (
            self.state.mounted
            and self.state.authority_key == authority_key
            and (self.state.page_error == "capture_authority_unavailable")
            == unavailable
        ):
            return False
        self._invalidate_all()
        self.state = CollectionsCaptureControllerState(
            mounted=True,
            authority_key=authority_key,
            page_error="capture_authority_unavailable" if unavailable else None,
            visible_archive_receipts=(
                self._visible_receipts(authority_key) if authority_key else ()
            ),
        )
        return True

    def unmount(self) -> None:
        """Invalidate pending work and clear renderable authority data."""
        self._invalidate_all()
        self.state = CollectionsCaptureControllerState(mounted=False)

    def _invalidate_all(self) -> None:
        for operation in self._OPERATIONS:
            self._generations[operation] += 1
        self._fences.clear()

    def _invalidate(self, operation: str) -> None:
        self._generations[operation] += 1
        self._fences.pop(operation, None)

    @staticmethod
    def _scope_key(request: CapturePageRequest | None) -> str:
        if request is None:
            return "unscoped"
        values = (
            request.authority_key,
            request.search,
            request.statuses,
            request.favorite,
            request.tags,
            request.domain,
            request.date_from,
            request.date_to,
            request.sort,
            request.page,
            request.size,
        )
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()[:24]

    def _new_fence(
        self,
        operation: str,
        *,
        request: CapturePageRequest | None = None,
        identity: CaptureIdentity | None = None,
        revision: str | int | None = None,
    ) -> CaptureRequestFence:
        authority_key = self.state.authority_key
        if not self.state.mounted or authority_key is None:
            raise CollectionsCaptureError("capture_session_unavailable")
        self._generations[operation] += 1
        fence = CaptureRequestFence(
            "collections",
            authority_key,
            self._scope_key(request or self.state.requested_scope),
            identity.capture_id if identity is not None else None,
            revision,
            self._generations[operation],
        )
        self._fences[operation] = fence
        return fence

    def _is_current(self, operation: str, fence: CaptureRequestFence) -> bool:
        return (
            self.state.mounted
            and self.state.authority_key == fence.authority_key
            and self._fences.get(operation) == fence
        )

    @staticmethod
    def _reason(error: BaseException, fallback: str) -> str:
        if isinstance(error, CollectionsCaptureError):
            return error.reason
        return fallback

    def _validate_authority(self, authority_key: str) -> None:
        if authority_key != self.state.authority_key:
            raise CollectionsCaptureError("authority_mismatch")

    async def load_page(self, request: CapturePageRequest) -> bool:
        """Load one exact page, retrying a concurrent last-page shrink once."""
        self._validate_authority(request.authority_key)
        return await self._start_page_load(request, retried=False)

    async def _start_page_load(
        self,
        request: CapturePageRequest,
        *,
        retried: bool,
    ) -> bool:
        fence = self._new_fence("list", request=request)
        self.state = replace(
            self.state,
            requested_scope=request,
            page_loading=True,
            page_stale=self.state.page is not None,
            page_error=None,
        )
        try:
            page = await self.scope_service.list_page(request)
        except Exception as exc:
            if not self._is_current("list", fence):
                return False
            self.state = replace(
                self.state,
                page_loading=False,
                page_stale=self.state.page is not None,
                page_error=self._reason(exc, "page_load_failed"),
            )
            return False
        if not self._is_current("list", fence):
            return False
        if page.applied != request:
            self.state = replace(
                self.state,
                page_loading=False,
                page_stale=self.state.page is not None,
                page_error="page_scope_mismatch",
            )
            return False
        if self._is_out_of_range(page):
            if retried:
                self.state = replace(
                    self.state,
                    page_loading=False,
                    page_stale=self.state.page is not None,
                    page_error="page_changed_again",
                )
                return False
            last_page = max(
                1, (page.total + CAPTURE_PAGE_SIZE - 1) // CAPTURE_PAGE_SIZE
            )
            retry = replace(request, page=last_page)
            return await self._start_page_load(retry, retried=True)
        self._apply_page(page)
        return True

    @staticmethod
    def _is_out_of_range(page: CapturePage) -> bool:
        return (
            page.applied.page > 1
            and not page.items
            and (page.applied.page - 1) * page.applied.size >= page.total
        )

    def _apply_page(self, page: CapturePage) -> None:
        identities = tuple(item.identity for item in page.items)
        selected = self.state.selected_identity
        if selected not in identities:
            selected = identities[0] if identities else None
            self._invalidate("detail")
        loaded = self.state.loaded_detail
        if selected is None:
            loaded = None
        self.state = replace(
            self.state,
            requested_scope=page.applied,
            applied_scope=page.applied,
            page=page,
            page_loading=False,
            page_stale=False,
            page_error=None,
            selected_identity=selected,
            loaded_detail=loaded,
            detail_loading=False,
            detail_error=None,
        )

    async def select_item(self, identity: CaptureIdentity) -> bool:
        """Select immediately, then settle before loading its detail."""
        self._validate_selectable(identity)
        fence = self._begin_detail(identity)
        if self._loaded_identity() == identity:
            self.state = replace(self.state, detail_loading=False)
            return True
        await self._sleep(self.detail_settle_seconds)
        if not self._is_current("detail", fence):
            return False
        return await self._load_detail(identity, fence)

    async def load_selected_now(self) -> bool:
        """Bypass settling and load the selected capture immediately."""
        identity = self.state.selected_identity
        if identity is None:
            raise CollectionsCaptureError("capture_selection_unavailable")
        self._validate_selectable(identity)
        fence = self._begin_detail(identity)
        if self._loaded_identity() == identity:
            self.state = replace(self.state, detail_loading=False)
            return True
        return await self._load_detail(identity, fence)

    def _begin_detail(self, identity: CaptureIdentity) -> CaptureRequestFence:
        revision = self._summary_revision(identity)
        fence = self._new_fence("detail", identity=identity, revision=revision)
        self.state = replace(
            self.state,
            selected_identity=identity,
            detail_loading=True,
            detail_error=None,
            conflict=None,
            conflict_draft=None,
        )
        return fence

    async def _load_detail(
        self,
        identity: CaptureIdentity,
        fence: CaptureRequestFence,
    ) -> bool:
        try:
            detail = await self.scope_service.get_detail(identity)
        except Exception as exc:
            if not self._is_current("detail", fence):
                return False
            self.state = replace(
                self.state,
                detail_loading=False,
                detail_error=self._reason(exc, "detail_load_failed"),
            )
            return False
        if not self._is_current("detail", fence):
            return False
        if detail.capture.identity != identity:
            self.state = replace(
                self.state,
                detail_loading=False,
                detail_error="detail_identity_mismatch",
            )
            return False
        self.state = replace(
            self.state,
            loaded_detail=detail,
            detail_loading=False,
            detail_error=None,
        )
        return True

    def _validate_selectable(self, identity: CaptureIdentity) -> None:
        self._validate_authority(identity.authority_key)
        if self.state.page is None or identity not in {
            item.identity for item in self.state.page.items
        }:
            raise CollectionsCaptureError("capture_not_in_applied_page")

    def _loaded_identity(self) -> CaptureIdentity | None:
        if self.state.loaded_detail is None:
            return None
        return self.state.loaded_detail.capture.identity

    def _summary_revision(self, identity: CaptureIdentity) -> int | None:
        if self.state.page is None:
            return None
        item = next(
            (item for item in self.state.page.items if item.identity == identity),
            None,
        )
        return item.revision if item is not None else None

    async def update_selected(self, changes: Mapping[str, Any]) -> bool:
        """Apply one revisioned edit and refresh the applied page."""
        identity, current = self._mutation_target()
        draft = MappingProxyType(dict(changes))
        fence = self._begin_mutation(identity, current.revision)
        try:
            changed = await self.scope_service.update_capture(
                identity,
                current.revision,
                dict(changes),
            )
        except CaptureConflictError as exc:
            if not self._is_current("mutation", fence):
                return False
            self.state = replace(
                self.state,
                mutation_loading=False,
                mutation_error=exc.reason,
                conflict=exc.conflict,
                conflict_draft=draft,
            )
            return False
        except Exception as exc:
            if not self._is_current("mutation", fence):
                return False
            self.state = replace(
                self.state,
                mutation_loading=False,
                mutation_error=self._reason(exc, "mutation_failed"),
                conflict_draft=draft,
            )
            return False
        if not self._is_current("mutation", fence):
            return False
        self._apply_mutation(changed)
        await self._refresh_after_mutation()
        return True

    async def archive_selected(self) -> bool:
        """Move the loaded capture to Archive and retain one authority receipt."""
        identity, current = self._mutation_target()
        previous_status = current.status
        fence = self._begin_mutation(identity, current.revision)
        try:
            changed = await self.scope_service.archive(identity, current.revision)
        except Exception as exc:
            return self._finish_mutation_error(fence, exc)
        if not self._is_current("mutation", fence):
            return False
        receipt = CaptureArchiveReceipt(
            identity,
            previous_status,
            changed.revision,
            self._clock(),
        )
        self._archive_receipts[(identity.authority_key, identity.capture_id)] = receipt
        self._apply_mutation(changed)
        self._sync_visible_receipts()
        await self._refresh_after_mutation()
        return True

    async def undo_archive(self, identity: CaptureIdentity) -> bool:
        """Restore the prior status recorded by the originating authority receipt."""
        self._validate_authority(identity.authority_key)
        key = (identity.authority_key, identity.capture_id)
        receipt = self._archive_receipts.get(key)
        if receipt is None:
            raise CollectionsCaptureError("archive_receipt_unavailable")
        fence = self._begin_mutation(identity, receipt.archived_revision)
        try:
            changed = await self.scope_service.undo_archive(
                identity,
                receipt.archived_revision,
            )
        except Exception as exc:
            return self._finish_mutation_error(fence, exc)
        if not self._is_current("mutation", fence):
            return False
        self._archive_receipts.pop(key, None)
        if self._loaded_identity() == identity:
            self._apply_mutation(changed)
        else:
            self.state = replace(
                self.state,
                mutation_loading=False,
                mutation_error=None,
            )
        self._sync_visible_receipts()
        await self._refresh_after_mutation()
        return True

    def _mutation_target(self) -> tuple[CaptureIdentity, CaptureDetail]:
        if not self.state.identity_actions_enabled or self.state.loaded_detail is None:
            raise CollectionsCaptureError("identity_actions_unavailable")
        return (
            self.state.loaded_detail.capture.identity,
            self.state.loaded_detail.capture,
        )

    def _begin_mutation(
        self,
        identity: CaptureIdentity,
        revision: int,
    ) -> CaptureRequestFence:
        self._invalidate("detail")
        fence = self._new_fence("mutation", identity=identity, revision=revision)
        self.state = replace(
            self.state,
            detail_loading=False,
            mutation_loading=True,
            mutation_error=None,
            conflict=None,
            conflict_draft=None,
        )
        return fence

    def _finish_mutation_error(
        self,
        fence: CaptureRequestFence,
        error: BaseException,
    ) -> bool:
        if not self._is_current("mutation", fence):
            return False
        self.state = replace(
            self.state,
            mutation_loading=False,
            mutation_error=self._reason(error, "mutation_failed"),
        )
        return False

    def _apply_mutation(self, changed: CaptureDetail) -> None:
        loaded = self.state.loaded_detail
        if loaded is not None and loaded.capture.identity == changed.identity:
            loaded = replace(loaded, capture=changed)
        page = self.state.page
        if page is not None:
            items = tuple(
                self._as_summary(changed) if item.identity == changed.identity else item
                for item in page.items
            )
            page = replace(page, items=items)
        self.state = replace(
            self.state,
            loaded_detail=loaded,
            page=page,
            mutation_loading=False,
            mutation_error=None,
            conflict=None,
            conflict_draft=None,
        )

    @staticmethod
    def _as_summary(detail: CaptureDetail) -> CaptureSummary:
        return CaptureSummary(
            **{
                field.name: getattr(detail, field.name)
                for field in fields(CaptureSummary)
            }
        )

    async def _refresh_after_mutation(self) -> None:
        request = self.state.applied_scope or self.state.requested_scope
        if request is not None and request.authority_key == self.state.authority_key:
            await self.load_page(request)

    async def retry_extraction(self) -> bool:
        """Retry extraction under its own fence and refresh current data."""
        identity, current = self._mutation_target()
        fence = self._new_fence(
            "extraction",
            identity=identity,
            revision=current.revision,
        )
        self.state = replace(
            self.state,
            extraction_loading=True,
            extraction_error=None,
        )
        try:
            await self.scope_service.retry_extraction(identity)
        except Exception as exc:
            if not self._is_current("extraction", fence):
                return False
            self.state = replace(
                self.state,
                extraction_loading=False,
                extraction_error=self._reason(exc, "extraction_retry_failed"),
            )
            return False
        if not self._is_current("extraction", fence):
            return False
        self.state = replace(
            self.state,
            extraction_loading=False,
            extraction_error=None,
        )
        await self._refresh_after_mutation()
        return True

    def _visible_receipts(
        self,
        authority_key: str,
    ) -> tuple[CaptureArchiveReceipt, ...]:
        return tuple(
            sorted(
                (
                    receipt
                    for (owner, _capture_id), receipt in self._archive_receipts.items()
                    if owner == authority_key
                ),
                key=lambda receipt: receipt.created_at,
                reverse=True,
            )
        )

    def _sync_visible_receipts(self) -> None:
        authority_key = self.state.authority_key
        visible = self._visible_receipts(authority_key) if authority_key else ()
        self.state = replace(self.state, visible_archive_receipts=visible)


__all__ = [
    "CaptureArchiveReceipt",
    "CaptureRequestFence",
    "CollectionsCaptureControllerState",
    "LibraryCollectionsCaptureController",
]
