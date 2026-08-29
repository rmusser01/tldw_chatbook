"""Non-visual orchestration for exact top-level Library Collection pages."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Mapping

from loguru import logger

from ...Library.library_collections_state import (
    COLLECTION_BROWSE_PAGE_SIZE,
    CollectionBrowseResult,
    CollectionBrowseScope,
    CollectionLocatorResult,
    build_collection_browse_result,
    build_collection_locator_result,
    validate_collection_browse_items,
)
from ...Library.library_pager_state import (
    LibraryPagerDisplay,
    PageFreshness,
    build_library_pager_display,
)


_WORKER_GROUP = "library-collections-browse"
_SERVICE_ERROR = "Couldn't load Collections. Check the local Library and retry."
_LOCATOR_ERROR = "Couldn't locate that Collection."
_SHRINK_COPY = "Source changed again; try again."
_MUTATION_COPY = "Collections changed; retry to load a current page."


class LibraryCollectionsBrowseController:
    """Own requested/applied Collection pages and stable-ID locator reads."""

    def __init__(
        self,
        *,
        screen: Any,
        run_service_call: Callable[[], Callable[..., Awaitable[Any]]],
        collections_service: Callable[[], Any],
        sync_view: Callable[[], Callable[..., None]],
        request_is_active: Callable[[], bool],
    ) -> None:
        self._screen = screen
        self._run_service_call = run_service_call
        self._collections_service = collections_service
        self._sync_view = sync_view
        self._request_is_active = request_is_active

        self.requested_scope = CollectionBrowseScope()
        self.inflight_scope: CollectionBrowseScope | None = None
        self.applied_result: CollectionBrowseResult | None = None
        self.retained_items: tuple[Mapping[str, Any], ...] = ()
        self.freshness: PageFreshness = "uninitialized"
        self.loading = False
        self.error_copy = ""
        self.stale_copy = ""
        self.located_target_id: str | None = None
        self._retry_locator_target_id: str | None = None
        self._generation = 0

    @property
    def _run_worker(self) -> Callable[..., Any]:
        return self._screen.run_worker

    @property
    def applied_scope(self) -> CollectionBrowseScope | None:
        return self.applied_result.scope if self.applied_result is not None else None

    @property
    def mutation_refresh_scope(self) -> CollectionBrowseScope:
        return self.applied_scope or self.requested_scope

    def scope_for_page(self, page: int) -> CollectionBrowseScope:
        """Return a page-only scope derived from the last applied request."""

        return self.mutation_refresh_scope.with_page(page)

    @property
    def pager(self) -> LibraryPagerDisplay:
        """Return the complete truthful pager projection."""

        applied = self.applied_result
        return build_library_pager_display(
            applied_page=applied.scope.page if applied is not None else None,
            requested_page=(
                self.inflight_scope.page
                if self.loading and self.inflight_scope is not None
                else self.requested_scope.page
                if self.error_copy or applied is None
                else applied.scope.page
            ),
            page_size=(
                applied.limit
                if applied is not None
                else COLLECTION_BROWSE_PAGE_SIZE
            ),
            row_count=len(self.retained_items),
            total=(
                applied.total
                if applied is not None and self.freshness == "fresh"
                else None
            ),
            freshness=self.freshness,
            loading=self.loading,
            error_copy=self.error_copy,
            stale_copy=self.stale_copy,
        )

    def _sync(self, focus_identity: str | None) -> None:
        self._sync_view()(focus_identity)

    def begin(self, scope: CollectionBrowseScope) -> int:
        """Fence older work and begin one page request generation."""

        if not isinstance(scope, CollectionBrowseScope):
            raise TypeError("scope must be a CollectionBrowseScope.")
        self._generation += 1
        self.requested_scope = scope
        self.inflight_scope = scope
        self.loading = True
        self.error_copy = ""
        self.located_target_id = None
        self._retry_locator_target_id = None
        return self._generation

    def request(
        self, scope: CollectionBrowseScope, *, focus_identity: str | None
    ) -> Any | None:
        """Dispatch one exact Collection page read when the route is active."""

        if not isinstance(scope, CollectionBrowseScope):
            raise TypeError("scope must be a CollectionBrowseScope.")
        if not self._request_is_active():
            return None
        generation = self.begin(scope)
        self._sync(focus_identity)
        return self._run_worker(
            self._load(scope, generation=generation, focus_identity=focus_identity),
            exclusive=True,
            group=_WORKER_GROUP,
        )

    def retry(self, *, focus_identity: str | None) -> Any | None:
        """Retry the failed locator or exact requested page."""

        if self._retry_locator_target_id is not None:
            return self.request_locator(
                self._retry_locator_target_id, focus_identity=focus_identity
            )
        return self.request(self.requested_scope, focus_identity=focus_identity)

    async def _list(self, scope: CollectionBrowseScope) -> CollectionBrowseResult:
        service = self._collections_service()
        list_collections = getattr(service, "list_library_collections", None)
        if not callable(list_collections):
            raise RuntimeError("Collections service unavailable")
        payload = await self._run_service_call()(
            list_collections,
            limit=scope.page_size,
            offset=scope.offset,
            isolate_in_worker=True,
        )
        return build_collection_browse_result(scope, payload)

    def _current(self, generation: int) -> bool:
        return generation == self._generation and self._request_is_active()

    async def _load(
        self,
        scope: CollectionBrowseScope,
        *,
        generation: int,
        focus_identity: str | None,
    ) -> None:
        fetched_scope = scope
        clamped = False
        try:
            while True:
                result = await self._list(fetched_scope)
                if not self._current(generation):
                    return
                if not result.out_of_range:
                    self._apply(
                        result, generation=generation, focus_identity=focus_identity
                    )
                    return
                if clamped:
                    self.loading = False
                    self.inflight_scope = None
                    if self.applied_result is not None:
                        self.freshness = "stale"
                        self.error_copy = ""
                        self.stale_copy = _SHRINK_COPY
                    else:
                        self.error_copy = _SERVICE_ERROR
                    self._sync(focus_identity)
                    return
                clamped = True
                fetched_scope = scope.with_page(result.last_page)
                self.inflight_scope = fetched_scope
                self._sync(focus_identity)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._current(generation):
                return
            logger.warning(
                "Library Collections browse failed; operation=list_collections "
                "exception_type={}",
                type(exc).__name__,
            )
            self.loading = False
            self.inflight_scope = None
            if self.freshness != "stale":
                self.error_copy = self._failure_copy(scope)
            self._sync(focus_identity)

    def _apply(
        self,
        result: CollectionBrowseResult,
        *,
        generation: int,
        focus_identity: str | None,
    ) -> bool:
        if not self._current(generation):
            return False
        self.applied_result = result
        self.retained_items = result.items
        self.freshness = "fresh"
        self.loading = False
        self.inflight_scope = None
        self.error_copy = ""
        self.stale_copy = ""
        self._retry_locator_target_id = None
        self._sync(focus_identity)
        return True

    def _failure_copy(self, failed_scope: CollectionBrowseScope) -> str:
        if self.applied_scope is None:
            return _SERVICE_ERROR
        return f"Couldn't load page {failed_scope.page}."

    def request_locator(
        self, target_id: str, *, focus_identity: str | None
    ) -> Any | None:
        """Dispatch one stable-ID owning-page read."""

        if type(target_id) is not str or not target_id or target_id != target_id.strip():
            raise ValueError("target_id must be stable non-blank text.")
        if not self._request_is_active():
            return None
        generation = self.begin(self.mutation_refresh_scope)
        self._retry_locator_target_id = target_id
        self._sync(focus_identity)
        return self._run_worker(
            self._load_locator(
                target_id, generation=generation, focus_identity=focus_identity
            ),
            exclusive=True,
            group=_WORKER_GROUP,
        )

    async def _load_locator(
        self,
        target_id: str,
        *,
        generation: int,
        focus_identity: str | None,
    ) -> None:
        try:
            service = self._collections_service()
            locate = getattr(service, "locate_library_collection_page", None)
            if not callable(locate):
                raise RuntimeError("Collection locator unavailable")
            payload = await self._run_service_call()(
                locate,
                collection_id=target_id,
                limit=COLLECTION_BROWSE_PAGE_SIZE,
                isolate_in_worker=True,
            )
            if payload is None:
                raise LookupError("Collection is absent")
            result = build_collection_locator_result(target_id, payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._current(generation):
                return
            logger.warning(
                "Library Collections browse failed; operation=locate_collection "
                "exception_type={}",
                type(exc).__name__,
            )
            self.loading = False
            self.inflight_scope = None
            if self.freshness != "stale":
                self.error_copy = _LOCATOR_ERROR
            self._retry_locator_target_id = target_id
            self._sync(focus_identity)
            return
        self._apply_locator(
            result, generation=generation, focus_identity=focus_identity
        )

    def _apply_locator(
        self,
        result: CollectionLocatorResult,
        *,
        generation: int,
        focus_identity: str | None,
    ) -> bool:
        if not self._current(generation):
            return False
        self.requested_scope = result.browse_result.scope
        self.located_target_id = result.target_id
        return self._apply(
            result.browse_result,
            generation=generation,
            focus_identity=focus_identity,
        )

    def retain_stale_items(
        self,
        items: tuple[Mapping[str, Any], ...],
        *,
        stale_copy: str,
    ) -> None:
        """Retain known rows without retaining exact source metadata."""

        if self.applied_result is None:
            raise ValueError("Cannot retain stale items before a page applies.")
        if type(items) is not tuple:
            raise TypeError("items must be an exact tuple.")
        if not isinstance(stale_copy, str) or not stale_copy.strip():
            raise ValueError("stale_copy must be non-empty text.")
        self.retained_items = validate_collection_browse_items(items)
        self.freshness = "stale"
        self.error_copy = ""
        self.stale_copy = stale_copy.strip()

    def begin_mutation(self) -> CollectionBrowseScope:
        """Fence reads before a durable write and preserve its applied scope."""

        scope = self.mutation_refresh_scope
        self.invalidate(scope)
        return scope

    def reconcile_committed_mutation(
        self,
        *,
        remove_ids: tuple[str, ...] = (),
        upsert_items: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        """Retain a locally known committed view without forging a total."""

        if type(remove_ids) is not tuple or any(
            type(collection_id) is not str or not collection_id
            for collection_id in remove_ids
        ):
            raise ValueError("remove_ids must be an exact tuple of non-empty ids.")
        if type(upsert_items) is not tuple:
            raise TypeError("upsert_items must be an exact tuple.")
        normalized_upserts = validate_collection_browse_items(upsert_items)
        removed = set(remove_ids)
        upsert_ids = {str(item["collection_id"]) for item in normalized_upserts}
        retained = normalized_upserts + tuple(
            item
            for item in self.retained_items
            if item["collection_id"] not in removed
            and item["collection_id"] not in upsert_ids
        )
        if self.applied_result is None:
            return
        self.retain_stale_items(
            retained[: self.applied_result.limit],
            stale_copy=_MUTATION_COPY,
        )

    def invalidate(self, scope: CollectionBrowseScope | None = None) -> int:
        """Fence all current work without discarding retained rows."""

        self._generation += 1
        if scope is not None:
            if not isinstance(scope, CollectionBrowseScope):
                raise TypeError("scope must be a CollectionBrowseScope.")
            self.requested_scope = scope
        self.inflight_scope = None
        self.loading = False
        self.located_target_id = None
        self._retry_locator_target_id = None
        return self._generation
