"""Non-visual orchestration for exact Library Media pages and type facets."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Mapping

from loguru import logger

from ...Library.library_media_state import (
    MediaBrowseResult,
    MediaBrowseScope,
    build_media_browse_result,
    validate_media_browse_items,
)
from ...Library.library_pager_state import (
    LibraryPagerDisplay,
    PageFreshness,
    build_library_pager_display,
)

_PAGE_WORKER_GROUP = "library-media-browse"
_FACET_WORKER_GROUP = "library-media-types"
_SERVICE_ERROR = "Couldn't load media. Check the local Library and retry."
_FACET_ERROR = "Couldn't load media types. Retry."
_SHRINK_COPY = "List changed while paging; retry to load a current page."
_MUTATION_COPY = "Media changed; retry to load a current page."
_RETRY_FAILED_PREFIX = "Retry failed · "
_TIMEOUT_REASON = "Library took longer than 5 s to answer"


def _retry_failure_reason(exc: BaseException) -> str:
    """Name a failed refresh in the reader's terms, never as a bare class.

    Args:
        exc: The exception the failed page request raised.

    Returns:
        A short human-readable reason for the failure.
    """
    # ``asyncio.TimeoutError`` IS ``TimeoutError`` on 3.11+, and
    # ``TimeoutError`` subclasses ``OSError`` -- so it must be tested first.
    if isinstance(exc, TimeoutError):
        return _TIMEOUT_REASON
    if isinstance(exc, (OSError, sqlite3.OperationalError)):
        # One bounded line: this lands in a ~36-col status Static, so an
        # embedded newline or a long path would push the pager off screen.
        message = " ".join(str(exc).split())[:80]
        return message or type(exc).__name__
    return type(exc).__name__


class LibraryMediaBrowseController:
    """Own requested/applied Media pages and an independently fenced facet list."""

    def __init__(
        self,
        *,
        screen: Any,
        run_service_call: Callable[[], Callable[..., Awaitable[Any]]],
        media_service: Callable[[], Any],
        sync_view: Callable[[], Callable[..., None]],
        request_is_active: Callable[[], bool],
    ) -> None:
        self._screen = screen
        self._run_service_call = run_service_call
        self._media_service = media_service
        self._sync_view = sync_view
        self._request_is_active = request_is_active

        self.requested_scope = MediaBrowseScope()
        self.inflight_scope: MediaBrowseScope | None = None
        self.applied_result: MediaBrowseResult | None = None
        self.retained_items: tuple[Mapping[str, Any], ...] = ()
        self.freshness: PageFreshness = "uninitialized"
        self.loading = False
        self.error_copy = ""
        self.stale_copy = ""
        self._page_generation = 0

        self.type_options: tuple[str, ...] = ()
        self.facet_loading = False
        self.facet_error_copy = ""
        self.facet_fingerprint = ""
        self._facet_generation = 0

    @property
    def _run_worker(self) -> Callable[..., Any]:
        return self._screen.run_worker

    @property
    def applied_scope(self) -> MediaBrowseScope | None:
        return self.applied_result.scope if self.applied_result is not None else None

    @property
    def mutation_refresh_scope(self) -> MediaBrowseScope:
        return self.applied_scope or self.requested_scope

    def scope_for_page(self, page: int) -> MediaBrowseScope:
        return self.mutation_refresh_scope.with_page(page)

    @property
    def pager(self) -> LibraryPagerDisplay:
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
            page_size=applied.limit if applied is not None else 20,
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

    def begin(self, scope: MediaBrowseScope) -> int:
        if not isinstance(scope, MediaBrowseScope):
            raise TypeError("scope must be a MediaBrowseScope.")
        self._page_generation += 1
        self.requested_scope = scope
        self.inflight_scope = scope
        self.loading = True
        self.error_copy = ""
        return self._page_generation

    def request(
        self, scope: MediaBrowseScope, *, focus_identity: str | None
    ) -> Any | None:
        generation = self.begin(scope)
        if not self._request_is_active():
            return None
        self._sync(focus_identity)
        return self._run_worker(
            self._load(scope, generation=generation, focus_identity=focus_identity),
            exclusive=True,
            group=_PAGE_WORKER_GROUP,
        )

    def retry(self, *, focus_identity: str | None) -> Any | None:
        return self.request(self.requested_scope, focus_identity=focus_identity)

    async def _search(self, scope: MediaBrowseScope) -> MediaBrowseResult:
        service = self._media_service()
        search_media = getattr(service, "search_media", None)
        if not callable(search_media):
            raise RuntimeError("Media service unavailable")
        filters: dict[str, Any] = {"sort_by": scope.sort_by}
        if scope.media_type is not None:
            filters["media_types"] = [scope.media_type]
        payload = await self._run_service_call()(
            search_media,
            mode="local",
            query=scope.query,
            limit=scope.page_size,
            offset=scope.offset,
            library_summary=True,
            isolate_in_worker=True,
            **filters,
        )
        return build_media_browse_result(scope, payload)

    def _current(self, generation: int) -> bool:
        return generation == self._page_generation and self._request_is_active()

    async def _load(
        self,
        scope: MediaBrowseScope,
        *,
        generation: int,
        focus_identity: str | None,
    ) -> None:
        fetched_scope = scope
        clamped = False
        try:
            while True:
                result = await self._search(fetched_scope)
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
                "Library Media browse failed; operation=search_media exception_type={}",
                type(exc).__name__,
            )
            self.loading = False
            self.inflight_scope = None
            if self.freshness != "stale":
                self.error_copy = self._failure_copy(scope)
            else:
                # task-31220: on a stale page the stale copy is the ONLY
                # thing shown, and leaving it untouched is what made Retry
                # read as inert across repeated presses (critique #5).
                # ``_MUTATION_COPY``/``_SHRINK_COPY`` still describe why the
                # page went stale; this describes why recovering from it
                # just failed. ``_apply`` clears it on the next success.
                self.stale_copy = _RETRY_FAILED_PREFIX + _retry_failure_reason(exc)
            self._sync(focus_identity)

    def _apply(
        self,
        result: MediaBrowseResult,
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
        self._sync(focus_identity)
        return True

    def _failure_copy(self, failed_scope: MediaBrowseScope) -> str:
        applied = self.applied_scope
        if applied is None:
            return _SERVICE_ERROR
        if failed_scope.same_except_page(applied):
            return f"Couldn't load page {failed_scope.page}."
        return "Filter wasn't applied; showing previous results."

    def retain_stale_items(
        self,
        items: tuple[Mapping[str, Any], ...],
        *,
        stale_copy: str,
    ) -> None:
        if self.applied_result is None:
            raise ValueError("Cannot retain stale items before a page applies.")
        if type(items) is not tuple:
            raise TypeError("items must be an exact tuple.")
        if not isinstance(stale_copy, str) or not stale_copy.strip():
            raise ValueError("stale_copy must be non-empty text.")
        self.retained_items = validate_media_browse_items(items)
        self.freshness = "stale"
        self.error_copy = ""
        self.stale_copy = stale_copy.strip()

    def begin_mutation(self) -> MediaBrowseScope:
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
        """Retain one locally known committed view without forging metadata."""
        if type(remove_ids) is not tuple or any(
            type(media_id) is not str or not media_id for media_id in remove_ids
        ):
            raise ValueError("remove_ids must be an exact tuple of non-empty ids.")
        if type(upsert_items) is not tuple:
            raise TypeError("upsert_items must be an exact tuple.")
        normalized_upserts = validate_media_browse_items(upsert_items)
        applied_scope = self.applied_scope
        if applied_scope is not None and applied_scope.query:
            normalized_upserts = ()
        elif applied_scope is not None and applied_scope.media_type is not None:
            normalized_upserts = tuple(
                item
                for item in normalized_upserts
                if item["media_type"] == applied_scope.media_type
            )
        removed = set(remove_ids)
        upsert_ids = {str(item["id"]) for item in normalized_upserts}
        retained = normalized_upserts + tuple(
            item
            for item in self.retained_items
            if item["id"] not in removed and item["id"] not in upsert_ids
        )
        if self.applied_result is None:
            return
        self.retain_stale_items(
            retained[: self.applied_result.limit],
            stale_copy=_MUTATION_COPY,
        )

    def invalidate(self, scope: MediaBrowseScope | None = None) -> int:
        self._page_generation += 1
        if scope is not None:
            self.requested_scope = scope
        self.inflight_scope = None
        self.loading = False
        self.invalidate_facets()
        return self._page_generation

    def request_facets(self, *, fingerprint: str) -> Any | None:
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("facet fingerprint must be non-empty text.")
        self._facet_generation += 1
        generation = self._facet_generation
        self.facet_fingerprint = fingerprint
        self.facet_loading = True
        self.facet_error_copy = ""
        if not self._request_is_active():
            return None
        self._sync(None)
        return self._run_worker(
            self._load_facets(generation=generation, fingerprint=fingerprint),
            exclusive=True,
            group=_FACET_WORKER_GROUP,
        )

    async def _load_facets(self, *, generation: int, fingerprint: str) -> None:
        try:
            service = self._media_service()
            list_types = getattr(service, "list_library_media_types", None)
            if not callable(list_types):
                raise RuntimeError("Media type service unavailable")
            values = await self._run_service_call()(
                list_types, mode="local", isolate_in_worker=True
            )
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise TypeError("Media types must be a sequence.")
            if any(type(value) is not str or not value.strip() for value in values):
                raise ValueError("Media types must be non-empty strings.")
            normalized = tuple(sorted(set(values)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if generation != self._facet_generation or not self._request_is_active():
                return
            logger.warning(
                "Library Media facets failed; operation=list_media_types exception_type={}",
                type(exc).__name__,
            )
            self.facet_loading = False
            self.facet_error_copy = _FACET_ERROR
            self._sync(None)
            return
        if (
            generation != self._facet_generation
            or fingerprint != self.facet_fingerprint
            or not self._request_is_active()
        ):
            return
        self.type_options = normalized
        self.facet_loading = False
        self.facet_error_copy = ""
        self._sync(None)

    def invalidate_facets(self, *, fingerprint: str = "") -> int:
        self._facet_generation += 1
        self.facet_fingerprint = fingerprint
        self.facet_loading = False
        return self._facet_generation
