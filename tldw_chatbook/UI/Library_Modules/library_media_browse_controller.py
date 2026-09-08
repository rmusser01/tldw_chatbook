"""Non-visual orchestration for exact Library Media pages and type facets."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from loguru import logger

from ...Library.library_media_state import (
    LIBRARY_MEDIA_SERVICE_ERROR as _SERVICE_ERROR,
    MediaBrowseResult,
    MediaBrowseScope,
    _redact_paths as _redact_paths,
    _retry_failure_reason,
    build_media_browse_result,
    build_media_retry_failure_copy,
)
from .library_media_browse_state import (
    MediaBrowseState,
    _SERVICE_WHAT,
    _load_failure,
    _raised_failure,
)

_PAGE_WORKER_GROUP = "library-media-browse"
_FACET_WORKER_GROUP = "library-media-types"
_FACET_ERROR = "Couldn't load media types. Retry."
_FACET_WHAT = "Couldn't load media types"
_SHRINK_COPY = "List changed while paging; retry to load a current page."


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
        self.state = MediaBrowseState()
        self._page_generation = 0
        self._facet_generation = 0

    @property
    def _run_worker(self) -> Callable[..., Any]:
        return self._screen.run_worker

    def begin(self, scope: MediaBrowseScope) -> int:
        if not isinstance(scope, MediaBrowseScope):
            raise TypeError("scope must be a MediaBrowseScope.")
        self._page_generation += 1
        self.state.requested_scope = scope
        self.state.inflight_scope = scope
        self.state.loading = True
        self.state.error_copy = ""
        self.state.page_failure = None
        return self._page_generation

    def request(
        self, scope: MediaBrowseScope, *, focus_identity: str | None
    ) -> Any | None:
        generation = self.begin(scope)
        if not self._request_is_active():
            return None
        self._sync_view()(focus_identity)
        return self._run_worker(
            self._load(scope, generation=generation, focus_identity=focus_identity),
            exclusive=True,
            group=_PAGE_WORKER_GROUP,
        )

    def retry(self, *, focus_identity: str | None) -> Any | None:
        # task-31632: the callout advertises ONE Retry for whichever load
        # failed, so a facet failure it names has to be one this button can
        # clear -- the type list has no retry control of its own.
        # Qodo PR G finding 4: decide per fence from `page_failure`/
        # `facet_failure`, not from the copy strings -- a facet-only
        # failure must retry the facet fence ALONE. An unconditional page
        # reload here would replace a live facet failure with a page one
        # if that unnecessary page request itself failed. `page_failed`
        # also covers the "neither is live" edge case, defaulting to the
        # page fence like this method always has.
        facet_failed = self.state.facet_failure is not None
        page_failed = self.state.page_failure is not None or not facet_failed
        if facet_failed:
            self.request_facets(fingerprint=self.state.requested_scope.fingerprint)
        if page_failed:
            return self.request(
                self.state.requested_scope, focus_identity=focus_identity
            )
        return None

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
            # Qodo on #2475: this is the ONE caller that renders task-28008's
            # keyword-only reasons ("· matched keyword X" on a row whose title
            # and body hold nothing the user typed), so it is the one caller
            # that pays the probe's extra SELECT. Every other library-summary
            # pager -- "Review these", the selection ordering pass -- discards
            # them and now asks for nothing.
            match_reasons=True,
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
                    self.state.loading = False
                    self.state.inflight_scope = None
                    if self.state.applied_result is not None:
                        self.state.freshness = "stale"
                        self.state.error_copy = ""
                        self.state.page_failure = None
                        self.state.stale_copy = _SHRINK_COPY
                        self.state.stale_reason = _SHRINK_COPY
                    else:
                        self.state.error_copy = _SERVICE_ERROR
                        # No exception here: the clamped page came back
                        # out of range too, so the reason is the shrink
                        # itself rather than anything raised.
                        self.state.page_failure = _load_failure(
                            _SERVICE_WHAT,
                            "the list changed while loading",
                            timed_out=False,
                        )
                    self._sync_view()(focus_identity)
                    return
                clamped = True
                fetched_scope = scope.with_page(result.last_page)
                self.state.inflight_scope = fetched_scope
                self._sync_view()(focus_identity)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._current(generation):
                return
            logger.warning(
                "Library Media browse failed; operation=search_media exception_type={}",
                type(exc).__name__,
            )
            self.state.loading = False
            self.state.inflight_scope = None
            if self.state.freshness != "stale":
                self.state.error_copy, failure_what = self.state._failure_copy(scope)
                reason = _retry_failure_reason(exc)
                fingerprint = scope.fingerprint
                repeated = (
                    reason == self.state._page_fault_reason
                    and fingerprint == self.state._page_fault_context
                )
                self.state._page_fault_reason = reason
                self.state._page_fault_context = fingerprint
                self.state.page_failure = _raised_failure(
                    failure_what, exc, repeated=repeated
                )
            else:
                # task-31220: on a stale page the stale copy is the ONLY
                # thing shown, and leaving it untouched is what made Retry
                # read as inert across repeated presses (critique #5).
                # ``_MUTATION_COPY``/``_SHRINK_COPY`` still describe why the
                # page went stale; this describes why recovering from it
                # just failed. ``_apply`` clears it on the next success.
                self.state.stale_copy = build_media_retry_failure_copy(exc)
            self._sync_view()(focus_identity)

    def _apply(
        self,
        result: MediaBrowseResult,
        *,
        generation: int,
        focus_identity: str | None,
    ) -> bool:
        if not self._current(generation):
            return False
        self.state.applied_result = result
        self.state.retained_items = result.items
        self.state.freshness = "fresh"
        self.state.loading = False
        self.state.inflight_scope = None
        self.state.error_copy = ""
        self.state.page_failure = None
        self.state._page_fault_reason = ""
        self.state._page_fault_context = ""
        self.state.stale_copy = ""
        self.state.stale_reason = ""
        self._sync_view()(focus_identity)
        return True

    def begin_mutation(self) -> MediaBrowseScope:
        """Fence reads before a durable write and preserve its applied scope."""
        scope = self.state.mutation_refresh_scope
        self.invalidate(scope)
        return scope

    def invalidate(self, scope: MediaBrowseScope | None = None) -> int:
        self._page_generation += 1
        if scope is not None:
            self.state.requested_scope = scope
        self.state.inflight_scope = None
        self.state.loading = False
        self.invalidate_facets()
        return self._page_generation

    def request_facets(self, *, fingerprint: str) -> Any | None:
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("facet fingerprint must be non-empty text.")
        self._facet_generation += 1
        generation = self._facet_generation
        self.state.facet_fingerprint = fingerprint
        self.state.facet_loading = True
        self.state.facet_error_copy = ""
        self.state.facet_failure = None
        if not self._request_is_active():
            return None
        self._sync_view()(None)
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
            self.state.facet_loading = False
            self.state.facet_error_copy = _FACET_ERROR
            reason = _retry_failure_reason(exc)
            repeated = (
                reason == self.state._facet_fault_reason
                and fingerprint == self.state._facet_fault_context
            )
            self.state._facet_fault_reason = reason
            self.state._facet_fault_context = fingerprint
            self.state.facet_failure = _raised_failure(
                _FACET_WHAT, exc, repeated=repeated
            )
            self._sync_view()(None)
            return
        if (
            generation != self._facet_generation
            or fingerprint != self.state.facet_fingerprint
            or not self._request_is_active()
        ):
            return
        self.state.type_options = normalized
        self.state.facet_loading = False
        self.state.facet_error_copy = ""
        self.state.facet_failure = None
        self.state._facet_fault_reason = ""
        self.state._facet_fault_context = ""
        self._sync_view()(None)

    def invalidate_facets(self, *, fingerprint: str = "") -> int:
        self._facet_generation += 1
        self.state.facet_fingerprint = fingerprint
        self.state.facet_loading = False
        return self._facet_generation
