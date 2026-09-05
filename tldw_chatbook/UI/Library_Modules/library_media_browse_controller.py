"""Non-visual orchestration for exact Library Media pages and type facets."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Mapping

from loguru import logger

from ...Library.library_media_state import (
    LIBRARY_MEDIA_SERVICE_ERROR as _SERVICE_ERROR,
    MediaBrowseResult,
    MediaBrowseScope,
    _redact_paths as _redact_paths,
    _retry_failure_reason,
    _REOPEN_RECOVERY,
    build_media_browse_result,
    build_media_load_failure_copy,
    build_media_retry_failure_copy,
    validate_media_browse_items,
)
from ...Library.library_pager_state import (
    LibraryPagerDisplay,
    PageFreshness,
    build_library_pager_display,
)
from ..destination_recovery import (
    DestinationRecoveryState,
    load_failure_recovery_state,
)

_PAGE_WORKER_GROUP = "library-media-browse"
_FACET_WORKER_GROUP = "library-media-types"
_SERVICE_WHAT = "Couldn't load media"
_FACET_ERROR = "Couldn't load media types. Retry."
_FACET_WHAT = "Couldn't load media types"
# task-31632: the single Media Retry, rendered INSIDE the failure callout,
# and the callout's own selector -- both failure fences publish one state
# through ``failure`` because the canvas paints one callout.
_RETRY_ID = "library-media-retry"
_FAILURE_SELECTOR = "#library-media-load-failure"
_SHRINK_COPY = "List changed while paging; retry to load a current page."
_MUTATION_COPY = "Media changed; retry to load a current page."


def _load_failure(
    what: str, reason: str, *, timed_out: bool
) -> DestinationRecoveryState:
    """Build the Media callout state for one failed load."""
    return load_failure_recovery_state(
        what=what,
        reason=reason,
        retry_id=_RETRY_ID,
        stable_selector=_FAILURE_SELECTOR,
        kind="timeout" if timed_out else "error",
    )


def _raised_failure(
    what: str, exc: BaseException, *, repeated: bool = False
) -> DestinationRecoveryState:
    """Name a raised load failure through the shared reason mapping.

    Args:
        what: What could not be loaded, as a clause.
        exc: The exception the failed request raised.
        repeated: True when this same reason has just recurred on a
            consecutive Retry (task-31982 AC#2). The failure then names the
            reopen recovery step instead of repeating its one sentence.
    """
    reason = _retry_failure_reason(exc)
    if repeated:
        reason = f"{reason} · {_REOPEN_RECOVERY}"
    return _load_failure(
        what,
        reason,
        timed_out=isinstance(exc, TimeoutError),
    )


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
        # Final review M-3: the reason the PAGE went stale, kept separate
        # from ``stale_copy`` (the pager's own status line, which a failed
        # Retry overwrites with "Couldn't retry · <reason>"). Every gated
        # action's tooltip reads this one instead, so it keeps explaining
        # why the action is off across repeated failed retries.
        self.stale_reason = ""
        # task-31632: the recovery state behind ``error_copy``/
        # ``facet_error_copy`` -- same event, with the reason and a Retry
        # target. Each is cleared by its OWN success, so a page failure
        # never outlives a facet reload (or the reverse); ``failure`` is
        # the one the canvas paints.
        self.page_failure: DestinationRecoveryState | None = None
        self.facet_failure: DestinationRecoveryState | None = None
        # task-31982 AC#2: the reason each fence last failed with, kept
        # across Retries (``begin`` clears ``page_failure`` on every request,
        # so it cannot tell a repeat from a first failure). A consecutive
        # failure with the SAME reason names the reopen recovery step; a
        # success on either fence clears its own tracker.
        # task-32039 AC#1: the reason alone was not enough -- a resume
        # auto-refresh or a page/query/type change that hit the same reason
        # read as a consecutive Retry. Each reason is now paired with the
        # context fingerprint it failed in (the page scope, the facet
        # request), so "repeated" means the SAME context failed again; a new
        # visit clears the episode via ``clear_fault_episode``.
        self._page_fault_reason = ""
        self._page_fault_context = ""
        self._facet_fault_reason = ""
        self._facet_fault_context = ""
        self._page_generation = 0

        self.type_options: tuple[str, ...] = ()
        self.facet_loading = False
        self.facet_error_copy = ""
        self.facet_fingerprint = ""
        self._facet_generation = 0

    @property
    def failure(self) -> DestinationRecoveryState | None:
        """Return the load failure to show: the page's, else the facets'.

        Returns:
            ``page_failure`` when a page load has failed; otherwise
            ``facet_failure`` when the type facets have failed; otherwise
            ``None`` when both fences are clean.
        """
        return self.page_failure or self.facet_failure

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

    def begin(self, scope: MediaBrowseScope) -> int:
        if not isinstance(scope, MediaBrowseScope):
            raise TypeError("scope must be a MediaBrowseScope.")
        self._page_generation += 1
        self.requested_scope = scope
        self.inflight_scope = scope
        self.loading = True
        self.error_copy = ""
        self.page_failure = None
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
        facet_failed = self.facet_failure is not None
        page_failed = self.page_failure is not None or not facet_failed
        if facet_failed:
            self.request_facets(fingerprint=self.requested_scope.fingerprint)
        if page_failed:
            return self.request(self.requested_scope, focus_identity=focus_identity)
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
                    self.loading = False
                    self.inflight_scope = None
                    if self.applied_result is not None:
                        self.freshness = "stale"
                        self.error_copy = ""
                        self.page_failure = None
                        self.stale_copy = _SHRINK_COPY
                        self.stale_reason = _SHRINK_COPY
                    else:
                        self.error_copy = _SERVICE_ERROR
                        # No exception here: the clamped page came back
                        # out of range too, so the reason is the shrink
                        # itself rather than anything raised.
                        self.page_failure = _load_failure(
                            _SERVICE_WHAT,
                            "the list changed while loading",
                            timed_out=False,
                        )
                    self._sync_view()(focus_identity)
                    return
                clamped = True
                fetched_scope = scope.with_page(result.last_page)
                self.inflight_scope = fetched_scope
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
            self.loading = False
            self.inflight_scope = None
            if self.freshness != "stale":
                self.error_copy, failure_what = self._failure_copy(scope)
                reason = _retry_failure_reason(exc)
                fingerprint = scope.fingerprint
                repeated = (
                    reason == self._page_fault_reason
                    and fingerprint == self._page_fault_context
                )
                self._page_fault_reason = reason
                self._page_fault_context = fingerprint
                self.page_failure = _raised_failure(
                    failure_what, exc, repeated=repeated
                )
            else:
                # task-31220: on a stale page the stale copy is the ONLY
                # thing shown, and leaving it untouched is what made Retry
                # read as inert across repeated presses (critique #5).
                # ``_MUTATION_COPY``/``_SHRINK_COPY`` still describe why the
                # page went stale; this describes why recovering from it
                # just failed. ``_apply`` clears it on the next success.
                self.stale_copy = build_media_retry_failure_copy(exc)
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
        self.applied_result = result
        self.retained_items = result.items
        self.freshness = "fresh"
        self.loading = False
        self.inflight_scope = None
        self.error_copy = ""
        self.page_failure = None
        self._page_fault_reason = ""
        self._page_fault_context = ""
        self.stale_copy = ""
        self.stale_reason = ""
        self._sync_view()(focus_identity)
        return True

    def _failure_copy(self, failed_scope: MediaBrowseScope) -> tuple[str, str]:
        """Return the pager sentence and the callout's "what failed" clause.

        Args:
            failed_scope: Scope of the request that failed.

        Returns:
            The existing plain sentence and the same failure as a clause, so
            the two can never describe different failures.
        """
        applied = self.applied_scope
        copy = build_media_load_failure_copy(applied, failed_scope)
        if applied is None:
            return copy, _SERVICE_WHAT
        if failed_scope.same_except_page(applied):
            return (
                copy,
                f"Couldn't load page {failed_scope.page}",
            )
        return (
            copy,
            "Filter wasn't applied",
        )

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
        self.page_failure = None
        self.stale_copy = stale_copy.strip()
        self.stale_reason = self.stale_copy

    def note_analysis_state(self, media_id: str, *, has_analysis: bool) -> bool:
        """Re-project one retained row's ``has_analysis`` after an analysis write.

        Qodo on #2475: ``has_analysis`` is a SQL projection frozen into the
        retained row when the page applied, so an analysis saved from the
        Reader (or by the bulk Analyze run) left its own row unmarked until
        something re-paged the list. The caller supplies the value from that
        same projection, re-read for this ONE id after the write
        (``LibraryScreen._reproject_library_media_analysis_row``); nothing
        here derives it, and nothing runs on the page path.

        Freshness is deliberately untouched: this is not a page change, it
        is the same page carrying a fact the projection has already been
        asked about.

        Args:
            media_id: Canonical ``local:media:<id>`` row id.
            has_analysis: Whether that item now carries analysis text.

        Returns:
            True when a retained row actually changed (so the caller can
            skip a repaint it does not need).
        """
        target = str(media_id)
        if not any(
            str(item["id"]) == target and bool(item["has_analysis"]) != has_analysis
            for item in self.retained_items
        ):
            return False
        self.retained_items = validate_media_browse_items(
            tuple(
                {**item, "has_analysis": has_analysis}
                if str(item["id"]) == target
                else item
                for item in self.retained_items
            )
        )
        return True

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
        self.facet_failure = None
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
            self.facet_loading = False
            self.facet_error_copy = _FACET_ERROR
            reason = _retry_failure_reason(exc)
            repeated = (
                reason == self._facet_fault_reason
                and fingerprint == self._facet_fault_context
            )
            self._facet_fault_reason = reason
            self._facet_fault_context = fingerprint
            self.facet_failure = _raised_failure(_FACET_WHAT, exc, repeated=repeated)
            self._sync_view()(None)
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
        self.facet_failure = None
        self._facet_fault_reason = ""
        self._facet_fault_context = ""
        self._sync_view()(None)

    def clear_fault_episode(self) -> None:
        """Forget the repeated-fault history so a new visit is not a Retry.

        task-32039 AC#1: ``begin``/``request_facets`` cannot tell a Library
        screen-RESUME auto-refresh from a consecutive Retry -- both re-issue
        the same scope with the same fingerprint. The screen clears the
        episode on resume, so the first failure of a new visit never wears the
        reopen recovery step even when its reason matches the last visit's. A
        genuine same-context Retry within a visit still escalates.
        """
        self._page_fault_reason = ""
        self._page_fault_context = ""
        self._facet_fault_reason = ""
        self._facet_fault_context = ""

    def invalidate_facets(self, *, fingerprint: str = "") -> int:
        self._facet_generation += 1
        self.facet_fingerprint = fingerprint
        self.facet_loading = False
        return self._facet_generation
