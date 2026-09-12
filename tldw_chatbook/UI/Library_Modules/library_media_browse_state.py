"""UI-local Media browse data and pure presentation projections (ADR-128)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...Library.library_media_state import (
    MediaBrowseResult,
    MediaBrowseScope,
    _REOPEN_RECOVERY,
    _retry_failure_reason,
    build_media_load_failure_copy,
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

_SERVICE_WHAT = "Couldn't load media"
# TASK-31632: both failure fences share the canvas's one Retry callout.
_RETRY_ID = "library-media-retry"
_FAILURE_SELECTOR = "#library-media-load-failure"
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
        repeated: Same-reason consecutive Retry; add the reopen recovery step.
    """
    reason = _retry_failure_reason(exc)
    if repeated:
        reason = f"{reason} · {_REOPEN_RECOVERY}"
    return _load_failure(
        what,
        reason,
        timed_out=isinstance(exc, TimeoutError),
    )


@dataclass
class MediaBrowseState:
    """Retain source-owned page/facet presentation without runtime collaborators."""

    requested_scope: MediaBrowseScope = field(default_factory=MediaBrowseScope)
    inflight_scope: MediaBrowseScope | None = None
    applied_result: MediaBrowseResult | None = None
    retained_items: tuple[Mapping[str, Any], ...] = ()
    freshness: PageFreshness = "uninitialized"
    loading: bool = False
    error_copy: str = ""
    stale_copy: str = ""
    # M-3: gated-action tooltips retain the stale reason when Retry changes
    # the pager's stale_copy to "Couldn't retry · <reason>".
    stale_reason: str = ""
    # TASK-31632: each fence clears only its own recovery state on success;
    # failure selects the page/facet callout shown by the canvas.
    page_failure: DestinationRecoveryState | None = None
    facet_failure: DestinationRecoveryState | None = None
    # TASK-31982/32039: retain each reason and request fingerprint across
    # begin(), which clears page_failure. Only a same-context Retry escalates;
    # that fence's success or clear_fault_episode() resets its history.
    _page_fault_reason: str = ""
    _page_fault_context: str = ""
    _facet_fault_reason: str = ""
    _facet_fault_context: str = ""

    type_options: tuple[str, ...] = ()
    facet_loading: bool = False
    facet_error_copy: str = ""
    facet_fingerprint: str = ""

    @property
    def failure(self) -> DestinationRecoveryState | None:
        """Return the load failure to show: the page's, else the facets'.

        Returns:
            The page failure, else facet failure, else None when both are clean.
        """
        return self.page_failure or self.facet_failure

    @property
    def applied_scope(self) -> MediaBrowseScope | None:
        """Read the scope whose result is retained.

        Returns:
            The applied result's scope, or None before a result applies.
        """
        return self.applied_result.scope if self.applied_result is not None else None

    @property
    def mutation_refresh_scope(self) -> MediaBrowseScope:
        """Return the applied scope, falling back to the requested scope.

        Refresh visible items even when a different requested page has not applied.

        Returns:
            The retained result's scope, or the requested scope without a result.
        """
        return self.applied_scope or self.requested_scope

    def scope_for_page(self, page: int) -> MediaBrowseScope:
        """Preserve the mutation-refresh filters while choosing another page.

        Args:
            page: Requested page number, validated by MediaBrowseScope.

        Returns:
            The mutation-refresh scope with its page replaced.

        Raises:
            ValueError: The page is invalid or its offset exceeds SQLite's range.
        """
        return self.mutation_refresh_scope.with_page(page)

    @property
    def pager(self) -> LibraryPagerDisplay:
        """Build the pager from retained results and current request state.

        Returns:
            Display state using the in-flight page while loading, the requested
            page after an error or before any result, and the applied page
            otherwise. Retained rows supply the count; totals are trusted only
            for fresh results, with a page-size fallback before any result.
        """
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

        Qodo #2475: the Reader/bulk writer re-reads this one SQL projection
        through _reproject_library_media_analysis_row; otherwise its retained
        row stays unmarked until re-paging. Do not derive the value here or
        change freshness: this updates a known fact on the same page.

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

    def clear_fault_episode(self) -> None:
        """Forget the repeated-fault history so a new visit is not a Retry.

        TASK-32039: resume can reissue the same fingerprint as a Retry. Clear
        on resume so only same-context retries within one visit escalate.
        """
        self._page_fault_reason = ""
        self._page_fault_context = ""
        self._facet_fault_reason = ""
        self._facet_fault_context = ""
