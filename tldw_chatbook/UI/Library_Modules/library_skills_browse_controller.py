"""Non-visual orchestration for the Library's exact local Skills page."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, Mapping

from loguru import logger

from ...Library.library_pager_state import (
    LibraryPagerDisplay,
    PageFreshness,
    build_library_pager_display,
)
from ...Library.library_skills_state import (
    SkillBrowseResult,
    SkillBrowseScope,
    apply_skill_browse_result,
    begin_skill_browse,
    build_skill_browse_error,
    build_skill_browse_result,
    validate_skill_browse_items,
)

_WORKER_GROUP = "library-skills-browse"
_SERVICE_UNAVAILABLE = (
    "Couldn't load Skills. The local Skills service is unavailable; retry."
)
_SERVICE_ERROR = "Couldn't load Skills. Check the local Library and retry."
_MAX_PAGE_FETCH_ATTEMPTS = 3


class LibrarySkillsBrowseController:
    """Own requested/applied Skills scopes, generations, rows, and recovery.

    Args:
        screen: Owning screen whose worker runner executes source requests.
        run_service_call: Factory for the async service-call boundary.
        skills_service: Factory for the current local Skills service.
        sync_view: Factory for the accepted-state projection callback.
        request_is_active: Predicate fencing requests to the active route.
    """

    def __init__(
        self,
        *,
        screen: Any,
        run_service_call: Callable[[], Callable[..., Awaitable[Any]]],
        skills_service: Callable[[], Any],
        sync_view: Callable[[], Callable[[SkillBrowseResult, str | None], None]],
        request_is_active: Callable[[], bool],
    ) -> None:
        self._screen = screen
        self._run_service_call = run_service_call
        self._skills_service = skills_service
        self._sync_view = sync_view
        self._request_is_active = request_is_active
        self._request_counter = 1
        self.scope = SkillBrowseScope()
        self.result = begin_skill_browse(self.scope, request_token=1)
        self.applied_result: SkillBrowseResult | None = None
        self.retained_items: tuple[Mapping[str, Any], ...] = ()
        self.freshness: PageFreshness = "uninitialized"
        self.error_copy = ""
        self.stale_copy = ""

    @property
    def visible_result(self) -> SkillBrowseResult:
        """Return the page that should drive visible rows.

        Returns:
            The last applied page, falling back to current request state.
        """
        return self.applied_result or self.result

    @property
    def blocked_total(self) -> int:
        """Return the source-owned blocked Skill total.

        Returns:
            The applied source total, or zero before a page has applied.
        """
        return self.applied_result.blocked_total if self.applied_result else 0

    @property
    def first_blocked_skill_name(self) -> str | None:
        """Return the first blocked Skill identity.

        Returns:
            The source-reported identity, or ``None`` before a page applies.
        """
        return (
            self.applied_result.first_blocked_skill_name
            if self.applied_result
            else None
        )

    @property
    def pager(self) -> LibraryPagerDisplay:
        """Build display-only pager state.

        Returns:
            Pager copy and control state derived from requested/applied coordinates.
        """
        applied = self.applied_result
        requested_page = (
            self.scope.page
            if self.result.status in {"loading", "error"} or applied is None
            else applied.page
        )
        return build_library_pager_display(
            applied_page=applied.page if applied is not None else None,
            requested_page=requested_page,
            page_size=(
                applied.scope.page_size if applied is not None else self.scope.page_size
            ),
            row_count=len(self.retained_items),
            total=(
                applied.total_items
                if applied is not None and self.freshness == "fresh"
                else None
            ),
            freshness=self.freshness,
            loading=self.result.status == "loading",
            error_copy=self.error_copy,
            stale_copy=self.stale_copy,
        )

    @property
    def mutation_refresh_scope(self) -> SkillBrowseScope:
        """Return the stable scope to refresh after a committed mutation.

        Returns:
            The applied scope, or the current requested scope before first apply.
        """
        return self.applied_result.scope if self.applied_result else self.scope

    def scope_for_page(self, page: int) -> SkillBrowseScope:
        """Return the stable mutation scope at a different page.

        Args:
            page: One-based page number to request.

        Returns:
            A new scope preserving the applied query, sort, and page size.

        Raises:
            ValueError: If ``page`` is outside the supported one-based range.
        """
        return replace(self.mutation_refresh_scope, page=page)

    def retain_stale_items(
        self,
        items: tuple[Mapping[str, Any], ...],
        *,
        stale_copy: str,
    ) -> None:
        """Keep an applied page visible but inert during a source refresh.

        Args:
            items: Exact detached rows to retain.
            stale_copy: Non-empty explanation shown beside the stale rows.

        Raises:
            TypeError: If ``items`` is not an exact tuple or contains bad rows.
            ValueError: If copy is blank or no page has applied yet.
        """
        if type(items) is not tuple:
            raise TypeError("items must be an exact tuple.")
        if not isinstance(stale_copy, str) or not stale_copy.strip():
            raise ValueError("stale_copy must be non-empty text.")
        if self.applied_result is None:
            raise ValueError("Cannot retain stale items before a page applies.")
        self.retained_items = validate_skill_browse_items(items)
        self.freshness = "stale"
        self.error_copy = ""
        self.stale_copy = stale_copy.strip()

    @property
    def _run_worker(self) -> Callable[..., Any]:
        return self._screen.run_worker

    def _next_request_token(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def begin(self, scope: SkillBrowseScope) -> int:
        """Start a new loading generation for an exact scope.

        Args:
            scope: Exact Skills page coordinates to request.

        Returns:
            The monotonic request token assigned to this generation.
        """
        token = self._next_request_token()
        self.scope = scope
        self.result = begin_skill_browse(scope, request_token=token)
        self.error_copy = ""
        return token

    def dispatch(
        self,
        scope: SkillBrowseScope,
        *,
        request_token: int,
        focus_identity: str | None,
    ) -> Any | None:
        """Dispatch a still-current loading generation to the screen worker.

        Args:
            scope: Exact Skills page coordinates to load.
            request_token: Generation returned by :meth:`begin`.
            focus_identity: Widget identity to restore after settlement.

        Returns:
            The screen worker handle, or ``None`` when the request is stale.
        """
        current = self.result
        if (
            not self._request_is_active()
            or current.status != "loading"
            or current.request_token != request_token
            or current.request_fingerprint != scope.fingerprint
        ):
            return None
        self._sync_view()(current, focus_identity)
        return self._run_worker(
            self._load(
                scope,
                request_token=request_token,
                focus_identity=focus_identity,
            ),
            exclusive=True,
            group=_WORKER_GROUP,
        )

    def request(
        self,
        scope: SkillBrowseScope,
        *,
        focus_identity: str | None,
    ) -> Any | None:
        """Begin and dispatch one exact Skills page request.

        Args:
            scope: Exact Skills page coordinates to load.
            focus_identity: Widget identity to restore after settlement.

        Returns:
            The screen worker handle, or ``None`` when browsing is inactive.
        """
        token = self.begin(scope)
        return self.dispatch(
            scope,
            request_token=token,
            focus_identity=focus_identity,
        )

    def retry(self, *, focus_identity: str | None) -> Any | None:
        """Retry the last requested scope.

        Args:
            focus_identity: Widget identity to restore after settlement.

        Returns:
            The screen worker handle, or ``None`` when browsing is inactive.
        """
        return self.request(self.scope, focus_identity=focus_identity)

    async def _service_page(
        self,
        list_skills: Callable[..., Any],
        scope: SkillBrowseScope,
        *,
        offset: int,
    ) -> Any:
        return await self._run_service_call()(
            list_skills,
            mode="local",
            query=scope.query,
            sort=scope.sort,
            limit=scope.page_size,
            offset=offset,
            isolate_in_worker=True,
        )

    async def _load(
        self,
        scope: SkillBrowseScope,
        *,
        request_token: int,
        focus_identity: str | None,
    ) -> None:
        service = self._skills_service()
        list_skills = getattr(service, "list_skills", None)
        if not callable(list_skills):
            result = build_skill_browse_error(
                scope,
                request_token=request_token,
                error=_SERVICE_UNAVAILABLE,
            )
        else:
            try:
                offset = (scope.page - 1) * scope.page_size
                for _attempt in range(_MAX_PAGE_FETCH_ATTEMPTS):
                    record = await self._service_page(
                        list_skills,
                        scope,
                        offset=offset,
                    )
                    if not isinstance(record, Mapping):
                        break
                    total = record.get("total")
                    if type(total) is not int or total < 0:
                        break
                    total_pages = max(
                        1,
                        (total + scope.page_size - 1) // scope.page_size,
                    )
                    resolved_offset = (
                        min(scope.page, total_pages) - 1
                    ) * scope.page_size
                    if offset == resolved_offset:
                        break
                    offset = resolved_offset
                result = build_skill_browse_result(
                    scope,
                    record,
                    request_token=request_token,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Library Skills browse failed; operation=list_skills "
                    "exception_type={}",
                    type(exc).__name__,
                )
                result = build_skill_browse_error(
                    scope,
                    request_token=request_token,
                    error=_SERVICE_ERROR,
                )
        self.apply(result, focus_identity=focus_identity)

    def apply(
        self,
        result: SkillBrowseResult,
        *,
        focus_identity: str | None,
    ) -> bool:
        """Apply one matching result and synchronize the visible projection.

        Args:
            result: Candidate settled state from the Skills source.
            focus_identity: Widget identity to restore after synchronization.

        Returns:
            True when the result owned the active generation and was applied.
        """
        if not self._request_is_active():
            return False
        applied = apply_skill_browse_result(self.result, result)
        if applied is self.result:
            return False
        self.result = applied
        if applied.status == "error":
            failure_copy = self._failure_copy(applied.scope)
            prior = self.applied_result
            failed_scope_changed = (
                prior is not None
                and replace(applied.scope, page=prior.page) != prior.scope
            )
            if failed_scope_changed:
                self.freshness = "stale"
                self.error_copy = ""
                self.stale_copy = failure_copy
            elif self.freshness != "stale":
                self.error_copy = failure_copy
        else:
            self.scope = applied.scope
            self.applied_result = applied
            self.retained_items = applied.items
            self.freshness = "fresh"
            self.error_copy = ""
            self.stale_copy = ""
        self._sync_view()(applied, focus_identity)
        return True

    def _failure_copy(self, failed_scope: SkillBrowseScope) -> str:
        applied = self.applied_result
        if applied is None:
            return self.result.error
        if replace(failed_scope, page=applied.page) == applied.scope:
            return f"Couldn't load page {failed_scope.page}."
        return "Filter wasn't applied; showing previous results."

    def invalidate(self, scope: SkillBrowseScope | None = None) -> int:
        """Invalidate in-flight work and begin a replacement generation.

        Args:
            scope: Optional replacement scope; the current scope is reused when
                omitted.

        Returns:
            The new monotonic request token.
        """
        if scope is not None:
            self.scope = scope
        return self.begin(self.scope)
