"""Non-visual orchestration for the Library's exact Prompt browse page."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, Mapping

from loguru import logger

from ...Library.library_prompts_state import (
    PromptBrowseResult,
    PromptBrowseScope,
    apply_prompt_browse_result,
    begin_prompt_browse,
    build_prompt_browse_error,
    build_prompt_browse_result,
    validate_prompt_browse_items,
)
from ...Library.library_pager_state import (
    LibraryPagerDisplay,
    PageFreshness,
    build_library_pager_display,
)

_WORKER_GROUP = "library-prompt-browse"
_SERVICE_UNAVAILABLE = (
    "Couldn't load prompts. The local Prompt service is unavailable; retry."
)
_SERVICE_ERROR = "Couldn't load prompts. Check the local Library and retry."


class LibraryPromptBrowseController:
    """Own exact Prompt browse state, tokens, and isolated service work.

    All replaceable framework and service seams are accessed lazily. This keeps
    runtime/test replacement authoritative even after the controller exists.

    Args:
        screen: Owning screen, retained only to live-read ``run_worker``.
        run_service_call: Accessor for the current thread-isolating call seam.
        prompt_service: Accessor for the current local Prompt scope service.
        sync_view: Accessor for the screen's current non-state presentation seam.
        request_is_active: Late-bound navigation guard for the Prompt list.
    """

    def __init__(
        self,
        *,
        screen: Any,
        run_service_call: Callable[[], Callable[..., Awaitable[Any]]],
        prompt_service: Callable[[], Any],
        sync_view: Callable[[], Callable[[PromptBrowseResult, str | None], None]],
        request_is_active: Callable[[], bool],
    ) -> None:
        self._screen = screen
        self._run_service_call = run_service_call
        self._prompt_service = prompt_service
        self._sync_view = sync_view
        self._request_is_active = request_is_active
        self._request_counter = 1
        self.scope = PromptBrowseScope()
        self.result = begin_prompt_browse(self.scope, request_token=1)
        self.applied_result: PromptBrowseResult | None = None
        self.retained_items: tuple[Mapping[str, Any], ...] = ()
        self.freshness: PageFreshness = "uninitialized"
        self.error_copy = ""
        self.stale_copy = ""

    @property
    def visible_result(self) -> PromptBrowseResult:
        """Return exact applied metadata when last-good rows are visible."""
        return self.applied_result or self.result

    @property
    def pager(self) -> LibraryPagerDisplay:
        """Derive Prompt pager presentation from requested and applied state."""
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
    def mutation_refresh_scope(self) -> PromptBrowseScope:
        """Return the complete applied scope used to refresh after mutation."""
        return (
            self.applied_result.scope if self.applied_result is not None else self.scope
        )

    def scope_for_page(self, page: int) -> PromptBrowseScope:
        """Change only the page of the full last-applied Prompt scope."""
        return replace(self.mutation_refresh_scope, page=page)

    def retain_stale_items(
        self,
        items: tuple[Mapping[str, Any], ...],
        *,
        stale_copy: str,
    ) -> None:
        """Retain a locally reconciled page without forging exact metadata."""
        if type(items) is not tuple:
            raise TypeError("items must be an exact tuple.")
        if not isinstance(stale_copy, str) or not stale_copy.strip():
            raise ValueError("stale_copy must be non-empty text.")
        if self.applied_result is None:
            raise ValueError("Cannot retain stale items before a page applies.")
        self.retained_items = validate_prompt_browse_items(items)
        self.freshness = "stale"
        self.error_copy = ""
        self.stale_copy = stale_copy.strip()

    @property
    def _run_worker(self) -> Callable[..., Any]:
        """Live-read Textual's worker starter so replacement remains visible."""
        return self._screen.run_worker

    def _next_request_token(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def begin(self, scope: PromptBrowseScope) -> int:
        """Publish a fresh loading token without dispatching service work."""
        token = self._next_request_token()
        self.scope = scope
        self.result = begin_prompt_browse(scope, request_token=token)
        self.error_copy = ""
        return token

    def dispatch(
        self,
        scope: PromptBrowseScope,
        *,
        request_token: int,
        focus_identity: str | None,
    ) -> Any | None:
        """Dispatch a still-current loading request through one worker."""
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
        scope: PromptBrowseScope,
        *,
        focus_identity: str | None,
    ) -> Any | None:
        """Begin and dispatch one exact browse request immediately."""
        token = self.begin(scope)
        return self.dispatch(
            scope,
            request_token=token,
            focus_identity=focus_identity,
        )

    def retry(self, *, focus_identity: str | None) -> Any | None:
        """Request the last settled or failed scope with a fresh token."""
        return self.request(self.scope, focus_identity=focus_identity)

    async def _load(
        self,
        scope: PromptBrowseScope,
        *,
        request_token: int,
        focus_identity: str | None,
    ) -> None:
        service = self._prompt_service()
        browse_prompts = getattr(service, "browse_prompts", None)
        if not callable(browse_prompts):
            result = build_prompt_browse_error(
                scope,
                request_token=request_token,
                error=_SERVICE_UNAVAILABLE,
            )
        else:
            try:
                result = build_prompt_browse_result(
                    scope,
                    await self._run_service_call()(
                        browse_prompts,
                        mode="local",
                        query=scope.query,
                        collection_id=scope.collection_id,
                        sort_by=scope.sort_by,
                        sort_order=scope.sort_order,
                        page=scope.page,
                        page_size=scope.page_size,
                        isolate_in_worker=True,
                    ),
                    request_token=request_token,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Library Prompt browse failed; operation=browse_prompts "
                    "exception_type={}",
                    type(exc).__name__,
                )
                result = build_prompt_browse_error(
                    scope,
                    request_token=request_token,
                    error=_SERVICE_ERROR,
                )
        self.apply(result, focus_identity=focus_identity)

    def apply(
        self,
        result: PromptBrowseResult,
        *,
        focus_identity: str | None,
    ) -> bool:
        """Apply only an active matching result, rejecting every late outcome."""
        if not self._request_is_active():
            return False
        applied = apply_prompt_browse_result(self.result, result)
        if applied is self.result:
            return False
        self.result = applied
        if applied.status == "error":
            if self.freshness != "stale":
                self.error_copy = self._failure_copy(applied.scope)
        else:
            self.scope = applied.scope
            self.applied_result = applied
            self.retained_items = applied.items
            self.freshness = "fresh"
            self.error_copy = ""
            self.stale_copy = ""
        self._sync_view()(applied, focus_identity)
        return True

    def _failure_copy(self, failed_scope: PromptBrowseScope) -> str:
        """Describe whether a page or broader Prompt scope failed to apply."""
        applied = self.applied_result
        if applied is None:
            return self.result.error
        applied_scope = applied.scope
        if replace(failed_scope, page=applied_scope.page) == applied_scope:
            return f"Couldn't load page {failed_scope.page}."
        return "Filter wasn't applied; showing previous results."

    def invalidate(self, scope: PromptBrowseScope | None = None) -> int:
        """Supersede work after navigation or restored-scope changes."""
        if scope is not None:
            self.scope = scope
        return self.begin(self.scope)
