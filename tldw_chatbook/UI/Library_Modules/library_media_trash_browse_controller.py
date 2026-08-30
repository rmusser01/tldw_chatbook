"""Source-owned orchestration for exact local Media Trash pages."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from ...Library.library_media_state import (
    MediaTrashBrowseState,
    MediaTrashMutationTarget,
    MediaTrashRequestOrigin,
    MediaTrashResult,
    MediaTrashScope,
    apply_media_trash_result,
    begin_media_trash_mutation,
    begin_media_trash_request,
    build_media_trash_result,
    cancel_media_trash_delete_confirmation,
    commit_media_trash_mutation,
    fail_media_trash_mutation,
    fail_media_trash_request,
    open_media_trash_delete_confirmation,
    select_media_trash_item,
)
from ...Library.library_pager_state import (
    LibraryPagerDisplay,
    build_library_pager_display,
)

_WORKER_GROUP = "library-media-trash-browse"
_SERVICE_ERROR = "Could not load Trash."
_SHRINK_COPY = "Source changed again; try again."
_STALE_COPY = "List may be out of date."


class LibraryMediaTrashBrowseController:
    """Own one immutable Trash state plus request-generation worker handles."""

    def __init__(
        self,
        *,
        screen: Any,
        run_service_call: Callable[[], Callable[..., Awaitable[Any]]],
        media_service: Callable[[], Any],
        sync_view: Callable[[], Callable[[str | None], None]],
        request_is_active: Callable[[], bool],
    ) -> None:
        self._screen = screen
        self._run_service_call = run_service_call
        self._media_service = media_service
        self._sync_view = sync_view
        self._request_is_active = request_is_active
        self._generation = 0
        self.state = MediaTrashBrowseState()

    @property
    def _run_worker(self) -> Callable[..., Any]:
        return self._screen.run_worker

    @property
    def pager(self) -> LibraryPagerDisplay:
        """Derive the complete pager display from immutable source state."""
        applied = self.state.applied_result
        requested_page = (
            self.state.requested_scope.page
            if self.state.loading or self.state.error_copy or applied is None
            else applied.scope.page
        )
        return build_library_pager_display(
            applied_page=applied.scope.page if applied is not None else None,
            requested_page=requested_page,
            page_size=applied.limit if applied is not None else 20,
            row_count=len(self.state.retained_items),
            total=(
                applied.total
                if applied is not None and self.state.freshness == "fresh"
                else None
            ),
            freshness=self.state.freshness,
            loading=self.state.loading,
            error_copy=self.state.error_copy,
            stale_copy=self.state.stale_copy,
        )

    def _current(self, generation: int) -> bool:
        return generation == self._generation and self._request_is_active()

    def _publish(self, generation: int, focus_identity: str | None) -> bool:
        if not self._current(generation):
            return False
        self._sync_view()(focus_identity)
        return True

    def request(
        self,
        scope: MediaTrashScope,
        *,
        origin: MediaTrashRequestOrigin,
        focus_identity: str | None,
    ) -> Any | None:
        """Dispatch one exact local Trash request for the active route."""
        next_state = begin_media_trash_request(self.state, scope, origin=origin)
        if not self._request_is_active():
            return None
        self._generation += 1
        generation = self._generation
        self.state = next_state
        if not self._publish(generation, focus_identity):
            return None
        return self._run_worker(
            self._load(
                scope,
                generation=generation,
                focus_identity=focus_identity,
            ),
            exclusive=True,
            group=_WORKER_GROUP,
        )

    def retry(self, *, focus_identity: str | None) -> Any | None:
        """Repeat the exact failed target with its original focus authority."""
        scope = self.state.failed_scope or self.state.requested_scope
        origin = self.state.failed_origin or "retry"
        return self.request(scope, origin=origin, focus_identity=focus_identity)

    async def _list(self, scope: MediaTrashScope) -> MediaTrashResult:
        service = self._media_service()
        list_trash = getattr(service, "list_library_media_trash", None)
        if not callable(list_trash):
            raise RuntimeError("Media Trash service unavailable")
        payload = await self._run_service_call()(
            list_trash,
            mode="local",
            query=scope.query,
            media_type=scope.media_type,
            limit=scope.page_size,
            offset=scope.offset,
            isolate_in_worker=True,
        )
        return build_media_trash_result(scope, payload)

    async def _load(
        self,
        scope: MediaTrashScope,
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
                    self.state = apply_media_trash_result(self.state, result)
                    self._publish(generation, focus_identity)
                    return
                if clamped:
                    self.state = fail_media_trash_request(
                        self.state,
                        fetched_scope,
                        copy=_SHRINK_COPY,
                    )
                    self._publish(generation, focus_identity)
                    return
                clamped = True
                fetched_scope = scope.with_page(result.last_page)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._current(generation):
                return
            logger.warning(
                "Library Media Trash browse failed; "
                "operation=list_library_media_trash page={} page_size={} "
                "has_query={} has_type={} exception_type={}",
                fetched_scope.page,
                fetched_scope.page_size,
                bool(fetched_scope.query),
                fetched_scope.media_type is not None,
                type(exc).__name__,
            )
            copy = (
                _SHRINK_COPY
                if clamped
                else _STALE_COPY
                if self.state.freshness == "stale"
                else self._failure_copy(scope)
            )
            self.state = fail_media_trash_request(
                self.state,
                fetched_scope if clamped else scope,
                copy=copy,
            )
            self._publish(generation, focus_identity)

    def _failure_copy(self, failed_scope: MediaTrashScope) -> str:
        applied = self.state.applied_result
        if applied is None:
            return _SERVICE_ERROR
        if failed_scope.same_except_page(applied.scope):
            return (
                f"Page {failed_scope.page} not loaded — "
                f"showing page {applied.scope.page}."
            )
        previous = (
            "All Trash"
            if not applied.scope.query and applied.scope.media_type is None
            else "previous results"
        )
        return f"Filter not applied — showing {previous}."

    def select(self, stable_id: str) -> None:
        """Select one visible fresh Trash identity."""
        generation = self._generation
        if not self._current(generation):
            return
        self.state = select_media_trash_item(self.state, stable_id)
        self._publish(generation, None)

    def open_delete_confirmation(self) -> MediaTrashMutationTarget | None:
        """Capture and expose the selected permanent-delete target."""
        generation = self._generation
        if not self._current(generation):
            return None
        self.state = open_media_trash_delete_confirmation(self.state)
        self._publish(generation, None)
        return self.state.confirmation_target

    def cancel_delete_confirmation(self) -> None:
        """Dismiss permanent-delete confirmation without changing selection."""
        generation = self._generation
        if not self._current(generation):
            return
        self.state = cancel_media_trash_delete_confirmation(self.state)
        self._publish(generation, None)

    def claim_mutation(self) -> MediaTrashMutationTarget | None:
        """Fence reads and claim the currently selected fresh target."""
        if not self._request_is_active():
            return None
        target = self.state.confirmation_target
        if target is None:
            candidate = open_media_trash_delete_confirmation(self.state)
            target = candidate.confirmation_target
        next_state = begin_media_trash_mutation(self.state)
        if target is None or next_state is self.state:
            return None
        self._generation += 1
        generation = self._generation
        self.state = next_state
        self._publish(generation, None)
        return target

    def finish_mutation_failure(
        self, target: MediaTrashMutationTarget, copy: str
    ) -> None:
        """Publish a recoverable pre-commit mutation failure."""
        generation = self._generation
        if not self._current(generation):
            return
        self.state = fail_media_trash_mutation(self.state, target, copy=copy)
        self._publish(generation, None)

    def finish_mutation_commit(
        self, target: MediaTrashMutationTarget, notice: str
    ) -> None:
        """Publish a committed removal as stale before authoritative refresh."""
        generation = self._generation
        if not self._current(generation):
            return
        self.state = commit_media_trash_mutation(self.state, target, notice=notice)
        self._publish(generation, None)

    def request_after_mutation(self, *, focus_identity: str | None) -> Any | None:
        """Refresh the applied page after a committed mutation."""
        scope = (
            self.state.applied_result.scope
            if self.state.applied_result is not None
            else self.state.requested_scope
        )
        return self.request(scope, origin="mutation", focus_identity=focus_identity)

    def invalidate(self) -> int:
        """Fence every outstanding local read without publishing route state."""
        self._generation += 1
        return self._generation
