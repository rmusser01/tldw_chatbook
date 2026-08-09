"""Non-visual orchestration for the Library's exact Prompt browse page."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from ...Library.library_prompts_state import (
    PromptBrowseResult,
    PromptBrowseScope,
    apply_prompt_browse_result,
    begin_prompt_browse,
    build_prompt_browse_error,
    build_prompt_browse_result,
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
        """Request the current scope with a fresh monotonic token."""
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
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to browse the local Prompt library."
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
        self.scope = applied.scope
        self._sync_view()(applied, focus_identity)
        return True

    def invalidate(self, scope: PromptBrowseScope | None = None) -> int:
        """Supersede work after navigation or restored-scope changes."""
        if scope is not None:
            self.scope = scope
        return self.begin(self.scope)
