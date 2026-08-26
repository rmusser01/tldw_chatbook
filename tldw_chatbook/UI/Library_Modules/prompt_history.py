"""Non-visual retained Prompt history orchestration for the Library screen.

The controller owns one immutable ``PromptHistoryState`` and the workers that
advance it through the Task-196 pure reducers. It owns no DOM: ``sync_view`` is
the single screen-provided paint seam, while ``run_worker`` and
``run_service_call`` are the only Textual/service framework dependencies.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from loguru import logger

from ...Prompt_Management.prompt_restore_errors import PromptRestoreError
from ...Library.library_prompts_state import (
    PromptHistoryPageRequest,
    PromptHistoryRequest,
    PromptHistoryRestoreGate,
    PromptHistoryRestoreOutcome,
    PromptHistoryRestoreRequest,
    PromptHistoryState,
    apply_prompt_history_count,
    apply_prompt_history_page,
    apply_prompt_history_preview,
    apply_prompt_history_restore,
    begin_prompt_history_count,
    begin_prompt_history_page,
    begin_prompt_history_preview,
    begin_prompt_history_restore,
    build_prompt_history_page,
    build_prompt_history_state,
    close_prompt_history,
    format_prompt_history_restore_outcome,
    history_restore_gate,
    reset_prompt_history_page,
)

_PAGE_SIZE = 10
_COUNT_ERROR = "Couldn't load retained history count. Try again."
_PAGE_ERROR = "Couldn't load retained history. Try again."


class LibraryPromptHistoryController:
    """Own retained-history state and off-UI-path service calls.

    Args:
        screen: Owning screen, retained only to live-read its replaceable
            Textual ``run_worker`` framework seam.
        run_service_call: Late-binding service-call seam that can isolate work in a
            thread.
        prompt_service: Late-binding accessor for the app-wired scope service.
        sync_view: Late-binding targeted callback that repaints only the history
            region.
    """

    def __init__(
        self,
        *,
        screen: Any,
        run_service_call: Callable[..., Awaitable[Any]],
        prompt_service: Callable[[], Any],
        sync_view: Callable[[PromptHistoryState | None], None],
    ) -> None:
        self._screen = screen
        self._run_service_call = run_service_call
        self._prompt_service = prompt_service
        self._sync_view = sync_view
        self.state: PromptHistoryState | None = None
        self._scope_counter = 0
        self._request_counter = 0

    @property
    def _run_worker(self) -> Callable[..., Any]:
        """Live-read Textual's worker starter so test/runtime replacement works."""
        return self._screen.run_worker

    def _next_request_token(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def _publish(self, state: PromptHistoryState | None) -> None:
        self.state = state
        self._sync_view(state)

    def invalidate(self) -> None:
        """Drop history when navigation or prompt identity changes."""
        self._scope_counter += 1
        self._publish(None)

    def initialize(
        self, detail: Mapping[str, Any], *, open_history: bool = False
    ) -> None:
        """Start an index-only count for one freshly loaded local Prompt."""
        prompt_uuid = detail.get("uuid")
        current_version = detail.get("version")
        if not isinstance(prompt_uuid, str) or not prompt_uuid:
            self.invalidate()
            return
        if type(current_version) is not int or current_version < 1:
            current_version = None
        self._scope_counter += 1
        state = build_prompt_history_state(
            prompt_uuid=prompt_uuid,
            current_version=current_version,
            scope_token=self._scope_counter,
        )
        state, request = begin_prompt_history_count(
            state, request_token=self._next_request_token()
        )
        self._publish(state)
        self._run_worker(
            self._load_count(request),
            exclusive=True,
            group="library_prompt_history_count",
            name="library_prompt_history_count",
        )
        if open_history:
            self.request_page()

    async def _load_count(self, request: PromptHistoryRequest) -> None:
        service = self._prompt_service()
        count_versions = getattr(service, "count_prompt_versions", None)
        if not callable(count_versions):
            total_count = None
            error = _COUNT_ERROR
        else:
            try:
                total_count = await self._run_service_call(
                    count_versions,
                    mode="local",
                    prompt_identifier=request.prompt_uuid,
                    isolate_in_worker=True,
                )
                error = ""
            except Exception:
                # Retained bodies and arbitrary exception text are private.
                logger.warning("Library retained-history count failed.")
                total_count = None
                error = _COUNT_ERROR
        state = self.state
        if state is None:
            return
        next_state = apply_prompt_history_count(
            state, request, total_count=total_count, error=error
        )
        if next_state is not state:
            self._publish(next_state)

    def retry_count(self) -> None:
        """Retry the scalar count while leaving loaded pages untouched."""
        state = self.state
        if state is None or state.count_request is not None:
            return
        state, request = begin_prompt_history_count(
            state, request_token=self._next_request_token()
        )
        self._publish(state)
        self._run_worker(
            self._load_count(request),
            exclusive=True,
            group="library_prompt_history_count",
            name="library_prompt_history_count",
        )

    def request_page(self) -> None:
        """Open/retry/load one cursor page, suppressing duplicate requests."""
        state = self.state
        if state is None or state.page_request is not None:
            return
        if state.rows and not state.has_more:
            return
        state, request = begin_prompt_history_page(
            state, request_token=self._next_request_token()
        )
        self._publish(state)
        self._run_worker(
            self._load_page(request),
            exclusive=True,
            group="library_prompt_history_page",
            name="library_prompt_history_page",
        )

    async def _load_page(self, request: PromptHistoryPageRequest) -> None:
        service = self._prompt_service()
        list_versions = getattr(service, "list_prompt_versions", None)
        page = None
        error = ""
        if not callable(list_versions):
            error = _PAGE_ERROR
        else:
            try:
                result = await self._run_service_call(
                    list_versions,
                    mode="local",
                    prompt_identifier=request.prompt_uuid,
                    page_size=_PAGE_SIZE,
                    before_change_id=request.before_change_id,
                    isolate_in_worker=True,
                )
                page = build_prompt_history_page(result)
            except Exception:
                # Never surface or log retained Prompt bodies / exception text.
                logger.warning("Library retained-history page load failed.")
                error = _PAGE_ERROR
        state = self.state
        if state is None:
            return
        next_state = apply_prompt_history_page(state, request, page, error=error)
        if next_state is not state:
            self._publish(next_state)

    def close(self) -> None:
        """Apply the pure collapsed reset without re-requesting the count."""
        state = self.state
        if state is not None and state.is_open:
            self._publish(close_prompt_history(state))

    def matches_scope(self, *, prompt_uuid: str, scope_token: int) -> bool:
        """Return whether a semantic action targets the live Prompt scope."""
        state = self.state
        return bool(
            state is not None
            and state.prompt_uuid == prompt_uuid
            and state.scope_token == scope_token
        )

    def reload_page(self) -> None:
        """Reset to page zero and reload without repeating a settled count."""
        state = self.state
        if state is None:
            return
        self._publish(reset_prompt_history_page(state))
        if state.retained_count is None and state.count_request is None:
            self.retry_count()
        self.request_page()

    def select(self, *, change_id: int, source_version: int) -> None:
        """Select one already-loaded immutable preview."""
        state = self.state
        if state is None:
            return
        if not any(
            row.change_id == change_id and row.version == source_version
            for row in state.rows
        ):
            return
        state, request = begin_prompt_history_preview(
            state,
            change_id=change_id,
            source_version=source_version,
            request_token=self._next_request_token(),
        )
        self._publish(apply_prompt_history_preview(state, request))

    def restore_gate(self, *, dirty: bool) -> PromptHistoryRestoreGate | None:
        state = self.state
        return history_restore_gate(state, dirty=dirty) if state is not None else None

    def begin_restore(
        self,
        *,
        dirty: bool,
        expected_target: tuple[str, int, int, int],
    ) -> PromptHistoryRestoreRequest | None:
        """Revalidate a modal's captured target, then begin one restore."""
        state = self.state
        if state is None or state.restore_request is not None:
            return None
        gate = history_restore_gate(state, dirty=dirty)
        if (
            gate.target is None
            or (
                gate.target.prompt_uuid,
                gate.target.change_id,
                gate.target.source_version,
                gate.target.expected_current_version,
            )
            != expected_target
        ):
            self._sync_view(state)
            return None
        state, request, _gate = begin_prompt_history_restore(
            state, request_token=self._next_request_token(), dirty=dirty
        )
        if request is not None:
            self._publish(state)
        return request

    async def restore(
        self, request: PromptHistoryRestoreRequest
    ) -> PromptHistoryRestoreOutcome | None:
        """Run a conditional restore and publish only an exact guarded result."""
        service = self._prompt_service()
        restore_version = getattr(service, "restore_prompt_version", None)
        result = None
        caught: Exception | None = None
        if not callable(restore_version):
            caught = RuntimeError()
        else:
            try:
                result = await self._run_service_call(
                    restore_version,
                    mode="local",
                    prompt_identifier=request.prompt_uuid,
                    version=request.source_version,
                    change_id=request.change_id,
                    expected_version=request.expected_current_version,
                    isolate_in_worker=True,
                )
            except PromptRestoreError as exc:
                # Only bounded service categories are suitable for UI translation.
                caught = exc
            except Exception:
                # Never include arbitrary exception text or retained bodies.
                logger.warning("Library retained-history restore failed.")
                caught = RuntimeError()
        outcome = format_prompt_history_restore_outcome(result, error=caught)
        state = self.state
        if state is None:
            return None
        next_state = apply_prompt_history_restore(state, request, outcome)
        if next_state is state:
            return None
        self._publish(next_state)
        return outcome
