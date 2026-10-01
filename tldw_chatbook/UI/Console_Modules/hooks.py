"""DOM-free hook review and one captured Console Send continuation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from textual.worker import NoActiveWorker, get_current_worker

from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    ConsolePromptDispatchResult,
    ConsolePromptDispatchStatus,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash

if TYPE_CHECKING:
    from tldw_chatbook.Agents.hook_permissions import (
        HookPermissions,
        HookReviewSnapshot,
    )


@dataclass(frozen=True, slots=True)
class HookReviewResult:
    kind: Literal["ready", "cancel", "settings"]
    snapshot: HookReviewSnapshot | None = None


def same_captured_draft(
    current: ConsoleDraftStash | None, expected: ConsoleDraftStash | None
) -> bool:
    if current is None or expected is None:
        return current is expected
    return (current.text, current.edit_serial, current.generation) == (
        expected.text,
        expected.edit_serial,
        expected.generation,
    )


def _in_worker() -> bool:
    try:
        get_current_worker()
    except NoActiveWorker:
        return False
    return True


class ConsoleHooksController:
    """Keep review cancellation separate from persistent explicit consent."""

    def __init__(
        self,
        *,
        hook_permissions_accessor: Callable[[], HookPermissions],
        request_review: Callable[
            [HookReviewSnapshot, bool, Callable[[], None]], Awaitable[HookReviewResult]
        ],
        current_session: Callable[[], str],
        current_stash: Callable[[], ConsoleDraftStash | None],
        on_state: Callable[[HookReviewSnapshot], None],
        notify: Callable[[str, str], None],
        start_worker: (
            Callable[[Coroutine[Any, Any, ConsolePromptDispatchResult]], object] | None
        ) = None,
    ) -> None:
        """Wire the controller to its owners.

        Args:
            start_worker: Runs a Send's review-then-dispatch continuation in a
                Textual worker. When given, ``dispatch`` called outside a
                worker hands its review to it instead of awaiting the review
                itself (TASK-33621.28). ``None`` keeps every review inline,
                for callers that own their own task.
        """
        self._permissions = hook_permissions_accessor
        self._review = request_review
        self._session = current_session
        self._stash = current_stash
        self._on_state = on_state
        self._notify = notify
        self._start_worker = start_worker
        self._generation = 0
        self._busy = False
        self.review_open = False

    async def _request_review(self, snapshot, waiting):
        self.review_open = True
        try:
            return await self._review(snapshot, waiting, self.cancel_pending)
        finally:
            self.review_open = False

    def cancel_pending(self) -> None:
        """Invalidate a continuation synchronously before navigation/dismissal."""
        self._generation += 1

    async def refresh(self) -> None:
        generation = self._generation
        try:
            owner = self._permissions()
        except RuntimeError:  # Runtime disposal may precede a queued UI refresh.
            return
        snapshot = await asyncio.to_thread(owner.snapshot)
        if generation == self._generation:
            self._on_state(snapshot)

    async def review_current(self) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            snapshot = await asyncio.to_thread(self._permissions().snapshot)
            self._on_state(snapshot)
            await self._request_review(snapshot, False)
            await self.refresh()
        finally:
            self._busy = False

    async def dispatch(
        self,
        draft: str,
        *,
        session_id: str,
        stash: ConsoleDraftStash | None,
        dispatch: Callable[[], Awaitable[ConsolePromptDispatchResult]],
    ) -> ConsolePromptDispatchResult:
        """Review hooks if the Send needs it, then run the captured dispatch.

        Called outside a Textual worker -- a Send button or Workbench handler
        on the Console's pump, or Enter's ``app.call_later`` callback on the
        APP pump -- a Send that needs review returns ``AWAITING_REVIEW`` at
        once and its review-then-dispatch continuation runs in a worker. That
        caller's pump must stay free: the app pump delivers every key and
        click the review needs, so awaiting the review there froze the whole
        app, Ctrl+Q included (TASK-33621.28). A worker caller (spoken "send")
        awaits the whole continuation and gets its settled outcome.
        """
        if self._busy:
            return self._refused(
                session_id, "Hook review or Send is already in progress."
            )
        self._busy = True
        self._generation += 1
        generation = self._generation
        owns_busy = True
        try:
            snapshot = await asyncio.to_thread(self._permissions().snapshot)
            self._on_state(snapshot)
            if (
                not snapshot.ready
                and self._start_worker is not None
                and not _in_worker()
            ):
                continuation = self._continue_in_worker(
                    snapshot, generation, session_id, stash, dispatch
                )
                try:
                    self._start_worker(continuation)
                except BaseException:
                    continuation.close()
                    raise
                owns_busy = False  # The continuation releases it when it settles.
                return ConsolePromptDispatchResult(
                    ConsolePromptDispatchStatus.AWAITING_REVIEW, session_id
                )
            return await self._continue(
                snapshot, generation, session_id, stash, dispatch
            )
        finally:
            if owns_busy:
                self._busy = False

    async def _continue_in_worker(
        self,
        snapshot: HookReviewSnapshot,
        generation: int,
        session_id: str,
        stash: ConsoleDraftStash | None,
        dispatch: Callable[[], Awaitable[ConsolePromptDispatchResult]],
    ) -> ConsolePromptDispatchResult:
        try:
            return await self._continue(
                snapshot, generation, session_id, stash, dispatch
            )
        finally:
            self._busy = False

    async def _continue(
        self,
        snapshot: HookReviewSnapshot,
        generation: int,
        session_id: str,
        stash: ConsoleDraftStash | None,
        dispatch: Callable[[], Awaitable[ConsolePromptDispatchResult]],
    ) -> ConsolePromptDispatchResult:
        if not snapshot.ready:
            result = await self._request_review(snapshot, True)
            if result.kind != "ready":
                return self._refused(session_id, "Send cancelled; draft kept.")
            snapshot = await asyncio.to_thread(self._permissions().snapshot)
            self._on_state(snapshot)
        if (
            not snapshot.ready
            or generation != self._generation
            or self._session() != session_id
            or not same_captured_draft(self._stash(), stash)
        ):
            return self._refused(session_id, "Draft, chat or hooks changed; Send again.")
        # The captured continuation is consumed before the normal dispatcher awaits.
        self._generation += 1
        return await dispatch()

    def _refused(self, session_id: str, detail: str) -> ConsolePromptDispatchResult:
        self._notify(detail, "warning")
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.REFUSED, session_id, detail
        )
