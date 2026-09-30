"""DOM-free hook review and one captured Console Send continuation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
    ) -> None:
        self._permissions = hook_permissions_accessor
        self._review = request_review
        self._session = current_session
        self._stash = current_stash
        self._on_state = on_state
        self._notify = notify
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
        if self._busy:
            return self._refused(
                session_id, "Hook review or Send is already in progress."
            )
        self._busy = True
        self._generation += 1
        generation = self._generation
        try:
            snapshot = await asyncio.to_thread(self._permissions().snapshot)
            self._on_state(snapshot)
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
                return self._refused(
                    session_id, "Draft, chat or hooks changed; Send again."
                )
            # The captured continuation is consumed before the normal dispatcher awaits.
            self._generation += 1
            return await dispatch()
        finally:
            self._busy = False

    def _refused(self, session_id: str, detail: str) -> ConsolePromptDispatchResult:
        self._notify(detail, "warning")
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.REFUSED, session_id, detail
        )
