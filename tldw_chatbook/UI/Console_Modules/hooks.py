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


async def _request_console_hooks_review(self, snapshot, waiting, cancel):
    from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
        request_hook_review,
    )

    return await request_hook_review(
        self,
        self._console_runtime().ensure_hook_permissions(),
        snapshot,
        waiting,
        cancel,
    )


async def _open_console_hooks_review(self) -> None:
    await self._hooks.review_current()


async def _refresh_console_hooks(self) -> None:
    # ADR-097: indicator disk reads and hook imports start after first paint.
    while not getattr(self.app, "_ui_ready", True):
        await asyncio.sleep(0.1)
    await self._hooks.refresh()


def _apply_console_hooks_state(self, snapshot) -> None:
    from ..Screens import chat_screen as owner

    self._console_hook_review_snapshot = snapshot
    self._sync_console_control_bar()
    try:
        self.query_one("#console-control-hooks", owner.Button).set_class(
            snapshot.pending_count > 0, "hooks-attention"
        )
    except owner.QueryError:
        pass


async def _dispatch_console_draft_send(
    self,
    draft: str,
    stash: ConsoleDraftStash | None = None,
    *,
    session_id: str | None = None,
) -> bool:
    """Compatibility delegate for the one typed queue-aware dispatcher."""

    from tldw_chatbook.Chat.console_send_diagnostics import send_diagnostic_scope

    async with send_diagnostic_scope(
        "ui_dispatch", self._ui_responsiveness_monitor()
    ) as diagnostic:
        if session_id is None:
            session_id = self._console_visible_send_session_id()
        if stash is None:
            composer = self._console_composer_or_none()
            stash = composer.capture_draft_for_send() if composer else None
        result = await self._hooks.dispatch(
            draft,
            session_id=session_id,
            stash=stash,
            dispatch=lambda: self._prompt_queue.dispatch(
                draft, session_id=session_id, stash=stash
            ),
        )
        diagnostic.outcome = result.status.value
        return result.status is not ConsolePromptDispatchStatus.REFUSED
